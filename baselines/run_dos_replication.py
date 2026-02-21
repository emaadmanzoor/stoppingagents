#!/usr/bin/env python3

import csv
import math
import os
import time

from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn


TABLE1_CONFIGS = [
    (20, 90),
    (20, 100),
    (20, 110),
    (50, 90),
    (50, 100),
    (50, 110),
    (100, 90),
    (100, 100),
    (100, 110),
    (200, 90),
    (200, 100),
    (200, 110),
]

NUM_EXERCISE_OPPORTUNITIES = 9  # L

SEED = 20260222

VAL_FRACTION = 0.20
BATCH_SIZE = 64
INFERENCE_BATCH_SIZE = 8192
SAMPLES_PER_EPOCH = 200
DOS_EPOCHS = 100
LR = 1e-3
CLIPNORM = 5.0
EARLY_STOPPING_PATIENCE = 5
NUM_STACKED_LAYERS = 2
UNITS_HIDDEN_OFFSET = 20

BN_EPS = 1e-3
BN_MOMENTUM = 0.01  # PyTorch momentum = 1 - Keras momentum (0.99)

OMIT_TIME_ZERO = True
SATURATION_EPS = 1e-3  # Count phi as saturated if phi <= eps or phi >= 1-eps.

device = torch.device("cuda")
torch.set_default_dtype(torch.float32)

if not os.path.exists("train.npz"):
    raise FileNotFoundError("Missing train NPZ: train.npz")
if not os.path.exists("test.npz"):
    raise FileNotFoundError("Missing test NPZ: test.npz")
train_npz = np.load("train.npz")
test_npz = np.load("test.npz")

train_data = {k: np.array(train_npz[k], dtype=np.float32, copy=False) for k in train_npz.files}
test_data = {k: np.array(test_npz[k], dtype=np.float32, copy=False) for k in test_npz.files}
train_npz.close()
test_npz.close()
del train_npz
del test_npz

results = []

with tqdm(
    TABLE1_CONFIGS,
    total=len(TABLE1_CONFIGS),
    desc="DOS configs",
    unit="cfg",
    dynamic_ncols=True,
) as outer_bar:
    for d_value, s0_value in outer_bar:
        cfg_key = f"d{int(d_value)}_s0{int(s0_value)}"
        x_key = f"x_{cfg_key}"
        g_key = f"g_{cfg_key}"
        outer_bar.set_postfix_str(f"d={int(d_value)} p0={int(s0_value)}", refresh=False)

        if x_key not in train_data or g_key not in train_data:
            raise KeyError(f"Missing keys in train.npz: {x_key} / {g_key}")
        if x_key not in test_data or g_key not in test_data:
            raise KeyError(f"Missing keys in test.npz: {x_key} / {g_key}")

        x_train_t = train_data[x_key]  # [N, L, d]
        g_train_t = train_data[g_key]  # [N, L]
        x_test_t = test_data[x_key]
        g_test_t = test_data[g_key]

        if x_train_t.ndim != 3 or g_train_t.ndim != 2:
            raise ValueError(
                f"Bad shapes for {cfg_key}: x={tuple(x_train_t.shape)} g={tuple(g_train_t.shape)}"
            )
        if x_train_t.shape[0] != g_train_t.shape[0] or x_train_t.shape[1] != g_train_t.shape[1]:
            raise ValueError(
                f"Train mismatch for {cfg_key}: x={tuple(x_train_t.shape)} g={tuple(g_train_t.shape)}"
            )
        if x_test_t.shape[0] != g_test_t.shape[0] or x_test_t.shape[1] != g_test_t.shape[1]:
            raise ValueError(
                f"Test mismatch for {cfg_key}: x={tuple(x_test_t.shape)} g={tuple(g_test_t.shape)}"
            )

        n_train, seq_len, d_in = x_train_t.shape
        if int(seq_len) != int(NUM_EXERCISE_OPPORTUNITIES):
            raise ValueError(f"Expected L={NUM_EXERCISE_OPPORTUNITIES}, got L={seq_len} for {cfg_key}")
        if int(d_in) != int(d_value):
            raise ValueError(f"d mismatch: expected {d_value}, got {d_in} for {cfg_key}")

        row_seed = int(SEED + int(d_value) * 1_000 + int(s0_value))
        torch.manual_seed(row_seed)
        rng = np.random.default_rng(row_seed)

        n_val = int(max(1, math.floor(VAL_FRACTION * n_train)))
        perm = rng.permutation(n_train)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        if train_idx.size < 1:
            raise ValueError(f"Train split empty for {cfg_key} (n_train={n_train}, n_val={n_val}).")

        # Load full train/test tensors to GPU once per config.
        # Feature vectors are [x_t, g_stop] (so feature_dim = d + 1).
        feature_dim = int(d_in) + 1
        with torch.no_grad():
            x_train_all = torch.from_numpy(np.asarray(x_train_t, dtype=np.float32)).to(
                device=device, dtype=torch.float32
            )
            g_train_all = torch.from_numpy(np.asarray(g_train_t, dtype=np.float32)).to(
                device=device, dtype=torch.float32
            )
            x_test_all = torch.from_numpy(np.asarray(x_test_t, dtype=np.float32)).to(
                device=device, dtype=torch.float32
            )
            g_test_all = torch.from_numpy(np.asarray(g_test_t, dtype=np.float32)).to(
                device=device, dtype=torch.float32
            )

            train_features_all = torch.empty(
                (int(n_train), int(seq_len), int(feature_dim)), device=device, dtype=torch.float32
            )
            train_features_all[:, :, :-1].copy_(x_train_all)
            train_features_all[:, :, -1].copy_(g_train_all)

            n_test = int(x_test_t.shape[0])
            test_features_all = torch.empty(
                (int(n_test), int(seq_len), int(feature_dim)), device=device, dtype=torch.float32
            )
            test_features_all[:, :, :-1].copy_(x_test_all)
            test_features_all[:, :, -1].copy_(g_test_all)

            val_idx_t = torch.from_numpy(np.asarray(val_idx, dtype=np.int64)).to(device=device)
            val_features_all = train_features_all.index_select(0, val_idx_t)

            # Continuation values start at terminal payoff.
            cont_train = g_train_all[:, seq_len - 1 : seq_len].clone()
            cont_val = g_train_all.index_select(0, val_idx_t)[:, seq_len - 1 : seq_len].clone()

            del x_train_all
            del g_train_all
            del x_test_all
            del val_idx_t

        # Models stored for inference: index by time n (0..L-2); last step is forced stop.
        # If OMIT_TIME_ZERO is True, index 0 is left as None (we force continue at t=0).
        dos_models = [None] * (seq_len - 1)

        n_values = list(range(int(seq_len - 2), -1, -1))
        if OMIT_TIME_ZERO:
            n_values = list(range(int(seq_len - 2), 0, -1))

        sat_train_count_t = torch.zeros((), device=device, dtype=torch.int64)
        sat_train_total = 0

        with tqdm(
            total=int(len(n_values) * DOS_EPOCHS),
            desc=f"DOS d={int(d_value)} p0={int(s0_value)}",
            position=1,
            leave=False,
            dynamic_ncols=True,
            unit="epoch",
        ) as bar:
            t_train_started = time.time()

            for n in n_values:
                input_dim = int(d_in) + 1
                hidden_dim = int(d_in) + int(UNITS_HIDDEN_OFFSET)

                model = nn.Module()
                model.input_bn = (
                    nn.BatchNorm1d(input_dim, eps=float(BN_EPS), momentum=float(BN_MOMENTUM))
                )
                model.hidden_layers = nn.ModuleList()
                model.hidden_bns = nn.ModuleList()

                in_dim = int(input_dim)
                for _ in range(int(NUM_STACKED_LAYERS)):
                    model.hidden_layers.append(nn.Linear(in_dim, int(hidden_dim)))
                    model.hidden_bns.append(
                        nn.BatchNorm1d(int(hidden_dim), eps=float(BN_EPS), momentum=float(BN_MOMENTUM))
                    )
                    in_dim = int(hidden_dim)

                model.output_layer = nn.Linear(in_dim, 1)
                model.relu = nn.ReLU()
                model.sigmoid = nn.Sigmoid()
                model.to(device)

                with torch.no_grad():
                    for layer in model.hidden_layers:
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.zeros_(layer.bias)
                    nn.init.xavier_uniform_(model.output_layer.weight)
                    nn.init.zeros_(model.output_layer.bias)

                optimizer = torch.optim.Adam(model.parameters(), lr=float(LR))

                best_val_reward_hard = -float("inf")
                best_state = None
                patience = 0

                # Training targets are [g_stop, continuation_value].
                for _epoch in range(int(DOS_EPOCHS)):
                    model.train()
                    for _ in range(int(SAMPLES_PER_EPOCH)):
                        batch_sel = rng.choice(train_idx, size=int(BATCH_SIZE), replace=True)
                        batch_sel_t = torch.from_numpy(np.asarray(batch_sel, dtype=np.int64)).to(device=device)
                        feats = train_features_all.index_select(0, batch_sel_t)[:, n, :]
                        g_stop = feats[:, -1]
                        g_cont = cont_train.index_select(0, batch_sel_t)[:, 0]

                        values = model.input_bn(feats)
                        for layer, bn in zip(model.hidden_layers, model.hidden_bns):
                            values = layer(values)
                            values = model.relu(values)
                            values = bn(values)
                        phi = model.sigmoid(model.output_layer(values)).squeeze(1)

                        sat_train_count_t += (
                            (phi <= float(SATURATION_EPS)) | (phi >= float(1.0 - SATURATION_EPS))
                        ).to(dtype=torch.int64).sum()
                        sat_train_total += int(phi.numel())

                        loss = -(phi * g_stop + (1.0 - phi) * g_cont)
                        loss = loss.float().mean()

                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        if float(CLIPNORM) > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(CLIPNORM))
                        optimizer.step()

                    # Validation loss over the whole val split.
                    model.eval()
                    with torch.no_grad():
                        feats_val = val_features_all[:, n, :]
                        g_stop_val = feats_val[:, -1]
                        g_cont_val = cont_val[:, 0]

                        v = model.input_bn(feats_val)
                        for layer, bn in zip(model.hidden_layers, model.hidden_bns):
                            v = layer(v)
                            v = model.relu(v)
                            v = bn(v)
                        phi_val = model.sigmoid(model.output_layer(v)).squeeze(1)

                    stop_val = phi_val > 0.5
                    val_reward_hard = float(
                        torch.where(stop_val, g_stop_val, g_cont_val).float().mean().item()
                    )
                    improved = val_reward_hard > best_val_reward_hard
                    if improved:
                        best_val_reward_hard = val_reward_hard
                        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                        patience = 0
                    else:
                        patience += 1

                    bar.set_postfix_str(
                        f"t={n} val_reward={val_reward_hard:.3f} best_val_reward={best_val_reward_hard:.3f}",
                        refresh=False,
                    )
                    bar.update(1)

                    if patience > int(EARLY_STOPPING_PATIENCE):
                        break

                if best_state is not None:
                    model.load_state_dict(best_state)
                model.eval()
                dos_models[n] = model

                # Update continuation values on train and val using hard decisions at time n.
                with torch.no_grad():
                    # Train split (all N): inference in chunks to limit working-set size.
                    for start in range(0, int(n_train), int(INFERENCE_BATCH_SIZE)):
                        end = min(int(n_train), start + int(INFERENCE_BATCH_SIZE))
                        feats = train_features_all[start:end, n, :]
                        v = model.input_bn(feats)
                        for layer, bn in zip(model.hidden_layers, model.hidden_bns):
                            v = layer(v)
                            v = model.relu(v)
                            v = bn(v)
                        phi_chunk = model.sigmoid(model.output_layer(v)).squeeze(1)
                        stop_mask = phi_chunk > 0.5
                        cont_train[start:end, 0] = torch.where(stop_mask, feats[:, -1], cont_train[start:end, 0])

                    # Val split only.
                    feats_val = val_features_all[:, n, :]
                    v = model.input_bn(feats_val)
                    for layer, bn in zip(model.hidden_layers, model.hidden_bns):
                        v = layer(v)
                        v = model.relu(v)
                        v = bn(v)
                    phi_val_best = model.sigmoid(model.output_layer(v)).squeeze(1)
                    stop_val_mask = phi_val_best > 0.5
                    cont_val[:, 0] = torch.where(stop_val_mask, feats_val[:, -1], cont_val[:, 0])

            t_train_sec = time.time() - t_train_started

            # Inference on test set: stop at first time where phi>0.5 (and force stop at terminal).
            t_eval_started = time.time()
            stop_idxs = torch.full((int(n_test),), int(seq_len - 1), device=device, dtype=torch.int64)
            active = torch.ones((int(n_test),), device=device, dtype=torch.bool)

            with torch.no_grad():
                for n in range(int(seq_len - 1)):
                    if OMIT_TIME_ZERO and int(n) == 0:
                        continue
                    model = dos_models[n]
                    assert model is not None

                    stop_now = torch.zeros((int(n_test),), device=device, dtype=torch.bool)
                    for start in range(0, int(n_test), int(INFERENCE_BATCH_SIZE)):
                        end = min(int(n_test), start + int(INFERENCE_BATCH_SIZE))
                        feats = test_features_all[start:end, n, :]
                        v = model.input_bn(feats)
                        for layer, bn in zip(model.hidden_layers, model.hidden_bns):
                            v = layer(v)
                            v = model.relu(v)
                            v = bn(v)
                        phi_chunk = model.sigmoid(model.output_layer(v)).squeeze(1)
                        stop_now[start:end] = phi_chunk > 0.5

                    stop_now = stop_now & active
                    if bool(torch.any(stop_now).item()):
                        stop_idxs[stop_now] = int(n)
                        active[stop_now] = False

            payoff = g_test_all.gather(1, stop_idxs.view(-1, 1)).squeeze(1)
            dos_mean = float(payoff.mean().item())
            sat_frac_train = float("nan")
            if int(sat_train_total) > 0:
                sat_frac_train = float(sat_train_count_t.item()) / float(sat_train_total)

            t_eval_sec = time.time() - t_eval_started

            bar.set_postfix_str(f"done mean={dos_mean:.2f}", refresh=False)
            bar.total = bar.n

            row = {
                "d": int(d_value),
                "p0": int(s0_value),
                "dos_point": float(dos_mean),
                "sat_frac_train": float(sat_frac_train),
                "t_train_sec": float(t_train_sec),
                "t_eval_sec": float(t_eval_sec),
            }
            results.append(row)

        tqdm.write(
            f"Done DOS d={int(d_value)} p0={int(s0_value)} | mean={row['dos_point']:.3f} sat_train={row['sat_frac_train']:.4f}"
        )

results = sorted(results, key=lambda r: (int(r["d"]), int(r["p0"])))

with open("baselines_dos_results.csv", "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["d", "p0", "DOS_point", "sat_frac_train"])
    writer.writeheader()
    for row in results:
        writer.writerow(
            {
                "d": int(row["d"]),
                "p0": int(row["p0"]),
                "DOS_point": f'{float(row["dos_point"]):.2f}',
                "sat_frac_train": f'{float(row["sat_frac_train"]):.6f}',
            }
        )
