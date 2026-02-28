#!/usr/bin/env python3

import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression

CALLS_H5_PATH = "subcorpus_train_val_test60.hd5"
EMBEDDINGS_PARQUET_PATH = "embedding_subcorpus_train_val_test60_output.parquet"

COST_PER_SECOND = 1.0 / 193.0 * 5.5 / 100.0
BENEFIT_PER_SALE = 1.0

seed = int(os.environ.get("SEED", "42"))
embedding_dim_max = int(os.environ.get("BC_DIM_IN", "1500000"))

# Replication note (best benchmark seen so far; 2026-02-27):
# - T-1 gain (first stop @ 90s): 4.2496%
# - T-2 gain (first stop @ 60s, then 90s): 10.1074%
# Config (env vars / defaults):
# - SEED=42
# - BC_DIM_IN=15000  (effective dim_in=min(BC_DIM_IN, embedding_dim_available)=3072)
# - LR_SOLVER=lbfgs
# - LR_C=20.0
# - LR_MAX_ITER=1000
# Notes:
# - Training always uses train+val ("tv").

lr_solver = os.environ.get("LR_SOLVER", "lbfgs")
lr_c = float(os.environ.get("LR_C", "20.0"))
lr_max_iter = int(os.environ.get("LR_MAX_ITER", "100000"))

assert os.path.exists(CALLS_H5_PATH)
assert os.path.exists(EMBEDDINGS_PARQUET_PATH)

calls_train = pd.read_hdf(CALLS_H5_PATH, key="train")
calls_val = pd.read_hdf(CALLS_H5_PATH, key="val")
calls_test = pd.read_hdf(CALLS_H5_PATH, key="test")

calls_train = calls_train.loc[calls_train["transcript_speaker_30"].notna()].copy()
calls_val = calls_val.loc[calls_val["transcript_speaker_30"].notna()].copy()
calls_test = calls_test.loc[calls_test["transcript_speaker_30"].notna()].copy()

for df in [calls_train, calls_val, calls_test]:
    df["conversation_id"] = df["conversation_id"].astype(str)
    df["duration"] = df["duration"].astype(float)
    df["is_sale"] = df["is_sale"].astype(np.float32)
    assert set(df["is_sale"].dropna().unique().tolist()) <= {0.0, 1.0}

duration_train = calls_train["duration"].astype(np.float32).to_numpy()
is_sale_train = calls_train["is_sale"].astype(np.float32).to_numpy()
duration_val = calls_val["duration"].astype(np.float32).to_numpy()
is_sale_val = calls_val["is_sale"].astype(np.float32).to_numpy()
duration_test = calls_test["duration"].astype(np.float32).to_numpy()
is_sale_test = calls_test["is_sale"].astype(np.float32).to_numpy()

parquet_file = pq.ParquetFile(EMBEDDINGS_PARQUET_PATH)
parquet_meta = parquet_file.schema_arrow.metadata or {}
embedding_dim_available = int(parquet_meta[b"embedding_dim"].decode("utf-8"))
assert embedding_dim_available > 0
assert parquet_meta[b"embedding_dtype"].decode("utf-8") == "float32"
d_in = int(min(embedding_dim_max, embedding_dim_available))

embeddings = pd.read_parquet(
    EMBEDDINGS_PARQUET_PATH,
    columns=["split_code", "conversation_id", "times", "embedding"],
)
embeddings["conversation_id"] = embeddings["conversation_id"].astype(str)
embeddings["time"] = embeddings["times"].astype(int)
embeddings.drop(columns=["times"], inplace=True)
embeddings["split_code"] = embeddings["split_code"].astype(int)
embeddings["split"] = embeddings["split_code"].map({0: "train", 1: "val", 2: "test"})
assert embeddings["split"].isna().sum() == 0
embeddings.drop(columns=["split_code"], inplace=True)
assert set(embeddings["time"].unique().tolist()) <= {30, 60, 90}
assert embeddings.duplicated(subset=["split", "conversation_id", "time"]).sum() == 0

train_ids = set(calls_train["conversation_id"].to_numpy().tolist())
val_ids = set(calls_val["conversation_id"].to_numpy().tolist())
test_ids = set(calls_test["conversation_id"].to_numpy().tolist())
embeddings = embeddings.loc[
    ((embeddings["split"] == "train") & (embeddings["conversation_id"].isin(train_ids)))
    | ((embeddings["split"] == "val") & (embeddings["conversation_id"].isin(val_ids)))
    | ((embeddings["split"] == "test") & (embeddings["conversation_id"].isin(test_ids)))
]

Ts = sorted(embeddings["time"].unique().tolist())  # [30, 60, 90]

calls_train_ids = calls_train["conversation_id"].to_numpy()
calls_val_ids = calls_val["conversation_id"].to_numpy()
calls_test_ids = calls_test["conversation_id"].to_numpy()

embeddings_train = np.empty((len(calls_train), d_in, len(Ts)), dtype=np.float32)
embeddings_val = np.empty((len(calls_val), d_in, len(Ts)), dtype=np.float32)
embeddings_test = np.empty((len(calls_test), d_in, len(Ts)), dtype=np.float32)
nan = np.full(d_in, np.nan, dtype=np.float32).tobytes()

for split, call_ids, out in [
    ("train", calls_train_ids, embeddings_train),
    ("val", calls_val_ids, embeddings_val),
    ("test", calls_test_ids, embeddings_test),
]:
    e = embeddings.loc[embeddings["split"] == split, ["conversation_id", "time", "embedding"]].pivot(
        index="conversation_id", columns="time", values="embedding"
    )
    e = e.reindex(index=call_ids, columns=Ts)

    for t_idx, t in enumerate(Ts):
        out[:, :, t_idx] = np.stack(
            [
                np.frombuffer(
                    b if isinstance(b, (bytes, bytearray, memoryview)) else nan,
                    dtype=np.float32,
                    count=d_in,
                )
                for b in e[t].to_numpy()
            ]
        )

assert embeddings_train.shape == (len(calls_train), d_in, len(Ts))
assert embeddings_val.shape == (len(calls_val), d_in, len(Ts))
assert embeddings_test.shape == (len(calls_test), d_in, len(Ts))

print(
    "LR config: "
    f"dim_in={d_in} solver={lr_solver} C={lr_c} max_iter={lr_max_iter}"
)
# g_stop(n, X_n^k) = cumulative reward of stopping call k at time n given state X_n^k.
# If call k has duration < n, then g_stop(n, X_n^k) = -inf so it never wins in a max comparison.

# stopping values
g_train_stop = np.full((len(calls_train), len(Ts) + 1), -np.inf, dtype=np.float32)
g_val_stop = np.full((len(calls_val), len(Ts) + 1), -np.inf, dtype=np.float32)
g_test_stop = np.full((len(calls_test), len(Ts) + 1), -np.inf, dtype=np.float32)

# terminal stopping values are known
g_train_stop[:, -1] = BENEFIT_PER_SALE * is_sale_train - COST_PER_SECOND * duration_train
g_val_stop[:, -1] = BENEFIT_PER_SALE * is_sale_val - COST_PER_SECOND * duration_val
g_test_stop[:, -1] = BENEFIT_PER_SALE * is_sale_test - COST_PER_SECOND * duration_test

# stopping decisions (policy)
yhat_train = np.full((len(calls_train), len(Ts) + 1), -1, dtype=np.int8)
yhat_val = np.full((len(calls_val), len(Ts) + 1), -1, dtype=np.int8)
yhat_test = np.full((len(calls_test), len(Ts) + 1), -1, dtype=np.int8)

# terminal stopping is enforced (call has ended)
yhat_train[:, -1] = 1
yhat_val[:, -1] = 1
yhat_test[:, -1] = 1

# -----------------------------
# Base case: n = T - 1 (t=90)
# -----------------------------
n = Ts[-1]

g_train_continue = g_train_stop[:, -1].copy().reshape(-1, 1)
g_val_continue = g_val_stop[:, -1].copy().reshape(-1, 1)
g_test_continue = g_test_stop[:, -1].copy().reshape(-1, 1)

g_train_continue[duration_train < n, -1] = -np.inf
g_val_continue[duration_val < n, -1] = -np.inf
g_test_continue[duration_test < n, -1] = -np.inf

g_train_stop[:, -2] = np.where(duration_train >= n, -COST_PER_SECOND * n, -np.inf)
g_val_stop[:, -2] = np.where(duration_val >= n, -COST_PER_SECOND * n, -np.inf)
g_test_stop[:, -2] = np.where(duration_test >= n, -COST_PER_SECOND * n, -np.inf)

mask_train = duration_train >= n
mask_val = duration_val >= n
mask_test = duration_test >= n

X_train = embeddings_train[mask_train, :, -1]
X_val = embeddings_val[mask_val, :, -1]
X_test = embeddings_test[mask_test, :, -1]

y_train = (g_train_stop[mask_train, -2] >= g_train_continue[mask_train, -1]).astype(int)
y_val = (g_val_stop[mask_val, -2] >= g_val_continue[mask_val, -1]).astype(int)
y_test = (g_test_stop[mask_test, -2] >= g_test_continue[mask_test, -1]).astype(int)

w_train = g_train_stop[mask_train, -2] - g_train_continue[mask_train, -1]
w_val = g_val_stop[mask_val, -2] - g_val_continue[mask_val, -1]
w_test = g_test_stop[mask_test, -2] - g_test_continue[mask_test, -1]

sample_weight_train = np.abs(w_train)
sample_weight_val = np.abs(w_val)
sample_weight_test = np.abs(w_test)

assert np.unique(y_train).size == 2
assert np.unique(y_val).size == 2
assert np.unique(y_test).size == 2

print("Fitting classifier at time T - 1...")
X_fit = np.vstack([X_train, X_val])
y_fit = np.concatenate([y_train, y_val])
sample_weight_fit = np.concatenate([sample_weight_train, sample_weight_val])

clf = LogisticRegression(
    C=lr_c,
    solver=lr_solver,
    tol=1e-24,
    max_iter=lr_max_iter,
    random_state=seed,
)
clf.fit(X_fit, y_fit, sample_weight=sample_weight_fit)

p_train = clf.predict_proba(X_train)[:, 1]
p_val = clf.predict_proba(X_val)[:, 1]
p_test = clf.predict_proba(X_test)[:, 1]

yhat_train[mask_train, -2] = (p_train >= 0.5).astype(int)
yhat_val[mask_val, -2] = (p_val >= 0.5).astype(int)
yhat_test[mask_test, -2] = (p_test >= 0.5).astype(int)

d_test_T = duration_test[mask_test]
time_saved_test = np.sum((d_test_T - n) * yhat_test[mask_test, -2])
print(f"  Total time saved on test (duration > {n}s): {time_saved_test:.2f} sec")

sales_made_test = np.sum(is_sale_test[~mask_test]) + np.sum(is_sale_test[mask_test] * (1 - yhat_test[mask_test, -2]))
print(f"  Total sales made on test: {int(sales_made_test)}")

sales_per_second_test = float(np.sum(is_sale_test) / np.sum(duration_test))
expected_sales_from_time_saved_test = float(time_saved_test) * sales_per_second_test
print(
    "  Expected sales from time saved (same call distribution): "
    f"{expected_sales_from_time_saved_test:.4f}"
)

expected_total_sales_test = float(sales_made_test) + expected_sales_from_time_saved_test
print(
    f"  Expected total sales (assuming first stop opportunity at {n}s): "
    f"{expected_total_sales_test:.4f}"
)

print(f"  Status quo sales = {int(np.sum(is_sale_test))}")

status_quo_sales = float(np.sum(is_sale_test))
expected_sales_gain_pct = 100.0 * (expected_total_sales_test - status_quo_sales) / status_quo_sales
print(f"  Expected sales gain (%): {expected_sales_gain_pct:.4f}")

# ------------------------------------
# Inductive case: n = T - 2 (t=60)
# ------------------------------------
n = Ts[-2]

future_stops_train = yhat_train[:, -2:]
future_stops_val = yhat_val[:, -2:]
future_stops_test = yhat_test[:, -2:]

first_j_train = future_stops_train.argmax(axis=1)
first_j_val = future_stops_val.argmax(axis=1)
first_j_test = future_stops_test.argmax(axis=1)

start_col_train = len(Ts) + 1 - 2
start_col_val = len(Ts) + 1 - 2
start_col_test = len(Ts) + 1 - 2

first_col_train = start_col_train + first_j_train
first_col_val = start_col_val + first_j_val
first_col_test = start_col_test + first_j_test

g_future_stop_train = g_train_stop[np.arange(yhat_train.shape[0]), first_col_train]
g_future_stop_val = g_val_stop[np.arange(yhat_val.shape[0]), first_col_val]
g_future_stop_test = g_test_stop[np.arange(yhat_test.shape[0]), first_col_test]

g_train_continue = np.where(duration_train >= n, g_future_stop_train, -np.inf).reshape(-1, 1)
g_val_continue = np.where(duration_val >= n, g_future_stop_val, -np.inf).reshape(-1, 1)
g_test_continue = np.where(duration_test >= n, g_future_stop_test, -np.inf).reshape(-1, 1)

g_train_stop[:, -3] = np.where(duration_train >= n, -COST_PER_SECOND * n, -np.inf)
g_val_stop[:, -3] = np.where(duration_val >= n, -COST_PER_SECOND * n, -np.inf)
g_test_stop[:, -3] = np.where(duration_test >= n, -COST_PER_SECOND * n, -np.inf)

mask_train = duration_train >= n
mask_val = duration_val >= n
mask_test = duration_test >= n

X_train = embeddings_train[mask_train, :, -2]
X_val = embeddings_val[mask_val, :, -2]
X_test = embeddings_test[mask_test, :, -2]

y_train = (g_train_stop[mask_train, -3] >= g_train_continue[mask_train, -1]).astype(int)
y_val = (g_val_stop[mask_val, -3] >= g_val_continue[mask_val, -1]).astype(int)
y_test = (g_test_stop[mask_test, -3] >= g_test_continue[mask_test, -1]).astype(int)

w_train = g_train_stop[mask_train, -3] - g_train_continue[mask_train, -1]
w_val = g_val_stop[mask_val, -3] - g_val_continue[mask_val, -1]
w_test = g_test_stop[mask_test, -3] - g_test_continue[mask_test, -1]

sample_weight_train = np.abs(w_train)
sample_weight_val = np.abs(w_val)
sample_weight_test = np.abs(w_test)

assert np.unique(y_train).size == 2
assert np.unique(y_val).size == 2
assert np.unique(y_test).size == 2

print("Fitting classifier at time T - 2...")
X_fit = np.vstack([X_train, X_val])
y_fit = np.concatenate([y_train, y_val])
sample_weight_fit = np.concatenate([sample_weight_train, sample_weight_val])

clf = LogisticRegression(
    C=lr_c,
    solver=lr_solver,
    tol=1e-24,
    max_iter=lr_max_iter,
    random_state=seed,
)
clf.fit(X_fit, y_fit, sample_weight=sample_weight_fit)

p_train = clf.predict_proba(X_train)[:, 1]
p_val = clf.predict_proba(X_val)[:, 1]
p_test = clf.predict_proba(X_test)[:, 1]

yhat_train[mask_train, -3] = (p_train >= 0.5).astype(int)
yhat_val[mask_val, -3] = (p_val >= 0.5).astype(int)
yhat_test[mask_test, -3] = (p_test >= 0.5).astype(int)

d_test_T = duration_test[mask_test]

future_stops_test = yhat_test[mask_test, -3:]
first_j_test = future_stops_test.argmax(axis=1)
start_col_test = len(Ts) + 1 - 3
first_col_test = start_col_test + first_j_test
Ts_arr = np.asarray(Ts, dtype=np.float32)

stop_time_test = d_test_T.copy()
mask_not_terminal = first_col_test < len(Ts)
stop_time_test[mask_not_terminal] = Ts_arr[first_col_test[mask_not_terminal]]

time_saved_test = np.sum(d_test_T - stop_time_test)
print(f"  Total time saved on test (first stop >= {n}s): {time_saved_test:.2f} sec")

sales_made_test = np.sum(is_sale_test[~mask_test]) + np.sum(is_sale_test[mask_test] * (first_col_test == len(Ts)))
print(f"  Total sales made on test: {int(sales_made_test)}")

expected_sales_from_time_saved_test = float(time_saved_test) * sales_per_second_test
print(
    "  Expected sales from time saved (same call distribution): "
    f"{expected_sales_from_time_saved_test:.4f}"
)

expected_total_sales_test = float(sales_made_test) + expected_sales_from_time_saved_test
print(
    f"  Expected total sales (assuming first stop opportunity at {n}s): "
    f"{expected_total_sales_test:.4f}"
)

print(f"  Status quo sales = {int(np.sum(is_sale_test))}")

expected_sales_gain_pct = 100.0 * (expected_total_sales_test - status_quo_sales) / status_quo_sales
print(f"  Expected sales gain (%): {expected_sales_gain_pct:.4f}")

# ------------------------------------
# Inductive case: n = T - 3 (t=30)
# ------------------------------------
n = Ts[-3]

future_stops_train = yhat_train[:, -3:]
future_stops_val = yhat_val[:, -3:]
future_stops_test = yhat_test[:, -3:]

first_j_train = future_stops_train.argmax(axis=1)
first_j_val = future_stops_val.argmax(axis=1)
first_j_test = future_stops_test.argmax(axis=1)

start_col_train = len(Ts) + 1 - 3
start_col_val = len(Ts) + 1 - 3
start_col_test = len(Ts) + 1 - 3

first_col_train = start_col_train + first_j_train
first_col_val = start_col_val + first_j_val
first_col_test = start_col_test + first_j_test

g_future_stop_train = g_train_stop[np.arange(yhat_train.shape[0]), first_col_train]
g_future_stop_val = g_val_stop[np.arange(yhat_val.shape[0]), first_col_val]
g_future_stop_test = g_test_stop[np.arange(yhat_test.shape[0]), first_col_test]

g_train_continue = np.where(duration_train >= n, g_future_stop_train, -np.inf).reshape(-1, 1)
g_val_continue = np.where(duration_val >= n, g_future_stop_val, -np.inf).reshape(-1, 1)
g_test_continue = np.where(duration_test >= n, g_future_stop_test, -np.inf).reshape(-1, 1)

g_train_stop[:, -4] = np.where(duration_train >= n, -COST_PER_SECOND * n, -np.inf)
g_val_stop[:, -4] = np.where(duration_val >= n, -COST_PER_SECOND * n, -np.inf)
g_test_stop[:, -4] = np.where(duration_test >= n, -COST_PER_SECOND * n, -np.inf)

mask_train = duration_train >= n
mask_val = duration_val >= n
mask_test = duration_test >= n

X_train = embeddings_train[mask_train, :, -3]
X_val = embeddings_val[mask_val, :, -3]
X_test = embeddings_test[mask_test, :, -3]

y_train = (g_train_stop[mask_train, -4] >= g_train_continue[mask_train, -1]).astype(int)
y_val = (g_val_stop[mask_val, -4] >= g_val_continue[mask_val, -1]).astype(int)
y_test = (g_test_stop[mask_test, -4] >= g_test_continue[mask_test, -1]).astype(int)

w_train = g_train_stop[mask_train, -4] - g_train_continue[mask_train, -1]
w_val = g_val_stop[mask_val, -4] - g_val_continue[mask_val, -1]
w_test = g_test_stop[mask_test, -4] - g_test_continue[mask_test, -1]

sample_weight_train = np.abs(w_train)
sample_weight_val = np.abs(w_val)
sample_weight_test = np.abs(w_test)

assert np.unique(y_train).size == 2
assert np.unique(y_val).size == 2
assert np.unique(y_test).size == 2

print("Fitting classifier at time T - 3...")
X_fit = np.vstack([X_train, X_val])
y_fit = np.concatenate([y_train, y_val])
sample_weight_fit = np.concatenate([sample_weight_train, sample_weight_val])

clf = LogisticRegression(
    C=lr_c,
    solver=lr_solver,
    tol=1e-24,
    max_iter=lr_max_iter,
    random_state=seed,
)
clf.fit(X_fit, y_fit, sample_weight=sample_weight_fit)

p_train = clf.predict_proba(X_train)[:, 1]
p_val = clf.predict_proba(X_val)[:, 1]
p_test = clf.predict_proba(X_test)[:, 1]

yhat_train[mask_train, -4] = (p_train >= 0.5).astype(int)
yhat_val[mask_val, -4] = (p_val >= 0.5).astype(int)
yhat_test[mask_test, -4] = (p_test >= 0.5).astype(int)

d_test_T = duration_test[mask_test]

future_stops_test = yhat_test[mask_test, -4:]
first_j_test = future_stops_test.argmax(axis=1)
start_col_test = len(Ts) + 1 - 4
first_col_test = start_col_test + first_j_test
Ts_arr = np.asarray(Ts, dtype=np.float32)

stop_time_test = d_test_T.copy()
mask_not_terminal = first_col_test < len(Ts)
stop_time_test[mask_not_terminal] = Ts_arr[first_col_test[mask_not_terminal]]

time_saved_test = np.sum(d_test_T - stop_time_test)
print(f"  Total time saved on test (first stop >= {n}s): {time_saved_test:.2f} sec")

sales_made_test = np.sum(is_sale_test[~mask_test]) + np.sum(is_sale_test[mask_test] * (first_col_test == len(Ts)))
print(f"  Total sales made on test: {int(sales_made_test)}")

expected_sales_from_time_saved_test = float(time_saved_test) * sales_per_second_test
print(
    "  Expected sales from time saved (same call distribution): "
    f"{expected_sales_from_time_saved_test:.4f}"
)

expected_total_sales_test = float(sales_made_test) + expected_sales_from_time_saved_test
print(
    f"  Expected total sales (assuming first stop opportunity at {n}s): "
    f"{expected_total_sales_test:.4f}"
)

print(f"  Status quo sales = {int(np.sum(is_sale_test))}")

expected_sales_gain_pct = 100.0 * (expected_total_sales_test - status_quo_sales) / status_quo_sales
print(f"  Expected sales gain (%): {expected_sales_gain_pct:.4f}")
