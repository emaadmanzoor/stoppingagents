#!/usr/bin/env python3

# Independent implementation by Emaad Manzoor (emaadmanzoor@cornell.edu)
# of the data generation process described in:
#
# Damera Venkata, N., & Bhattacharyya, C. (2023). Deep recurrent optimal stopping.
# In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, & S. Levine (Eds.),
# Advances in Neural Information Processing Systems (Vol. 36). Curran Associates, Inc.

import numpy as np
from tqdm import tqdm

# Mirroring Deep Recurrent Optimal Stopping ("Exercise of a Bermudan max-call option"):
# K = 100, r = 0.05, sigma_m = 0.2, delta_m = 0.1, T = 3, num_exercise_opportunities = 9.
#
# Mapping to Equation (14):
# - Continuous-time model:
#     S_t^m = s0^m * exp( (r - δ_m - σ_m^2/2) t + σ_m W_t^m )
#     R_t   = exp(-r t) * ( max_m S_t^m - K )_+
# - Discretization used here (exact simulation on a grid):
#     1) Follow the authors' convention: let L = num_exercise_opportunities - 1 and
#        set dt = T / L. This yields L+1 exercise dates.
#     2) For each path k, time step j=1..L, and asset component m=1..d, sample
#        an independent standard normal Z_{k,j,m} ~ N(0, 1). This corresponds to a
#        Brownian increment via ΔW_{k,j,m} = sqrt(dt) * Z_{k,j,m}, since W has
#        constant off-diagonal correlation rho_ij (set to 0 in the paper experiment).
#     3) Plug ΔW into the closed-form Black–Scholes update over one time step:
#          log S_{t_j}^{k,m} = log S_{t_{j-1}}^{k,m}
#                             + (r - δ - 0.5 σ^2) * dt
#                             + σ * ΔW_{k,j,m}
#        so log-paths are just a cumulative sum of per-step increments.
#     4) Exponentiate to get prices S_{t_j}^{k,m}, and compute rewards on the grid:
#          g_{k,j} = exp(-r * t_j) * (max_m S_{t_j}^{k,m} - K)_+.
#   In code: `drift=(r-δ-0.5σ^2)dt`, `vol=σ*sqrt(dt)`, then `increments=drift+vol*Z`,
#   `log_paths=log(s0)+cumsum(increments)`, `x=exp(log_paths)`, and `g` as above.
# Output files:
# - Writes `train.npz` and `test.npz` into the current working directory.
# - Each `.npz` contains per-configuration arrays (each key is stored as `<key>.npy` inside the zip):
#     - `x_d{d}_s0{s0}`: float32[N_split, L+1, d] (states / prices X_t)
#     - `g_d{d}_s0{s0}`: float32[N_split, L+1]     (rewards R_t = g(t, X_t))
#   where `g` is the discounted payoff exp(-r t) * (max_m X_t^m - K)_+ at each exercise time.
K_VALUE = 100.0
R_VALUE = 0.05
DELTA_VALUE = 0.10
SIGMA_VALUE = 0.20
T_VALUE = 3.0
NUM_EXERCISE_OPPORTUNITIES = 9
RHO_IJ = 0.0

# Table 1 evaluated configurations (d, p0) only.
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

# The paper generates 40,000 trajectories and uses random 50/50 train-test splits.
# This script produces one deterministic 50/50 split per (d, s0) using BASE_SPLIT_SEED.
N_TOTAL_PATHS = 40_000
N_TRAIN_PATHS = 20_000
N_TEST_PATHS = 20_000

# Reproducibility (data generation and split shuffling).
BASE_DATA_SEED = 20260220
BASE_SPLIT_SEED = 20260222

# -----------------------------------------------------------------------------

L_STEPS = NUM_EXERCISE_OPPORTUNITIES - 1

dt_value = T_VALUE / float(L_STEPS)
times = np.linspace(0.0, T_VALUE, L_STEPS + 1, dtype=np.float32)
discount = np.exp(-np.float32(R_VALUE) * times).astype(np.float32)

assert N_TRAIN_PATHS + N_TEST_PATHS == N_TOTAL_PATHS

mu = np.float32(R_VALUE - DELTA_VALUE)
drift = np.float32((mu - 0.5 * SIGMA_VALUE * SIGMA_VALUE) * dt_value)
sqrt_dt = np.float32(np.sqrt(dt_value))

chol_by_d = {}

full_payload = {}
permutations = {}

for d_value, p0_value in tqdm(TABLE1_CONFIGS, desc="Simulating configs", unit="cfg"):
    d_value = int(d_value)
    s0_value = float(p0_value)
    cfg_key = f"d{d_value}_s0{int(s0_value)}"

    data_seed = int(BASE_DATA_SEED + 10_000 * d_value + int(s0_value))
    split_seed = int(BASE_SPLIT_SEED + 10_000 * d_value + int(s0_value))

    rng_data = np.random.default_rng(data_seed)
    rng_split = np.random.default_rng(split_seed)

    perm = rng_split.permutation(N_TOTAL_PATHS)

    C = chol_by_d.get(d_value)
    if C is None:
        rho = np.eye(d_value, dtype=np.float32)
        rho[~np.eye(d_value, dtype=bool)] = np.float32(RHO_IJ)
        C = np.linalg.cholesky(rho).astype(np.float32)
        chol_by_d[d_value] = C

    dw = rng_data.normal(
        loc=0.0,
        scale=float(sqrt_dt),
        size=(L_STEPS, d_value, N_TOTAL_PATHS),
    ).astype(np.float32)
    correlated_dw = np.transpose(np.dot(C, dw), (1, 2, 0))
    log_increments = drift + np.float32(SIGMA_VALUE) * correlated_dw
    log_paths = np.cumsum(log_increments, axis=0, dtype=np.float32)

    st = np.float32(s0_value) * np.exp(log_paths, dtype=np.float32)
    st = np.concatenate(
        (np.float32(s0_value) * np.ones((1, N_TOTAL_PATHS, d_value), dtype=np.float32), st),
        axis=0,
    )
    x = np.transpose(st, (1, 0, 2))

    max_component = x.max(axis=2)
    intrinsic = np.maximum(max_component - np.float32(K_VALUE), np.float32(0.0)).astype(np.float32)
    g = (intrinsic * discount[None, :]).astype(np.float32)

    del dw, correlated_dw, log_increments, log_paths, st, max_component, intrinsic

    full_payload[f"x_{cfg_key}"] = x
    full_payload[f"g_{cfg_key}"] = g
    permutations[cfg_key] = perm

train_payload = {}
test_payload = {}

for d_value, p0_value in TABLE1_CONFIGS:
    d_value = int(d_value)
    s0_value = float(p0_value)
    cfg_key = f"d{d_value}_s0{int(s0_value)}"
    perm = permutations[cfg_key]

    x = full_payload[f"x_{cfg_key}"][perm]
    g = full_payload[f"g_{cfg_key}"][perm]

    train_payload[f"x_{cfg_key}"] = x[:N_TRAIN_PATHS]
    train_payload[f"g_{cfg_key}"] = g[:N_TRAIN_PATHS]
    test_payload[f"x_{cfg_key}"] = x[N_TRAIN_PATHS:]
    test_payload[f"g_{cfg_key}"] = g[N_TRAIN_PATHS:]

print("Saving train.npz (np.savez_compressed)...")
np.savez_compressed("train.npz", **train_payload)
print("Saving test.npz (np.savez_compressed)...")
np.savez_compressed("test.npz", **test_payload)
