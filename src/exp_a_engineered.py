"""Experiment A — engineered projected-quantum-kernel advantage (Huang 2021).

For n in {8,10,20} qubits and seeds {0,1,2}: build a quantum-tailored task by
relabelling random inputs along the max-geometric-difference direction of the
projected quantum kernel, then compare a projected-quantum-kernel SVM against a
full classical baseline suite (and, as a within-quantum control, the naive
fidelity quantum kernel). Repeated stratified splits give error bars + a paired
significance test.
"""
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import common as C
from quantum_kernels import (make_bloch_fn, make_state_fn, projected_kernel,
                             fidelity_kernel)

QUBITS = [8, 10, 20]
SEEDS = [0, 1, 2]
N_SAMPLES = 250
GAMMA = 0.1
REPS = 2
N_SPLITS = 10
RAW = "results/raw"
os.makedirs(RAW, exist_ok=True)


def engineer_labels(K_C, K_Q, lam=1e-3):
    """Labels along the max-quantum-advantage direction (median-thresholded
    for ~50/50 balance). Returns y in {-1,+1} and the geometric difference g."""
    n = K_C.shape[0]
    wq, vq = np.linalg.eigh(K_Q)
    sqrtKQ = (vq * np.sqrt(np.clip(wq, 0, None))) @ vq.T
    inv = np.linalg.inv(K_C + lam * np.eye(n))
    M = (sqrtKQ @ inv @ sqrtKQ)
    M = (M + M.T) / 2
    w, v = np.linalg.eigh(M)
    g = float(np.sqrt(max(w[-1], 0.0)))
    proj = sqrtKQ @ v[:, -1]
    y = np.where(proj >= np.median(proj), 1, -1).astype(np.int8)
    return y, g


def build_dataset(n_qubits, seed, with_states):
    """Generate (or load cached) engineered dataset for (n_qubits, seed)."""
    cache = os.path.join(RAW, f"engk_n{n_qubits}_s{seed}.npz")
    rng = np.random.default_rng(seed)
    X = rng.uniform(-np.pi, np.pi, size=(N_SAMPLES, n_qubits)).astype(np.float64)

    if os.path.exists(cache):
        d = np.load(cache)
        B, K_Q, K_C, y = d["B"], d["K_Q"], d["K_C"], d["y"]
        g = float(d["g"])
    else:
        B = make_bloch_fn(n_qubits, REPS)(X)
        K_Q = projected_kernel(B, GAMMA, auto_bandwidth=False)
        sq = np.sum(X ** 2, axis=1)
        d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0.0)
        med = np.median(d2[d2 > 0])
        K_C = np.exp(-d2 / (med + 1e-9))               # classical RBF on inputs
        y, g = engineer_labels(K_C, K_Q)
        np.savez_compressed(cache, X=X, B=B, K_Q=K_Q, K_C=K_C, y=y, g=g)

    K_F = None
    if with_states:
        fcache = os.path.join(RAW, f"engk_fid_n{n_qubits}_s{seed}.npy")
        if os.path.exists(fcache):
            K_F = np.load(fcache)
        else:
            states = make_state_fn(n_qubits, REPS)(X)
            K_F = fidelity_kernel(states)
            np.save(fcache, K_F)
            del states
    return X, y, K_Q, K_C, K_F, g


def run():
    C.ensure_root_cwd()
    C.set_seed()
    results = {}
    for n in QUBITS:
        for seed in SEEDS:
            t0 = time.time()
            with_states = (seed == 0)          # fidelity control: seed 0 only
            X, y, K_Q, K_C, K_F, g = build_dataset(n, seed, with_states)
            balance = float(np.mean(y == 1))

            # classical feature models (defines the shared splits)
            feat_res, splits = C.eval_feature_models(X, y, n_splits=N_SPLITS)
            # projected quantum kernel on the SAME splits
            proj = C.eval_precomputed_kernel(K_Q, y, splits)
            # classical RBF precomputed kernel on the SAME splits
            crbf = C.eval_precomputed_kernel(K_C, y, splits)

            models = {"Quantum: Projected kernel": proj,
                      "Classical: RBF kernel (precomp.)": crbf}
            models.update(feat_res)
            if K_F is not None:
                models["Quantum: Fidelity kernel"] = C.eval_precomputed_kernel(K_F, y, splits)

            key = f"n{n}_s{seed}"
            results[key] = {
                "n_qubits": n, "seed": seed, "g": g, "class_balance": balance,
                "models": {m: {"test": summ(v["test"]), "train": summ(v["train"]),
                               "test_raw": v["test"]}
                           for m, v in models.items()},
            }
            # paired test: projected quantum vs best classical (by mean test acc)
            classical = {m: v for m, v in models.items()
                         if m.startswith("Classical") or m.startswith("Baseline")}
            best = max(classical, key=lambda m: np.mean(models[m]["test"]))
            results[key]["best_classical"] = best
            results[key]["paired_vs_best_classical"] = C.paired_test(
                models["Quantum: Projected kernel"]["test"], models[best]["test"])
            dt = time.time() - t0
            print(f"[{key}] g={g:.2f} balance={balance:.2f} "
                  f"Qproj={np.mean(proj['test']):.3f} bestC({best.split(':')[1].strip()})="
                  f"{np.mean(models[best]['test']):.3f} ({dt:.0f}s)", flush=True)
    with open("results/exp_a.json", "w") as f:
        json.dump(results, f, indent=2)
    print("saved results/exp_a.json")


def summ(s):
    a = np.asarray(s)
    return {"mean": float(a.mean()), "std": float(a.std())}


if __name__ == "__main__":
    run()
