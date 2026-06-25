"""Experiment B — discrete-log task (Liu, Arunachalam & Temme 2021).

Off-the-shelf classical ML on the bit features should sit at chance across
n=12/16/20 bits. We also build the quantum *interval-state* fidelity kernel —
the feature map a fault-tolerant quantum computer realises via Shor's period
finding — and show a precomputed-kernel SVM separates the task near-perfectly.

HONEST CAVEAT: the interval kernel is computed here from the discrete-log table,
which is only tractable because p < 2^21 is classically simulable. On hardware
the log is obtained inside the feature map by Shor's algorithm WITHOUT a
precomputed table; at this bit size the task is NOT asymptotically hard, so this
demonstrates "off-the-shelf classical ML fails", not a fault-tolerant separation.
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "datasets", "generators"))
import common as C
from discrete_log_dataset import build_log_table

BITS = [12, 16, 20]
N_SPLITS = 10
DATA = "datasets/discrete_log"


def interval_fidelity_kernel(x_int, p, g):
    """K(x,x') = |<phi(x)|phi(x')>|^2 for Liu interval states.
    phi(x) = uniform superposition over the length-k=(p-1)/2 arc of logs
    [L(x), L(x)+k). Overlap fraction f = 1 - 2*circ_dist/P; fidelity = f^2."""
    log = build_log_table(p, g)
    P = p - 1
    L = log[x_int].astype(np.int64)          # discrete logs (the quantum feature)
    d = np.abs(L[:, None] - L[None, :])
    circ = np.minimum(d, P - d)              # circular distance on Z_{p-1}
    f = np.clip(1.0 - 2.0 * circ / P, 0.0, None)   # arc overlap fraction
    return (f ** 2).astype(np.float64), L


def run():
    C.ensure_root_cwd()
    C.set_seed()
    results = {}
    for n in BITS:
        d = np.load(os.path.join(DATA, f"dlp_n{n}.npz"))
        meta = json.load(open(os.path.join(DATA, f"dlp_n{n}_meta.json")))
        X_bits, y, x_int = d["x_bits"], d["y"].astype(int), d["x_int"]
        p, g = meta["p"], meta["g"]

        # off-the-shelf classical ML on bit features (defines shared splits)
        feat_res, splits = C.eval_feature_models(X_bits, y, n_splits=N_SPLITS)

        # quantum interval-state fidelity kernel on the SAME splits
        K_Q, L = interval_fidelity_kernel(x_int, p, g)
        gdiff = None
        try:
            # classical RBF kernel on bits, for geometric difference context
            sq = np.sum(X_bits ** 2, axis=1)
            d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * X_bits @ X_bits.T, 0.0)
            med = np.median(d2[d2 > 0])
            K_C = np.exp(-d2 / (med + 1e-9))
            gdiff = C.geometric_difference(K_C, K_Q)
        except Exception as e:
            print("g failed:", e)
        qk = C.eval_precomputed_kernel(K_Q, y, splits, C=10.0)

        models = {"Quantum: Interval-state kernel": qk}
        models.update(feat_res)
        results[f"n{n}"] = {
            "n_bits": meta["n_bits"], "p": p, "n_samples": len(y),
            "class_balance": float(np.mean(y == 1)), "geometric_difference": gdiff,
            "models": {m: {"test": C.summarize(v["test"]),
                           "train": C.summarize(v["train"]),
                           "test_raw": v["test"]} for m, v in models.items()},
        }
        cls_best = max(
            (m for m in models if not m.startswith("Quantum")),
            key=lambda m: np.mean(models[m]["test"]))
        qmean = np.mean(qk["test"])
        print(f"[n={n}] Q-interval={qmean:.3f}  best off-the-shelf "
              f"classical({cls_best.split(':')[-1].strip()})="
              f"{np.mean(models[cls_best]['test']):.3f}  g={gdiff}", flush=True)
    with open("results/exp_b.json", "w") as f:
        json.dump(results, f, indent=2)
    print("saved results/exp_b.json")


if __name__ == "__main__":
    run()
