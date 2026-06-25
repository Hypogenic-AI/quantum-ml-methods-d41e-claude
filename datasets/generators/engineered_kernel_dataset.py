#!/usr/bin/env python3
"""
Engineered quantum-advantage dataset generator (label-engineering method).

Reproduces the dataset-relabelling procedure of
Huang, Broughton, Mohseni, Babbush, Boixo, Neven & McClean,
"Power of data in quantum machine learning", Nature Communications 12,
2631 (2021), arXiv:2011.01938.

Idea
----
For real-world data, classical models that *learn from data* are often as
good as quantum models. To expose a genuine separation, Huang et al.
RELABEL an input set so that the new labels are easy for a quantum kernel
but hard for the best classical kernel. They quantify the potential
advantage by the geometric difference

    g(K_C || K_Q) = sqrt( || sqrt(K_Q) (K_C + lambda I)^-1 sqrt(K_Q) ||_inf )

and construct labels along the eigenvector that maximises it:

    y = sign( sqrt(K_Q) v ),   v = top eigenvector of
        sqrt(K_Q) (K_C + lambda I)^-1 sqrt(K_Q).

These labels have a small quantum-model complexity (separable by K_Q with
large margin) but a large classical-model complexity, so an SVM with the
quantum kernel generalises while SVMs with classical kernels do not.

Here we use the *projected quantum kernel* (the variant that shows the
clearest advantage in the paper and resists the "power of data"
dequantisation):

    K_Q(x_i, x_j) = exp( -gamma * sum_k || b_k(x_i) - b_k(x_j) ||^2 )

where b_k(x) = (<X_k>, <Y_k>, <Z_k>) is the Bloch vector (1-qubit reduced
density matrix) of qubit k for the encoded state |phi(x)>. This is
classically hard to evaluate in general but cheap to *simulate* at the
20-30 qubit scale, which is exactly the regime of the hypothesis.

Outputs
-------
X (inputs, here random or PCA features), engineered binary labels y,
the geometric difference g, and the quantum-kernel SVM vs classical-kernel
SVM separation observed on a held-out split (sanity check).
"""
import argparse
import json
import os
import numpy as np

try:
    import pennylane as qml
except ImportError:
    qml = None


def make_feature_map(n_qubits, reps=2):
    """Hardware-efficient encoding: per-feature RY/RZ angle encoding with
    CZ entangling layers, repeated `reps` times. Inputs scaled into angles."""
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def bloch(x):
        for r in range(reps):
            for w in range(n_qubits):
                qml.RY(x[w] * (r + 1), wires=w)
                qml.RZ(x[(w + r) % n_qubits], wires=w)
            for w in range(n_qubits - 1):
                qml.CZ(wires=[w, w + 1])
            qml.CZ(wires=[n_qubits - 1, 0])
        return [qml.expval(qml.PauliX(w)) for w in range(n_qubits)] + \
               [qml.expval(qml.PauliY(w)) for w in range(n_qubits)] + \
               [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    def blochvecs(x):
        vals = np.array(bloch(x))
        # reshape to (n_qubits, 3): columns X,Y,Z
        return np.stack([vals[:n_qubits], vals[n_qubits:2 * n_qubits],
                         vals[2 * n_qubits:]], axis=1)
    return blochvecs


def projected_kernel(B, gamma, auto_bandwidth=True):
    """B: (N, n_qubits, 3) Bloch vectors -> projected quantum kernel (N,N).

    With auto_bandwidth, gamma is interpreted *relative* to the median
    pairwise projected distance, so the same gamma lands in a comparable
    regime across qubit counts. gamma ~ 0.1 (relative) gives a clean
    quantum/classical separation on engineered labels (see sanity check)."""
    N = B.shape[0]
    flat = B.reshape(N, -1)
    sq = np.sum(flat ** 2, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2 * flat @ flat.T
    d2 = np.maximum(d2, 0.0)
    np.fill_diagonal(d2, 0.0)
    if auto_bandwidth:
        med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
        gamma = gamma / (med + 1e-12)
    return np.exp(-gamma * d2)


def geometric_difference(K_C, K_Q, lam=1e-3):
    """g(K_C||K_Q) per Huang et al. Both kernels normalised to unit trace*."""
    n = K_C.shape[0]
    # symmetric sqrt of K_Q
    wq, vq = np.linalg.eigh(K_Q)
    wq = np.clip(wq, 0, None)
    sqrtKQ = (vq * np.sqrt(wq)) @ vq.T
    inv = np.linalg.inv(K_C + lam * np.eye(n))
    M = sqrtKQ @ inv @ sqrtKQ
    M = (M + M.T) / 2
    w, v = np.linalg.eigh(M)
    g = np.sqrt(max(w[-1], 0.0))
    return g, sqrtKQ, v[:, -1]


def engineer_labels(K_C, K_Q, lam=1e-3, balance=True):
    """Labels along the max-quantum-advantage direction. Thresholding the
    projection at its median (balance=True) yields ~50/50 classes so that
    accuracy is a meaningful separation metric (otherwise the eigenvector
    sign can be highly imbalanced)."""
    g, sqrtKQ, vtop = geometric_difference(K_C, K_Q, lam)
    proj = sqrtKQ @ vtop
    thr = np.median(proj) if balance else 0.0
    y = np.where(proj >= thr, 1, -1)
    return y.astype(np.int8), g


def generate(n_qubits=20, n_samples=240, gamma=1.0, seed=0, reps=2):
    if qml is None:
        raise SystemExit("pennylane not installed; run: uv add pennylane pennylane-lightning")
    rng = np.random.default_rng(seed)
    # inputs: random features in [-pi, pi] (stand-in for PCA-reduced real data)
    X = rng.uniform(-np.pi, np.pi, size=(n_samples, n_qubits)).astype(np.float64)
    bloch = make_feature_map(n_qubits, reps)
    B = np.array([bloch(X[i]) for i in range(n_samples)])
    K_Q = projected_kernel(B, gamma, auto_bandwidth=False)
    # best simple classical kernel: RBF on raw inputs (median heuristic)
    sq = np.sum(X ** 2, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2 * X @ X.T
    med = np.median(d2[d2 > 0])
    K_C = np.exp(-d2 / (med + 1e-9))
    y, g = engineer_labels(K_C, K_Q)
    meta = dict(n_qubits=int(n_qubits), n_samples=int(n_samples), gamma=float(gamma),
                reps=int(reps), geometric_difference_g=float(g),
                class_balance=float(np.mean(y == 1)),
                reference="Huang et al 2021 (arXiv:2011.01938), projected quantum kernel")
    return X, B, y, K_Q, K_C, meta


def sanity_check(X, y, K_Q, K_C, seed=0):
    """Train kernel SVMs (precomputed) on a split; report quantum vs classical."""
    from sklearn.svm import SVC
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    tr, te = idx[:int(0.7 * n)], idx[int(0.7 * n):]
    out = {}
    for name, K in [("projected_quantum", K_Q), ("classical_rbf", K_C)]:
        clf = SVC(kernel="precomputed", C=1.0)
        clf.fit(K[np.ix_(tr, tr)], y[tr])
        acc = clf.score(K[np.ix_(te, tr)], y[te])
        out[name + "_test_acc"] = float(acc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-qubits", type=int, default=20)
    ap.add_argument("--n-samples", type=int, default=240)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="datasets/engineered_kernel")
    ap.add_argument("--sanity", action="store_true")
    args = ap.parse_args()

    X, B, y, K_Q, K_C, meta = generate(args.n_qubits, args.n_samples,
                                       args.gamma, args.seed, args.reps)
    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, f"engk_q{args.n_qubits}.npz"),
                        X=X, bloch=B, y=y, K_Q=K_Q, K_C=K_C)
    if args.sanity:
        meta["sanity_check"] = sanity_check(X, y, K_Q, K_C, args.seed)
    with open(os.path.join(args.out, f"engk_q{args.n_qubits}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
