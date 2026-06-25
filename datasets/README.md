# Datasets

Two **generator-based** dataset families built to test the project hypothesis —
*tasks where classical ML struggles but a 20+ qubit (simulated) quantum method
excels*. Both implement published constructions and are **regenerated from
code** (raw `.npz` arrays are git-ignored; generators, metadata, and small
samples are committed).

All commands assume the workspace venv:
```bash
source .venv/bin/activate
```

---

## Dataset 1 — Discrete-Logarithm (DLP) classification  `discrete_log/`

The canonical *provable* classical-hard / quantum-easy task from
**Liu, Arunachalam & Temme, Nature Physics 2021 (arXiv:2010.02174)**. Inputs are
elements of `Z_p*`; the binary label is a half-interval of the discrete log
`log_g(x)`. After taking the discrete log the task is trivial 1-D clustering, but
in the raw input space the labels look random to any learner that cannot compute
discrete logs.

### Generate
```bash
python datasets/generators/discrete_log_dataset.py --n-bits 20 --n-samples 2000 --seed 0
# also: --n-bits 12 / 16 for scaling studies
```
Produces `discrete_log/dlp_n{N}.npz` with `x_int` (group elements), `x_bits`
(bit-vector features for the learner), `y ∈ {-1,+1}`, plus `*_meta.json`.

### Validated premise (this workspace)
Standard classical models trained on the bit features score at chance:

| Model | Test acc (n=16) |
|---|---|
| RBF-SVM | **0.505 ± 0.016** |
| Random Forest | **0.489 ± 0.013** |
| *(random guessing)* | 0.500 |

### Honest caveat
At `n=20` bits the discrete log is *not* asymptotically hard — a dedicated
classical adversary could solve DLP for a 20-bit prime. This dataset therefore
demonstrates that **off-the-shelf classical ML struggles** (the wording of the
hypothesis), not a fault-tolerant-scale asymptotic separation. The true
asymptotic separation (Liu 2021) needs hundreds of qubits and is **not**
simulable — see `../literature_review.md` §1.

---

## Dataset 2 — Engineered projected-quantum-kernel advantage  `engineered_kernel/`

The *simulable empirical* advantage construction from
**Huang et al., "Power of data", Nat. Commun. 2021 (arXiv:2011.01938)**. Inputs
are encoded into an `n`-qubit state; the **projected quantum kernel**
`K_Q(x,x') = exp(-γ Σ_k ‖b_k(x)-b_k(x')‖²)` is built from per-qubit Bloch vectors
`b_k` (1-qubit reduced density matrices). Labels are **engineered** along the
direction of maximal *geometric difference* `g(K_C ‖ K_Q)`, i.e. relabeled so the
quantum kernel separates them with margin while the best classical (RBF) kernel
cannot. This is exactly the 20–30 qubit simulable regime of the hypothesis.

### Generate (uses PennyLane `lightning.qubit`)
```bash
python datasets/generators/engineered_kernel_dataset.py --n-qubits 20 --n-samples 250 --gamma 0.1 --sanity --seed 0
# small/fast: --n-qubits 8 ; intermediate: --n-qubits 10
```
Produces `engineered_kernel/engk_q{N}.npz` with `X` (inputs), `bloch` (Bloch
vectors), `y ∈ {-1,+1}`, precomputed kernels `K_Q`,`K_C`, plus `*_meta.json`
(includes `geometric_difference_g` and the sanity check).

### Validated advantage (this workspace, SVM on precomputed kernels, 70/30 split)

| n qubits | Projected **quantum** kernel | Classical RBF kernel | geom. diff `g` |
|---|---|---|---|
| 8  | **0.933** | 0.317 | 18.9 |
| 10 | **0.879** | 0.561 | ~6 |
| **20** | **0.893** | 0.453 | 6.35 |

The quantum kernel generalizes; the classical kernel is at/near chance. The gap
is the headline evidence for the hypothesis in the simulable regime.

### Notes / scaling
- Kernel construction is `O(N²)` circuit evaluations; `N≈250` at 20 qubits runs
  in a few minutes on `lightning.qubit`. Increase `N` and `n` for stronger
  scaling claims (memory ~ `2^n` statevector; ≤~28 qubits is practical on a
  workstation).
- `gamma=0.1` (absolute) sits in the clean-separation regime; `--reps` controls
  feature-map depth.

---

## Loading
```python
import numpy as np
d = np.load("datasets/engineered_kernel/engk_q20.npz")
X, y, K_Q, K_C = d["X"], d["y"], d["K_Q"], d["K_C"]

from sklearn.svm import SVC               # train directly on precomputed kernels
clf = SVC(kernel="precomputed").fit(K_Q[:175][:, :175], y[:175])
```

## Standard real-data benchmarks (for context / optional)
Used by the literature for *feasibility* (not advantage). Download on demand:
```python
# Fashion-MNIST (Huang 2021 / Jerbi 2023 relabeling base)
from sklearn.datasets import fetch_openml
fm = fetch_openml("Fashion-MNIST", version=1, as_frame=False)
```
The engineered-kernel generator can be adapted to relabel PCA-reduced
Fashion-MNIST instead of random inputs (set `X` to PCA features) to reproduce
Huang/Jerbi exactly.

## Reproduce everything
```bash
bash datasets/generate_all.sh      # regenerates all dataset sizes + sanity checks
```
