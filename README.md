# Quantum Methods on ML

Does a **quantum** ML method, *simulated at 20+ qubits*, succeed on tasks where
**classical** ML fails? Two reproducible experiments say **yes — with honest
scope**: the win is on quantum-tailored / number-theoretic tasks, from a
problem-matched quantum inductive bias, under noiseless simulation.

## Key findings
- **Engineered task, 20 qubits** (Huang et al. 2021 projected quantum kernel):
  quantum-kernel SVM **0.87** test acc vs **0.63** for the best of 6 tuned
  classical models (gradient boosting); paired t-test **p ≈ 4×10⁻⁸**, Cohen's
  d = 5.4. Gap is stable **~0.23 across n = 8/10/20 qubits**.
- **Control:** the *naive fidelity* quantum kernel collapses to **chance (0.49)**
  at 20 qubits — the win is the **projected** kernel's inductive bias, not
  "quantumness." A generic quantum kernel overfits just like the MLP.
- **Discrete-log task** (Liu et al. 2021), 12/17/21 bits: all 6 off-the-shelf
  classical models at **chance (0.49–0.52)**; quantum interval-state kernel
  **0.997 → 1.000**. (Trees hit train acc 1.0, test 0.50 → memorize, can't generalize.)
- **Honest caveats:** Exp A labels are engineered (existence-of-task); Exp B at
  ≤21 bits is not asymptotically hard and the quantum kernel uses a simulable log
  table (what Shor does on hardware). A *naturally labeled* simulable advantage
  remains an open problem. See `REPORT.md` §6–7.

## Reproduce
```bash
source .venv/bin/activate                      # Python 3.12, PennyLane 0.45, sklearn 1.9
python src/exp_a_engineered.py                 # engineered projected-kernel task (~6 min)
python src/exp_b_discrete_log.py               # discrete-log task (~3 min)
python src/make_figures.py                     # figures from results/*.json
```
Datasets regenerate from `datasets/generators/` (see `datasets/README.md`); raw
arrays are git-ignored. Seeds fixed (42; dataset seeds 0/1/2); 10 stratified
splits, all models share splits (paired comparisons).

## Structure
```
planning.md              Phase-0 motivation + experimental plan
REPORT.md                Full research report (PRIMARY deliverable)
literature_review.md     15-paper synthesis (pre-gathered)
resources.md             resource catalog (pre-gathered)
src/                     experiment code
  common.py              seeding, baselines, eval, geometric difference, stats
  quantum_kernels.py     PennyLane feature maps; projected & fidelity kernels
  exp_a_engineered.py    Experiment A
  exp_b_discrete_log.py  Experiment B
  make_figures.py        figure generation
results/                 exp_a.json, exp_b.json, environment.json, raw/ caches
figures/                 fig1–fig4 (.png)
datasets/                generators + metadata (Huang engineered; Liu discrete-log)
papers/                  15 source PDFs
```

Full details, tables, statistical tests, limitations → **`REPORT.md`**.
