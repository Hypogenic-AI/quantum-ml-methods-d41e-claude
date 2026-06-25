# Planning — Quantum Methods on ML

## Motivation & Novelty Assessment

### Why This Research Matters
Quantum machine learning (QML) is widely promoted as a path to learning tasks
that classical ML cannot handle, but the field's central question — *when, if
ever, does a quantum model beat the best classical model on a learning task?* —
is unsettled and frequently overstated. Clarifying **exactly which tasks** admit
a quantum-over-classical separation, and **at what scale that separation is
actually demonstrable**, protects practitioners from hype and points research at
the regimes where quantum methods can genuinely help.

### Gap in Existing Work
From `literature_review.md` (15 deep-read papers), the landscape is two-sided:
- **Provable separations** (Liu 2021 discrete-log; Jerbi 2023 shadows) exist but
  embed Shor's algorithm and need *fault-tolerant, hundreds-of-qubit* hardware —
  **not simulable** at 20–30 qubits (at that size the number-theoretic problem is
  trivially classically solvable).
- **Simulable empirical separations** (Huang 2021; Jerbi 2023 beyond-kernel) are
  real and reproducible at 20–30 qubits, but only on **engineered/relabeled**
  data, not naturally occurring tasks.
- **Cautionary results** (Kübler 2021, Schuld 2021, Tang 2019) show expressivity
  alone destroys generalization and many advantages "dequantize."

The gap: most published demonstrations are either small-scale (≤12 qubits), on
hardware (≤17–27 qubits, no defeated classical baseline), or purely theoretical.
A **clean, reproducible, ≥20-qubit simulation** that (a) puts the quantum kernel
against a *full* classical baseline suite, (b) reports the `g` diagnostic, (c)
quantifies scaling and statistical significance, and (d) states the engineered-
task caveat honestly, is not collected in one place.

### Our Novel Contribution
A self-contained, reproducible **20-qubit simulation study** that demonstrates
*both sides* of the hypothesis with honest scoping:
1. **Engineered task (Huang 2021):** a projected quantum kernel separates a
   20-qubit task that the best tuned classical models cannot — with the geometric
   difference `g` reported *before* training, repeated-split error bars, a paired
   significance test, a within-quantum control (fidelity vs projected kernel),
   and an accuracy-vs-qubit-count scaling curve.
2. **Discrete-log task (Liu 2021):** off-the-shelf classical ML sits at chance
   across n=12/16/20 bits, while the quantum feature-map kernel (interval states,
   here computed from the simulable log table — what Shor builds on hardware)
   separates the task near-perfectly. We state the asymptotic-hardness caveat.

### Experiment Justification
- **Exp A (engineered projected kernel, n=8/10/20):** directly tests "quantum
  excels where classical struggles" in the *simulable* regime the hypothesis
  names. Needed because it is the only place a ≥20-qubit win is demonstrable.
- **Exp A-control (fidelity vs projected kernel):** tests Huang's claim that the
  naive fidelity kernel concentrates/fails in high-dim Hilbert space — guards
  against attributing the win to "quantumness" rather than the right kernel.
- **Exp A-baselines (RBF/linear SVM, RF, gradient boosting, MLP):** Huang stresses
  beating *non-kernel* classical models, not just RBF. Needed for an honest claim.
- **Exp B (discrete log, n=12/16/20):** shows the *off-the-shelf classical
  failure* the user asked about on a task with a rigorous backing, and exhibits
  the quantum-easy side concretely. Scaling across n shows the trend.

---

## Research Question
Are there tasks where classical ML methods struggle while quantum methods —
simulated at **20 qubits or more** — excel? Demonstrate on real/realistic
datasets with simulation.

## Hypothesis Decomposition
- **H1 (engineered, simulable):** On a quantum-tailored 20-qubit task, a projected
  quantum-kernel SVM achieves substantially higher test accuracy than the best
  tuned classical model. *Measurable:* Δacc > 0.2, paired test p < 0.05.
- **H1a:** The advantage tracks the geometric difference `g` and **widens with
  qubit count** n ∈ {8,10,20}.
- **H1b (control):** The *projected* quantum kernel wins, but the naive *fidelity*
  quantum kernel does not (concentration in high-dim Hilbert space).
- **H2 (off-the-shelf classical failure):** On the discrete-log task, standard
  classical models (SVM/RF/GB/MLP/logreg) stay at chance (~0.50) for n=12/16/20,
  while the quantum interval-state kernel separates the task (acc ≫ 0.5).
- **H2-caveat:** At 20 bits DLP is *not* asymptotically hard; this shows
  off-the-shelf classical ML fails, not a fault-tolerant-scale separation.

## Proposed Methodology

### Approach
Quantum kernels reduce to classical kernel SVMs on a quantum-computed kernel
matrix (Schuld 2021), so we simulate the quantum feature maps in PennyLane
`lightning.qubit` (≥20 qubits verified), build kernel matrices, and train
`sklearn` SVMs on precomputed kernels. Classical baselines see the **same input
features**. We report the `g` diagnostic before training.

### Experimental Steps
1. **Exp A:** regenerate engineered projected-kernel datasets at n=8,10,20 (and a
   fresh seed set for error bars). Compute `g(K_C‖K_Q)`. Train projected-quantum-
   kernel SVM + fidelity-kernel SVM + classical suite over repeated stratified
   splits. Report mean±std, paired t-test (quantum vs best classical),
   generalization gap, accuracy-vs-n curve.
2. **Exp B:** load discrete-log datasets n=12,16,20. Train off-the-shelf classical
   suite (bit features). Build the quantum interval-state fidelity kernel from the
   log table; train precomputed-kernel SVM. Report accuracy vs chance across n.

### Baselines
RBF-SVM (CV-tuned C, γ), linear SVM, random forest, gradient boosting, MLP, and
(Exp B) logistic regression + majority-class dummy. These cover kernel and
non-kernel classical ML per Huang 2021 / lit-review §3.

### Evaluation Metrics
Test accuracy (primary), geometric difference `g`, generalization gap
(train−test), accuracy vs qubit count, paired t-test p-value + Cohen's d.
Accuracy is appropriate: all tasks are class-balanced ~50/50 (verified in meta).

### Statistical Analysis Plan
Repeated stratified train/test splits (≥10 seeds) per model; report mean ± std.
Compare quantum vs best classical with a paired t-test (same splits), α = 0.05,
plus Cohen's d effect size. Bonferroni note for multiple model comparisons.

## Expected Outcomes
- Exp A: projected quantum kernel ≈ 0.85–0.93, best classical ≈ 0.45–0.60, gap
  significant and widening with n; fidelity kernel near chance at n=20.
- Exp B: all classical ≈ 0.50 across n; quantum interval kernel ≫ 0.5.
- Refutation would be: classical matches quantum on Exp A, or quantum kernel also
  fails — would mean the engineered construction / `g` diagnostic did not hold.

## Timeline and Milestones
Planning ✓ → env/data setup (10m) → implement pipelines (40m) → run Exp A+B (40m)
→ analysis + figures (30m) → REPORT.md/README.md (25m). ~30% buffer for debugging.

## Potential Challenges
- **Memory at 20 qubits:** fidelity kernel needs 2^20 statevectors. Mitigation:
  store statevectors in batches / fall back to n≤14 for the fidelity control if
  RAM-bound; projected kernel only needs 3n expvals (cheap).
- **Kernel concentration:** projected-kernel bandwidth γ must sit in the
  separation regime (γ≈0.1, validated). Report median-distance scaling.
- **Honesty on engineered labels:** labels use the full kernel (existence-of-task
  construction). We disclose this and still do proper train/test SVM fitting.
- **DLP quantum side uses log table:** disclose that on hardware Shor builds this
  without precomputing logs; at simulable n it is precomputable.

## Success Criteria
H1 and H2 confirmed with the stated thresholds and a significant paired test on
Exp A; all caveats documented; everything reproducible from `src/` + seeds.
