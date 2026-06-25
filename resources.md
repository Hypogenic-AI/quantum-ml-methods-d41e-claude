# Resources Catalog — Quantum Methods on ML

Catalog of all resources gathered for the project. Companion to
`literature_review.md` (synthesis) and the per-directory READMEs.

## Summary

| Resource type | Count | Location |
|---|---|---|
| Papers (deep-read, PDF) | 15 | `papers/` |
| Search corpus (ranked, deduped) | 219 | `paper_search_results/all_papers.json` |
| Dataset generators (published constructions) | 2 families | `datasets/generators/` |
| Generated datasets (validated) | 6 (`n`=12/16/20 DLP; `n`=8/10/20 engineered) | `datasets/` |
| Cloned code repos | 2 | `code/` |
| Installed QML frameworks | 4 | `.venv` |

**Bottom line.** The hypothesis is **demonstrable in the simulable regime** —
validated in this workspace: at **20 qubits**, a simulated projected-quantum-kernel
SVM scores **0.89** vs **0.45** for a classical RBF-SVM on the engineered task,
and standard classical ML scores **~0.50** (chance) on the 20-bit discrete-log
task. The honest scope is *engineered / quantum-tailored* tasks; a *naturally
labeled* simulable advantage remains an open problem (see `literature_review.md`).

## Papers

15 deep-read (PDFs + notes). Full subsections in `literature_review.md` §8;
per-file index in `papers/README.md`. Top of the list:

| Title | Year | File | arXiv | Role |
|---|---|---|---|---|
| A rigorous and robust quantum speed-up in supervised ML | 2021 | `liu2021_rigorous_quantum_speedup.pdf` | 2010.02174 | provable DLP separation (not simulable) |
| Power of data in quantum machine learning | 2021 | `huang2021_power_of_data.pdf` | 2011.01938 | **engineered advantage + geom. diff `g`** |
| Supervised learning with quantum-enhanced feature spaces | 2019 | `havlicek2019_quantum_feature_spaces.pdf` | 1804.11326 | foundational QSVM / feature maps |
| Machine learning of high-dim data on a noisy quantum processor | 2021 | `peters2021_projected_kernel_hardware.pdf` | 2101.09581 | NISQ feasibility, supernova data |
| Covariant quantum kernels for data with group structure | 2021 | `glick2021_covariant_kernels.pdf` | 2105.03406 | group-structured data, 27 qubits |
| Quantum machine learning beyond kernel methods | 2023 | `jerbi2023_beyond_kernel.pdf` | 2110.13162 | **data re-uploading; relabeled FMNIST win** |
| Supervised QML models are kernel methods | 2021 | `schuld2021_models_are_kernels.pdf` | 2101.11020 | unifying theory (cautionary) |
| The inductive bias of quantum kernels | 2021 | `kubler2021_inductive_bias_quantum_kernels.pdf` | 2106.03747 | **when kernels help/hurt** (cautionary) |
| The power of quantum neural networks | 2021 | `abbas2021_power_qnn.pdf` | 2011.00027 | effective dimension / trainability |
| Effect of data encoding on expressive power (Fourier) | 2021 | `schuld2021_data_encoding_fourier.pdf` | 2008.08605 | expressivity = Fourier series |
| The expressive power of parameterized quantum circuits | 2020 | `du2020_expressive_power_pqc.pdf` | 1810.11922 | generative expressivity separation |
| Enhancing generative models via quantum correlations | 2022 | `gao2022_quantum_generative.pdf` | 2101.08354 | nonlocality/contextuality |
| Quantum embeddings for machine learning | 2020 | `lloyd2020_quantum_embeddings.pdf` | 2001.03622 | metric learning in Hilbert space |
| Shadows of quantum machine learning | 2023 | `jerbi2023_shadows_qml.pdf` | 2306.00061 | quantum-train/classical-deploy |
| A quantum-inspired classical algorithm for recommendation | 2019 | `tang2019_dequantization_recommendation.pdf` | 1807.04271 | **dequantization** (counter-evidence) |

## Datasets

| Name | Source / paper | Size | Task | Location | Validated separation |
|---|---|---|---|---|---|
| Discrete-log (DLP) | Liu 2021 (2010.02174) | `n`=12/16/20, ~2k pts | binary classification | `datasets/discrete_log/` | classical ~0.50 (chance) |
| Engineered projected-kernel | Huang 2021 (2011.01938) | `n`=8/10/20, ~250 pts | binary classification | `datasets/engineered_kernel/` | **Q 0.89 vs C 0.45 @ n=20** |

Both are **generator-based** (`datasets/generators/*.py`, regenerated via
`datasets/generate_all.sh`); raw arrays git-ignored, metadata + samples kept.
See `datasets/README.md` for generate/load instructions and the full validation
table.

## Code repositories

| Name | URL | Purpose | Location |
|---|---|---|---|
| PennyLane | github.com/PennyLaneAI/pennylane | primary simulator (`lightning.qubit`, kernels) | `code/pennylane/` |
| Qiskit ML | github.com/Qiskit/qiskit-machine-learning | `FidelityQuantumKernel`, `QSVC`, `EffectiveDimension` | `code/qiskit-machine-learning/` |

Installed (venv): `pennylane` 0.45, `qiskit` 2.4.2, `qiskit-machine-learning`
0.9.0, `qiskit-aer` 0.17.2, `scikit-learn` 1.9.0. Referenced paper code:
`sjerbi/QML-beyond-kernel`, `jmkuebler/quantumbias`. See `code/README.md`.

## Resource-gathering notes

**Search strategy.** `paper-finder` (Semantic Scholar backend) with 6 query
angles (advantage, kernels, VQC, dequantization, QNN/trainability, engineered
benchmarks); 219 unique papers ranked by relevance×citations; top 15 downloaded
via the arXiv export API and deep-read in parallel (15-agent workflow).

**Selection criteria.** Prioritized (1) the canonical advantage constructions
(Liu, Huang, Havlíček), (2) the cautionary/conditions papers essential for honest
claims (Kübler, Schuld, Tang), and (3) the most-cited foundations. Skewed toward
2019–2023 with a few seminal earlier works.

**Challenges.** The `paper-finder` service was intermittently flaky (occasional
500s / missing `httpx`); worked around by installing `httpx` and retrying. The
installed `arxiv` PyPI package was non-standard (no `download_pdf`); replaced with
direct arXiv export-API + `requests` download. Two titles failed title-search and
were fetched by arXiv ID. TensorFlow Quantum (used by Huang/Jerbi) was not
installed (heavy); reproduced their methodology in PennyLane instead.

**Gaps / workarounds.** No naturally labeled simulable-advantage dataset exists in
the literature — by design we provide the *engineered* construction the hypothesis
permits, plus the discrete-log task against off-the-shelf classical ML, each with
documented caveats.

## Recommendations for experiment design

1. **Primary dataset**: `engineered_kernel/engk_q20.npz` (projected quantum kernel,
   20 qubits) — the headline simulable demonstration.
2. **Secondary dataset**: `discrete_log/dlp_n20.npz` — off-the-shelf classical
   failure; scale `n` to show the trend.
3. **Baselines**: tuned RBF-SVM, linear SVM, random forest, gradient boosting,
   small MLP — *and* report geometric difference `g` before training.
4. **Quantum models**: projected quantum kernel (preferred), fidelity kernel,
   optional data re-uploading VQC; simulate with PennyLane `lightning.qubit`.
5. **Metrics**: test accuracy, `g`, accuracy-vs-qubit-count, generalization gap
   vs `N`, shots-to-estimate-kernel.
6. **Guardrails**: same features to classical baselines; include non-kernel
   classical models; state that the advantage task is engineered.
