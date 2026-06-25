# Code Repositories & Frameworks

The QML simulation stack is **installed in the workspace venv** (`.venv`, managed
by `uv`/`pyproject.toml`) and ready for the experiment runner. Two reference
source repos are cloned here for code/example lookup.

## Installed frameworks (verified to import)

| Package | Version | Use |
|---|---|---|
| `pennylane` + `pennylane-lightning` | 0.45.0 | **Primary** simulator. `lightning.qubit` verified to initialize **22+ qubits**; projected/fidelity kernels, VQCs, data re-uploading |
| `qiskit` / `qiskit-aer` | 2.4.2 / 0.17.2 | Statevector/Aer simulation, `ZZFeatureMap` |
| `qiskit-machine-learning` | 0.9.0 | `FidelityQuantumKernel`, `QSVC`, `QSVR`, `EffectiveDimension` |
| `scikit-learn` | 1.9.0 | classical baselines (SVM, RF, GB, MLP), PCA, `SVC(kernel="precomputed")` |
| `numpy`/`scipy`/`pandas`/`matplotlib` | — | numerics, analysis, plots |

The project's own dataset generators in `../datasets/generators/` implement the
key constructions (discrete-log; projected-quantum-kernel engineered advantage)
directly in PennyLane + NumPy.

## Cloned repositories

### `pennylane/`  (75 MB, shallow clone)
- **URL**: https://github.com/PennyLaneAI/pennylane
- **Purpose**: source + demos for quantum kernels, kernel-based training, and
  variational classifiers. Relevant: `pennylane/kernels/` (kernel utilities,
  target/kernel alignment). The official QML demos (kernel-based training,
  data-reuploading classifier, Fourier expressivity) correspond to several
  papers in `../papers/`.
- **Key entry points**: `qml.kernels.square_kernel_matrix`,
  `qml.kernels.target_alignment`; `qml.device("lightning.qubit", wires=n)`.

### `qiskit-machine-learning/`  (13 MB, shallow clone)
- **URL**: https://github.com/Qiskit/qiskit-machine-learning
- **Purpose**: reference implementations of the Havlíček 2019 methods
  (`ZZFeatureMap`, `FidelityQuantumKernel`, `QSVC`) and `EffectiveDimension`
  (Abbas 2021). See `qiskit_machine_learning/kernels/` and `.../algorithms/`.
- **Key entry points**: `FidelityQuantumKernel`, `QSVC`, `EffectiveDimension`.

## Paper code referenced (not cloned — links for the experiment runner)
- **Jerbi 2023 "beyond kernel"** — https://github.com/sjerbi/QML-beyond-kernel
  (TensorFlow Quantum; relabeled Fashion-MNIST experiment we partly reproduce).
- **Kübler 2021 "inductive bias"** — https://github.com/jmkuebler/quantumbias
  (spectral analysis of quantum kernels).
- **Huang 2021 "Power of data"** — built on TensorFlow Quantum (no standalone
  repo; methodology reproduced in `../datasets/generators/engineered_kernel_dataset.py`).

> TensorFlow Quantum is **not** installed (heavy, TF-pinned). The two papers that
> use it are reproduced here in PennyLane instead, which is lighter and the
> project's primary simulator.

## Quick start for the experiment runner
```python
import pennylane as qml, numpy as np
dev = qml.device("lightning.qubit", wires=20)   # 20-qubit simulation
# projected quantum kernel + engineered labels already in:
#   datasets/generators/engineered_kernel_dataset.py
# classical baselines: sklearn SVC / RandomForestClassifier / GradientBoosting
```
