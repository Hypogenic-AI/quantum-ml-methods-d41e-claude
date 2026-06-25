# Downloaded Papers

15 papers, deep-read and annotated (see `../literature_review.md` for full synthesis).
Ordered by relevance to the hypothesis. arXiv PDFs in this directory.

### 1. A rigorous and robust quantum speed-up in supervised machine learning
- **File**: `liu2021_rigorous_quantum_speedup.pdf`
- **Authors / Year**: Yunchao Liu, Srinivasan Arunachalam, Kristan Temme — 2021 (arXiv 2020, Nature Physics 2021)
- **arXiv**: [2010.02174](https://arxiv.org/abs/2010.02174)
- **Contribution**: Constructs a discrete-logarithm-based classification task that is provably hard for any classical learner (no better than random guessing under the DLP hardness assumption) yet is efficiently and noise-robustly learnable by a support vector machine using a quantum-kernel feature map, giving the first end-to-end provable quantum speed-up in supervised learning with only classical data access.
- **Qubits**: n = ceil(log2 p) qubits, where p is a large prime (chosen large enough that DLP is classically intractable; in practice p would be hundreds of bits, so n is in the hundreds-to-thousands of qubits and far exceeds 20). No simulation or hardware experiment was run, so no concrete qubit count <=20 or <=
- **Code**: none (purely theoretical paper; no repository, no simulation, no Qiskit/PennyLane/TFQ implementation reported)
- **Relevance**: High relevance as the foundational theory reference for "classical-hard / quantum-easy" supervised learning, but LOW relevance as direct support for the specific hypothesis of demonstrating advantage via simulation with 20-30 qubits. It rigorously establishes that such separating tasks EXIST (under DLP hardness) and that quantum kernel methods can be provably advantageous and noise-robust; however

### 2. Power of data in quantum machine learning
- **File**: `huang2021_power_of_data.pdf`
- **Authors / Year**: Hsin-Yuan Huang, Michael Broughton, Masoud Mohseni, Ryan Babbush, Sergio Boixo, Hartmut Neven, Jarrod R. McClean (Google Research / Caltech) — 2021
- **arXiv**: [2011.01938](https://arxiv.org/abs/2011.01938)
- **Contribution**: Introduces a geometric-difference test (g) and rigorous prediction-error bounds showing that data can let classical ML match quantum models even on classically-hard-to-compute data, and proposes the projected quantum kernel which empirically beats all tested classical models on engineered datasets up to 30 qubits and gives a provable fault-tolerant speedup on a discrete-log task.
- **Qubits**: Simulation only (no hardware); engineered-dataset experiments scale up to 30 qubits (system size n in plots runs to ~30); original quantum kernel reported only up to n=28 due to simulation cost and poor performance. Well above the 20-qubit threshold of interest.
- **Code**: Built on TensorFlow and TensorFlow-Quantum (Broughton et al. 2020) with scikit-learn for PCA/SVM and Neural Tangents referenced; no explicit public repository link given in the paper text.
- **Relevance**: high -- Directly central to the hypothesis: it is the canonical paper formalizing WHEN classical ML struggles and quantum wins, provides the geometric-difference diagnostic, demonstrates an empirical quantum-over-classical ML separation via simulation at exactly the 20-30 qubit scale of interest, and supplies a concrete recipe (projected quantum kernel + engineered datasets) for constructing such 

### 3. Supervised learning with quantum enhanced feature spaces
- **File**: `havlicek2019_quantum_feature_spaces.pdf`
- **Authors / Year**: Vojtech Havlicek, Antonio D. Corcoles, Kristan Temme, Aram W. Harrow, Abhinav Kandala, Jerry M. Chow, Jay M. Gambetta (IBM T.J. Watson Research Center; MIT) — 2019 (Nature 567:209-212; arXiv:1804.11326, 2018)
- **arXiv**: [1804.11326](https://arxiv.org/abs/1804.11326)
- **Contribution**: Introduces two SVM-style classifiers that encode classical data into a quantum Hilbert space via a conjecturally-hard-to-simulate feature map—a variational quantum classifier and a quantum kernel estimator—and demonstrates them on a 2-qubit superconducting processor with up to 100% classification accuracy.
- **Qubits**: Experiment: 2 qubits (Q0, Q1) on a 5-qubit superconducting transmon device; only n=d=2 used. Theory/algorithms stated for general n qubits with feature space dimension 4^n, but no simulation or hardware runs at >=20 qubits are reported. All concrete results are at n=2.
- **Code**: None provided in the paper (no repository link). Methods later became QSVM / ZZFeatureMap in IBM Qiskit (qiskit-machine-learning), but no code is referenced in the manuscript.
- **Relevance**: High. This is a foundational, widely-cited core paper that directly frames the hypothesis: it proposes encoding classical data into a high-dimensional quantum feature/Hilbert space and argues a quantum advantage can arise specifically when the kernel is classically hard to estimate. It clearly delineates WHEN classical methods would and would not struggle (product/shallow maps = no advantage; conj

### 4. Machine learning of high dimensional data on a noisy quantum processor
- **File**: `peters2021_projected_kernel_hardware.pdf`
- **Authors / Year**: Evan Peters, Joao Caldeira, Alan Ho, Stefan Leichenauer, Masoud Mohseni, Hartmut Neven, Panagiotis Spentzouris, Doug Strain, Gabriel N. Perdue — 2021
- **arXiv**: [2101.09581](https://arxiv.org/abs/2101.09581)
- **Contribution**: Demonstrates a fidelity quantum-kernel SVM on Google's Sycamore superconducting processor that classifies real 67-dimensional supernova data with no dimensionality reduction, achieving test accuracy competitive with noiseless simulation and a classical RBF-SVM despite ~30% circuit fidelity.
- **Qubits**: Hardware: n in {10, 14, 17} qubits on Sycamore (23 active qubits, max 19 assignable with their path connectivity). Noiseless simulation used the same up to 17 qubits. Type 1 ablation simulated 2-17 qubits. Does NOT use >=20 qubits in this paper's own experiments (max 17); the authors cite a companio
- **Code**: No dedicated repository released. Built on Cirq (quantum simulation/execution) and scikit-learn (SVM); references TensorFlow Quantum (TFQ). Hardware accessed via Google Quantum Computing Service (Sycamore). Graph routines via NetworkX.
- **Relevance**: medium - It is a landmark NISQ-hardware quantum-kernel demonstration on real, high-dimensional (67-dim) data and is methodologically central to the quantum-kernel-SVM literature the hypothesis targets. However, relevance to the specific hypothesis (classical struggles, quantum via 20+ qubit simulation excels) is only partial: it uses at most 17 qubits (not >=20), runs primarily on hardware rather 

### 5. Covariant quantum kernels for data with group structure
- **File**: `glick2021_covariant_kernels.pdf`
- **Authors / Year**: Jennifer R. Glick, Tanvi P. Gujarati, Antonio D. Córcoles, Youngseok Kim, Abhinav Kandala, Jay M. Gambetta, Kristan Temme (IBM Quantum) — 2021 (arXiv 2105.03406; v2 Mar 2022)
- **arXiv**: [2105.03406](https://arxiv.org/abs/2105.03406)
- **Contribution**: Introduces covariant quantum kernels for data with group structure—built from a unitary group representation and an optimizable fiducial state—and demonstrates them on a 27-qubit superconducting processor for a coset-labeling classification task.
- **Qubits**: 27 qubits on hardware (ibmq_kolkata, heavy-hexagon), G=SU(2)^⊗27. This is >=20 qubits. The authors explicitly state the circuit is shallow enough to be classically simulable via 2D tensor-network methods [37], so 27 qubits here does NOT imply classical intractability.
- **Code**: none (no repository or framework linked in the read chunks; experiments on IBM Quantum hardware with SPSA-based kernel alignment).
- **Relevance**: High relevance to the hypothesis as a methodological and conceptual anchor: it defines a structured class (group/coset data) argued to be a natural fit for quantum advantage and connects to a proven hardness-based separation. But for the specific simulable (<=30-qubit) "quantum wins where classical fails" claim, its evidence is LOW—the 27-qubit demo is explicitly simulable and on a fabricated zero

### 6. Quantum Machine Learning Beyond Kernel Methods
- **File**: `jerbi2023_beyond_kernel.pdf`
- **Authors / Year**: Sofiene Jerbi, Lukas J. Fiderer, Hendrik Poulsen Nautrup, Jonas M. Kübler, Hans J. Briegel, Vedran Dunjko — 2023
- **arXiv**: [2110.13162](https://arxiv.org/abs/2110.13162)
- **Contribution**: Unifies explicit, implicit (kernel), and data re-uploading quantum models as linear models in quantum feature spaces, and proves exponential resource separations (in qubits and data) showing that data re-uploading and explicit models can dramatically outperform quantum kernel methods, which overfit despite the representer theorem's training-loss guarantee.
- **Qubits**: Numerical simulations span n=2 to 12 qubits (no hardware; TensorFlow Quantum statevector simulation; no run uses >=20 qubits). Theoretical separations are stated asymptotically in d (qubits scale as Omega(d), unbounded). Data re-uploading construction for parities uses a single qubit.
- **Code**: TensorFlow Quantum; code at https://github.com/sjerbi/QML-beyond-kernel (classical baselines via scikit-learn, ADAM optimizer).
- **Relevance**: High. Directly central to the hypothesis: it studies, both theoretically and via <=12-qubit simulation, tasks where quantum (especially data re-uploading and explicit) models outperform alternatives, and empirically shows a quantum-over-classical test-error gap on a simulable task. Importantly for the hypothesis, it is a sharp cautionary reference: it shows the parity separations are classically e

### 7. Supervised quantum machine learning models are kernel methods
- **File**: `schuld2021_models_are_kernels.pdf`
- **Authors / Year**: Maria Schuld (Xanadu) — 2021
- **arXiv**: [2101.11020](https://arxiv.org/abs/2101.11020)
- **Contribution**: A technical manuscript proving that essentially all supervised quantum models (data-encoding circuit + measurement) are mathematically equivalent to classical kernel methods whose kernel is computed on a quantum computer, so the data-encoding strategy (which fixes the kernel) is the only ingredient that can possibly separate quantum from classical ML.
- **Qubits**: Theoretical: encodings use O(n) qubits (basis, amplitude, rotation) or O(rn) (repeated amplitude); general feature space is 2^n x 2^n. No specific hardware runs and no simulations at fixed large qubit counts; toy illustrations are single-qubit (n=1) or few-qubit. No experiments at >=20 qubits.
- **Code**: No paper-specific repository. Uses PennyLane (Bergholm et al. 2018, ref. [23]) for kernel-plot simulations; references a public PennyLane demo on kernel-based training (pennylane.ai/qml/demos/tutorial_kernel_based_training.html).
- **Relevance**: High - this is a foundational/conceptual anchor for the hypothesis, but as a cautionary one. It rigorously frames WHERE any quantum-vs-classical ML separation can come from (only a classically-hard-to-compute data-encoding kernel) and makes clear that such separations are conjectural and that common encodings are dequantizable. It is essential context for critically scoping claims of "quantum exce

### 8. The Inductive Bias of Quantum Kernels
- **File**: `kubler2021_inductive_bias_quantum_kernels.pdf`
- **Authors / Year**: Jonas M. Kübler, Simon Buchholz, Bernhard Schölkopf — 2021
- **arXiv**: [2106.03747](https://arxiv.org/abs/2106.03747)
- **Contribution**: A spectral analysis of quantum kernels showing that an expressive (exponentially large) quantum feature space destroys generalization, so a quantum advantage in kernel learning requires encoding a problem-specific inductive bias that is classically hard to encode — which the authors argue is plausible only for quantum-generated data, not generic classical datasets.
- **Qubits**: Simulation only (full state vector), d = 1 to 7 qubits. No quantum hardware. No experiments at >=20 qubits; theoretical claims are asymptotic in d but empirical demonstration is limited to <=7 qubits.
- **Code**: Yes — https://github.com/jmkuebler/quantumbias (builds on standard open-source packages, refs [50,51]; full-state simulation).
- **Relevance**: High. The paper directly addresses the hypothesis (tasks where classical struggles and quantum wins) and provides a rigorous spectral framework for *when* such an advantage can exist. It is a crucial cautionary/conditions paper: it shows the advantage requires a classically-inexpressible inductive bias, demonstrates it only in a constructed synthetic task at <=7 simulated qubits, and identifies th

### 9. The power of quantum neural networks
- **File**: `abbas2021_power_qnn.pdf`
- **Authors / Year**: Amira Abbas, David Sutter, Christa Zoufal, Aurelien Lucchi, Alessio Figalli, Stefan Woerner — 2021 (arXiv 2020, Nature Computational Science 2021)
- **arXiv**: [2011.00027](https://arxiv.org/abs/2011.00027)
- **Contribution**: Introduces the Fisher-information-based "effective dimension" as a data-aware capacity measure, proves a generalisation bound for it, and uses it (plus the Fisher spectrum) to argue that a QNN with a hard-to-simulate feature map achieves higher effective dimension and faster/more resilient training than comparable classical neural networks.
- **Qubits**: Simulation: S = s_in qubits with s_in in {4,6,8,10}, i.e. 4-10 qubits (all <20). Hardware: 4 qubits used on the 27-qubit ibmq_montreal device. No experiment reaches 20+ qubits.
- **Code**: No dedicated repository given; implemented in Qiskit (IBM Quantum Experience), classical nets standard. scikit-learn used for make_blobs/Iris.
- **Relevance**: High. It is a foundational, frequently-cited theoretical/empirical argument that a well-designed QNN can be "more powerful" (higher effective dimension, better-conditioned Fisher landscape, barren-plateau resilience) than comparable classical nets, directly addressing the hypothesis. Relevance is tempered for the specific "20+ qubit simulation where classical struggles" angle because experiments t

### 10. The effect of data encoding on the expressive power of variational quantum machine learning models
- **File**: `schuld2021_data_encoding_fourier.pdf`
- **Authors / Year**: Maria Schuld, Ryan Sweke, Johannes Jakob Meyer — 2021
- **arXiv**: [2008.08605](https://arxiv.org/abs/2008.08605)
- **Contribution**: Shows that variational quantum models with Hamiltonian time-evolution data encoding can be written exactly as partial Fourier series whose accessible frequencies are set by the encoding-gate eigenvalues and whose coefficients are set by the rest of the circuit, yielding an asymptotic universal-function-approximation result.
- **Qubits**: No fixed/large qubit count; this is a theory paper. Demonstrations use a single qubit (Fig. 3) or r qubits for parallel Pauli encodings with r ∈ {1,3,5} (Fig. 4). Universality is an asymptotic statement requiring the number of subsystems m→∞ (m=K subsystems to reach degree K). No experiments at or a
- **Code**: Experiments performed with the PennyLane software library (Ref. [36]); no dedicated repository link given in the read chunks. Framework: PennyLane.
- **Relevance**: High relevance as a foundational/theory reference, low as direct evidence. It is a canonical paper for understanding what function classes 20+ qubit simulable variational models can express (partial Fourier series), which is essential for designing tasks and reasoning about expressivity/frequency matching. But it provides no empirical or proven classical-vs-quantum advantage, and its framing actua

### 11. The Expressive Power of Parameterized Quantum Circuits
- **File**: `du2020_expressive_power_pqc.pdf`
- **Authors / Year**: Yuxuan Du, Min-Hsiu Hsieh, Tongliang Liu, Dacheng Tao — 2018 (arXiv:1810.11922; published Phys. Rev. Research 2, 033125, 2020)
- **arXiv**: [1810.11922](https://arxiv.org/abs/1810.11922)
- **Contribution**: Proves that parameterized quantum circuits (MPQCs) have provably stronger expressive power than classical generative neural networks (DBM, RBM) and tensor-network PQCs for generative modeling, unless the polynomial hierarchy collapses, and introduces a post-selection-based Bayesian Quantum Circuit (BQC) for learning prior distributions.
- **Qubits**: All experiments are small simulations on Rigetti's pyQuil Quantum Virtual Machine (QVM): BAS 2x2 uses 7 qubits (4 data + 3 ancilla), BAS 3x3 uses 13 qubits (9 data + 4 ancilla), prior-learning uses 8 qubits (7 data + 1 ancilla). No hardware runs. No experiment reaches 20 qubits; the >=20-qubit regim
- **Code**: No public repository. Experiments implemented in Python via pyQuil on Rigetti Forest QVM; paper states "Python codes for our experiments are available upon request." Framework: pyQuil / Rigetti Forest.
- **Relevance**: Medium-high. Directly on-topic for the hypothesis as a foundational theoretical pillar: it gives one of the cleanest arguments that PQC generative models are strictly more expressive than classical neural generative models (RBM/DBM) under PH-non-collapse. However, relevance to the SPECIFIC hypothesis (simulable 20+-qubit quantum methods empirically beating classical ML) is limited because all sepa

### 12. Enhancing Generative Models via Quantum Correlations
- **File**: `gao2022_quantum_generative.pdf`
- **Authors / Year**: Xun Gao, Eric R. Anschuetz, Sheng-Tao Wang, J. Ignacio Cirac, Mikhail D. Lukin — 2022 (arXiv 2101.08354, Jan 2021; published Phys. Rev. X 2022)
- **arXiv**: [2101.08354](https://arxiv.org/abs/2101.08354)
- **Contribution**: Proves that quantum correlations (nonlocality and contextuality) give a minimal "basis-enhanced" quantum extension of Bayesian networks unconditionally greater expressive power than classical Bayesian networks/HMMs, and demonstrates the gap numerically on real ML datasets via classically-simulable tensor networks.
- **Qubits**: No large hardware run and no explicit >=20-qubit simulation count is reported. Constructions are 1D and classically simulable: the basis-enhanced 2-gram is a cluster/GHZ state where each variable = 2 qubits (so the n=16/23/57-variable datasets correspond to roughly 2n ~ 32/46/114 logical qubits wort
- **Code**: None explicitly linked in the read sections; custom Riemannian/unitary gradient-descent training of MPS/MPO tensor networks (method adapted from their Ref. [61]). No Qiskit/PennyLane/TFQ repo mentioned.
- **Relevance**: Medium. Highly relevant conceptually: it is one of the cleanest results identifying WHY quantum models can outdo classical generative models (nonlocality/contextuality) and shows an empirical expressivity gap on real datasets. But it is medium rather than high for the specific hypothesis because the demonstrated advantage uses classically-simulable 1D tensor networks (quantum-inspired classical me

### 13. Quantum embeddings for machine learning
- **File**: `lloyd2020_quantum_embeddings.pdf`
- **Authors / Year**: Seth Lloyd, Maria Schuld, Aroosa Ijaz, Josh Izaac, Nathan Killoran — 2020
- **arXiv**: [2001.03622](https://arxiv.org/abs/2001.03622)
- **Contribution**: Proposes "quantum metric learning": instead of training a variational measurement, train the quantum feature map (embedding) to maximally separate data classes in Hilbert space, after which the optimal classifying measurement is known in closed form (Helstrom for trace/l1 distance, fidelity/overlap for Hilbert-Schmidt/l2 distance).
- **Qubits**: Very small in experiments: 1-qubit and 2-qubit embeddings (simulated analytically in PennyLane). No experiment uses >=20 qubits. Larger numbers appear only in feasibility/illustrative discussion: a 2^50 ~ 10^15 dim embedding needing 50 controlled-SWAPs on 101 qubits; near-term hardware estimates cit
- **Code**: No dedicated repo released; experiments implemented in the PennyLane hybrid-optimization framework (Bergholm et al., arXiv:1811.04968). Classifiers simulated analytically.
- **Relevance**: Medium. Highly relevant conceptually—it gives a clean, analytically grounded framework (metric learning in Hilbert space with known-optimal measurements) that is well suited to simulation and near-term devices, and motivates why quantum embeddings could access classifications classical models cannot. But it provides no empirical quantum-advantage evidence, no classical baselines, and uses only 1-2

### 14. Shadows of quantum machine learning
- **File**: `jerbi2023_shadows_qml.pdf`
- **Authors / Year**: Sofiene Jerbi, Casper Gyurik, Simon C. Marshall, Riccardo Molteni, Vedran Dunjko — 2023 (v2 2024)
- **arXiv**: [2306.00061](https://arxiv.org/abs/2306.00061)
- **Contribution**: Introduces "shadow models" — quantum ML models trained on a quantum computer but deployed entirely classically via shadow tomography of a "flipped" linear model — and proves they retain a provable learning advantage over fully classical learners (under cryptographic assumptions) while being strictly weaker than fully quantum models.
- **Qubits**: Theoretical, parameter n (generic n-qubit / m=n+1-qubit constructions); no specific simulated or hardware qubit count is run. No experiments at 20+ qubits (or any qubit count) were performed — the paper is entirely analytical.
- **Code**: None. No repository, no Qiskit/PennyLane/TFQ implementation; explicitly no data or code (theoretical paper).
- **Relevance**: Medium. Highly relevant conceptually — it rigorously establishes that a quantum-trained, classically-deployed model can solve a task classical ML provably cannot (supporting the "classical struggles, quantum wins" premise) and clarifies the BPP < BPP/qgenpoly < BQP landscape. But low on the specific operational angle of the hypothesis: there are no simulations, no 20+-qubit experiments, no dataset

### 15. A quantum-inspired classical algorithm for recommendation systems
- **File**: `tang2019_dequantization_recommendation.pdf`
- **Authors / Year**: Ewin Tang — 2019
- **arXiv**: [1807.04271](https://arxiv.org/abs/1807.04271)
- **Contribution**: A classical ("quantum-inspired") sampling algorithm that solves the recommendation-systems task in time polylogarithmic in matrix dimensions, "dequantizing" the Kerenidis-Prakash quantum recommendation algorithm and proving it gives no exponential quantum speedup.
- **Qubits**: None. No quantum circuits are simulated or run; no qubit counts appear. The entire contribution is a classical algorithm. The referenced quantum algorithm (KP17b) is discussed only abstractly.
- **Code**: none (theoretical paper; no implementation, repository, or framework provided).
- **Relevance**: high - directly relevant as counter-evidence. It is the seminal dequantization paper showing that a flagship QML "exponential speedup" task is in fact efficiently classically solvable, sharpening any claim that quantum methods excel where classical ones fail and establishing the l2-norm-sampling framework needed to fairly compare QML and classical ML.

