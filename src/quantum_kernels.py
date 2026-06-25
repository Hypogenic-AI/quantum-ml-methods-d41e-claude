"""Quantum feature maps and kernels, simulated with PennyLane lightning.qubit.

Two kernels on the SAME hardware-efficient encoding (RY/RZ angle encoding + CZ
entanglers, `reps` layers), matching datasets/generators/engineered_kernel_dataset.py:

  * projected quantum kernel  K_PQ(x,x') = exp(-gamma * sum_k ||b_k(x)-b_k(x')||^2)
    where b_k = (<X_k>,<Y_k>,<Z_k>) is the qubit-k Bloch vector (1-RDM).
    -> Huang et al. 2021; resists data-dequantization, clearest advantage.

  * fidelity quantum kernel    K_F(x,x') = |<phi(x)|phi(x')>|^2
    -> Havlicek 2019 style; expected to CONCENTRATE (-> identity) at large n.

The projected kernel needs only 3n expectation values per input (cheap at 20q);
the fidelity kernel needs full statevectors (2^n) and is memory-bounded.
"""
import numpy as np
import pennylane as qml


def _encode_ops(x, n_qubits, reps):
    """The shared encoding circuit (identical to the dataset generator)."""
    for r in range(reps):
        for w in range(n_qubits):
            qml.RY(x[w] * (r + 1), wires=w)
            qml.RZ(x[(w + r) % n_qubits], wires=w)
        for w in range(n_qubits - 1):
            qml.CZ(wires=[w, w + 1])
        qml.CZ(wires=[n_qubits - 1, 0])


def make_bloch_fn(n_qubits, reps=2):
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def bloch(x):
        _encode_ops(x, n_qubits, reps)
        return ([qml.expval(qml.PauliX(w)) for w in range(n_qubits)] +
                [qml.expval(qml.PauliY(w)) for w in range(n_qubits)] +
                [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)])

    def blochvecs(X):
        out = np.empty((len(X), n_qubits, 3))
        for i in range(len(X)):
            v = np.array(bloch(X[i]))
            out[i] = np.stack([v[:n_qubits], v[n_qubits:2 * n_qubits],
                               v[2 * n_qubits:]], axis=1)
        return out
    return blochvecs


def make_state_fn(n_qubits, reps=2):
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def state(x):
        _encode_ops(x, n_qubits, reps)
        return qml.state()

    def states(X):
        out = np.empty((len(X), 2 ** n_qubits), dtype=np.complex64)
        for i in range(len(X)):
            out[i] = np.asarray(state(X[i]), dtype=np.complex64)
        return out
    return states


def projected_kernel(B, gamma=0.1, auto_bandwidth=True):
    """B: (N, n_qubits, 3) -> (N,N). gamma relative to median pairwise distance
    when auto_bandwidth (keeps the same regime across qubit counts)."""
    N = B.shape[0]
    flat = B.reshape(N, -1)
    sq = np.sum(flat ** 2, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * flat @ flat.T, 0.0)
    np.fill_diagonal(d2, 0.0)
    if auto_bandwidth:
        med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
        gamma = gamma / (med + 1e-12)
    return np.exp(-gamma * d2)


def fidelity_kernel(states):
    """|<phi_i|phi_j>|^2 from a stack of statevectors. PSD by construction."""
    G = states.conj() @ states.T          # <phi_i|phi_j>
    K = np.abs(G) ** 2
    return np.real(K)
