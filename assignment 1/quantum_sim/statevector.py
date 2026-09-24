"""Statevector representation and gate application."""

import numpy as np

# A useful utility function
def is_unitary(matrix: np.ndarray, atol: float = 1e-9,) -> bool:
    matrix = np.asarray(matrix, dtype=complex)

    if matrix.ndim != 2:
        return False

    if matrix.shape[0] != matrix.shape[1]:
        return False

    identity = np.eye(matrix.shape[0], dtype=complex)

    return np.allclose(
        matrix.conj().T @ matrix,
        identity,
        atol=atol,
    )

class Statevector:
    """A normalized n-qubit pure statevector.

    Qubit indexing convention:
    - Qubit 0 is the least-significant bit of a computational-basis index.
    - For one qubit: [alpha, beta] represents alpha|0> + beta|1>.
    """

    def __init__(self, data: np.ndarray):
        data = np.asarray(data, dtype=complex).flatten()
        n = data.shape[0]
        if n < 2 or (n & (n - 1)) != 0:
            raise ValueError("Statevector length must be a power of 2 and at least 2.")

        norm = np.linalg.norm(data)
        if np.isclose(norm, 0.0):
            raise ValueError("Cannot construct a Statevector from the zero vector.")

        self.data = data / norm  # normalize, don't reject

    @property
    def num_qubits(self) -> int:
        return int(np.log2(self.data.shape[0]))

    @classmethod
    def zero(cls, num_qubits: int = 1) -> "Statevector":
        """Return |0...0>."""
        if num_qubits < 1:
            raise ValueError("Number of qubits must be at least 1.")
        data = np.zeros(2 ** num_qubits, dtype=complex)
        data[0] = 1.0
        return cls(data)

    def copy(self) -> "Statevector":
        """Return a copy of the statevector."""
        return Statevector(self.data.copy())

    def probabilities(self) -> np.ndarray:
        """Return computational-basis probabilities."""
        return np.abs(self.data) ** 2


    def apply_unitary(self, unitary: np.ndarray, target: int = 0) -> None:
        """Apply a 2x2 unitary to one target qubit.

        Assignment TODO:
        Implement the general n-qubit version without constructing the full
        2**n by 2**n matrix.
        """
        
        ## Ensure unitary is a 2 x 2 unitary and the target is a single qubit.
        unitary = np.asarray(unitary, dtype=complex)
        if unitary.shape != (2, 2) or not is_unitary(unitary):
            raise ValueError("Unitary must be a 2x2 unitary matrix.")
        if target < 0 or target >= self.num_qubits:
            raise ValueError("Target qubit index out of range.")
        if not np.isclose(np.linalg.norm(self.data), 1.0):
            raise ValueError("Statevector must be normalized.")
        
        
        ## You only need to define a one-qubit state vector at this point ##
        if self.num_qubits != 1:
            raise NotImplementedError(
                "Multi-qubit gate application is an extension."
            )

        self.data = unitary @ self.data

    def expectation_pauli(self, pauli: np.ndarray, target: int = 0) -> float:
        """Return <psi|P|psi> for a one-qubit Pauli observable."""
        if self.num_qubits != 1 or target != 0:
            raise NotImplementedError("Multi-qubit observables are an extension.")

        pauli = np.asarray(pauli, dtype=complex)
        value = np.vdot(self.data, pauli @ self.data)  # <psi| P |psi>

        # Expectation values of Hermitian observables are real; clean up
        # floating-point imaginary dust and snap to the exact values you'd
        # expect for eigenstates of a Pauli operator.
        value = value.real if np.isclose(value.imag, 0.0, atol=1e-9) else value

        for target_val in (1.0, 0.0, -1.0):
            if np.isclose(value, target_val, atol=1e-9):
                return target_val
        return float(value)