import numpy as np
import pytest

from quantum_sim.gates import H, X
from quantum_sim.statevector import Statevector


def test_zero_returns_zero_state_for_one_qubit():
    state = Statevector.zero()
    assert np.allclose(state.data, np.array([1.0, 0.0], dtype=complex))


def test_zero_returns_all_zero_basis_state_for_two_qubits():
    state = Statevector.zero(2)
    assert np.allclose(state.data, np.array([1.0, 0.0, 0.0, 0.0], dtype=complex))


def test_constructor_normalizes_nonzero_vector():
    state = Statevector(np.array([3.0 + 4.0j, 0.0j]))
    assert np.allclose(state.data, np.array([0.6 + 0.8j, 0.0j]))
    assert np.isclose(np.linalg.norm(state.data), 1.0)


def test_constructor_rejects_zero_vector():
    with pytest.raises(ValueError):
        Statevector(np.array([0.0, 0.0], dtype=complex))


@pytest.mark.parametrize("data", [
    np.array([1.0]),
    np.array([1.0, 0.0, 0.0]),
    np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
])
def test_constructor_rejects_dimension_that_is_not_a_power_of_two(data):
    with pytest.raises(ValueError):
        Statevector(data)


def test_copy_is_independent():
    original = Statevector.zero()
    copied = original.copy()
    copied.apply_unitary(X)

    assert np.allclose(original.data, np.array([1.0, 0.0], dtype=complex))
    assert np.allclose(copied.data, np.array([0.0, 1.0], dtype=complex))


def test_probabilities_of_plus_state():
    state = Statevector.zero()
    state.apply_unitary(H)
    assert np.allclose(state.probabilities(), np.array([0.5, 0.5]))


def test_apply_unitary_applies_x_to_zero_state():
    state = Statevector.zero()
    state.apply_unitary(X)
    assert np.allclose(state.data, np.array([0.0, 1.0], dtype=complex))


def test_apply_unitary_rejects_non_unitary_matrix():
    state = Statevector.zero()
    non_unitary = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=complex)
    with pytest.raises(ValueError):
        state.apply_unitary(non_unitary)


def test_apply_unitary_rejects_wrong_shape():
    state = Statevector.zero()
    with pytest.raises(ValueError):
        state.apply_unitary(np.eye(4, dtype=complex))


def test_expectation_pauli_for_zero_state():
    state = Statevector.zero()
    assert np.isclose(state.expectation_pauli(np.array([[1, 0], [0, -1]])), 1.0)


def test_expectation_pauli_for_plus_state():
    state = Statevector.zero()
    state.apply_unitary(H)
    assert np.isclose(state.expectation_pauli(np.array([[0, 1], [1, 0]])), 1.0)
