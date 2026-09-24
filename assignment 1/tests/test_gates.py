import numpy as np
import pytest

from quantum_sim.gates import H, I, S, SX, T, X, Y, Z, rx, ry, rz


STANDARD_GATES = [I, X, Y, Z, H, S, T, SX]


def assert_unitary(U: np.ndarray) -> None:
    assert U.shape == (2, 2)
    assert np.allclose(U.conj().T @ U, np.eye(2), atol=1e-12)


@pytest.mark.parametrize("gate", STANDARD_GATES)
def test_standard_gates_are_unitary(gate):
    assert_unitary(gate)


@pytest.mark.parametrize("rotation", [rx, ry, rz])
@pytest.mark.parametrize("theta", [0.0, np.pi / 7, -np.pi / 3, np.pi, 2 * np.pi])
def test_rotation_gates_are_unitary(rotation, theta):
    assert_unitary(rotation(theta))


@pytest.mark.parametrize("rotation", [rx, ry, rz])
def test_zero_angle_rotation_is_identity(rotation):
    assert np.allclose(rotation(0.0), I)


def test_rx_pi_is_x_up_to_global_phase():
    assert np.allclose(rx(np.pi), -1j * X)


def test_ry_pi_is_y_up_to_global_phase():
    assert np.allclose(ry(np.pi), -1j * Y)


def test_rz_pi_is_z_up_to_global_phase():
    assert np.allclose(rz(np.pi), -1j * Z)


def test_h_squared_is_identity():
    assert np.allclose(H @ H, I)


def test_pauli_squares_are_identity():
    for pauli in (X, Y, Z):
        assert np.allclose(pauli @ pauli, I)


def test_s_is_t_squared():
    assert np.allclose(T @ T, S)


def test_z_is_t_to_the_fourth():
    assert np.allclose(np.linalg.matrix_power(T, 4), Z)


def test_sx_squared_is_x():
    assert np.allclose(SX @ SX, X)


def test_hzh_is_x():
    assert np.allclose(H @ Z @ H, X)
