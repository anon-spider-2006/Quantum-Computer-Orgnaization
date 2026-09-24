import numpy as np
import pytest

from quantum_sim.decomposition import (
    ZYZAngles,
    equivalent_up_to_global_phase,
    is_unitary,
    zyz_decompose,
    zyz_unitary,
)
from quantum_sim.gates import H, I, S, T, X, Y, Z


def reconstruct(angles: ZYZAngles) -> np.ndarray:
    return (
        np.exp(1j * angles.phase)
        * zyz_unitary(angles.theta, angles.phi, angles.lam)
    )


def random_unitary(rng: np.random.Generator) -> np.ndarray:
    A = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    Q, R = np.linalg.qr(A)
    phases = np.diag(R) / np.abs(np.diag(R))
    return Q @ np.diag(phases)


@pytest.mark.parametrize("U", [I, X, Y, Z, H, S, T])
def test_is_unitary_accepts_standard_unitaries(U):
    assert is_unitary(U)


def test_is_unitary_rejects_non_unitary_matrix():
    U = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=complex)
    assert not is_unitary(U)


def test_is_unitary_rejects_non_2_by_2_matrix():
    assert not is_unitary(np.eye(3, dtype=complex))


def test_zyz_unitary_identity_angles():
    assert np.allclose(zyz_unitary(0.0, 0.0, 0.0), I)


def test_zyz_unitary_matches_explicit_product():
    theta = 0.71
    phi = -1.23
    lam = 2.04

    from quantum_sim.gates import ry, rz

    expected = rz(phi) @ ry(theta) @ rz(lam)
    assert np.allclose(zyz_unitary(theta, phi, lam), expected)


@pytest.mark.parametrize("U", [I, X, Y, Z, H, S, T])
def test_standard_gate_decompositions_reconstruct_exactly(U):
    angles = zyz_decompose(U)
    assert np.allclose(reconstruct(angles), U, atol=1e-9)


@pytest.mark.parametrize("seed", range(5))
def test_random_unitary_decompositions_reconstruct_exactly(seed):
    U = random_unitary(np.random.default_rng(seed))
    angles = zyz_decompose(U)
    assert np.allclose(reconstruct(angles), U, atol=1e-9)


def test_zyz_decompose_rejects_wrong_shape():
    with pytest.raises(ValueError):
        zyz_decompose(np.eye(3, dtype=complex))


def test_zyz_decompose_rejects_non_unitary_input():
    U = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=complex)
    with pytest.raises(ValueError):
        zyz_decompose(U)


def test_identity_uses_theta_zero_convention():
    angles = zyz_decompose(I)
    assert np.isclose(angles.theta, 0.0)
    assert np.isclose(angles.phi, 0.0)
    assert np.isclose(angles.lam, 0.0)
    assert np.allclose(reconstruct(angles), I, atol=1e-9)


def test_z_rotation_uses_theta_zero_convention():
    angle = 0.83
    U = zyz_unitary(0.0, 0.0, angle)

    angles = zyz_decompose(U)

    assert np.isclose(angles.theta, 0.0)
    assert np.isclose(angles.phi, 0.0)
    assert np.allclose(reconstruct(angles), U, atol=1e-9)


def test_theta_pi_uses_lambda_zero_convention():
    phi = -0.61
    lam = 1.17
    U = zyz_unitary(np.pi, phi, lam)

    angles = zyz_decompose(U)

    assert np.isclose(angles.theta, np.pi)
    assert np.isclose(angles.lam, 0.0)
    assert np.allclose(reconstruct(angles), U, atol=1e-9)


def test_equivalent_up_to_global_phase_accepts_global_phase_difference():
    U = H
    V = np.exp(1j * 0.73) * H
    assert equivalent_up_to_global_phase(U, V)


def test_equivalent_up_to_global_phase_rejects_distinct_gates():
    assert not equivalent_up_to_global_phase(I, X)


def test_equivalent_up_to_global_phase_rejects_mismatched_shapes():
    assert not equivalent_up_to_global_phase(np.eye(2), np.eye(4))
