"""Single-qubit ZYZ Euler decomposition."""

from dataclasses import dataclass
import numpy as np

from .gates import ry, rz


@dataclass(frozen=True)
class ZYZAngles:
    """Angles satisfying U = exp(i phase) Rz(phi) Ry(theta) Rz(lam)."""

    theta: float
    phi: float
    lam: float
    phase: float


def is_unitary(U: np.ndarray, atol: float = 1e-9) -> bool:
    """Return True when U dagger U is approximately identity."""
    U = np.asarray(U)
    if U.ndim != 2 or U.shape != (2, 2):
        return False
    return np.allclose(U.conj().T @ U, np.eye(2), atol=atol)


def zyz_unitary(theta: float, phi: float, lam: float) -> np.ndarray:
    """Return Rz(phi) Ry(theta) Rz(lam)."""
    return rz(phi) @ ry(theta) @ rz(lam)


def _wrap_angle(angle: float) -> float:
    """Wrap an angle to the interval (-pi, pi]."""
    a = float(angle)% (2 * np.pi)
    if a > np.pi:
        a -= 2 * np.pi
    return a

def zyz_decompose(U: np.ndarray, atol: float = 1e-12) -> ZYZAngles:
    """Decompose a 2x2 unitary as exp(i * phase) Rz(phi) Ry(theta) Rz(lam).

    Convention
    ----------
    Returns angles satisfying

        U = exp(1j * phase) @ Rz(phi) @ Ry(theta) @ Rz(lam)

    up to numerical floating-point error, with canonical ranges

        0 <= theta <= pi
        -pi < phi, lam <= pi
        -pi < phase <= pi

    At the Euler-angle singularities, the decomposition is chosen as follows:

    - If theta is approximately 0, return phi = 0 and absorb the combined
      Z rotation into lam.

    - If theta is approximately pi, return lam = 0 and absorb the remaining
      Z rotation into phi.

    Parameters
    ----------
    U:
        A 2x2 unitary NumPy array.

    atol:
        Absolute tolerance used for unitary and singularity checks.

    Returns
    -------
    ZYZAngles
        Angles theta, phi, lam, and global phase satisfying the convention
        above.

    Raises
    ------
    ValueError
        If U is not a 2x2 matrix or is not unitary.
    """

    U = np.asarray(U, dtype=complex)
    if U.shape != (2, 2):
        raise ValueError("U must be a 2x2 matrix.")
    if not is_unitary(U, atol=atol):
        raise ValueError("U must be unitary.")

    # For U = exp(i phase) V with V in SU(2),
    #
    #     det(U) = exp(2 i phase).
    #
    # Taking one square root removes the determinant phase.  The sign
    # ambiguity of the square root corresponds only to an SU(2) factor of -I.

    det_u = U[0, 0] * U[1, 1] - U[0, 1] * U[1, 0]
    phase = 0.5 * np.angle(det_u)
    V = U * np.exp(-1j * phase)

    # Improve numerical behavior: after global-phase removal, force V into
    # the expected SU(2) structure
    #
    #     V = [[alpha, -beta.conjugate()],
    #          [beta,   alpha.conjugate()]]
    #
    # up to floating-point roundoff.

    alpha = 0.5 * (V[0, 0] + V[1, 1].conjugate())
    beta = 0.5 * (V[1, 0] - V[0, 1].conjugate())
    norm = np.sqrt(np.abs(alpha) ** 2 + np.abs(beta) ** 2)
    if norm > 0:
        alpha /= norm
        beta /= norm
    theta = 2 * np.arctan2(np.abs(beta), np.abs(alpha))
    theta_singular_tol = max(atol, 1e-12) ** 0.5

    # Generic case: 0 < theta < pi.
    #
    # For V = Rz(phi) Ry(theta) Rz(lam),
    #
    #     V[0,0] = cos(theta/2) exp[-i(phi + lam)/2]
    #     V[1,0] = sin(theta/2) exp[ i(phi - lam)/2].
    #
    # Therefore:
    #
    #     phi + lam = -2 arg(V[0,0])
    #     phi - lam =  2 arg(V[1,0]).
    #

    if theta_singular_tol < theta < np.pi - theta_singular_tol:
        phi = -np.angle(alpha) + np.angle(beta)
        lam = -np.angle(alpha) - np.angle(beta)

    # theta approximately 0:
    #
    #     Rz(phi) Ry(0) Rz(lam) = Rz(phi + lam).
    #
    # The two Z angles cannot be uniquely separated.  Choose phi = 0 and
    # assign the entire Z rotation to lam.

    elif theta <= theta_singular_tol:
        theta = 0.0
        phi = 0.0
        lam = _wrap_angle(-2 * np.angle(alpha))

    # theta approximately pi:
    #
    #     Rz(phi) Ry(pi) Rz(lam)
    #
    # depends only on phi - lam.  The two Z angles cannot be uniquely
    # separated.  Choose lam = 0 and assign the remaining angle to phi.

    elif theta >= np.pi - theta_singular_tol:
        theta = np.pi
        lam = 0.0
        phi = _wrap_angle(2 * np.angle(beta))

    # Wrap phi and lam into canonical range (-pi, pi] before the candidate
    # check. Shifting an angle by 2*pi flips the sign of Rz because
    # Rz(angle + 2*pi) = -Rz(angle).
    phi = _wrap_angle(phi)
    lam = _wrap_angle(lam)

    # The square root used to isolate `phase` has a sign ambiguity: both
    # phase and phase + pi satisfy det(U) = exp(2i*phase), and theta, phi,
    # lam turn out to be identical either way. Directly test which phase
    # choice actually reconstructs U, rather than guessing from alpha/beta.

    candidate = zyz_unitary(theta, phi, lam)
    if not np.allclose(np.exp(1j * phase) * candidate, U, atol=1e-6):
        phase += np.pi

    return ZYZAngles(
        theta=float(theta),
        phi=float(phi),
        lam=float(lam),
        phase=float(_wrap_angle(phase)),
    )


def equivalent_up_to_global_phase(
    U: np.ndarray,
    V: np.ndarray,
    atol: float = 1e-9,
) -> bool:
    """Test whether U and V differ only by one global phase."""

    U = np.asarray(U, dtype=complex)
    V = np.asarray(V, dtype=complex)
    if U.shape != V.shape:
        return False
    
    idx = np.unravel_index(np.argmax(np.abs(U)), U.shape)
    if np.abs(U[idx]) < atol:
        return np.allclose(V, 0, atol=atol) and np.allclose(U, 0, atol=atol)
    
    phase = V[idx] / U[idx]
    if not np.isclose(np.abs(phase), 1.0, atol=atol):
        return False
    
    return np.allclose(
        V,
        phase * U,
        rtol=0.0,
        atol=atol,
        )