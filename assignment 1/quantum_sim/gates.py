"""Standard one-qubit gate matrices."""

import numpy as np
from typing import TypeAlias

ComplexMatrix: TypeAlias = np.ndarray


def rx(theta: float) -> ComplexMatrix:
    """Return Rx(theta) = exp(-i theta X / 2)."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s],[-1j * s, c]], dtype=complex,)

def ry(theta: float) -> ComplexMatrix:
    """Return Ry(theta) = exp(-i theta Y / 2)."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s],[s, c]], dtype=complex,)

def rz(theta: float) -> ComplexMatrix:
    """Return Rz(theta) = exp(-i theta Z / 2)."""
    return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex)

## The identity gate is defined for you ##
I = np.eye(2, dtype=complex)

## Complete the definitions for the remaining standard gates
X = np.array([[0, 1], [1, 0]], dtype=complex)

Y = np.array([[0, -1j], [1j, 0]], dtype=complex)

Z = np.array([[1, 0], [0, -1]], dtype=complex)

H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)

S = np.array([[1, 0], [0, 1j]], dtype=complex)

T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

SX = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex)


STANDARD_GATES: dict[str, ComplexMatrix] = {
    "id": I,
    "x": X,
    "y": Y,
    "z": Z,
    "h": H,
    "s": S,
    "t": T,
    "sx": SX,
}

