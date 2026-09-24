"""Demonstrate ZYZ decompositions of standard and random one-qubit unitaries.

Run from the project root:
    python examples/euler_demo.py
"""
import json
from pathlib import Path

import numpy as np

from quantum_sim.decomposition import (
    equivalent_up_to_global_phase,
    zyz_decompose,
    zyz_unitary,
    ZYZAngles,
    is_unitary
)
from quantum_sim.gates import H, I, S, SX, T, X, Y, Z, rx, ry, rz
from quantum_sim.measurement import sample_counts
from quantum_sim.statevector import Statevector

def complex_to_json(z: complex) -> dict:
    return {"real": float(np.real(z)), "imag": float(np.imag(z))}


def state_to_json(vector: np.ndarray) -> list:
    return [complex_to_json(x) for x in np.asarray(vector).flatten()]


def matrix_to_json(matrix: np.ndarray) -> list:
    matrix = np.asarray(matrix)
    return [[complex_to_json(x) for x in row] for row in matrix]


def probs_to_json(probabilities: np.ndarray) -> list:
    return [float(p) for p in np.asarray(probabilities).flatten()]

# Build a dictionary of the required tests.

def build_euler_tests() -> list[dict]:
    standard_gates = {
        "I": I,
        "X": X,
        "Y": Y,
        "Z": Z,
        "H": H,
        "S": S,
        "T": T,
    }

    tests = []

    def euler_entry(name: str, U: np.ndarray) -> dict:
        angles, reconstructed = reconstructed_unitary(U)
        equivalent = equivalent_up_to_global_phase(U, reconstructed)
        max_error = float(np.max(np.abs(np.asarray(U) - reconstructed)))
        return {
            "name": name,
            "input_matrix": matrix_to_json(U),
            "theta": float(angles.theta),
            "phi": float(angles.phi),
            "lam": float(angles.lam),
            "phase": float(angles.phase),
            "reconstructed_matrix": matrix_to_json(reconstructed),
            "equivalent_up_to_global_phase": bool(equivalent),
            "maximum_absolute_error": max_error,
            "passed": bool(equivalent),
        }

    for name, U in standard_gates.items():
        tests.append(euler_entry(name.lower(), U))
    tests.append(euler_entry("rz_0p83_theta_zero_case", rz(0.83)))
    special_matrix = rz(-0.61) @ ry(np.pi) @ rz(1.17)
    tests.append(euler_entry("rz_ry_pi_rz_theta_pi_case", special_matrix))
    rng = np.random.default_rng(42)
    for index in range(1, 6):
        U = random_unitary(rng)
        tests.append(euler_entry(f"random_unitary_{index}", U))

    return tests


# Build all the required gate tests.

def build_gate_tests() -> list[dict]:
    gates = {
        "I": I,
        "X": X,
        "Y": Y,
        "Z": Z,
        "H": H,
        "S": S,
        "T": T,
        "SX": SX,
    }

    tests = []

    for name, U in gates.items():
        state = Statevector.zero(1)
        state.apply_unitary(U, target=0)
        unitary_ok = is_unitary(U, atol=1e-9)
        tests.append({
            "name": f"gate_{name.lower()}",
            "matrix": matrix_to_json(U),
            "is_unitary": bool(unitary_ok),
            "output_state": state_to_json(state.data),
            "probabilities": probs_to_json(state.probabilities()),
            "passed": bool(unitary_ok),
        })

    def identity_entry(name, left_expr, right_expr, obtained, expected):
        equivalent = equivalent_up_to_global_phase(obtained, expected, atol=1e-9)
        return {
            "name": name,
            "input": {
                "left_expression": left_expr,
                "right_expression": right_expr,
            },
            "obtained_matrix": matrix_to_json(obtained),
            "expected_matrix": matrix_to_json(expected),
            "equivalent_up_to_global_phase": bool(equivalent),
            "passed": bool(equivalent),
        }

    tests.append(identity_entry("h_squared_equals_identity", "H @ H", "I", H @ H, I))
    tests.append(identity_entry("x_squared_equals_identity", "X @ X", "I", X @ X, I))
    tests.append(identity_entry("y_squared_equals_identity", "Y @ Y", "I", Y @ Y, I))
    tests.append(identity_entry("z_squared_equals_identity", "Z @ Z", "I", Z @ Z, I))
    tests.append(identity_entry("t_squared_equals_s", "T @ T", "S", T @ T, S))
    tests.append(identity_entry("t_fourth_equals_identity", "T @ T @ T @ T", "I", T @ T @ T @ T, I))
    tests.append(identity_entry("sx_squared_equals_x", "SX @ SX", "X", SX @ SX, X))
    tests.append(identity_entry("hzh_equals_x", "H @ Z @ H", "X", H @ Z @ H, X))
    for name, rot, label in (("rx", rx, "Rx"), ("ry", ry, "Ry"), ("rz", rz, "Rz")):
        tests.append(identity_entry(f"{name}_zero_equals_identity", f"{label}(0)", "I", rot(0.0), I))
    tests.append(identity_entry("rx_pi_equivalent_to_minus_i_x", "Rx(pi)", "-i * X", rx(np.pi), -1j * X))
    tests.append(identity_entry("ry_pi_equivalent_to_minus_i_y", "Ry(pi)", "-i * Y", ry(np.pi), -1j * Y))
    tests.append(identity_entry("rz_pi_equivalent_to_minus_i_z", "Rz(pi)", "-i * Z", rz(np.pi), -1j * Z))
    
    return tests


# Build all the measurement tests

def build_measurement_tests() -> list[dict]:

    tests = []
    shots = 1000
    seed = 42

    def measurement_entry(name, state, preparation_operations, theoretical_probs):
        counts = sample_counts(state, shots=shots, seed=seed)
        obtained_counts = {b: counts.get(b, 0) for b in ("0", "1")}
        return {
            "name": name,
            "input_state": state_to_json(state.data),
            "preparation_operations": preparation_operations,
            "shots": shots,
            "seed": seed,
            "theoretical_probabilities": probs_to_json(theoretical_probs),
            "obtained_counts": obtained_counts,
            "total_counts": sum(obtained_counts.values()),
            "passed": sum(obtained_counts.values()) == shots,
        }

    state = Statevector.zero(1)
    tests.append(measurement_entry("zero_state_measurement", state, [], np.array([1.0, 0.0])))
    state = Statevector.zero(1)
    state.apply_unitary(X, target=0)
    tests.append(measurement_entry("one_state_measurement", state, ["X"], np.array([0.0, 1.0])))
    state = Statevector.zero(1)
    state.apply_unitary(H, target=0)
    tests.append(measurement_entry("plus_state_measurement", state, ["H"], np.array([0.5, 0.5])))
    state = Statevector.zero(1)
    state.apply_unitary(ry(np.pi / 3), target=0)
    tests.append(measurement_entry("ry_pi_over_3_measurement", state, ["Ry(pi/3)"], np.array([0.75, 0.25])))

    return tests


# Build the non-commutativity test.

def build_noncommutativity_test() -> dict:
    """Demonstrate that different rotation orderings generally differ."""
    theta = float(np.pi / 3)
    phi = float(np.pi / 5)

    U_zy = rz(phi) @ ry(theta)
    U_yz = ry(theta) @ rz(phi)

    state_zy = Statevector.zero(1)
    state_zy.apply_unitary(U_zy, target=0)

    state_yz = Statevector.zero(1)
    state_yz.apply_unitary(U_yz, target=0)

    unitaries_equivalent = equivalent_up_to_global_phase(U_zy, U_yz, atol=1e-9)
    states_equivalent = equivalent_up_to_global_phase(state_zy.data, state_yz.data, atol=1e-9)

    lam = float(np.pi / 7)
    U_alt = ry(theta) @ rz(phi) @ rz(lam)
    state_alt = Statevector.zero(1)
    state_alt.apply_unitary(U_alt, target=0)

    return {
        "theta": theta,
        "phi": phi,
        "zy_unitary": matrix_to_json(U_zy),
        "yz_unitary": matrix_to_json(U_yz),
        "zy_output_state": state_to_json(state_zy.data),
        "yz_output_state": state_to_json(state_yz.data),
        "zy_probabilities": probs_to_json(state_zy.probabilities()),
        "yz_probabilities": probs_to_json(state_yz.probabilities()),
        "unitaries_equivalent_up_to_global_phase": bool(unitaries_equivalent),
        "states_equivalent_up_to_global_phase": bool(states_equivalent),
        "alternate_ordering": {
            "expression": "Ry(theta) @ Rz(phi) @ Rz(lambda)",
            "equivalent_to_zy_unitary": bool(equivalent_up_to_global_phase(U_alt, U_zy, atol=1e-9)),
            "equivalent_to_zy_state": bool(equivalent_up_to_global_phase(state_alt.data, state_zy.data, atol=1e-9)),
        },
    }

#
# The required statevector tests from section 1.5.5
#
def build_statevector_tests() -> list[dict]:
    """Build the required statevector and expectation-value records."""
    tests = []

    # 1. Construct |0>.
    state = Statevector.zero(1)
    expected_state = np.array([1.0, 0.0], dtype=complex)
    tests.append({
        "name": "construct_zero_state",
        "input_state": state_to_json(state.data),
        "operations": [],
        "obtained_state": state_to_json(state.data),
        "obtained_probabilities": probs_to_json(state.probabilities()),
        "expected_state": state_to_json(expected_state),
        "expected_probabilities": probs_to_json(np.array([1.0, 0.0])),
        "passed": bool(np.allclose(state.data, expected_state, atol=1e-9)),
    })
    
    # 2. Normalize [3 + 4i, 0].
    raw = np.array([3 + 4j, 0], dtype=complex)
    sv_norm = Statevector(raw)
    expected_normalized = np.array([0.6 + 0.8j, 0.0], dtype=complex)
    tests.append({
        "name": "normalize_3_plus_4i_state",
        "input_state": state_to_json(raw),
        "operations": ["normalize"],
        "obtained_state": state_to_json(sv_norm.data),
        "obtained_probabilities": probs_to_json(sv_norm.probabilities()),
        "expected_state": state_to_json(expected_normalized),
        "expected_probabilities": probs_to_json(np.array([1.0, 0.0])),
        "passed": bool(np.allclose(sv_norm.data, expected_normalized, atol=1e-9)),
    })

    # 3. X|0> = |1>.
    input_state = state.data.copy()
    state.apply_unitary(X, target=0)
    expected_state = np.array([0.0, 1.0], dtype=complex)
    tests.append({
        "name": "x_on_zero",
        "input_state": state_to_json(input_state),
        "operations": ["X"],
        "obtained_state": state_to_json(state.data),
        "obtained_probabilities": probs_to_json(state.probabilities()),
        "expected_state": state_to_json(expected_state),
        "expected_probabilities": probs_to_json(np.array([0.0, 1.0])),
        "passed": bool(np.allclose(state.data, expected_state, atol=1e-9)),
    })

    # 4. H|0> = |+>.
    state.apply_unitary(H, target=0)
    expected_state = np.array([1, 1], dtype=complex) / np.sqrt(2)
    tests.append({
        "name": "h_on_zero",
        "input_state": state_to_json(input_state),
        "operations": ["H"],
        "obtained_state": state_to_json(state.data),
        "obtained_probabilities": probs_to_json(state.probabilities()),
        "expected_state": state_to_json(expected_state),
        "expected_probabilities": probs_to_json(np.array([0.5, 0.5])),
        "passed": bool(np.allclose(state.data, expected_state, atol=1e-9)),
    })

    # 5. H^2|0> = |0>.
    state.apply_unitary(H, target=0)
    state.apply_unitary(H, target=0)
    expected_state = np.array([1.0, 0.0], dtype=complex)
    tests.append({
        "name": "h_squared_on_zero",
        "input_state": state_to_json(input_state),
        "operations": ["H", "H"],
        "obtained_state": state_to_json(state.data),
        "obtained_probabilities": probs_to_json(state.probabilities()),
        "expected_state": state_to_json(expected_state),
        "expected_probabilities": probs_to_json(np.array([1.0, 0.0])),
        "passed": bool(np.allclose(state.data, expected_state, atol=1e-9)),
    })

    # 6. <0|Z|0> = 1.
    obtained = state.expectation_pauli(Z, target=0)
    tests.append({
        "name": "expectation_z_on_zero",
        "input_state": state_to_json(state.data),
        "observable": "Z",
        "obtained_expectation": float(obtained),
        "expected_expectation": 1.0,
        "passed": bool(np.isclose(obtained, 1.0, atol=1e-9)),
    })

    # 7. <+|X|+> = 1.
    state = Statevector.zero(1)
    state.apply_unitary(H, target=0)
    obtained_x = state.expectation_pauli(X, target=0)
    tests.append({
        "name": "expectation_x_on_plus",
        "input_state": state_to_json(state.data),
        "observable": "X",
        "obtained_expectation": float(obtained_x),
        "expected_expectation": 1.0,
        "passed": bool(np.isclose(obtained_x, 1.0, atol=1e-9)),
    })

    return tests

# Use rgn.normal to generate randome unitarys
def random_unitary(rng: np.random.Generator) -> np.ndarray:
    """Return a random 2x2 unitary using QR factorization."""
    A = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    Q, R = np.linalg.qr(A)
    phases = np.diag(R) / np.abs(np.diag(R))
    return Q @ np.diag(phases)

def reconstructed_unitary(U: np.ndarray) -> tuple[ZYZAngles, np.ndarray]:
    """Decompose U and reconstruct it from the returned parameters."""
    angles = zyz_decompose(U)
    reconstructed = (np.exp(1j * angles.phase) * zyz_unitary(angles.theta, angles.phi, angles.lam))
    return angles, reconstructed


def print_result(name: str, U: np.ndarray) -> None:
    angles, reconstructed = reconstructed_unitary(U)
    equivalent = equivalent_up_to_global_phase(U, reconstructed)
    exact = np.allclose(U, reconstructed, atol=1e-9)

    print(f"{name}:")
    print(f"  theta = {angles.theta:+.12f}")
    print(f"  phi   = {angles.phi:+.12f}")
    print(f"  lam   = {angles.lam:+.12f}")
    print(f"  phase = {angles.phase:+.12f}")
    print(f"  equivalent up to global phase: {equivalent}")
    print(f"  exact reconstruction:          {exact}")
    print(f"  max |U - U_reconstructed|:     "
          f"{np.max(np.abs(U - reconstructed)):.3e}")
    print()


def main() -> None:
    standard_gates = {
        "I": I,
        "X": X,
        "Y": Y,
        "Z": Z,
        "H": H,
        "S": S,
        "T": T,
    }

    print("ZYZ decomposition of standard gates")
    print("=" * 46)
    
    for name, U in standard_gates.items():
        print_result(name, U)

    rng = np.random.default_rng(42)
    
    print("ZYZ decomposition of five random one-qubit unitaries")
    print("=" * 55)
    for index in range(1, 6):
        print_result(
            f"Random unitary {index}",
            random_unitary(rng)
            )

    print("Alternative rotation ordering")
    print("============================")

    alternate = build_noncommutativity_test()
    alternate_data = alternate["alternate_ordering"]

    print(
        "Expression:",
        alternate_data["expression"],
    )
    print(
        "Equivalent to ZY unitary:",
        alternate_data["equivalent_to_zy_unitary"],
    )
    print(
        "Equivalent output state:",
        alternate_data["equivalent_to_zy_state"],
    )
    
    euler_tests = build_euler_tests()
    gate_tests = build_gate_tests()
    measurement_tests = build_measurement_tests()
    state_vector_tests = build_statevector_tests()
    noncommutativity_test = build_noncommutativity_test()

    results = {
        "assignment": "assignment2",
        "measurement_seed": 42,
        "statevector_tests": state_vector_tests,
        "gate_tests": gate_tests,
        "measurement_tests": measurement_tests,
        "euler_decomposition_tests": euler_tests,
        "noncommutativity_test": noncommutativity_test,
    }

    output_path = (
        Path(__file__).resolve().parent
        / "assignment2_results.json"
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=2,
            allow_nan=False,
        )
        file.write("\n")

    print("Wrote", output_path)        

        

if __name__ == "__main__":
    main()
