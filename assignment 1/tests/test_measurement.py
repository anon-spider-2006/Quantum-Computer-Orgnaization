import numpy as np
import pytest

from quantum_sim.gates import H, X
from quantum_sim.measurement import sample_counts
from quantum_sim.statevector import Statevector


def test_zero_state_always_measures_zero():
    counts = sample_counts(Statevector.zero(), shots=100, seed=7)
    assert counts == {"0": 100}


def test_one_state_always_measures_one():
    state = Statevector.zero()
    state.apply_unitary(X)

    counts = sample_counts(state, shots=100, seed=7)
    assert counts == {"1": 100}


def test_sampling_is_reproducible_with_a_seed():
    state = Statevector.zero()
    state.apply_unitary(H)

    first = sample_counts(state, shots=200, seed=12345)
    second = sample_counts(state, shots=200, seed=12345)

    assert first == second


def test_plus_state_counts_sum_to_shots_and_are_approximately_balanced():
    state = Statevector.zero()
    state.apply_unitary(H)

    shots = 10_000
    counts = sample_counts(state, shots=shots, seed=20260902)

    assert set(counts).issubset({"0", "1"})
    assert sum(counts.values()) == shots
    assert abs(counts.get("0", 0) - shots / 2) < 5 * np.sqrt(shots / 4)
    assert abs(counts.get("1", 0) - shots / 2) < 5 * np.sqrt(shots / 4)


@pytest.mark.parametrize("shots", [0, -1, -25])
def test_sample_counts_rejects_nonpositive_shots(shots):
    with pytest.raises(ValueError):
        sample_counts(Statevector.zero(), shots=shots)
