"""Computational-basis measurement and sampling."""

import numpy as np

from .statevector import Statevector


def sample_counts(
    state: Statevector,
    shots: int,
    seed: int | None = None,
) -> dict[str, int]:
    """Sample full-register computational-basis measurements."""
    if shots <= 0:
        raise ValueError("shots must be positive.")

    ## Get the probabilities of each computational basis state
    rng = np.random.default_rng(seed)

    probs = state.probabilities()
    num_outcomes = len(probs)

    ## Repeatedly measure and collect the results and return the dictinary
    ## and counts

    probs = probs / probs.sum()
    outcomes = rng.choice(num_outcomes, size=shots, p=probs)

    counts: dict[str, int] = {}
    width = state.num_qubits

    for outcome in outcomes:
        bitstring = format(outcome, f"0{width}b")
        counts[bitstring] = counts.get(bitstring, 0) + 1
    
    return counts

