"""Generate one-bit quantum random-number counts with the local simulator.

Run from the project root:
    python examples/quantum_rng.py
"""

import numpy as np

from quantum_sim.gates import H
from quantum_sim.measurement import sample_counts
from quantum_sim.statevector import Statevector


def main() -> None:
    """Prepare |+>, sample it, and print counts and estimated probabilities."""
    shots = 1000
    seed = 42

    state = Statevector.zero(1)
    state.apply_unitary(H, target=0)

    counts = sample_counts(state, shots=shots, seed=seed)
    estimated_probabilities = {bitstring: counts.get(bitstring, 0) / shots for bitstring in ("0", "1")}
    
    print("One-bit quantum random-number generator")
    print(f"Shots: {shots}")
    print(f"Seed: {seed}")
    print()
    print("Prepared statevector |+>:")
    print(np.array2string(state.data, precision=12, suppress_small=True))
    print()
    print("Theoretical computational-basis probabilities:")
    print(np.array2string(state.probabilities(), precision=12, suppress_small=True))
    print()
    print("Measurement counts:")
    print({
        bitstring: counts.get(bitstring, 0)
        for bitstring in ("0", "1")
    })
    print()
    print("Estimated probabilities:")
    for bitstring, probability in estimated_probabilities.items():
        print(f"  P({bitstring}) = {probability:.6f}")


if __name__ == "__main__":
    main()
