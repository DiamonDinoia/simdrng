"""Basic usage — simdrng is a drop-in for np.random."""

import numpy as np
import simdrng


def main() -> None:
    # Every factory returns a real np.random.Generator
    rng = simdrng.XoshiroSIMD(42)
    assert isinstance(rng, np.random.Generator)

    # Standard NumPy API — fast path fires transparently
    print("uniform float64:", rng.random(5))
    print("uniform float32:", rng.random(5, dtype=np.float32))
    print("integers:       ", rng.integers(0, 100, size=5))
    print("normal:         ", rng.normal(0, 1, size=5))


if __name__ == "__main__":
    main()
