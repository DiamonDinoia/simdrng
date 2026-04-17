"""SplitMix generator with NumPy."""

import numpy as np
import simdrng


def main() -> None:
    rng = simdrng.SplitMix(123)

    print("uniform:", rng.random((3, 4)))
    print("integers:", rng.integers(0, 10, size=(2, 5)))
    print("normal:", rng.normal(0.0, 1.0, size=5))

    # State serialization
    state = rng.bit_generator.state
    a = rng.random(10)
    rng.bit_generator.state = state
    b = rng.random(10)
    np.testing.assert_array_equal(a, b)
    print("state roundtrip: OK")


if __name__ == "__main__":
    main()
