"""Xoshiro (scalar) generator with jump for parallel streams."""

import numpy as np
import simdrng


def main() -> None:
    rng1 = simdrng.Xoshiro(42)
    rng2 = simdrng.Xoshiro(42)
    rng2.bit_generator.jump()  # advance to independent stream

    a = rng1.random(5)
    b = rng2.random(5)
    print("stream 1:", a)
    print("stream 2:", b)
    print("independent:", not np.array_equal(a, b))


if __name__ == "__main__":
    main()
