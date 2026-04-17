"""ChaCha generators — cryptographic-quality PRNG."""

import numpy as np
import simdrng


def main() -> None:
    # Seed-based (simple — key expansion happens in C++)
    rng = simdrng.ChaCha20(42)
    print("ChaCha20:", rng.random(5))

    # Key-based (explicit 256-bit key)
    key = [0x03020100, 0x07060504, 0x0B0A0908, 0x0F0E0D0C,
           0x13121110, 0x17161514, 0x1B1A1918, 0x1F1E1D1C]
    rng = simdrng.ChaCha20(key, counter=0, nonce=0)
    print("ChaCha20 (key):", rng.random(5))

    # Fewer rounds for speed
    rng8 = simdrng.ChaCha8(42)
    print("ChaCha8:", rng8.random(5))

    # All ChaCha variants work with SciPy
    from scipy import stats
    print("ChaCha12 normal:", stats.norm.rvs(size=3, random_state=simdrng.ChaCha12(42)))


if __name__ == "__main__":
    main()
