"""State serialization and pickle — checkpoint/resume any generator."""

import pickle

import numpy as np
import simdrng


def main() -> None:
    rng = simdrng.XoshiroSIMD(42)
    rng.random(1000)  # advance state

    # Save state
    state = rng.bit_generator.state
    expected = rng.random(10)

    # Restore state → same sequence
    rng.bit_generator.state = state
    actual = rng.random(10)
    np.testing.assert_array_equal(actual, expected)
    print("state roundtrip: OK")

    # Pickle roundtrip
    rng.bit_generator.state = state
    data = pickle.dumps(rng.bit_generator)
    bg_restored = pickle.loads(data)
    rng2 = np.random.Generator(bg_restored)
    np.testing.assert_array_equal(rng2.random(10), expected)
    print("pickle roundtrip: OK")

    # Works for all generators
    for name, factory in [
        ("SplitMix", lambda: simdrng.SplitMix(42)),
        ("ChaCha20", lambda: simdrng.ChaCha20(42)),
        ("XoshiroNative", lambda: simdrng.XoshiroNative(42)),
    ]:
        rng = factory()
        rng.random(100)
        s = rng.bit_generator.state
        a = rng.random(10)
        rng.bit_generator.state = s
        b = rng.random(10)
        np.testing.assert_array_equal(a, b)
        print(f"{name}: OK")


if __name__ == "__main__":
    main()
