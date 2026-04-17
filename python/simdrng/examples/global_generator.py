"""Global generator — drop-in replacement with zero code changes.

Use simdrng.seed() once, then call simdrng.random() / simdrng.normal()
etc. as module-level functions.  This is the easiest way to speed up an
existing NumPy codebase.
"""

import simdrng


def main() -> None:
    # Seed the global XoshiroSIMD generator
    simdrng.seed(42)

    # Module-level functions use the global generator
    print("random:", simdrng.random(5))
    print("normal:", simdrng.normal(size=5))
    print("integers:", simdrng.integers(0, 100, size=5))
    print("uniform:", simdrng.uniform(low=-1, high=1, size=5))

    # Reproducible: same seed → same sequence
    simdrng.seed(42)
    a = simdrng.random(10)
    simdrng.seed(42)
    b = simdrng.random(10)
    import numpy as np
    np.testing.assert_array_equal(a, b)
    print("reproducible: OK")

    # Get the underlying Generator for advanced use
    rng = simdrng.default_rng()
    print("choice:", rng.choice([1, 2, 3, 4, 5], size=3))


if __name__ == "__main__":
    main()
