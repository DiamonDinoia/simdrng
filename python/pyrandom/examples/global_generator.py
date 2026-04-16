"""Global generator — drop-in replacement with zero code changes.

Use pyrandom.seed() once, then call pyrandom.random() / pyrandom.normal()
etc. as module-level functions.  This is the easiest way to speed up an
existing NumPy codebase.
"""

import pyrandom


def main() -> None:
    # Seed the global XoshiroSIMD generator
    pyrandom.seed(42)

    # Module-level functions use the global generator
    print("random:", pyrandom.random(5))
    print("normal:", pyrandom.normal(size=5))
    print("integers:", pyrandom.integers(0, 100, size=5))
    print("uniform:", pyrandom.uniform(low=-1, high=1, size=5))

    # Reproducible: same seed → same sequence
    pyrandom.seed(42)
    a = pyrandom.random(10)
    pyrandom.seed(42)
    b = pyrandom.random(10)
    import numpy as np
    np.testing.assert_array_equal(a, b)
    print("reproducible: OK")

    # Get the underlying Generator for advanced use
    rng = pyrandom.default_rng()
    print("choice:", rng.choice([1, 2, 3, 4, 5], size=3))


if __name__ == "__main__":
    main()
