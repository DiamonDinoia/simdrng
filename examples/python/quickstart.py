"""Quickstart: bulk-fill a numpy array and demonstrate reproducibility."""

from __future__ import annotations

import numpy as np
import simdrng


def main() -> None:
    rng = simdrng.XoshiroSIMD(42)
    assert isinstance(rng, np.random.Generator)

    samples = rng.random(1_000_000)  # dispatches to C++ bulk fill
    print(f"Drew {samples.size} float64 samples; mean={samples.mean():.4f}")

    # Reproducibility: re-seeding yields identical output.
    a = simdrng.XoshiroSIMD(123).random(8)
    b = simdrng.XoshiroSIMD(123).random(8)
    assert np.array_equal(a, b), "identical seeds must yield identical streams"
    print("Reproducibility check: OK")

    # float32 fast path
    f32 = rng.random(1_000_000, dtype=np.float32)
    print(f"float32 samples dtype={f32.dtype} size={f32.size}")


if __name__ == "__main__":
    main()
