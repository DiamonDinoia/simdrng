"""Use a simdrng generator as a drop-in source for scipy.stats distributions."""

from __future__ import annotations

import numpy as np
import simdrng
from scipy import stats


def main() -> None:
    rng = simdrng.Philox4x64(seed=2026)

    # scipy.stats accepts any numpy.random.Generator via random_state=.
    samples = stats.norm.rvs(loc=0.0, scale=1.0, size=100_000, random_state=rng)
    print(f"normal samples: mean={samples.mean():.3f} std={samples.std():.3f}")

    expo = stats.expon.rvs(scale=2.0, size=100_000, random_state=rng)
    print(f"expon samples:  mean={expo.mean():.3f} (expected 2.0)")


if __name__ == "__main__":
    main()
