"""Quick timing comparison: simdrng vs numpy.random.default_rng."""

from __future__ import annotations

import time

import numpy as np
import simdrng

N = 10_000_000


def time_it(label: str, fn) -> None:
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    rate = out.size / dt / 1e6
    print(f"{label:<28s} {dt*1e3:7.2f} ms  {rate:8.1f} Msamples/s")


def main() -> None:
    np_rng = np.random.default_rng(42)
    x_rng = simdrng.XoshiroSIMD(42)
    p_rng = simdrng.Philox4x64(42)

    time_it("numpy.default_rng",       lambda: np_rng.random(N))
    time_it("simdrng.XoshiroSIMD",    lambda: x_rng.random(N))
    time_it("simdrng.Philox4x64",     lambda: p_rng.random(N))


if __name__ == "__main__":
    main()
