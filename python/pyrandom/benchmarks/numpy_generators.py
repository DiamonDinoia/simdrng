"""Benchmark suite for pyrandom generators vs NumPy builtins.

Six benchmark categories:
1. Throughput comparison — all generators, 100M float64 samples
2. Per-sample overhead — single-sample latency
3. Bulk fill vs Generator path — quantifies fast-path benefit
4. State serialization cost — get/set state round-trip
5. Scaling — throughput vs array size
6. Integer generation — uint64 throughput
"""

from __future__ import annotations

import pickle
import time
from typing import Callable

import numpy as np

import pyrandom


def _bench(
    fn: Callable[[], None], *, repeat: int = 10, warmup: int = 1,
) -> float:
    """Return best time in seconds over *repeat* runs after *warmup*."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def _header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ── 1. Throughput comparison ────────────────────────────────────────────────


def bench_throughput(size: int = 100_000_000, repeat: int = 10) -> None:
    _header(f"1. Throughput (float64, N={size:_})")
    out = np.empty(size, dtype=np.float64)
    generators: dict[str, Callable[[], np.random.Generator]] = {
        "XoshiroSIMD": lambda: pyrandom.XoshiroSIMD(1234),
        "XoshiroNative": lambda: pyrandom.XoshiroNative(1234),
        "Xoshiro": lambda: pyrandom.Xoshiro(1234),
        "SplitMix": lambda: pyrandom.SplitMix(1234),
        "ChaCha20": lambda: pyrandom.ChaCha20(1234),
        "ChaCha8": lambda: pyrandom.ChaCha8(1234),
        "PCG64": lambda: np.random.Generator(np.random.PCG64(1234)),
        "Philox": lambda: np.random.Generator(np.random.Philox(1234)),
        "MT19937": lambda: np.random.Generator(np.random.MT19937(1234)),
        "default_rng": lambda: np.random.default_rng(1234),
    }
    for name, factory in generators.items():
        rng = factory()

        def run(r=rng, o=out):
            r.random(size, out=o)

        best = _bench(run, repeat=repeat)
        rate = size / best
        print(f"  {name:20s}: {best:8.3f} s  ({rate / 1e6:8.2f} M samples/s)")


# ── 2. Per-sample overhead ──────────────────────────────────────────────────


def bench_per_sample(repeat: int = 100_000) -> None:
    _header("2. Per-sample overhead (single float64)")
    generators: dict[str, Callable[[], np.random.Generator]] = {
        "XoshiroSIMD": lambda: pyrandom.XoshiroSIMD(1234),
        "XoshiroNative": lambda: pyrandom.XoshiroNative(1234),
        "ChaCha20": lambda: pyrandom.ChaCha20(1234),
        "PCG64": lambda: np.random.Generator(np.random.PCG64(1234)),
        "default_rng": lambda: np.random.default_rng(1234),
    }
    for name, factory in generators.items():
        rng = factory()

        def run(r=rng, n=repeat):
            for _ in range(n):
                r.random()

        best = _bench(run, repeat=3)
        ns_per = best / repeat * 1e9
        print(f"  {name:20s}: {ns_per:8.1f} ns/sample")


# ── 3. Bulk fill vs Generator path ─────────────────────────────────────────


def bench_bulk_vs_generator(size: int = 10_000_000, repeat: int = 10) -> None:
    _header(f"3. pyrandom.random() vs NumPy.random() (N={size:_})")
    out = np.empty(size, dtype=np.float64)

    # pyrandom (transparent C++ bulk fill via Generator subclass)
    for name, rng in [
        ("XoshiroSIMD", pyrandom.XoshiroSIMD(1234)),
        ("XoshiroNative", pyrandom.XoshiroNative(1234)),
    ]:
        def run(r=rng, o=out):
            r.random(size, out=o)

        t = _bench(run, repeat=repeat)
        print(f"  {name:20s}: {t:.3f}s  ({size / t / 1e6:8.2f} M/s)")

    # NumPy (standard per-sample callback path)
    for name, rng in [
        ("PCG64 (numpy)", np.random.Generator(np.random.PCG64(1234))),
        ("default_rng", np.random.default_rng(1234)),
    ]:
        def run(r=rng, o=out):
            r.random(size, out=o)

        t = _bench(run, repeat=repeat)
        print(f"  {name:20s}: {t:.3f}s  ({size / t / 1e6:8.2f} M/s)")


# ── 4. State serialization cost ────────────────────────────────────────────


def bench_serialization(repeat: int = 10_000) -> None:
    _header("4. State serialization cost")
    generators = {
        "XoshiroSIMD": pyrandom.XoshiroSIMD(1234),
        "ChaCha20": pyrandom.ChaCha20(1234),
        "SplitMix": pyrandom.SplitMix(1234),
    }
    for name, rng in generators.items():
        bg = rng.bit_generator

        # get/set round-trip
        def run_state(b=bg, n=repeat):
            for _ in range(n):
                s = b.state
                b.state = s

        t_state = _bench(run_state, repeat=3)

        # pickle round-trip
        def run_pickle(b=bg, n=repeat):
            for _ in range(n):
                pickle.loads(pickle.dumps(b))

        t_pickle = _bench(run_pickle, repeat=3)
        print(
            f"  {name:20s}: state={t_state / repeat * 1e6:.1f} us  "
            f"pickle={t_pickle / repeat * 1e6:.1f} us",
        )


# ── 5. Scaling — throughput vs array size ───────────────────────────────────


def bench_scaling() -> None:
    _header("5. Scaling (XoshiroSIMD throughput vs array size)")
    sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]
    for size in sizes:
        out = np.empty(size, dtype=np.float64)
        rng = pyrandom.XoshiroSIMD(1234)

        def run(r=rng, o=out, s=size):
            r.random(s, out=o)

        best = _bench(run, repeat=max(3, 100_000_000 // size))
        rate = size / best
        print(f"  N={size:>12_}: {rate / 1e6:8.2f} M samples/s")


# ── 6. Integer generation ──────────────────────────────────────────────────


def bench_integers(size: int = 10_000_000, repeat: int = 10) -> None:
    _header(f"6. Integer generation (uint64, N={size:_})")
    out = np.empty(size, dtype=np.uint64)
    generators = {
        "XoshiroSIMD bulk": pyrandom.XoshiroSIMD(1234),
        "XoshiroNative bulk": pyrandom.XoshiroNative(1234),
        "ChaCha20 bulk": pyrandom.ChaCha20(1234),
    }
    for name, rng in generators.items():
        def run(r=rng, o=out):
            r.bit_generator.fill_uint64(o)

        best = _bench(run, repeat=repeat)
        rate = size / best
        print(f"  {name:25s}: {best:.3f}s  ({rate / 1e6:8.2f} M samples/s)")

    # NumPy path comparison
    numpy_gens = {
        "PCG64 integers": np.random.Generator(np.random.PCG64(1234)),
        "default_rng integers": np.random.default_rng(1234),
    }
    for name, rng in numpy_gens.items():
        def run(r=rng, s=size):
            r.integers(0, 2**63, size=s, dtype=np.int64)

        best = _bench(run, repeat=repeat)
        rate = size / best
        print(f"  {name:25s}: {best:.3f}s  ({rate / 1e6:8.2f} M samples/s)")


# ── 7. Float32 throughput ───────────────────────────────────────────────────


def bench_float32(size: int = 100_000_000, repeat: int = 10) -> None:
    _header(f"7. Float32 throughput (N={size:_}, 2 samples per uint64)")
    out = np.empty(size, dtype=np.float32)
    generators: dict[str, Callable[[], np.random.Generator]] = {
        "XoshiroSIMD": lambda: pyrandom.XoshiroSIMD(1234),
        "XoshiroNative": lambda: pyrandom.XoshiroNative(1234),
        "ChaCha8": lambda: pyrandom.ChaCha8(1234),
        "PCG64": lambda: np.random.Generator(np.random.PCG64(1234)),
        "default_rng": lambda: np.random.default_rng(1234),
    }
    for name, factory in generators.items():
        rng = factory()

        def run(r=rng, o=out):
            r.random(size, dtype=np.float32, out=o)

        best = _bench(run, repeat=repeat)
        rate = size / best
        print(f"  {name:20s}: {best:8.3f} s  ({rate / 1e6:8.2f} M samples/s)")


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    print("pyrandom benchmark suite")
    print(f"NumPy version: {np.__version__}")
    bench_throughput()
    bench_per_sample()
    bench_bulk_vs_generator()
    bench_serialization()
    bench_scaling()
    bench_integers()
    bench_float32()


if __name__ == "__main__":
    main()
