Benchmarks
==========

The benchmark harness uses Google Benchmark, optionally augmented with
hardware counters via libpfm on Linux. Each push to ``main`` runs the full
matrix across several compilers; JSON results and generated SVG charts are
committed to the orphan ``benchmark-results`` branch.

The latest charts (gcc-15):

.. image:: https://raw.githubusercontent.com/DiamonDinoia/simdrng/benchmark-results/charts/gcc-15/latency.svg
   :alt: Single-value latency (gcc-15)
   :align: center

.. image:: https://raw.githubusercontent.com/DiamonDinoia/simdrng/benchmark-results/charts/gcc-15/throughput.svg
   :alt: Buffer-fill throughput, scalar vs SIMD (gcc-15)
   :align: center

.. image:: https://raw.githubusercontent.com/DiamonDinoia/simdrng/benchmark-results/charts/gcc-15/scalar_vs_simd.svg
   :alt: Scalar vs SIMD speedup (gcc-15)
   :align: center

CodSpeed tracks per-benchmark CPU-cycle regressions on every PR:

.. image:: https://img.shields.io/endpoint?url=https://codspeed.io/badge.json&project=DiamonDinoia/simdrng
   :alt: CodSpeed badge

Performance
-----------

The charts above measure the two regimes that matter when picking a generator:
*single-value latency* and *bulk throughput*. They tell different stories, so
read them together.

Generating ``u64``
~~~~~~~~~~~~~~~~~~~

``u64`` is the native output of every engine (``result_type`` is
``std::uint64_t``): ``operator()`` returns one value, ``generate(buf, n)`` fills
a buffer. Everything else — doubles, distributions — is built on top of it, so
this is the number to look at first.

Every SIMD engine generates a **256-value block** at a time into an internal
buffer and hands it out one value per ``operator()`` call; the block is refilled
only when it drains. The benchmarks run millions of iterations, so the
single-value latency you see already includes that refill, amortised across the
256 calls between refills (well under 0.5 ns/call) — it is *not* dominated by
the fill cost.

What the per-call number *is* dominated by is the buffered read itself: load the
cursor, load ``buffer[cursor]``, write the cursor back. That memory round-trip
costs a few nanoseconds, and for an engine as cheap as Xoshiro it is more
expensive than just computing the next value in registers — which is why
**scalar Xoshiro single-value (~1 ns) beats SIMD/Native (~3 ns)**. (Native shows
the same ~3 ns as the dispatched SIMD path, confirming the cost is the buffered
read, not runtime dispatch.) For engines whose per-value compute is expensive,
the buffered SIMD block wins even one-at-a-time: ChaCha20 is ~3.5x and Philox4x32
~1.8x faster in SIMD than scalar.

Bulk ``generate()`` is a different story. The SIMD lanes run flat out, writing
straight to the caller's buffer with no per-element cursor, so the read overhead
vanishes: Xoshiro reaches ~2.3x its scalar loop (sub-0.5 ns/value). **Use the
bulk API whenever you need many values at once** (filling a tensor, a batch of
draws) — it is the fastest path by a wide margin. Reserve single-value calls for
when results are consumed one at a time (rejection sampling, branchy
Monte-Carlo inner loops), and prefer the scalar engine there for cheap
generators.

Generating doubles: ``uniform()`` vs ``std::uniform_real_distribution``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two ways to turn ``u64`` into a uniform ``double`` in ``[0, 1)``:

* The built-in ``uniform()`` takes the top 53 bits and scales them:
  ``(x >> 11) * 0x1.0p-53``. One shift and one multiply on a value the engine
  already produced — essentially free on top of the ``u64`` cost.
* ``std::uniform_real_distribution<double>`` is the portable standard-library
  route. It is correct and engine-agnostic, but does considerably more work per
  draw (range handling, potentially multiple generator pulls).

The gap is large: scalar Xoshiro produces a double in ~1.6 ns via ``uniform()``
versus ~5.2 ns through ``std::uniform_real_distribution`` — roughly **3x**. Both
yield a uniform ``[0, 1)``; reach for the standard distribution only when you
need its exact semantics or a non-unit range. For buffers, ``fill_uniform(buf,
n)`` applies the shift-and-multiply over a bulk ``generate()``.

32-bit integer / ``float`` output paths are not currently exposed; a narrower
element type would roughly double per-second element counts and halve the buffer
footprint, since cost is dominated by bytes generated.

Memory requirements
~~~~~~~~~~~~~~~~~~~~~

Each generator instance is self-contained and heap-free: a small core state (the
Xoshiro lanes, or the Philox counter+key — tens to ~100 bytes) plus the fixed
**256-element output buffer** the single-value path draws from (2 KiB for u64).
State and buffer are over-aligned to the SIMD register width so bulk stores stay
aligned. There is no shared or global state, so instances are trivially copyable
and safe to keep one-per-thread.

Running benchmarks locally
--------------------------

.. code-block:: sh

   cmake --preset release
   cmake --build build/release --target benchmarks
   ./build/release/tests/benchmarks \
       --benchmark_perf_counters=CYCLES,INSTRUCTIONS,CACHE-MISSES,BRANCH-MISSES,BRANCHES \
       --benchmark_format=json --benchmark_out=bench.json

   python scripts/generate_charts.py bench.json
