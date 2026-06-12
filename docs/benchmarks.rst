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

The two charts above measure the two regimes that matter when picking a
generator. They tell different stories, so read them together.

Latency
   *Single-value* cost of one ``operator()`` (u64) or ``uniform()`` (double).
   Every engine buffers a block of SIMD-generated results and hands them out
   one at a time, so the common case is a load from that buffer plus an index
   bump — a few nanoseconds. The buffer is refilled in bulk only once it
   drains, amortising the SIMD work across hundreds of calls. This is the
   number that matters when results are consumed one at a time (rejection
   sampling, branchy Monte-Carlo inner loops).

Throughput
   *Buffer-fill* rate of ``generate()`` (u64) and ``fill_uniform()`` (double),
   in Mop/s. Here the SIMD lanes run flat out, writing straight to the caller's
   buffer with no per-element dispatch, so this is where vectorisation pays
   off — e.g. Xoshiro's ``generate()`` is ~2.3x its scalar loop. Use the bulk
   API whenever you need many values at once (filling a tensor, a batch of
   draws); it is the fastest path by a wide margin.

Output types
   The engines produce **64-bit** integers (``result_type`` is
   ``std::uint64_t``) and **double-precision** uniforms in ``[0, 1)`` built
   from the top 53 bits (``(x >> 11) * 2^-53``). Philox's 4x32 variant computes
   in 32-bit words internally but still hands back packed u64 results; 32-bit
   integer / ``float`` output paths are not currently exposed. A narrower
   element type would roughly double per-second element counts and halve the
   buffer footprint, since cost is dominated by bytes generated.

Memory requirements
   Each generator instance is self-contained and heap-free: a small core state
   (the Xoshiro lanes, or the Philox counter+key — tens to ~100 bytes) plus a
   fixed **256-element output buffer** that the single-value path draws from
   (2 KiB for u64). State and buffer are over-aligned to the SIMD register
   width so bulk stores stay aligned. There is no shared or global state, so
   instances are trivially copyable and safe to keep one-per-thread.

Running benchmarks locally
--------------------------

.. code-block:: sh

   cmake --preset release
   cmake --build build/release --target benchmarks
   ./build/release/tests/benchmarks \
       --benchmark_perf_counters=CYCLES,INSTRUCTIONS,CACHE-MISSES,BRANCH-MISSES,BRANCHES \
       --benchmark_format=json --benchmark_out=bench.json

   python scripts/generate_charts.py bench.json
