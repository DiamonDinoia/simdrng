Benchmarks
==========

The benchmark harness uses Google Benchmark, optionally augmented with
hardware counters via libpfm on Linux. Each push to ``main`` runs the full
matrix across several compilers; JSON results and generated SVG charts are
committed to the orphan ``benchmark-results`` branch.

The latest charts:

.. image:: https://raw.githubusercontent.com/DiamonDinoia/simdrng/benchmark-results/charts/gcc-15/overview.svg
   :alt: Throughput overview (gcc-15)
   :align: center

.. image:: https://raw.githubusercontent.com/DiamonDinoia/simdrng/benchmark-results/charts/gcc-15/scalar_vs_simd.svg
   :alt: Scalar vs SIMD speedup (gcc-15)
   :align: center

CodSpeed tracks per-benchmark CPU-cycle regressions on every PR:

.. image:: https://img.shields.io/endpoint?url=https://codspeed.io/badge.json&project=DiamonDinoia/simdrng
   :alt: CodSpeed badge

Running benchmarks locally
--------------------------

.. code-block:: sh

   cmake --preset release
   cmake --build build/release --target benchmarks
   ./build/release/tests/benchmarks \
       --benchmark_perf_counters=CYCLES,INSTRUCTIONS,CACHE-MISSES,BRANCH-MISSES,BRANCHES \
       --benchmark_format=json --benchmark_out=bench.json

   python scripts/generate_charts.py bench.json
