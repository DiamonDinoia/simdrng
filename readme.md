# simdrng

[![CI](https://github.com/DiamonDinoia/simdrng/actions/workflows/ci.yml/badge.svg)](https://github.com/DiamonDinoia/simdrng/actions/workflows/ci.yml)
[![Docs](https://readthedocs.org/projects/simdrng/badge/?version=latest)](https://simdrng.readthedocs.io/en/latest/)
[![CodSpeed](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json&project=DiamonDinoia/simdrng)](https://codspeed.io/DiamonDinoia/simdrng)
[![codecov](https://codecov.io/gh/DiamonDinoia/simdrng/branch/main/graph/badge.svg)](https://codecov.io/gh/DiamonDinoia/simdrng)
[![License: BSD-3-Clause-Attribution](https://img.shields.io/badge/License-BSD--3--Clause--Attribution-blue.svg)](LICENSE)

> **Note:** `simdrng` is pre-1.0 — the API may change before the first tagged release.

C++20 header-only (plus a tiny dispatch TU) library of scalar and SIMD-accelerated
random number generators, with nanobind-powered Python bindings that drop directly
into NumPy and SciPy.

## Generators

| Family      | Scalar | SIMD dispatch | ``-march=native`` |
|-------------|:------:|:-------------:|:-----------------:|
| Xoshiro256++| ✅     | ✅            | ✅                |
| SplitMix64  | ✅     | —             | —                 |
| ChaCha 8/12/20 | ✅  | ✅            | ✅                |
| Philox 2x/4x × 32/64 | ✅ | ✅       | ✅                |

All generators are compatible with ``std::uniform_int_distribution`` and friends.
A ``uniform()`` method returning a `double` in ``[0, 1)`` is provided for a faster
path than ``std::uniform_real_distribution``; see
<https://prng.di.unimi.it/#remarks> for the rationale.

## Multi-threading and cluster environments

Each generator takes optional ``thread_id`` and ``cluster_id`` parameters that
produce independent streams per thread and per node:

```cpp
#include <simdrng/xoshiro.hpp>

simdrng::Xoshiro rng(42, 1, 2);                     // seed, thread, cluster
simdrng::Xoshiro rng(42, std::this_thread::get_id());
simdrng::Xoshiro rng(42, omp_get_thread_num());     // OpenMP
simdrng::Xoshiro rng(42, omp_get_thread_num(), MPI_rank);  // MPI + OpenMP
```

See [`examples/cpp/threaded_openmp.cpp`](examples/cpp/threaded_openmp.cpp).

## Build

```sh
git clone https://github.com/DiamonDinoia/simdrng.git
cd simdrng
cmake --preset release
cmake --build build/release
ctest --preset test
```

CMake options:

| Option                   | Default | Purpose                                              |
|--------------------------|---------|------------------------------------------------------|
| `SIMDRNG_WITH_XSIMD`     | ON      | Build the SIMD backends with xsimd (OFF = scalar-only header library) |
| `SIMDRNG_BUILD_TESTS`    | ON\*    | Build Catch2 tests and benchmarks (\*follows `BUILD_TESTING`) |
| `SIMDRNG_BUILD_PYTHON`   | OFF     | Build the nanobind Python extension (requires `SIMDRNG_WITH_XSIMD`) |
| `SIMDRNG_BUILD_EXAMPLES` | OFF     | Build C++ examples under `examples/cpp`              |
| `SIMDRNG_BUILD_DOCS`     | OFF     | Generate Sphinx/Doxygen docs                         |
| `SIMDRNG_MARCH_NATIVE`   | OFF     | Compile benchmarks with `-march=native`              |
| `SIMDRNG_ENABLE_CODSPEED`| OFF     | Link codspeed-cpp into the bench harness             |
| `SIMDRNG_USE_SANITIZERS` | OFF     | `ON` = ASan+UBSan, `TSAN` = ThreadSanitizer          |

`CMakePresets.json` ships ready-made profiles so you rarely need raw `-D` flags:
`release`, `debug`, `ci`, `bench`, `codspeed`, `sanitizers`, `tsan`, `valgrind`,
`static-analysis`, `coverage`. For example `cmake --preset sanitizers && cmake
--build --preset sanitizers && ctest --preset sanitizers`.

With `SIMDRNG_WITH_XSIMD=OFF` the library is header-only and depends on nothing —
only the scalar generators are built and `simdrng::Xoshiro` aliases the scalar
implementation.

Consuming from another CMake project — install and `find_package`:

```cmake
find_package(simdrng CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE simdrng::simdrng)
```

The installed package config pulls in its dependencies (`xsimd`, `poet`) via
`find_dependency` automatically — but only when built with xsimd; a scalar-only
install requires no dependencies.

Or vendor it directly with FetchContent:

```cmake
include(FetchContent)
FetchContent_Declare(simdrng
    GIT_REPOSITORY https://github.com/DiamonDinoia/simdrng.git
    GIT_TAG        main)
FetchContent_MakeAvailable(simdrng)
target_link_libraries(my_target PRIVATE simdrng::simdrng)
```

### Single header

Two amalgamated headers are published to the
[`single-header`](https://github.com/DiamonDinoia/simdrng/tree/single-header)
branch:

- **`simdrng-scalar.hpp`** — fully self-contained, scalar generators only. No
  xsimd, no poet, no include path needed.
- **`simdrng.hpp`** — SIMD-capable. poet is inlined; xsimd is kept as an external
  `<xsimd/...>` include, so compile with the xsimd headers available. The
  compile-time-arch `*Native` generators (`XoshiroNative`, `Philox4x64Native`, …)
  work header-only; the runtime-dispatch types (`XoshiroSIMD`, `Philox*SIMD`, and
  the default `simdrng::Xoshiro` alias) require linking the compiled library and
  are not available from the single header alone.

Ready-to-run [Compiler Explorer links](https://github.com/DiamonDinoia/simdrng/tree/godbolt-links)
(the SIMD one uses Compiler Explorer's vendored xsimd library) are published to
the `godbolt-links` branch.

## Examples

C++ — see [`examples/cpp/`](examples/cpp/):

- [`basic_xoshiro.cpp`](examples/cpp/basic_xoshiro.cpp) — scalar usage
- [`simd_bulk_fill.cpp`](examples/cpp/simd_bulk_fill.cpp) — SIMD hot loop
- [`xoshiro_native.cpp`](examples/cpp/xoshiro_native.cpp) — `-march=native`
- [`threaded_openmp.cpp`](examples/cpp/threaded_openmp.cpp) — per-thread streams
- [`philox_counter_based.cpp`](examples/cpp/philox_counter_based.cpp) — Philox counter-based reproducibility

Python — see [`examples/python/`](examples/python/):

```python
import simdrng
import numpy as np

rng = simdrng.XoshiroSIMD(seed=42)          # returns numpy.random.Generator
samples = rng.random(10_000_000)            # GIL-released C++ bulk fill
```

## Python install

```sh
pip install .
# or, for development:
pip install -e '.[test]' --no-build-isolation
```

## Benchmarks

The full matrix runs in CI and is rendered to SVG on the
[`benchmark-results`](https://github.com/DiamonDinoia/simdrng/tree/benchmark-results)
branch; CodSpeed tracks regressions per-PR. See
[docs → benchmarks](https://simdrng.readthedocs.io/en/latest/benchmarks.html).

![Throughput overview (gcc-15)](https://raw.githubusercontent.com/DiamonDinoia/simdrng/benchmark-results/charts/gcc-15/overview.svg)

## Contributing

Formatting and linting run through [pre-commit](https://pre-commit.com/)
(clang-format, gersemi, darker/ruff, shfmt, markdownlint). Install once:

```sh
pip install pre-commit && pre-commit install
git config blame.ignoreRevsFile .git-blame-ignore-revs   # skip the bulk-format commit in blame
```

## References

- D. Blackman and S. Vigna, *Scrambled Linear Pseudorandom Number Generators*,
  ACM TOMS (2021) — <https://vigna.di.unimi.it/papers.php#BlVSLPNG>.
- S. Vigna, the xoshiro/xoroshiro family and PRNG shootout — <https://prng.di.unimi.it/>.
  xoshiro256++ has period 2²⁵⁶−1; SplitMix64 is used to seed it. `uniform()`'s
  `(x >> 11) * 0x1.0p-53` follows the [equidistribution rationale](https://prng.di.unimi.it/#remarks)
  (*"all dyadic rationals of the form k·2⁻⁵³ will be equally likely"*).
- The reference `splitmix64.c` / `xoshiro256plusplus.c` used by the tests are
  the authors' own public-domain (CC0) code from <https://prng.di.unimi.it/>.

More detail in [docs → references](https://simdrng.readthedocs.io/en/latest/references.html).

## License

BSD-3-Clause-Attribution — see [`LICENSE`](LICENSE).
