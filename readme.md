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
#include <random/xoshiro.hpp>

prng::Xoshiro rng(42, 1, 2);                     // seed, thread, cluster
prng::Xoshiro rng(42, std::this_thread::get_id());
prng::Xoshiro rng(42, omp_get_thread_num());     // OpenMP
prng::Xoshiro rng(42, omp_get_thread_num(), MPI_rank);  // MPI + OpenMP
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

| Option              | Default | Purpose                                |
|---------------------|---------|----------------------------------------|
| `ENABLE_TESTS`      | ON      | Build Catch2 tests and benchmarks      |
| `ENABLE_PYTHON`     | ON      | Build nanobind Python extension        |
| `BUILD_EXAMPLES`    | OFF     | Build C++ examples under `examples/cpp`|
| `BUILD_DOCS`        | OFF     | Generate Sphinx/Doxygen docs           |
| `MARCH_NATIVE`      | OFF     | Compile benchmarks with `-march=native`|
| `ENABLE_CODSPEED`   | OFF     | Link codspeed-cpp into the bench harness|

Consuming from another CMake project:

```cmake
include(FetchContent)
FetchContent_Declare(simdrng
    GIT_REPOSITORY https://github.com/DiamonDinoia/simdrng.git
    GIT_TAG        main)
FetchContent_MakeAvailable(simdrng)
target_link_libraries(my_target PRIVATE simdrng::simdrng)
```

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

## License

BSD-3-Clause-Attribution — see [`LICENSE`](LICENSE).
