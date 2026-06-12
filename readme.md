# simdrng

[![CI](https://github.com/DiamonDinoia/simdrng/actions/workflows/ci.yml/badge.svg)](https://github.com/DiamonDinoia/simdrng/actions/workflows/ci.yml)
[![Docs](https://readthedocs.org/projects/simdrng/badge/?version=latest)](https://simdrng.readthedocs.io/en/latest/)
[![CodSpeed](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json&project=DiamonDinoia/simdrng)](https://codspeed.io/DiamonDinoia/simdrng)
[![codecov](https://codecov.io/gh/DiamonDinoia/simdrng/branch/main/graph/badge.svg)](https://codecov.io/gh/DiamonDinoia/simdrng)
[![License: BSD-3-Clause-Attribution](https://img.shields.io/badge/License-BSD--3--Clause--Attribution-blue.svg)](LICENSE)

> **Note:** `simdrng` is pre-1.0 — the API may change before the first tagged release.

C++20 random number generators that run scalar **and** SIMD-accelerated, with
nanobind-powered Python bindings that drop straight into NumPy and SciPy.

- **Header-only** (plus one tiny dispatch TU for runtime SIMD selection); the
  scalar path is dependency-free.
- **Four generator families** — Xoshiro256++, SplitMix64, ChaCha, Philox — each
  available scalar, runtime-SIMD-dispatched, and `-march=native`.
- **Drop-in** with `std::uniform_int_distribution` and friends, plus a fast
  `uniform()` returning a `double` in `[0, 1)`.
- **Parallel-ready** — per-thread and per-node independent streams via
  `thread_id` / `cluster_id`, or Philox's counter-based reproducibility.
- **Python** — every generator is a `numpy.random.Generator` with a GIL-released
  C++ bulk-fill fast path.

## Why simdrng

General-purpose PRNGs are written one number at a time, leaving most of a modern
CPU's vector units idle. simdrng pairs best-in-class scalar generators
(Blackman & Vigna's xoshiro256++, the Random123 Philox counter-based design,
the ChaCha cipher core) with hand-tuned SIMD backends, and picks the right
implementation **at runtime** via [xsimd](https://github.com/xtensor-stack/xsimd)
dispatch — so the same binary uses AVX-512 on a server and NEON on an ARM
laptop with no recompilation. When you can commit to a target with
`-march=native`, the `*Native` generators inline the widest available width
directly. The result is NumPy-beating bulk throughput from C++ or Python while
keeping single-value calls as fast as the reference scalar code.

## Quick start

```cpp
#include <simdrng/xoshiro.hpp>

simdrng::Xoshiro rng(42);          // seed
std::uint64_t x = rng();           // next 64-bit value
double u = rng.uniform();          // double in [0, 1)
```

Bulk fill is where SIMD pays off. `XoshiroSIMD` refills an internal cache with
wide SIMD batches, so a plain generate loop runs vectorised under the hood — no
special bulk API to learn:

```cpp
#include <simdrng/xoshiro_simd.hpp>
#include <vector>

simdrng::XoshiroSIMD rng(42);
std::vector<double> out(1'000'000);
for (auto &x : out)
  x = rng.uniform();              // cache refilled with SIMD batches
```

From Python it is a `numpy.random.Generator`:

```python
import simdrng

rng = simdrng.XoshiroSIMD(seed=42)          # a numpy.random.Generator
samples = rng.random(10_000_000)            # GIL-released C++ bulk fill
```

More runnable programs live in [`examples/`](examples/) (C++ and Python), and
ready-to-run [Compiler Explorer links](https://github.com/DiamonDinoia/simdrng/tree/godbolt-links)
are published to the `godbolt-links` branch.

## Generators

| Family               | Scalar | SIMD dispatch | `-march=native` |
|----------------------|:------:|:-------------:|:---------------:|
| Xoshiro256++         | ✅     | ✅            | ✅              |
| SplitMix64           | ✅     | —             | —               |
| ChaCha 8/12/20       | ✅     | ✅            | ✅              |
| Philox 2x/4x × 32/64 | ✅     | ✅            | ✅              |

All generators satisfy the standard `UniformRandomBitGenerator` requirements, so
they compose with `std::uniform_int_distribution` and friends. The `uniform()`
method returns a `double` in `[0, 1)` faster than
`std::uniform_real_distribution`; see <https://prng.di.unimi.it/#remarks> for
the rationale. Per-family guidance is in the
[documentation](https://simdrng.readthedocs.io/en/latest/) (Guides section).

## Parallel streams

Each generator takes optional `thread_id` and `cluster_id` parameters that carve
out independent, non-overlapping streams per thread and per node (via xoshiro's
`jump()` / `long_jump()`):

```cpp
#include <simdrng/xoshiro.hpp>

simdrng::Xoshiro rng(42, 1, 2);                            // seed, thread, cluster
simdrng::Xoshiro rng(42, omp_get_thread_num());            // OpenMP
simdrng::Xoshiro rng(42, omp_get_thread_num(), MPI_rank);  // MPI + OpenMP
```

Philox takes the counter-based route instead: each work item derives its own
sub-stream from `(seed, counter)` with no coordination — see
[`examples/cpp/philox_counter_based.cpp`](examples/cpp/philox_counter_based.cpp).
For the threaded example, see
[`examples/cpp/threaded_openmp.cpp`](examples/cpp/threaded_openmp.cpp).

## Install

```sh
git clone https://github.com/DiamonDinoia/simdrng.git
cd simdrng
cmake --preset release
cmake --build build/release
ctest --preset test
```

`CMakePresets.json` ships ready-made profiles so you rarely need raw `-D` flags:
`release`, `debug`, `ci`, `bench`, `codspeed`, `sanitizers`, `tsan`, `valgrind`,
`static-analysis`, `coverage`.

CMake options:

| Option                    | Default | Purpose                                                               |
|---------------------------|---------|-----------------------------------------------------------------------|
| `SIMDRNG_WITH_XSIMD`      | ON      | Build the SIMD backends with xsimd (OFF = scalar-only header library) |
| `SIMDRNG_BUILD_TESTS`     | ON\*    | Build Catch2 tests and benchmarks (\*follows `BUILD_TESTING`)         |
| `SIMDRNG_BUILD_PYTHON`    | OFF     | Build the nanobind Python extension (requires `SIMDRNG_WITH_XSIMD`)   |
| `SIMDRNG_BUILD_EXAMPLES`  | OFF     | Build C++ examples under `examples/cpp`                               |
| `SIMDRNG_BUILD_DOCS`      | OFF     | Generate Sphinx/Doxygen docs                                          |
| `SIMDRNG_MARCH_NATIVE`    | OFF     | Compile benchmarks with `-march=native`                               |
| `SIMDRNG_ENABLE_CODSPEED` | OFF     | Link codspeed-cpp into the bench harness                              |
| `SIMDRNG_USE_SANITIZERS`  | OFF     | `ON` = ASan+UBSan, `TSAN` = ThreadSanitizer                           |

With `SIMDRNG_WITH_XSIMD=OFF` the library is header-only and depends on nothing —
only the scalar generators are built and `simdrng::Xoshiro` aliases the scalar
implementation.

**Consume from another CMake project** — install and `find_package`:

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

### Python

```sh
pip install .
# or, for development:
pip install -e '.[test]' --no-build-isolation
```

## Benchmarks

The full matrix runs in CI and is rendered to SVG on the
[`benchmark-results`](https://github.com/DiamonDinoia/simdrng/tree/benchmark-results)
branch; CodSpeed tracks regressions per-PR.

The chart below shows `u64` bulk throughput — the native output of every
engine — comparing the scalar loop against SIMD `generate()`. Per-call latency,
double generation (`uniform()` vs `std::uniform_real_distribution`), and memory
cost are covered in
[docs → benchmarks](https://simdrng.readthedocs.io/en/latest/benchmarks.html).

![Buffer-fill throughput, scalar vs SIMD (gcc-15)](https://raw.githubusercontent.com/DiamonDinoia/simdrng/benchmark-results/charts/gcc-15/throughput.svg)

## Documentation

Full documentation — quick start, per-family guides, the auto-generated C++ and
Python API reference, and the technical references — is hosted at
**<https://simdrng.readthedocs.io/>**.

## Contributing

Formatting and linting run through [pre-commit](https://pre-commit.com/)
(clang-format, gersemi, darker/ruff, shfmt, markdownlint). Install once:

```sh
pip install pre-commit && pre-commit install
git config blame.ignoreRevsFile .git-blame-ignore-revs   # skip the bulk-format commit in blame
```

## References

Grouped by the family each source backs — full annotations (periods, scrambling,
seeding, the `uniform()` rationale) live in
[docs → references](https://simdrng.readthedocs.io/en/latest/references.html):

- **Xoshiro256++ & SplitMix64** — D. Blackman and S. Vigna, *Scrambled Linear
  Pseudorandom Number Generators*, ACM TOMS (2021),
  <https://vigna.di.unimi.it/papers.php#BlVSLPNG>; and S. Vigna's
  xoshiro/xoroshiro shootout, SplitMix64 seeding, and `uniform()` rationale,
  <https://prng.di.unimi.it/>.
- **ChaCha 8/12/20** — D. J. Bernstein, *ChaCha, a variant of Salsa20*,
  <https://cr.yp.to/chacha.html>; cross-checked against
  [Monocypher](https://monocypher.org/) in the tests.
- **Philox** — J. K. Salmon et al., *Parallel Random Numbers: As Easy as
  1, 2, 3* (Philox), SC '11.

The reference `splitmix64.c` / `xoshiro256plusplus.c` used by the tests are the
authors' own public-domain (CC0) code from <https://prng.di.unimi.it/>.

## License

BSD-3-Clause-Attribution — see [`LICENSE`](LICENSE).
