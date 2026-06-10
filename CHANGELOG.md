# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Pre-1.0, the public API may change between minor versions.

## [Unreleased]

## [0.0.1]

First tagged release. The project was renamed from `VectorXoshiro`/`prng` to
`simdrng` and hardened for distribution.

### Added
- Optional xsimd backend via `-DSIMDRNG_WITH_XSIMD` (default ON). With it `OFF`,
  the library is dependency-free and header-only, exposing only the scalar
  generators; `simdrng::Xoshiro` then aliases the scalar implementation.
- `VERSION` file plus a generated `include/simdrng/version.hpp`
  (`SIMDRNG_VERSION_*` / `SIMDRNG_VERSION_STRING`), produced by
  `cmake/GenerateVersion.cmake` with a git-aware `-dev.N+g<sha>` suffix.
- Installable CMake package: `find_package(simdrng CONFIG)` with an exported
  `simdrng::simdrng` target. The package config emits `find_dependency(xsimd)` /
  `find_dependency(poet)` only when built with xsimd.
- Single-header amalgamation (`tools/amalgamate.py`) publishing two variants to
  the `single-header` branch: a fully self-contained `simdrng-scalar.hpp` and a
  SIMD-capable `simdrng.hpp` (poet inlined, xsimd kept external).
- Working Compiler Explorer links for the single-header demos, published to the
  `godbolt-links` branch (the SIMD demo uses Compiler Explorer's vendored xsimd).
- `.clang-tidy`, a `cmake/dev_helpers.cmake` split (CPM bootstrap, warnings,
  sanitizers, static analysis, coverage), and a CI analysis job.
- Compiler Explorer link generation workflow.
- `Install & Consume` CI workflow that installs the library (xsimd and
  scalar-only matrices) and builds a downstream `find_package` consumer against it.

### Changed
- **Renamed** the public surface: include directory `random/` → `simdrng/`,
  namespace `prng::` → `simdrng::`, and macros `PRNG_*` → `SIMDRNG_*`. Downstream
  code now uses `#include <simdrng/...>` and `simdrng::`.
- Switched xsimd from a force-pushed personal fork to upstream
  `xtensor-stack/xsimd` (pinned by SHA); `mulhilo` is now the upstream
  `xsimd::mul_hilo`.
- Pinned `poet` to its `v0.0.0` release.
- AVX-512 runtime dispatch now targets `avx512bw` (x86-64-v4: F+CD+BW+DQ+VL)
  instead of `avx512f`, unlocking `vpmullq` (AVX512DQ) for Philox `mul_hilo`.
- Normalized all CMake options to the `SIMDRNG_*` prefix; `SIMDRNG_BUILD_PYTHON`
  now defaults to `OFF` (scikit-build enables it explicitly for wheels).
- Rewrote the Philox round (scalar and SIMD) as a single data-driven loop over
  `N/2` counter pairs (multipliers/Weyl increments/output permutation in
  `PhiloxConstants`) instead of `if constexpr (N == 4)` branches; adding a new N
  is now just a constants specialization. The SIMD round uses `poet::static_for`.

[Unreleased]: https://github.com/DiamonDinoia/simdrng/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/DiamonDinoia/simdrng/releases/tag/v0.0.1
