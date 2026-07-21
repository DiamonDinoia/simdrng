#pragma once

#if defined(_MSC_VER)
#define SIMDRNG_ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define SIMDRNG_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define SIMDRNG_ALWAYS_INLINE inline
#endif

// Lambda call-operators can't take the `inline` keyword, so force-inlining a
// lambda body (e.g. a poet::static_for callable) needs the bare attribute. GCC
// otherwise outlines and clones heavy generic-lambda bodies per instantiation.
#if defined(_MSC_VER)
#define SIMDRNG_ALWAYS_INLINE_LAMBDA [[msvc::forceinline]]
#elif defined(__GNUC__) || defined(__clang__)
#define SIMDRNG_ALWAYS_INLINE_LAMBDA __attribute__((always_inline))
#else
#define SIMDRNG_ALWAYS_INLINE_LAMBDA
#endif

#if defined(__GNUC__) || defined(__clang__)
#define SIMDRNG_FLATTEN __attribute__((flatten))
#else
#define SIMDRNG_FLATTEN
#endif

#if defined(_MSC_VER)
#define SIMDRNG_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define SIMDRNG_RESTRICT __restrict__
#else
#define SIMDRNG_RESTRICT
#endif

#if defined(_MSC_VER)
#define SIMDRNG_NEVER_INLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define SIMDRNG_NEVER_INLINE __attribute__((cold, noinline))
#else
#define SIMDRNG_NEVER_INLINE
#endif

// simdrng ships as a STATIC library (and header-only public API). The per-arch
// dispatch-functor instantiations and the out-of-line XoshiroSIMD ctor need
// external linkage so the explicit instantiations resolve at static-link time,
// but they are implementation details and must NOT be re-exported from a shared
// object that links the static lib (e.g. the Python extension, which should
// export only its module init). Mark them hidden; MSVC static libs don't export
// without dllexport, so nothing is needed there.
#if defined(__GNUC__) || defined(__clang__)
#define SIMDRNG_LOCAL __attribute__((visibility("hidden")))
#else
#define SIMDRNG_LOCAL
#endif

// Runtime SIMD dispatch. When 1, the *SIMD types dispatch over the compiled
// per-arch tiers (portable, set by the CMake SIMDRNG_DISPATCH option, which
// propagates this via a PUBLIC compile definition). Default 0: no tiers are
// built and the default aliases resolve to the native (xsimd::best_arch) types.
#ifndef SIMDRNG_DISPATCH
#define SIMDRNG_DISPATCH 0
#endif

// Architecture detection
#if defined(__x86_64__) || defined(_M_X64)
#define SIMDRNG_ARCH_X86_64 1
#elif defined(__aarch64__) || defined(_M_ARM64)
#define SIMDRNG_ARCH_AARCH64 1
#elif defined(__riscv) && (__riscv_xlen == 64)
#define SIMDRNG_ARCH_RISCV64 1
#endif
