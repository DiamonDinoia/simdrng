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

#if defined(__GNUC__) || defined(__clang__)
#define SIMDRNG_EXPORT __attribute__((visibility("default")))
#elif defined(_MSC_VER) && defined(SIMDRNG_BUILDING_SHARED)
#define SIMDRNG_EXPORT __declspec(dllexport)
#elif defined(_MSC_VER) && defined(SIMDRNG_USING_SHARED)
#define SIMDRNG_EXPORT __declspec(dllimport)
#else
#define SIMDRNG_EXPORT
#endif

// Architecture detection
#if defined(__x86_64__) || defined(_M_X64)
#define SIMDRNG_ARCH_X86_64 1
#elif defined(__aarch64__) || defined(_M_ARM64)
#define SIMDRNG_ARCH_AARCH64 1
#elif defined(__riscv) && (__riscv_xlen == 64)
#define SIMDRNG_ARCH_RISCV64 1
#endif
