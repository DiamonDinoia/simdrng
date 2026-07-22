/* simdrng C ABI.
 *
 * A stable, extern-"C" surface over the SIMD generators so non-C++ languages
 * (Fortran, Julia, JS, MATLAB/Octave, ...) can drive simdrng through one small
 * shared/static library. All heavy lifting stays in the C++ headers; this is a
 * thin opaque-handle wrapper.
 *
 * Throughput note: bulk fills (simdrng_fill_*) amortize the one indirect call
 * per array over the whole buffer and run the generator's own SIMD path.
 * Per-draw calls (simdrng_next_*) pay one indirect call each — prefer the bulk
 * API in hot loops.
 */
#ifndef SIMDRNG_CAPI_H
#define SIMDRNG_CAPI_H

#include <stddef.h>
#include <stdint.h>

/* Export/visibility. Consumers get nothing (import on Windows); the library TUs
 * are compiled with SIMDRNG_C_BUILD; a fully static link defines SIMDRNG_STATIC. */
#if defined(SIMDRNG_STATIC)
#define SIMDRNG_CAPI_EXPORT
#elif defined(_WIN32)
#if defined(SIMDRNG_C_BUILD)
#define SIMDRNG_CAPI_EXPORT __declspec(dllexport)
#else
#define SIMDRNG_CAPI_EXPORT __declspec(dllimport)
#endif
#elif defined(SIMDRNG_C_BUILD) && (defined(__GNUC__) || defined(__clang__))
#define SIMDRNG_CAPI_EXPORT __attribute__((visibility("default")))
#else
#define SIMDRNG_CAPI_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Generator family. All are constructed from a single 64-bit seed; the SIMD
 * variants select the best ISA at runtime. */
typedef enum simdrng_kind {
  SIMDRNG_SPLITMIX = 0, /* scalar SplitMix64 (no bulk SIMD) */
  SIMDRNG_XOSHIRO,      /* xoshiro256++, runtime-dispatched SIMD */
  SIMDRNG_CHACHA8,
  SIMDRNG_CHACHA12,
  SIMDRNG_CHACHA20,
  SIMDRNG_PHILOX4X32,
  SIMDRNG_PHILOX2X32,
  SIMDRNG_PHILOX4X64,
  SIMDRNG_PHILOX2X64,
  SIMDRNG_KIND_COUNT
} simdrng_kind;

/* Opaque generator handle. */
typedef struct simdrng_state *simdrng_t;

/* Create a generator of the given kind seeded with `seed`.
 * Returns NULL on failure (unknown kind or allocation failure); the reason is
 * available via simdrng_last_error(). */
SIMDRNG_CAPI_EXPORT simdrng_t simdrng_create(simdrng_kind kind, uint64_t seed);

/* Destroy a handle. Passing NULL is a no-op. Always returns NULL, so callers
 * can write `h = simdrng_free(h);`. */
SIMDRNG_CAPI_EXPORT simdrng_t simdrng_free(simdrng_t g);

/* The kind a handle was created with. */
SIMDRNG_CAPI_EXPORT simdrng_kind simdrng_get_kind(simdrng_t g);

/* Single draws. */
SIMDRNG_CAPI_EXPORT uint64_t simdrng_next_u64(simdrng_t g);
SIMDRNG_CAPI_EXPORT double simdrng_next_double(simdrng_t g); /* uniform [0,1) */

/* Bulk fills. `out` must hold at least `n` elements. Uniform variants produce
 * values in [0,1). These release no locks and allocate nothing. */
SIMDRNG_CAPI_EXPORT void simdrng_fill_u64(simdrng_t g, uint64_t *out, size_t n);
SIMDRNG_CAPI_EXPORT void simdrng_fill_u32(simdrng_t g, uint32_t *out, size_t n);
SIMDRNG_CAPI_EXPORT void simdrng_fill_double(simdrng_t g, double *out, size_t n);
SIMDRNG_CAPI_EXPORT void simdrng_fill_float(simdrng_t g, float *out, size_t n);

/* Linked-library version, e.g. "0.0.2-dev.146+g...". Never NULL. */
SIMDRNG_CAPI_EXPORT const char *simdrng_version(void);

/* Thread-local message describing the most recent failure on this thread, or
 * "" if none. The returned pointer is valid until the next failing call on the
 * same thread. */
SIMDRNG_CAPI_EXPORT const char *simdrng_last_error(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SIMDRNG_CAPI_H */
