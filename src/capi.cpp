// simdrng C ABI implementation.
//
// One virtual IRng interface dispatches to a templated RngImpl<Rng> per family.
// The switch on kind lives only in simdrng_create; every other entry point is a
// single indirect call, so bulk fills pay it once per array and run the
// generator's own SIMD path. The bulk-fill bodies mirror the nanobind binding
// (python/simdrng.cpp) so the C ABI and Python produce identical streams.

#include "simdrng/capi.h"

#include "simdrng/chacha_simd.hpp"
#include "simdrng/philox_simd.hpp"
#include "simdrng/splitmix.hpp"
#include "simdrng/version.hpp"
#include "simdrng/xoshiro_simd.hpp"

#include <cstddef>
#include <cstdint>
#include <new>
#include <string>
#include <type_traits>
#include <utility>

namespace {

constexpr double kInvPow53 = 0x1.0p-53;         // 2^-53, for [0,1) doubles
constexpr float kInvPow24 = 1.0f / 16777216.0f; // 2^-24, for [0,1) floats

// Generators exposing generate(uint64*, n) — every SIMD family. Scalar
// generators (SplitMix) fall back to the per-sample loop.
template <typename T, typename = void> struct has_bulk_fill : std::false_type {};
template <typename T>
struct has_bulk_fill<
    T, std::void_t<decltype(std::declval<T &>().generate(std::declval<std::uint64_t *>(), std::size_t{}))>>
    : std::true_type {};

struct IRng {
  virtual ~IRng() = default;
  virtual std::uint64_t next_u64() noexcept = 0;
  virtual void fill_u64(std::uint64_t *out, std::size_t n) noexcept = 0;
  virtual void fill_u32(std::uint32_t *out, std::size_t n) noexcept = 0;
  virtual void fill_double(double *out, std::size_t n) noexcept = 0;
  virtual void fill_float(float *out, std::size_t n) noexcept = 0;
};

template <typename Rng> struct RngImpl final : IRng {
  Rng rng;
  template <typename... Args> explicit RngImpl(Args &&...args) : rng(std::forward<Args>(args)...) {}

  std::uint64_t next_u64() noexcept override { return rng(); }

  void fill_u64(std::uint64_t *out, std::size_t n) noexcept override {
    if constexpr (has_bulk_fill<Rng>::value) {
      rng.generate(out, n);
    } else {
      for (std::size_t i = 0; i < n; ++i)
        out[i] = rng();
    }
  }

  void fill_double(double *out, std::size_t n) noexcept override {
    if constexpr (has_bulk_fill<Rng>::value) {
      rng.fill_uniform(out, n);
    } else {
      for (std::size_t i = 0; i < n; ++i)
        out[i] = static_cast<double>(rng() >> 11) * kInvPow53;
    }
  }

  // 32-bit outputs pack 2 samples per uint64 (high word first), matching the
  // Python binding. Bulk generators fill a raw uint64 buffer in chunks; scalar
  // ones split inline.
  void fill_u32(std::uint32_t *out, std::size_t n) noexcept override {
    if constexpr (has_bulk_fill<Rng>::value) {
      constexpr std::size_t kChunk = 256;
      alignas(64) std::uint64_t buf[kChunk];
      std::size_t pos = 0;
      while (pos < n) {
        const std::size_t need = (n - pos + 1) / 2;
        const std::size_t take = need < kChunk ? need : kChunk;
        rng.generate(buf, take);
        for (std::size_t j = 0; j < take; ++j) {
          out[pos++] = static_cast<std::uint32_t>(buf[j] >> 32);
          if (pos < n)
            out[pos++] = static_cast<std::uint32_t>(buf[j]);
        }
      }
    } else {
      std::size_t i = 0;
      for (; i + 1 < n; i += 2) {
        const std::uint64_t raw = rng();
        out[i] = static_cast<std::uint32_t>(raw >> 32);
        out[i + 1] = static_cast<std::uint32_t>(raw);
      }
      if (i < n)
        out[i] = static_cast<std::uint32_t>(rng() >> 32);
    }
  }

  void fill_float(float *out, std::size_t n) noexcept override {
    if constexpr (has_bulk_fill<Rng>::value) {
      constexpr std::size_t kChunk = 256;
      alignas(64) std::uint64_t buf[kChunk];
      std::size_t pos = 0;
      while (pos < n) {
        const std::size_t need = (n - pos + 1) / 2;
        const std::size_t take = need < kChunk ? need : kChunk;
        rng.generate(buf, take);
        for (std::size_t j = 0; j < take; ++j) {
          out[pos++] = static_cast<float>(static_cast<std::uint32_t>(buf[j] >> 32) >> 8) * kInvPow24;
          if (pos < n)
            out[pos++] = static_cast<float>(static_cast<std::uint32_t>(buf[j]) >> 8) * kInvPow24;
        }
      }
    } else {
      std::size_t i = 0;
      for (; i + 1 < n; i += 2) {
        const std::uint64_t raw = rng();
        out[i] = static_cast<float>(static_cast<std::uint32_t>(raw >> 32) >> 8) * kInvPow24;
        out[i + 1] = static_cast<float>(static_cast<std::uint32_t>(raw) >> 8) * kInvPow24;
      }
      if (i < n)
        out[i] = static_cast<float>(static_cast<std::uint32_t>(rng() >> 32) >> 8) * kInvPow24;
    }
  }
};

thread_local std::string g_last_error;

void set_error(const char *msg) noexcept {
  try {
    g_last_error = msg;
  } catch (...) { // never let an allocation failure escape the C ABI
  }
}

IRng *make(simdrng_kind kind, std::uint64_t seed) {
  using namespace simdrng;
  switch (kind) {
  case SIMDRNG_SPLITMIX:
    return new RngImpl<SplitMix>(seed);
  case SIMDRNG_XOSHIRO:
    return new RngImpl<XoshiroSIMD>(seed);
  case SIMDRNG_CHACHA8:
    return new RngImpl<ChaChaSIMD<8>>(seed);
  case SIMDRNG_CHACHA12:
    return new RngImpl<ChaChaSIMD<12>>(seed);
  case SIMDRNG_CHACHA20:
    return new RngImpl<ChaChaSIMD<20>>(seed);
  case SIMDRNG_PHILOX4X32:
    return new RngImpl<PhiloxSIMD<4, 32, 10>>(seed);
  case SIMDRNG_PHILOX2X32:
    return new RngImpl<PhiloxSIMD<2, 32, 10>>(seed);
  case SIMDRNG_PHILOX4X64:
    return new RngImpl<PhiloxSIMD<4, 64, 10>>(seed);
  case SIMDRNG_PHILOX2X64:
    return new RngImpl<PhiloxSIMD<2, 64, 10>>(seed);
  default:
    return nullptr;
  }
}

} // namespace

struct simdrng_state {
  IRng *impl;
  simdrng_kind kind;
};

extern "C" {

simdrng_t simdrng_create(simdrng_kind kind, uint64_t seed) {
  IRng *impl = nullptr;
  try {
    impl = make(kind, seed);
  } catch (const std::exception &e) {
    set_error(e.what());
    return nullptr;
  } catch (...) {
    set_error("simdrng_create: unknown exception");
    return nullptr;
  }
  if (!impl) {
    set_error("simdrng_create: unknown generator kind");
    return nullptr;
  }
  auto *g = new (std::nothrow) simdrng_state{impl, kind};
  if (!g) {
    delete impl;
    set_error("simdrng_create: out of memory");
    return nullptr;
  }
  return g;
}

simdrng_t simdrng_free(simdrng_t g) {
  if (g) {
    delete g->impl;
    delete g;
  }
  return nullptr;
}

simdrng_kind simdrng_get_kind(simdrng_t g) { return g->kind; }

uint64_t simdrng_next_u64(simdrng_t g) { return g->impl->next_u64(); }
double simdrng_next_double(simdrng_t g) { return static_cast<double>(g->impl->next_u64() >> 11) * kInvPow53; }

void simdrng_fill_u64(simdrng_t g, uint64_t *out, size_t n) { g->impl->fill_u64(out, n); }
void simdrng_fill_u32(simdrng_t g, uint32_t *out, size_t n) { g->impl->fill_u32(out, n); }
void simdrng_fill_double(simdrng_t g, double *out, size_t n) { g->impl->fill_double(out, n); }
void simdrng_fill_float(simdrng_t g, float *out, size_t n) { g->impl->fill_float(out, n); }

const char *simdrng_version(void) { return SIMDRNG_VERSION_STRING; }

const char *simdrng_last_error(void) { return g_last_error.c_str(); }

} // extern "C"
