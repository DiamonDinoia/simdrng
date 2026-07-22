// Exercises the C ABI (include/simdrng/capi.h) through the static library, so
// CI covers the surface every non-Python binding links against.

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <string>
#include <vector>

#include "simdrng/capi.h"

namespace {
constexpr simdrng_kind kAllKinds[] = {
    SIMDRNG_SPLITMIX,   SIMDRNG_XOSHIRO,    SIMDRNG_CHACHA8,    SIMDRNG_CHACHA12,   SIMDRNG_CHACHA20,
    SIMDRNG_PHILOX4X32, SIMDRNG_PHILOX2X32, SIMDRNG_PHILOX4X64, SIMDRNG_PHILOX2X64,
};
} // namespace

TEST_CASE("capi version is reported", "[capi]") { REQUIRE(std::string(simdrng_version()).size() > 0); }

TEST_CASE("capi create/free round-trips every kind", "[capi]") {
  static_assert(sizeof(kAllKinds) / sizeof(kAllKinds[0]) == SIMDRNG_KIND_COUNT);
  for (auto kind : kAllKinds) {
    simdrng_t g = simdrng_create(kind, 123);
    REQUIRE(g != nullptr);
    REQUIRE(simdrng_get_kind(g) == kind);
    (void)simdrng_next_u64(g);
    REQUIRE(simdrng_free(g) == nullptr);
  }
}

TEST_CASE("capi is deterministic for a fixed seed", "[capi]") {
  for (auto kind : kAllKinds) {
    simdrng_t a = simdrng_create(kind, 42);
    simdrng_t b = simdrng_create(kind, 42);
    for (int i = 0; i < 64; ++i)
      REQUIRE(simdrng_next_u64(a) == simdrng_next_u64(b));
    simdrng_free(a);
    simdrng_free(b);
  }
}

TEST_CASE("capi bulk u64 fill equals sequential draws", "[capi]") {
  constexpr std::size_t n = 4096;
  for (auto kind : kAllKinds) {
    simdrng_t bulk = simdrng_create(kind, 7);
    simdrng_t seq = simdrng_create(kind, 7);
    std::vector<std::uint64_t> buf(n);
    simdrng_fill_u64(bulk, buf.data(), n);
    for (std::size_t i = 0; i < n; ++i)
      REQUIRE(buf[i] == simdrng_next_u64(seq));
    simdrng_free(bulk);
    simdrng_free(seq);
  }
}

TEST_CASE("capi uniform fills stay in range", "[capi]") {
  constexpr std::size_t n = 4096;
  for (auto kind : kAllKinds) {
    simdrng_t g = simdrng_create(kind, 99);
    std::vector<double> d(n);
    std::vector<float> f(n);
    simdrng_fill_double(g, d.data(), n);
    simdrng_fill_float(g, f.data(), n);
    for (double x : d)
      REQUIRE((x >= 0.0 && x < 1.0));
    for (float x : f)
      REQUIRE((x >= 0.0F && x < 1.0F));
    simdrng_free(g);
  }
}

TEST_CASE("capi rejects an unknown kind", "[capi]") {
  REQUIRE(simdrng_create(SIMDRNG_KIND_COUNT, 0) == nullptr);
  REQUIRE(std::string(simdrng_last_error()).size() > 0);
}
