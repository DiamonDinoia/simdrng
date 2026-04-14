#include <random>
#include <vector>

#include <random/xoshiro_simd.hpp>
#include <catch2/catch_all.hpp>

static constexpr auto tests = 1 << 12; // 4096
constexpr auto SIMD_WIDTH = xsimd::simd_type<prng::XoshiroNative::result_type>::size;

TEST_CASE("SEED", "[xoshiro256++]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  prng::XoshiroScalar reference(seed);
  prng::XoshiroNative rng(seed);
  REQUIRE(rng.getState(0) == reference.getState());
  for (auto i = 1UL; i < SIMD_WIDTH; ++i) {
    reference.jump();
    REQUIRE(rng.getState(i) == reference.getState());
  }
}

TEST_CASE("JUMP", "[xoshiro256++]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  prng::XoshiroScalar reference(seed);
  prng::XoshiroNative rng(seed);
  for (auto i = 0UL; i < SIMD_WIDTH; ++i) {
    REQUIRE(rng.getState(i) == reference.getState());
    reference.jump();
  }
  for (auto i = 0UL; i < SIMD_WIDTH; ++i) {
    rng.jump();
  }
  for (auto i = 0U; i < SIMD_WIDTH; ++i) {
    INFO("i: " << i);
    REQUIRE(rng.getState(i) == reference.getState());
    reference.jump();
  }
}

TEST_CASE("LONG JUMP", "[xoshiro256++]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  prng::XoshiroScalar reference(seed);
  prng::XoshiroNative rng(seed);
  rng.long_jump();
  reference.long_jump();
  for (auto i = 0UL; i < SIMD_WIDTH; ++i) {
    INFO("i: " << i);
    REQUIRE(rng.getState(i) == reference.getState());
    reference.jump();
  }
}

TEST_CASE("GENERATE UINT64", "[xoshiro256++]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  prng::XoshiroNative rng(seed);
  std::vector<prng::XoshiroScalar> reference;
  reference.reserve(SIMD_WIDTH);
  for (auto i = 0UL; i < SIMD_WIDTH; ++i) {
    reference.emplace_back(seed);
  }
  for (auto i = 1U; i < SIMD_WIDTH; ++i) {
    for (auto j = 0UL; j < i; ++j) {
      reference[i].jump();
    }
  }
  for (auto i = 0; i < SIMD_WIDTH; ++i) {
    REQUIRE(rng.getState(i) == reference[i].getState());
  }
  for (auto i = 0; i < tests; i += SIMD_WIDTH) {
    for (auto j = 0; j < SIMD_WIDTH; ++j) {
      INFO("i: " << i << " j: " << j);
      REQUIRE(rng() == reference[j]());
    }
  }
}

TEST_CASE("GENERATE DOUBLE", "[xoshiro256++]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  prng::XoshiroNative rng(seed);
  std::vector<prng::XoshiroScalar> reference;
  reference.reserve(SIMD_WIDTH);
  for (auto i = 0UL; i < SIMD_WIDTH; ++i) {
    reference.emplace_back(seed);
  }
  for (auto i = 1U; i < SIMD_WIDTH; ++i) {
    for (auto j = 0UL; j < i; ++j) {
      reference[i].jump();
    }
  }
  for (auto i = 0; i < SIMD_WIDTH; ++i) {
    REQUIRE(rng.getState(i) == reference[i].getState());
  }
  for (auto i = 0; i < tests; i += SIMD_WIDTH) {
    for (auto j = 0; j < SIMD_WIDTH; ++j) {
      INFO("i: " << i << " j: " << j);
      REQUIRE(rng.uniform() == reference[j].uniform());
    }
  }
}
