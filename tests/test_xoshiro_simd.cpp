#include <random>
#include <vector>

#include <catch2/catch_all.hpp>
#include <simdrng/xoshiro_simd.hpp>

static constexpr auto tests = 1 << 12; // 4096

TEST_CASE("SIMD DISPATCH JUMP", "[xoshiro256++]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  simdrng::XoshiroSIMD simd(seed);
  simdrng::XoshiroSIMD simd2(seed);
  // Both should produce the same output
  for (int i = 0; i < 1024; ++i) {
    REQUIRE(simd() == simd2());
  }
  // After jump, they should diverge
  simd.jump();
  REQUIRE(simd() != simd2());
}

TEST_CASE("SIMD DISPATCH MID JUMP", "[xoshiro256++]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  simdrng::XoshiroSIMD simd(seed);
  simdrng::XoshiroSIMD simd2(seed);
  simd.mid_jump();
  REQUIRE(simd() != simd2());
}

#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE
constexpr auto SIMD_WIDTH = xsimd::simd_type<simdrng::XoshiroNative::result_type>::size;

TEST_CASE("SEED", "[xoshiro256++]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  simdrng::XoshiroScalar reference(seed);
  simdrng::XoshiroNative rng(seed);
  REQUIRE(rng.getState(0) == reference.getState());
  for (auto i = 1UL; i < SIMD_WIDTH; ++i) {
    reference.jump();
    REQUIRE(rng.getState(i) == reference.getState());
  }
}

TEST_CASE("JUMP", "[xoshiro256++]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  simdrng::XoshiroScalar reference(seed);
  simdrng::XoshiroNative rng(seed);
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
  simdrng::XoshiroScalar reference(seed);
  simdrng::XoshiroNative rng(seed);
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
  simdrng::XoshiroNative rng(seed);
  std::vector<simdrng::XoshiroScalar> reference;
  reference.reserve(SIMD_WIDTH);
  for (auto i = 0UL; i < SIMD_WIDTH; ++i) {
    reference.emplace_back(seed);
  }
  for (auto i = 1U; i < SIMD_WIDTH; ++i) {
    for (auto j = 0UL; j < i; ++j) {
      reference[i].jump();
    }
  }
  for (auto i = 0UL; i < SIMD_WIDTH; ++i) {
    REQUIRE(rng.getState(i) == reference[i].getState());
  }
  for (auto i = 0; i < tests; i += SIMD_WIDTH) {
    for (auto j = 0UL; j < SIMD_WIDTH; ++j) {
      INFO("i: " << i << " j: " << j);
      REQUIRE(rng() == reference[j]());
    }
  }
}

TEST_CASE("MID JUMP", "[xoshiro256++]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  simdrng::XoshiroScalar reference(seed);
  simdrng::XoshiroNative rng(seed);
  rng.mid_jump();
  reference.mid_jump();
  for (auto i = 0UL; i < SIMD_WIDTH; ++i) {
    INFO("i: " << i);
    REQUIRE(rng.getState(i) == reference.getState());
    reference.jump();
  }
}

TEST_CASE("JUMP_N PER LANE", "[xoshiro256++][jump_n]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  // Lane i is seeded as the scalar generator after i jumps; jump(k) then
  // advances every lane by k. So lane i must match scalar(seed) + i jumps + jump(k).
  for (const std::uint64_t k :
       {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{5}, std::uint64_t{1000}, std::uint64_t{1} << 20}) {
    INFO("k: " << k);
    simdrng::XoshiroNative rng(seed);
    std::vector<simdrng::XoshiroScalar> reference;
    reference.reserve(SIMD_WIDTH);
    for (auto i = 0UL; i < SIMD_WIDTH; ++i) {
      reference.emplace_back(seed);
      for (auto j = 0UL; j < i; ++j)
        reference[i].jump();
    }
    rng.jump(k);
    for (auto &ref : reference)
      ref.jump(k);
    for (auto i = 0U; i < SIMD_WIDTH; ++i) {
      INFO("i: " << i);
      REQUIRE(rng.getState(i) == reference[i].getState());
    }
  }
}

TEST_CASE("SIMD DISPATCH JUMP_N", "[xoshiro256++][jump_n]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  const std::uint64_t k = (std::uint64_t{1} << 30) + 12345;
  simdrng::XoshiroSIMD dispatched(seed);
  simdrng::XoshiroNative native(seed);
  dispatched.jump(k);
  native.jump(k);
  for (auto i = 0U; i < SIMD_WIDTH; ++i) {
    INFO("i: " << i);
    REQUIRE(dispatched() == native());
  }
}

TEST_CASE("JUMP_N POW2 PER LANE", "[xoshiro256++][jump_n]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  // jump(pow2{128}) per lane must match the fixed jump() on the scalar reference.
  simdrng::XoshiroNative rng(seed);
  std::vector<simdrng::XoshiroScalar> reference;
  reference.reserve(SIMD_WIDTH);
  for (auto i = 0UL; i < SIMD_WIDTH; ++i) {
    reference.emplace_back(seed);
    for (auto j = 0UL; j < i; ++j)
      reference[i].jump();
  }
  rng.jump(simdrng::pow2{128});
  for (auto &ref : reference)
    ref.jump(); // jump() == 2^128 == pow2{128}
  for (auto i = 0U; i < SIMD_WIDTH; ++i) {
    INFO("i: " << i);
    REQUIRE(rng.getState(i) == reference[i].getState());
  }
}

TEST_CASE("SIMD DISPATCH JUMP_N POW2", "[xoshiro256++][jump_n]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  simdrng::XoshiroSIMD dispatched(seed);
  simdrng::XoshiroNative native(seed);
  dispatched.jump(simdrng::pow2{200});
  native.jump(simdrng::pow2{200});
  for (auto i = 0U; i < SIMD_WIDTH; ++i) {
    INFO("i: " << i);
    REQUIRE(dispatched() == native());
  }
}

TEST_CASE("GENERATE DOUBLE", "[xoshiro256++]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  simdrng::XoshiroNative rng(seed);
  std::vector<simdrng::XoshiroScalar> reference;
  reference.reserve(SIMD_WIDTH);
  for (auto i = 0UL; i < SIMD_WIDTH; ++i) {
    reference.emplace_back(seed);
  }
  for (auto i = 1U; i < SIMD_WIDTH; ++i) {
    for (auto j = 0UL; j < i; ++j) {
      reference[i].jump();
    }
  }
  for (auto i = 0UL; i < SIMD_WIDTH; ++i) {
    REQUIRE(rng.getState(i) == reference[i].getState());
  }
  for (auto i = 0; i < tests; i += SIMD_WIDTH) {
    for (auto j = 0UL; j < SIMD_WIDTH; ++j) {
      INFO("i: " << i << " j: " << j);
      REQUIRE(rng.uniform() == reference[j].uniform());
    }
  }
}
#endif // XSIMD_NO_SUPPORTED_ARCHITECTURE
