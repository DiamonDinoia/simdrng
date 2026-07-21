#include <catch2/catch_all.hpp>
#include <cstdint>
#include <random>
#include <simdrng/xoshiro_scalar.hpp>

#include "xoshiro256plusplus.c"

static constexpr auto tests = 1 << 15;

TEST_CASE("xoshiro256++", "[xoshiro256++]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  simdrng::XoshiroScalar rng(seed);
  s[0] = rng.getState()[0];
  s[1] = rng.getState()[1];
  s[2] = rng.getState()[2];
  s[3] = rng.getState()[3];
  for (int i = 0; i < tests; ++i) {
    REQUIRE(rng() == next());
  }
  rng.jump();
  jump();
  REQUIRE(rng.getState()[0] == s[0]);
  REQUIRE(rng.getState()[1] == s[1]);
  REQUIRE(rng.getState()[2] == s[2]);
  REQUIRE(rng.getState()[3] == s[3]);
  for (int i = 0; i < tests; ++i) {
    const auto result = rng.uniform();
    REQUIRE(result >= 0);
    REQUIRE(result < 1);
  }
}

TEST_CASE("xoshiro256++ jump(n) forward oracle", "[xoshiro256++][jump_n]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  // jump(k) must land on the same state as k raw next() calls.
  for (const std::uint64_t k : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{2}, std::uint64_t{3},
                                std::uint64_t{5}, std::uint64_t{1000}, std::uint64_t{1} << 20}) {
    INFO("k: " << k);
    simdrng::XoshiroScalar jumped(seed);
    simdrng::XoshiroScalar forward(seed);
    jumped.jump(k);
    for (std::uint64_t i = 0; i < k; ++i)
      forward();
    REQUIRE(jumped.getState() == forward.getState());
  }
}

TEST_CASE("xoshiro256++ jump(n) composition", "[xoshiro256++][jump_n]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  std::mt19937_64 gen(seed);
  std::uniform_int_distribution<std::uint64_t> dist(0, (std::uint64_t{1} << 40) - 1);
  // jump(a); jump(b) == jump(a + b), for random a, b < 2^40 (exercises the
  // high bits of n in square-and-multiply that the forward oracle can't reach).
  for (int t = 0; t < 64; ++t) {
    const std::uint64_t a = dist(gen);
    const std::uint64_t b = dist(gen);
    INFO("a: " << a << " b: " << b);
    simdrng::XoshiroScalar split(seed);
    simdrng::XoshiroScalar combined(seed);
    split.jump(a);
    split.jump(b);
    combined.jump(a + b);
    REQUIRE(split.getState() == combined.getState());
  }
}

TEST_CASE("xoshiro256++ jump(0) is identity", "[xoshiro256++][jump_n]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  simdrng::XoshiroScalar rng(seed);
  const auto before = rng.getState();
  rng.jump(0);
  REQUIRE(rng.getState() == before);
}

TEST_CASE("xoshiro256++ jump(pow2) matches step count", "[xoshiro256++][jump_n]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  // For e < 64 the 2^e stride also fits the uint64 step count, so the two
  // independent poly paths (jump_poly_pow2 vs jump_poly_n) must agree.
  for (const std::uint64_t e : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{5}, std::uint64_t{20},
                                std::uint64_t{40}, std::uint64_t{63}}) {
    INFO("e: " << e);
    simdrng::XoshiroScalar by_pow2(seed);
    simdrng::XoshiroScalar by_steps(seed);
    by_pow2.jump(simdrng::pow2{e});
    by_steps.jump(std::uint64_t{1} << e);
    REQUIRE(by_pow2.getState() == by_steps.getState());
  }
}

TEST_CASE("xoshiro256++ jump(pow2) matches fixed jumps", "[xoshiro256++][jump_n]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  // The 2^128/2^160/2^192 strides exceed uint64, so validate jump_poly_pow2 at
  // those exponents against the trusted hardcoded jump constants.
  {
    simdrng::XoshiroScalar a(seed), b(seed);
    a.jump(simdrng::pow2{128});
    b.jump();
    REQUIRE(a.getState() == b.getState());
  }
  {
    simdrng::XoshiroScalar a(seed), b(seed);
    a.jump(simdrng::pow2{160});
    b.mid_jump();
    REQUIRE(a.getState() == b.getState());
  }
  {
    simdrng::XoshiroScalar a(seed), b(seed);
    a.jump(simdrng::pow2{192});
    b.long_jump();
    REQUIRE(a.getState() == b.getState());
  }
}
