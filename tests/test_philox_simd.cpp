#include <random>

#include <catch2/catch_all.hpp>

#include <random/philox.hpp>
#include <random/philox_simd.hpp>

// Type aliases to avoid comma issues in Catch2 TEMPLATE_TEST_CASE macros
#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE
using Native4x32 = prng::PhiloxNative<4, 32, 10>;
using Native2x32 = prng::PhiloxNative<2, 32, 10>;
using Native4x64 = prng::PhiloxNative<4, 64, 10>;
using Native2x64 = prng::PhiloxNative<2, 64, 10>;

using Scalar4x32 = prng::Philox<4, 32, 10>;
using Scalar2x32 = prng::Philox<2, 32, 10>;
using Scalar4x64 = prng::Philox<4, 64, 10>;
using Scalar2x64 = prng::Philox<2, 64, 10>;

using Pair4x32 = std::pair<Native4x32, Scalar4x32>;
using Pair2x32 = std::pair<Native2x32, Scalar2x32>;
using Pair4x64 = std::pair<Native4x64, Scalar4x64>;
using Pair2x64 = std::pair<Native2x64, Scalar2x64>;

// Test SIMD (Native) vs scalar consistency for all four NxW combos
TEMPLATE_TEST_CASE("Philox SIMD vs scalar consistency", "[philox]",
    Pair4x32, Pair2x32, Pair4x64, Pair2x64) {
  using Native = typename TestType::first_type;
  using Scalar = typename TestType::second_type;

  static constexpr auto tests = 1 << 14;
  auto seed = std::random_device{}();
  INFO("SEED: " << seed);
  std::mt19937_64 rng64(seed);

  auto key_seed = rng64();
  auto counter_seed = rng64();

  auto key = Scalar::seed_to_key(key_seed);
  auto counter = Scalar::counter_from_uint64(counter_seed);

  Native simd(key, counter);
  Scalar scalar(key, counter);

  for (auto i = 0; i < tests; ++i) {
    REQUIRE(simd() == scalar());
  }
}

// Test counter overflow with carry propagation within a SIMD batch
TEMPLATE_TEST_CASE("Philox SIMD counter overflow", "[philox]",
    Pair4x32, Pair2x32, Pair4x64, Pair2x64) {
  using Native = typename TestType::first_type;
  using Scalar = typename TestType::second_type;
  using word_type = typename Scalar::word_type;
  using counter_type = typename Scalar::counter_type;
  using key_type = typename Scalar::key_type;

  static constexpr auto overflow_tests = 1 << 12;
  auto seed = std::random_device{}();
  INFO("SEED: " << seed);
  std::mt19937_64 rng64(seed);

  auto key = Scalar::seed_to_key(rng64());
  const auto simd_size = Native(key, counter_type{}).getSIMDSize();

  std::uniform_int_distribution<int> rngOverflow(1, static_cast<int>(simd_size - 1));
  std::mt19937 rng32(seed);

  for (auto i = 0; i < overflow_tests; ++i) {
    counter_type counter{};
    counter[0] = static_cast<word_type>(~word_type{0}) - static_cast<word_type>(rngOverflow(rng32));

    Native simd(key, counter);
    Scalar scalar(key, counter);

    auto total = simd_size * Scalar::RESULTS_PER_BLOCK;
    for (std::size_t j = 0; j < total; ++j) {
      INFO("iteration: " << i << " element: " << j);
      REQUIRE(simd() == scalar());
    }
  }
}

#endif // XSIMD_NO_SUPPORTED_ARCHITECTURE

// Dispatch class type aliases
using SIMD4x32 = prng::PhiloxSIMD<4, 32, 10>;
using SIMD2x32 = prng::PhiloxSIMD<2, 32, 10>;
using SIMD4x64 = prng::PhiloxSIMD<4, 64, 10>;
using SIMD2x64 = prng::PhiloxSIMD<2, 64, 10>;

// Test dispatch-based class: two instances with same seed produce same output
TEMPLATE_TEST_CASE("Philox SIMD dispatch consistency", "[philox]",
    SIMD4x32, SIMD2x32, SIMD4x64, SIMD2x64) {
  auto seed = std::random_device{}();
  INFO("SEED: " << seed);

  TestType a(static_cast<uint64_t>(seed));
  TestType b(static_cast<uint64_t>(seed));
  for (int i = 0; i < 4096; ++i) {
    REQUIRE(a() == b());
  }
}
