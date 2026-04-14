#include <random>

#include <catch2/catch_all.hpp>

#include <random/chacha.hpp>
#include <random/chacha_simd.hpp>

using ChaCha20Reference = prng::ChaCha<20>;
using ChaCha20SIMD = prng::ChaChaNative<20>;

TEST_CASE("SEED", "[chacha]") {
  static constexpr auto seed_tests = 1 << 14;
  auto seed = std::random_device{}();
  INFO("SEED: " << seed);
  std::mt19937 rng32(seed);
  std::mt19937_64 rng64(seed);
  ChaCha20SIMD::input_word counter = rng64(), nonce = rng64();
  std::array<ChaCha20SIMD::matrix_word, 8> key;
  for (int i = 0; i < 8; i++) {
    key[i] = rng32();
  }

  ChaCha20Reference chaCha20Reference(key, counter, nonce);
  ChaCha20SIMD chaCha20SIMD(key, counter, nonce);
  for (auto i = 0; i < seed_tests; ++i) {
    REQUIRE(chaCha20SIMD.getState() == chaCha20Reference.getState());
    REQUIRE(chaCha20SIMD.block() == chaCha20Reference.block());
  }
}

TEST_CASE("COUNTER OVERFLOW", "[chacha]") {
  static constexpr auto overflow_tests = 1 << 14;
  const auto simd_size = ChaCha20SIMD({}, 0, 0).getSIMDSize();
  auto seed = std::random_device{}();
  INFO("SEED: " << seed);
  std::mt19937 rng32(seed);
  std::mt19937_64 rng64(seed);
  std::uniform_int_distribution<> rngOverflow(1, (simd_size - 1));

  ChaCha20SIMD::input_word nonce = rng64();
  std::array<ChaCha20SIMD::matrix_word, 8> key;
  for (int i = 0; i < 8; i++) {
    key[i] = rng32();
  }

  for (auto i = 0; i < overflow_tests; ++i) {
    // This next line generates a random counter value that will always result in an overflow
    // of at least one batched block upon the next invocation of the chacha20 simd implementation.
    ChaCha20SIMD::input_word counter = (static_cast<uint64_t>(rng32()) << 32) | (0xFFFFFFFF - rngOverflow(rng32));
    ChaCha20Reference chaCha20Reference(key, counter, nonce);
    ChaCha20SIMD chaCha20SIMD(key, counter, nonce);
    for (std::size_t lane = 0; lane < simd_size; ++lane) {
      INFO("lane: " << lane);
      REQUIRE(chaCha20SIMD.getState() == chaCha20Reference.getState());
      REQUIRE(chaCha20SIMD.block() == chaCha20Reference.block());
    }
  }
}
