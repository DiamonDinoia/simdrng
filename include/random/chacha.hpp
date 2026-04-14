#pragma once

#include <array>
#include <bit>
#include <cstdint>
#include <limits>

#include "macros.hpp"

namespace prng {

template<std::uint8_t R = 20>
class ChaCha {

protected:
  static constexpr auto MATRIX_WORDCOUNT = std::uint8_t{16};
  static constexpr auto KEY_WORDCOUNT = std::uint8_t{8};

public:
  using result_type = std::uint64_t;
  using input_word = std::uint64_t;
  using matrix_word = std::uint32_t;
  using matrix_type = std::array<matrix_word, MATRIX_WORDCOUNT>;
  using result_cache_type = std::array<result_type, MATRIX_WORDCOUNT / 2>;

  static constexpr PRNG_ALWAYS_INLINE auto(min)() noexcept {
    return (std::numeric_limits<result_type>::min)();
  }

  static constexpr PRNG_ALWAYS_INLINE auto(max)() noexcept {
    return (std::numeric_limits<result_type>::max)();
  }

  /**
   * @brief Construct a scalar ChaCha generator with given key, counter and nonce
   * @param key A 256-bit key, divided up into eight 32-bit words.
   * @param counter Initial value of the counter.
   * @param nonce Initial value of the nonce.
   */
  PRNG_ALWAYS_INLINE explicit ChaCha(
    const std::array<matrix_word, KEY_WORDCOUNT> key,
    const input_word counter,
    const input_word nonce
  ) noexcept {
    // First four words (i.e. top-row) are always the same constants
    // They spell out "expand 2-byte k" in ASCII (little-endian)
    m_state[0] = 0x61707865;
    m_state[1] = 0x3320646e;
    m_state[2] = 0x79622d32;
    m_state[3] = 0x6b206574;

    for (auto i = 0; i < 8; ++i) {
      m_state[4 + i] = key[i];
    }

    // ChaCha assumes little-endianness
    m_state[12] = static_cast<matrix_word>(counter & 0xFFFFFFFF);
    m_state[13] = static_cast<matrix_word>(counter >> 32);
    m_state[14] = static_cast<matrix_word>(nonce & 0xFFFFFFFF);
    m_state[15] = static_cast<matrix_word>(nonce >> 32);
  }

  /**
   * @brief Generates the next 64-bit output.
   * @return The next 64-bit output.
   */
  PRNG_ALWAYS_INLINE constexpr result_type(operator())() noexcept { return next_result(); }

  /**
   * @brief Generates a uniform random number in the range [0, 1).
   * @return A uniform random number.
   */
  PRNG_ALWAYS_INLINE constexpr double uniform() noexcept {
    return static_cast<double>(operator()() >> 11) * 0x1.0p-53;
  }

  /**
   * @brief Generates the next 64-byte ChaCha block.
   * @return The next 64-byte ChaCha block.
   */
  PRNG_ALWAYS_INLINE constexpr matrix_type block() noexcept {
    if (m_result_index < m_result_cache.size()) {
      auto cached_block = results_to_block(m_result_cache);
      m_result_index = static_cast<std::uint8_t>(m_result_cache.size());
      return cached_block;
    }
    return next_block();
  }

  /**
   * @brief Returns the state of the generator; a 4x4 matrix.
   * @return State of the generator.
   */
  PRNG_ALWAYS_INLINE constexpr matrix_type getState() const noexcept {
    matrix_type state = m_state;
    if (m_result_index < m_result_cache.size()) {
      const input_word counter =
        (static_cast<input_word>(state[13]) << 32) | static_cast<input_word>(state[12]);
      const input_word current_counter = counter - 1;
      state[12] = static_cast<matrix_word>(current_counter & 0xFFFFFFFF);
      state[13] = static_cast<matrix_word>(current_counter >> 32);
    }
    return state;
  }

private:
  matrix_type m_state;
  result_cache_type m_result_cache{};
  std::uint8_t m_result_index = static_cast<std::uint8_t>(m_result_cache.size());

  static constexpr PRNG_ALWAYS_INLINE auto rotl(const matrix_word x, const int k) noexcept {
    return std::rotl(x, k);
  }

  static constexpr PRNG_ALWAYS_INLINE void quarter_round(
    matrix_type &m,
    const unsigned int a,
    const unsigned int b,
    const unsigned int c,
    const unsigned int d
  ) noexcept {
    m[a] += m[b]; m[d] ^= m[a]; m[d] = rotl(m[d], 16);
    m[c] += m[d]; m[b] ^= m[c]; m[b] = rotl(m[b], 12);
    m[a] += m[b]; m[d] ^= m[a]; m[d] = rotl(m[d],  8);
    m[c] += m[d]; m[b] ^= m[c]; m[b] = rotl(m[b],  7);
  }

  constexpr PRNG_ALWAYS_INLINE void inc_counter() noexcept {
    if (++m_state[12] == 0) {
      ++m_state[13];
    }
  }

  static constexpr PRNG_ALWAYS_INLINE result_cache_type block_to_results(const matrix_type& block) noexcept {
    return std::bit_cast<result_cache_type>(block);
  }

  static constexpr PRNG_ALWAYS_INLINE matrix_type results_to_block(const result_cache_type& results) noexcept {
    return std::bit_cast<matrix_type>(results);
  }

  constexpr PRNG_ALWAYS_INLINE result_type next_result() noexcept {
    if (m_result_index >= m_result_cache.size()) [[unlikely]] {
      m_result_cache = block_to_results(next_block());
      m_result_index = 0;
    }
    return m_result_cache[m_result_index++];
  }

  /**
   * @brief Returns the next output from the generator, then increases state's counter by 1.
   * @return The output for the current internal state.
  */
  PRNG_FLATTEN constexpr PRNG_ALWAYS_INLINE matrix_type next_block() noexcept {
    matrix_type x = m_state;

    // Note that we perform both an odd and even round at the same time.
    // As a result the amount of rounds performed is always rounded up to an even number.
    for (auto i = 0; i < R; i += 2) {
      // Odd round
      quarter_round(x, 0, 4, 8,12);
      quarter_round(x, 1, 5, 9,13);
      quarter_round(x, 2, 6,10,14);
      quarter_round(x, 3, 7,11,15);

      // Even round
      quarter_round(x, 0, 5,10,15);
      quarter_round(x, 1, 6,11,12);
      quarter_round(x, 2, 7, 8,13);
      quarter_round(x, 3, 4, 9,14);
    }

    for (auto i = 0; i < MATRIX_WORDCOUNT; ++i) {
      x[i] += m_state[i];
    }

    inc_counter();

    return x;
  }
};

}
