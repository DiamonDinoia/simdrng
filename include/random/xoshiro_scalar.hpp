/*  Written in 2019 by David Blackman and Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide.

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR
IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

Ported to C++, vectorized, and optimized by Marco Barbone.
Original implementation by David Blackman and Sebastiano
Vigna.
*/

#pragma once

#include <array>
#include <cstdint>
#include <limits>

#include "macros.hpp"
#include "splitmix.hpp"

#include <bit>

namespace prng {

/**
 * @class XoshiroScalar
 * @brief A class implementing the XoshiroScalar random number generator.
 */
class XoshiroScalar {
public:
  using result_type = std::uint64_t;
  static constexpr PRNG_ALWAYS_INLINE auto(min)() noexcept { return std::numeric_limits<result_type>::min(); }
  static constexpr PRNG_ALWAYS_INLINE auto(max)() noexcept { return std::numeric_limits<result_type>::max(); }

  /**
   * @brief Constructs the XoshiroScalar generator with a given seed.
   * @param seed The seed value.
   */
  PRNG_ALWAYS_INLINE constexpr explicit XoshiroScalar(const result_type seed) noexcept : m_state{} {
    SplitMix splitmix{seed};
    for (auto &element : m_state) {
      element = splitmix();
    }
  }

  /**
   * @brief Constructs the XoshiroScalar generator with a given seed and thread ID.
   * @param seed The seed value.
   * @param thread_id The thread ID.
   */
  PRNG_ALWAYS_INLINE constexpr explicit XoshiroScalar(const result_type seed, const result_type thread_id) noexcept
      : XoshiroScalar(seed) {
    for (result_type i = 0; i < thread_id; ++i) {
      jump();
    }
  }

  /**
   * @brief Constructs the XoshiroScalar generator with a given seed and thread ID.
   * @param seed The seed value.
   * @param thread_id The thread ID.
   * @param cluster_id The cluster ID.
   */
  PRNG_ALWAYS_INLINE constexpr explicit XoshiroScalar(const result_type seed, const result_type thread_id,
                                                      const result_type cluster_id) noexcept
      : XoshiroScalar(seed, thread_id) {
    for (result_type i = 0; i < cluster_id; ++i) {
      long_jump();
    }
  }

  /**
   * @brief Generates the next random number.
   * @return The next random number.
   */
  PRNG_ALWAYS_INLINE constexpr result_type(operator())() noexcept { return next(); }

  /**
   * @brief Generates a uniform random number in the range [0, 1).
   * @return A uniform random number.
   */
  PRNG_ALWAYS_INLINE constexpr double(uniform)() noexcept { return static_cast<double>(next() >> 11) * 0x1.0p-53; }

  /**
   * @brief Returns the state of the generator.
   * @return The state of the generator.
   */
  PRNG_ALWAYS_INLINE constexpr std::array<result_type, 4> getState() const noexcept { return m_state; }
  PRNG_ALWAYS_INLINE constexpr void setState(const std::array<result_type, 4> &state) noexcept { m_state = state; }

  /**
   * @brief Returns the size of the state array.
   * @return The size of the state array.
   */
  static constexpr PRNG_ALWAYS_INLINE result_type stateSize() noexcept { return 4; }

  /**
   * @brief Jump function for the generator. It is equivalent to 2^128 calls to next().
   * It can be used to generate 2^128 non-overlapping subsequences for simd computations.
   */
  PRNG_ALWAYS_INLINE constexpr void jump() noexcept {
    constexpr result_type JUMP[] = {0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c};
    result_type s0 = 0;
    result_type s1 = 0;
    result_type s2 = 0;
    result_type s3 = 0;
    for (const auto i : JUMP)
      for (auto b = 0; b < 64; b++) {
        if (i & result_type{1} << b) {
          s0 ^= m_state[0];
          s1 ^= m_state[1];
          s2 ^= m_state[2];
          s3 ^= m_state[3];
        }
        next();
      }
    m_state[0] = s0;
    m_state[1] = s1;
    m_state[2] = s2;
    m_state[3] = s3;
  }

  /**
   * @brief Jump function for the generator. It is equivalent to 2^160 calls to next().
   * It can be used to generate 2^96 non-overlapping subsequences for parallel computations.
   */
  PRNG_ALWAYS_INLINE constexpr void mid_jump() noexcept {
    constexpr result_type JUMP[] = {0xc04b4f9c5d26c200, 0x69e6e6e431a2d40b, 0x4823b45b89dc689c, 0xf567382197055bf0};
    result_type s0 = 0;
    result_type s1 = 0;
    result_type s2 = 0;
    result_type s3 = 0;
    for (const auto i : JUMP)
      for (auto b = 0; b < 64; b++) {
        if (i & result_type{1} << b) {
          s0 ^= m_state[0];
          s1 ^= m_state[1];
          s2 ^= m_state[2];
          s3 ^= m_state[3];
        }
        next();
      }
    m_state[0] = s0;
    m_state[1] = s1;
    m_state[2] = s2;
    m_state[3] = s3;
  }

  /**
   * @brief Long-jump function for the generator. It is equivalent to 2^192 calls to next().
   * It can be used to generate 2^64 starting points, from each of which jump() will generate 2^64 non-overlapping
   * subsequences for parallel distributed computations.
   */
  PRNG_ALWAYS_INLINE constexpr void long_jump() noexcept {
    constexpr result_type LONG_JUMP[] = {0x76e15d3efefdcbbf, 0xc5004e441c522fb3, 0x77710069854ee241,
                                         0x39109bb02acbe635};
    result_type s0 = 0;
    result_type s1 = 0;
    result_type s2 = 0;
    result_type s3 = 0;
    for (unsigned long i : LONG_JUMP)
      for (int b = 0; b < 64; b++) {
        if (i & result_type{1} << b) {
          s0 ^= m_state[0];
          s1 ^= m_state[1];
          s2 ^= m_state[2];
          s3 ^= m_state[3];
        }
        next();
      }

    m_state[0] = s0;
    m_state[1] = s1;
    m_state[2] = s2;
    m_state[3] = s3;
  }

private:
  std::array<result_type, 4> m_state;

  /**
   * @brief Rotates the bits of a 64-bit integer to the left.
   * @param x The integer to rotate.
   * @param k The number of bits to rotate.
   * @return The rotated integer.
   */
  static constexpr PRNG_ALWAYS_INLINE auto rotl(const result_type x, const int k) noexcept {
    return std::rotl(x, k);
  }

  /**
   * @brief Generates the next state of the generator.
   * @return The next state.
   */
  PRNG_ALWAYS_INLINE constexpr result_type next() noexcept {
    const auto result = rotl(m_state[0] + m_state[3], 23) + m_state[0];
    const auto t_shift = m_state[1] << 17;

    m_state[2] ^= m_state[0];
    m_state[3] ^= m_state[1];
    m_state[1] ^= m_state[2];
    m_state[0] ^= m_state[3];

    m_state[2] ^= t_shift;
    m_state[3] = rotl(m_state[3], 45);

    return result;
  }
};

} // namespace prng
