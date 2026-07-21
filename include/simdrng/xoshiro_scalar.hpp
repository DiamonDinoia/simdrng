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
#include <cassert>
#include <cstdint>
#include <limits>

#include "macros.hpp"
#include "splitmix.hpp"

#include <bit>

namespace simdrng {

namespace detail {

/// 4×u64 jump polynomial: bit i (LSB-first) = coefficient of x^i.
using jump_poly_t = std::array<std::uint64_t, 4>;

/// Low 256 coefficients of the degree-256 characteristic polynomial P(x) of the
/// xoshiro256 linear core (the x^256 term is implicit). Identical for ++/** —
/// the scrambler does not touch the linear core. Generated and cross-checked by
/// devel/xoshiro_coeffs.py against the published 2^128/2^160/2^192 constants.
inline constexpr jump_poly_t CHARPOLY = {0x9d116f2bb0f0f001, 0x0280002bcefd1a5e, 0x04b4edcf26259f85,
                                         0x0003c03c3f3ecb19};

/// Multiply a degree-<256 polynomial by x modulo P(x), in place: shift the
/// 256-bit value left by one bit across the 4 little-endian words; if the
/// x^256 term appears, reduce by XOR-ing CHARPOLY (since x^256 ≡ P(x) - x^256).
constexpr void poly_mul_x_mod(jump_poly_t &p) noexcept {
  const std::uint64_t carry = p[3] >> 63;
  p[3] = (p[3] << 1) | (p[2] >> 63);
  p[2] = (p[2] << 1) | (p[1] >> 63);
  p[1] = (p[1] << 1) | (p[0] >> 63);
  p[0] = p[0] << 1;
  if (carry) {
    p[0] ^= CHARPOLY[0];
    p[1] ^= CHARPOLY[1];
    p[2] ^= CHARPOLY[2];
    p[3] ^= CHARPOLY[3];
  }
}

/// Multiply two degree-<256 polynomials modulo P(x). Horner over a's bits from
/// x^255 down to x^0: ×x then conditional XOR of b.
constexpr jump_poly_t polymul_mod(const jump_poly_t &a, const jump_poly_t &b) noexcept {
  jump_poly_t r{};
  for (int w = 3; w >= 0; --w) {
    for (int bit = 63; bit >= 0; --bit) {
      poly_mul_x_mod(r);
      if ((a[static_cast<std::size_t>(w)] >> bit) & 1U) {
        r[0] ^= b[0];
        r[1] ^= b[1];
        r[2] ^= b[2];
        r[3] ^= b[3];
      }
    }
  }
  return r;
}

/// Jump polynomial for an arbitrary count: x^n mod P(x) via square-and-multiply
/// in GF(2)[x]. Pure integer code ⇒ usable in constant expressions.
constexpr jump_poly_t jump_poly_n(std::uint64_t n) noexcept {
  jump_poly_t result = {1, 0, 0, 0}; // x^0
  jump_poly_t base = {2, 0, 0, 0};   // x^1
  while (n != 0) {
    if ((n & 1U) != 0U) {
      result = polymul_mod(result, base);
    }
    n >>= 1;
    if (n != 0) {
      base = polymul_mod(base, base);
    }
  }
  return result;
}

/// Jump polynomial for a power-of-two count: x^(2^e) mod P(x), i.e. squaring x
/// e times. Reaches strides up to 2^255 that don't fit a uint64 step count.
/// ord_2(2^256-1) = 256, so strides repeat with period 256 in e — every
/// distinct power-of-two stride is e in [0, 255]; larger e is redundant and
/// would loop e times, so it is rejected rather than silently spun on.
constexpr jump_poly_t jump_poly_pow2(std::uint64_t e) noexcept {
  assert(e < 256 && "pow2 jump exponent must be < 256 (larger strides repeat)");
  jump_poly_t p = {2, 0, 0, 0}; // x = x^(2^0)
  for (std::uint64_t i = 0; i < e; ++i) {
    p = polymul_mod(p, p);
  }
  return p;
}

} // namespace detail

/// Tag for a power-of-two jump: jump(pow2{e}) advances the generator by 2^e
/// outputs, the same convention as the fixed jump()/mid_jump()/long_jump()
/// strides (2^128 / 2^160 / 2^192).
struct pow2 {
  std::uint64_t exponent;
};

/**
 * @class XoshiroScalar
 * @brief A class implementing the XoshiroScalar random number generator.
 */
class XoshiroScalar {
public:
  using result_type = std::uint64_t;
  static constexpr SIMDRNG_ALWAYS_INLINE auto(min)() noexcept { return std::numeric_limits<result_type>::min(); }
  static constexpr SIMDRNG_ALWAYS_INLINE auto(max)() noexcept { return std::numeric_limits<result_type>::max(); }

  /**
   * @brief Constructs the XoshiroScalar generator with a given seed.
   * @param seed The seed value.
   */
  SIMDRNG_ALWAYS_INLINE constexpr explicit XoshiroScalar(const result_type seed) noexcept : m_state{} {
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
  SIMDRNG_ALWAYS_INLINE constexpr explicit XoshiroScalar(const result_type seed, const result_type thread_id) noexcept
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
  SIMDRNG_ALWAYS_INLINE constexpr explicit XoshiroScalar(const result_type seed, const result_type thread_id,
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
  SIMDRNG_ALWAYS_INLINE constexpr result_type operator()() noexcept { return next(); }

  /**
   * @brief Generates a uniform random number in the range [0, 1).
   * @return A uniform random number.
   */
  SIMDRNG_ALWAYS_INLINE constexpr double(uniform)() noexcept { return static_cast<double>(next() >> 11) * 0x1.0p-53; }

  /**
   * @brief Returns the state of the generator.
   * @return The state of the generator.
   */
  SIMDRNG_ALWAYS_INLINE constexpr const std::array<result_type, 4> &getState() const noexcept { return m_state; }
  SIMDRNG_ALWAYS_INLINE constexpr void setState(const std::array<result_type, 4> &state) noexcept { m_state = state; }

  /**
   * @brief Returns the size of the state array.
   * @return The size of the state array.
   */
  static constexpr SIMDRNG_ALWAYS_INLINE result_type stateSize() noexcept { return 4; }

  /**
   * @brief Jump function for the generator. It is equivalent to 2^128 calls to next().
   * It can be used to generate 2^128 non-overlapping subsequences for simd computations.
   */
  SIMDRNG_ALWAYS_INLINE constexpr void jump() noexcept {
    apply_jump_poly({0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c});
  }

  /**
   * @brief Jump function for the generator. It is equivalent to 2^160 calls to next().
   * It can be used to generate 2^96 non-overlapping subsequences for parallel computations.
   */
  SIMDRNG_ALWAYS_INLINE constexpr void mid_jump() noexcept {
    apply_jump_poly({0xc04b4f9c5d26c200, 0x69e6e6e431a2d40b, 0x4823b45b89dc689c, 0xf567382197055bf0});
  }

  /**
   * @brief Long-jump function for the generator. It is equivalent to 2^192 calls to next().
   * It can be used to generate 2^64 starting points, from each of which jump() will generate 2^64 non-overlapping
   * subsequences for parallel distributed computations.
   */
  SIMDRNG_ALWAYS_INLINE constexpr void long_jump() noexcept {
    apply_jump_poly({0x76e15d3efefdcbbf, 0xc5004e441c522fb3, 0x77710069854ee241, 0x39109bb02acbe635});
  }

  /**
   * @brief Arbitrary jump-ahead: advances the state by exactly @p n calls to next().
   * Computes the jump polynomial x^n mod P(x) on the fly, so any offset in [0, 2^64) works.
   * Overload of jump(); the no-argument jump() remains the fixed 2^128 stride.
   * @param n The number of steps to jump ahead.
   */
  SIMDRNG_ALWAYS_INLINE constexpr void jump(std::uint64_t n) noexcept { apply_jump_poly(detail::jump_poly_n(n)); }

  /**
   * @brief Power-of-two jump-ahead: advances the state by exactly 2^p.exponent calls to next().
   * Reaches strides too large for the uint64 step count (e.g. jump(pow2{128}) == jump()).
   * @param p The exponent (as a pow2 tag) of the 2^e stride.
   */
  SIMDRNG_ALWAYS_INLINE constexpr void jump(pow2 p) noexcept { apply_jump_poly(detail::jump_poly_pow2(p.exponent)); }

private:
  std::array<result_type, 4> m_state;

  /**
   * @brief Applies a jump polynomial to the state: sets the state to poly(T) · state, where T is the
   * linear transition. Shared by all jump variants — they differ only in how the polynomial is obtained.
   */
  SIMDRNG_ALWAYS_INLINE constexpr void apply_jump_poly(const detail::jump_poly_t &poly) noexcept {
    result_type s0 = 0;
    result_type s1 = 0;
    result_type s2 = 0;
    result_type s3 = 0;
    for (const auto word : poly)
      for (auto b = 0; b < 64; b++) {
        if ((word & result_type{1} << b) != 0U) {
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
   * @brief Rotates the bits of a 64-bit integer to the left.
   * @param x The integer to rotate.
   * @param k The number of bits to rotate.
   * @return The rotated integer.
   */
  static constexpr SIMDRNG_ALWAYS_INLINE auto rotl(const result_type x, const int k) noexcept {
    return std::rotl(x, k);
  }

  /**
   * @brief Generates the next state of the generator.
   * @return The next state.
   */
  SIMDRNG_ALWAYS_INLINE constexpr result_type next() noexcept {
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

} // namespace simdrng
