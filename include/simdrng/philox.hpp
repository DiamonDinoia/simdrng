#pragma once

#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "macros.hpp"

namespace simdrng {

namespace internal {

// Per-variant Philox constants, laid out as N/2-wide arrays so the round is a
// single data-driven loop (see Philox::single_round) rather than an N-specific
// branch:
//   MUL  - the per-lane multipliers (one per counter pair).
//   BUMP - the per-key Weyl increments applied each round.
//   PERM - the Philox output permutation: output pair j takes its high/low words
//          from the product of pair PERM[j]. This cross-wiring is what makes the
//          round a bijection; it is the only part that differs between 2x and 4x.
// Adding support for another N is purely adding a specialization here; the round
// code is already generic over N/2.
template <std::uint8_t N, std::uint8_t W>
struct PhiloxConstants;

template <>
struct PhiloxConstants<4, 32> {
  static constexpr std::array<std::uint32_t, 2> MUL = {0xD2511F53, 0xCD9E8D57};
  static constexpr std::array<std::uint32_t, 2> BUMP = {0x9E3779B9, 0xBB67AE85};
  static constexpr std::array<std::uint8_t, 2> PERM = {1, 0};
};

template <>
struct PhiloxConstants<2, 32> {
  static constexpr std::array<std::uint32_t, 1> MUL = {0xD256D193};
  static constexpr std::array<std::uint32_t, 1> BUMP = {0x9E3779B9};
  static constexpr std::array<std::uint8_t, 1> PERM = {0};
};

template <>
struct PhiloxConstants<4, 64> {
  static constexpr std::array<std::uint64_t, 2> MUL = {0xD2E7470EE14C6C93ULL, 0xCA5A826395121157ULL};
  static constexpr std::array<std::uint64_t, 2> BUMP = {0x9E3779B97F4A7C15ULL, 0xBB67AE8584CAA73BULL};
  static constexpr std::array<std::uint8_t, 2> PERM = {1, 0};
};

template <>
struct PhiloxConstants<2, 64> {
  static constexpr std::array<std::uint64_t, 1> MUL = {0xD2B74407B1CE6E93ULL};
  static constexpr std::array<std::uint64_t, 1> BUMP = {0x9E3779B97F4A7C15ULL};
  static constexpr std::array<std::uint8_t, 1> PERM = {0};
};

} // namespace internal

template <std::uint8_t N = 4, std::uint8_t W = 32, std::uint8_t R = 10>
class Philox {
  static_assert(N == 2 || N == 4, "Philox N must be 2 or 4");
  static_assert(W == 32 || W == 64, "Philox W must be 32 or 64");
  static_assert(R > 0, "Philox rounds must be > 0");

public:
  using result_type = std::uint64_t;
  using word_type = std::conditional_t<W == 32, std::uint32_t, std::uint64_t>;
  using counter_type = std::array<word_type, N>;
  using key_type = std::array<word_type, N / 2>;

  static constexpr auto RESULTS_PER_BLOCK = std::uint8_t{N * W / 64};
  using result_block_type = std::array<result_type, RESULTS_PER_BLOCK>;

  static constexpr SIMDRNG_ALWAYS_INLINE auto(min)() noexcept {
    return (std::numeric_limits<result_type>::min)();
  }

  static constexpr SIMDRNG_ALWAYS_INLINE auto(max)() noexcept {
    return (std::numeric_limits<result_type>::max)();
  }

  explicit SIMDRNG_ALWAYS_INLINE Philox(result_type seed, result_type counter = 0) noexcept
      : m_counter(counter_from_uint64(counter)), m_key(seed_to_key(seed)) {}

  explicit SIMDRNG_ALWAYS_INLINE Philox(key_type key, counter_type counter) noexcept
      : m_counter(counter), m_key(key) {}

  SIMDRNG_ALWAYS_INLINE constexpr result_type operator()() noexcept {
    if (m_result_index >= RESULTS_PER_BLOCK) [[unlikely]] {
      m_result_cache = next_block();
      m_result_index = 0;
    }
    return m_result_cache[m_result_index++];
  }

  SIMDRNG_ALWAYS_INLINE constexpr double uniform() noexcept {
    return static_cast<double>(operator()() >> 11) * 0x1.0p-53;
  }

  counter_type getCounter() const noexcept {
    if (m_result_index < RESULTS_PER_BLOCK) {
      counter_type ctr = m_counter;
      dec_counter(ctr);
      return ctr;
    }
    return m_counter;
  }

  const key_type& getKey() const noexcept { return m_key; }

  void setCounter(const counter_type &ctr) noexcept {
    m_counter = ctr;
    m_result_index = RESULTS_PER_BLOCK;
  }

  void setKey(const key_type &key) noexcept {
    m_key = key;
    m_result_index = RESULTS_PER_BLOCK;
  }

  const counter_type& getCounterForSerde() const noexcept {
    return m_counter;
  }

  void setState(const counter_type &ctr, const key_type &key) noexcept {
    m_counter = ctr;
    m_key = key;
    m_result_index = RESULTS_PER_BLOCK;
  }

  const result_block_type &result_cache() const noexcept { return m_result_cache; }
  void set_result_cache(const result_block_type &cache) noexcept { m_result_cache = cache; }
  std::uint8_t result_index() const noexcept { return m_result_index; }
  void set_result_index(std::uint8_t idx) noexcept { m_result_index = idx; }

  static constexpr key_type seed_to_key(result_type seed) noexcept {
    key_type key{};
    auto state = seed;
    auto splitmix = [&state]() -> std::uint64_t {
      state += 0x9e3779b97f4a7c15ULL;
      auto z = state;
      z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
      z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
      return z ^ (z >> 31);
    };
    if constexpr (W == 32) {
      auto z = splitmix();
      key[0] = static_cast<word_type>(z);
      if constexpr (N == 4) {
        key[1] = static_cast<word_type>(z >> 32);
      }
    } else {
      key[0] = static_cast<word_type>(splitmix());
      if constexpr (N == 4) {
        key[1] = static_cast<word_type>(splitmix());
      }
    }
    return key;
  }

  static constexpr counter_type counter_from_uint64(result_type counter) noexcept {
    counter_type ctr{};
    if constexpr (W == 32) {
      ctr[0] = static_cast<word_type>(counter & 0xFFFFFFFF);
      if constexpr (N >= 2) {
        ctr[1] = static_cast<word_type>(counter >> 32);
      }
    } else {
      ctr[0] = static_cast<word_type>(counter);
    }
    return ctr;
  }

private:
  using C = internal::PhiloxConstants<N, W>;

  counter_type m_counter;
  key_type m_key;
  result_block_type m_result_cache{};
  std::uint8_t m_result_index = RESULTS_PER_BLOCK;

  static constexpr SIMDRNG_ALWAYS_INLINE void mulhilo(word_type a, word_type b,
                                                     word_type &hi, word_type &lo) noexcept {
    if constexpr (W == 32) {
      auto product = static_cast<std::uint64_t>(a) * static_cast<std::uint64_t>(b);
      lo = static_cast<word_type>(product);
      hi = static_cast<word_type>(product >> 32);
    } else {
#if defined(__SIZEOF_INT128__)
      auto product = static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b);
      lo = static_cast<word_type>(product);
      hi = static_cast<word_type>(product >> 64);
#else
      constexpr std::uint64_t MASK32 = 0xFFFFFFFF;
      std::uint64_t a_lo = a & MASK32, a_hi = a >> 32;
      std::uint64_t b_lo = b & MASK32, b_hi = b >> 32;
      std::uint64_t p_ll = a_lo * b_lo;
      std::uint64_t p_lh = a_lo * b_hi;
      std::uint64_t p_hl = a_hi * b_lo;
      std::uint64_t p_hh = a_hi * b_hi;
      std::uint64_t mid = (p_ll >> 32) + (p_lh & MASK32) + (p_hl & MASK32);
      lo = (p_ll & MASK32) | (mid << 32);
      hi = p_hh + (p_lh >> 32) + (p_hl >> 32) + (mid >> 32);
#endif
    }
  }

  static constexpr SIMDRNG_ALWAYS_INLINE void single_round(counter_type &ctr, key_type &key) noexcept {
    // N/2 is a compile-time constant, so these loops fully unroll; they stay
    // constexpr and keep this scalar header free of any SIMD/poet dependency.
    constexpr std::uint8_t PAIRS = N / 2;
    std::array<word_type, PAIRS> hi{};
    std::array<word_type, PAIRS> lo{};
    for (std::uint8_t j = 0; j < PAIRS; ++j) {
      mulhilo(C::MUL[j], ctr[2 * j], hi[j], lo[j]);
    }
    counter_type out{};
    for (std::uint8_t j = 0; j < PAIRS; ++j) {
      const auto s = C::PERM[j];
      out[2 * j] = hi[s] ^ ctr[2 * j + 1] ^ key[j];
      out[2 * j + 1] = lo[s];
    }
    ctr = out;
    for (std::uint8_t j = 0; j < PAIRS; ++j) {
      key[j] += C::BUMP[j];
    }
  }

  static constexpr SIMDRNG_ALWAYS_INLINE counter_type philox_rounds(counter_type ctr, key_type key) noexcept {
    for (std::uint8_t i = 0; i < R; ++i) {
      single_round(ctr, key);
    }
    return ctr;
  }

  constexpr SIMDRNG_ALWAYS_INLINE void inc_counter() noexcept {
    for (std::uint8_t i = 0; i < N; ++i) {
      if (++m_counter[i] != 0) break;
    }
  }

  static constexpr SIMDRNG_ALWAYS_INLINE void dec_counter(counter_type &ctr) noexcept {
    for (std::uint8_t i = 0; i < N; ++i) {
      if (ctr[i]-- != 0) break;
    }
  }

  SIMDRNG_FLATTEN constexpr SIMDRNG_ALWAYS_INLINE result_block_type next_block() noexcept {
    auto output = philox_rounds(m_counter, m_key);
    inc_counter();
    return std::bit_cast<result_block_type>(output);
  }
};

using Philox4x32 = Philox<4, 32, 10>;
using Philox2x32 = Philox<2, 32, 10>;
using Philox4x64 = Philox<4, 64, 10>;
using Philox2x64 = Philox<2, 64, 10>;

} // namespace simdrng
