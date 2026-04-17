#pragma once

#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "macros.hpp"

namespace prng {

namespace internal {

template <std::uint8_t N, std::uint8_t W>
struct PhiloxConstants;

template <>
struct PhiloxConstants<4, 32> {
  static constexpr std::uint32_t M0 = 0xD2511F53;
  static constexpr std::uint32_t M1 = 0xCD9E8D57;
  static constexpr std::uint32_t W0 = 0x9E3779B9;
  static constexpr std::uint32_t W1 = 0xBB67AE85;
};

template <>
struct PhiloxConstants<2, 32> {
  static constexpr std::uint32_t M0 = 0xD256D193;
  static constexpr std::uint32_t W0 = 0x9E3779B9;
};

template <>
struct PhiloxConstants<4, 64> {
  static constexpr std::uint64_t M0 = 0xD2E7470EE14C6C93ULL;
  static constexpr std::uint64_t M1 = 0xCA5A826395121157ULL;
  static constexpr std::uint64_t W0 = 0x9E3779B97F4A7C15ULL;
  static constexpr std::uint64_t W1 = 0xBB67AE8584CAA73BULL;
};

template <>
struct PhiloxConstants<2, 64> {
  static constexpr std::uint64_t M0 = 0xD2B74407B1CE6E93ULL;
  static constexpr std::uint64_t W0 = 0x9E3779B97F4A7C15ULL;
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

  static constexpr PRNG_ALWAYS_INLINE auto(min)() noexcept {
    return (std::numeric_limits<result_type>::min)();
  }

  static constexpr PRNG_ALWAYS_INLINE auto(max)() noexcept {
    return (std::numeric_limits<result_type>::max)();
  }

  explicit PRNG_ALWAYS_INLINE Philox(result_type seed, result_type counter = 0) noexcept
      : m_counter(counter_from_uint64(counter)), m_key(seed_to_key(seed)) {}

  explicit PRNG_ALWAYS_INLINE Philox(key_type key, counter_type counter) noexcept
      : m_counter(counter), m_key(key) {}

  PRNG_ALWAYS_INLINE constexpr result_type operator()() noexcept {
    if (m_result_index >= RESULTS_PER_BLOCK) [[unlikely]] {
      m_result_cache = next_block();
      m_result_index = 0;
    }
    return m_result_cache[m_result_index++];
  }

  PRNG_ALWAYS_INLINE constexpr double uniform() noexcept {
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

  key_type getKey() const noexcept { return m_key; }

  void setCounter(const counter_type &ctr) noexcept {
    m_counter = ctr;
    m_result_index = RESULTS_PER_BLOCK;
  }

  void setKey(const key_type &key) noexcept {
    m_key = key;
    m_result_index = RESULTS_PER_BLOCK;
  }

  counter_type getCounterForSerde() const noexcept {
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

  static constexpr PRNG_ALWAYS_INLINE void mulhilo(word_type a, word_type b,
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

  static constexpr PRNG_ALWAYS_INLINE void single_round(counter_type &ctr, key_type &key) noexcept {
    if constexpr (N == 4) {
      word_type hi0, lo0, hi1, lo1;
      mulhilo(C::M0, ctr[0], hi0, lo0);
      mulhilo(C::M1, ctr[2], hi1, lo1);
      ctr = {hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0};
      key[0] += C::W0;
      key[1] += C::W1;
    } else {
      word_type hi, lo;
      mulhilo(C::M0, ctr[0], hi, lo);
      ctr = {hi ^ ctr[1] ^ key[0], lo};
      key[0] += C::W0;
    }
  }

  static constexpr PRNG_ALWAYS_INLINE counter_type philox_rounds(counter_type ctr, key_type key) noexcept {
    for (std::uint8_t i = 0; i < R; ++i) {
      single_round(ctr, key);
    }
    return ctr;
  }

  constexpr PRNG_ALWAYS_INLINE void inc_counter() noexcept {
    for (std::uint8_t i = 0; i < N; ++i) {
      if (++m_counter[i] != 0) break;
    }
  }

  static constexpr PRNG_ALWAYS_INLINE void dec_counter(counter_type &ctr) noexcept {
    for (std::uint8_t i = 0; i < N; ++i) {
      if (ctr[i]-- != 0) break;
    }
  }

  PRNG_FLATTEN constexpr PRNG_ALWAYS_INLINE result_block_type next_block() noexcept {
    auto output = philox_rounds(m_counter, m_key);
    inc_counter();
    return std::bit_cast<result_block_type>(output);
  }
};

using Philox4x32 = Philox<4, 32, 10>;
using Philox2x32 = Philox<2, 32, 10>;
using Philox4x64 = Philox<4, 64, 10>;
using Philox2x64 = Philox<2, 64, 10>;

} // namespace prng
