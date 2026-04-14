#pragma once

#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <xsimd/xsimd.hpp>

#include "macros.hpp"

namespace prng {

namespace internal {

/**
 * SIMD state for the ChaCha generator.
 * Contains the matrix state, SIMD block cache, and all generation logic.
 *
 * @tparam Arch The xsimd architecture type.
 * @tparam R Number of ChaCha rounds.
 */
template <class Arch, std::uint8_t R> struct ChaChaState {
  static constexpr auto MATRIX_WORDCOUNT = std::uint8_t{16};
  static constexpr auto KEY_WORDCOUNT = std::uint8_t{8};

  using input_word = std::uint64_t;
  using matrix_word = std::uint32_t;
  using matrix_type = std::array<matrix_word, MATRIX_WORDCOUNT>;
  using simd_type = xsimd::batch<matrix_word, Arch>;
  using working_state_type = std::array<simd_type, MATRIX_WORDCOUNT>;

  static constexpr std::uint8_t SIMD_WIDTH = std::uint8_t{simd_type::size};
  // Guard: SIMD_WIDTH may be 0 when an arch is instantiated in the byte storage
  // but not available at compile time. Operations are only invoked from dispatch
  // TUs compiled with the correct -march flags.
  static_assert(SIMD_WIDTH == 0 || std::has_single_bit(static_cast<unsigned int>(SIMD_WIDTH)),
                "ChaCha SIMD width must be a power of two");
  static constexpr std::uint8_t SIMD_WIDTH_SHIFT =
      SIMD_WIDTH > 0 ? static_cast<std::uint8_t>(std::countr_zero(static_cast<unsigned int>(SIMD_WIDTH))) : 0;
  static_assert(SIMD_WIDTH == 0 || MATRIX_WORDCOUNT % SIMD_WIDTH == 0,
                "ChaCha state must divide evenly into SIMD segments");
  static constexpr std::uint8_t SIMD_WIDTH_MASK = SIMD_WIDTH > 0 ? std::uint8_t(SIMD_WIDTH - 1) : 0;
  static constexpr std::uint8_t BLOCK_SEGMENTCOUNT =
      SIMD_WIDTH > 0 ? static_cast<std::uint8_t>(MATRIX_WORDCOUNT / SIMD_WIDTH) : 0;
  static constexpr std::uint8_t cache_batchcount() noexcept {
    if constexpr (std::is_base_of_v<xsimd::avx512f, Arch>) {
      return 2;
    } else {
      return 1;
    }
  }

  static constexpr auto CACHE_BATCHCOUNT = cache_batchcount();
  static constexpr auto CACHE_BLOCKCOUNT = std::uint8_t{CACHE_BATCHCOUNT * SIMD_WIDTH};
  using cache_block_type = std::array<simd_type, BLOCK_SEGMENTCOUNT>;
  using cache_batch_type = std::array<cache_block_type, SIMD_WIDTH>;
  static_assert(sizeof(cache_block_type) == sizeof(matrix_type),
                "Cache blocks must have the same layout size as a ChaCha block");
  static_assert(std::is_trivially_copyable_v<cache_block_type>, "Cache blocks must be trivially copyable for bit_cast");
  static_assert(std::is_trivially_copyable_v<matrix_type>, "ChaCha blocks must be trivially copyable for bit_cast");

  matrix_type m_state;
  alignas(simd_type::arch_type::alignment()) std::array<cache_batch_type, CACHE_BATCHCOUNT> m_cache;
  std::uint8_t m_cache_index = CACHE_BLOCKCOUNT;

  explicit PRNG_ALWAYS_INLINE ChaChaState(const std::array<matrix_word, KEY_WORDCOUNT> key, const input_word counter,
                                          const input_word nonce) {
    m_state[0] = 0x61707865;
    m_state[1] = 0x3320646e;
    m_state[2] = 0x79622d32;
    m_state[3] = 0x6b206574;

    for (auto i = std::size_t{0}; i < KEY_WORDCOUNT; ++i) {
      m_state[4 + i] = key[i];
    }

    m_state[12] = static_cast<matrix_word>(counter & 0xFFFFFFFF);
    m_state[13] = static_cast<matrix_word>(counter >> 32);
    m_state[14] = static_cast<matrix_word>(nonce & 0xFFFFFFFF);
    m_state[15] = static_cast<matrix_word>(nonce >> 32);
  }

  PRNG_ALWAYS_INLINE matrix_type getState(bool prev) const noexcept {
    matrix_type state = m_state;
    if (m_cache_index < CACHE_BLOCKCOUNT || prev) {
      input_word counter = (static_cast<input_word>(state[13]) << 32) | static_cast<input_word>(state[12]);
      counter -= static_cast<input_word>(CACHE_BLOCKCOUNT - m_cache_index);
      if (prev) {
        --counter;
      }
      state[12] = static_cast<matrix_word>(counter & 0xFFFFFFFF);
      state[13] = static_cast<matrix_word>(counter >> 32);
    }
    return state;
  }

  PRNG_ALWAYS_INLINE matrix_type next_block() noexcept {
    if (m_cache_index >= CACHE_BLOCKCOUNT) [[unlikely]] {
      gen_next_blocks_in_cache();
      m_cache_index = 0;
    }

    const auto cache_batch = m_cache_index >> SIMD_WIDTH_SHIFT;
    const auto lane = m_cache_index & SIMD_WIDTH_MASK;
    ++m_cache_index;
    return std::bit_cast<matrix_type>(m_cache[cache_batch][lane]);
  }

private:
  static inline constexpr std::array<matrix_word, SIMD_WIDTH> LANE_OFFSETS = [] {
    std::array<matrix_word, SIMD_WIDTH> offsets{};
    for (auto i = std::size_t{0}; i < SIMD_WIDTH; ++i) {
      offsets[i] = static_cast<matrix_word>(i);
    }
    return offsets;
  }();

  PRNG_ALWAYS_INLINE static simd_type make_higher_counter_inc(matrix_word overflow_index) noexcept {
    if (overflow_index >= SIMD_WIDTH) [[likely]] {
      return simd_type::broadcast(0);
    }

    std::array<matrix_word, SIMD_WIDTH> incs{};
    for (auto i = std::size_t{1}; i < SIMD_WIDTH; ++i) {
      incs[i] = static_cast<matrix_word>(overflow_index < i);
    }
    return simd_type::load_unaligned(incs.data());
  }

  PRNG_ALWAYS_INLINE static void init_state_batches(working_state_type &x, const matrix_type &state,
                                                     simd_type lower_counter_inc,
                                                     simd_type higher_counter_inc) noexcept {
    for (auto i = std::size_t{0}; i < MATRIX_WORDCOUNT; ++i) {
      x[i] = simd_type::broadcast(state[i]);
    }
    x[12] += lower_counter_inc;
    x[13] += higher_counter_inc;
  }

  PRNG_ALWAYS_INLINE static void add_original_state(working_state_type &x, const matrix_type &state,
                                                     simd_type lower_counter_inc,
                                                     simd_type higher_counter_inc) noexcept {
    for (auto i = std::size_t{0}; i < MATRIX_WORDCOUNT; ++i) {
      x[i] += simd_type::broadcast(state[i]);
    }
    x[12] += lower_counter_inc;
    x[13] += higher_counter_inc;
  }

  static void transpose_into_cache(cache_batch_type &cache, working_state_type &x) noexcept {
    auto *PRNG_RESTRICT cache_lanes = cache.data();
    auto *PRNG_RESTRICT working = x.data();
    for (auto segment = std::size_t{0}; segment < BLOCK_SEGMENTCOUNT; ++segment) {
      auto *PRNG_RESTRICT segment_begin = working + segment * SIMD_WIDTH;
      xsimd::transpose(segment_begin, segment_begin + SIMD_WIDTH);
      for (auto lane = std::size_t{0}; lane < SIMD_WIDTH; ++lane) {
        cache_lanes[lane][segment] = segment_begin[lane];
      }
    }
  }

  PRNG_ALWAYS_INLINE static constexpr void advance_counter(matrix_type &state) noexcept {
    state[12] += SIMD_WIDTH;
    state[13] += state[12] < SIMD_WIDTH;
  }

  PRNG_ALWAYS_INLINE static void gen_block_batch(cache_batch_type &cache, const matrix_type &state) noexcept {
    const simd_type lower_counter_inc = simd_type::load_unaligned(LANE_OFFSETS.data());
    matrix_word overflow_index = std::numeric_limits<matrix_word>::max() - state[12];
    const simd_type higher_counter_inc = make_higher_counter_inc(overflow_index);

    working_state_type x;
    init_state_batches(x, state, lower_counter_inc, higher_counter_inc);

    for (auto i = 0; i < R; i += 2) {
      x[0] += x[4];
      x[1] += x[5];
      x[2] += x[6];
      x[3] += x[7];

      x[12] ^= x[0];
      x[13] ^= x[1];
      x[14] ^= x[2];
      x[15] ^= x[3];

      x[12] = xsimd::rotl<16>(x[12]);
      x[13] = xsimd::rotl<16>(x[13]);
      x[14] = xsimd::rotl<16>(x[14]);
      x[15] = xsimd::rotl<16>(x[15]);

      x[8] += x[12];
      x[9] += x[13];
      x[10] += x[14];
      x[11] += x[15];

      x[4] ^= x[8];
      x[5] ^= x[9];
      x[6] ^= x[10];
      x[7] ^= x[11];

      x[4] = xsimd::rotl<12>(x[4]);
      x[5] = xsimd::rotl<12>(x[5]);
      x[6] = xsimd::rotl<12>(x[6]);
      x[7] = xsimd::rotl<12>(x[7]);

      x[0] += x[4];
      x[1] += x[5];
      x[2] += x[6];
      x[3] += x[7];

      x[12] ^= x[0];
      x[13] ^= x[1];
      x[14] ^= x[2];
      x[15] ^= x[3];

      x[12] = xsimd::rotl<8>(x[12]);
      x[13] = xsimd::rotl<8>(x[13]);
      x[14] = xsimd::rotl<8>(x[14]);
      x[15] = xsimd::rotl<8>(x[15]);

      x[8] += x[12];
      x[9] += x[13];
      x[10] += x[14];
      x[11] += x[15];

      x[4] ^= x[8];
      x[5] ^= x[9];
      x[6] ^= x[10];
      x[7] ^= x[11];

      x[4] = xsimd::rotl<7>(x[4]);
      x[5] = xsimd::rotl<7>(x[5]);
      x[6] = xsimd::rotl<7>(x[6]);
      x[7] = xsimd::rotl<7>(x[7]);

      x[0] += x[5];
      x[1] += x[6];
      x[2] += x[7];
      x[3] += x[4];

      x[15] ^= x[0];
      x[12] ^= x[1];
      x[13] ^= x[2];
      x[14] ^= x[3];

      x[15] = xsimd::rotl<16>(x[15]);
      x[12] = xsimd::rotl<16>(x[12]);
      x[13] = xsimd::rotl<16>(x[13]);
      x[14] = xsimd::rotl<16>(x[14]);

      x[10] += x[15];
      x[11] += x[12];
      x[8] += x[13];
      x[9] += x[14];

      x[5] ^= x[10];
      x[6] ^= x[11];
      x[7] ^= x[8];
      x[4] ^= x[9];

      x[5] = xsimd::rotl<12>(x[5]);
      x[6] = xsimd::rotl<12>(x[6]);
      x[7] = xsimd::rotl<12>(x[7]);
      x[4] = xsimd::rotl<12>(x[4]);

      x[0] += x[5];
      x[1] += x[6];
      x[2] += x[7];
      x[3] += x[4];

      x[15] ^= x[0];
      x[12] ^= x[1];
      x[13] ^= x[2];
      x[14] ^= x[3];

      x[15] = xsimd::rotl<8>(x[15]);
      x[12] = xsimd::rotl<8>(x[12]);
      x[13] = xsimd::rotl<8>(x[13]);
      x[14] = xsimd::rotl<8>(x[14]);

      x[10] += x[15];
      x[11] += x[12];
      x[8] += x[13];
      x[9] += x[14];

      x[5] ^= x[10];
      x[6] ^= x[11];
      x[7] ^= x[8];
      x[4] ^= x[9];

      x[5] = xsimd::rotl<7>(x[5]);
      x[6] = xsimd::rotl<7>(x[6]);
      x[7] = xsimd::rotl<7>(x[7]);
      x[4] = xsimd::rotl<7>(x[4]);
    }

    add_original_state(x, state, lower_counter_inc, higher_counter_inc);
    transpose_into_cache(cache, x);
  }

  PRNG_ALWAYS_INLINE constexpr void gen_next_blocks_in_cache() noexcept {
    auto state = m_state;
    for (auto batch = std::uint8_t{0}; batch < CACHE_BATCHCOUNT; ++batch) {
      gen_block_batch(m_cache[batch], state);
      advance_counter(state);
    }
    m_state = state;
  }
};

/**
 * Result from the runtime dispatch initialization for ChaChaSIMD.
 */
struct ChaChaSIMDInitResult {
  using matrix_type = std::array<std::uint32_t, 16>;
  using next_block_fn = matrix_type (*)(void *) noexcept;
  using get_state_fn = matrix_type (*)(const void *, bool) noexcept;
  next_block_fn next_block;
  get_state_fn get_state;
  std::size_t simd_size;
};

/**
 * Functor used by xsimd::dispatch to initialize a ChaChaSIMD instance.
 */
template <std::uint8_t R> struct ChaChaSIMDInitFunctor {
  void *state_storage;
  const std::array<std::uint32_t, 8> key;
  const std::uint64_t counter, nonce;

  template <class Arch> ChaChaSIMDInitResult operator()(Arch) const noexcept;
};

template <std::uint8_t R>
template <class Arch>
ChaChaSIMDInitResult ChaChaSIMDInitFunctor<R>::operator()(Arch) const noexcept {
  using State = ChaChaState<Arch, R>;
  static_assert(sizeof(State) <= 2176, "ChaChaState exceeds StateStorage capacity");
  static_assert(alignof(State) <= 64, "ChaChaState exceeds StateStorage alignment");
  new (state_storage) State(key, counter, nonce);
  return {
      +[](void *s) noexcept -> ChaChaSIMDInitResult::matrix_type {
        return static_cast<State *>(s)->next_block();
      },
      +[](const void *s, bool prev) noexcept -> ChaChaSIMDInitResult::matrix_type {
        return static_cast<const State *>(s)->getState(prev);
      },
      std::size_t{State::SIMD_WIDTH},
  };
}

extern template PRNG_EXPORT ChaChaSIMDInitResult
ChaChaSIMDInitFunctor<20>::operator()<xsimd::sse2>(xsimd::sse2) const noexcept;
extern template PRNG_EXPORT ChaChaSIMDInitResult
ChaChaSIMDInitFunctor<20>::operator()<xsimd::avx2>(xsimd::avx2) const noexcept;
extern template PRNG_EXPORT ChaChaSIMDInitResult
ChaChaSIMDInitFunctor<20>::operator()<xsimd::avx512f>(xsimd::avx512f) const noexcept;

} // namespace internal

/**
 * ChaChaSIMD: runtime SIMD dispatch via inline union + function pointers.
 * No heap allocation, no virtual dispatch.
 *
 * @tparam R Number of ChaCha rounds.
 */
template <std::uint8_t R = 20> class ChaChaSIMD {
public:
  static constexpr auto MATRIX_WORDCOUNT = std::uint8_t{16};
  static constexpr auto KEY_WORDCOUNT = std::uint8_t{8};

  using result_type = std::uint64_t;
  using input_word = std::uint64_t;
  using matrix_word = std::uint32_t;
  using matrix_type = std::array<matrix_word, MATRIX_WORDCOUNT>;
  using result_cache_type = std::array<result_type, MATRIX_WORDCOUNT / 2>;

  static constexpr PRNG_ALWAYS_INLINE auto(min)() noexcept { return (std::numeric_limits<result_type>::min)(); }
  static constexpr PRNG_ALWAYS_INLINE auto(max)() noexcept { return (std::numeric_limits<result_type>::max)(); }

  static constexpr PRNG_ALWAYS_INLINE matrix_type results_to_block(const result_cache_type &results) noexcept {
    return std::bit_cast<matrix_type>(results);
  }

  static constexpr PRNG_ALWAYS_INLINE result_cache_type block_to_results(const matrix_type &block) noexcept {
    return std::bit_cast<result_cache_type>(block);
  }

  explicit PRNG_ALWAYS_INLINE ChaChaSIMD(const std::array<matrix_word, KEY_WORDCOUNT> key, const input_word counter,
                                          const input_word nonce) {
    auto result =
        xsimd::dispatch<xsimd::arch_list<xsimd::avx512f, xsimd::avx2, xsimd::sse2>>(
            internal::ChaChaSIMDInitFunctor<R>{m_state.data, key, counter, nonce})();
    m_next_block = result.next_block;
    m_get_state = result.get_state;
    m_simd_size = result.simd_size;
  }

  PRNG_ALWAYS_INLINE constexpr result_type operator()() noexcept {
    if (m_result_index >= m_result_cache.size()) [[unlikely]] {
      m_result_cache = block_to_results(m_next_block(m_state.data));
      m_result_index = 0;
    }
    return m_result_cache[m_result_index++];
  }

  PRNG_ALWAYS_INLINE constexpr double uniform() noexcept {
    return static_cast<double>(operator()() >> 11) * 0x1.0p-53;
  }

  PRNG_ALWAYS_INLINE constexpr matrix_type block() noexcept {
    if (m_result_index < m_result_cache.size()) {
      auto cached_block = results_to_block(m_result_cache);
      m_result_index = static_cast<std::uint8_t>(m_result_cache.size());
      return cached_block;
    }
    return m_next_block(m_state.data);
  }

  PRNG_ALWAYS_INLINE constexpr matrix_type getState() const noexcept {
    return m_get_state(m_state.data, m_result_index < m_result_cache.size());
  }

  PRNG_ALWAYS_INLINE size_t getSIMDSize() const noexcept { return m_simd_size; }

private:
  using next_block_fn = internal::ChaChaSIMDInitResult::next_block_fn;
  using get_state_fn = internal::ChaChaSIMDInitResult::get_state_fn;

  // Raw byte storage for the arch-specific ChaChaState.
  // Typed union is not viable: xsimd batch types have different sizeof
  // across TUs compiled with different -march flags (ODR divergence).
  // Max is avx512f: 64B state + 2048B cache + 1B index + padding = 2176 bytes.
  struct StateStorage {
    static constexpr std::size_t SIZE = 2176;
    static constexpr std::size_t ALIGN = 64;
    alignas(ALIGN) unsigned char data[SIZE];
  };

  alignas(64) StateStorage m_state;
  next_block_fn m_next_block = nullptr;
  get_state_fn m_get_state = nullptr;
  std::size_t m_simd_size = 0;
  result_cache_type m_result_cache{};
  std::uint8_t m_result_index = static_cast<std::uint8_t>(m_result_cache.size());
};

/**
 * ChaChaNative: uses the best architecture available at compile time.
 * Zero indirection — direct calls to ChaChaState methods.
 *
 * @tparam R Number of ChaCha rounds.
 */
template <std::uint8_t R = 20> class ChaChaNative {
public:
  static constexpr auto MATRIX_WORDCOUNT = std::uint8_t{16};
  static constexpr auto KEY_WORDCOUNT = std::uint8_t{8};

  using result_type = std::uint64_t;
  using input_word = std::uint64_t;
  using matrix_word = std::uint32_t;
  using matrix_type = std::array<matrix_word, MATRIX_WORDCOUNT>;
  using result_cache_type = std::array<result_type, MATRIX_WORDCOUNT / 2>;

  static constexpr PRNG_ALWAYS_INLINE auto(min)() noexcept { return (std::numeric_limits<result_type>::min)(); }
  static constexpr PRNG_ALWAYS_INLINE auto(max)() noexcept { return (std::numeric_limits<result_type>::max)(); }

  ChaChaNative(const std::array<matrix_word, KEY_WORDCOUNT> key, const input_word counter, const input_word nonce)
      : m_state(key, counter, nonce) {}

  PRNG_ALWAYS_INLINE constexpr result_type operator()() noexcept {
    if (m_result_index >= m_result_cache.size()) [[unlikely]] {
      m_result_cache = ChaChaSIMD<R>::block_to_results(m_state.next_block());
      m_result_index = 0;
    }
    return m_result_cache[m_result_index++];
  }

  PRNG_ALWAYS_INLINE constexpr double uniform() noexcept {
    return static_cast<double>(operator()() >> 11) * 0x1.0p-53;
  }

  PRNG_ALWAYS_INLINE constexpr matrix_type block() noexcept {
    if (m_result_index < m_result_cache.size()) {
      auto cached_block = ChaChaSIMD<R>::results_to_block(m_result_cache);
      m_result_index = static_cast<std::uint8_t>(m_result_cache.size());
      return cached_block;
    }
    return m_state.next_block();
  }

  PRNG_ALWAYS_INLINE constexpr matrix_type getState() const noexcept {
    return m_state.getState(m_result_index < m_result_cache.size());
  }

  PRNG_ALWAYS_INLINE size_t getSIMDSize() const noexcept { return std::size_t{State::SIMD_WIDTH}; }

private:
  using State = internal::ChaChaState<xsimd::best_arch, R>;
  State m_state;
  result_cache_type m_result_cache{};
  std::uint8_t m_result_index = static_cast<std::uint8_t>(m_result_cache.size());
};

} // namespace prng
