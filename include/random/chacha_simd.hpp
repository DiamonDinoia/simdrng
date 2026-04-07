#pragma once

#include <array>
#if __cplusplus >= 202002L
#include <bit>
#endif
#include <cstdint>
#include <limits>
#include <type_traits>
#include <xsimd/xsimd.hpp>

#include "random/macros.hpp"

namespace prng {

// Forward declare ChaChaSIMDCreator for dynamic dispatch help
namespace internal {
  template <std::uint8_t R>
  struct ChaChaSIMDCreator;
}

template <std::uint8_t R = 20>
class ChaChaSIMD {
public:
  static constexpr auto MATRIX_WORDCOUNT = std::uint8_t{16};
  static constexpr auto KEY_WORDCOUNT = std::uint8_t{8};

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

  static constexpr PRNG_ALWAYS_INLINE matrix_type results_to_block(const result_cache_type& results) noexcept {
#if __cplusplus >= 202002L
    return std::bit_cast<matrix_type>(results);
#else
    matrix_type block{};
    for (auto i = std::size_t{0}; i < results.size(); ++i) {
      block[2 * i] = static_cast<matrix_word>(results[i] & 0xFFFFFFFF);
      block[2 * i + 1] = static_cast<matrix_word>(results[i] >> 32);
    }
    return block;
#endif
  }

  static constexpr PRNG_ALWAYS_INLINE result_cache_type block_to_results(const matrix_type& block) noexcept {
#if __cplusplus >= 202002L
    return std::bit_cast<result_cache_type>(block);
#else
    result_cache_type results{};
    for (auto i = std::size_t{0}; i < results.size(); ++i) {
      results[i] =
        static_cast<result_type>(block[2 * i]) |
        (static_cast<result_type>(block[2 * i + 1]) << 32);
    }
    return results;
#endif
  }

  /**
   * @brief Construct a SIMD ChaCha generator with given key, counter and nonce
   * @param key A 256-bit key, divided up into eight 32-bit words.
   * @param counter Initial value of the counter.
   * @param nonce Initial value of the nonce.
   */
  explicit PRNG_ALWAYS_INLINE ChaChaSIMD(
    const std::array<matrix_word, KEY_WORDCOUNT> key,
    const input_word counter,
    const input_word nonce
  ) {
    // TODO: Make sure you properly initalize the implementation's state with these!
    pImpl = xsimd::dispatch<xsimd::arch_list<xsimd::default_arch>>(internal::ChaChaSIMDCreator<R>{key, counter, nonce})();
  }

  /**
   * @brief Generates the next 64-bit output.
   * @return The next 64-bit output.
   */
  PRNG_ALWAYS_INLINE constexpr result_type operator()() noexcept {
    if (m_result_index >= m_result_cache.size()) [[unlikely]] {
      m_result_cache = block_to_results(pImpl->next_block());
      m_result_index = 0;
    }
    return m_result_cache[m_result_index++];
  }
  
  /**
   * @brief Generates a uniform random number in the range [0, 1).
   * @return A uniform random number.
   */
  PRNG_ALWAYS_INLINE constexpr double uniform() noexcept {
    return static_cast<double>(operator()() >> 11) * 0x1.0p-53;
  }

  // TODO: block()
  // if we're not at the end of the result cache, should return the whole result cache
  // as matrix_type and then set the result cache index past the end to invalidate it.
  // Otherwise it should return the next block. Means in the implementation we can return
  // just the next block, which leaves just a 'return next_block' call, so we can cut it in the implementation
  /**
   * @brief Generates the next 64-byte ChaCha block.
   * @return The next 64-byte ChaCha block.
   */
  PRNG_ALWAYS_INLINE constexpr matrix_type block() noexcept {
    if (m_result_index < m_result_cache.size()) {
      auto cached_block = results_to_block(m_result_cache);
      m_result_index = static_cast<std::uint8_t>(m_result_cache.size()); // Mark result cache as exhausted
      return cached_block;
    }
    return pImpl->next_block();
  }

  // TODO: getState() will likely need direct access to impl for this
  // /**
  //  * @brief Returns the state of the generator; a 4x4 matrix.
  //  * @return State of the generator.
  //  */
  // PRNG_ALWAYS_INLINE constexpr matrix_type getState() const noexcept {
  //   matrix_type state = m_state;
  //   if (m_cache_index < CACHE_BLOCKCOUNT || m_result_index < m_result_cache.size()) {
  //     input_word counter = (static_cast<input_word>(state[13]) << 32) | static_cast<input_word>(state[12]);
  //     counter -= static_cast<input_word>(CACHE_BLOCKCOUNT - m_cache_index);
  //     if (m_result_index < m_result_cache.size()) {
  //       --counter;
  //     }
  //     state[12] = static_cast<matrix_word>(counter & 0xFFFFFFFF);
  //     state[13] = static_cast<matrix_word>(counter >> 32);
  //   }
  //   return state;
  // }

  struct IChaChaSIMD {
    virtual ~IChaChaSIMD() = default;
    virtual matrix_type next_block() = 0;
  };

private:
  std::unique_ptr<IChaChaSIMD> pImpl;

  result_cache_type m_result_cache {};
  // Initialize to "past end of the cache" since cache starts empty.
  std::uint8_t m_result_index = static_cast<std::uint8_t>(m_result_cache.size());
};


namespace internal {

template <class Arch, std::uint8_t R>
class ChaChaSIMDImpl : public ChaChaSIMD<>::IChaChaSIMD {
protected:
  static constexpr auto MATRIX_WORDCOUNT = std::uint8_t{16};
  static constexpr auto KEY_WORDCOUNT = std::uint8_t{8};

public:
  using input_word = ChaChaSIMD<>::input_word;
  using matrix_word = ChaChaSIMD<>::matrix_word;
  using matrix_type = ChaChaSIMD<>::matrix_type;
  using simd_type = xsimd::batch<matrix_word, Arch>;
  using working_state_type = std::array<simd_type, MATRIX_WORDCOUNT>;

protected:
  static constexpr std::uint8_t SIMD_WIDTH = std::uint8_t{simd_type::size};
#if __cplusplus >= 202002L
  static_assert(std::has_single_bit(static_cast<unsigned int>(SIMD_WIDTH)),
                "ChaCha SIMD width must be a power of two");
  static constexpr std::uint8_t SIMD_WIDTH_SHIFT =
    static_cast<std::uint8_t>(std::countr_zero(static_cast<unsigned int>(SIMD_WIDTH)));
#else
  static constexpr bool is_power_of_two(std::uint8_t value) noexcept {
    return value != 0 && (value & (value - 1)) == 0;
  }

  static constexpr std::uint8_t bit_shift(std::uint8_t value) noexcept {
    std::uint8_t shift = 0;
    while (value > 1) {
      value >>= 1;
      ++shift;
    }
    return shift;
  }

  static_assert(is_power_of_two(SIMD_WIDTH), "ChaCha SIMD width must be a power of two");
  static constexpr std::uint8_t SIMD_WIDTH_SHIFT = bit_shift(SIMD_WIDTH);
#endif
  static_assert(MATRIX_WORDCOUNT % SIMD_WIDTH == 0, "ChaCha state must divide evenly into SIMD segments");
  static constexpr std::uint8_t SIMD_WIDTH_MASK = std::uint8_t{SIMD_WIDTH - 1};
  static constexpr std::uint8_t BLOCK_SEGMENTCOUNT = std::uint8_t{MATRIX_WORDCOUNT / SIMD_WIDTH};
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
#if __cplusplus >= 202002L
  static_assert(sizeof(cache_block_type) == sizeof(matrix_type),
                "Cache blocks must have the same layout size as a ChaCha block");
  static_assert(std::is_trivially_copyable_v<cache_block_type>,
                "Cache blocks must be trivially copyable for bit_cast");
  static_assert(std::is_trivially_copyable_v<matrix_type>,
                "ChaCha blocks must be trivially copyable for bit_cast");
#endif

public:
  /**
   * @brief Construct a SIMD ChaCha generator with given key, counter and nonce
   * @param key A 256-bit key, divided up into eight 32-bit words.
   * @param counter Initial value of the counter.
   * @param nonce Initial value of the nonce.
   */
  explicit PRNG_ALWAYS_INLINE ChaChaSIMDImpl(
    const std::array<matrix_word, KEY_WORDCOUNT> key,
    const input_word counter,
    const input_word nonce
  ) {
    // First four words (i.e. top-row) are always the same constants
    // They spell out "expand 2-byte k" in ASCII (little-endian)
    m_state[0] = 0x61707865;
    m_state[1] = 0x3320646e;
    m_state[2] = 0x79622d32;
    m_state[3] = 0x6b206574;

    for (auto i = std::size_t{0}; i < KEY_WORDCOUNT; ++i) {
      m_state[4 + i] = key[i];
    }

    // ChaCha assumes little-endianness
    m_state[12] = static_cast<matrix_word>(counter & 0xFFFFFFFF);
    m_state[13] = static_cast<matrix_word>(counter >> 32);
    m_state[14] = static_cast<matrix_word>(nonce & 0xFFFFFFFF);
    m_state[15] = static_cast<matrix_word>(nonce >> 32);
  }
  // /**
  //  * @brief Returns the state of the generator; a 4x4 matrix.
  //  * @return State of the generator.
  //  */
  // PRNG_ALWAYS_INLINE constexpr matrix_type getState() const noexcept {
  //   matrix_type state = m_state;
  //   if (m_cache_index < CACHE_BLOCKCOUNT || m_result_index < m_result_cache.size()) {
  //     input_word counter = (static_cast<input_word>(state[13]) << 32) | static_cast<input_word>(state[12]);
  //     counter -= static_cast<input_word>(CACHE_BLOCKCOUNT - m_cache_index);
  //     if (m_result_index < m_result_cache.size()) {
  //       --counter;
  //     }
  //     state[12] = static_cast<matrix_word>(counter & 0xFFFFFFFF);
  //     state[13] = static_cast<matrix_word>(counter >> 32);
  //   }
  //   return state;
  // }


private:
  matrix_type m_state;
  alignas(simd_type::arch_type::alignment()) std::array<cache_batch_type, CACHE_BATCHCOUNT> m_cache;
  // Initialize to "past end of the cache" since cache starts empty.
  std::uint8_t m_cache_index = CACHE_BLOCKCOUNT;

  static inline constexpr std::array<matrix_word, SIMD_WIDTH> LANE_OFFSETS = [] {
    std::array<matrix_word, SIMD_WIDTH> offsets{};
    for (auto i = std::size_t{0}; i < SIMD_WIDTH; ++i) {
      offsets[i] = static_cast<matrix_word>(i);
    }
    return offsets;
  }();

  /**
   * Return an array { 0, n < 1, n < 2, ..., n < (SIMD_WIDTH - 1) }. Can be used to initialize a
   * batch for incremetinng elements in a batch of consecutive high counter words, depending on at
   * what index the corresponding lower counter words had an overflow.
   */
  PRNG_ALWAYS_INLINE static simd_type make_higher_counter_inc(matrix_word overflow_index) noexcept {
    if (overflow_index >= SIMD_WIDTH) [[likely]] {
      return simd_type::broadcast(0);
    }

    std::array<matrix_word, SIMD_WIDTH> incs{};
    for (auto i = std::size_t{1}; i < SIMD_WIDTH; ++i) {
      incs[i] = static_cast<matrix_word>(overflow_index < i);
    }
    return xsimd::load_unaligned(incs.data());
  }


  PRNG_ALWAYS_INLINE static void init_state_batches(working_state_type& x, const matrix_type& state,
                                                    simd_type lower_counter_inc, simd_type higher_counter_inc) noexcept {
    for (auto i = std::size_t{0}; i < MATRIX_WORDCOUNT; ++i) {
      x[i] = simd_type::broadcast(state[i]);
    }
    x[12] += lower_counter_inc;
    x[13] += higher_counter_inc;
  }

  PRNG_ALWAYS_INLINE static void add_original_state(working_state_type& x, const matrix_type& state,
                                                    simd_type lower_counter_inc, simd_type higher_counter_inc) noexcept {
    for (auto i = std::size_t{0}; i < MATRIX_WORDCOUNT; ++i) {
      x[i] += simd_type::broadcast(state[i]);
    }
    x[12] += lower_counter_inc;
    x[13] += higher_counter_inc;
  }

  static void transpose_into_cache(cache_batch_type& cache, working_state_type& x) noexcept {
    auto* PRNG_RESTRICT cache_lanes = cache.data();
    auto* PRNG_RESTRICT working = x.data();
    for (auto segment = std::size_t{0}; segment < BLOCK_SEGMENTCOUNT; ++segment) {
      auto* PRNG_RESTRICT segment_begin = working + segment * SIMD_WIDTH;
      xsimd::transpose(segment_begin, segment_begin + SIMD_WIDTH);
      for (auto lane = std::size_t{0}; lane < SIMD_WIDTH; ++lane) {
        cache_lanes[lane][segment] = segment_begin[lane];
      }
    }
  }

  /**
   * Advances the 64-bit ChaCha block counter by a vector-width worth of blocks.
   */
  PRNG_ALWAYS_INLINE static constexpr void advance_counter(matrix_type& state) noexcept {
    state[12] += SIMD_WIDTH;
    state[13] += state[12] < SIMD_WIDTH;
  }

  /**
   * Generates `SIMD_WIDTH` new ChaCha blocks into one cache batch.
   */
  PRNG_ALWAYS_INLINE static void gen_block_batch(cache_batch_type& cache, const matrix_type& state) noexcept {
    const auto lower_counter_inc = xsimd::load_unaligned(LANE_OFFSETS.data());
    matrix_word overflow_index = std::numeric_limits<matrix_word>::max() - state[12];
    const auto higher_counter_inc = make_higher_counter_inc(overflow_index);

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

  PRNG_ALWAYS_INLINE matrix_type next_block() noexcept override {
    if (m_cache_index >= CACHE_BLOCKCOUNT) [[unlikely]] {
      gen_next_blocks_in_cache();
      m_cache_index = 0;
    }

    const auto cache_batch = m_cache_index >> SIMD_WIDTH_SHIFT;
    const auto lane = m_cache_index & SIMD_WIDTH_MASK;
    ++m_cache_index;
#if __cplusplus >= 202002L
    return std::bit_cast<matrix_type>(m_cache[cache_batch][lane]);
#else
    matrix_type block;
    const auto& cached_block = m_cache[cache_batch][lane];
    const auto* PRNG_RESTRICT cached_segments = cached_block.data();
    auto* PRNG_RESTRICT out = block.data();
    for (auto segment = std::size_t{0}; segment < BLOCK_SEGMENTCOUNT; ++segment) {
      cached_segments[segment].store_unaligned(out + segment * SIMD_WIDTH);
    }
    return block;
#endif
  }

  /**
   * Generates one or two SIMD batches, depending on the target ISA, and writes them into the cache.
   * Also advances the state's counter words by the amount of ChaCha blocks generated.
   */
  PRNG_ALWAYS_INLINE constexpr void gen_next_blocks_in_cache() noexcept {
    auto state = m_state;
    for (auto batch = std::uint8_t{0}; batch < CACHE_BATCHCOUNT; ++batch) {
      gen_block_batch(m_cache[batch], state);
      advance_counter(state);
    }
    m_state = state;
  }
};

template <std::uint8_t R>
struct ChaChaSIMDCreator {
  const typename std::array<typename ChaChaSIMD<R>::matrix_word, ChaChaSIMD<R>::KEY_WORDCOUNT> key;
  const typename ChaChaSIMD<R>::input_word counter, nonce;
  template <class Arch>
  std::unique_ptr<typename ChaChaSIMD<R>::IChaChaSIMD> operator()(Arch) const;
};

template <std::uint8_t R>
template <class Arch>
std::unique_ptr<typename ChaChaSIMD<R>::IChaChaSIMD> ChaChaSIMDCreator<R>::operator()(Arch) const {
  return std::make_unique<ChaChaSIMDImpl<Arch, R>>(key, counter, nonce);
};

} // namespace internal

} // namespace prng
