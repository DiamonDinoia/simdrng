/* Auto-generated single-header for simdrng.
 * Do not edit directly.
 *
 * poet is inlined; xsimd is kept as an external <xsimd/...> include, so
 * compile with the xsimd headers available (installed, or the Compiler
 * Explorer xsimd library). The *Native generators (XoshiroNative,
 * Philox4x64Native, ...) work header-only. The runtime-dispatch types
 * (XoshiroSIMD/Philox*SIMD and the default simdrng::Xoshiro alias) need the
 * compiled libsimdrng.a and are not available from this header alone.
 */

#ifndef SIMDRNG_SINGLE_HEADER_HPP
#define SIMDRNG_SINGLE_HEADER_HPP

// BEGIN_FILE: simdrng.hpp

// Umbrella header for the simdrng public API. This is also the entry point the
// single-header amalgamation (tools/amalgamate.py) inlines from.

/* Begin inline (quoted): version.hpp */
// BEGIN_FILE: version.hpp

// Auto-generated from version.hpp.in by CMake (GenerateVersion.cmake).
// Do not edit by hand; edit the top-level VERSION file instead.

#define SIMDRNG_VERSION_MAJOR 0
#define SIMDRNG_VERSION_MINOR 0
#define SIMDRNG_VERSION_PATCH 2

// Full version string, including a `-dev.N+g<sha>` suffix on untagged builds.
#define SIMDRNG_VERSION_STRING "0.0.2-dev.151+g2b3a9aa"

// Numeric encoding (MAJOR * 10000 + MINOR * 100 + PATCH) for comparisons.
#define SIMDRNG_VERSION \
  (SIMDRNG_VERSION_MAJOR * 10000 + SIMDRNG_VERSION_MINOR * 100 + SIMDRNG_VERSION_PATCH)
// END_FILE: version.hpp
/* End inline (quoted): version.hpp */

/* Begin inline (quoted): chacha.hpp */
// BEGIN_FILE: chacha.hpp

#include <array>
#include <bit>
#include <cstdint>
#include <limits>

/* Begin inline (quoted): macros.hpp */
// BEGIN_FILE: macros.hpp

#if defined(_MSC_VER)
#define SIMDRNG_ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define SIMDRNG_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define SIMDRNG_ALWAYS_INLINE inline
#endif

// Lambda call-operators can't take the `inline` keyword, so force-inlining a
// lambda body (e.g. a poet::static_for callable) needs the bare attribute. GCC
// otherwise outlines and clones heavy generic-lambda bodies per instantiation.
#if defined(_MSC_VER)
#define SIMDRNG_ALWAYS_INLINE_LAMBDA [[msvc::forceinline]]
#elif defined(__GNUC__) || defined(__clang__)
#define SIMDRNG_ALWAYS_INLINE_LAMBDA __attribute__((always_inline))
#else
#define SIMDRNG_ALWAYS_INLINE_LAMBDA
#endif

#if defined(__GNUC__) || defined(__clang__)
#define SIMDRNG_FLATTEN __attribute__((flatten))
#else
#define SIMDRNG_FLATTEN
#endif

#if defined(_MSC_VER)
#define SIMDRNG_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define SIMDRNG_RESTRICT __restrict__
#else
#define SIMDRNG_RESTRICT
#endif

#if defined(_MSC_VER)
#define SIMDRNG_NEVER_INLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define SIMDRNG_NEVER_INLINE __attribute__((cold, noinline))
#else
#define SIMDRNG_NEVER_INLINE
#endif

// simdrng ships as a STATIC library (and header-only public API). The per-arch
// dispatch-functor instantiations and the out-of-line XoshiroSIMD ctor need
// external linkage so the explicit instantiations resolve at static-link time,
// but they are implementation details and must NOT be re-exported from a shared
// object that links the static lib (e.g. the Python extension, which should
// export only its module init). Mark them hidden; MSVC static libs don't export
// without dllexport, so nothing is needed there.
#if defined(__GNUC__) || defined(__clang__)
#define SIMDRNG_LOCAL __attribute__((visibility("hidden")))
#else
#define SIMDRNG_LOCAL
#endif

// Runtime SIMD dispatch. When 1, the *SIMD types dispatch over the compiled
// per-arch tiers (portable, set by the CMake SIMDRNG_DISPATCH option, which
// propagates this via a PUBLIC compile definition). Default 0: no tiers are
// built and the default aliases resolve to the native (xsimd::best_arch) types.
#ifndef SIMDRNG_DISPATCH
#define SIMDRNG_DISPATCH 0
#endif

// Architecture detection
#if defined(__x86_64__) || defined(_M_X64)
#define SIMDRNG_ARCH_X86_64 1
#elif defined(__aarch64__) || defined(_M_ARM64)
#define SIMDRNG_ARCH_AARCH64 1
#elif defined(__riscv) && (__riscv_xlen == 64)
#define SIMDRNG_ARCH_RISCV64 1
#endif
// END_FILE: macros.hpp
/* End inline (quoted): macros.hpp */

namespace simdrng {

/**
 * @class ChaCha
 * @brief ChaCha stream-cipher core repurposed as a counter-based RNG.
 *
 * Each 64-byte block is produced by running @p R rounds of the ChaCha
 * quarter-round over a 4x4 state matrix built from a 256-bit key, a 64-bit
 * counter and a 64-bit nonce, then adding the original matrix. Because the
 * block is a pure function of (key, counter, nonce), streams are reproducible
 * and seekable: bumping the counter selects the next non-overlapping block.
 *
 * @tparam R Number of rounds (default 20, i.e. ChaCha20; always rounded up to
 *           an even number since odd and even rounds are issued in pairs).
 */
template <std::uint8_t R = 20> class ChaCha {

protected:
  static constexpr auto MATRIX_WORDCOUNT = std::uint8_t{16};
  static constexpr auto KEY_WORDCOUNT = std::uint8_t{8};

public:
  using result_type = std::uint64_t;
  using input_word = std::uint64_t;
  using matrix_word = std::uint32_t;
  using matrix_type = std::array<matrix_word, MATRIX_WORDCOUNT>;
  using result_cache_type = std::array<result_type, MATRIX_WORDCOUNT / 2>;

  /**
   * @brief Smallest value operator() can return.
   * @return 0.
   */
  static constexpr SIMDRNG_ALWAYS_INLINE auto(min)() noexcept { return (std::numeric_limits<result_type>::min)(); }

  /**
   * @brief Largest value operator() can return.
   * @return 2^64 - 1.
   */
  static constexpr SIMDRNG_ALWAYS_INLINE auto(max)() noexcept { return (std::numeric_limits<result_type>::max)(); }

  /**
   * @brief Construct a scalar ChaCha generator with given key, counter and nonce
   * @param key A 256-bit key, divided up into eight 32-bit words.
   * @param counter Initial value of the counter.
   * @param nonce Initial value of the nonce.
   */
  SIMDRNG_ALWAYS_INLINE explicit ChaCha(const std::array<matrix_word, KEY_WORDCOUNT> &key, const input_word counter,
                                        const input_word nonce) noexcept {
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
  SIMDRNG_ALWAYS_INLINE constexpr result_type operator()() noexcept { return next_result(); }

  /**
   * @brief Generates a uniform random number in the range [0, 1).
   * @return A uniform random number.
   */
  SIMDRNG_ALWAYS_INLINE constexpr double uniform() noexcept {
    return static_cast<double>(operator()() >> 11) * 0x1.0p-53;
  }

  /**
   * @brief Generates the next 64-byte ChaCha block.
   * @return The next 64-byte ChaCha block.
   */
  SIMDRNG_ALWAYS_INLINE constexpr matrix_type block() noexcept {
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
  SIMDRNG_ALWAYS_INLINE constexpr matrix_type getState() const noexcept {
    matrix_type state = m_state;
    if (m_result_index < m_result_cache.size()) {
      const input_word counter = (static_cast<input_word>(state[13]) << 32) | static_cast<input_word>(state[12]);
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

  static constexpr SIMDRNG_ALWAYS_INLINE auto rotl(const matrix_word x, const int k) noexcept {
    return std::rotl(x, k);
  }

  static constexpr SIMDRNG_ALWAYS_INLINE void quarter_round(matrix_type &m, const unsigned int a, const unsigned int b,
                                                            const unsigned int c, const unsigned int d) noexcept {
    m[a] += m[b];
    m[d] ^= m[a];
    m[d] = rotl(m[d], 16);
    m[c] += m[d];
    m[b] ^= m[c];
    m[b] = rotl(m[b], 12);
    m[a] += m[b];
    m[d] ^= m[a];
    m[d] = rotl(m[d], 8);
    m[c] += m[d];
    m[b] ^= m[c];
    m[b] = rotl(m[b], 7);
  }

  constexpr SIMDRNG_ALWAYS_INLINE void inc_counter() noexcept {
    if (++m_state[12] == 0) {
      ++m_state[13];
    }
  }

  static constexpr SIMDRNG_ALWAYS_INLINE result_cache_type block_to_results(const matrix_type &block) noexcept {
    return std::bit_cast<result_cache_type>(block);
  }

  static constexpr SIMDRNG_ALWAYS_INLINE matrix_type results_to_block(const result_cache_type &results) noexcept {
    return std::bit_cast<matrix_type>(results);
  }

  constexpr SIMDRNG_ALWAYS_INLINE result_type next_result() noexcept {
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
  SIMDRNG_FLATTEN constexpr SIMDRNG_ALWAYS_INLINE matrix_type next_block() noexcept {
    matrix_type x = m_state;

    // Note that we perform both an odd and even round at the same time.
    // As a result the amount of rounds performed is always rounded up to an even number.
    for (auto i = 0; i < R; i += 2) {
      // Odd round
      quarter_round(x, 0, 4, 8, 12);
      quarter_round(x, 1, 5, 9, 13);
      quarter_round(x, 2, 6, 10, 14);
      quarter_round(x, 3, 7, 11, 15);

      // Even round
      quarter_round(x, 0, 5, 10, 15);
      quarter_round(x, 1, 6, 11, 12);
      quarter_round(x, 2, 7, 8, 13);
      quarter_round(x, 3, 4, 9, 14);
    }

    for (auto i = 0; i < MATRIX_WORDCOUNT; ++i) {
      x[i] += m_state[i];
    }

    inc_counter();

    return x;
  }
};

} // namespace simdrng
// END_FILE: chacha.hpp
/* End inline (quoted): chacha.hpp */
/* Begin inline (quoted): macros.hpp */
/* Skipped already inlined: macros.hpp */
/* End inline (quoted): macros.hpp */
/* Begin inline (quoted): philox.hpp */
// BEGIN_FILE: philox.hpp

#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <type_traits>

/* Begin inline (quoted): macros.hpp */
/* Skipped already inlined: macros.hpp */
/* End inline (quoted): macros.hpp */

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
template <std::uint8_t N, std::uint8_t W> struct PhiloxConstants;

template <> struct PhiloxConstants<4, 32> {
  static constexpr std::array<std::uint32_t, 2> MUL = {0xD2511F53, 0xCD9E8D57};
  static constexpr std::array<std::uint32_t, 2> BUMP = {0x9E3779B9, 0xBB67AE85};
  static constexpr std::array<std::uint8_t, 2> PERM = {1, 0};
};

template <> struct PhiloxConstants<2, 32> {
  static constexpr std::array<std::uint32_t, 1> MUL = {0xD256D193};
  static constexpr std::array<std::uint32_t, 1> BUMP = {0x9E3779B9};
  static constexpr std::array<std::uint8_t, 1> PERM = {0};
};

template <> struct PhiloxConstants<4, 64> {
  static constexpr std::array<std::uint64_t, 2> MUL = {0xD2E7470EE14C6C93ULL, 0xCA5A826395121157ULL};
  static constexpr std::array<std::uint64_t, 2> BUMP = {0x9E3779B97F4A7C15ULL, 0xBB67AE8584CAA73BULL};
  static constexpr std::array<std::uint8_t, 2> PERM = {1, 0};
};

template <> struct PhiloxConstants<2, 64> {
  static constexpr std::array<std::uint64_t, 1> MUL = {0xD2B74407B1CE6E93ULL};
  static constexpr std::array<std::uint64_t, 1> BUMP = {0x9E3779B97F4A7C15ULL};
  static constexpr std::array<std::uint8_t, 1> PERM = {0};
};

} // namespace internal

/**
 * @class Philox
 * @brief Philox counter-based RNG (Salmon et al., "Parallel Random Numbers: As
 * Easy as 1, 2, 3", SC11).
 *
 * Philox is a counter-based generator: each output block is a keyed bijection of
 * an incrementing counter, so an arbitrary stream position can be computed
 * directly with no sequential dependency. That makes it a natural fit for
 * parallel / GPU-style workloads where each work item derives its own
 * non-overlapping sub-stream from (seed, counter).
 *
 * @tparam N Number of counter words per block (2 or 4).
 * @tparam W Word width in bits (32 or 64).
 * @tparam R Number of rounds (default 10, the reference strength).
 */
template <std::uint8_t N = 4, std::uint8_t W = 32, std::uint8_t R = 10> class Philox {
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

  /**
   * @brief Smallest value operator() can return.
   * @return 0.
   */
  static constexpr SIMDRNG_ALWAYS_INLINE auto(min)() noexcept { return (std::numeric_limits<result_type>::min)(); }

  /**
   * @brief Largest value operator() can return.
   * @return 2^64 - 1.
   */
  static constexpr SIMDRNG_ALWAYS_INLINE auto(max)() noexcept { return (std::numeric_limits<result_type>::max)(); }

  /**
   * @brief Constructs the generator from a 64-bit seed and starting counter.
   * @param seed Expanded into the Philox key via SplitMix.
   * @param counter Initial counter (stream position); defaults to 0.
   */
  explicit SIMDRNG_ALWAYS_INLINE Philox(result_type seed, result_type counter = 0) noexcept
      : m_counter(counter_from_uint64(counter)), m_key(seed_to_key(seed)) {}

  /**
   * @brief Constructs the generator from an explicit key and counter block.
   * @param key The Philox key (N/2 words).
   * @param counter The full counter block (N words).
   */
  explicit SIMDRNG_ALWAYS_INLINE Philox(const key_type &key, const counter_type &counter) noexcept
      : m_counter(counter), m_key(key) {}

  /**
   * @brief Returns the next 64-bit output, generating a fresh block when the
   * current one is exhausted.
   * @return The next 64-bit output.
   */
  SIMDRNG_ALWAYS_INLINE constexpr result_type operator()() noexcept {
    if (m_result_index >= RESULTS_PER_BLOCK) [[unlikely]] {
      m_result_cache = next_block();
      m_result_index = 0;
    }
    return m_result_cache[m_result_index++];
  }

  /**
   * @brief Generates a uniform random number in the range [0, 1).
   * @return A uniform random number.
   */
  SIMDRNG_ALWAYS_INLINE constexpr double uniform() noexcept {
    return static_cast<double>(operator()() >> 11) * 0x1.0p-53;
  }

  /**
   * @brief Returns the counter of the block the most recent output came from.
   * @return The current counter block (decremented past the in-flight block).
   */
  counter_type getCounter() const noexcept {
    if (m_result_index < RESULTS_PER_BLOCK) {
      counter_type ctr = m_counter;
      dec_counter(ctr);
      return ctr;
    }
    return m_counter;
  }

  /**
   * @brief Returns the Philox key.
   * @return The current key.
   */
  const key_type &getKey() const noexcept { return m_key; }

  /**
   * @brief Sets the counter and forces a fresh block on the next output.
   * @param ctr The counter block to seek to.
   */
  void setCounter(const counter_type &ctr) noexcept {
    m_counter = ctr;
    m_result_index = RESULTS_PER_BLOCK;
  }

  /**
   * @brief Sets the key and forces a fresh block on the next output.
   * @param key The key to install.
   */
  void setKey(const key_type &key) noexcept {
    m_key = key;
    m_result_index = RESULTS_PER_BLOCK;
  }

  /**
   * @brief Returns the raw internal counter (for serialization).
   * @return The internal counter block, including the in-flight block.
   */
  const counter_type &getCounterForSerde() const noexcept { return m_counter; }

  /**
   * @brief Restores both counter and key, forcing a fresh block next.
   * @param ctr The counter block.
   * @param key The key.
   */
  void setState(const counter_type &ctr, const key_type &key) noexcept {
    m_counter = ctr;
    m_key = key;
    m_result_index = RESULTS_PER_BLOCK;
  }

  /// @brief Returns the cached output block (for serialization).
  const result_block_type &result_cache() const noexcept { return m_result_cache; }
  /// @brief Restores the cached output block (for serialization).
  void set_result_cache(const result_block_type &cache) noexcept { m_result_cache = cache; }
  /// @brief Returns the index of the next output within the cached block.
  std::uint8_t result_index() const noexcept { return m_result_index; }
  /// @brief Restores the index of the next output within the cached block.
  void set_result_index(std::uint8_t idx) noexcept { m_result_index = idx; }

  /**
   * @brief Expands a 64-bit seed into a Philox key via SplitMix.
   * @param seed The user seed.
   * @return The derived key.
   */
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

  /**
   * @brief Packs a 64-bit counter into the N-word counter block.
   * @param counter The scalar counter value.
   * @return The counter block.
   */
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

  static constexpr SIMDRNG_ALWAYS_INLINE void mulhilo(word_type a, word_type b, word_type &hi, word_type &lo) noexcept {
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
      if (++m_counter[i] != 0)
        break;
    }
  }

  static constexpr SIMDRNG_ALWAYS_INLINE void dec_counter(counter_type &ctr) noexcept {
    for (std::uint8_t i = 0; i < N; ++i) {
      if (ctr[i]-- != 0)
        break;
    }
  }

  SIMDRNG_FLATTEN constexpr SIMDRNG_ALWAYS_INLINE result_block_type next_block() noexcept {
    auto output = philox_rounds(m_counter, m_key);
    inc_counter();
    return std::bit_cast<result_block_type>(output);
  }
};

/// @brief Philox4x32-10: four 32-bit counter words, 10 rounds (the common default).
using Philox4x32 = Philox<4, 32, 10>;
/// @brief Philox2x32-10: two 32-bit counter words, 10 rounds.
using Philox2x32 = Philox<2, 32, 10>;
/// @brief Philox4x64-10: four 64-bit counter words, 10 rounds (widest stream).
using Philox4x64 = Philox<4, 64, 10>;
/// @brief Philox2x64-10: two 64-bit counter words, 10 rounds.
using Philox2x64 = Philox<2, 64, 10>;

} // namespace simdrng
// END_FILE: philox.hpp
/* End inline (quoted): philox.hpp */
/* Begin inline (quoted): splitmix.hpp */
// BEGIN_FILE: splitmix.hpp
/*  Written in 2014 originally by Guy Steele

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

Ported to C++ by Marco Barbone. Original implementation by Guy Steele.
*/


#include <cstdint>
#include <limits>

/* Begin inline (quoted): macros.hpp */
/* Skipped already inlined: macros.hpp */
/* End inline (quoted): macros.hpp */

namespace simdrng {

/**
 * @class SplitMix
 * @brief SplitMix64 — a fast 64-bit splittable generator used to seed the other engines.
 *
 * Given a single 64-bit seed it advances a Weyl sequence (the state is bumped by
 * the golden-ratio constant 0x9e3779b97f4a7c15 each call) and applies an
 * avalanche mix, producing a well-distributed stream of 64-bit values. simdrng
 * uses it to expand a user seed into the larger state of the other generators,
 * so that even low-entropy seeds (e.g. 0 or 1) yield sound starting states.
 */
class SplitMix {
public:
  /**
   * @brief Constructs the generator from a 64-bit seed state.
   * @param state The initial 64-bit state.
   */
  SIMDRNG_ALWAYS_INLINE constexpr explicit SplitMix(const std::uint64_t state) noexcept : m_state(state) {}

  /**
   * @brief Advances the state and returns the next 64-bit value.
   * @return The next 64-bit output.
   */
  SIMDRNG_ALWAYS_INLINE constexpr std::uint64_t operator()() noexcept {
    std::uint64_t z = (m_state += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
  }

  /**
   * @brief Smallest value operator() can return.
   * @return 0.
   */
  static constexpr SIMDRNG_ALWAYS_INLINE std::uint64_t(min)() noexcept {
    return std::numeric_limits<std::uint64_t>::lowest();
  }

  /**
   * @brief Largest value operator() can return.
   * @return 2^64 - 1.
   */
  static constexpr SIMDRNG_ALWAYS_INLINE std::uint64_t(max)() noexcept {
    return std::numeric_limits<std::uint64_t>::max();
  }

  /**
   * @brief Returns the current 64-bit state.
   * @return The current state.
   */
  SIMDRNG_ALWAYS_INLINE constexpr std::uint64_t getState() const noexcept { return m_state; }

  /**
   * @brief Restores the state, e.g. when resuming a saved stream.
   * @param state The state to restore.
   */
  SIMDRNG_ALWAYS_INLINE constexpr void setState(std::uint64_t state) noexcept { m_state = state; }

private:
  std::uint64_t m_state;
};

} // namespace simdrng
// END_FILE: splitmix.hpp
/* End inline (quoted): splitmix.hpp */
/* Begin inline (quoted): xoshiro_scalar.hpp */
// BEGIN_FILE: xoshiro_scalar.hpp
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


#include <array>
#include <cassert>
#include <cstdint>
#include <limits>

/* Begin inline (quoted): macros.hpp */
/* Skipped already inlined: macros.hpp */
/* End inline (quoted): macros.hpp */
/* Begin inline (quoted): splitmix.hpp */
/* Skipped already inlined: splitmix.hpp */
/* End inline (quoted): splitmix.hpp */

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
// END_FILE: xoshiro_scalar.hpp
/* End inline (quoted): xoshiro_scalar.hpp */

/* Begin inline (quoted): chacha_simd.hpp */
// BEGIN_FILE: chacha_simd.hpp

#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>

/* Begin inline (angle): poet.hpp */
// BEGIN_FILE: poet.hpp

/// \file poet.hpp
/// \brief Umbrella header for the public POET API.

// clang-format off
// Include order matters: macros.hpp must come first and undef_macros.hpp last.
// NOLINTBEGIN(llvm-include-order)
/* Begin inline (angle): macros.hpp */
// BEGIN_FILE: macros.hpp

/// \file macros.hpp
/// \brief Compiler-specific macros for portability and optimization.

// ============================================================================
// POET_UNREACHABLE
// ============================================================================
/// Marks code path as unreachable. UB if reached at runtime.
#if defined(__GNUC__) || defined(__clang__)
#define POET_UNREACHABLE() __builtin_unreachable()// NOLINT(cppcoreguidelines-macro-usage)
#elif defined(_MSC_VER)
#define POET_UNREACHABLE() __assume(false)// NOLINT(cppcoreguidelines-macro-usage)
#else
#define POET_UNREACHABLE() \
    do {                   \
    } while (false)// NOLINT(cppcoreguidelines-macro-usage)
#endif

// ============================================================================
// POET_FORCEINLINE
// ============================================================================
/// Forces function inlining regardless of compiler heuristics.
#ifdef _MSC_VER
#define POET_FORCEINLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define POET_FORCEINLINE inline __attribute__((always_inline))
#else
#define POET_FORCEINLINE inline
#endif

// ============================================================================
// POET_ALWAYS_INLINE_LAMBDA
// ============================================================================
/// Forces inlining of lambda call operators. Place after the parameter list:
///
///   auto fn = [&](auto x) POET_ALWAYS_INLINE_LAMBDA { return x; };
///
/// Uses __attribute__((always_inline)) on GCC/Clang (the only syntax that
/// applies to the call operator) and [[msvc::forceinline]] on MSVC.
/// GCC 15+ / Clang 22+: attributed generic lambdas must be assigned to a
/// variable before passing to template functions.
#if defined(_MSC_VER) && !defined(__clang__)
#define POET_ALWAYS_INLINE_LAMBDA [[msvc::forceinline]]
#elif defined(__GNUC__) || defined(__clang__)
#define POET_ALWAYS_INLINE_LAMBDA __attribute__((always_inline))
#else
#define POET_ALWAYS_INLINE_LAMBDA
#endif

// ============================================================================
// POET_ASSUME
// ============================================================================
/// Generic assumption hint. UB if expression is false at runtime.
/// Uses [[assume(expr)]] when the compiler reports support via __has_cpp_attribute
/// (GCC >= 13, Clang >= 19), otherwise falls back to compiler builtins.
#ifdef __has_cpp_attribute
#if __has_cpp_attribute(assume)
#define POET_ASSUME(expr) [[assume(expr)]]// NOLINT(cppcoreguidelines-macro-usage)
#endif
#endif
#ifndef POET_ASSUME
#if defined(__clang__)
#define POET_ASSUME(expr) __builtin_assume(expr)// NOLINT(cppcoreguidelines-macro-usage)
#elif defined(__GNUC__)
#define POET_ASSUME(expr)                \
    do {                                 \
        if (!(expr)) POET_UNREACHABLE(); \
    } while (false)// NOLINT(cppcoreguidelines-macro-usage)
#elif defined(_MSC_VER)
#define POET_ASSUME(expr) __assume(expr)// NOLINT(cppcoreguidelines-macro-usage)
#else
#define POET_ASSUME(expr) \
    do {                  \
    } while (false)// NOLINT(cppcoreguidelines-macro-usage)
#endif
#endif// ifndef POET_ASSUME

// ============================================================================
// POET_NOINLINE_FLATTEN
// ============================================================================
/// Prevents a function from being inlined into its caller (register isolation)
/// while forcing all functions it calls to be inlined into it.
///
/// This is critical for GCC codegen in static_for isolated blocks: without
/// flatten, GCC's ISRA pass extracts each functor operator() instantiation
/// into a separate out-of-line clone, causing redundant constant reloads
/// from .rodata on every call (5 FMA constants * 32 iterations = 160
/// wasted loads per block).  With flatten, GCC inlines all functor calls
/// within the block so constants are hoisted into registers once at block
/// entry — matching Clang's default behavior.
///
/// Clang already inlines everything within noinline blocks, so flatten is
/// harmless but included for consistency.
#ifdef _MSC_VER
#define POET_NOINLINE_FLATTEN __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define POET_NOINLINE_FLATTEN __attribute__((noinline, flatten))
#else
#define POET_NOINLINE_FLATTEN
#endif

// ============================================================================
// POET_LIKELY / POET_UNLIKELY
// ============================================================================
/// Branch prediction hints. Use for conditions true/false >95% of the time.
#if defined(__GNUC__) || defined(__clang__)
#define POET_LIKELY(x) __builtin_expect(!!(x), 1)// NOLINT(cppcoreguidelines-macro-usage)
#define POET_UNLIKELY(x) __builtin_expect(!!(x), 0)// NOLINT(cppcoreguidelines-macro-usage)
#else
#define POET_LIKELY(x) (x)// NOLINT(cppcoreguidelines-macro-usage)
#define POET_UNLIKELY(x) (x)// NOLINT(cppcoreguidelines-macro-usage)
#endif

// ============================================================================
// poet_count_trailing_zeros
// ============================================================================
/// Counts trailing zero bits. UB if value is 0.
/// Guarded separately so it is defined only once even when macros.hpp is
/// re-included after undef_macros.hpp.
#ifndef POET_COUNT_TRAILING_ZEROS_DEFINED
#define POET_COUNT_TRAILING_ZEROS_DEFINED
#if __cplusplus >= 202002L
#include <bit>

constexpr auto poet_count_trailing_zeros(unsigned int value) noexcept -> unsigned int {
    return static_cast<unsigned int>(std::countr_zero(value));
}

constexpr auto poet_count_trailing_zeros(unsigned long value) noexcept -> unsigned int {// NOLINT(google-runtime-int)
    return static_cast<unsigned int>(std::countr_zero(value));
}

constexpr auto poet_count_trailing_zeros(unsigned long long value) noexcept
  -> unsigned int {// NOLINT(google-runtime-int)
    return static_cast<unsigned int>(std::countr_zero(value));
}

#elif defined(__GNUC__) || defined(__clang__)

constexpr auto poet_count_trailing_zeros(unsigned int value) noexcept -> unsigned int {
    return static_cast<unsigned int>(__builtin_ctz(value));
}

constexpr auto poet_count_trailing_zeros(unsigned long value) noexcept -> unsigned int {// NOLINT(google-runtime-int)
    return static_cast<unsigned int>(__builtin_ctzl(value));
}

constexpr auto poet_count_trailing_zeros(unsigned long long value) noexcept
  -> unsigned int {// NOLINT(google-runtime-int)
    return static_cast<unsigned int>(__builtin_ctzll(value));
}

#elif defined(_MSC_VER)

#include <intrin.h>

inline unsigned int poet_count_trailing_zeros(unsigned long value) noexcept {
    unsigned long index;
    _BitScanForward(&index, value);
    return static_cast<unsigned int>(index);
}

#if defined(_WIN64)
inline unsigned int poet_count_trailing_zeros(unsigned long long value) noexcept {
    unsigned long index;
    _BitScanForward64(&index, value);
    return static_cast<unsigned int>(index);
}
#endif

inline unsigned int poet_count_trailing_zeros(unsigned int value) noexcept {
    return poet_count_trailing_zeros(static_cast<unsigned long>(value));
}

#else
#error "poet_count_trailing_zeros: no implementation for this compiler (need C++20 <bit>, GCC/Clang builtins, or MSVC)"
#endif
#endif// POET_COUNT_TRAILING_ZEROS_DEFINED

// ============================================================================
// Optimization level detection
// ============================================================================
#if defined(__OPTIMIZE__) && !defined(__OPTIMIZE_SIZE__)
#define POET_HIGH_OPTIMIZATION 1// NOLINT(cppcoreguidelines-macro-usage)
#elif defined(_MSC_VER) && !defined(_DEBUG) && defined(NDEBUG)
#define POET_HIGH_OPTIMIZATION 1// NOLINT(cppcoreguidelines-macro-usage)
#else
#define POET_HIGH_OPTIMIZATION 0// NOLINT(cppcoreguidelines-macro-usage)
#endif

// ============================================================================
// POET_HOT_LOOP
// ============================================================================
/// Marks hot-path functions for aggressive optimization and inlining.
#if defined(__GNUC__) || defined(__clang__)
#define POET_HOT_LOOP inline __attribute__((hot, always_inline))
#elif defined(_MSC_VER)
#define POET_HOT_LOOP __forceinline
#else
#define POET_HOT_LOOP inline
#endif

// ============================================================================
// C++20 Feature Detection
// ============================================================================
/// Use `consteval` for C++20+, fallback to `constexpr` for C++17.
#if __cplusplus >= 202002L
#define POET_CPP20_CONSTEVAL consteval
#else
#define POET_CPP20_CONSTEVAL constexpr
#endif

// END_FILE: macros.hpp
/* End inline (angle): macros.hpp */
/* Begin inline (angle): version.hpp */
// BEGIN_FILE: version.hpp

/// \file version.hpp
/// \brief POET version macros and constants.
///
/// Generated from version.hpp.in by cmake/GenerateVersion.cmake.
/// Do not edit by hand; re-run CMake configure or the pre-commit hook.

// NOLINTBEGIN(cppcoreguidelines-macro-usage,cppcoreguidelines-macro-to-enum,modernize-macro-to-enum)
#define POET_VERSION_MAJOR 0
#define POET_VERSION_MINOR 0
#define POET_VERSION_PATCH 0
#define POET_VERSION_STRING "0.0.0"
#define POET_VERSION_FULL "0.0.0-dev.1"
// NOLINTEND(cppcoreguidelines-macro-usage,cppcoreguidelines-macro-to-enum,modernize-macro-to-enum)

namespace poet {

inline constexpr int version_major = POET_VERSION_MAJOR;
inline constexpr int version_minor = POET_VERSION_MINOR;
inline constexpr int version_patch = POET_VERSION_PATCH;
inline constexpr const char *version_string = POET_VERSION_STRING;
inline constexpr const char *version_full = POET_VERSION_FULL;

}// namespace poet
// END_FILE: version.hpp
/* End inline (angle): version.hpp */
/* Begin inline (angle): cpu_info.hpp */
// BEGIN_FILE: cpu_info.hpp

#include <cstddef>
/* Begin inline (angle): macros.hpp */
/* Skipped already inlined: macros.hpp */
/* End inline (angle): macros.hpp */

namespace poet {

enum class instruction_set : unsigned char {
    generic,///< Generic/unknown ISA
    sse2,///< x86-64 SSE2 (128-bit vectors)
    sse4_2,///< x86-64 SSE4.2 (128-bit vectors)
    avx,///< x86-64 AVX (256-bit vectors)
    avx2,///< x86-64 AVX2 (256-bit vectors, integer ops)
    avx_512,///< x86-64 AVX-512 (512-bit vectors)
    arm_neon,///< ARM NEON (128-bit vectors)
    arm_sve,///< ARM SVE (scalable vectors)
    arm_sve2,///< ARM SVE2 (scalable vectors, enhanced)
    ppc_altivec,///< PowerPC AltiVec (128-bit vectors)
    ppc_vsx,///< PowerPC VSX (128/256-bit vectors)
    mips_msa,///< MIPS MSA (128-bit vectors)
};

/// Register and vector characteristics for a target ISA.
struct register_info {
    size_t gp_registers;
    size_t vector_registers;
    size_t vector_width_bits;
    size_t lanes_64bit;
    size_t lanes_32bit;
    instruction_set isa;
};

/// Cache line sizes used for padding and alignment decisions.
struct cache_line_info {
    size_t destructive_size;
    size_t constructive_size;
};

namespace detail {

    POET_CPP20_CONSTEVAL auto detect_instruction_set() noexcept -> instruction_set {
#ifdef __AVX512F__
        return instruction_set::avx_512;
#endif

#ifdef __AVX2__
        return instruction_set::avx2;
#endif

#ifdef __AVX__
        return instruction_set::avx;
#endif

#ifdef __SSE4_2__
        return instruction_set::sse4_2;
#endif

#ifdef __SSE2__
        return instruction_set::sse2;
#endif

#if defined(__ARM_FEATURE_SVE2) || defined(__ARM_FEATURE_SVE2__)
        return instruction_set::arm_sve2;
#endif

#if defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_SVE__)
        return instruction_set::arm_sve;
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        return instruction_set::arm_neon;
#endif

#ifdef __VSX__
        return instruction_set::ppc_vsx;
#endif

#ifdef __ALTIVEC__
        return instruction_set::ppc_altivec;
#endif

#ifdef __mips_msa
        return instruction_set::mips_msa;
#endif

        return instruction_set::generic;
    }

    POET_CPP20_CONSTEVAL auto get_register_info(instruction_set isa) noexcept -> register_info {
        switch (isa) {
        case instruction_set::sse2:
        case instruction_set::sse4_2:
            return register_info{
                16,// gp_registers
                16,// vector_registers
                128,// vector_width_bits
                2,// lanes_64bit
                4,// lanes_32bit
                isa,
            };

        case instruction_set::avx:
        case instruction_set::avx2:
            return register_info{
                16,// gp_registers
                16,// vector_registers
                256,// vector_width_bits
                4,// lanes_64bit
                8,// lanes_32bit
                isa,
            };

        case instruction_set::avx_512:
            return register_info{
                16,// gp_registers
                32,// vector_registers
                512,// vector_width_bits
                8,// lanes_64bit
                16,// lanes_32bit
                isa,
            };

        case instruction_set::arm_neon:
        case instruction_set::arm_sve:
        case instruction_set::arm_sve2:
            return register_info{
                31,// gp_registers
                32,// vector_registers
                128,// vector_width_bits
                2,// lanes_64bit
                4,// lanes_32bit
                isa,
            };

        case instruction_set::ppc_altivec:
            return register_info{
                32,// gp_registers
                32,// vector_registers
                128,// vector_width_bits
                2,// lanes_64bit
                4,// lanes_32bit
                isa,
            };

        case instruction_set::ppc_vsx:
            return register_info{
                32,// gp_registers
                64,// vector_registers
                128,// vector_width_bits
                2,// lanes_64bit
                4,// lanes_32bit
                isa,
            };

        case instruction_set::mips_msa:
            return register_info{
                32,// gp_registers
                32,// vector_registers
                128,// vector_width_bits
                2,// lanes_64bit
                4,// lanes_32bit
                isa,
            };

        case instruction_set::generic:
        default:
            return register_info{
                16,// gp_registers
                16,// vector_registers
                128,// vector_width_bits
                2,// lanes_64bit
                4,// lanes_32bit
                instruction_set::generic,
            };
        }
    }

    POET_CPP20_CONSTEVAL auto detect_cache_line_info() noexcept -> cache_line_info {
#if defined(__GCC_DESTRUCTIVE_SIZE) && defined(__GCC_CONSTRUCTIVE_SIZE)
        return cache_line_info{ __GCC_DESTRUCTIVE_SIZE, __GCC_CONSTRUCTIVE_SIZE };
#else
        switch (detect_instruction_set()) {
        case instruction_set::sse2:
        case instruction_set::sse4_2:
        case instruction_set::avx:
        case instruction_set::avx2:
        case instruction_set::avx_512:
        case instruction_set::arm_neon:
        case instruction_set::arm_sve:
        case instruction_set::arm_sve2:
            return cache_line_info{ 64, 64 };

        case instruction_set::ppc_altivec:
        case instruction_set::ppc_vsx:
            return cache_line_info{ 128, 128 };

        case instruction_set::mips_msa:
            return cache_line_info{ 32, 32 };

        case instruction_set::generic:
        default:
            return cache_line_info{ 64, 64 };
        }
#endif
    }

}// namespace detail

/// Detects the compile target ISA from compiler defines.
POET_CPP20_CONSTEVAL auto detected_isa() noexcept -> instruction_set { return detail::detect_instruction_set(); }

/// Register information for the detected ISA.
POET_CPP20_CONSTEVAL auto available_registers() noexcept -> register_info {
    return detail::get_register_info(detected_isa());
}

/// Register information for a specific ISA.
POET_CPP20_CONSTEVAL auto registers_for(instruction_set isa) noexcept -> register_info {
    return detail::get_register_info(isa);
}

POET_CPP20_CONSTEVAL auto vector_register_count() noexcept -> size_t { return available_registers().vector_registers; }

POET_CPP20_CONSTEVAL auto vector_width_bits() noexcept -> size_t { return available_registers().vector_width_bits; }

POET_CPP20_CONSTEVAL auto vector_lanes_64bit() noexcept -> size_t { return available_registers().lanes_64bit; }

POET_CPP20_CONSTEVAL auto vector_lanes_32bit() noexcept -> size_t { return available_registers().lanes_32bit; }

POET_CPP20_CONSTEVAL auto cache_line() noexcept -> cache_line_info { return detail::detect_cache_line_info(); }

POET_CPP20_CONSTEVAL auto destructive_interference_size() noexcept -> size_t { return cache_line().destructive_size; }

POET_CPP20_CONSTEVAL auto constructive_interference_size() noexcept -> size_t { return cache_line().constructive_size; }

}// namespace poet
// END_FILE: cpu_info.hpp
/* End inline (angle): cpu_info.hpp */
/* Begin inline (angle): dynamic_for.hpp */
// BEGIN_FILE: dynamic_for.hpp

/// \file dynamic_for.hpp
/// \brief Runtime loops emitted as compile-time unrolled blocks.

#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

/* Begin inline (angle): macros.hpp */
/* Skipped already inlined: macros.hpp */
/* End inline (angle): macros.hpp */


namespace poet {

namespace detail {

    template<typename...> inline constexpr bool always_false_v = false;

    struct lane_by_value_tag {};///< func(integral_constant<size_t, Lane>{}, index)
    struct index_only_tag {};///< func(index)

    template<typename Func, typename T> constexpr auto detect_callable_form() {
        if constexpr (std::is_invocable_v<Func &, std::integral_constant<std::size_t, 0>, T>) {
            return lane_by_value_tag{};
        } else if constexpr (std::is_invocable_v<Func &, T>) {
            return index_only_tag{};
        } else {
            static_assert(always_false_v<Func>, "dynamic_for callable must accept (lane, index) or (index)");
            return index_only_tag{};
        }
    }

    template<typename Func, typename T> using callable_form_t = decltype(detect_callable_form<Func, T>());

    template<std::size_t Lane, typename Func, typename T>
    POET_FORCEINLINE constexpr void invoke_lane(lane_by_value_tag /*tag*/, Func &func, T index) {
        func(std::integral_constant<std::size_t, Lane>{}, index);
    }

    template<std::size_t Lane, typename Func, typename T>
    POET_FORCEINLINE constexpr void invoke_lane(index_only_tag /*tag*/, Func &func, T index) {
        func(index);
    }

    // Emits `Count` calls as a single expanded pack; the comma-fold carries `index` forward
    // so each lane sees a distinct compile-time `Lane` and the running runtime `index`.
    template<typename FormTag, typename Callable, typename T, std::size_t... Lanes>
    POET_FORCEINLINE constexpr void
      emit_carried(Callable &callable, T index, T stride, std::index_sequence<Lanes...> /*seq*/) {
        ((invoke_lane<Lanes>(FormTag{}, callable, index), index += stride), ...);
    }

    template<typename FormTag, typename Callable, typename T, std::size_t Count>
    POET_FORCEINLINE constexpr void emit_block(FormTag /*tag*/,
      [[maybe_unused]] Callable &callable,
      [[maybe_unused]] T base,
      [[maybe_unused]] T stride) {
        if constexpr (Count > 0) { emit_carried<FormTag>(callable, base, stride, std::make_index_sequence<Count>{}); }
    }

    template<std::ptrdiff_t Step, typename FormTag, typename Callable, typename T, std::size_t... Lanes>
    POET_FORCEINLINE constexpr void
      emit_carried_ct(Callable &callable, T index, std::index_sequence<Lanes...> /*seq*/) {
        ((invoke_lane<Lanes>(FormTag{}, callable, index), index += static_cast<T>(Step)), ...);
    }

    template<std::ptrdiff_t Step, typename FormTag, typename Callable, typename T, std::size_t Count>
    POET_FORCEINLINE constexpr void
      emit_block_ct(FormTag /*tag*/, [[maybe_unused]] Callable &callable, [[maybe_unused]] T base) {
        if constexpr (Count > 0) { emit_carried_ct<Step, FormTag>(callable, base, std::make_index_sequence<Count>{}); }
    }

    // Handles a leftover count in [0, N) by emitting at most log2(N) fixed-size unrolled
    // blocks — picks the largest power of two <= N/2, optionally emits it, then recurses on
    // the remainder. Each level has a compile-time `half`, so codegen stays fully unrolled.
    template<std::size_t N, typename FormTag, typename Callable, typename T>
    POET_FORCEINLINE void tail_binary(std::size_t count, Callable &callable, T index, T stride) {
        if constexpr (N <= 1) {
        } else {
            // Largest power of two strictly less than N — the block size we might emit here.
            constexpr std::size_t half = []() constexpr -> std::size_t {
                std::size_t pow2 = 1;
                while (pow2 * 2 < N) { pow2 *= 2; }
                return pow2;
            }();
            // If this level fires, it consumes exactly `half` iterations; otherwise all
            // `count` pass through to the smaller level.
            const std::size_t rem = (count >= half) ? (count - half) : count;
            tail_binary<half, FormTag>(rem, callable, index, stride);
            if (count >= half) {
                // Smaller blocks run first over the low indices; this block picks up at
                // `index + rem*stride` so iteration order is preserved.
                emit_block<FormTag, Callable, T, half>(
                  FormTag{}, callable, static_cast<T>(index + (static_cast<T>(rem) * stride)), stride);
            }
        }
    }

    template<std::size_t N, typename FormTag, typename Callable, typename T>
    POET_NOINLINE_FLATTEN void tail_binary_noinline(std::size_t count, Callable &callable, T index, T stride) {
        tail_binary<N, FormTag>(count, callable, index, stride);
    }

    template<std::size_t N, std::ptrdiff_t Step, typename FormTag, typename Callable, typename T>
    POET_FORCEINLINE void tail_binary_ct(std::size_t count, Callable &callable, T index) {
        if constexpr (N <= 1) {
        } else {
            constexpr std::size_t half = []() constexpr -> std::size_t {
                std::size_t pow2 = 1;
                while (pow2 * 2 < N) { pow2 *= 2; }
                return pow2;
            }();
            const std::size_t rem = (count >= half) ? (count - half) : count;
            tail_binary_ct<half, Step, FormTag>(rem, callable, index);
            if (count >= half) {
                emit_block_ct<Step, FormTag, Callable, T, half>(
                  FormTag{}, callable, static_cast<T>(index + static_cast<T>(static_cast<std::ptrdiff_t>(rem) * Step)));
            }
        }
    }

    template<std::size_t N, std::ptrdiff_t Step, typename FormTag, typename Callable, typename T>
    POET_NOINLINE_FLATTEN void tail_binary_ct_noinline(std::size_t count, Callable &callable, T index) {
        tail_binary_ct<N, Step, FormTag>(count, callable, index);
    }

    // Handles signed and unsigned-wrapped-negative strides uniformly. For unsigned T, a
    // "negative" stride arrives as a large positive value (> half_max); we detect and flip
    // it so both directions share the same (dist + |stride| - 1) / |stride| ceiling formula.
    template<typename T>
    POET_FORCEINLINE constexpr auto calculate_iteration_count_complex(T begin, T end, T stride) -> std::size_t {
        constexpr bool is_unsigned = !std::is_signed_v<T>;
        constexpr T half_max = std::numeric_limits<T>::max() / 2;
        const bool is_wrapped_negative = is_unsigned && (stride > half_max);

        if (POET_UNLIKELY(stride < 0 || is_wrapped_negative)) {
            // Descending: empty unless begin > end.
            if (POET_UNLIKELY(begin <= end)) { return 0; }
            T abs_stride;
            if constexpr (std::is_signed_v<T>) {
                abs_stride = static_cast<T>(-stride);
            } else {
                // Unsigned two's-complement negation recovers the original magnitude.
                abs_stride = static_cast<T>(0) - stride;
            }
            auto dist = static_cast<std::size_t>(begin - end);
            auto ustride = static_cast<std::size_t>(abs_stride);
            return (dist + ustride - 1) / ustride;
        }

        if (begin >= end) { return 0; }

        auto dist = static_cast<std::size_t>(end - begin);
        auto ustride = static_cast<std::size_t>(stride);
        // Classic `x & (x-1) == 0` power-of-two test; replaces the divide with a shift.
        // Worth ~18x cycles on znver4 (`tzcntq+shrxq` ≈ 1c block-RT vs `divq` ≈ 18c) —
        // see /tmp/poet-asm/SUMMARY.md (T5) for the simdref+llvm-mca cross-check.
        const bool is_power_of_2 = (ustride & (ustride - 1)) == 0;

        if (is_power_of_2) {
            const unsigned int shift = poet_count_trailing_zeros(ustride);
            return (dist + ustride - 1) >> shift;
        }
        return (dist + ustride - 1) / ustride;
    }

    template<std::ptrdiff_t Step, typename T>
    POET_FORCEINLINE constexpr auto calculate_iteration_count_ct(T begin, T end) -> std::size_t {
        static_assert(Step != 0, "Step must be non-zero");
        if constexpr (Step > 0) {
            if (begin >= end) { return 0; }
            auto dist = static_cast<std::size_t>(end - begin);
            constexpr auto ustride = static_cast<std::size_t>(Step);
            return (dist + ustride - 1) / ustride;
        } else {
            if (begin <= end) { return 0; }
            auto dist = static_cast<std::size_t>(begin - end);
            constexpr auto ustride = static_cast<std::size_t>(-Step);
            return (dist + ustride - 1) / ustride;
        }
    }

    template<typename T, typename Callable, std::size_t Unroll, typename FormTag>
    POET_HOT_LOOP void
      dynamic_for_impl_general(const T begin, const T end, const T stride, Callable &callable, const FormTag tag) {
        if (POET_UNLIKELY(stride == 0)) { return; }

        std::size_t count = calculate_iteration_count_complex(begin, end, stride);
        if (POET_UNLIKELY(count == 0)) { return; }

        if constexpr (Unroll == 1) {
            T index = begin;
            for (std::size_t i = 0; i < count; ++i) {
                invoke_lane<0>(tag, callable, index);
                index += stride;
            }
        } else {
            T index = begin;
            std::size_t remaining = count;

            if (POET_UNLIKELY(count < Unroll)) {
                if (count > 0) { tail_binary<Unroll, FormTag>(count, callable, index, stride); }
                return;
            }

            const T stride_times_unroll = static_cast<T>(Unroll) * stride;
            while (remaining >= Unroll) {
                emit_block<FormTag, Callable, T, Unroll>(tag, callable, index, stride);
                index += stride_times_unroll;
                remaining -= Unroll;
            }

            if (remaining > 0) { tail_binary_noinline<Unroll, FormTag>(remaining, callable, index, stride); }
        }
    }

    template<std::ptrdiff_t Step, typename T, typename Callable, std::size_t Unroll, typename FormTag>
    POET_HOT_LOOP void dynamic_for_impl_ct_stride(const T begin, const T end, Callable &callable, const FormTag tag) {
        std::size_t count = calculate_iteration_count_ct<Step>(begin, end);
        if (POET_UNLIKELY(count == 0)) { return; }

        if constexpr (Unroll == 1) {
            T index = begin;
            constexpr T ct_stride = static_cast<T>(Step);
            for (std::size_t i = 0; i < count; ++i) {
                invoke_lane<0>(tag, callable, index);
                index += ct_stride;
            }
        } else {
            T index = begin;
            std::size_t remaining = count;

            if (POET_UNLIKELY(count < Unroll)) {
                if (count > 0) { tail_binary_ct<Unroll, Step, FormTag>(count, callable, index); }
                return;
            }

            constexpr T stride_times_unroll = static_cast<T>(static_cast<std::ptrdiff_t>(Unroll) * Step);
            while (remaining >= Unroll) {
                emit_block_ct<Step, FormTag, Callable, T, Unroll>(tag, callable, index);
                index += stride_times_unroll;
                remaining -= Unroll;
            }

            if (remaining > 0) { tail_binary_ct_noinline<Unroll, Step, FormTag>(remaining, callable, index); }
        }
    }

}// namespace detail

// ============================================================================
// Public API
// ============================================================================

/// \brief Runs `[begin, end)` with compile-time unrolled blocks.
///
/// `func` may take `(index)` or `(lane, index)`, where `lane` is
/// `std::integral_constant<std::size_t, L>`. Use the lane form for
/// multi-accumulator kernels; prefer a plain `for` loop for trivial index-only work.
template<std::size_t Unroll, typename T1, typename T2, typename T3, typename Func>
POET_FORCEINLINE constexpr void dynamic_for(T1 begin, T2 end, T3 step, Func &&func) {
    static_assert(Unroll > 0, "dynamic_for requires Unroll > 0");

    using T = std::common_type_t<T1, T2, T3>;
    const T stride = static_cast<T>(step);

    auto run = [&](auto &callable) POET_ALWAYS_INLINE_LAMBDA -> void {
        using callable_t = std::remove_reference_t<decltype(callable)>;
        using form_tag = detail::callable_form_t<callable_t, T>;
        if (stride == static_cast<T>(1)) {
            detail::dynamic_for_impl_ct_stride<1, T, callable_t, Unroll>(
              static_cast<T>(begin), static_cast<T>(end), callable, form_tag{});
        } else {
            detail::dynamic_for_impl_general<T, callable_t, Unroll>(
              static_cast<T>(begin), static_cast<T>(end), stride, callable, form_tag{});
        }
    };

    if constexpr (std::is_lvalue_reference_v<Func>) {
        run(func);
    } else {
        std::remove_reference_t<Func> local(std::forward<Func>(func));
        run(local);
    }
}

/// \brief Runs `[begin, end)` with a compile-time stride.
template<std::size_t Unroll, std::ptrdiff_t Step, typename T1, typename T2, typename Func>
POET_FORCEINLINE constexpr void dynamic_for(T1 begin, T2 end, Func &&func) {
    static_assert(Unroll > 0, "dynamic_for requires Unroll > 0");
    static_assert(Step != 0, "dynamic_for requires Step != 0");

    using T = std::common_type_t<T1, T2>;

    auto run = [&](auto &callable) POET_ALWAYS_INLINE_LAMBDA -> void {
        using callable_t = std::remove_reference_t<decltype(callable)>;
        using form_tag = detail::callable_form_t<callable_t, T>;
        detail::dynamic_for_impl_ct_stride<Step, T, callable_t, Unroll>(
          static_cast<T>(begin), static_cast<T>(end), callable, form_tag{});
    };

    if constexpr (std::is_lvalue_reference_v<Func>) {
        run(func);
    } else {
        std::remove_reference_t<Func> local(std::forward<Func>(func));
        run(local);
    }
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// \brief Runs `[begin, end)` with an inferred step of `+1` or `-1`.
template<std::size_t Unroll, typename T1, typename T2, typename Func>
POET_FORCEINLINE constexpr void dynamic_for(T1 begin, T2 end, Func &&func) {
    using T = std::common_type_t<T1, T2>;
    T s_begin = static_cast<T>(begin);
    T s_end = static_cast<T>(end);
    T step = (s_begin <= s_end) ? static_cast<T>(1) : static_cast<T>(-1);

    dynamic_for<Unroll>(s_begin, s_end, step, std::forward<Func>(func));
}

/// \brief Convenience overload for `[0, count)`.
template<std::size_t Unroll, typename Func>
POET_FORCEINLINE constexpr void dynamic_for(std::size_t count, Func &&func) {
    dynamic_for<Unroll>(static_cast<std::size_t>(0), count, std::size_t{ 1 }, std::forward<Func>(func));
}
#endif

}// namespace poet


#if __cplusplus >= 202002L
#include <ranges>
#include <tuple>

namespace poet {

template<typename Func, std::size_t Unroll> struct dynamic_for_adaptor {
    Func func;
    constexpr explicit dynamic_for_adaptor(Func f) : func(std::move(f)) {}
};

template<typename Func, std::size_t Unroll, typename Range>
requires std::ranges::range<Range> void operator|(Range const &r, dynamic_for_adaptor<Func, Unroll> const &ad) {
    auto it = std::ranges::begin(r);
    auto it_end = std::ranges::end(r);

    if (it == it_end) return;// empty range

    using ValT = std::remove_reference_t<decltype(*it)>;
    ValT start = *it;

    std::size_t count = 0;
    for (auto jt = it; jt != it_end; ++jt) ++count;

    // Treat the range as a consecutive [start, start + count) sequence.
    poet::dynamic_for<Unroll>(start, static_cast<ValT>(start + static_cast<ValT>(count)), ad.func);
}

template<typename Func, std::size_t Unroll, typename B, typename E, typename S>
void operator|(std::tuple<B, E, S> const &t, dynamic_for_adaptor<Func, Unroll> const &ad) {
    auto [b, e, s] = t;
    poet::dynamic_for<Unroll>(b, e, s, ad.func);
}

template<std::size_t U, typename F> constexpr auto make_dynamic_for(F &&f) -> dynamic_for_adaptor<std::decay_t<F>, U> {
    return dynamic_for_adaptor<std::decay_t<F>, U>(std::forward<F>(f));
}

}// namespace poet
#endif// __cplusplus >= 202002L
// END_FILE: dynamic_for.hpp
/* End inline (angle): dynamic_for.hpp */
/* Begin inline (angle): dispatch.hpp */
// BEGIN_FILE: dispatch.hpp

/// \file dispatch.hpp
/// \brief Runtime-to-compile-time dispatch for integer choices and tuples.

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

/* Begin inline (angle): macros.hpp */
/* Skipped already inlined: macros.hpp */
/* End inline (angle): macros.hpp */
/* Begin inline (angle): mdspan_utils.hpp */
// BEGIN_FILE: mdspan_utils.hpp

/// \file mdspan_utils.hpp
/// \brief Multidimensional index utilities for N-D dispatch table generation.
///
/// Provides row-major stride computation and total-size calculation used by
/// the N-D function-pointer-table dispatch in dispatch.hpp.

#include <array>
#include <cstddef>
/* Begin inline (angle): macros.hpp */
/* Skipped already inlined: macros.hpp */
/* End inline (angle): macros.hpp */

namespace poet::detail {

/// Total size (product of all dimensions).
template<std::size_t N>
POET_CPP20_CONSTEVAL auto compute_total_size(const std::array<std::size_t, N> &dims) -> std::size_t {
    std::size_t total = 1;
    for (std::size_t i = 0; i < N; ++i) { total *= dims[i]; }
    return total;
}

/// Compute row-major strides. stride[i] = product of dims[i+1..N-1].
template<std::size_t N>
POET_CPP20_CONSTEVAL auto compute_strides(const std::array<std::size_t, N> &dims) -> std::array<std::size_t, N> {
    std::array<std::size_t, N> strides{};
    if constexpr (N > 0) {
        strides[N - 1] = 1;
        for (std::size_t i = N - 1; i > 0; --i) { strides[i - 1] = strides[i] * dims[i]; }
    }
    return strides;
}

}// namespace poet::detail
// END_FILE: mdspan_utils.hpp
/* End inline (angle): mdspan_utils.hpp */

namespace poet {

/// \brief Concise tuple syntax for `dispatch_set`.
template<auto... Vs> struct tuple_ {};

namespace detail {

    template<typename T>
    using result_holder = std::conditional_t<std::is_void_v<T>, std::optional<std::monostate>, std::optional<T>>;

    template<typename Functor, typename ResultType, typename RuntimeTuple, typename... Args> struct seq_matcher;

    template<typename ValueType,
      ValueType... V,
      typename ResultType,
      typename RuntimeTuple,
      typename Functor,
      typename... Args>
    struct seq_matcher<std::integer_sequence<ValueType, V...>, ResultType, RuntimeTuple, Functor, Args...> {
        template<std::size_t... Idx, typename F>
        static auto
          impl(std::index_sequence<Idx...> /*idx_seq*/, const RuntimeTuple &runtime_tuple, F &&func, Args &&...args)
            -> result_holder<ResultType> {
            result_holder<ResultType> res;
            // Short-circuiting AND fold: all runtime slots must equal their compile-time counterparts.
            if (((std::get<Idx>(runtime_tuple) == V) && ...)) {
                if constexpr (std::is_void_v<ResultType>) {
                    std::forward<F>(func).template operator()<V...>(std::forward<Args>(args)...);
                    res = std::monostate{};
                } else {
                    res = std::forward<F>(func).template operator()<V...>(std::forward<Args>(args)...);
                }
            }
            return res;
        }

        template<typename F>
        static auto match_and_call(const RuntimeTuple &runtime_tuple, F &&func, Args &&...args)
          -> result_holder<ResultType> {
            return impl(std::make_index_sequence<sizeof...(V)>{},
              runtime_tuple,
              std::forward<F>(func),
              std::forward<Args>(args)...);
        }
    };

    template<typename Seq, typename Functor, typename... Args> struct seq_call_result;

    template<typename ValueType, ValueType... V, typename Functor, typename... Args>
    struct seq_call_result<std::integer_sequence<ValueType, V...>, Functor, Args...> {
        using type = decltype(std::declval<Functor>().template operator()<V...>(std::declval<Args>()...));
    };

    template<int Start, int... Is>
    auto inclusive_range_impl(std::integer_sequence<int, Is...>) -> std::integer_sequence<int, (Start + Is)...>;

}// namespace detail

/// \brief Inclusive integer sequence `[Start, End]`.
template<int Start, int End>
using inclusive_range =
  decltype(detail::inclusive_range_impl<Start>(std::make_integer_sequence<int, End - Start + 1>{}));

/// \brief Runtime value paired with the compile-time candidates to probe.
template<typename Seq> struct dispatch_param {
    int runtime_val;
    using seq_type = Seq;
};

namespace detail {
    template<typename T> struct is_dispatch_param : std::false_type {};
    template<typename Seq> struct is_dispatch_param<dispatch_param<Seq>> : std::true_type {};

    template<typename T> inline constexpr bool is_dispatch_param_v = is_dispatch_param<std::decay_t<T>>::value;

    template<typename T> struct is_dispatch_param_tuple : std::false_type {};

    template<typename... Ts>
    struct is_dispatch_param_tuple<std::tuple<Ts...>> : std::bool_constant<(is_dispatch_param_v<Ts> && ...)> {};

    template<typename T>
    inline constexpr bool is_dispatch_param_tuple_v = is_dispatch_param_tuple<std::decay_t<T>>::value;
}// namespace detail

namespace detail {

    template<typename Sequence> struct sequence_size;

    template<typename T, T... Values>
    struct sequence_size<std::integer_sequence<T, Values...>>
      : std::integral_constant<std::size_t, sizeof...(Values)> {};

    template<typename Seq> struct is_contiguous_sequence : std::false_type {};

    template<int First, int... Rest>
    struct is_contiguous_sequence<std::integer_sequence<int, First, Rest...>>
      : std::bool_constant<(
          std::max({ First, Rest... }) - std::min({ First, Rest... }) + 1 == static_cast<int>(1 + sizeof...(Rest)))> {};

    template<typename ParamTuple, typename = std::make_index_sequence<std::tuple_size_v<std::decay_t<ParamTuple>>>>
    struct all_contiguous;

    template<typename ParamTuple, std::size_t... Idx>
    struct all_contiguous<ParamTuple, std::index_sequence<Idx...>>
      : std::bool_constant<(
          is_contiguous_sequence<typename std::tuple_element_t<Idx, std::decay_t<ParamTuple>>::seq_type>::value
          && ...)> {};

    template<typename ParamTuple> inline constexpr bool all_contiguous_v = all_contiguous<ParamTuple>::value;

    template<typename Sequence> struct sequence_first;

    template<typename Seq> struct sparse_index;

    template<int... Values> struct sparse_index<std::integer_sequence<int, Values...>> {
        using seq_type = std::integer_sequence<int, Values...>;
        static constexpr std::size_t value_count = sizeof...(Values);

        struct sorted_data_t {
            std::array<int, value_count> sorted_keys{};
            std::array<std::size_t, value_count> sorted_indices{};
        };

        // Insertion sort that carries original positions alongside keys, so the dispatch table
        // preserves user-declared slot order while lookups can use ordered search (binary/strided).
        static constexpr sorted_data_t sorted_data = []() constexpr -> sorted_data_t {
            sorted_data_t out{};
            out.sorted_keys = std::array<int, value_count>{ Values... };
            for (std::size_t i = 0; i < value_count; ++i) { out.sorted_indices[i] = i; }
            for (std::size_t i = 1; i < value_count; ++i) {
                const int current_key = out.sorted_keys[i];
                const std::size_t current_index = out.sorted_indices[i];
                std::size_t insert_pos = i;
                // Shift larger keys (and their original-position tags) right in lockstep
                // until we find the slot where `current_key` belongs.
                while (insert_pos > 0 && out.sorted_keys[insert_pos - 1] > current_key) {
                    out.sorted_keys[insert_pos] = out.sorted_keys[insert_pos - 1];
                    out.sorted_indices[insert_pos] = out.sorted_indices[insert_pos - 1];
                    --insert_pos;
                }
                out.sorted_keys[insert_pos] = current_key;
                out.sorted_indices[insert_pos] = current_index;
            }
            return out;
        }();

        static constexpr std::size_t unique_count = []() constexpr -> std::size_t {
            if constexpr (value_count == 0) { return 0; }
            std::size_t count = 1;
            for (std::size_t i = 1; i < value_count; ++i) {
                if (sorted_data.sorted_keys[i] != sorted_data.sorted_keys[i - 1]) { ++count; }
            }
            return count;
        }();

        static constexpr std::array<int, unique_count> keys = []() constexpr -> std::array<int, unique_count> {
            std::array<int, unique_count> out{};
            if constexpr (value_count > 0) {
                std::size_t out_i = 0;
                out[out_i++] = sorted_data.sorted_keys[0];
                for (std::size_t i = 1; i < value_count; ++i) {
                    if (sorted_data.sorted_keys[i] != sorted_data.sorted_keys[i - 1]) {
                        out[out_i++] = sorted_data.sorted_keys[i];
                    }
                }
            }
            return out;
        }();

        static constexpr std::array<std::size_t, unique_count> indices =
          []() constexpr -> std::array<std::size_t, unique_count> {
            std::array<std::size_t, unique_count> out{};
            if constexpr (value_count > 0) {
                std::size_t out_i = 0;
                out[out_i++] = sorted_data.sorted_indices[0];
                for (std::size_t i = 1; i < value_count; ++i) {
                    if (sorted_data.sorted_keys[i] != sorted_data.sorted_keys[i - 1]) {
                        out[out_i++] = sorted_data.sorted_indices[i];
                    }
                }
            }
            return out;
        }();
    };

    inline constexpr std::size_t dispatch_npos = static_cast<std::size_t>(-1);

    template<typename Seq, bool IsContiguous = is_contiguous_sequence<Seq>::value> struct seq_lookup;

    template<int... Values> struct seq_lookup<std::integer_sequence<int, Values...>, true> {
        static constexpr int first = sequence_first<std::integer_sequence<int, Values...>>::value;
        static constexpr std::size_t len = sizeof...(Values);
        static constexpr bool ascending = (first == std::min({ Values... }));

        static POET_FORCEINLINE auto find(int value) -> std::size_t {
            // Unsigned subtraction folds "below first" into "far above len", so a single
            // `idx < len` check handles both underflow and overflow with no extra branch.
            std::size_t idx = 0;
            if constexpr (ascending) {
                idx = static_cast<std::size_t>(static_cast<unsigned int>(value) - static_cast<unsigned int>(first));
            } else {
                idx = static_cast<std::size_t>(static_cast<unsigned int>(first) - static_cast<unsigned int>(value));
            }
            if (idx < len) { return idx; }
            return dispatch_npos;
        }
    };

    // Non-contiguous sequences: detect a uniform positive stride at compile time and
    // specialise `find` to a div/mod (strided) instead of a binary search (truly sparse).
    template<int... Values> struct seq_lookup<std::integer_sequence<int, Values...>, false> {
        using sparse_data = sparse_index<std::integer_sequence<int, Values...>>;

        static constexpr bool is_strided = []() constexpr -> bool {
            if constexpr (sparse_data::unique_count < 2) {
                return false;
            } else {
                // Reject non-positive strides up front so `find` can use unsigned math.
                constexpr int stride0 = sparse_data::keys[1] - sparse_data::keys[0];
                if constexpr (stride0 <= 0) { return false; }
                // cppcheck-suppress syntaxError
                // All adjacent gaps must match `stride0`, otherwise fall back to binary search.
                for (std::size_t i = 2; i < sparse_data::unique_count; ++i) {
                    if (sparse_data::keys[i] - sparse_data::keys[i - 1] != stride0) { return false; }
                }
                return true;
            }
        }();

        static POET_FORCEINLINE auto find(int value) -> std::size_t {
            if constexpr (is_strided) {
                static constexpr int first = sparse_data::keys[0];
                static constexpr int stride = sparse_data::keys[1] - sparse_data::keys[0];
                const int diff = value - first;
                // Miss when below range or not aligned to the stride grid.
                if (diff < 0 || diff % stride != 0) { return dispatch_npos; }
                const auto idx = static_cast<std::size_t>(diff / stride);
                // Remap sorted position back to the user's declared slot.
                if (idx < sparse_data::unique_count) { return sparse_data::indices[idx]; }
                return dispatch_npos;
            } else {
                // Sorted keys → binary search; `indices` undoes the sort to the original slot.
                const auto pos = std::lower_bound(sparse_data::keys.begin(), sparse_data::keys.end(), value);
                if (pos != sparse_data::keys.end() && *pos == value) {
                    return sparse_data::indices[static_cast<std::size_t>(pos - sparse_data::keys.begin())];
                }
                return dispatch_npos;
            }
        }
    };

    template<int First, int... Rest>
    struct sequence_first<std::integer_sequence<int, First, Rest...>> : std::integral_constant<int, First> {};

    template<typename ParamTuple, std::size_t... Idx>
    POET_CPP20_CONSTEVAL auto dimensions_of_impl(std::index_sequence<Idx...> /*idxs*/)
      -> std::array<std::size_t, sizeof...(Idx)> {
        using P = std::decay_t<ParamTuple>;
        return std::array<std::size_t, sizeof...(Idx)>{
            sequence_size<typename std::tuple_element_t<Idx, P>::seq_type>::value...
        };
    }

    template<typename ParamTuple>
    POET_CPP20_CONSTEVAL auto dimensions_of() -> std::array<std::size_t, std::tuple_size_v<std::decay_t<ParamTuple>>> {
        return dimensions_of_impl<ParamTuple>(std::make_index_sequence<std::tuple_size_v<std::decay_t<ParamTuple>>>{});
    }

    template<typename ParamTuple, std::size_t... Idx>
    POET_FORCEINLINE auto flat_index_sparse(const ParamTuple &params, std::index_sequence<Idx...> /*idxs*/)
      -> std::size_t {
        using P = std::decay_t<ParamTuple>;
        constexpr auto strides = compute_strides(dimensions_of<P>());

        const std::array<std::size_t, sizeof...(Idx)> indices = {
            seq_lookup<typename std::tuple_element_t<Idx, P>::seq_type>::find(std::get<Idx>(params).runtime_val)...
        };

        const bool all_hit = ((indices[Idx] != dispatch_npos) && ...);
        if (POET_UNLIKELY(!all_hit)) { return dispatch_npos; }

        return ((indices[Idx] * strides[Idx]) + ...);
    }

    template<typename Seq> POET_FORCEINLINE constexpr auto contiguous_offset(int value) noexcept -> std::size_t {
        constexpr auto ufirst = static_cast<unsigned int>(sequence_first<Seq>::value);
        const auto uval = static_cast<unsigned int>(value);
        if constexpr (seq_lookup<Seq>::ascending) {
            return static_cast<std::size_t>(uval - ufirst);
        } else {
            return static_cast<std::size_t>(ufirst - uval);
        }
    }

    template<typename ParamTuple, std::size_t... Idx>
    POET_FORCEINLINE auto flat_index_contiguous(const ParamTuple &params, std::index_sequence<Idx...> /*idxs*/)
      -> std::size_t {
        using P = std::decay_t<ParamTuple>;
        constexpr auto strides = compute_strides(dimensions_of<P>());

        const std::array<std::size_t, sizeof...(Idx)> mapped = {
            contiguous_offset<typename std::tuple_element_t<Idx, P>::seq_type>(std::get<Idx>(params).runtime_val)...
        };

        // Bitwise-OR fold (not logical) so each bound check is evaluated branch-free;
        // the aggregate OOB flag is consumed once at the bottom.
        const std::size_t oob = (static_cast<std::size_t>(
                                   mapped[Idx] >= sequence_size<typename std::tuple_element_t<Idx, P>::seq_type>::value)
                                 | ...);

        const std::size_t flat = ((mapped[Idx] * strides[Idx]) + ...);

        return (oob == 0) ? flat : dispatch_npos;
    }

    template<typename ParamTuple> POET_FORCEINLINE auto extract_flat_index(const ParamTuple &params) -> std::size_t {
        constexpr std::size_t num_dims = std::tuple_size_v<std::decay_t<ParamTuple>>;
        if constexpr (all_contiguous_v<ParamTuple>) {
            return flat_index_contiguous(params, std::make_index_sequence<num_dims>{});
        } else {
            return flat_index_sparse(params, std::make_index_sequence<num_dims>{});
        }
    }

    template<typename A, typename B> struct seq_equal;
    template<typename T, T... A, T... B>
    struct seq_equal<std::integer_sequence<T, A...>, std::integer_sequence<T, B...>>
      : std::bool_constant<((A == B) && ...)> {};

    template<typename... S> struct unique_helper;
    template<> struct unique_helper<> : std::true_type {};
    template<typename Head, typename... Rest>
    struct unique_helper<Head, Rest...>
      : std::bool_constant<(!(seq_equal<Head, Rest>::value || ...) && unique_helper<Rest...>::value)> {};

    template<typename T> struct is_tuple : std::false_type {};
    template<typename... Ts> struct is_tuple<std::tuple<Ts...>> : std::true_type {};

    template<typename S> POET_CPP20_CONSTEVAL auto as_seq_tuple(S /*seq*/) {
        if constexpr (is_tuple<S>::value) {
            return S{};
        } else {
            return std::tuple<S>{ S{} };
        }
    }

    template<typename Tuple, std::size_t... Indices>
    POET_CPP20_CONSTEVAL auto extract_sequences_impl(std::index_sequence<Indices...> /*idxs*/) {
        using TupleType = std::remove_reference_t<Tuple>;
        return std::tuple_cat(as_seq_tuple(typename std::tuple_element_t<Indices, TupleType>::seq_type{})...);
    }

    template<typename Tuple> POET_CPP20_CONSTEVAL auto extract_sequences() {
        using TupleType = std::remove_reference_t<Tuple>;
        return extract_sequences_impl<TupleType>(std::make_index_sequence<std::tuple_size_v<TupleType>>{});
    }

    // Computes the functor's return type by probing both calling conventions the dispatcher
    // supports: `func(integral_constant<int, V>{}, args...)` (value form) and
    // `func.template operator()<V>(args...)` (template form). Value form is preferred when viable.
    template<typename Functor, typename... Seq> struct dispatch_result_helper {
        // First preference: value-argument form (passes std::integral_constant values as parameters).
        template<typename... Args>
        static auto compute_impl(std::true_type /*use_value_args*/)
          -> decltype(std::declval<Functor &>()(std::integral_constant<int, sequence_first<Seq>::value>{}...,
            std::declval<Args>()...));

        // Fallback: template-parameter form.
        template<typename... Args>
        static auto compute_impl(std::false_type /*use_value_args*/)
          -> decltype(std::declval<Functor &>().template operator()<sequence_first<Seq>::value...>(
            std::declval<Args>()...));

        // Detection of value-argument viability using std::is_invocable
        template<typename... Args>
        static auto compute() -> decltype(compute_impl<Args...>(std::integral_constant<bool,
          std::is_invocable_v<Functor &, std::integral_constant<int, sequence_first<Seq>::value>..., Args...>>{})) {
            return compute_impl<Args...>(std::integral_constant<bool,
              std::is_invocable_v<Functor &, std::integral_constant<int, sequence_first<Seq>::value>..., Args...>>{});
        }
    };

    template<typename Functor, typename SequenceTuple, typename... Args> struct dispatch_result;

    template<typename Functor, typename... Seq, typename... Args>
    struct dispatch_result<Functor, std::tuple<Seq...>, Args...> {
        using type = decltype(dispatch_result_helper<Functor, Seq...>::template compute<Args...>());
    };

    template<typename Functor, typename SequenceTuple, typename... Args>
    using dispatch_result_t = typename dispatch_result<Functor, SequenceTuple, Args...>::type;

    template<typename... Args> struct arg_pack {};

    template<typename T>
    inline constexpr bool is_stateless_v = std::is_empty_v<T> && std::is_default_constructible_v<T>;

    // Picks the calling convention for each forwarded arg through the function-pointer table.
    // Small trivially-copyable rvalue/const-lvalue args are passed by value (cheaper than
    // synthesising a reference); everything else keeps its original reference category.
    template<typename T> struct arg_pass {
        using raw = std::remove_reference_t<T>;
        using raw_unqual = std::remove_cv_t<raw>;
        static constexpr bool is_small_trivial =
          std::is_trivially_copyable_v<raw_unqual> && (sizeof(raw_unqual) <= 2 * sizeof(void *));

        static constexpr bool caller_allows_copy =
          std::is_rvalue_reference_v<T> || (std::is_lvalue_reference_v<T> && std::is_const_v<raw>);

        static constexpr bool by_value = is_small_trivial && caller_allows_copy;

        using type = std::conditional_t<by_value, raw_unqual, T>;
    };

    template<typename T> using pass_t = typename arg_pass<T>::type;

    template<typename Functor, int Value, typename ArgPack> struct can_use_value_form : std::false_type {};

    template<typename Functor, int Value, typename... Args>
    struct can_use_value_form<Functor, Value, arg_pack<Args...>>
      : std::bool_constant<std::is_invocable_v<Functor &, std::integral_constant<int, Value>, Args &&...>> {};

    template<typename Functor, typename ArgPack, typename R, int... Values> struct table_builder;

    template<typename Functor, typename... Args, typename R, int... Values>
    struct table_builder<Functor, arg_pack<Args...>, R, Values...> {
        static constexpr int first_value = sequence_first<std::integer_sequence<int, Values...>>::value;

        // Each entry is a plain function pointer. Stateless functors are default-constructed
        // inside the thunk (no closure needed); stateful functors take the functor by ref so
        // the signature stays identical across all entries in the array.
        template<int V> static POET_CPP20_CONSTEVAL auto make_entry() {
            if constexpr (is_stateless_v<Functor>) {
                return +[](pass_t<Args &&>... args) -> R {
                    Functor func{};
                    constexpr bool use_value_form = can_use_value_form<Functor, V, arg_pack<Args...>>::value;
                    if constexpr (use_value_form) {
                        return func(std::integral_constant<int, V>{}, std::forward<Args>(args)...);
                    } else {
                        return func.template operator()<V>(std::forward<Args>(args)...);
                    }
                };
            } else {
                return +[](Functor &func, pass_t<Args &&>... args) -> R {
                    constexpr bool use_value_form = can_use_value_form<Functor, V, arg_pack<Args...>>::value;
                    if constexpr (use_value_form) {
                        return func(std::integral_constant<int, V>{}, std::forward<Args>(args)...);
                    } else {
                        return func.template operator()<V>(std::forward<Args>(args)...);
                    }
                };
            }
        }

        static POET_CPP20_CONSTEVAL auto make() {
            using fn_type = decltype(make_entry<first_value>());
            return std::array<fn_type, sizeof...(Values)>{ make_entry<Values>()... };
        }
    };

    template<typename Functor, typename ArgPack, typename R, int... Values>
    POET_CPP20_CONSTEVAL auto make_dispatch_table(std::integer_sequence<int, Values...> /*seq*/) {
        return table_builder<Functor, ArgPack, R, Values...>::make();
    }

    template<typename Functor, typename ArgPack, typename SeqTuple, typename IndexSeq> struct nd_table_builder;

    template<typename Functor, typename... Args, typename... Seqs, std::size_t... FlatIndices>
    struct nd_table_builder<Functor, arg_pack<Args...>, std::tuple<Seqs...>, std::index_sequence<FlatIndices...>> {

        static constexpr std::array<std::size_t, sizeof...(Seqs)> dims_ = { sequence_size<Seqs>::value... };
        static constexpr std::array<std::size_t, sizeof...(Seqs)> strides_ = compute_strides(dims_);

        template<std::size_t I, typename Seq> struct get_sequence_value;

        template<std::size_t I, int... Values> struct get_sequence_value<I, std::integer_sequence<int, Values...>> {
            static constexpr std::array<int, sizeof...(Values)> values = { Values... };
            static constexpr int value = values[I];
        };

        // Decode a flat table index back to its per-dimension coordinate via row-major strides.
        template<std::size_t FlatIdx, std::size_t DimIdx>
        static constexpr std::size_t dim_index_v = FlatIdx / strides_[DimIdx] % dims_[DimIdx];

        // For a given flat index, materialises the tuple of per-dim values as an array and
        // exposes each as `integral_constant<int, V>` via `ic<N>` — that's what the functor sees.
        template<std::size_t FlatIdx, std::size_t... SeqIdx> struct value_extractor {
            static constexpr std::array<int, sizeof...(SeqIdx)> values = {
                get_sequence_value<dim_index_v<FlatIdx, SeqIdx>,
                  std::tuple_element_t<SeqIdx, std::tuple<Seqs...>>>::value...
            };

            template<std::size_t N> using ic = std::integral_constant<int, values[N]>;
        };

        template<std::size_t FlatIdx> struct nd_index_caller {
            template<std::size_t... Is>
            static auto make_ve(std::index_sequence<Is...>) -> value_extractor<FlatIdx, Is...>;
            using VE = decltype(make_ve(std::make_index_sequence<sizeof...(Seqs)>{}));

            template<typename R, std::size_t... SeqIdx>
            static POET_FORCEINLINE auto invoke(Functor &func, std::index_sequence<SeqIdx...> /*idx*/, Args &&...args)
              -> R {
                using VE_local = value_extractor<FlatIdx, SeqIdx...>;
                constexpr bool use_value_form =
                  std::is_invocable_v<Functor &, typename VE_local::template ic<SeqIdx>..., Args &&...>;
                if constexpr (use_value_form) {
                    return func(typename VE_local::template ic<SeqIdx>{}..., std::forward<Args>(args)...);
                } else {
                    return func.template operator()<VE_local::template ic<SeqIdx>::value...>(
                      std::forward<Args>(args)...);
                }
            }

            template<typename R> static POET_FORCEINLINE auto call(Functor &func, pass_t<Args &&>... args) -> R {
                return invoke<R>(func, std::make_index_sequence<sizeof...(Seqs)>{}, std::forward<Args>(args)...);
            }

            template<typename R> static POET_FORCEINLINE auto call_stateless(pass_t<Args &&>... args) -> R {
                Functor func{};
                return invoke<R>(func, std::make_index_sequence<sizeof...(Seqs)>{}, std::forward<Args>(args)...);
            }
        };

        template<typename R> static constexpr auto make_table() {
            if constexpr (is_stateless_v<Functor>) {
                using fn_type = decltype(&nd_index_caller<0>::template call_stateless<R>);
                return std::array<fn_type, sizeof...(FlatIndices)>{
                    &nd_index_caller<FlatIndices>::template call_stateless<R>...
                };
            } else {
                using fn_type = decltype(&nd_index_caller<0>::template call<R>);
                return std::array<fn_type, sizeof...(FlatIndices)>{
                    &nd_index_caller<FlatIndices>::template call<R>...
                };
            }
        }
    };

    template<typename Functor, typename ArgPack, typename R, typename... Seqs>
    POET_CPP20_CONSTEVAL auto make_nd_dispatch_table(std::tuple<Seqs...> /*seqs*/) {
        constexpr std::size_t total_size = (sequence_size<Seqs>::value * ... * 1);
        return nd_table_builder<Functor, ArgPack, std::tuple<Seqs...>, std::make_index_sequence<total_size>>::
          template make_table<R>();
    }

}// namespace detail

/// \brief Exact set of allowed tuples for sparse dispatch.
template<typename ValueType, typename... Tuples> struct dispatch_set {
    template<typename TupleHelper> struct convert_tuple;

    template<auto... Vs> struct convert_tuple<tuple_<Vs...>> {
        using type = std::integer_sequence<ValueType, static_cast<ValueType>(Vs)...>;
    };

    using seq_type = std::tuple<typename convert_tuple<Tuples>::type...>;
    using first_t = std::tuple_element_t<0, seq_type>;

    static_assert(sizeof...(Tuples) >= 1, "dispatch_set requires at least one allowed tuple");

    static constexpr std::size_t tuple_arity = detail::sequence_size<first_t>::value;

    template<typename S> struct same_arity : std::bool_constant<detail::sequence_size<S>::value == tuple_arity> {};

    static_assert((same_arity<typename convert_tuple<Tuples>::type>::value && ...),
      "All tuples in dispatch_set must have the same arity");

    static_assert(detail::unique_helper<typename convert_tuple<Tuples>::type...>::value,
      "dispatch_set contains duplicate allowed tuples");

    using runtime_array_t = std::array<ValueType, tuple_arity>;

  private:
    runtime_array_t runtime_val;

  public:
    template<typename... Args, typename = std::enable_if_t<sizeof...(Args) == tuple_arity>>
    explicit dispatch_set(Args &&...args) : runtime_val{ static_cast<ValueType>(std::forward<Args>(args))... } {}

    template<std::size_t... Idx> [[nodiscard]] auto runtime_tuple_impl(std::index_sequence<Idx...> /*idxs*/) const {
        return std::make_tuple(runtime_val[Idx]...);
    }

    [[nodiscard]] auto runtime_tuple() const { return runtime_tuple_impl(std::make_index_sequence<tuple_arity>{}); }
};

struct throw_on_no_match_t {};
inline constexpr throw_on_no_match_t throw_on_no_match{};

/// \brief Thrown when a `throw_on_no_match` dispatch has no matching specialization.
struct no_match_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

namespace detail {

    template<typename R, typename EntryFn, typename FunctorFwd, typename... Args>
    POET_FORCEINLINE auto invoke_table_entry(FunctorFwd &functor, EntryFn entry, Args &&...args) -> R {
        using FT = std::decay_t<FunctorFwd>;
        if constexpr (is_stateless_v<FT>) {
            if constexpr (std::is_void_v<R>) {
                entry(std::forward<Args>(args)...);
                return;
            } else {
                return entry(std::forward<Args>(args)...);
            }
        } else {
            if constexpr (std::is_void_v<R>) {
                entry(static_cast<FT &>(functor), std::forward<Args>(args)...);
                return;
            } else {
                return entry(static_cast<FT &>(functor), std::forward<Args>(args)...);
            }
        }
    }

    template<bool ThrowOnNoMatch, typename R, typename Functor, typename ParamTuple, typename... Args>
    POET_FORCEINLINE auto dispatch_1d(Functor &functor, ParamTuple const &params, Args &&...args) -> R {
        using FirstParam = std::tuple_element_t<0, std::remove_reference_t<ParamTuple>>;
        using Seq = typename FirstParam::seq_type;
        const int runtime_val = std::get<0>(params).runtime_val;
        const std::size_t idx = seq_lookup<Seq>::find(runtime_val);

        if (idx != dispatch_npos) {
            using FunctorT = std::decay_t<Functor>;
            static constexpr auto table = make_dispatch_table<FunctorT, arg_pack<Args...>, R>(Seq{});
            return invoke_table_entry<R>(functor, table[idx], std::forward<Args>(args)...);
        }
        if constexpr (ThrowOnNoMatch) {
            throw no_match_error("poet::dispatch: no matching compile-time combination for runtime inputs");
        } else if constexpr (!std::is_void_v<R>) {
            return R{};
        }
    }

    template<bool ThrowOnNoMatch, typename R, typename Functor, typename ParamTuple, typename... Args>
    POET_FORCEINLINE auto dispatch_nd(Functor &functor, ParamTuple const &params, Args &&...args) -> R {
        const std::size_t flat_idx = extract_flat_index(params);
        if (POET_LIKELY(flat_idx != dispatch_npos)) {
            using sequences_t = decltype(extract_sequences<ParamTuple>());
            static constexpr sequences_t sequences{};

            using FunctorT = std::decay_t<Functor>;
            static constexpr auto table = make_nd_dispatch_table<FunctorT, arg_pack<Args...>, R>(sequences);
            return invoke_table_entry<R>(functor, table[flat_idx], std::forward<Args>(args)...);
        }
        if constexpr (ThrowOnNoMatch) {
            throw no_match_error("poet::dispatch: no matching compile-time combination for runtime inputs");
        } else if constexpr (!std::is_void_v<R>) {
            return R{};
        }
    }

    template<bool ThrowOnNoMatch, typename Functor, typename ParamTuple, typename... Args>
    POET_FORCEINLINE auto dispatch_impl(Functor &functor, ParamTuple const &params, Args &&...args) -> decltype(auto) {
        constexpr std::size_t param_count = std::tuple_size_v<std::remove_reference_t<ParamTuple>>;
        using sequences_t = decltype(extract_sequences<ParamTuple>());
        using result_type = dispatch_result_t<Functor, sequences_t, Args &&...>;

        if constexpr (param_count == 1) {
            return dispatch_1d<ThrowOnNoMatch, result_type>(functor, params, std::forward<Args>(args)...);
        } else {
            return dispatch_nd<ThrowOnNoMatch, result_type>(functor, params, std::forward<Args>(args)...);
        }
    }

}// namespace detail

namespace detail {
    template<typename... Ts> struct leading_param_count;

    template<> struct leading_param_count<> {
        static constexpr std::size_t value = 0;
    };

    template<typename First, typename... Rest> struct leading_param_count<First, Rest...> {
        static constexpr std::size_t value = is_dispatch_param_v<First> ? (1 + leading_param_count<Rest...>::value) : 0;
    };

    template<bool ThrowOnNoMatch, typename Functor, std::size_t... ParamIdx, std::size_t... ArgIdx, typename... All>
    POET_FORCEINLINE auto dispatch_split_impl(Functor &functor,
      std::index_sequence<ParamIdx...> /*p*/,
      std::index_sequence<ArgIdx...> /*a*/,
      All &&...all) -> decltype(auto) {

        constexpr std::size_t num_params = sizeof...(ParamIdx);
        // Reference-tuple view of the entire pack so we can index it twice without copies.
        auto all_refs = std::forward_as_tuple(std::forward<All>(all)...);

        // Leading `num_params` entries are the dispatch_params → copy into a value tuple
        // (they're small structs holding a runtime int).
        auto params = std::make_tuple(std::get<ParamIdx>(all_refs)...);

        // Remaining entries are forwarded with their original value categories preserved
        // via `std::move(all_refs)` (the references inside are unaffected).
        return dispatch_impl<ThrowOnNoMatch>(functor,
          params,
          std::get<num_params + ArgIdx>(
            std::move(all_refs))...);// NOLINT(bugprone-use-after-move,hicpp-invalid-access-moved)
    }

    // Splits the variadic pack into [leading dispatch_params | trailing regular args] by
    // counting dispatch_param types until the first non-dispatch_param — everything after is
    // forwarded as plain args into the chosen specialisation.
    template<bool ThrowOnNoMatch, typename Functor, typename FirstParam, typename... Rest>
    POET_FORCEINLINE auto dispatch_variadic_impl(Functor &functor, FirstParam &&first_param, Rest &&...rest)
      -> decltype(auto) {
        // `first_param` is known to be a dispatch_param (enable_if on the public overload);
        // count contiguous dispatch_params in the rest, the remainder is the regular arg pack.
        constexpr std::size_t num_params = 1 + leading_param_count<Rest...>::value;
        constexpr std::size_t num_args = sizeof...(Rest) + 1 - num_params;

        if constexpr (num_args == 0) {
            auto params = std::make_tuple(std::forward<FirstParam>(first_param), std::forward<Rest>(rest)...);
            return dispatch_impl<ThrowOnNoMatch>(functor, params);
        } else {
            return dispatch_split_impl<ThrowOnNoMatch>(functor,
              std::make_index_sequence<num_params>{},
              std::make_index_sequence<num_args>{},
              std::forward<FirstParam>(first_param),
              std::forward<Rest>(rest)...);
        }
    }
}// namespace detail

/// \brief Dispatches runtime integers to compile-time specializations.
///
/// Accepts either leading `dispatch_param` arguments or a tuple of them. On miss,
/// the non-throwing overload returns `void` or a default-constructed result.
template<typename Functor,
  typename FirstParam,
  typename... Rest,
  std::enable_if_t<detail::is_dispatch_param_v<FirstParam>, int> = 0>
auto dispatch(Functor &&functor,// NOLINT(cppcoreguidelines-missing-std-forward) — accepted as universal ref to avoid
                                // copy; internally always used by lvalue ref
  FirstParam &&first_param,
  Rest &&...rest) -> decltype(auto) {
    return detail::dispatch_variadic_impl<false>(
      functor, std::forward<FirstParam>(first_param), std::forward<Rest>(rest)...);
}

/// \brief Tuple overload for `dispatch_param` dispatch.
template<typename Functor,
  typename ParamTuple,
  typename... Args,
  std::enable_if_t<detail::is_dispatch_param_tuple_v<ParamTuple>, int> = 0>
auto dispatch(Functor &&functor,// NOLINT(cppcoreguidelines-missing-std-forward) — accepted as universal ref to avoid
                                // copy; internally always used by lvalue ref
  ParamTuple const &params,
  Args &&...args) -> decltype(auto) {
    return detail::dispatch_impl<false>(functor, params, std::forward<Args>(args)...);
}

namespace detail {
    template<bool ThrowOnNoMatch, typename Functor, typename TupleList, typename RuntimeTuple, typename... Args>
    auto dispatch_tuples_impl(Functor &&functor,
      TupleList const & /*tl*/,
      const RuntimeTuple &runtime_tuple,
      Args &&...args)// NOLINT(cppcoreguidelines-missing-std-forward) forwarded inside short-circuiting fold
      -> decltype(auto) {
        using TL = std::decay_t<TupleList>;
        static_assert(std::tuple_size_v<TL> >= 1, "tuple list must contain at least one allowed tuple");

        using first_seq = std::tuple_element_t<0, TL>;
        using result_type = typename seq_call_result<first_seq, std::decay_t<Functor>, std::decay_t<Args>...>::type;

        result_holder<result_type> out;

        using FunctorT = std::decay_t<Functor>;
        FunctorT functor_copy(std::forward<Functor>(functor));

        const bool matched = std::apply(
          [&](auto... seqs) POET_ALWAYS_INLINE_LAMBDA -> bool {
              return ([&](auto &seq) POET_ALWAYS_INLINE_LAMBDA -> bool {
                  using SeqType = std::decay_t<decltype(seq)>;
                  auto result = seq_matcher<SeqType, result_type, RuntimeTuple, FunctorT, Args...>::match_and_call(
                    runtime_tuple, functor_copy, std::forward<Args>(args)...);

                  if (result.has_value()) {
                      out = std::move(result);
                      return true;
                  }
                  return false;
              }(seqs) || ...);
          },
          TL{});

        if (matched) {
            if constexpr (std::is_void_v<result_type>) {
                return;
            } else {
                return result_type(std::move(*out));
            }
        }
        if constexpr (ThrowOnNoMatch) {
            throw no_match_error("poet::dispatch_tuples: no matching compile-time tuple for runtime inputs");
        } else if constexpr (!std::is_void_v<result_type>) {
            return result_type{};
        }
    }
}// namespace detail

/// \brief Dispatches using a `dispatch_set`.
template<typename Functor, typename... Tuples, typename... Args>
auto dispatch(Functor &&functor, const dispatch_set<Tuples...> &set, Args &&...args) -> decltype(auto) {
    return detail::dispatch_tuples_impl<false>(std::forward<Functor>(functor),
      typename dispatch_set<Tuples...>::seq_type{},
      set.runtime_tuple(),
      std::forward<Args>(args)...);
}

/// \brief Throwing overload for `dispatch_set` dispatch.
template<typename Functor, typename... Tuples, typename... Args>
auto dispatch(throw_on_no_match_t /*tag*/, Functor &&functor, const dispatch_set<Tuples...> &set, Args &&...args)
  -> decltype(auto) {
    return detail::dispatch_tuples_impl<true>(std::forward<Functor>(functor),
      typename dispatch_set<Tuples...>::seq_type{},
      set.runtime_tuple(),
      std::forward<Args>(args)...);
}

/// \brief Throwing `dispatch_param` overload.
template<typename Functor,
  typename FirstParam,
  typename... Rest,
  std::enable_if_t<detail::is_dispatch_param_v<FirstParam>, int> = 0>
auto dispatch(throw_on_no_match_t /*tag*/,
  Functor &&functor,// NOLINT(cppcoreguidelines-missing-std-forward) — accepted as universal ref to avoid copy;
                    // internally always used by lvalue ref
  FirstParam &&first_param,
  Rest &&...rest) -> decltype(auto) {
    return detail::dispatch_variadic_impl<true>(
      functor, std::forward<FirstParam>(first_param), std::forward<Rest>(rest)...);
}

/// \brief Throwing tuple overload for `dispatch_param` dispatch.
template<typename Functor,
  typename ParamTuple,
  typename... Args,
  std::enable_if_t<detail::is_dispatch_param_tuple_v<ParamTuple>, int> = 0>
auto dispatch(throw_on_no_match_t /*tag*/,
  Functor &&functor,// NOLINT(cppcoreguidelines-missing-std-forward) — accepted as universal ref to avoid copy;
                    // internally always used by lvalue ref
  ParamTuple const &params,
  Args &&...args) -> decltype(auto) {
    return detail::dispatch_impl<true>(functor, params, std::forward<Args>(args)...);
}

}// namespace poet
// END_FILE: dispatch.hpp
/* End inline (angle): dispatch.hpp */
/* Begin inline (angle): static_for.hpp */
// BEGIN_FILE: static_for.hpp

/// \file static_for.hpp
/// \brief Compile-time loop unrolling over integer ranges.

#include <cstddef>
#include <type_traits>
#include <utility>

/* Begin inline (angle): for_utils.hpp */
// BEGIN_FILE: for_utils.hpp

/// \file for_utils.hpp
/// \brief Internal helpers shared by the loop primitives.

#include <cstddef>
#include <type_traits>
#include <utility>

/* Begin inline (angle): macros.hpp */
/* Skipped already inlined: macros.hpp */
/* End inline (angle): macros.hpp */

namespace poet::detail {

template<std::ptrdiff_t Begin, std::ptrdiff_t End, std::ptrdiff_t Step>
[[nodiscard]] POET_CPP20_CONSTEVAL auto compute_range_count() noexcept -> std::size_t {
    static_assert(Step != 0, "static_for requires a non-zero step");
    if constexpr (Step > 0) {
        static_assert(Begin <= End, "static_for with a positive step requires Begin <= End");
    } else {
        static_assert(Begin >= End, "static_for with a negative step requires Begin >= End");
    }
    if constexpr (Begin == End) { return 0; }
    constexpr auto distance = (End - Begin) < 0 ? -(End - Begin) : (End - Begin);
    constexpr auto magnitude = Step < 0 ? -Step : Step;
    return static_cast<std::size_t>((distance + magnitude - 1) / magnitude);
}

template<typename Func, std::ptrdiff_t Begin, std::ptrdiff_t Step, std::size_t StartIndex, std::size_t... Is>
POET_FORCEINLINE constexpr auto run_block(Func &func, std::index_sequence<Is...> /*seq*/) -> void {
    constexpr std::ptrdiff_t Base = Begin + (Step * static_cast<std::ptrdiff_t>(StartIndex));
    (func(std::integral_constant<std::ptrdiff_t, Base + (Step * static_cast<std::ptrdiff_t>(Is))>{}), ...);
}

template<typename Func, std::ptrdiff_t Begin, std::ptrdiff_t Step, std::size_t StartIndex, std::size_t... Is>
POET_NOINLINE_FLATTEN constexpr auto run_block_iso(Func &func, std::index_sequence<Is...> /*seq*/) -> void {
    constexpr std::ptrdiff_t Base = Begin + (Step * static_cast<std::ptrdiff_t>(StartIndex));
    (func(std::integral_constant<std::ptrdiff_t, Base + (Step * static_cast<std::ptrdiff_t>(Is))>{}), ...);
}

template<typename Func, std::ptrdiff_t Begin, std::ptrdiff_t Step, std::size_t BlockSize, std::size_t... Is>
POET_FORCEINLINE constexpr auto emit_blocks(Func &func, std::index_sequence<Is...> /*seq*/) -> void {
    (run_block<Func, Begin, Step, Is * BlockSize>(func, std::make_index_sequence<BlockSize>{}), ...);
}

template<typename Func, std::ptrdiff_t Begin, std::ptrdiff_t Step, std::size_t BlockSize, std::size_t... Is>
POET_FORCEINLINE constexpr auto emit_blocks_iso(Func &func, std::index_sequence<Is...> /*seq*/) -> void {
    (run_block_iso<Func, Begin, Step, Is * BlockSize>(func, std::make_index_sequence<BlockSize>{}), ...);
}

template<typename Functor> struct template_invoker {
    Functor &functor;

    template<std::ptrdiff_t Value>
    POET_FORCEINLINE constexpr auto operator()(std::integral_constant<std::ptrdiff_t, Value> /*ic*/) const -> void {
        functor.template operator()<Value>();
    }
};

}// namespace poet::detail
// END_FILE: for_utils.hpp
/* End inline (angle): for_utils.hpp */

namespace poet {

namespace detail {

    template<typename Callable,
      std::ptrdiff_t Begin,
      std::ptrdiff_t Step,
      std::size_t BlockSize,
      std::size_t FullBlocks,
      std::size_t Remainder>
    POET_FORCEINLINE constexpr void run_blocks(Callable &callable) {
        if constexpr (FullBlocks > 0) {
            if constexpr (FullBlocks > 1) {
                emit_blocks_iso<Callable, Begin, Step, BlockSize>(callable, std::make_index_sequence<FullBlocks>{});
            } else {
                emit_blocks<Callable, Begin, Step, BlockSize>(callable, std::make_index_sequence<FullBlocks>{});
            }
        }

        if constexpr (Remainder > 0) {
            run_block<Callable, Begin, Step, FullBlocks * BlockSize>(callable, std::make_index_sequence<Remainder>{});
        }
    }

    template<std::ptrdiff_t Begin, std::ptrdiff_t End, std::ptrdiff_t Step>
    POET_CPP20_CONSTEVAL auto default_block_size() noexcept -> std::size_t {
        constexpr auto count = detail::compute_range_count<Begin, End, Step>();
        return count == 0 ? 1 : count;
    }

}// namespace detail

/// \brief Runs a compile-time unrolled loop over `[Begin, End)`.
///
/// `func` may take `std::integral_constant<std::ptrdiff_t, I>` or expose
/// `template <auto I> operator()()`. `BlockSize` defaults to the full range;
/// pass a smaller value to isolate heavier bodies into separate outlined blocks.
///
/// \tparam Begin Initial value of the range.
/// \tparam End Exclusive terminator of the range.
/// \tparam Step Increment applied between iterations (defaults to `1`).
/// \tparam BlockSize Number of iterations expanded per block (defaults to the
///                   total iteration count, or `1` for empty ranges).
/// \tparam Func Callable type.
/// \param func Callable instance invoked once per iteration.
template<std::ptrdiff_t Begin,
  std::ptrdiff_t End,
  std::ptrdiff_t Step = 1,
  std::size_t BlockSize = detail::default_block_size<Begin, End, Step>(),
  typename Func>
POET_FORCEINLINE constexpr void static_for(Func &&func) {
    static_assert(BlockSize > 0, "static_for requires BlockSize > 0");

    constexpr auto count = detail::compute_range_count<Begin, End, Step>();
    if constexpr (count == 0) { return; }

    constexpr auto full_blocks = count / BlockSize;
    constexpr auto remainder = count % BlockSize;

    using callable_t = std::remove_reference_t<Func>;

    auto do_for = [&](auto &ref) POET_ALWAYS_INLINE_LAMBDA -> void {
        if constexpr (std::is_invocable_v<callable_t &, std::integral_constant<std::ptrdiff_t, Begin>>) {
            detail::run_blocks<callable_t, Begin, Step, BlockSize, full_blocks, remainder>(ref);
        } else {
            using invoker_t = detail::template_invoker<callable_t>;
            invoker_t invoker{ ref };
            detail::run_blocks<invoker_t, Begin, Step, BlockSize, full_blocks, remainder>(invoker);
        }
    };

    if constexpr (std::is_lvalue_reference_v<Func>) {
        do_for(func);
    } else {
        callable_t callable(std::forward<Func>(func));
        do_for(callable);
    }
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// \brief Convenience overload for `static_for<0, End>(func)`.
template<std::ptrdiff_t End, typename Func> POET_FORCEINLINE constexpr void static_for(Func &&func) {
    static_for<0, End>(std::forward<Func>(func));
}
#endif

}// namespace poet
// END_FILE: static_for.hpp
/* End inline (angle): static_for.hpp */
/* Begin inline (angle): undef_macros.hpp */
// BEGIN_FILE: undef_macros.hpp
// NOLINTBEGIN(llvm-header-guard)
// NOLINTEND(llvm-header-guard)

/// \file undef_macros.hpp
/// \brief Undefines all POET macros to prevent namespace pollution.
///
/// The umbrella header `<poet/poet.hpp>` includes this header automatically as
/// its last include, so macros are cleaned up by default.  If you include
/// individual POET headers instead, you can include this header manually after
/// all code that uses POET macros.
///
/// POET defines several utility macros for portability and optimization:
/// - POET_UNREACHABLE: Marks unreachable code paths
/// - POET_FORCEINLINE: Forces function inlining
/// - POET_ALWAYS_INLINE_LAMBDA: Forces lambda call-operator inlining
/// - POET_NOINLINE_FLATTEN: noinline+flatten for register-isolated blocks
/// - POET_HOT_LOOP: Hot path optimization with aggressive inlining
/// - POET_LIKELY / POET_UNLIKELY: Branch prediction hints
/// - POET_ASSUME: Compiler assumption hint
/// - POET_CPP20_CONSTEVAL: Feature detection
/// - poet_count_trailing_zeros: (function, not macro — unaffected)
///
/// **Usage with individual headers:**
///
/// ```cpp
/// #include <poet/core/static_for.hpp>
///
/// void my_poet_code() {
///     if (POET_LIKELY(condition)) {
///         // ...
///     }
/// }
///
/// // Clean up macro namespace before including other headers
/// #include <poet/core/undef_macros.hpp>
/// ```
///
/// **Re-include to restore macros:**
/// After this header is included, re-including <poet/core/macros.hpp> will
/// redefine all POET macros.
///
/// **Important Notes:**
/// 1. Include this header ONLY after all code that uses POET macros.
/// 2. The poet_count_trailing_zeros function remains available (it's not a macro).
/// 3. Template-based POET utilities (static_for, dynamic_for, dispatch) are unaffected.

// Re-arm macros.hpp so a subsequent #include <poet/core/macros.hpp> redefines
// everything.
#ifdef POET_CORE_MACROS_HPP
#undef POET_CORE_MACROS_HPP
#endif

// ============================================================================
// Undefine POET_UNREACHABLE
// ============================================================================
#ifdef POET_UNREACHABLE
#undef POET_UNREACHABLE
#endif

// ============================================================================
// Undefine POET_FORCEINLINE
// ============================================================================
#ifdef POET_FORCEINLINE
#undef POET_FORCEINLINE
#endif

// ============================================================================
// Undefine POET_ALWAYS_INLINE_LAMBDA
// ============================================================================
#ifdef POET_ALWAYS_INLINE_LAMBDA
#undef POET_ALWAYS_INLINE_LAMBDA
#endif

// ============================================================================
// Undefine POET_NOINLINE_FLATTEN
// ============================================================================
#ifdef POET_NOINLINE_FLATTEN
#undef POET_NOINLINE_FLATTEN
#endif

// ============================================================================
// Undefine POET_LIKELY / POET_UNLIKELY
// ============================================================================
#ifdef POET_LIKELY
#undef POET_LIKELY
#endif

#ifdef POET_UNLIKELY
#undef POET_UNLIKELY
#endif

// ============================================================================
// Undefine POET_ASSUME
// ============================================================================
#ifdef POET_ASSUME
#undef POET_ASSUME
#endif

// ============================================================================
// Undefine POET_HOT_LOOP
// ============================================================================
#ifdef POET_HOT_LOOP
#undef POET_HOT_LOOP
#endif

// ============================================================================
// Undefine POET_HIGH_OPTIMIZATION
// ============================================================================
#ifdef POET_HIGH_OPTIMIZATION
#undef POET_HIGH_OPTIMIZATION
#endif

// ============================================================================
// Undefine C++20/C++23 feature detection macros
// ============================================================================
#ifdef POET_CPP20_CONSTEVAL
#undef POET_CPP20_CONSTEVAL
#endif

// END_FILE: undef_macros.hpp
/* End inline (angle): undef_macros.hpp */
// NOLINTEND(llvm-include-order)
// clang-format on
// END_FILE: poet.hpp
/* End inline (angle): poet.hpp */

/* Begin inline (quoted): dispatch_arch.hpp */
// BEGIN_FILE: dispatch_arch.hpp

#include <type_traits>
#include <xsimd/xsimd.hpp>

namespace simdrng {

// Dispatch arch list, selected at compile time via the xsimd inheritance hierarchy.
// x86:     best_arch derives from sse2  -> dispatch across avx512bw / avx2 / sse2
// aarch64: best_arch derives from neon  -> dispatch to neon64
// riscv64: fallback                     -> dispatch to rvv (128-bit fixed VLEN)
//
// Note: xsimd::rvv is detail::rvv<__riscv_v_fixed_vlen> when RVV is active,
// but detail::rvv<0xFFFFFFFF> as a fallback sentinel otherwise.  The dispatch
// TU is compiled with -march=rv64gcv -mrvv-vector-bits=zvl (VLEN=128), so we
// must use the concrete rvv<128> to match the exported symbols.
using rvv128 = xsimd::detail::rvv<128>;

// xsimd's arch hierarchy splits the x86 family: avx* derive from `common`,
// and avx512* derive from `common` independently (avx512bw : avx512dq :
// avx512cd : avx512f : common). So avx512bw is NOT a descendant of sse2 or
// avx. Check all three roots to cover every x86 ISA variant.
//
// The AVX-512 variant targets avx512bw (the x86-64-v4 floor: F+CD+BW+DQ+VL).
// Compared to plain avx512f this unlocks vpmullq (AVX512DQ, used by Philox
// mulhilo) and byte/word permutes (AVX512BW, used by ChaCha shuffles).
// avx512bw derives from avx512f, so the is_base_of_v<avx512f, ...> root check
// below is still correct.
using dispatch_arch_list =
    std::conditional_t<std::is_base_of_v<xsimd::sse2, xsimd::best_arch> ||
                           std::is_base_of_v<xsimd::avx, xsimd::best_arch> ||
                           std::is_base_of_v<xsimd::avx512f, xsimd::best_arch>,
                       xsimd::arch_list<xsimd::avx512bw, xsimd::avx2, xsimd::sse2>,
                       std::conditional_t<std::is_base_of_v<xsimd::neon, xsimd::best_arch>,
                                          xsimd::arch_list<xsimd::neon64>, xsimd::arch_list<rvv128>>>;

} // namespace simdrng
// END_FILE: dispatch_arch.hpp
/* End inline (quoted): dispatch_arch.hpp */
/* Begin inline (quoted): macros.hpp */
/* Skipped already inlined: macros.hpp */
/* End inline (quoted): macros.hpp */

namespace simdrng {

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
  static constexpr std::uint8_t SIMD_WIDTH_MASK = SIMD_WIDTH > 0 ? static_cast<std::uint8_t>(SIMD_WIDTH - 1) : 0;
  static constexpr std::uint8_t BLOCK_SEGMENTCOUNT = SIMD_WIDTH > 0
                                                         ? static_cast<std::uint8_t>(MATRIX_WORDCOUNT / SIMD_WIDTH)
                                                         : 0;
  static constexpr std::uint8_t cache_batchcount() noexcept {
    // Use 2 batches for 512-bit+ SIMD (fewer but wider blocks per batch).
    if constexpr (simd_type::size >= 16) {
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

  explicit SIMDRNG_ALWAYS_INLINE ChaChaState(const std::array<matrix_word, KEY_WORDCOUNT> &key,
                                             const input_word counter, const input_word nonce) {
    m_state[0] = 0x61707865;
    m_state[1] = 0x3320646e;
    m_state[2] = 0x79622d32;
    m_state[3] = 0x6b206574;

    poet::static_for<0, KEY_WORDCOUNT>([&](auto I) { m_state[4 + I] = key[I]; });

    m_state[12] = static_cast<matrix_word>(counter & 0xFFFFFFFF);
    m_state[13] = static_cast<matrix_word>(counter >> 32);
    m_state[14] = static_cast<matrix_word>(nonce & 0xFFFFFFFF);
    m_state[15] = static_cast<matrix_word>(nonce >> 32);
  }

  SIMDRNG_ALWAYS_INLINE matrix_type getState(bool prev) const noexcept {
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

  SIMDRNG_ALWAYS_INLINE matrix_type next_block() noexcept {
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
  static constexpr std::array<matrix_word, SIMD_WIDTH> LANE_OFFSETS = [] {
    std::array<matrix_word, SIMD_WIDTH> offsets{};
    poet::static_for<0, SIMD_WIDTH>([&](auto I) { offsets[I] = static_cast<matrix_word>(I.value); });
    return offsets;
  }();

  SIMDRNG_ALWAYS_INLINE static simd_type make_higher_counter_inc(matrix_word overflow_index) noexcept {
    if (overflow_index >= SIMD_WIDTH) [[likely]] {
      return simd_type::broadcast(0);
    }

    std::array<matrix_word, SIMD_WIDTH> incs{};
    poet::static_for<1, SIMD_WIDTH>(
        [&](auto I) { incs[I] = static_cast<matrix_word>(overflow_index < static_cast<matrix_word>(I.value)); });
    return simd_type::load_unaligned(incs.data());
  }

  SIMDRNG_ALWAYS_INLINE static void init_state_batches(working_state_type &x, const matrix_type &state,
                                                       simd_type lower_counter_inc,
                                                       simd_type higher_counter_inc) noexcept {
    poet::static_for<0, MATRIX_WORDCOUNT>([&](auto I) { x[I] = simd_type::broadcast(state[I]); });
    x[12] += lower_counter_inc;
    x[13] += higher_counter_inc;
  }

  SIMDRNG_ALWAYS_INLINE static void add_original_state(working_state_type &x, const matrix_type &state,
                                                       simd_type lower_counter_inc,
                                                       simd_type higher_counter_inc) noexcept {
    poet::static_for<0, MATRIX_WORDCOUNT>([&](auto I) { x[I] += simd_type::broadcast(state[I]); });
    x[12] += lower_counter_inc;
    x[13] += higher_counter_inc;
  }

  static void transpose_into_cache(cache_batch_type &cache, working_state_type &x) noexcept {
    auto *SIMDRNG_RESTRICT cache_lanes = cache.data();
    auto *SIMDRNG_RESTRICT working = x.data();
    poet::static_for<0, BLOCK_SEGMENTCOUNT>([&](auto Seg) {
      auto *SIMDRNG_RESTRICT segment_begin = working + Seg * SIMD_WIDTH;
      xsimd::transpose(segment_begin, segment_begin + SIMD_WIDTH);
      poet::static_for<0, SIMD_WIDTH>([&](auto Lane) { cache_lanes[Lane][Seg] = segment_begin[Lane]; });
    });
  }

  SIMDRNG_ALWAYS_INLINE static constexpr void advance_counter(matrix_type &state) noexcept {
    state[12] += SIMD_WIDTH;
    state[13] += static_cast<matrix_word>(state[12] < SIMD_WIDTH);
  }

  template <unsigned A, unsigned B, unsigned C, unsigned D>
  SIMDRNG_ALWAYS_INLINE static void quarter_round(working_state_type &x) noexcept {
    x[A] += x[B];
    x[D] ^= x[A];
    x[D] = xsimd::rotl<16>(x[D]);
    x[C] += x[D];
    x[B] ^= x[C];
    x[B] = xsimd::rotl<12>(x[B]);
    x[A] += x[B];
    x[D] ^= x[A];
    x[D] = xsimd::rotl<8>(x[D]);
    x[C] += x[D];
    x[B] ^= x[C];
    x[B] = xsimd::rotl<7>(x[B]);
  }

  SIMDRNG_ALWAYS_INLINE static void gen_block_batch(cache_batch_type &cache, const matrix_type &state) noexcept {
    const simd_type lower_counter_inc = simd_type::load_unaligned(LANE_OFFSETS.data());
    matrix_word overflow_index = std::numeric_limits<matrix_word>::max() - state[12];
    const simd_type higher_counter_inc = make_higher_counter_inc(overflow_index);

    working_state_type x;
    init_state_batches(x, state, lower_counter_inc, higher_counter_inc);

    poet::static_for<0, R / 2>([&](auto) {
      // Column round: QR(i, i+4, i+8, i+12)
      poet::static_for<0, 4>([&](auto I) {
        constexpr auto i = static_cast<unsigned>(I.value);
        quarter_round<i, i + 4, i + 8, i + 12>(x);
      });
      // Diagonal round: QR(i, ((i+1)%4)+4, ((i+2)%4)+8, ((i+3)%4)+12)
      poet::static_for<0, 4>([&](auto I) {
        constexpr auto i = static_cast<unsigned>(I.value);
        quarter_round<i, ((i + 1) % 4) + 4, ((i + 2) % 4) + 8, ((i + 3) % 4) + 12>(x);
      });
    });

    add_original_state(x, state, lower_counter_inc, higher_counter_inc);
    transpose_into_cache(cache, x);
  }

  SIMDRNG_ALWAYS_INLINE constexpr void gen_next_blocks_in_cache() noexcept {
    auto state = m_state;
    poet::static_for<0, CACHE_BATCHCOUNT>([&](auto Batch) {
      gen_block_batch(m_cache[Batch], state);
      advance_counter(state);
    });
    m_state = state;
  }
};

/**
 * Result from the runtime dispatch initialization for ChaChaSIMD.
 */
struct ChaChaSIMDInitResult {
  using matrix_type = std::array<std::uint32_t, 16>;
  using next_block_fn = matrix_type (*)(void *SIMDRNG_RESTRICT) noexcept;
  using get_state_fn = matrix_type (*)(const void *, bool) noexcept;
  using set_state_fn = void (*)(void *, const matrix_type &) noexcept;
  using get_cache_index_fn = std::uint8_t (*)(const void *) noexcept;
  next_block_fn next_block;
  get_state_fn get_state;
  set_state_fn set_state;
  get_cache_index_fn get_cache_index;
  std::size_t simd_size;
};

/**
 * Functor used by xsimd::dispatch to initialize a ChaChaSIMD instance.
 */
template <std::uint8_t R> struct ChaChaSIMDInitFunctor {
  void *state_storage;
  const std::array<std::uint32_t, 8> key;
  const std::uint64_t counter, nonce;

  template <class Arch> ChaChaSIMDInitResult operator()(Arch /*arch*/) const noexcept;
};

template <std::uint8_t R>
template <class Arch>
ChaChaSIMDInitResult ChaChaSIMDInitFunctor<R>::operator()(Arch /*arch*/) const noexcept {
  using State = ChaChaState<Arch, R>;
  static_assert(sizeof(State) <= 2176, "ChaChaState exceeds StateStorage capacity");
  static_assert(alignof(State) <= 64, "ChaChaState exceeds StateStorage alignment");
  std::construct_at(static_cast<State *>(state_storage), key, counter, nonce);
  return {
      +[](void *SIMDRNG_RESTRICT s) noexcept -> ChaChaSIMDInitResult::matrix_type {
        return static_cast<State *>(s)->next_block();
      },
      +[](const void *s, bool prev) noexcept -> ChaChaSIMDInitResult::matrix_type {
        return static_cast<const State *>(s)->getState(prev);
      },
      +[](void *s, const ChaChaSIMDInitResult::matrix_type &matrix) noexcept {
        auto *state = static_cast<State *>(s);
        state->m_state = matrix;
        state->m_cache_index = State::CACHE_BLOCKCOUNT;
      },
      +[](const void *s) noexcept -> std::uint8_t { return static_cast<const State *>(s)->m_cache_index; },
      std::size_t{State::SIMD_WIDTH},
  };
}

#define SIMDRNG_CHACHA_EXTERN_TEMPLATE(R, Arch)                                                                        \
  extern template SIMDRNG_LOCAL ChaChaSIMDInitResult ChaChaSIMDInitFunctor<R>::operator()<Arch>(Arch) const noexcept

#if SIMDRNG_ARCH_X86_64
SIMDRNG_CHACHA_EXTERN_TEMPLATE(8, xsimd::sse2);
SIMDRNG_CHACHA_EXTERN_TEMPLATE(12, xsimd::sse2);
SIMDRNG_CHACHA_EXTERN_TEMPLATE(20, xsimd::sse2);
SIMDRNG_CHACHA_EXTERN_TEMPLATE(8, xsimd::avx2);
SIMDRNG_CHACHA_EXTERN_TEMPLATE(12, xsimd::avx2);
SIMDRNG_CHACHA_EXTERN_TEMPLATE(20, xsimd::avx2);
SIMDRNG_CHACHA_EXTERN_TEMPLATE(8, xsimd::avx512bw);
SIMDRNG_CHACHA_EXTERN_TEMPLATE(12, xsimd::avx512bw);
SIMDRNG_CHACHA_EXTERN_TEMPLATE(20, xsimd::avx512bw);
#elif SIMDRNG_ARCH_AARCH64
SIMDRNG_CHACHA_EXTERN_TEMPLATE(8, xsimd::neon64);
SIMDRNG_CHACHA_EXTERN_TEMPLATE(12, xsimd::neon64);
SIMDRNG_CHACHA_EXTERN_TEMPLATE(20, xsimd::neon64);
#if XSIMD_WITH_SVE
SIMDRNG_CHACHA_EXTERN_TEMPLATE(8, xsimd::sve);
SIMDRNG_CHACHA_EXTERN_TEMPLATE(12, xsimd::sve);
SIMDRNG_CHACHA_EXTERN_TEMPLATE(20, xsimd::sve);
#endif
#elif SIMDRNG_ARCH_RISCV64
SIMDRNG_CHACHA_EXTERN_TEMPLATE(8, xsimd::detail::rvv<128>);
SIMDRNG_CHACHA_EXTERN_TEMPLATE(12, xsimd::detail::rvv<128>);
SIMDRNG_CHACHA_EXTERN_TEMPLATE(20, xsimd::detail::rvv<128>);
#endif

#undef SIMDRNG_CHACHA_EXTERN_TEMPLATE

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

  static constexpr SIMDRNG_ALWAYS_INLINE auto(min)() noexcept { return (std::numeric_limits<result_type>::min)(); }
  static constexpr SIMDRNG_ALWAYS_INLINE auto(max)() noexcept { return (std::numeric_limits<result_type>::max)(); }

  static constexpr SIMDRNG_ALWAYS_INLINE matrix_type results_to_block(const result_cache_type &results) noexcept {
    return std::bit_cast<matrix_type>(results);
  }

  static constexpr SIMDRNG_ALWAYS_INLINE result_cache_type block_to_results(const matrix_type &block) noexcept {
    return std::bit_cast<result_cache_type>(block);
  }

  static constexpr std::array<matrix_word, KEY_WORDCOUNT> seed_to_key(result_type seed) noexcept {
    std::array<matrix_word, KEY_WORDCOUNT> key{};
    // SplitMix64 expansion: 1 uint64 seed -> 4 uint64 -> 8 uint32
    auto state = seed;
    for (std::uint8_t i = 0; i < 4; ++i) {
      state += 0x9e3779b97f4a7c15ULL;
      auto z = state;
      z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
      z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
      z = z ^ (z >> 31);
      key[std::size_t{i} * 2] = static_cast<matrix_word>(z);
      key[std::size_t{i} * 2 + 1] = static_cast<matrix_word>(z >> 32);
    }
    return key;
  }

  explicit SIMDRNG_ALWAYS_INLINE ChaChaSIMD(result_type seed, const input_word counter = 0, const input_word nonce = 0)
      : ChaChaSIMD(seed_to_key(seed), counter, nonce) {}

  explicit SIMDRNG_ALWAYS_INLINE ChaChaSIMD(const std::array<matrix_word, KEY_WORDCOUNT> &key, const input_word counter,
                                            const input_word nonce) {
    auto result =
        xsimd::dispatch<dispatch_arch_list>(internal::ChaChaSIMDInitFunctor<R>{m_state.data, key, counter, nonce})();
    m_next_block = result.next_block;
    m_get_state = result.get_state;
    m_set_state = result.set_state;
    m_get_cache_index = result.get_cache_index;
    m_simd_size = result.simd_size;
  }

  SIMDRNG_ALWAYS_INLINE constexpr result_type operator()() noexcept {
    if (m_result_index >= m_result_cache.size()) [[unlikely]] {
      m_result_cache = block_to_results(m_next_block(m_state.data));
      m_result_index = 0;
    }
    return m_result_cache[m_result_index++];
  }

  SIMDRNG_ALWAYS_INLINE constexpr double uniform() noexcept {
    return static_cast<double>(operator()() >> 11) * 0x1.0p-53;
  }

  // Bulk fill, bit-identical to n consecutive operator() calls. Drains the
  // partial result-cache prefix, then refills a block (8 results) at a time via
  // the dispatched next_block and copies straight to `out`, always parking the
  // last block in m_result_cache and the cursor so the next operator() resumes
  // the exact stream.
  void generate(result_type *SIMDRNG_RESTRICT out, std::size_t n) noexcept {
    constexpr std::size_t sz = MATRIX_WORDCOUNT / 2; // results per block (8)
    std::size_t produced = 0;
    std::uint8_t ri = m_result_index;
    if (ri < sz) {
      const std::size_t avail = sz - ri;
      const std::size_t take = avail < n ? avail : n;
      std::memcpy(out, m_result_cache.data() + ri, take * sizeof(result_type));
      ri = static_cast<std::uint8_t>(ri + take);
      produced = take;
    }
    while (produced < n) {
      m_result_cache = block_to_results(m_next_block(m_state.data));
      const std::size_t rem = n - produced;
      const std::size_t take = rem < sz ? rem : sz;
      std::memcpy(out + produced, m_result_cache.data(), take * sizeof(result_type));
      produced += take;
      ri = static_cast<std::uint8_t>(take); // sz -> cursor at end (exhausted)
    }
    m_result_index = ri;
  }

  void fill_uniform(double *SIMDRNG_RESTRICT out, std::size_t n) noexcept {
    constexpr std::size_t kChunk = 256;
    alignas(64) std::array<result_type, kChunk> buf;
    std::size_t done = 0;
    while (done < n) {
      const std::size_t take = (n - done) < kChunk ? (n - done) : kChunk;
      generate(buf.data(), take);
      for (std::size_t i = 0; i < take; ++i)
        out[done + i] = static_cast<double>(buf[i] >> 11) * 0x1.0p-53;
      done += take;
    }
  }

  SIMDRNG_ALWAYS_INLINE constexpr matrix_type block() noexcept {
    if (m_result_index < m_result_cache.size()) {
      auto cached_block = results_to_block(m_result_cache);
      m_result_index = static_cast<std::uint8_t>(m_result_cache.size());
      return cached_block;
    }
    return m_next_block(m_state.data);
  }

  SIMDRNG_ALWAYS_INLINE constexpr matrix_type getState() const noexcept {
    return m_get_state(m_state.data, m_result_index < m_result_cache.size());
  }

  matrix_type getStateForSerde() const noexcept { return m_get_state(m_state.data, false); }

  void setState(const matrix_type &matrix) noexcept { m_set_state(m_state.data, matrix); }

  const result_cache_type &result_cache() const noexcept { return m_result_cache; }
  void set_result_cache(const result_cache_type &cache) noexcept { m_result_cache = cache; }
  std::uint8_t result_index() const noexcept { return m_result_index; }
  void set_result_index(std::uint8_t idx) noexcept { m_result_index = idx; }

  SIMDRNG_ALWAYS_INLINE size_t getSIMDSize() const noexcept { return m_simd_size; }

private:
  using next_block_fn = internal::ChaChaSIMDInitResult::next_block_fn;
  using get_state_fn = internal::ChaChaSIMDInitResult::get_state_fn;
  using set_state_fn = internal::ChaChaSIMDInitResult::set_state_fn;
  using get_cache_index_fn = internal::ChaChaSIMDInitResult::get_cache_index_fn;

  // Raw byte storage for the arch-specific ChaChaState.
  // Typed union is not viable: xsimd batch types have different sizeof
  // across TUs compiled with different -march flags (ODR divergence).
  // Max is avx512bw: 64B state + 2048B cache + 1B index + padding = 2176 bytes.
  struct StateStorage {
    static constexpr std::size_t SIZE = 2176;
    static constexpr std::size_t ALIGN = 64;
    alignas(ALIGN) unsigned char data[SIZE];
  };

  // Value-initialised so the type-erased storage is never read uninitialised
  // (the per-arch State is written into m_state.data by the dispatch functor in
  // the constructor body). Zeroing 2176 B once per construction is negligible
  // next to the dispatch call that follows.
  alignas(64) StateStorage m_state{};
  next_block_fn m_next_block = nullptr;
  get_state_fn m_get_state = nullptr;
  set_state_fn m_set_state = nullptr;
  get_cache_index_fn m_get_cache_index = nullptr;
  std::size_t m_simd_size = 0;
  result_cache_type m_result_cache{};
  std::uint8_t m_result_index = static_cast<std::uint8_t>(m_result_cache.size());
};

#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE
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

  static constexpr SIMDRNG_ALWAYS_INLINE auto(min)() noexcept { return (std::numeric_limits<result_type>::min)(); }
  static constexpr SIMDRNG_ALWAYS_INLINE auto(max)() noexcept { return (std::numeric_limits<result_type>::max)(); }

  explicit ChaChaNative(result_type seed, const input_word counter = 0, const input_word nonce = 0)
      : ChaChaNative(ChaChaSIMD<R>::seed_to_key(seed), counter, nonce) {}

  ChaChaNative(const std::array<matrix_word, KEY_WORDCOUNT> &key, const input_word counter, const input_word nonce)
      : m_state(key, counter, nonce) {}

  SIMDRNG_ALWAYS_INLINE constexpr result_type operator()() noexcept {
    if (m_result_index >= m_result_cache.size()) [[unlikely]] {
      m_result_cache = ChaChaSIMD<R>::block_to_results(m_state.next_block());
      m_result_index = 0;
    }
    return m_result_cache[m_result_index++];
  }

  SIMDRNG_ALWAYS_INLINE constexpr double uniform() noexcept {
    return static_cast<double>(operator()() >> 11) * 0x1.0p-53;
  }

  // Bulk fill, bit-identical to n consecutive operator() calls (see ChaChaSIMD).
  void generate(result_type *SIMDRNG_RESTRICT out, std::size_t n) noexcept {
    constexpr std::size_t sz = MATRIX_WORDCOUNT / 2;
    std::size_t produced = 0;
    std::uint8_t ri = m_result_index;
    if (ri < sz) {
      const std::size_t avail = sz - ri;
      const std::size_t take = avail < n ? avail : n;
      std::memcpy(out, m_result_cache.data() + ri, take * sizeof(result_type));
      ri = static_cast<std::uint8_t>(ri + take);
      produced = take;
    }
    while (produced < n) {
      m_result_cache = ChaChaSIMD<R>::block_to_results(m_state.next_block());
      const std::size_t rem = n - produced;
      const std::size_t take = rem < sz ? rem : sz;
      std::memcpy(out + produced, m_result_cache.data(), take * sizeof(result_type));
      produced += take;
      ri = static_cast<std::uint8_t>(take);
    }
    m_result_index = ri;
  }

  void fill_uniform(double *SIMDRNG_RESTRICT out, std::size_t n) noexcept {
    constexpr std::size_t kChunk = 256;
    alignas(64) std::array<result_type, kChunk> buf;
    std::size_t done = 0;
    while (done < n) {
      const std::size_t take = (n - done) < kChunk ? (n - done) : kChunk;
      generate(buf.data(), take);
      for (std::size_t i = 0; i < take; ++i)
        out[done + i] = static_cast<double>(buf[i] >> 11) * 0x1.0p-53;
      done += take;
    }
  }

  SIMDRNG_ALWAYS_INLINE constexpr matrix_type block() noexcept {
    if (m_result_index < m_result_cache.size()) {
      auto cached_block = ChaChaSIMD<R>::results_to_block(m_result_cache);
      m_result_index = static_cast<std::uint8_t>(m_result_cache.size());
      return cached_block;
    }
    return m_state.next_block();
  }

  SIMDRNG_ALWAYS_INLINE constexpr matrix_type getState() const noexcept {
    return m_state.getState(m_result_index < m_result_cache.size());
  }

  matrix_type getStateForSerde() const noexcept { return m_state.getState(false); }

  void setState(const matrix_type &matrix) noexcept {
    m_state.m_state = matrix;
    m_state.m_cache_index = State::CACHE_BLOCKCOUNT;
  }

  const result_cache_type &result_cache() const noexcept { return m_result_cache; }
  void set_result_cache(const result_cache_type &cache) noexcept { m_result_cache = cache; }
  std::uint8_t result_index() const noexcept { return m_result_index; }
  void set_result_index(std::uint8_t idx) noexcept { m_result_index = idx; }

  SIMDRNG_ALWAYS_INLINE size_t getSIMDSize() const noexcept { return std::size_t{State::SIMD_WIDTH}; }

private:
  using State = internal::ChaChaState<xsimd::best_arch, R>;
  State m_state;
  result_cache_type m_result_cache{};
  std::uint8_t m_result_index = static_cast<std::uint8_t>(m_result_cache.size());
};
#endif // XSIMD_NO_SUPPORTED_ARCHITECTURE

// Convenience aliases for common ChaCha variants.
using ChaCha8SIMD = ChaChaSIMD<8>;
using ChaCha12SIMD = ChaChaSIMD<12>;
using ChaCha20SIMD = ChaChaSIMD<20>;

#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE
using ChaCha8Native = ChaChaNative<8>;
using ChaCha12Native = ChaChaNative<12>;
using ChaCha20Native = ChaChaNative<20>;
#endif // XSIMD_NO_SUPPORTED_ARCHITECTURE

} // namespace simdrng
// END_FILE: chacha_simd.hpp
/* End inline (quoted): chacha_simd.hpp */
/* Begin inline (quoted): philox_simd.hpp */
// BEGIN_FILE: philox_simd.hpp

#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>

/* Begin inline (angle): poet.hpp */
/* Skipped already inlined: poet.hpp */
/* End inline (angle): poet.hpp */

/* Begin inline (quoted): dispatch_arch.hpp */
/* Skipped already inlined: dispatch_arch.hpp */
/* End inline (quoted): dispatch_arch.hpp */
/* Begin inline (quoted): macros.hpp */
/* Skipped already inlined: macros.hpp */
/* End inline (quoted): macros.hpp */
/* Begin inline (quoted): philox.hpp */
/* Skipped already inlined: philox.hpp */
/* End inline (quoted): philox.hpp */

namespace simdrng {

namespace internal {

template <class Arch, std::uint8_t N, std::uint8_t W, std::uint8_t R> struct PhiloxState {
  static_assert(N == 2 || N == 4, "Philox N must be 2 or 4");
  static_assert(W == 32 || W == 64, "Philox W must be 32 or 64");

  using word_type = std::conditional_t<W == 32, std::uint32_t, std::uint64_t>;
  using counter_type = std::array<word_type, N>;
  using key_type = std::array<word_type, N / 2>;
  using result_type = std::uint64_t;
  static constexpr auto RESULTS_PER_BLOCK = std::uint8_t{N * W / 64};

  using simd_type = xsimd::batch<word_type, Arch>;
  static constexpr std::uint8_t SIMD_WIDTH = std::uint8_t{simd_type::size};
  static_assert(SIMD_WIDTH == 0 || std::has_single_bit(static_cast<unsigned int>(SIMD_WIDTH)),
                "Philox SIMD width must be a power of two");

  static constexpr std::uint16_t CACHE_SIZE = 256;
  static constexpr std::uint16_t BLOCKS_PER_CACHE = CACHE_SIZE / RESULTS_PER_BLOCK;
  static_assert(CACHE_SIZE % RESULTS_PER_BLOCK == 0, "CACHE_SIZE must be a multiple of RESULTS_PER_BLOCK");
  static constexpr std::uint16_t BATCHES_PER_CACHE = SIMD_WIDTH == 0 ? 1 : BLOCKS_PER_CACHE / SIMD_WIDTH;
  static_assert(SIMD_WIDTH == 0 || BLOCKS_PER_CACHE % SIMD_WIDTH == 0,
                "BLOCKS_PER_CACHE must be a multiple of SIMD_WIDTH");

  // How many independent counter batches to run with their R-round chains
  // interleaved. Each batch is a serial multiply-dependency chain that leaves
  // most ALU ports idle while it waits on latency; interleaving K of them fills
  // those ports. K is budgeted from the vector register file for *this* arch
  // (poet::vector_register_count() is consteval and reflects the ISA the TU is
  // compiled for: 16 for SSE2/AVX2, 32 for AVX-512), shared between the K*N live
  // counter words and the per-round multiply transients — no per-combo literals.
  //
  // The scratch reserve scales with the widening-multiply cost, which depends on
  // W. The 64-bit mul_hilo synthesises 64x64->128 from four 32x32 partials plus
  // a carry/shift tree (~half the file of live transients); reserving VREG/2
  // gives K=(VREG/2)/N. On avx2 N=4 that is K=2, which is the *measured optimum*:
  // the 64-bit kernel is ALU-port-bound (~925 vector-ALU uops per 2-batch group
  // / 3 ports ~= 308 cyc floor), so more interleave only adds register pressure
  // and slips (K=3..6 all regress; a K=2..6 sweep confirms K=2 is the peak).
  //
  // The 32-bit path multiplies with a cheap mulhi and leaves the kernel
  // latency-bound rather than port-bound, so extra interleave keeps paying off:
  // devote the *whole* file to counters (reserve 0 -> K=VREG/N) and let the wider
  // ILP hide the multiply latency. The counter spills this costs land on the
  // otherwise-idle load ports. Measured avx2 N=4 W=32 optimum is K=4 (+5% over
  // the old K=2; the K=2..6 sweep peaks there and falls off by K=5). For W=64
  // this expression is identical to the prior (VREG/2)/N, so the 64-bit codegen
  // is unchanged.
  static constexpr std::uint16_t INTERLEAVE = []() constexpr -> std::uint16_t {
#if defined(_MSC_VER) && !defined(__clang__)
    // MSVC's frontend cost is superlinear in inlined-body size, and gen_block_group<K>
    // expands to R*K fully-inlined round chains (each with the multi-partial fused
    // mul_hilo). At the register-budgeted K (up to 8 on AVX-512) that is minutes per
    // dispatch TU — the build blew past the CI cap. K only reorders independent batches
    // for ILP; the cache output is bit-identical for any K (see gen_block_group). MSVC's
    // codegen isn't what this knob tunes, so pin K=1 there: same results, tractable build.
    return std::uint16_t{1};
#else
    const std::size_t vreg = poet::vector_register_count();
    const std::size_t scratch = (W >= 64) ? vreg / 2 : 0;
    std::size_t k = (vreg - scratch) / N;
    if (k < 1)
      k = 1;
    if (k > BATCHES_PER_CACHE)
      k = BATCHES_PER_CACHE;
    return static_cast<std::uint16_t>(k);
#endif
  }();

  counter_type m_counter;
  key_type m_key;

  explicit SIMDRNG_ALWAYS_INLINE PhiloxState(const key_type &key, const counter_type &counter) noexcept
      : m_counter(counter), m_key(key) {}

  SIMDRNG_ALWAYS_INLINE void populate_cache(std::array<result_type, CACHE_SIZE> &SIMDRNG_RESTRICT cache) noexcept {
    constexpr std::uint16_t K = INTERLEAVE;
    // Index/extent arithmetic uses std::size_t (the native pointer-width index
    // on this target): a narrow fixed-width loop counter would force the
    // compiler to re-mask to width on every increment/compare. word_type stays
    // exact (it defines the RNG/SIMD-lane semantics).
    constexpr std::size_t STRIDE = static_cast<std::size_t>(SIMD_WIDTH) * RESULTS_PER_BLOCK;
    constexpr std::size_t FULL_GROUPS = BATCHES_PER_CACHE / K;
    constexpr std::size_t REMAINDER = BATCHES_PER_CACHE % K;

    auto counter = m_counter;
    result_type *out = cache.data();
    // Runtime loop (not static_for) over the full K-batch groups: this bounds
    // the live set GCC sees to one group, so it can't unroll all
    // BATCHES_PER_CACHE batches into one block and spill. The loop-carried
    // counter/out keep iterations from being interleaved (and re-spilling).
    for (std::size_t g = 0; g < FULL_GROUPS; ++g) {
      gen_block_group<K>(out, counter, m_key);
      out += static_cast<std::size_t>(K) * STRIDE;
      advance_counter(counter, static_cast<word_type>(K * SIMD_WIDTH));
    }
    if constexpr (REMAINDER > 0) {
      gen_block_group<REMAINDER>(out, counter, m_key);
      advance_counter(counter, static_cast<word_type>(REMAINDER * SIMD_WIDTH));
    }
    m_counter = counter;
  }

  counter_type getCounter(bool prev, std::uint16_t cache_index) const noexcept {
    counter_type ctr = m_counter;
    auto consumed = static_cast<word_type>(cache_index / RESULTS_PER_BLOCK);
    auto back_off = static_cast<word_type>(BLOCKS_PER_CACHE) - consumed;
    if (prev)
      ++back_off;
    // Two's-complement: subtract back_off via advance_counter(-back_off).
    advance_counter(ctr, static_cast<word_type>(0) - back_off);
    return ctr;
  }

  const counter_type &getRawCounter() const noexcept { return m_counter; }

  const key_type &getKey() const noexcept { return m_key; }

  void setState(const counter_type &ctr, const key_type &key) noexcept {
    m_counter = ctr;
    m_key = key;
  }

private:
  using C = PhiloxConstants<N, W>;

  static SIMDRNG_ALWAYS_INLINE void advance_counter(counter_type &ctr, word_type amount) noexcept {
    word_type old = ctr[0];
    ctr[0] += amount;
    if constexpr (N >= 2) {
      if (ctr[0] < old) { // carry
        for (std::uint8_t i = 1; i < N; ++i) {
          if (++ctr[i] != 0)
            break;
        }
      }
    }
  }

  static SIMDRNG_ALWAYS_INLINE void simd_single_round(std::array<simd_type, N> &ctr, key_type &key) noexcept {
    // Data-driven round, generic over N/2 counter pairs (see PhiloxConstants).
    // poet::static_for unrolls at compile time so the PERM lookups and lane
    // indices fold away. The lambda bodies carry SIMDRNG_ALWAYS_INLINE_LAMBDA so
    // GCC inlines them instead of cloning each heavy round into its own function.
    constexpr std::uint8_t PAIRS = N / 2;
    std::array<simd_type, PAIRS> hi;
    std::array<simd_type, PAIRS> lo;
    poet::static_for<0, PAIRS>([&](auto J) SIMDRNG_ALWAYS_INLINE_LAMBDA {
      auto [h, l] = xsimd::mul_hilo(ctr[2 * J], simd_type::broadcast(C::MUL[J]));
      hi[J] = h;
      lo[J] = l;
    });
    // Write the permuted result back into `ctr` in place. Pair J reads only its
    // own old odd word ctr[2J+1] (consumed before being overwritten) and the
    // already-computed hi/lo[PERM[J]]; the even word ctr[2J] is dead after the
    // multiply above. So no extra `out[N]` array is needed — the per-batch
    // persistent footprint stays at the N counter registers, which is the
    // headroom the interleaved kernel below spends on ILP.
    poet::static_for<0, PAIRS>([&](auto J) SIMDRNG_ALWAYS_INLINE_LAMBDA {
      constexpr std::uint8_t S = C::PERM[J];
      const simd_type even = hi[S] ^ ctr[2 * J + 1] ^ simd_type::broadcast(key[J]);
      ctr[2 * J + 1] = lo[S];
      ctr[2 * J] = even;
    });
    poet::static_for<0, PAIRS>([&](auto J) SIMDRNG_ALWAYS_INLINE_LAMBDA { key[J] += C::BUMP[J]; });
  }

  static SIMDRNG_ALWAYS_INLINE void init_counter_batch(std::array<simd_type, N> &ctr_simd,
                                                       const counter_type &counter) noexcept {
    alignas(simd_type::arch_type::alignment()) std::array<word_type, SIMD_WIDTH> offsets{};
    poet::static_for<0, SIMD_WIDTH>([&](auto I)
                                        SIMDRNG_ALWAYS_INLINE_LAMBDA { offsets[I] = static_cast<word_type>(I.value); });

    auto carry = simd_type::load_aligned(offsets.data());
    poet::static_for<0, N>([&](auto I) SIMDRNG_ALWAYS_INLINE_LAMBDA {
      auto base = simd_type::broadcast(counter[I]);
      auto val = base + carry;
      ctr_simd[I] = val;
      if constexpr (I + 1 < N) {
        carry = xsimd::select(val < base, simd_type::broadcast(word_type{1}), simd_type::broadcast(word_type{0}));
      }
    });
  }

  static SIMDRNG_ALWAYS_INLINE void store_blocks_to_cache(result_type *cache,
                                                          const std::array<simd_type, N> &ctr_simd) noexcept {
    // A block is RESULTS_PER_BLOCK u64 outputs; each output is assembled from
    // WORDS_PER_RESULT = 64/W consecutive lane-words (1 for W=64, 2 for W=32).
    // When each word maps 1:1 to an output (RESULTS_PER_BLOCK == N, i.e. every
    // counter word is already a full u64) and the matrix is square
    // (N == SIMD_WIDTH), the block-major layout is exactly the transpose of the
    // N x SIMD_WIDTH counter matrix — a few shuffles instead of N*SIMD_WIDTH
    // scalar lane extracts. This geometric predicate replaces the old `W == 64`
    // literal; the generic scatter below handles every other (N, W, Arch) combo.
    constexpr bool kSquareTranspose = (RESULTS_PER_BLOCK == N) && (N == SIMD_WIDTH);
    if constexpr (kSquareTranspose) {
      std::array<simd_type, SIMD_WIDTH> regs;
      poet::static_for<0, N>([&](auto I) SIMDRNG_ALWAYS_INLINE_LAMBDA { regs[I] = ctr_simd[I]; });
      xsimd::transpose(regs.data(), regs.data() + SIMD_WIDTH);
      poet::static_for<0, SIMD_WIDTH>(
          [&](auto Lane) SIMDRNG_ALWAYS_INLINE_LAMBDA { regs[Lane].store_aligned(cache + Lane * RESULTS_PER_BLOCK); });
    } else {
      constexpr std::uint8_t WORDS_PER_RESULT = static_cast<std::uint8_t>(64 / W);
      alignas(simd_type::arch_type::alignment()) std::array<word_type, SIMD_WIDTH> regs[N];
      poet::static_for<0, N>([&](auto I) SIMDRNG_ALWAYS_INLINE_LAMBDA { ctr_simd[I].store_aligned(regs[I].data()); });
      // One dynamic_for over the lanes, in the (lane) running-index form: Unroll ==
      // SIMD_WIDTH so the block is fully unrolled at compile time, but `lane` stays a
      // *runtime* running value, so the per-lane base address is one induction value
      // rather than SIMD_WIDTH materialized constants — no scatter-offset spills.
      // The within-block assembly (RESULTS_PER_BLOCK results, each WORDS_PER_RESULT =
      // 64/W lane-words) is two plain loops over built-in std::size_t with constant
      // bounds: the optimizer unrolls them (so codegen matches a full static unroll),
      // the size_t subscript dodges the MSVC integral_constant chained-subscript parse
      // bug, and — critically — flattening to a single dynamic_for avoids the nested
      // pack-expansion-of-pack-expansion that makes MSVC's frontend blow up (the
      // triple-nested form compiled but never finished within the 6h CI build cap).
      poet::dynamic_for<SIMD_WIDTH>(std::size_t{SIMD_WIDTH}, [&](std::size_t lane) SIMDRNG_ALWAYS_INLINE_LAMBDA {
        result_type *const out = cache + lane * RESULTS_PER_BLOCK;
        for (std::size_t k = 0; k < RESULTS_PER_BLOCK; ++k) {
          result_type acc = 0;
          for (std::size_t wd = 0; wd < WORDS_PER_RESULT; ++wd) {
            acc |= static_cast<result_type>(regs[k * WORDS_PER_RESULT + wd][lane]) << (wd * W);
          }
          out[k] = acc;
        }
      });
    }
  }

  // Generate G consecutive SIMD batches with their round chains interleaved.
  // `counter` is the base for batch 0; batch b uses counter + b*SIMD_WIDTH and
  // writes to cache + b*SIMD_WIDTH*RESULTS_PER_BLOCK, exactly as G separate
  // sequential gen_block_batch calls would — so the output stays bit-identical;
  // only the instruction schedule changes. G==1 is the plain single-batch path.
  template <std::uint16_t G>
  static SIMDRNG_ALWAYS_INLINE void gen_block_group(result_type *cache, counter_type counter,
                                                    const key_type &key) noexcept {
    std::array<std::array<simd_type, N>, G> ctr_simd;
    std::array<key_type, G> round_key;
    poet::static_for<0, G>([&](auto B) SIMDRNG_ALWAYS_INLINE_LAMBDA {
      init_counter_batch(ctr_simd[B.value], counter);
      round_key[B.value] = key;
      advance_counter(counter, static_cast<word_type>(SIMD_WIDTH));
    });

    // Round-major over batch-minor: advance every batch by one round before
    // moving to the next round, so the G independent multiply chains overlap.
    poet::static_for<0, R>([&](auto) SIMDRNG_ALWAYS_INLINE_LAMBDA {
      poet::static_for<0, G>(
          [&](auto B) SIMDRNG_ALWAYS_INLINE_LAMBDA { simd_single_round(ctr_simd[B.value], round_key[B.value]); });
    });

    poet::static_for<0, G>([&](auto B) SIMDRNG_ALWAYS_INLINE_LAMBDA {
      store_blocks_to_cache(cache + B.value * SIMD_WIDTH * RESULTS_PER_BLOCK, ctr_simd[B.value]);
    });
  }
};

// Dispatch result: function pointers for the type-erased PhiloxSIMD wrapper.
template <std::uint8_t N, std::uint8_t W> struct PhiloxSIMDInitResult {
  using word_type = std::conditional_t<W == 32, std::uint32_t, std::uint64_t>;
  using counter_type = std::array<word_type, N>;
  using key_type = std::array<word_type, N / 2>;
  using result_type = std::uint64_t;
  static constexpr std::uint16_t CACHE_SIZE = 256;

  using populate_fn = void (*)(void *SIMDRNG_RESTRICT, std::array<result_type, CACHE_SIZE> &SIMDRNG_RESTRICT) noexcept;
  using get_counter_fn = counter_type (*)(const void *, bool, std::uint16_t) noexcept;
  using get_raw_counter_fn = counter_type (*)(const void *) noexcept;
  using get_key_fn = key_type (*)(const void *) noexcept;
  using set_state_fn = void (*)(void *, const counter_type &, const key_type &) noexcept;

  populate_fn populate_cache;
  get_counter_fn get_counter;
  get_raw_counter_fn get_raw_counter;
  get_key_fn get_key;
  set_state_fn set_state;
  std::size_t simd_size;
};

template <std::uint8_t N, std::uint8_t W, std::uint8_t R> struct PhiloxSIMDInitFunctor {
  void *state_storage;
  using word_type = std::conditional_t<W == 32, std::uint32_t, std::uint64_t>;
  using counter_type = std::array<word_type, N>;
  using key_type = std::array<word_type, N / 2>;
  const key_type key;
  const counter_type counter;

  template <class Arch> PhiloxSIMDInitResult<N, W> operator()(Arch /*arch*/) const noexcept;
};

template <std::uint8_t N, std::uint8_t W, std::uint8_t R>
template <class Arch>
PhiloxSIMDInitResult<N, W> PhiloxSIMDInitFunctor<N, W, R>::operator()(Arch /*arch*/) const noexcept {
  using State = PhiloxState<Arch, N, W, R>;
  using InitResult = PhiloxSIMDInitResult<N, W>;
  static_assert(sizeof(State) <= 256, "PhiloxState exceeds StateStorage capacity");
  static_assert(alignof(State) <= 64, "PhiloxState exceeds StateStorage alignment");
  std::construct_at(static_cast<State *>(state_storage), key, counter);
  return {
      +[](void *SIMDRNG_RESTRICT s,
          std::array<typename InitResult::result_type, InitResult::CACHE_SIZE> &SIMDRNG_RESTRICT cache) noexcept {
        static_cast<State *>(s)->populate_cache(cache);
      },
      +[](const void *s, bool prev, std::uint16_t idx) noexcept ->
      typename InitResult::counter_type { return static_cast<const State *>(s)->getCounter(prev, idx); },
      +[](const void *s) noexcept ->
      typename InitResult::counter_type { return static_cast<const State *>(s)->getRawCounter(); },
      +[](const void *s) noexcept -> typename InitResult::key_type { return static_cast<const State *>(s)->getKey(); },
      +[](void *s, const typename InitResult::counter_type &ctr, const typename InitResult::key_type &k) noexcept {
        static_cast<State *>(s)->setState(ctr, k);
      },
      std::size_t{State::SIMD_WIDTH},
  };
}

// Extern template declarations for all NxW combos and architectures
#define SIMDRNG_PHILOX_EXTERN_TEMPLATE(N, W, R, Arch)                                                                  \
  extern template SIMDRNG_LOCAL PhiloxSIMDInitResult<N, W> PhiloxSIMDInitFunctor<N, W, R>::operator()<Arch>(Arch)      \
      const noexcept

#define SIMDRNG_PHILOX_EXTERN_TEMPLATES_FOR_ARCH(Arch)                                                                 \
  SIMDRNG_PHILOX_EXTERN_TEMPLATE(4, 32, 10, Arch);                                                                     \
  SIMDRNG_PHILOX_EXTERN_TEMPLATE(2, 32, 10, Arch);                                                                     \
  SIMDRNG_PHILOX_EXTERN_TEMPLATE(4, 64, 10, Arch);                                                                     \
  SIMDRNG_PHILOX_EXTERN_TEMPLATE(2, 64, 10, Arch)

#if SIMDRNG_ARCH_X86_64
SIMDRNG_PHILOX_EXTERN_TEMPLATES_FOR_ARCH(xsimd::sse2);
SIMDRNG_PHILOX_EXTERN_TEMPLATES_FOR_ARCH(xsimd::avx2);
SIMDRNG_PHILOX_EXTERN_TEMPLATES_FOR_ARCH(xsimd::avx512bw);
#elif SIMDRNG_ARCH_AARCH64
SIMDRNG_PHILOX_EXTERN_TEMPLATES_FOR_ARCH(xsimd::neon64);
#if XSIMD_WITH_SVE
SIMDRNG_PHILOX_EXTERN_TEMPLATES_FOR_ARCH(xsimd::sve);
#endif
#elif SIMDRNG_ARCH_RISCV64
SIMDRNG_PHILOX_EXTERN_TEMPLATES_FOR_ARCH(xsimd::detail::rvv<128>);
#endif

#undef SIMDRNG_PHILOX_EXTERN_TEMPLATES_FOR_ARCH
#undef SIMDRNG_PHILOX_EXTERN_TEMPLATE

} // namespace internal

// PhiloxSIMD: runtime SIMD dispatch via inline storage + function pointers.
template <std::uint8_t N = 4, std::uint8_t W = 32, std::uint8_t R = 10> class PhiloxSIMD {
public:
  using result_type = std::uint64_t;
  using word_type = std::conditional_t<W == 32, std::uint32_t, std::uint64_t>;
  using counter_type = std::array<word_type, N>;
  using key_type = std::array<word_type, N / 2>;
  static constexpr std::uint16_t CACHE_SIZE = 256;

  static constexpr SIMDRNG_ALWAYS_INLINE auto(min)() noexcept { return (std::numeric_limits<result_type>::min)(); }
  static constexpr SIMDRNG_ALWAYS_INLINE auto(max)() noexcept { return (std::numeric_limits<result_type>::max)(); }

  static constexpr key_type seed_to_key(result_type seed) noexcept { return Philox<N, W, R>::seed_to_key(seed); }

  static constexpr counter_type counter_from_uint64(result_type counter) noexcept {
    return Philox<N, W, R>::counter_from_uint64(counter);
  }

  explicit SIMDRNG_ALWAYS_INLINE PhiloxSIMD(result_type seed, result_type counter = 0) noexcept
      : PhiloxSIMD(seed_to_key(seed), counter_from_uint64(counter)) {}

  explicit SIMDRNG_ALWAYS_INLINE PhiloxSIMD(const key_type &key, const counter_type &counter) noexcept {
    auto result =
        xsimd::dispatch<dispatch_arch_list>(internal::PhiloxSIMDInitFunctor<N, W, R>{m_state.data, key, counter})();
    m_populate_cache = result.populate_cache;
    m_get_counter = result.get_counter;
    m_get_raw_counter = result.get_raw_counter;
    m_get_key = result.get_key;
    m_set_state = result.set_state;
    m_simd_size = result.simd_size;
  }

  SIMDRNG_ALWAYS_INLINE result_type operator()() noexcept {
    if (m_index == 0) [[unlikely]] {
      m_populate_cache(m_state.data, m_cache);
    }
    return m_cache[m_index++];
  }

  SIMDRNG_ALWAYS_INLINE double uniform() noexcept { return static_cast<double>(operator()() >> 11) * 0x1.0p-53; }

  // Bulk fill, bit-identical to n consecutive operator() calls. Drains the
  // partial cache prefix, then refills the (SIMD-aligned) cache a block at a
  // time and copies straight to `out`, parking the cursor for the tail so the
  // next operator() resumes the exact stream. The cursor is a non-escaping
  // local, so it stays in a register across the dispatched refill.
  void generate(result_type *SIMDRNG_RESTRICT out, std::size_t n) noexcept {
    std::size_t produced = 0;
    std::uint8_t idx = m_index;
    if (idx != 0) {
      const std::size_t avail = CACHE_SIZE - idx;
      const std::size_t take = avail < n ? avail : n;
      std::memcpy(out, m_cache.data() + idx, take * sizeof(result_type));
      idx = static_cast<std::uint8_t>(idx + take);
      produced = take;
    }
    while (produced < n) {
      m_populate_cache(m_state.data, m_cache);
      const std::size_t rem = n - produced;
      const std::size_t take = rem < CACHE_SIZE ? rem : CACHE_SIZE;
      std::memcpy(out + produced, m_cache.data(), take * sizeof(result_type));
      produced += take;
      idx = static_cast<std::uint8_t>(take); // CACHE_SIZE -> 0 (uint8 wrap)
    }
    m_index = idx;
  }

  void fill_uniform(double *SIMDRNG_RESTRICT out, std::size_t n) noexcept {
    alignas(64) std::array<result_type, CACHE_SIZE> buf;
    std::size_t done = 0;
    while (done < n) {
      const std::size_t take = (n - done) < CACHE_SIZE ? (n - done) : CACHE_SIZE;
      generate(buf.data(), take);
      for (std::size_t i = 0; i < take; ++i)
        out[done + i] = static_cast<double>(buf[i] >> 11) * 0x1.0p-53;
      done += take;
    }
  }

  counter_type getCounter() const noexcept { return m_get_counter(m_state.data, m_index != 0, m_index); }

  key_type getKey() const noexcept { return m_get_key(m_state.data); }

  counter_type getCounterForSerde() const noexcept { return m_get_raw_counter(m_state.data); }

  void setState(const counter_type &ctr, const key_type &key) noexcept {
    m_set_state(m_state.data, ctr, key);
    m_index = 0;
  }

  std::uint8_t cache_index() const noexcept { return m_index; }
  void set_cache_index(std::uint8_t idx) noexcept { m_index = idx; }
  const std::array<result_type, CACHE_SIZE> &cache() const noexcept { return m_cache; }
  std::array<result_type, CACHE_SIZE> &cache() noexcept { return m_cache; }

  SIMDRNG_ALWAYS_INLINE std::size_t getSIMDSize() const noexcept { return m_simd_size; }

private:
  using InitResult = internal::PhiloxSIMDInitResult<N, W>;
  using populate_fn = typename InitResult::populate_fn;
  using get_counter_fn = typename InitResult::get_counter_fn;
  using get_raw_counter_fn = typename InitResult::get_raw_counter_fn;
  using get_key_fn = typename InitResult::get_key_fn;
  using set_state_fn = typename InitResult::set_state_fn;

  struct StateStorage {
    static constexpr std::size_t SIZE = 256;
    static constexpr std::size_t ALIGN = 64;
    alignas(ALIGN) unsigned char data[SIZE];
  };

  alignas(64) std::array<result_type, CACHE_SIZE> m_cache{};
  // Value-initialised so the type-erased storage is never read uninitialised
  // (the per-arch State is written into m_state.data by the dispatch functor in
  // the constructor body).
  alignas(64) StateStorage m_state{};
  populate_fn m_populate_cache = nullptr;
  get_counter_fn m_get_counter = nullptr;
  get_raw_counter_fn m_get_raw_counter = nullptr;
  get_key_fn m_get_key = nullptr;
  set_state_fn m_set_state = nullptr;
  std::size_t m_simd_size = 0;
  std::uint8_t m_index = 0;
};

#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE
template <std::uint8_t N = 4, std::uint8_t W = 32, std::uint8_t R = 10> class PhiloxNative {
public:
  using result_type = std::uint64_t;
  using word_type = std::conditional_t<W == 32, std::uint32_t, std::uint64_t>;
  using counter_type = std::array<word_type, N>;
  using key_type = std::array<word_type, N / 2>;
  static constexpr std::uint16_t CACHE_SIZE = 256;

  static constexpr SIMDRNG_ALWAYS_INLINE auto(min)() noexcept { return (std::numeric_limits<result_type>::min)(); }
  static constexpr SIMDRNG_ALWAYS_INLINE auto(max)() noexcept { return (std::numeric_limits<result_type>::max)(); }

  explicit PhiloxNative(result_type seed, result_type counter = 0) noexcept
      : PhiloxNative(Philox<N, W, R>::seed_to_key(seed), Philox<N, W, R>::counter_from_uint64(counter)) {}

  PhiloxNative(const key_type &key, const counter_type &counter) noexcept : m_state(key, counter) {}

  SIMDRNG_ALWAYS_INLINE result_type operator()() noexcept {
    if (m_index == 0) [[unlikely]] {
      m_state.populate_cache(m_cache);
    }
    return m_cache[m_index++];
  }

  SIMDRNG_ALWAYS_INLINE double uniform() noexcept { return static_cast<double>(operator()() >> 11) * 0x1.0p-53; }

  // Bulk fill, bit-identical to n consecutive operator() calls (see PhiloxSIMD).
  void generate(result_type *SIMDRNG_RESTRICT out, std::size_t n) noexcept {
    std::size_t produced = 0;
    std::uint8_t idx = m_index;
    if (idx != 0) {
      const std::size_t avail = CACHE_SIZE - idx;
      const std::size_t take = avail < n ? avail : n;
      std::memcpy(out, m_cache.data() + idx, take * sizeof(result_type));
      idx = static_cast<std::uint8_t>(idx + take);
      produced = take;
    }
    while (produced < n) {
      m_state.populate_cache(m_cache);
      const std::size_t rem = n - produced;
      const std::size_t take = rem < CACHE_SIZE ? rem : CACHE_SIZE;
      std::memcpy(out + produced, m_cache.data(), take * sizeof(result_type));
      produced += take;
      idx = static_cast<std::uint8_t>(take);
    }
    m_index = idx;
  }

  void fill_uniform(double *SIMDRNG_RESTRICT out, std::size_t n) noexcept {
    alignas(64) std::array<result_type, CACHE_SIZE> buf;
    std::size_t done = 0;
    while (done < n) {
      const std::size_t take = (n - done) < CACHE_SIZE ? (n - done) : CACHE_SIZE;
      generate(buf.data(), take);
      for (std::size_t i = 0; i < take; ++i)
        out[done + i] = static_cast<double>(buf[i] >> 11) * 0x1.0p-53;
      done += take;
    }
  }

  counter_type getCounter() const noexcept { return m_state.getCounter(m_index != 0, m_index); }

  key_type getKey() const noexcept { return m_state.getKey(); }

  counter_type getCounterForSerde() const noexcept { return m_state.getRawCounter(); }

  void setState(const counter_type &ctr, const key_type &key) noexcept {
    m_state.setState(ctr, key);
    m_index = 0;
  }

  std::uint8_t cache_index() const noexcept { return m_index; }
  void set_cache_index(std::uint8_t idx) noexcept { m_index = idx; }
  const std::array<result_type, CACHE_SIZE> &cache() const noexcept { return m_cache; }
  std::array<result_type, CACHE_SIZE> &cache() noexcept { return m_cache; }

  SIMDRNG_ALWAYS_INLINE std::size_t getSIMDSize() const noexcept { return std::size_t{State::SIMD_WIDTH}; }

private:
  using State = internal::PhiloxState<xsimd::best_arch, N, W, R>;
  alignas(State::simd_type::arch_type::alignment()) std::array<result_type, CACHE_SIZE> m_cache{};
  State m_state;
  std::uint8_t m_index = 0;
};
#endif // XSIMD_NO_SUPPORTED_ARCHITECTURE

// Convenience aliases
using Philox4x32SIMD = PhiloxSIMD<4, 32, 10>;
using Philox2x32SIMD = PhiloxSIMD<2, 32, 10>;
using Philox4x64SIMD = PhiloxSIMD<4, 64, 10>;
using Philox2x64SIMD = PhiloxSIMD<2, 64, 10>;

#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE
using Philox4x32Native = PhiloxNative<4, 32, 10>;
using Philox2x32Native = PhiloxNative<2, 32, 10>;
using Philox4x64Native = PhiloxNative<4, 64, 10>;
using Philox2x64Native = PhiloxNative<2, 64, 10>;
#endif // XSIMD_NO_SUPPORTED_ARCHITECTURE

} // namespace simdrng
// END_FILE: philox_simd.hpp
/* End inline (quoted): philox_simd.hpp */
/* Begin inline (quoted): xoshiro.hpp */
// BEGIN_FILE: xoshiro.hpp

// Umbrella header: `Xoshiro` is the runtime-dispatched SIMD generator when the
// library is built with SIMDRNG_DISPATCH, otherwise the native (best_arch) one.
/* Begin inline (quoted): macros.hpp */
/* Skipped already inlined: macros.hpp */
/* End inline (quoted): macros.hpp */
/* Begin inline (quoted): xoshiro_simd.hpp */
// BEGIN_FILE: xoshiro_simd.hpp
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


#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>

/* Begin inline (angle): poet.hpp */
/* Skipped already inlined: poet.hpp */
/* End inline (angle): poet.hpp */

/* Begin inline (quoted): dispatch_arch.hpp */
/* Skipped already inlined: dispatch_arch.hpp */
/* End inline (quoted): dispatch_arch.hpp */
/* Begin inline (quoted): macros.hpp */
/* Skipped already inlined: macros.hpp */
/* End inline (quoted): macros.hpp */
/* Begin inline (quoted): xoshiro_scalar.hpp */
/* Skipped already inlined: xoshiro_scalar.hpp */
/* End inline (quoted): xoshiro_scalar.hpp */

namespace simdrng {

class XoshiroSIMD;

namespace internal {

/**
 * SIMD state for the vectorized Xoshiro256++ generator.
 * Contains the 4 SIMD register state and all PRNG operations.
 *
 * @tparam Arch The xsimd architecture type.
 */
template <class Arch> struct XoshiroState {
  using result_type = std::uint64_t;
  using simd_type = xsimd::batch<result_type, Arch>;
  static constexpr std::uint8_t RNG_WIDTH = 4;
  static constexpr std::uint8_t SIMD_WIDTH = simd_type::size;
  static constexpr std::uint16_t CACHE_SIZE = std::numeric_limits<std::uint8_t>::max() + 1;

  alignas(simd_type::arch_type::alignment()) std::array<simd_type, RNG_WIDTH> s{};

  /**
   * Seed the SIMD state from a scalar seed, with optional thread and cluster offsets.
   */
  SIMDRNG_ALWAYS_INLINE constexpr void seed(result_type seed_val, result_type thread_id = 0,
                                            result_type cluster_id = 0) noexcept {
    XoshiroScalar rng{seed_val};
    std::array<std::array<result_type, SIMD_WIDTH>, RNG_WIDTH> states{};
    for (auto i = 0UL; i < SIMD_WIDTH; ++i) {
      for (auto j = 0UL; j < RNG_WIDTH; ++j) {
        states[j][i] = rng.getState()[j];
      }
      rng.jump();
    }
    poet::static_for<0, RNG_WIDTH>([&](auto I) { s[I] = simd_type::load_unaligned(states[I].data()); });
    for (result_type i = 0; i < thread_id; ++i) {
      mid_jump();
    }
    for (result_type i = 0; i < cluster_id; ++i) {
      long_jump();
    }
  }

  SIMDRNG_ALWAYS_INLINE constexpr simd_type next() noexcept {
    const auto result = xsimd::rotl<23>(s[0] + s[3]) + s[0];
    const auto t = xsimd::bitwise_lshift<17>(s[1]);

    s[2] ^= s[0];
    s[3] ^= s[1];

    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;

    s[3] = xsimd::rotl<45>(s[3]);

    return result;
  }

  SIMDRNG_ALWAYS_INLINE constexpr void
  populate_cache(std::array<result_type, CACHE_SIZE> &SIMDRNG_RESTRICT cache) noexcept {
    poet::static_for<0, CACHE_SIZE / SIMD_WIDTH>([&](auto I) { next().store_aligned(cache.data() + I * SIMD_WIDTH); });
  }

  // Generate `blocks` full CACHE_SIZE-value blocks straight to `out` (wide
  // unaligned stores, no cache bounce). Advances the state by blocks*CACHE_SIZE
  // outputs, exactly as that many populate_cache() calls would.
  SIMDRNG_ALWAYS_INLINE void generate_blocks(result_type *SIMDRNG_RESTRICT out, std::size_t blocks) noexcept {
    for (std::size_t b = 0; b < blocks; ++b) {
      result_type *SIMDRNG_RESTRICT dst = out + b * CACHE_SIZE;
      poet::static_for<0, CACHE_SIZE / SIMD_WIDTH>([&](auto I) { next().store_unaligned(dst + I * SIMD_WIDTH); });
    }
  }

  /**
   * Apply a jump polynomial to every lane: state <- poly(T) · state. All lanes
   * share the same polynomial (same jump count). Shared by every jump variant.
   */
  SIMDRNG_ALWAYS_INLINE constexpr void jump_poly(const result_type *SIMDRNG_RESTRICT poly) noexcept {
    static_assert(std::tuple_size_v<detail::jump_poly_t> == RNG_WIDTH,
                  "jump polynomial word count must match the state width");
    simd_type s0(0);
    simd_type s1(0);
    simd_type s2(0);
    simd_type s3(0);
    for (std::uint8_t w = 0; w < RNG_WIDTH; ++w)
      for (auto b = 0; b < 64; b++) {
        if ((poly[w] & result_type{1} << b) != 0U) {
          s0 ^= s[0];
          s1 ^= s[1];
          s2 ^= s[2];
          s3 ^= s[3];
        }
        next();
      }
    s[0] = s0;
    s[1] = s1;
    s[2] = s2;
    s[3] = s3;
  }

  /**
   * Jump function. Equivalent to 2^128 calls to next().
   */
  SIMDRNG_ALWAYS_INLINE constexpr void jump() noexcept {
    constexpr result_type JUMP[] = {0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c};
    jump_poly(JUMP);
  }

  /**
   * Mid-jump function. Equivalent to 2^160 calls to next().
   */
  SIMDRNG_ALWAYS_INLINE constexpr void mid_jump() noexcept {
    constexpr result_type MID_JUMP[] = {0xc04b4f9c5d26c200, 0x69e6e6e431a2d40b, 0x4823b45b89dc689c, 0xf567382197055bf0};
    jump_poly(MID_JUMP);
  }

  /**
   * Long-jump function. Equivalent to 2^192 calls to next().
   */
  SIMDRNG_ALWAYS_INLINE constexpr void long_jump() noexcept {
    constexpr result_type LONG_JUMP[] = {0x76e15d3efefdcbbf, 0xc5004e441c522fb3, 0x77710069854ee241,
                                         0x39109bb02acbe635};
    jump_poly(LONG_JUMP);
  }

  /**
   * Arbitrary jump-ahead: advances every lane by exactly n calls to next() via x^n mod P(x).
   * Overload of jump(); no-argument jump() remains the fixed 2^128 stride.
   */
  SIMDRNG_ALWAYS_INLINE constexpr void jump(std::uint64_t n) noexcept { jump_poly(detail::jump_poly_n(n).data()); }

  /**
   * Power-of-two jump-ahead: advances every lane by 2^p.exponent calls to next() via x^(2^e) mod P(x).
   */
  SIMDRNG_ALWAYS_INLINE constexpr void jump(pow2 p) noexcept { jump_poly(detail::jump_poly_pow2(p.exponent).data()); }

  SIMDRNG_ALWAYS_INLINE constexpr std::array<result_type, RNG_WIDTH> getState(const std::size_t index) const noexcept {
    std::array<result_type, RNG_WIDTH> state{};
    poet::static_for<0, RNG_WIDTH>([&](auto I) { state[I] = s[I].get(index); });
    return state;
  }

  void get_flat_state(result_type *out) const noexcept {
    for (std::uint8_t reg = 0; reg < RNG_WIDTH; ++reg)
      for (std::uint8_t lane = 0; lane < SIMD_WIDTH; ++lane)
        out[reg * SIMD_WIDTH + lane] = s[reg].get(lane);
  }

  void set_flat_state(const result_type *in) noexcept {
    for (std::uint8_t reg = 0; reg < RNG_WIDTH; ++reg) {
      std::array<result_type, SIMD_WIDTH> vals;
      for (std::uint8_t lane = 0; lane < SIMD_WIDTH; ++lane)
        vals[lane] = in[reg * SIMD_WIDTH + lane];
      s[reg] = simd_type::load_unaligned(vals.data());
    }
  }
};

/**
 * Result from the runtime dispatch initialization.
 */
struct XoshiroSIMDInitResult {
  using populate_fn = void (*)(void *SIMDRNG_RESTRICT, std::array<std::uint64_t, 256> &SIMDRNG_RESTRICT) noexcept;
  using generate_fn = void (*)(void *SIMDRNG_RESTRICT, std::uint64_t *SIMDRNG_RESTRICT, std::size_t) noexcept;
  using jump_fn = void (*)(void *) noexcept;
  using jump_n_fn = void (*)(void *, const std::uint64_t *) noexcept;
  using get_state_fn = void (*)(const void *, std::uint64_t *) noexcept;
  using set_state_fn = void (*)(void *, const std::uint64_t *) noexcept;
  using simd_width_fn = std::uint8_t (*)() noexcept;
  populate_fn populate_cache;
  generate_fn generate_blocks;
  jump_fn jump;
  jump_fn mid_jump;
  jump_fn long_jump;
  jump_n_fn jump_n;
  get_state_fn get_state;
  set_state_fn set_state;
  simd_width_fn simd_width;
};

/**
 * Functor used by xsimd::dispatch to initialize a XoshiroSIMD instance.
 * Placement-news the correct XoshiroState<Arch> into the byte storage and returns function pointers.
 */
struct XoshiroSIMDInitFunctor {
  void *state_storage;
  std::uint64_t seed, thread_id, cluster_id;

  template <class Arch> XoshiroSIMDInitResult operator()(Arch /*arch*/) const noexcept;
};

template <class Arch> XoshiroSIMDInitResult XoshiroSIMDInitFunctor::operator()(Arch /*arch*/) const noexcept {
  using State = XoshiroState<Arch>;
  static_assert(sizeof(State) <= 256, "XoshiroState exceeds StateStorage capacity");
  static_assert(alignof(State) <= 64, "XoshiroState exceeds StateStorage alignment");
  auto *state = std::construct_at(static_cast<State *>(state_storage));
  state->seed(seed, thread_id, cluster_id);
  return {
      +[](void *SIMDRNG_RESTRICT s, std::array<std::uint64_t, 256> &SIMDRNG_RESTRICT cache) noexcept {
        static_cast<State *>(s)->populate_cache(cache);
      },
      +[](void *SIMDRNG_RESTRICT s, std::uint64_t *SIMDRNG_RESTRICT out, std::size_t blocks) noexcept {
        static_cast<State *>(s)->generate_blocks(out, blocks);
      },
      +[](void *s) noexcept { static_cast<State *>(s)->jump(); },
      +[](void *s) noexcept { static_cast<State *>(s)->mid_jump(); },
      +[](void *s) noexcept { static_cast<State *>(s)->long_jump(); },
      +[](void *s, const std::uint64_t *poly) noexcept { static_cast<State *>(s)->jump_poly(poly); },
      +[](const void *s, std::uint64_t *out) noexcept { static_cast<const State *>(s)->get_flat_state(out); },
      +[](void *s, const std::uint64_t *in) noexcept { static_cast<State *>(s)->set_flat_state(in); },
      +[]() noexcept -> std::uint8_t { return State::SIMD_WIDTH; },
  };
}

#if SIMDRNG_ARCH_X86_64
extern template SIMDRNG_LOCAL
    XoshiroSIMDInitResult XoshiroSIMDInitFunctor::operator()<xsimd::sse2>(xsimd::sse2) const noexcept;
extern template SIMDRNG_LOCAL XoshiroSIMDInitResult
XoshiroSIMDInitFunctor::operator()<xsimd::avx2>(xsimd::avx2) const noexcept;
extern template SIMDRNG_LOCAL XoshiroSIMDInitResult
XoshiroSIMDInitFunctor::operator()<xsimd::avx512bw>(xsimd::avx512bw) const noexcept;
#elif SIMDRNG_ARCH_AARCH64
extern template SIMDRNG_LOCAL
    XoshiroSIMDInitResult XoshiroSIMDInitFunctor::operator()<xsimd::neon64>(xsimd::neon64) const noexcept;
#if XSIMD_WITH_SVE
extern template SIMDRNG_LOCAL
    XoshiroSIMDInitResult XoshiroSIMDInitFunctor::operator()<xsimd::sve>(xsimd::sve) const noexcept;
#endif
#elif SIMDRNG_ARCH_RISCV64
extern template SIMDRNG_LOCAL XoshiroSIMDInitResult
XoshiroSIMDInitFunctor::operator()<xsimd::detail::rvv<128>>(xsimd::detail::rvv<128>) const noexcept;
#endif

} // namespace internal

#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE
/**
 * XoshiroNative: uses the best architecture available at compile time.
 * Zero indirection — direct calls to XoshiroState methods.
 */
class XoshiroNative {
  using State = internal::XoshiroState<xsimd::best_arch>;
  static constexpr auto CACHE_SIZE = State::CACHE_SIZE;

public:
  using result_type = std::uint64_t;
  static constexpr SIMDRNG_ALWAYS_INLINE auto(min)() noexcept { return (std::numeric_limits<result_type>::min)(); }
  static constexpr SIMDRNG_ALWAYS_INLINE auto(max)() noexcept { return (std::numeric_limits<result_type>::max)(); }
  static constexpr SIMDRNG_ALWAYS_INLINE auto stateSize() noexcept { return State::RNG_WIDTH; }

  SIMDRNG_ALWAYS_INLINE explicit XoshiroNative(const result_type seed) noexcept { m_state.seed(seed); }
  SIMDRNG_ALWAYS_INLINE explicit XoshiroNative(const result_type seed, const result_type thread_id) noexcept {
    m_state.seed(seed, thread_id);
  }
  SIMDRNG_ALWAYS_INLINE explicit XoshiroNative(const result_type seed, const result_type thread_id,
                                               const result_type cluster_id) noexcept {
    m_state.seed(seed, thread_id, cluster_id);
  }

  SIMDRNG_ALWAYS_INLINE result_type operator()() noexcept {
    if (m_index == 0) [[unlikely]] {
      m_state.populate_cache(m_cache);
    }
    return m_cache[m_index++];
  }

  SIMDRNG_ALWAYS_INLINE double uniform() noexcept { return static_cast<double>(operator()() >> 11) * 0x1.0p-53; }

  // Bulk fill, bit-identical to n consecutive operator() calls (see XoshiroSIMD).
  void generate(result_type *SIMDRNG_RESTRICT out, std::size_t n) noexcept {
    std::size_t produced = 0;
    std::uint8_t idx = m_index;
    if (idx != 0) {
      const std::size_t avail = CACHE_SIZE - idx;
      const std::size_t take = avail < n ? avail : n;
      std::memcpy(out, m_cache.data() + idx, take * sizeof(result_type));
      idx = static_cast<std::uint8_t>(idx + take);
      produced = take;
    }
    const std::size_t mid_blocks = (n - produced) / CACHE_SIZE;
    if (mid_blocks != 0) {
      m_state.generate_blocks(out + produced, mid_blocks);
      produced += mid_blocks * CACHE_SIZE;
    }
    if (produced < n) {
      m_state.populate_cache(m_cache);
      const std::size_t rem = n - produced;
      std::memcpy(out + produced, m_cache.data(), rem * sizeof(result_type));
      idx = static_cast<std::uint8_t>(rem);
    }
    m_index = idx;
  }

  void fill_uniform(double *SIMDRNG_RESTRICT out, std::size_t n) noexcept {
    alignas(64) std::array<result_type, CACHE_SIZE> buf;
    std::size_t done = 0;
    while (done < n) {
      const std::size_t take = (n - done) < CACHE_SIZE ? (n - done) : CACHE_SIZE;
      generate(buf.data(), take);
      for (std::size_t i = 0; i < take; ++i)
        out[done + i] = static_cast<double>(buf[i] >> 11) * 0x1.0p-53;
      done += take;
    }
  }

  SIMDRNG_ALWAYS_INLINE auto getState(const std::size_t index) const noexcept { return m_state.getState(index); }

  SIMDRNG_ALWAYS_INLINE void jump() noexcept { m_state.jump(); }
  SIMDRNG_ALWAYS_INLINE void mid_jump() noexcept { m_state.mid_jump(); }
  SIMDRNG_ALWAYS_INLINE void long_jump() noexcept { m_state.long_jump(); }
  SIMDRNG_ALWAYS_INLINE void jump(result_type n) noexcept { m_state.jump(n); }
  SIMDRNG_ALWAYS_INLINE void jump(pow2 p) noexcept { m_state.jump(p); }

  void get_flat_state(result_type *out) const noexcept { m_state.get_flat_state(out); }
  void set_flat_state(const result_type *in) noexcept { m_state.set_flat_state(in); }
  static constexpr std::uint8_t simd_width() noexcept { return State::SIMD_WIDTH; }

  std::uint8_t cache_index() const noexcept { return m_index; }
  void set_cache_index(std::uint8_t idx) noexcept { m_index = idx; }
  const std::array<result_type, CACHE_SIZE> &cache() const noexcept { return m_cache; }
  std::array<result_type, CACHE_SIZE> &cache() noexcept { return m_cache; }

private:
  alignas(State::simd_type::arch_type::alignment()) std::array<result_type, CACHE_SIZE> m_cache{};
  State m_state{};
  std::uint8_t m_index{0};
};
#endif // XSIMD_NO_SUPPORTED_ARCHITECTURE

/**
 * XoshiroSIMD: runtime SIMD dispatch via inline union + function pointers.
 * No heap allocation, no virtual dispatch.
 */
class XoshiroSIMD {
public:
  using result_type = std::uint64_t;
  static constexpr SIMDRNG_ALWAYS_INLINE result_type(min)() noexcept {
    return (std::numeric_limits<result_type>::min)();
  }
  static constexpr SIMDRNG_ALWAYS_INLINE result_type(max)() noexcept {
    return (std::numeric_limits<result_type>::max)();
  }

  SIMDRNG_LOCAL explicit XoshiroSIMD(result_type seed, result_type thread_id = 0, result_type cluster_id = 0) noexcept;

  SIMDRNG_ALWAYS_INLINE result_type operator()() noexcept {
    if (m_index == 0) [[unlikely]] {
      m_populate_cache(m_state.data, m_cache);
    }
    return m_cache[m_index++];
  }

  SIMDRNG_ALWAYS_INLINE double uniform() noexcept { return static_cast<double>(operator()() >> 11) * 0x1.0p-53; }

  // Bulk fill: bit-identical to n consecutive operator() calls. Drains the
  // partial cache prefix, generates the aligned middle with wide stores
  // straight to `out` via a single dispatched call, then refills the cache for
  // the < CACHE_SIZE tail and parks the cursor so the next operator() resumes
  // the exact stream.
  void generate(result_type *SIMDRNG_RESTRICT out, std::size_t n) noexcept {
    std::size_t produced = 0;
    std::uint8_t idx = m_index;
    if (idx != 0) {
      const std::size_t avail = CACHE_SIZE - idx;
      const std::size_t take = avail < n ? avail : n;
      std::memcpy(out, m_cache.data() + idx, take * sizeof(result_type));
      idx = static_cast<std::uint8_t>(idx + take);
      produced = take;
    }
    const std::size_t mid_blocks = (n - produced) / CACHE_SIZE;
    if (mid_blocks != 0) {
      m_generate(m_state.data, out + produced, mid_blocks);
      produced += mid_blocks * CACHE_SIZE;
    }
    if (produced < n) {
      m_populate_cache(m_state.data, m_cache);
      const std::size_t rem = n - produced;
      std::memcpy(out + produced, m_cache.data(), rem * sizeof(result_type));
      idx = static_cast<std::uint8_t>(rem);
    }
    m_index = idx;
  }

  void fill_uniform(double *SIMDRNG_RESTRICT out, std::size_t n) noexcept {
    alignas(64) std::array<result_type, CACHE_SIZE> buf;
    std::size_t done = 0;
    while (done < n) {
      const std::size_t take = (n - done) < CACHE_SIZE ? (n - done) : CACHE_SIZE;
      generate(buf.data(), take);
      for (std::size_t i = 0; i < take; ++i)
        out[done + i] = static_cast<double>(buf[i] >> 11) * 0x1.0p-53;
      done += take;
    }
  }

  SIMDRNG_ALWAYS_INLINE void jump() noexcept { m_jump(m_state.data); }
  SIMDRNG_ALWAYS_INLINE void mid_jump() noexcept { m_mid_jump(m_state.data); }
  SIMDRNG_ALWAYS_INLINE void long_jump() noexcept { m_long_jump(m_state.data); }

  // Arbitrary jump-ahead. The jump polynomial x^n mod P(x) is arch-independent
  // (pure scalar math), so it is computed once here and applied to every lane.
  SIMDRNG_ALWAYS_INLINE void jump(result_type n) noexcept { m_jump_n(m_state.data, detail::jump_poly_n(n).data()); }

  // Power-of-two jump-ahead: 2^p.exponent steps. Same arch-independent poly path.
  SIMDRNG_ALWAYS_INLINE void jump(pow2 p) noexcept {
    m_jump_n(m_state.data, detail::jump_poly_pow2(p.exponent).data());
  }

  void get_flat_state(result_type *out) const noexcept { m_get_state(m_state.data, out); }
  void set_flat_state(const result_type *in) noexcept { m_set_state(m_state.data, in); }
  std::uint8_t simd_width() const noexcept { return m_simd_width(); }

  std::uint8_t cache_index() const noexcept { return m_index; }
  void set_cache_index(std::uint8_t idx) noexcept { m_index = idx; }
  const std::array<result_type, std::numeric_limits<std::uint8_t>::max() + 1> &cache() const noexcept {
    return m_cache;
  }
  std::array<result_type, std::numeric_limits<std::uint8_t>::max() + 1> &cache() noexcept { return m_cache; }

protected:
  static constexpr std::uint16_t CACHE_SIZE = std::numeric_limits<std::uint8_t>::max() + 1;
  using populate_fn = void (*)(void *SIMDRNG_RESTRICT, std::array<result_type, CACHE_SIZE> &SIMDRNG_RESTRICT) noexcept;
  using generate_fn = void (*)(void *SIMDRNG_RESTRICT, result_type *SIMDRNG_RESTRICT, std::size_t) noexcept;
  using jump_fn = void (*)(void *) noexcept;
  using jump_n_fn = void (*)(void *, const result_type *) noexcept;
  using get_state_fn = void (*)(const void *, result_type *) noexcept;
  using set_state_fn = void (*)(void *, const result_type *) noexcept;
  using simd_width_fn = std::uint8_t (*)() noexcept;

  // Raw byte storage for the arch-specific XoshiroState.
  // Typed union is not viable: xsimd batch types have different sizeof
  // across TUs compiled with different -march flags (ODR divergence).
  // Max is avx512bw: 4 × sizeof(__m512i) = 4 × 64 = 256 bytes, align 64.
  struct StateStorage {
    static constexpr std::size_t SIZE = 256;
    static constexpr std::size_t ALIGN = 64;
    alignas(ALIGN) unsigned char data[SIZE];
  };

  alignas(64) std::array<result_type, CACHE_SIZE> m_cache{};
  alignas(64) StateStorage m_state{};
  populate_fn m_populate_cache = nullptr;
  generate_fn m_generate = nullptr;
  jump_fn m_jump = nullptr;
  jump_fn m_mid_jump = nullptr;
  jump_fn m_long_jump = nullptr;
  jump_n_fn m_jump_n = nullptr;
  get_state_fn m_get_state = nullptr;
  set_state_fn m_set_state = nullptr;
  simd_width_fn m_simd_width = nullptr;
  std::uint8_t m_index{0};
};

} // namespace simdrng
// END_FILE: xoshiro_simd.hpp
/* End inline (quoted): xoshiro_simd.hpp */

namespace simdrng {
#if SIMDRNG_DISPATCH
using Xoshiro = XoshiroSIMD;
#else
using Xoshiro = XoshiroNative;
#endif
} // namespace simdrng
// END_FILE: xoshiro.hpp
/* End inline (quoted): xoshiro.hpp */
// END_FILE: simdrng.hpp

#endif // SIMDRNG_SINGLE_HEADER_HPP
