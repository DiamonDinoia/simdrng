#pragma once

#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <poet/poet.hpp>

#include "dispatch_arch.hpp"
#include "macros.hpp"
#include "philox.hpp"

namespace prng {

namespace internal {

template <class Arch, std::uint8_t N, std::uint8_t W, std::uint8_t R>
struct PhiloxState {
  static_assert(N == 2 || N == 4, "Philox N must be 2 or 4");
  static_assert(W == 32 || W == 64, "Philox W must be 32 or 64");

  using word_type = std::conditional_t<W == 32, std::uint32_t, std::uint64_t>;
  using counter_type = std::array<word_type, N>;
  using key_type = std::array<word_type, N / 2>;
  using result_type = std::uint64_t;
  static constexpr auto RESULTS_PER_BLOCK = std::uint8_t{N * W / 64};
  using result_block_type = std::array<result_type, RESULTS_PER_BLOCK>;

  // Batch at native word width: W=32 uses batch<uint32_t> (2x lanes on x86),
  // W=64 uses batch<uint64_t>. xsimd::mulhilo handles both uniformly.
  using simd_type = xsimd::batch<word_type, Arch>;

  static constexpr std::uint8_t SIMD_WIDTH = std::uint8_t{simd_type::size};
  static_assert(SIMD_WIDTH == 0 || std::has_single_bit(static_cast<unsigned int>(SIMD_WIDTH)),
                "Philox SIMD width must be a power of two");

  static constexpr std::uint8_t cache_batchcount() noexcept {
    constexpr auto results_per_batch = static_cast<unsigned>(SIMD_WIDTH) * RESULTS_PER_BLOCK;
    if constexpr (results_per_batch == 0) {
      return 1;
    } else {
      constexpr auto count = 256u / results_per_batch;
      return count < 1 ? 1 : static_cast<std::uint8_t>(count);
    }
  }

  static constexpr auto CACHE_BATCHCOUNT = cache_batchcount();
  static constexpr auto CACHE_BLOCKCOUNT = std::uint16_t{CACHE_BATCHCOUNT * SIMD_WIDTH};
  static constexpr auto CACHE_SIZE = std::uint16_t{CACHE_BLOCKCOUNT * RESULTS_PER_BLOCK};

  counter_type m_counter;
  key_type m_key;
  alignas(simd_type::arch_type::alignment()) std::array<result_type, CACHE_SIZE> m_cache;
  std::uint16_t m_cache_index = CACHE_SIZE;

  explicit PRNG_ALWAYS_INLINE PhiloxState(const key_type &key, const counter_type &counter) noexcept
      : m_counter(counter), m_key(key) {}

  PRNG_ALWAYS_INLINE result_block_type next_block() noexcept {
    if (m_cache_index >= CACHE_SIZE) [[unlikely]] {
      gen_next_blocks_in_cache();
      m_cache_index = 0;
    }
    result_block_type blk;
    for (std::uint8_t i = 0; i < RESULTS_PER_BLOCK; ++i) {
      blk[i] = m_cache[m_cache_index + i];
    }
    m_cache_index += RESULTS_PER_BLOCK;
    return blk;
  }

  counter_type getCounter(bool prev) const noexcept {
    counter_type ctr = m_counter;
    if (m_cache_index < CACHE_SIZE || prev) {
      // m_counter points to start of next batch.
      // Current block = m_cache_index / RESULTS_PER_BLOCK
      auto blocks_consumed = m_cache_index / RESULTS_PER_BLOCK;
      auto blocks_to_subtract = static_cast<word_type>(CACHE_BLOCKCOUNT - blocks_consumed);
      if (prev) ++blocks_to_subtract;
      // Multi-word counter subtraction
      subtract_from_counter(ctr, blocks_to_subtract);
    }
    return ctr;
  }

  key_type getKey() const noexcept { return m_key; }

  void setState(const counter_type &ctr, const key_type &key) noexcept {
    m_counter = ctr;
    m_key = key;
    m_cache_index = CACHE_SIZE;
  }

private:
  using C = PhiloxConstants<N, W>;

  static PRNG_ALWAYS_INLINE void subtract_from_counter(counter_type &ctr, word_type amount) noexcept {
    // Subtract amount from multi-word counter (element 0 is least significant)
    word_type old = ctr[0];
    ctr[0] -= amount;
    if constexpr (N >= 2) {
      if (ctr[0] > old) { // borrow
        for (std::uint8_t i = 1; i < N; ++i) {
          if (ctr[i]-- != 0) break;
        }
      }
    }
  }

  static PRNG_ALWAYS_INLINE void advance_counter(counter_type &ctr, word_type amount) noexcept {
    word_type old = ctr[0];
    ctr[0] += amount;
    if constexpr (N >= 2) {
      if (ctr[0] < old) { // carry
        for (std::uint8_t i = 1; i < N; ++i) {
          if (++ctr[i] != 0) break;
        }
      }
    }
  }

  // SIMD mulhilo: widening W-bit multiply producing hi and lo halves.
  static PRNG_ALWAYS_INLINE void mulhilo_simd(simd_type a, word_type B,
                                                simd_type &hi, simd_type &lo) noexcept {
    auto hilo = xsimd::mulhilo(a, simd_type::broadcast(B));
    hi = hilo.first;
    lo = hilo.second;
  }

  static PRNG_ALWAYS_INLINE void simd_single_round(std::array<simd_type, N> &ctr,
                                                     key_type &key) noexcept {
    if constexpr (N == 4) {
      simd_type hi0, lo0, hi1, lo1;
      mulhilo_simd(ctr[0], C::M0, hi0, lo0);
      mulhilo_simd(ctr[2], C::M1, hi1, lo1);
      auto k0 = simd_type::broadcast(key[0]);
      auto k1 = simd_type::broadcast(key[1]);
      ctr[0] = hi1 ^ ctr[1] ^ k0;
      ctr[1] = lo1;
      ctr[2] = hi0 ^ ctr[3] ^ k1;
      ctr[3] = lo0;
      key[0] += C::W0;
      key[1] += C::W1;
    } else {
      simd_type hi, lo;
      mulhilo_simd(ctr[0], C::M0, hi, lo);
      auto k0 = simd_type::broadcast(key[0]);
      ctr[0] = hi ^ ctr[1] ^ k0;
      ctr[1] = lo;
      key[0] += C::W0;
    }
  }

  // Initialize SoA counter registers with per-lane offsets and carry propagation.
  // Each lane produces block (counter + lane_index). Higher elements receive
  // carry bits from the word_type addition overflow of lower elements.
  static PRNG_ALWAYS_INLINE void init_counter_batch(
      std::array<simd_type, N> &ctr_simd,
      const counter_type &counter) noexcept {
    alignas(simd_type::arch_type::alignment()) std::array<word_type, SIMD_WIDTH> offsets{};
    poet::static_for<0, SIMD_WIDTH>([&](auto I) {
      offsets[I] = static_cast<word_type>(I.value);
    });

    auto carry = simd_type::load_aligned(offsets.data());
    poet::static_for<0, N>([&](auto I) {
      auto base = simd_type::broadcast(counter[I]);
      auto val = base + carry;
      ctr_simd[I] = val;
      if constexpr (I + 1 < N) {
        // Detect overflow: val < base when carry caused wraparound
        carry = xsimd::select(val < base,
                              simd_type::broadcast(word_type{1}),
                              simd_type::broadcast(word_type{0}));
      }
    });
  }

  // Store SIMD_WIDTH blocks into cache at the given offset.
  // ctr_simd[i] contains element i across all SIMD_WIDTH lanes (SoA).
  // We need blocks in AoS: block0[0..N-1], block1[0..N-1], ...
  static PRNG_ALWAYS_INLINE void store_blocks_to_cache(result_type *cache,
                                     const std::array<simd_type, N> &ctr_simd) noexcept {
    if constexpr (W == 64 && N == SIMD_WIDTH) {
      // In-register SoA->AoS via xsimd::transpose (NxN square, uint64 lanes).
      std::array<simd_type, SIMD_WIDTH> regs;
      poet::static_for<0, N>([&](auto I) { regs[I] = ctr_simd[I]; });
      xsimd::transpose(regs.data(), regs.data() + SIMD_WIDTH);
      poet::static_for<0, SIMD_WIDTH>([&](auto Lane) {
        regs[Lane].store_aligned(cache + Lane * RESULTS_PER_BLOCK);
      });
    } else {
      // Store SoA registers to temps, scalar-pack into AoS cache.
      alignas(simd_type::arch_type::alignment()) std::array<word_type, SIMD_WIDTH> regs[N];
      poet::static_for<0, N>([&](auto I) {
        ctr_simd[I].store_aligned(regs[I].data());
      });

      for (std::uint8_t lane = 0; lane < SIMD_WIDTH; ++lane) {
        if constexpr (W == 32) {
          // Pack N uint32 values per lane into RESULTS_PER_BLOCK uint64 results.
          for (std::uint8_t k = 0; k < RESULTS_PER_BLOCK; ++k) {
            auto lo32 = static_cast<std::uint64_t>(regs[2 * k][lane]);
            auto hi32 = static_cast<std::uint64_t>(regs[2 * k + 1][lane]);
            cache[lane * RESULTS_PER_BLOCK + k] = lo32 | (hi32 << 32);
          }
        } else {
          for (std::uint8_t k = 0; k < N; ++k) {
            cache[lane * RESULTS_PER_BLOCK + k] = regs[k][lane];
          }
        }
      }
    }
  }

  // Generate one SIMD batch of blocks into the given cache pointer
  static PRNG_ALWAYS_INLINE void gen_block_batch(result_type *cache,
                                                  const counter_type &counter,
                                                  const key_type &key) noexcept {
    std::array<simd_type, N> ctr_simd;
    init_counter_batch(ctr_simd, counter);

    key_type round_key = key;
    poet::static_for<0, R>([&](auto) {
      simd_single_round(ctr_simd, round_key);
    });

    store_blocks_to_cache(cache, ctr_simd);
  }

  PRNG_ALWAYS_INLINE void gen_next_blocks_in_cache() noexcept {
    auto counter = m_counter;
    for (std::uint8_t batch = 0; batch < CACHE_BATCHCOUNT; ++batch) {
      gen_block_batch(m_cache.data() + batch * SIMD_WIDTH * RESULTS_PER_BLOCK,
                      counter, m_key);
      advance_counter(counter, static_cast<word_type>(SIMD_WIDTH));
    }
    m_counter = counter;
  }
};

// Dispatch result: function pointers for the type-erased PhiloxSIMD wrapper
template <std::uint8_t N, std::uint8_t W>
struct PhiloxSIMDInitResult {
  using word_type = std::conditional_t<W == 32, std::uint32_t, std::uint64_t>;
  using counter_type = std::array<word_type, N>;
  using key_type = std::array<word_type, N / 2>;
  using result_type = std::uint64_t;
  static constexpr auto RESULTS_PER_BLOCK = std::uint8_t{N * W / 64};
  using result_block_type = std::array<result_type, RESULTS_PER_BLOCK>;

  using next_block_fn = result_block_type (*)(void *) noexcept;
  using get_counter_fn = counter_type (*)(const void *, bool) noexcept;
  using get_key_fn = key_type (*)(const void *) noexcept;
  using set_state_fn = void (*)(void *, const counter_type &, const key_type &) noexcept;
  using get_cache_index_fn = std::uint16_t (*)(const void *) noexcept;

  next_block_fn next_block;
  get_counter_fn get_counter;
  get_key_fn get_key;
  set_state_fn set_state;
  get_cache_index_fn get_cache_index;
  std::size_t simd_size;
};

template <std::uint8_t N, std::uint8_t W, std::uint8_t R>
struct PhiloxSIMDInitFunctor {
  void *state_storage;
  using word_type = std::conditional_t<W == 32, std::uint32_t, std::uint64_t>;
  using counter_type = std::array<word_type, N>;
  using key_type = std::array<word_type, N / 2>;
  const key_type key;
  const counter_type counter;

  template <class Arch>
  PhiloxSIMDInitResult<N, W> operator()(Arch) const noexcept;
};

template <std::uint8_t N, std::uint8_t W, std::uint8_t R>
template <class Arch>
PhiloxSIMDInitResult<N, W> PhiloxSIMDInitFunctor<N, W, R>::operator()(Arch) const noexcept {
  using State = PhiloxState<Arch, N, W, R>;
  using InitResult = PhiloxSIMDInitResult<N, W>;
  static_assert(sizeof(State) <= 4096, "PhiloxState exceeds StateStorage capacity");
  static_assert(alignof(State) <= 64, "PhiloxState exceeds StateStorage alignment");
  new (state_storage) State(key, counter);
  return {
      +[](void *s) noexcept -> typename InitResult::result_block_type {
        return static_cast<State *>(s)->next_block();
      },
      +[](const void *s, bool prev) noexcept -> typename InitResult::counter_type {
        return static_cast<const State *>(s)->getCounter(prev);
      },
      +[](const void *s) noexcept -> typename InitResult::key_type {
        return static_cast<const State *>(s)->getKey();
      },
      +[](void *s, const typename InitResult::counter_type &ctr,
          const typename InitResult::key_type &key) noexcept {
        static_cast<State *>(s)->setState(ctr, key);
      },
      +[](const void *s) noexcept -> std::uint16_t {
        return static_cast<const State *>(s)->m_cache_index;
      },
      std::size_t{State::SIMD_WIDTH},
  };
}

// Extern template declarations for all NxW combos and architectures
#define PRNG_PHILOX_EXTERN_TEMPLATE(N, W, R, Arch)                                                  \
  extern template PRNG_EXPORT PhiloxSIMDInitResult<N, W>                                            \
  PhiloxSIMDInitFunctor<N, W, R>::operator()<Arch>(Arch) const noexcept

#define PRNG_PHILOX_EXTERN_TEMPLATES_FOR_ARCH(Arch)                                                 \
  PRNG_PHILOX_EXTERN_TEMPLATE(4, 32, 10, Arch);                                                    \
  PRNG_PHILOX_EXTERN_TEMPLATE(2, 32, 10, Arch);                                                    \
  PRNG_PHILOX_EXTERN_TEMPLATE(4, 64, 10, Arch);                                                    \
  PRNG_PHILOX_EXTERN_TEMPLATE(2, 64, 10, Arch)

#if PRNG_ARCH_X86_64
PRNG_PHILOX_EXTERN_TEMPLATES_FOR_ARCH(xsimd::sse2);
PRNG_PHILOX_EXTERN_TEMPLATES_FOR_ARCH(xsimd::avx2);
PRNG_PHILOX_EXTERN_TEMPLATES_FOR_ARCH(xsimd::avx512f);
#elif PRNG_ARCH_AARCH64
PRNG_PHILOX_EXTERN_TEMPLATES_FOR_ARCH(xsimd::neon64);
#  if XSIMD_WITH_SVE
PRNG_PHILOX_EXTERN_TEMPLATES_FOR_ARCH(xsimd::sve);
#  endif
#elif PRNG_ARCH_RISCV64
PRNG_PHILOX_EXTERN_TEMPLATES_FOR_ARCH(xsimd::detail::rvv<128>);
#endif

#undef PRNG_PHILOX_EXTERN_TEMPLATES_FOR_ARCH
#undef PRNG_PHILOX_EXTERN_TEMPLATE

} // namespace internal

// PhiloxSIMD: runtime SIMD dispatch via inline storage + function pointers
template <std::uint8_t N = 4, std::uint8_t W = 32, std::uint8_t R = 10>
class PhiloxSIMD {
public:
  using result_type = std::uint64_t;
  using word_type = std::conditional_t<W == 32, std::uint32_t, std::uint64_t>;
  using counter_type = std::array<word_type, N>;
  using key_type = std::array<word_type, N / 2>;
  static constexpr auto RESULTS_PER_BLOCK = std::uint8_t{N * W / 64};
  using result_block_type = std::array<result_type, RESULTS_PER_BLOCK>;

  static constexpr PRNG_ALWAYS_INLINE auto(min)() noexcept { return (std::numeric_limits<result_type>::min)(); }
  static constexpr PRNG_ALWAYS_INLINE auto(max)() noexcept { return (std::numeric_limits<result_type>::max)(); }

  static constexpr key_type seed_to_key(result_type seed) noexcept {
    return Philox<N, W, R>::seed_to_key(seed);
  }

  static constexpr counter_type counter_from_uint64(result_type counter) noexcept {
    return Philox<N, W, R>::counter_from_uint64(counter);
  }

  explicit PRNG_ALWAYS_INLINE PhiloxSIMD(result_type seed, result_type counter = 0) noexcept
      : PhiloxSIMD(seed_to_key(seed), counter_from_uint64(counter)) {}

  explicit PRNG_ALWAYS_INLINE PhiloxSIMD(key_type key, counter_type counter) noexcept {
    auto result = xsimd::dispatch<dispatch_arch_list>(
        internal::PhiloxSIMDInitFunctor<N, W, R>{m_state.data, key, counter})();
    m_next_block = result.next_block;
    m_get_counter = result.get_counter;
    m_get_key = result.get_key;
    m_set_state = result.set_state;
    m_get_cache_index = result.get_cache_index;
    m_simd_size = result.simd_size;
  }

  PRNG_ALWAYS_INLINE constexpr result_type operator()() noexcept {
    if (m_result_index >= RESULTS_PER_BLOCK) [[unlikely]] {
      m_result_cache = m_next_block(m_state.data);
      m_result_index = 0;
    }
    return m_result_cache[m_result_index++];
  }

  PRNG_ALWAYS_INLINE constexpr double uniform() noexcept {
    return static_cast<double>(operator()() >> 11) * 0x1.0p-53;
  }

  counter_type getCounter() const noexcept {
    return m_get_counter(m_state.data, m_result_index < RESULTS_PER_BLOCK);
  }

  key_type getKey() const noexcept { return m_get_key(m_state.data); }

  counter_type getCounterForSerde() const noexcept {
    return m_get_counter(m_state.data, false);
  }

  void setState(const counter_type &ctr, const key_type &key) noexcept {
    m_set_state(m_state.data, ctr, key);
  }

  const result_block_type &result_cache() const noexcept { return m_result_cache; }
  void set_result_cache(const result_block_type &cache) noexcept { m_result_cache = cache; }
  std::uint8_t result_index() const noexcept { return m_result_index; }
  void set_result_index(std::uint8_t idx) noexcept { m_result_index = idx; }

  PRNG_ALWAYS_INLINE std::size_t getSIMDSize() const noexcept { return m_simd_size; }

private:
  using InitResult = internal::PhiloxSIMDInitResult<N, W>;
  using next_block_fn = typename InitResult::next_block_fn;
  using get_counter_fn = typename InitResult::get_counter_fn;
  using get_key_fn = typename InitResult::get_key_fn;
  using set_state_fn = typename InitResult::set_state_fn;
  using get_cache_index_fn = typename InitResult::get_cache_index_fn;

  struct StateStorage {
    static constexpr std::size_t SIZE = 4096;
    static constexpr std::size_t ALIGN = 64;
    alignas(ALIGN) unsigned char data[SIZE];
  };

  alignas(64) StateStorage m_state;
  next_block_fn m_next_block = nullptr;
  get_counter_fn m_get_counter = nullptr;
  get_key_fn m_get_key = nullptr;
  set_state_fn m_set_state = nullptr;
  get_cache_index_fn m_get_cache_index = nullptr;
  std::size_t m_simd_size = 0;
  result_block_type m_result_cache{};
  std::uint8_t m_result_index = RESULTS_PER_BLOCK;
};

#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE
template <std::uint8_t N = 4, std::uint8_t W = 32, std::uint8_t R = 10>
class PhiloxNative {
public:
  using result_type = std::uint64_t;
  using word_type = std::conditional_t<W == 32, std::uint32_t, std::uint64_t>;
  using counter_type = std::array<word_type, N>;
  using key_type = std::array<word_type, N / 2>;
  static constexpr auto RESULTS_PER_BLOCK = std::uint8_t{N * W / 64};
  using result_block_type = std::array<result_type, RESULTS_PER_BLOCK>;

  static constexpr PRNG_ALWAYS_INLINE auto(min)() noexcept { return (std::numeric_limits<result_type>::min)(); }
  static constexpr PRNG_ALWAYS_INLINE auto(max)() noexcept { return (std::numeric_limits<result_type>::max)(); }

  explicit PhiloxNative(result_type seed, result_type counter = 0) noexcept
      : PhiloxNative(Philox<N, W, R>::seed_to_key(seed), Philox<N, W, R>::counter_from_uint64(counter)) {}

  PhiloxNative(key_type key, counter_type counter) noexcept
      : m_state(key, counter) {}

  PRNG_ALWAYS_INLINE constexpr result_type operator()() noexcept {
    if (m_result_index >= RESULTS_PER_BLOCK) [[unlikely]] {
      m_result_cache = m_state.next_block();
      m_result_index = 0;
    }
    return m_result_cache[m_result_index++];
  }

  PRNG_ALWAYS_INLINE constexpr double uniform() noexcept {
    return static_cast<double>(operator()() >> 11) * 0x1.0p-53;
  }

  counter_type getCounter() const noexcept {
    return m_state.getCounter(m_result_index < RESULTS_PER_BLOCK);
  }

  key_type getKey() const noexcept { return m_state.getKey(); }

  counter_type getCounterForSerde() const noexcept {
    return m_state.getCounter(false);
  }

  void setState(const counter_type &ctr, const key_type &key) noexcept {
    m_state.setState(ctr, key);
  }

  const result_block_type &result_cache() const noexcept { return m_result_cache; }
  void set_result_cache(const result_block_type &cache) noexcept { m_result_cache = cache; }
  std::uint8_t result_index() const noexcept { return m_result_index; }
  void set_result_index(std::uint8_t idx) noexcept { m_result_index = idx; }

  PRNG_ALWAYS_INLINE std::size_t getSIMDSize() const noexcept { return std::size_t{State::SIMD_WIDTH}; }

private:
  using State = internal::PhiloxState<xsimd::best_arch, N, W, R>;
  State m_state;
  result_block_type m_result_cache{};
  std::uint8_t m_result_index = RESULTS_PER_BLOCK;
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

} // namespace prng
