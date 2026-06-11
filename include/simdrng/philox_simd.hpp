#pragma once

#if SIMDRNG_WITH_XSIMD

#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>

#include <poet/poet.hpp>

#include "dispatch_arch.hpp"
#include "macros.hpp"
#include "philox.hpp"

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
    const std::size_t vreg = poet::vector_register_count();
    const std::size_t scratch = (W >= 64) ? vreg / 2 : 0;
    std::size_t k = (vreg - scratch) / N;
    if (k < 1)
      k = 1;
    if (k > BATCHES_PER_CACHE)
      k = BATCHES_PER_CACHE;
    return static_cast<std::uint16_t>(k);
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
      // dynamic_for fully unrolls each (lane, k, wd) — every bound is a constant,
      // so Unroll == count and the block expands at compile time — while threading
      // a *runtime* running index for the addressing. The runtime indices keep the
      // memory math a single running offset (no spilling SIMD_WIDTH*RESULTS_PER_BLOCK
      // live compile-time-indexed values) and keep the chained regs[..][..] subscript
      // over built-in integers, which MSVC mis-parses when fed integral_constant
      // arithmetic. The innermost uses the (lane, index) form so the compile-time
      // `wd` lane still folds the shift while `wd_idx` addresses the word.
      poet::dynamic_for<SIMD_WIDTH>(std::size_t{SIMD_WIDTH}, [&](std::size_t lane) SIMDRNG_ALWAYS_INLINE_LAMBDA {
        poet::dynamic_for<RESULTS_PER_BLOCK>(
            std::size_t{RESULTS_PER_BLOCK}, [&](std::size_t k) SIMDRNG_ALWAYS_INLINE_LAMBDA {
              result_type acc = 0;
              poet::dynamic_for<WORDS_PER_RESULT>(
                  std::size_t{WORDS_PER_RESULT}, [&](auto wd, std::size_t wd_idx) SIMDRNG_ALWAYS_INLINE_LAMBDA {
                    acc |= static_cast<result_type>(regs[k * WORDS_PER_RESULT + wd_idx][lane]) << (wd * W);
                  });
              cache[lane * RESULTS_PER_BLOCK + k] = acc;
            });
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
      +[](void *s, const typename InitResult::counter_type &ctr, const typename InitResult::key_type &key) noexcept {
        static_cast<State *>(s)->setState(ctr, key);
      },
      std::size_t{State::SIMD_WIDTH},
  };
}

// Extern template declarations for all NxW combos and architectures
#define SIMDRNG_PHILOX_EXTERN_TEMPLATE(N, W, R, Arch)                                                                  \
  extern template SIMDRNG_EXPORT PhiloxSIMDInitResult<N, W> PhiloxSIMDInitFunctor<N, W, R>::operator()<Arch>(Arch)     \
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
#endif // SIMDRNG_WITH_XSIMD
