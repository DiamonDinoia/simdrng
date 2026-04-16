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

#include <poet/poet.hpp>

#include "dispatch_arch.hpp"
#include "macros.hpp"
#include "xoshiro_scalar.hpp"

namespace prng {

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
  static constexpr std::uint16_t CACHE_SIZE =
      std::numeric_limits<std::uint8_t>::max() + 1;

  alignas(simd_type::arch_type::alignment()) std::array<simd_type, RNG_WIDTH> s{};

  /**
   * Seed the SIMD state from a scalar seed, with optional thread and cluster offsets.
   */
  PRNG_ALWAYS_INLINE constexpr void seed(result_type seed_val, result_type thread_id = 0,
                                          result_type cluster_id = 0) noexcept {
    XoshiroScalar rng{seed_val};
    std::array<std::array<result_type, SIMD_WIDTH>, RNG_WIDTH> states{};
    for (auto i = 0UL; i < SIMD_WIDTH; ++i) {
      for (auto j = 0UL; j < RNG_WIDTH; ++j) {
        states[j][i] = rng.getState()[j];
      }
      rng.jump();
    }
    poet::static_for<0, RNG_WIDTH>([&](auto I) {
      s[I] = simd_type::load_unaligned(states[I].data());
    });
    for (result_type i = 0; i < thread_id; ++i) {
      mid_jump();
    }
    for (result_type i = 0; i < cluster_id; ++i) {
      long_jump();
    }
  }

  PRNG_ALWAYS_INLINE constexpr simd_type next() noexcept {
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

  PRNG_ALWAYS_INLINE constexpr void populate_cache(std::array<result_type, CACHE_SIZE> &cache) noexcept {
    poet::static_for<0, CACHE_SIZE / SIMD_WIDTH>([&](auto I) {
      next().store_aligned(cache.data() + I * SIMD_WIDTH);
    });
  }

  /**
   * Jump function. Equivalent to 2^128 calls to next().
   */
  PRNG_ALWAYS_INLINE constexpr void jump() noexcept {
    constexpr result_type JUMP[] = {0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c};
    simd_type s0(0), s1(0), s2(0), s3(0);
    for (const auto i : JUMP)
      for (auto b = 0; b < 64; b++) {
        if (i & result_type{1} << b) {
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
   * Mid-jump function. Equivalent to 2^160 calls to next().
   */
  PRNG_ALWAYS_INLINE constexpr void mid_jump() noexcept {
    constexpr result_type MID_JUMP[] = {0xc04b4f9c5d26c200, 0x69e6e6e431a2d40b, 0x4823b45b89dc689c,
                                         0xf567382197055bf0};
    simd_type s0(0), s1(0), s2(0), s3(0);
    for (const auto i : MID_JUMP)
      for (auto b = 0; b < 64; b++) {
        if (i & result_type{1} << b) {
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
   * Long-jump function. Equivalent to 2^192 calls to next().
   */
  PRNG_ALWAYS_INLINE constexpr void long_jump() noexcept {
    constexpr result_type LONG_JUMP[] = {0x76e15d3efefdcbbf, 0xc5004e441c522fb3, 0x77710069854ee241,
                                          0x39109bb02acbe635};
    simd_type s0(0), s1(0), s2(0), s3(0);
    for (const auto i : LONG_JUMP)
      for (auto b = 0; b < 64; b++) {
        if (i & result_type{1} << b) {
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

  PRNG_ALWAYS_INLINE constexpr std::array<result_type, RNG_WIDTH> getState(const std::size_t index) const noexcept {
    std::array<result_type, RNG_WIDTH> state{};
    poet::static_for<0, RNG_WIDTH>([&](auto I) {
      state[I] = s[I].get(index);
    });
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
  using populate_fn = void (*)(void *, std::array<std::uint64_t, 256> &) noexcept;
  using jump_fn = void (*)(void *) noexcept;
  using get_state_fn = void (*)(const void *, std::uint64_t *) noexcept;
  using set_state_fn = void (*)(void *, const std::uint64_t *) noexcept;
  using simd_width_fn = std::uint8_t (*)() noexcept;
  populate_fn populate_cache;
  jump_fn jump;
  jump_fn mid_jump;
  jump_fn long_jump;
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

  template <class Arch> XoshiroSIMDInitResult operator()(Arch) const noexcept;
};

template <class Arch> XoshiroSIMDInitResult XoshiroSIMDInitFunctor::operator()(Arch) const noexcept {
  using State = XoshiroState<Arch>;
  static_assert(sizeof(State) <= 256, "XoshiroState exceeds StateStorage capacity");
  static_assert(alignof(State) <= 64, "XoshiroState exceeds StateStorage alignment");
  auto *state = new (state_storage) State{};
  state->seed(seed, thread_id, cluster_id);
  return {
      +[](void *s, std::array<std::uint64_t, 256> &cache) noexcept { static_cast<State *>(s)->populate_cache(cache); },
      +[](void *s) noexcept { static_cast<State *>(s)->jump(); },
      +[](void *s) noexcept { static_cast<State *>(s)->mid_jump(); },
      +[](void *s) noexcept { static_cast<State *>(s)->long_jump(); },
      +[](const void *s, std::uint64_t *out) noexcept { static_cast<const State *>(s)->get_flat_state(out); },
      +[](void *s, const std::uint64_t *in) noexcept { static_cast<State *>(s)->set_flat_state(in); },
      +[]() noexcept -> std::uint8_t { return State::SIMD_WIDTH; },
  };
}

#if PRNG_ARCH_X86_64
extern template PRNG_EXPORT XoshiroSIMDInitResult XoshiroSIMDInitFunctor::operator()<xsimd::sse2>(xsimd::sse2) const noexcept;
extern template PRNG_EXPORT XoshiroSIMDInitResult XoshiroSIMDInitFunctor::operator()<xsimd::avx2>(xsimd::avx2) const noexcept;
extern template PRNG_EXPORT XoshiroSIMDInitResult
XoshiroSIMDInitFunctor::operator()<xsimd::avx512f>(xsimd::avx512f) const noexcept;
#elif PRNG_ARCH_AARCH64
extern template PRNG_EXPORT XoshiroSIMDInitResult
XoshiroSIMDInitFunctor::operator()<xsimd::neon64>(xsimd::neon64) const noexcept;
#  if XSIMD_WITH_SVE
extern template PRNG_EXPORT XoshiroSIMDInitResult
XoshiroSIMDInitFunctor::operator()<xsimd::sve>(xsimd::sve) const noexcept;
#  endif
#elif PRNG_ARCH_RISCV64
extern template PRNG_EXPORT XoshiroSIMDInitResult
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
  static constexpr PRNG_ALWAYS_INLINE auto(min)() noexcept { return (std::numeric_limits<result_type>::min)(); }
  static constexpr PRNG_ALWAYS_INLINE auto(max)() noexcept { return (std::numeric_limits<result_type>::max)(); }
  static constexpr PRNG_ALWAYS_INLINE auto stateSize() noexcept { return State::RNG_WIDTH; }

  PRNG_ALWAYS_INLINE explicit XoshiroNative(const result_type seed) noexcept { m_state.seed(seed); }
  PRNG_ALWAYS_INLINE explicit XoshiroNative(const result_type seed, const result_type thread_id) noexcept {
    m_state.seed(seed, thread_id);
  }
  PRNG_ALWAYS_INLINE explicit XoshiroNative(const result_type seed, const result_type thread_id,
                                            const result_type cluster_id) noexcept {
    m_state.seed(seed, thread_id, cluster_id);
  }

  PRNG_ALWAYS_INLINE result_type operator()() noexcept {
    if (m_index == 0) [[unlikely]] {
      m_state.populate_cache(m_cache);
    }
    return m_cache[m_index++];
  }

  PRNG_ALWAYS_INLINE double uniform() noexcept {
    return static_cast<double>(operator()() >> 11) * 0x1.0p-53;
  }

  PRNG_ALWAYS_INLINE auto getState(const std::size_t index) const noexcept { return m_state.getState(index); }

  PRNG_ALWAYS_INLINE void jump() noexcept { m_state.jump(); }
  PRNG_ALWAYS_INLINE void mid_jump() noexcept { m_state.mid_jump(); }
  PRNG_ALWAYS_INLINE void long_jump() noexcept { m_state.long_jump(); }

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
  static constexpr PRNG_ALWAYS_INLINE result_type(min)() noexcept { return (std::numeric_limits<result_type>::min)(); }
  static constexpr PRNG_ALWAYS_INLINE result_type(max)() noexcept { return (std::numeric_limits<result_type>::max)(); }

  PRNG_EXPORT explicit XoshiroSIMD(result_type seed, result_type thread_id = 0, result_type cluster_id = 0) noexcept;

  PRNG_ALWAYS_INLINE result_type operator()() noexcept {
    if (m_index == 0) [[unlikely]] {
      m_populate_cache(m_state.data, m_cache);
    }
    return m_cache[m_index++];
  }

  PRNG_ALWAYS_INLINE double uniform() noexcept {
    return static_cast<double>(operator()() >> 11) * 0x1.0p-53;
  }

  PRNG_ALWAYS_INLINE void jump() noexcept { m_jump(m_state.data); }
  PRNG_ALWAYS_INLINE void mid_jump() noexcept { m_mid_jump(m_state.data); }
  PRNG_ALWAYS_INLINE void long_jump() noexcept { m_long_jump(m_state.data); }

  void get_flat_state(result_type *out) const noexcept { m_get_state(m_state.data, out); }
  void set_flat_state(const result_type *in) noexcept { m_set_state(m_state.data, in); }
  std::uint8_t simd_width() const noexcept { return m_simd_width(); }

  std::uint8_t cache_index() const noexcept { return m_index; }
  void set_cache_index(std::uint8_t idx) noexcept { m_index = idx; }
  const std::array<result_type, std::numeric_limits<std::uint8_t>::max() + 1> &cache() const noexcept { return m_cache; }
  std::array<result_type, std::numeric_limits<std::uint8_t>::max() + 1> &cache() noexcept { return m_cache; }

protected:
  static constexpr std::uint16_t CACHE_SIZE = std::numeric_limits<std::uint8_t>::max() + 1;
  using populate_fn = void (*)(void *, std::array<result_type, CACHE_SIZE> &) noexcept;
  using jump_fn = void (*)(void *) noexcept;
  using get_state_fn = void (*)(const void *, result_type *) noexcept;
  using set_state_fn = void (*)(void *, const result_type *) noexcept;
  using simd_width_fn = std::uint8_t (*)() noexcept;

  // Raw byte storage for the arch-specific XoshiroState.
  // Typed union is not viable: xsimd batch types have different sizeof
  // across TUs compiled with different -march flags (ODR divergence).
  // Max is avx512f: 4 × sizeof(__m512i) = 4 × 64 = 256 bytes, align 64.
  struct StateStorage {
    static constexpr std::size_t SIZE = 256;
    static constexpr std::size_t ALIGN = 64;
    alignas(ALIGN) unsigned char data[SIZE];
  };

  alignas(64) std::array<result_type, CACHE_SIZE> m_cache{};
  alignas(64) StateStorage m_state;
  populate_fn m_populate_cache = nullptr;
  jump_fn m_jump = nullptr;
  jump_fn m_mid_jump = nullptr;
  jump_fn m_long_jump = nullptr;
  get_state_fn m_get_state = nullptr;
  set_state_fn m_set_state = nullptr;
  simd_width_fn m_simd_width = nullptr;
  std::uint8_t m_index{0};
};

} // namespace prng
