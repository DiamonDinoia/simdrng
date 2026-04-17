#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>
#include <numpy/random/bitgen.h>
#include <cstring>
#include <type_traits>

#include <poet/poet.hpp>

#include "random/macros.hpp"
#include "random/splitmix.hpp"
#include "random/xoshiro_scalar.hpp"
#include "random/xoshiro_simd.hpp"
#include "random/chacha_simd.hpp"
#include "random/philox_simd.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {
constexpr double kInvPow53 = 0x1.0p-53;

// ---------------------------------------------------------------------------
// Type traits for dispatch in bulk fill and state serialization
template <typename T> struct is_xoshiro_cached : std::false_type {};
template <> struct is_xoshiro_cached<prng::XoshiroSIMD> : std::true_type {};
#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE
template <> struct is_xoshiro_cached<prng::XoshiroNative> : std::true_type {};
#endif

template <typename T> struct is_chacha : std::false_type {};
template <std::uint8_t R> struct is_chacha<prng::ChaChaSIMD<R>> : std::true_type {};
#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE
template <std::uint8_t R> struct is_chacha<prng::ChaChaNative<R>> : std::true_type {};
#endif

template <typename T> struct is_philox : std::false_type {};
template <std::uint8_t N, std::uint8_t W, std::uint8_t R>
struct is_philox<prng::PhiloxSIMD<N, W, R>> : std::true_type {};
#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE
template <std::uint8_t N, std::uint8_t W, std::uint8_t R>
struct is_philox<prng::PhiloxNative<N, W, R>> : std::true_type {};
#endif

// ---------------------------------------------------------------------------
// PyBitGenerator<Rng>: single template replacing DirectBitGen + per-gen wrappers.
// Contains: Rng instance + bitgen_t (stable address, no heap indirection).
// No dcache — generators use their own internal caches.
template <typename Rng>
struct PyBitGenerator {
  Rng rng;
  bitgen_t bitgen;

  template <typename... Args>
  explicit PyBitGenerator(Args &&...args)
      : rng(std::forward<Args>(args)...), bitgen{} {
    bitgen.state = this;
    bitgen.next_uint64 = &next_u64;
    bitgen.next_uint32 = &next_u32;
    bitgen.next_double = &next_f64;
    bitgen.next_raw = &next_u64;
  }

  // Static callbacks — called by NumPy's Generator per-sample.
  // No virtual dispatch, no lambda thunks.
  static uint64_t next_u64(void *s) noexcept {
    return static_cast<PyBitGenerator *>(s)->rng();
  }
  static uint32_t next_u32(void *s) noexcept {
    return static_cast<uint32_t>(static_cast<PyBitGenerator *>(s)->rng() >> 32);
  }
  static double next_f64(void *s) noexcept {
    return static_cast<double>(static_cast<PyBitGenerator *>(s)->rng() >> 11) *
           kInvPow53;
  }

  // Non-owning capsule — Python wrapper prevents GC via reference.
  nb::object capsule() {
    return nb::steal(PyCapsule_New(&bitgen, "BitGenerator", nullptr));
  }

  uint64_t random_raw() noexcept { return rng(); }

  // -- Bulk fill: float64 --------------------------------------------------
  void fill_uniform(
      nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::c_contig> out) noexcept {
    nb::gil_scoped_release release;
    auto *data = out.data();
    const std::size_t n = out.size();
    if constexpr (is_xoshiro_cached<Rng>::value) {
      fill_uniform_cached(data, n);
    } else {
      for (std::size_t i = 0; i < n; ++i)
        data[i] = static_cast<double>(rng() >> 11) * kInvPow53;
    }
  }

  // -- Bulk fill: uint64 ---------------------------------------------------
  void fill_uint64(
      nb::ndarray<nb::numpy, uint64_t, nb::ndim<1>, nb::c_contig> out) noexcept {
    nb::gil_scoped_release release;
    auto *data = out.data();
    const std::size_t n = out.size();
    if constexpr (is_xoshiro_cached<Rng>::value) {
      fill_uint64_cached(data, n);
    } else {
      for (std::size_t i = 0; i < n; ++i)
        data[i] = rng();
    }
  }

  // -- Bulk fill: float32 (2 samples per uint64) ---------------------------
  void fill_float32(
      nb::ndarray<nb::numpy, float, nb::ndim<1>, nb::c_contig> out) noexcept {
    nb::gil_scoped_release release;
    auto *data = out.data();
    const std::size_t n = out.size();
    constexpr float kScale = 1.0f / 16777216.0f; // 2^-24
    if constexpr (is_xoshiro_cached<Rng>::value) {
      fill_float32_cached(data, n);
    } else {
      // Generic: 2 float32 per uint64
      std::size_t i = 0;
      for (; i + 1 < n; i += 2) {
        uint64_t raw = rng();
        data[i] = static_cast<float>(static_cast<uint32_t>(raw >> 32) >> 8) *
                  kScale;
        data[i + 1] =
            static_cast<float>(static_cast<uint32_t>(raw) >> 8) * kScale;
      }
      if (i < n)
        data[i] =
            static_cast<float>(static_cast<uint32_t>(rng() >> 32) >> 8) *
            kScale;
    }
  }

  // -- Bulk fill: uint32 (2 samples per uint64) ----------------------------
  void fill_uint32(
      nb::ndarray<nb::numpy, uint32_t, nb::ndim<1>, nb::c_contig> out) noexcept {
    nb::gil_scoped_release release;
    auto *data = out.data();
    const std::size_t n = out.size();
    if constexpr (is_xoshiro_cached<Rng>::value) {
      fill_uint32_cached(data, n);
    } else {
      std::size_t i = 0;
      for (; i + 1 < n; i += 2) {
        uint64_t raw = rng();
        data[i] = static_cast<uint32_t>(raw >> 32);
        data[i + 1] = static_cast<uint32_t>(raw);
      }
      if (i < n) data[i] = static_cast<uint32_t>(rng() >> 32);
    }
  }

  // -- State serialization -------------------------------------------------
  nb::dict get_state() const;
  void set_state(nb::dict d);

private:
  // Cache-drain bulk fill for Xoshiro SIMD/Native (256-element cache).
  void fill_uniform_cached(double *out, std::size_t n) noexcept {
    constexpr std::size_t CACHE_SIZE = 256;
    std::size_t produced = 0;
    while (produced < n) {
      auto idx = rng.cache_index();
      if (idx == 0) {
        out[produced++] = static_cast<double>(rng() >> 11) * kInvPow53;
        idx = rng.cache_index();
        if (produced >= n) return;
      }
      const std::size_t available = CACHE_SIZE - idx;
      const std::size_t to_copy =
          (n - produced < available) ? (n - produced) : available;
      const auto &cache = rng.cache();
      poet::dynamic_for<4>(std::size_t{0}, to_copy, [&](std::size_t i) {
        out[produced + i] =
            static_cast<double>(cache[idx + i] >> 11) * kInvPow53;
      });
      rng.set_cache_index(static_cast<std::uint8_t>(idx + to_copy));
      produced += to_copy;
    }
  }

  void fill_float32_cached(float *out, std::size_t n) noexcept {
    constexpr std::size_t CACHE_SIZE = 256;
    constexpr float kScale = 1.0f / 16777216.0f;
    std::size_t produced = 0;
    while (produced < n) {
      auto idx = rng.cache_index();
      if (idx == 0) {
        uint64_t raw = rng();
        idx = rng.cache_index();
        data_from_u64_f32(out, produced, n, raw, kScale);
        if (produced >= n) return;
      }
      const std::size_t available = CACHE_SIZE - idx;
      const std::size_t need = (n - produced + 1) / 2;
      const std::size_t entries =
          (need < available) ? need : available;
      const auto &cache = rng.cache();
      for (std::size_t i = 0; i < entries; ++i) {
        data_from_u64_f32(out, produced, n, cache[idx + i], kScale);
      }
      rng.set_cache_index(static_cast<std::uint8_t>(idx + entries));
    }
  }

  void fill_uint32_cached(uint32_t *out, std::size_t n) noexcept {
    constexpr std::size_t CACHE_SIZE = 256;
    std::size_t produced = 0;
    while (produced < n) {
      auto idx = rng.cache_index();
      if (idx == 0) {
        uint64_t raw = rng();
        idx = rng.cache_index();
        data_from_u64_u32(out, produced, n, raw);
        if (produced >= n) return;
      }
      const std::size_t available = CACHE_SIZE - idx;
      const std::size_t need = (n - produced + 1) / 2;
      const std::size_t entries =
          (need < available) ? need : available;
      const auto &cache = rng.cache();
      for (std::size_t i = 0; i < entries; ++i) {
        data_from_u64_u32(out, produced, n, cache[idx + i]);
      }
      rng.set_cache_index(static_cast<std::uint8_t>(idx + entries));
    }
  }

  static void data_from_u64_f32(float *out, std::size_t &pos, std::size_t n,
                                 uint64_t raw, float scale) noexcept {
    out[pos++] =
        static_cast<float>(static_cast<uint32_t>(raw >> 32) >> 8) * scale;
    if (pos < n)
      out[pos++] =
          static_cast<float>(static_cast<uint32_t>(raw) >> 8) * scale;
  }

  static void data_from_u64_u32(uint32_t *out, std::size_t &pos,
                                 std::size_t n, uint64_t raw) noexcept {
    out[pos++] = static_cast<uint32_t>(raw >> 32);
    if (pos < n) out[pos++] = static_cast<uint32_t>(raw);
  }

  void fill_uint64_cached(uint64_t *out, std::size_t n) noexcept {
    constexpr std::size_t CACHE_SIZE = 256;
    std::size_t produced = 0;
    while (produced < n) {
      auto idx = rng.cache_index();
      if (idx == 0) {
        out[produced++] = rng();
        idx = rng.cache_index();
        if (produced >= n) return;
      }
      const std::size_t available = CACHE_SIZE - idx;
      const std::size_t to_copy =
          (n - produced < available) ? (n - produced) : available;
      std::memcpy(out + produced, rng.cache().data() + idx,
                  to_copy * sizeof(uint64_t));
      rng.set_cache_index(static_cast<std::uint8_t>(idx + to_copy));
      produced += to_copy;
    }
  }
};

// ---------------------------------------------------------------------------
// State serialization: SplitMix
template <>
nb::dict PyBitGenerator<prng::SplitMix>::get_state() const {
  nb::dict d;
  d["s"] = rng.getState();
  return d;
}
template <>
void PyBitGenerator<prng::SplitMix>::set_state(nb::dict d) {
  rng.setState(nb::cast<uint64_t>(d["s"]));
}

// ---------------------------------------------------------------------------
// State serialization: XoshiroScalar
template <>
nb::dict PyBitGenerator<prng::XoshiroScalar>::get_state() const {
  auto s = rng.getState();
  nb::dict d;
  d["s"] = std::vector<uint64_t>(s.begin(), s.end());
  return d;
}
template <>
void PyBitGenerator<prng::XoshiroScalar>::set_state(nb::dict d) {
  auto v = nb::cast<std::vector<uint64_t>>(d["s"]);
  std::array<uint64_t, 4> s;
  for (int i = 0; i < 4; ++i) s[i] = v[i];
  rng.setState(s);
}

// ---------------------------------------------------------------------------
// State serialization helper for Xoshiro SIMD/Native
template <typename Rng>
nb::dict get_xoshiro_cached_state(const Rng &rng) {
  const auto sw = rng.simd_width();
  const std::size_t state_len = 4 * sw;
  std::vector<uint64_t> flat(state_len);
  rng.get_flat_state(flat.data());
  const auto &cache = rng.cache();
  std::vector<uint64_t> cache_vec(cache.begin(), cache.end());
  nb::dict d;
  d["s"] = std::move(flat);
  d["cache"] = std::move(cache_vec);
  d["cache_index"] = rng.cache_index();
  d["simd_width"] = sw;
  return d;
}

template <typename Rng>
void set_xoshiro_cached_state(Rng &rng, nb::dict d) {
  auto flat = nb::cast<std::vector<uint64_t>>(d["s"]);
  rng.set_flat_state(flat.data());
  auto cache_vec = nb::cast<std::vector<uint64_t>>(d["cache"]);
  auto &cache = rng.cache();
  for (std::size_t i = 0; i < 256 && i < cache_vec.size(); ++i)
    cache[i] = cache_vec[i];
  rng.set_cache_index(nb::cast<uint8_t>(d["cache_index"]));
}

// XoshiroSIMD
template <>
nb::dict PyBitGenerator<prng::XoshiroSIMD>::get_state() const {
  return get_xoshiro_cached_state(rng);
}
template <>
void PyBitGenerator<prng::XoshiroSIMD>::set_state(nb::dict d) {
  set_xoshiro_cached_state(rng, d);
}

// XoshiroNative
#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE
template <>
nb::dict PyBitGenerator<prng::XoshiroNative>::get_state() const {
  return get_xoshiro_cached_state(rng);
}
template <>
void PyBitGenerator<prng::XoshiroNative>::set_state(nb::dict d) {
  set_xoshiro_cached_state(rng, d);
}
#endif

// ---------------------------------------------------------------------------
// State serialization helper for ChaCha SIMD/Native
template <typename Rng>
nb::dict get_chacha_state(const Rng &rng) {
  auto matrix = rng.getStateForSerde();
  auto rc = rng.result_cache();
  nb::dict d;
  d["matrix"] = std::vector<uint32_t>(matrix.begin(), matrix.end());
  d["result_cache"] = std::vector<uint64_t>(rc.begin(), rc.end());
  d["result_index"] = rng.result_index();
  return d;
}

template <typename Rng>
void set_chacha_state(Rng &rng, nb::dict d) {
  auto mv = nb::cast<std::vector<uint32_t>>(d["matrix"]);
  std::array<uint32_t, 16> matrix;
  for (int i = 0; i < 16; ++i) matrix[i] = mv[i];
  rng.setState(matrix);
  auto rcv = nb::cast<std::vector<uint64_t>>(d["result_cache"]);
  typename Rng::result_cache_type rc;
  for (std::size_t i = 0; i < rc.size() && i < rcv.size(); ++i) rc[i] = rcv[i];
  rng.set_result_cache(rc);
  rng.set_result_index(nb::cast<uint8_t>(d["result_index"]));
}

// ChaChaSIMD<8>, <12>, <20>
template <>
nb::dict PyBitGenerator<prng::ChaChaSIMD<8>>::get_state() const {
  return get_chacha_state(rng);
}
template <>
void PyBitGenerator<prng::ChaChaSIMD<8>>::set_state(nb::dict d) {
  set_chacha_state(rng, d);
}
template <>
nb::dict PyBitGenerator<prng::ChaChaSIMD<12>>::get_state() const {
  return get_chacha_state(rng);
}
template <>
void PyBitGenerator<prng::ChaChaSIMD<12>>::set_state(nb::dict d) {
  set_chacha_state(rng, d);
}
template <>
nb::dict PyBitGenerator<prng::ChaChaSIMD<20>>::get_state() const {
  return get_chacha_state(rng);
}
template <>
void PyBitGenerator<prng::ChaChaSIMD<20>>::set_state(nb::dict d) {
  set_chacha_state(rng, d);
}

// ChaChaNative<8>, <12>, <20>
#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE
template <>
nb::dict PyBitGenerator<prng::ChaChaNative<8>>::get_state() const {
  return get_chacha_state(rng);
}
template <>
void PyBitGenerator<prng::ChaChaNative<8>>::set_state(nb::dict d) {
  set_chacha_state(rng, d);
}
template <>
nb::dict PyBitGenerator<prng::ChaChaNative<12>>::get_state() const {
  return get_chacha_state(rng);
}
template <>
void PyBitGenerator<prng::ChaChaNative<12>>::set_state(nb::dict d) {
  set_chacha_state(rng, d);
}
template <>
nb::dict PyBitGenerator<prng::ChaChaNative<20>>::get_state() const {
  return get_chacha_state(rng);
}
template <>
void PyBitGenerator<prng::ChaChaNative<20>>::set_state(nb::dict d) {
  set_chacha_state(rng, d);
}
#endif

// ---------------------------------------------------------------------------
// State serialization helper for Philox SIMD/Native
template <typename Rng>
nb::dict get_philox_state(const Rng &rng) {
  using word_type = typename Rng::word_type;
  auto ctr = rng.getCounterForSerde();
  auto key = rng.getKey();
  const auto &cache = rng.cache();
  nb::dict d;
  d["counter"] = std::vector<word_type>(ctr.begin(), ctr.end());
  d["key"] = std::vector<word_type>(key.begin(), key.end());
  d["cache"] = std::vector<uint64_t>(cache.begin(), cache.end());
  d["cache_index"] = rng.cache_index();
  return d;
}

template <typename Rng>
void set_philox_state(Rng &rng, nb::dict d) {
  using word_type = typename Rng::word_type;
  auto cv = nb::cast<std::vector<word_type>>(d["counter"]);
  auto kv = nb::cast<std::vector<word_type>>(d["key"]);
  typename Rng::counter_type ctr;
  typename Rng::key_type key;
  for (std::size_t i = 0; i < ctr.size() && i < cv.size(); ++i) ctr[i] = cv[i];
  for (std::size_t i = 0; i < key.size() && i < kv.size(); ++i) key[i] = kv[i];
  rng.setState(ctr, key);
  auto cache_vec = nb::cast<std::vector<uint64_t>>(d["cache"]);
  auto &cache = rng.cache();
  for (std::size_t i = 0; i < cache.size() && i < cache_vec.size(); ++i)
    cache[i] = cache_vec[i];
  rng.set_cache_index(nb::cast<uint8_t>(d["cache_index"]));
}

// Philox SIMD state specializations for all four NxW combos
using Philox4x32SIMD_t = prng::PhiloxSIMD<4, 32, 10>;
using Philox2x32SIMD_t = prng::PhiloxSIMD<2, 32, 10>;
using Philox4x64SIMD_t = prng::PhiloxSIMD<4, 64, 10>;
using Philox2x64SIMD_t = prng::PhiloxSIMD<2, 64, 10>;

template <> nb::dict PyBitGenerator<Philox4x32SIMD_t>::get_state() const { return get_philox_state(rng); }
template <> void PyBitGenerator<Philox4x32SIMD_t>::set_state(nb::dict d) { set_philox_state(rng, d); }
template <> nb::dict PyBitGenerator<Philox2x32SIMD_t>::get_state() const { return get_philox_state(rng); }
template <> void PyBitGenerator<Philox2x32SIMD_t>::set_state(nb::dict d) { set_philox_state(rng, d); }
template <> nb::dict PyBitGenerator<Philox4x64SIMD_t>::get_state() const { return get_philox_state(rng); }
template <> void PyBitGenerator<Philox4x64SIMD_t>::set_state(nb::dict d) { set_philox_state(rng, d); }
template <> nb::dict PyBitGenerator<Philox2x64SIMD_t>::get_state() const { return get_philox_state(rng); }
template <> void PyBitGenerator<Philox2x64SIMD_t>::set_state(nb::dict d) { set_philox_state(rng, d); }

#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE
using Philox4x32Native_t = prng::PhiloxNative<4, 32, 10>;
using Philox2x32Native_t = prng::PhiloxNative<2, 32, 10>;
using Philox4x64Native_t = prng::PhiloxNative<4, 64, 10>;
using Philox2x64Native_t = prng::PhiloxNative<2, 64, 10>;

template <> nb::dict PyBitGenerator<Philox4x32Native_t>::get_state() const { return get_philox_state(rng); }
template <> void PyBitGenerator<Philox4x32Native_t>::set_state(nb::dict d) { set_philox_state(rng, d); }
template <> nb::dict PyBitGenerator<Philox2x32Native_t>::get_state() const { return get_philox_state(rng); }
template <> void PyBitGenerator<Philox2x32Native_t>::set_state(nb::dict d) { set_philox_state(rng, d); }
template <> nb::dict PyBitGenerator<Philox4x64Native_t>::get_state() const { return get_philox_state(rng); }
template <> void PyBitGenerator<Philox4x64Native_t>::set_state(nb::dict d) { set_philox_state(rng, d); }
template <> nb::dict PyBitGenerator<Philox2x64Native_t>::get_state() const { return get_philox_state(rng); }
template <> void PyBitGenerator<Philox2x64Native_t>::set_state(nb::dict d) { set_philox_state(rng, d); }
#endif

// ---------------------------------------------------------------------------
// Registration helpers to reduce boilerplate.

// Base registration: capsule, random_raw, state, fill methods.
template <typename Rng, typename Class>
void register_base(Class &cls) {
  cls.def("capsule", &PyBitGenerator<Rng>::capsule)
      .def("random_raw", &PyBitGenerator<Rng>::random_raw)
      .def("get_state", &PyBitGenerator<Rng>::get_state)
      .def("set_state", &PyBitGenerator<Rng>::set_state, "state"_a)
      .def("fill_uniform", &PyBitGenerator<Rng>::fill_uniform, "out"_a)
      .def("fill_float32", &PyBitGenerator<Rng>::fill_float32, "out"_a)
      .def("fill_uint64", &PyBitGenerator<Rng>::fill_uint64, "out"_a)
      .def("fill_uint32", &PyBitGenerator<Rng>::fill_uint32, "out"_a);
}

// Jump registration for Xoshiro family.
template <typename Rng>
struct has_jump : std::false_type {};
template <>
struct has_jump<prng::XoshiroScalar> : std::true_type {};
template <>
struct has_jump<prng::XoshiroSIMD> : std::true_type {};
#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE
template <>
struct has_jump<prng::XoshiroNative> : std::true_type {};
#endif

template <typename Rng, typename Class>
void register_jump(Class &cls) {
  if constexpr (has_jump<Rng>::value) {
    cls.def("jump", [](PyBitGenerator<Rng> &self) { self.rng.jump(); })
        .def("long_jump",
             [](PyBitGenerator<Rng> &self) { self.rng.long_jump(); });
  }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Python module
NB_MODULE(simdrng_ext, m) {
  using PySplitMix = PyBitGenerator<prng::SplitMix>;
  auto sm = nb::class_<PySplitMix>(m, "_SplitMix")
                .def(nb::init<uint64_t>(), "seed"_a);
  register_base<prng::SplitMix>(sm);

  using PyXoshiro = PyBitGenerator<prng::XoshiroScalar>;
  auto xo = nb::class_<PyXoshiro>(m, "_Xoshiro")
                .def(nb::init<uint64_t>(), "seed"_a)
                .def(nb::init<uint64_t, uint64_t>(), "seed"_a, "thread"_a)
                .def(nb::init<uint64_t, uint64_t, uint64_t>(), "seed"_a,
                     "thread"_a, "cluster"_a);
  register_base<prng::XoshiroScalar>(xo);
  register_jump<prng::XoshiroScalar>(xo);

  using PyXoshiroSIMD = PyBitGenerator<prng::XoshiroSIMD>;
  auto xs = nb::class_<PyXoshiroSIMD>(m, "_XoshiroSIMD")
                .def(nb::init<uint64_t>(), "seed"_a)
                .def(nb::init<uint64_t, uint64_t>(), "seed"_a, "thread"_a)
                .def(nb::init<uint64_t, uint64_t, uint64_t>(), "seed"_a,
                     "thread"_a, "cluster"_a);
  register_base<prng::XoshiroSIMD>(xs);
  register_jump<prng::XoshiroSIMD>(xs);

#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE
  using PyXoshiroNative = PyBitGenerator<prng::XoshiroNative>;
  auto xn = nb::class_<PyXoshiroNative>(m, "_XoshiroNative")
                .def(nb::init<uint64_t>(), "seed"_a)
                .def(nb::init<uint64_t, uint64_t>(), "seed"_a, "thread"_a)
                .def(nb::init<uint64_t, uint64_t, uint64_t>(), "seed"_a,
                     "thread"_a, "cluster"_a);
  register_base<prng::XoshiroNative>(xn);
  register_jump<prng::XoshiroNative>(xn);
#endif

  // ChaCha SIMD variants — accept both (seed) and (key, counter, nonce)
  using key_t = std::array<uint32_t, 8>;

  using PyChaCha8SIMD = PyBitGenerator<prng::ChaChaSIMD<8>>;
  auto c8s = nb::class_<PyChaCha8SIMD>(m, "_ChaCha8SIMD")
                 .def(nb::init<uint64_t>(), "seed"_a)
                 .def(nb::init<uint64_t, uint64_t, uint64_t>(), "seed"_a,
                      "counter"_a, "nonce"_a)
                 .def(nb::init<key_t, uint64_t, uint64_t>(), "key"_a,
                      "counter"_a, "nonce"_a);
  register_base<prng::ChaChaSIMD<8>>(c8s);

  using PyChaCha12SIMD = PyBitGenerator<prng::ChaChaSIMD<12>>;
  auto c12s = nb::class_<PyChaCha12SIMD>(m, "_ChaCha12SIMD")
                  .def(nb::init<uint64_t>(), "seed"_a)
                  .def(nb::init<uint64_t, uint64_t, uint64_t>(), "seed"_a,
                       "counter"_a, "nonce"_a)
                  .def(nb::init<key_t, uint64_t, uint64_t>(), "key"_a,
                       "counter"_a, "nonce"_a);
  register_base<prng::ChaChaSIMD<12>>(c12s);

  using PyChaCha20SIMD = PyBitGenerator<prng::ChaChaSIMD<20>>;
  auto c20s = nb::class_<PyChaCha20SIMD>(m, "_ChaCha20SIMD")
                  .def(nb::init<uint64_t>(), "seed"_a)
                  .def(nb::init<uint64_t, uint64_t, uint64_t>(), "seed"_a,
                       "counter"_a, "nonce"_a)
                  .def(nb::init<key_t, uint64_t, uint64_t>(), "key"_a,
                       "counter"_a, "nonce"_a);
  register_base<prng::ChaChaSIMD<20>>(c20s);

#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE
  // ChaCha Native variants — accept both (seed) and (key, counter, nonce)
  using PyChaCha8Native = PyBitGenerator<prng::ChaChaNative<8>>;
  auto c8n = nb::class_<PyChaCha8Native>(m, "_ChaCha8Native")
                 .def(nb::init<uint64_t>(), "seed"_a)
                 .def(nb::init<uint64_t, uint64_t, uint64_t>(), "seed"_a,
                      "counter"_a, "nonce"_a)
                 .def(nb::init<key_t, uint64_t, uint64_t>(), "key"_a,
                      "counter"_a, "nonce"_a);
  register_base<prng::ChaChaNative<8>>(c8n);

  using PyChaCha12Native = PyBitGenerator<prng::ChaChaNative<12>>;
  auto c12n = nb::class_<PyChaCha12Native>(m, "_ChaCha12Native")
                  .def(nb::init<uint64_t>(), "seed"_a)
                  .def(nb::init<uint64_t, uint64_t, uint64_t>(), "seed"_a,
                       "counter"_a, "nonce"_a)
                  .def(nb::init<key_t, uint64_t, uint64_t>(), "key"_a,
                       "counter"_a, "nonce"_a);
  register_base<prng::ChaChaNative<12>>(c12n);

  using PyChaCha20Native = PyBitGenerator<prng::ChaChaNative<20>>;
  auto c20n = nb::class_<PyChaCha20Native>(m, "_ChaCha20Native")
                  .def(nb::init<uint64_t>(), "seed"_a)
                  .def(nb::init<uint64_t, uint64_t, uint64_t>(), "seed"_a,
                       "counter"_a, "nonce"_a)
                  .def(nb::init<key_t, uint64_t, uint64_t>(), "key"_a,
                       "counter"_a, "nonce"_a);
  register_base<prng::ChaChaNative<20>>(c20n);
#endif

  // Philox SIMD variants — accept both (seed) and (key, counter)
  using PyPhilox4x32SIMD = PyBitGenerator<Philox4x32SIMD_t>;
  using key4x32_t = std::array<uint32_t, 2>;
  using ctr4x32_t = std::array<uint32_t, 4>;
  auto p4x32s = nb::class_<PyPhilox4x32SIMD>(m, "_Philox4x32SIMD")
                    .def(nb::init<uint64_t>(), "seed"_a)
                    .def(nb::init<uint64_t, uint64_t>(), "seed"_a, "counter"_a)
                    .def(nb::init<key4x32_t, ctr4x32_t>(), "key"_a, "counter"_a);
  register_base<Philox4x32SIMD_t>(p4x32s);

  using PyPhilox2x32SIMD = PyBitGenerator<Philox2x32SIMD_t>;
  using key2x32_t = std::array<uint32_t, 1>;
  using ctr2x32_t = std::array<uint32_t, 2>;
  auto p2x32s = nb::class_<PyPhilox2x32SIMD>(m, "_Philox2x32SIMD")
                    .def(nb::init<uint64_t>(), "seed"_a)
                    .def(nb::init<uint64_t, uint64_t>(), "seed"_a, "counter"_a)
                    .def(nb::init<key2x32_t, ctr2x32_t>(), "key"_a, "counter"_a);
  register_base<Philox2x32SIMD_t>(p2x32s);

  using PyPhilox4x64SIMD = PyBitGenerator<Philox4x64SIMD_t>;
  using key4x64_t = std::array<uint64_t, 2>;
  using ctr4x64_t = std::array<uint64_t, 4>;
  auto p4x64s = nb::class_<PyPhilox4x64SIMD>(m, "_Philox4x64SIMD")
                    .def(nb::init<uint64_t>(), "seed"_a)
                    .def(nb::init<uint64_t, uint64_t>(), "seed"_a, "counter"_a)
                    .def(nb::init<key4x64_t, ctr4x64_t>(), "key"_a, "counter"_a);
  register_base<Philox4x64SIMD_t>(p4x64s);

  using PyPhilox2x64SIMD = PyBitGenerator<Philox2x64SIMD_t>;
  using key2x64_t = std::array<uint64_t, 1>;
  using ctr2x64_t = std::array<uint64_t, 2>;
  auto p2x64s = nb::class_<PyPhilox2x64SIMD>(m, "_Philox2x64SIMD")
                    .def(nb::init<uint64_t>(), "seed"_a)
                    .def(nb::init<uint64_t, uint64_t>(), "seed"_a, "counter"_a)
                    .def(nb::init<key2x64_t, ctr2x64_t>(), "key"_a, "counter"_a);
  register_base<Philox2x64SIMD_t>(p2x64s);

#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE
  // Philox Native variants
  using PyPhilox4x32Native = PyBitGenerator<Philox4x32Native_t>;
  auto p4x32n = nb::class_<PyPhilox4x32Native>(m, "_Philox4x32Native")
                    .def(nb::init<uint64_t>(), "seed"_a)
                    .def(nb::init<uint64_t, uint64_t>(), "seed"_a, "counter"_a)
                    .def(nb::init<key4x32_t, ctr4x32_t>(), "key"_a, "counter"_a);
  register_base<Philox4x32Native_t>(p4x32n);

  using PyPhilox2x32Native = PyBitGenerator<Philox2x32Native_t>;
  auto p2x32n = nb::class_<PyPhilox2x32Native>(m, "_Philox2x32Native")
                    .def(nb::init<uint64_t>(), "seed"_a)
                    .def(nb::init<uint64_t, uint64_t>(), "seed"_a, "counter"_a)
                    .def(nb::init<key2x32_t, ctr2x32_t>(), "key"_a, "counter"_a);
  register_base<Philox2x32Native_t>(p2x32n);

  using PyPhilox4x64Native = PyBitGenerator<Philox4x64Native_t>;
  auto p4x64n = nb::class_<PyPhilox4x64Native>(m, "_Philox4x64Native")
                    .def(nb::init<uint64_t>(), "seed"_a)
                    .def(nb::init<uint64_t, uint64_t>(), "seed"_a, "counter"_a)
                    .def(nb::init<key4x64_t, ctr4x64_t>(), "key"_a, "counter"_a);
  register_base<Philox4x64Native_t>(p4x64n);

  using PyPhilox2x64Native = PyBitGenerator<Philox2x64Native_t>;
  auto p2x64n = nb::class_<PyPhilox2x64Native>(m, "_Philox2x64Native")
                    .def(nb::init<uint64_t>(), "seed"_a)
                    .def(nb::init<uint64_t, uint64_t>(), "seed"_a, "counter"_a)
                    .def(nb::init<key2x64_t, ctr2x64_t>(), "key"_a, "counter"_a);
  register_base<Philox2x64Native_t>(p2x64n);
#endif
}
