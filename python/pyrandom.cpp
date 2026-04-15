#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <numpy/random/bitgen.h>
#include <type_traits>  // std::void_t, std::true_type, std::false_type
#include <utility>      // std::forward
#include <stdexcept>
#include <cstring>
#include <vector>

#include <poet/poet.hpp>

#include "random/macros.hpp"
#include "random/splitmix.hpp"
#include "random/xoshiro.hpp"
#include "random/xoshiro_simd.hpp"

namespace {
// Scaling constant: use 53-bit mantissa mapping to [0,1)
constexpr double kInvPow53 = 0x1.0p-53;
}

namespace nb = nanobind;
using namespace prng;

// -----------------------------------------------------------------------------
// Python-exposed wrappers (unchanged API)
class PySplitMix {
public:
  PRNG_ALWAYS_INLINE explicit PySplitMix(uint64_t seed) noexcept : gen(seed) {}
  PRNG_ALWAYS_INLINE uint64_t random_raw() noexcept { return gen(); }
  PRNG_ALWAYS_INLINE uint64_t get_state() const noexcept { return gen.getState(); }
  PRNG_ALWAYS_INLINE void set_state(uint64_t s) noexcept { gen.setState(s); }

private:
  SplitMix gen;
};

#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE
class PyXoshiroNative {
public:
  PRNG_ALWAYS_INLINE explicit PyXoshiroNative(uint64_t seed) noexcept : rng(seed) {}
  PRNG_ALWAYS_INLINE PyXoshiroNative(uint64_t seed, uint64_t thread) noexcept : rng(seed, thread) {}
  PRNG_ALWAYS_INLINE PyXoshiroNative(uint64_t seed, uint64_t thread, uint64_t cluster) noexcept
      : rng(seed, thread, cluster) {}

  PRNG_ALWAYS_INLINE uint64_t random_raw() noexcept { return rng(); }
  PRNG_ALWAYS_INLINE double uniform() noexcept { return rng.uniform(); }
  PRNG_ALWAYS_INLINE void jump() noexcept { rng.jump(); }
  PRNG_ALWAYS_INLINE void long_jump() noexcept { rng.long_jump(); }

private:
  XoshiroNative rng;
};
#endif // XSIMD_NO_SUPPORTED_ARCHITECTURE

class PyXoshiroSIMD : public XoshiroSIMD {
public:
  using XoshiroSIMD::XoshiroSIMD;

  PRNG_ALWAYS_INLINE uint64_t random_raw() noexcept { return (*this)(); }
  PRNG_ALWAYS_INLINE double uniform() noexcept { return XoshiroSIMD::uniform(); }
  PRNG_ALWAYS_INLINE void jump() noexcept { XoshiroSIMD::jump(); }
  PRNG_ALWAYS_INLINE void long_jump() noexcept { XoshiroSIMD::long_jump(); }

  // Fast bulk fill using internal cache; accepts 1D C-contiguous
  PRNG_ALWAYS_INLINE void fill_uniform_array(nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::c_contig> arr) noexcept {
    nb::gil_scoped_release release;
    auto* out = arr.data();
    const std::size_t n = arr.size();
    std::size_t produced = 0;
    while (produced < n) {
      if (m_index == 0) [[unlikely]] {
        m_populate_cache(m_state.data, m_cache);
      }
      const std::size_t available = static_cast<std::size_t>(CACHE_SIZE) - m_index;
      const std::size_t to_copy = (n - produced < available) ? (n - produced) : available;
      poet::dynamic_for<4>(std::size_t{0}, to_copy, [&](std::size_t i) {
        out[produced + i] = static_cast<double>(m_cache[m_index + i] >> 11) * kInvPow53;
      });
      m_index = static_cast<std::uint8_t>(m_index + to_copy);
      if (m_index == CACHE_SIZE) {
        m_index = 0;
      }
      produced += to_copy;
    }
  }

  // Overload: accept 1D F-contiguous
  PRNG_ALWAYS_INLINE void fill_uniform_array(nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::f_contig> arr) noexcept {
    nb::gil_scoped_release release;
    auto* out = arr.data();
    const std::size_t n = arr.size();
    std::size_t produced = 0;
    while (produced < n) {
      if (m_index == 0) [[unlikely]] {
        m_populate_cache(m_state.data, m_cache);
      }
      const std::size_t available = static_cast<std::size_t>(CACHE_SIZE) - m_index;
      const std::size_t to_copy = (n - produced < available) ? (n - produced) : available;
      poet::dynamic_for<4>(std::size_t{0}, to_copy, [&](std::size_t i) {
        out[produced + i] = static_cast<double>(m_cache[m_index + i] >> 11) * kInvPow53;
      });
      m_index = static_cast<std::uint8_t>(m_index + to_copy);
      if (m_index == CACHE_SIZE) {
        m_index = 0;
      }
      produced += to_copy;
    }
  }

  PRNG_ALWAYS_INLINE void fill_uint64_array(
      nb::ndarray<nb::numpy, std::uint64_t, nb::ndim<1>, nb::c_contig> arr) noexcept {
    nb::gil_scoped_release release;
    auto* out = arr.data();
    const std::size_t n = arr.size();
    std::size_t produced = 0;
    while (produced < n) {
      if (m_index == 0) [[unlikely]] {
        m_populate_cache(m_state.data, m_cache);
      }
      const std::size_t available = static_cast<std::size_t>(CACHE_SIZE) - m_index;
      const std::size_t to_copy = (n - produced < available) ? (n - produced) : available;
      // Copy contiguous block directly
      std::memcpy(out + produced, m_cache.data() + m_index, to_copy * sizeof(std::uint64_t));
      m_index = static_cast<std::uint8_t>(m_index + to_copy);
      if (m_index == CACHE_SIZE) {
        m_index = 0;
      }
      produced += to_copy;
    }
  }

  // Overload: accept 1D F-contiguous for uint64
  PRNG_ALWAYS_INLINE void fill_uint64_array(
      nb::ndarray<nb::numpy, std::uint64_t, nb::ndim<1>, nb::f_contig> arr) noexcept {
    nb::gil_scoped_release release;
    auto* out = arr.data();
    const std::size_t n = arr.size();
    std::size_t produced = 0;
    while (produced < n) {
      if (m_index == 0) [[unlikely]] {
        m_populate_cache(m_state.data, m_cache);
      }
      const std::size_t available = static_cast<std::size_t>(CACHE_SIZE) - m_index;
      const std::size_t to_copy = (n - produced < available) ? (n - produced) : available;
      std::memcpy(out + produced, m_cache.data() + m_index, to_copy * sizeof(std::uint64_t));
      m_index = static_cast<std::uint8_t>(m_index + to_copy);
      if (m_index == CACHE_SIZE) {
        m_index = 0;
      }
      produced += to_copy;
    }
  }
};

// -----------------------------------------------------------------------------
 

// -----------------------------------------------------------------------------
// DirectBitGen: optimized adapter
template <typename Rng>
struct alignas(64) DirectBitGen {
  Rng      rng;
  bitgen_t base;
  // Local cache of doubles to amortize callback overhead in NumPy's per-sample loop.
  static constexpr std::size_t DCACHE = 8192;
  std::vector<double> dcache;
  std::size_t dpos;

  template <typename... Args>
  PRNG_ALWAYS_INLINE explicit DirectBitGen(Args&&... args) noexcept
  : rng(std::forward<Args>(args)...), base{}, dcache(DCACHE), dpos{DCACHE} {
    base.state       = this;
    base.next_uint64 = &DirectBitGen::next_u64;
    base.next_uint32 = &DirectBitGen::next_u32;
    base.next_double = &DirectBitGen::next_f64;
    base.next_raw    = base.next_uint64;
  }

  PRNG_ALWAYS_INLINE void refill() noexcept {
    poet::dynamic_for<4>(std::size_t{0}, DCACHE, [&](std::size_t i) {
      dcache[i] = static_cast<double>(rng() >> 11) * kInvPow53;
    });
    dpos = 0;
  }

  // Static callbacks — no capturing lambdas, no thunks
  PRNG_ALWAYS_INLINE static uint64_t next_u64(void* s) noexcept {
    auto* self = static_cast<DirectBitGen*>(s);
    return self->rng(); // rely on inlining in Rng::operator()
  }

  PRNG_ALWAYS_INLINE static uint32_t next_u32(void* s) noexcept {
    auto* self = static_cast<DirectBitGen*>(s);
    // If your RNG has a native 32-bit fast path, you can specialize it here.
    return static_cast<uint32_t>(self->rng() >> 32);
  }

  PRNG_ALWAYS_INLINE static double next_f64(void* s) noexcept {
    auto* self = static_cast<DirectBitGen*>(s);
    if (self->dpos == DCACHE) [[unlikely]] {
      self->refill();
    }
    return self->dcache[self->dpos++];
  }
};

// -----------------------------------------------------------------------------
// Capsule management: keep NumPy-facing pointer = bitgen_t*, but own the
// DirectBitGen<T> via the capsule *context*, so we can delete safely.
template <typename Generator>
static void capsule_destruct(PyObject* capsule) noexcept {
  // Retrieve the owner pointer from the context (not the capsule pointer)
  void* ctx = PyCapsule_GetContext(capsule);
  auto* gen = static_cast<Generator*>(ctx);
  delete gen;
}

template <typename Generator>
static nb::object make_direct_bitgenerator_capsule(Generator* gen) {
  // NumPy expects a capsule with a "BitGenerator" pointer to bitgen_t
  bitgen_t* base_ptr = &gen->base;
  PyObject* cap = PyCapsule_New(static_cast<void*>(base_ptr), "BitGenerator",
                                &capsule_destruct<Generator>);
  // Store the actual owner so the destructor can delete it
  PyCapsule_SetContext(cap, static_cast<void*>(gen));
  return nb::steal(cap);
}

// -----------------------------------------------------------------------------
// Factory helpers
template <typename Rng, typename... Args>
PRNG_ALWAYS_INLINE nb::object make_direct_bitgenerator(Args&&... args) {
  using Generator = DirectBitGen<Rng>;
  auto* gen = new Generator(std::forward<Args>(args)...);
  return make_direct_bitgenerator_capsule(gen);
}

PRNG_ALWAYS_INLINE nb::object make_splitmix_bitgenerator(uint64_t seed) {
  return make_direct_bitgenerator<SplitMix>(seed);
}

PRNG_ALWAYS_INLINE nb::object make_xoshiro_bitgenerator(uint64_t seed) {
  return make_direct_bitgenerator<Xoshiro>(seed);
}

PRNG_ALWAYS_INLINE nb::object make_xoshiro_simd_bitgenerator(uint64_t seed) {
  return make_direct_bitgenerator<XoshiroSIMD>(seed);
}

#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE
PRNG_ALWAYS_INLINE nb::object make_xoshiro_native_bitgenerator(uint64_t seed) {
  return make_direct_bitgenerator<XoshiroNative>(seed);
}
PRNG_ALWAYS_INLINE nb::object make_xoshiro_native_bitgenerator(uint64_t seed, uint64_t thread) {
  return make_direct_bitgenerator<XoshiroNative>(seed, thread);
}
PRNG_ALWAYS_INLINE nb::object make_xoshiro_native_bitgenerator(uint64_t seed, uint64_t thread, uint64_t cluster) {
  return make_direct_bitgenerator<XoshiroNative>(seed, thread, cluster);
}
#endif // XSIMD_NO_SUPPORTED_ARCHITECTURE

PRNG_ALWAYS_INLINE void fill_xoshiro_simd_array(uint64_t seed, nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::c_contig> arr) {
  nb::gil_scoped_release release;
  auto* out = arr.data();
  const std::size_t n = arr.size();
  prng::XoshiroSIMD rng(seed);
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = static_cast<double>(rng() >> 11) * kInvPow53;
  }
}

PRNG_ALWAYS_INLINE void fill_xoshiro_simd_uint64(uint64_t seed, nb::ndarray<nb::numpy, std::uint64_t, nb::ndim<1>, nb::c_contig> arr) {
  nb::gil_scoped_release release;
  auto* out = arr.data();
  const std::size_t n = arr.size();
  prng::XoshiroSIMD rng(seed);
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = rng();
  }
}

// -----------------------------------------------------------------------------
// Python module
NB_MODULE(pyrandom_ext, m) {
  nb::class_<PySplitMix>(m, "SplitMix")
      .def(nb::init<uint64_t>())
      .def("random_raw", &PySplitMix::random_raw)
      .def("get_state", &PySplitMix::get_state)
      .def("set_state", &PySplitMix::set_state);

#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE
  nb::class_<PyXoshiroNative>(m, "XoshiroNative")
      .def(nb::init<uint64_t>())
      .def(nb::init<uint64_t, uint64_t>())
      .def(nb::init<uint64_t, uint64_t, uint64_t>())
      .def("random_raw", &PyXoshiroNative::random_raw)
      .def("uniform", &PyXoshiroNative::uniform)
      .def("jump", &PyXoshiroNative::jump)
      .def("long_jump", &PyXoshiroNative::long_jump);
#endif // XSIMD_NO_SUPPORTED_ARCHITECTURE

  nb::class_<PyXoshiroSIMD>(m, "XoshiroSIMD")
      .def(nb::init<uint64_t>())
      .def(nb::init<uint64_t, uint64_t>())
      .def(nb::init<uint64_t, uint64_t, uint64_t>())
      .def("random_raw", &PyXoshiroSIMD::random_raw)
      .def("uniform", &PyXoshiroSIMD::uniform)
      .def("jump", &PyXoshiroSIMD::jump)
      .def("long_jump", &PyXoshiroSIMD::long_jump)
      // Private helpers for internal fast paths in the high-level wrapper
      .def("_fill_uniform", nb::overload_cast<nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::c_contig>>(&PyXoshiroSIMD::fill_uniform_array), nb::arg("out"))
      .def("_fill_uniform", nb::overload_cast<nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::f_contig>>(&PyXoshiroSIMD::fill_uniform_array), nb::arg("out"))
      .def("_fill_uint64", nb::overload_cast<nb::ndarray<nb::numpy, std::uint64_t, nb::ndim<1>, nb::c_contig>>(&PyXoshiroSIMD::fill_uint64_array), nb::arg("out"))
      .def("_fill_uint64", nb::overload_cast<nb::ndarray<nb::numpy, std::uint64_t, nb::ndim<1>, nb::f_contig>>(&PyXoshiroSIMD::fill_uint64_array), nb::arg("out"));

  // NumPy BitGenerator factories
  m.def("create_bit_generator", &make_xoshiro_simd_bitgenerator, nb::arg("seed"),
        "Return a NumPy BitGenerator backed by XoshiroSIMD");

  m.def("create_splitmix_bit_generator", &make_splitmix_bitgenerator, nb::arg("seed"),
        "Return a NumPy BitGenerator backed by SplitMix");

  m.def("create_xoshiro_bit_generator", &make_xoshiro_bitgenerator, nb::arg("seed"),
        "Return a NumPy BitGenerator backed by Xoshiro");

#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE
  m.def("create_xoshiro_native_bit_generator",
        nb::overload_cast<uint64_t>(&make_xoshiro_native_bitgenerator),
        nb::arg("seed"),
        "Return a NumPy BitGenerator backed by XoshiroNative (seed)");

  m.def("create_xoshiro_native_bit_generator",
        nb::overload_cast<uint64_t, uint64_t>(&make_xoshiro_native_bitgenerator),
        nb::arg("seed"), nb::arg("thread"),
        "Return a NumPy BitGenerator backed by XoshiroNative (seed, thread)");

  m.def("create_xoshiro_native_bit_generator",
        nb::overload_cast<uint64_t, uint64_t, uint64_t>(&make_xoshiro_native_bitgenerator),
        nb::arg("seed"), nb::arg("thread"), nb::arg("cluster"),
        "Return a NumPy BitGenerator backed by XoshiroNative (seed, thread, cluster)");
#endif // XSIMD_NO_SUPPORTED_ARCHITECTURE

  m.def("fill_xoshiro_simd_array", &fill_xoshiro_simd_array, nb::arg("seed"), nb::arg("out"),
        "Fill a 1D numpy.ndarray[float64, C-contiguous] using XoshiroSIMD core bulk-fill (releases GIL)");

  m.def("fill_xoshiro_simd_uint64", &fill_xoshiro_simd_uint64, nb::arg("seed"), nb::arg("out"),
        "Fill a 1D numpy.ndarray[uint64, C-contiguous] with raw XoshiroSIMD outputs (releases GIL)");

}
