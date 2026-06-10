#include "simdrng/philox_simd.hpp"

namespace simdrng {

// Explicit template instantiation macros
#define SIMDRNG_PHILOX_INSTANTIATE(N, W, R, Arch)                                                                      \
  template internal::PhiloxSIMDInitResult<N, W> internal::PhiloxSIMDInitFunctor<N, W, R>::operator()<Arch>(Arch)       \
      const noexcept

#define SIMDRNG_PHILOX_INSTANTIATE_ALL(Arch)                                                                           \
  SIMDRNG_PHILOX_INSTANTIATE(4, 32, 10, Arch);                                                                         \
  SIMDRNG_PHILOX_INSTANTIATE(2, 32, 10, Arch);                                                                         \
  SIMDRNG_PHILOX_INSTANTIATE(4, 64, 10, Arch);                                                                         \
  SIMDRNG_PHILOX_INSTANTIATE(2, 64, 10, Arch)

#if SIMDRNG_ARCH_X86_64

#if defined(__AVX512BW__)
SIMDRNG_PHILOX_INSTANTIATE_ALL(xsimd::avx512bw);
#elif defined(__AVX2__)
SIMDRNG_PHILOX_INSTANTIATE_ALL(xsimd::avx2);
#elif defined(__SSE2__)
SIMDRNG_PHILOX_INSTANTIATE_ALL(xsimd::sse2);
#else
#error "x86_64: no SIMD instruction set enabled"
#endif

#elif SIMDRNG_ARCH_AARCH64

#if XSIMD_WITH_SVE
SIMDRNG_PHILOX_INSTANTIATE_ALL(xsimd::sve);
#elif XSIMD_WITH_NEON64
SIMDRNG_PHILOX_INSTANTIATE_ALL(xsimd::neon64);
#else
#error "aarch64: no SIMD instruction set detected"
#endif

#elif SIMDRNG_ARCH_RISCV64

#if XSIMD_WITH_RVV
SIMDRNG_PHILOX_INSTANTIATE_ALL(xsimd::rvv);
#else
#error "riscv64: RVV not available (compile with -march=rv64gcv -mrvv-vector-bits=zvl)"
#endif

#else
#error "Unsupported architecture"
#endif

#undef SIMDRNG_PHILOX_INSTANTIATE_ALL
#undef SIMDRNG_PHILOX_INSTANTIATE

} // namespace simdrng
