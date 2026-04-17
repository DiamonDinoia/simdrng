#include "random/philox_simd.hpp"

namespace prng {

// Explicit template instantiation macros
#define PRNG_PHILOX_INSTANTIATE(N, W, R, Arch)                                                      \
  template internal::PhiloxSIMDInitResult<N, W>                                                     \
  internal::PhiloxSIMDInitFunctor<N, W, R>::operator()<Arch>(Arch) const noexcept

#define PRNG_PHILOX_INSTANTIATE_ALL(Arch)                                                           \
  PRNG_PHILOX_INSTANTIATE(4, 32, 10, Arch);                                                        \
  PRNG_PHILOX_INSTANTIATE(2, 32, 10, Arch);                                                        \
  PRNG_PHILOX_INSTANTIATE(4, 64, 10, Arch);                                                        \
  PRNG_PHILOX_INSTANTIATE(2, 64, 10, Arch)

#if PRNG_ARCH_X86_64

#if defined(__AVX512F__)
PRNG_PHILOX_INSTANTIATE_ALL(xsimd::avx512f);
#elif defined(__AVX2__)
PRNG_PHILOX_INSTANTIATE_ALL(xsimd::avx2);
#elif defined(__SSE2__)
PRNG_PHILOX_INSTANTIATE_ALL(xsimd::sse2);
#else
#error "x86_64: no SIMD instruction set enabled"
#endif

#elif PRNG_ARCH_AARCH64

#if XSIMD_WITH_SVE
PRNG_PHILOX_INSTANTIATE_ALL(xsimd::sve);
#elif XSIMD_WITH_NEON64
PRNG_PHILOX_INSTANTIATE_ALL(xsimd::neon64);
#else
#error "aarch64: no SIMD instruction set detected"
#endif

#elif PRNG_ARCH_RISCV64

#if XSIMD_WITH_RVV
PRNG_PHILOX_INSTANTIATE_ALL(xsimd::rvv);
#else
#error "riscv64: RVV not available (compile with -march=rv64gcv -mrvv-vector-bits=zvl)"
#endif

#else
#error "Unsupported architecture"
#endif

#undef PRNG_PHILOX_INSTANTIATE_ALL
#undef PRNG_PHILOX_INSTANTIATE

} // namespace prng
