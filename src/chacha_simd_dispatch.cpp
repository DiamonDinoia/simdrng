#include "random/chacha_simd.hpp"

namespace prng {

using InitFunctor20 = internal::ChaChaSIMDInitFunctor<20>;
using InitResult = internal::ChaChaSIMDInitResult;

#if PRNG_ARCH_X86_64

#if defined(__AVX512F__)
template InitResult InitFunctor20::operator()<xsimd::avx512f>(xsimd::avx512f) const noexcept;
#elif defined(__AVX2__)
template InitResult InitFunctor20::operator()<xsimd::avx2>(xsimd::avx2) const noexcept;
#elif defined(__SSE2__)
template InitResult InitFunctor20::operator()<xsimd::sse2>(xsimd::sse2) const noexcept;
#else
#error "x86_64: no SIMD instruction set enabled"
#endif

#elif PRNG_ARCH_AARCH64

#if XSIMD_WITH_SVE
template InitResult InitFunctor20::operator()<xsimd::sve>(xsimd::sve) const noexcept;
#elif XSIMD_WITH_NEON64
template InitResult InitFunctor20::operator()<xsimd::neon64>(xsimd::neon64) const noexcept;
#else
#error "aarch64: no SIMD instruction set detected"
#endif

#elif PRNG_ARCH_RISCV64

#if XSIMD_WITH_RVV
template InitResult InitFunctor20::operator()<xsimd::rvv>(xsimd::rvv) const noexcept;
#else
#error "riscv64: RVV not available (compile with -march=rv64gcv -mrvv-vector-bits=zvl)"
#endif

#else
#error "Unsupported architecture"
#endif

} // namespace prng
