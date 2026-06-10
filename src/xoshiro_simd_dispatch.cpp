#include <simdrng/xoshiro_simd.hpp>

namespace simdrng {

using InitFunctor = internal::XoshiroSIMDInitFunctor;
using InitResult = internal::XoshiroSIMDInitResult;

#if SIMDRNG_ARCH_X86_64
#if defined(__AVX512BW__)
template InitResult InitFunctor::operator()<xsimd::avx512bw>(xsimd::avx512bw) const noexcept;
#elif defined(__AVX2__)
template InitResult InitFunctor::operator()<xsimd::avx2>(xsimd::avx2) const noexcept;
#elif defined(__SSE2__)
template InitResult InitFunctor::operator()<xsimd::sse2>(xsimd::sse2) const noexcept;
#else
#error "x86_64: no SIMD instruction set enabled"
#endif

#elif SIMDRNG_ARCH_AARCH64
#if XSIMD_WITH_SVE
template InitResult InitFunctor::operator()<xsimd::sve>(xsimd::sve) const noexcept;
#elif XSIMD_WITH_NEON64
template InitResult InitFunctor::operator()<xsimd::neon64>(xsimd::neon64) const noexcept;
#else
#error "aarch64: no SIMD instruction set detected"
#endif

#elif SIMDRNG_ARCH_RISCV64
#if XSIMD_WITH_RVV
template InitResult InitFunctor::operator()<xsimd::rvv>(xsimd::rvv) const noexcept;
#else
#error "riscv64: RVV not available (compile with -march=rv64gcv -mrvv-vector-bits=zvl)"
#endif

#else
#error "Unsupported architecture"
#endif

} // namespace simdrng
