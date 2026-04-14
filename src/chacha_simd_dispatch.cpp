#include "random/chacha_simd.hpp"

namespace prng {

using InitFunctor20 = internal::ChaChaSIMDInitFunctor<20>;
using InitResult = internal::ChaChaSIMDInitResult;

#if defined(__x86_64__) || defined(_M_X64)

#if defined(__AVX512F__)
template InitResult InitFunctor20::operator()<xsimd::avx512f>(xsimd::avx512f) const noexcept;
#elif defined(__AVX2__)
template InitResult InitFunctor20::operator()<xsimd::avx2>(xsimd::avx2) const noexcept;
#elif defined(__SSE2__)
template InitResult InitFunctor20::operator()<xsimd::sse2>(xsimd::sse2) const noexcept;
#else
#error "no SIMD instruction set enabled"
#endif

#else
#error "Unsupported architecture"
#endif

} // namespace prng
