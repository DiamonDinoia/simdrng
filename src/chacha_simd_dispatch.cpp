#include "random/chacha_simd.hpp"

namespace prng {

using ChaChaSIMDCreator_20 = internal::ChaChaSIMDCreator<20>;
#if defined(__x86_64__) || defined(_M_X64)

#if defined(__AVX512F__)
template std::unique_ptr<ChaChaSIMD<>::IChaChaSIMD> ChaChaSIMDCreator_20::operator()<xsimd::avx512f>(xsimd::avx512f) const;
#elif defined(__AVX__) && defined(__AVX2__) && defined(__FMA__)
template std::unique_ptr<ChaChaSIMD<>::IChaChaSIMD> ChaChaSIMDCreator_20::operator()<xsimd::fma3<xsimd::avx2>>(xsimd::fma3<xsimd::avx2>) const;
#elif defined(__SSE4_1__) && defined(__SSE4_2__) && defined(__SSSE3__) && defined(__SSE3__)
template std::unique_ptr<ChaChaSIMD<>::IChaChaSIMD> ChaChaSIMDCreator_20::operator()<xsimd::sse4_2>(xsimd::sse4_2) const;
// template std::unique_ptr<ChaChaSIMD<>::IChaChaSIMD> ChaChaSIMDCreator_20::operator()<xsimd::sse4_2>(xsimd::sse4_2) const;
#elif defined(__SSE__) && defined(__SSE2__) && defined(__MMX__)
template std::unique_ptr<ChaChaSIMD<>::IChaChaSIMD> ChaChaSIMDCreator_20::operator()<xsimd::sse2>(xsimd::sse2) const;
#else
#error "no SIMD instruction set enabled"
#endif

#else
#error "Unsupported x86-64 architecture"

#endif

}
