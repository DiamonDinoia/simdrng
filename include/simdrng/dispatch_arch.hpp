#pragma once

#if SIMDRNG_WITH_XSIMD

#include <type_traits>
#include <xsimd/xsimd.hpp>

namespace simdrng {

// Dispatch arch list, selected at compile time via the xsimd inheritance hierarchy.
// x86:     best_arch derives from sse2  -> dispatch across avx512bw / avx2 / sse2
// aarch64: best_arch derives from neon  -> dispatch to neon64
// riscv64: fallback                     -> dispatch to rvv (128-bit fixed VLEN)
//
// Note: xsimd::rvv is detail::rvv<__riscv_v_fixed_vlen> when RVV is active,
// but detail::rvv<0xFFFFFFFF> as a fallback sentinel otherwise.  The dispatch
// TU is compiled with -march=rv64gcv -mrvv-vector-bits=zvl (VLEN=128), so we
// must use the concrete rvv<128> to match the exported symbols.
using rvv128 = xsimd::detail::rvv<128>;

// xsimd's arch hierarchy splits the x86 family: avx* derive from `common`,
// and avx512* derive from `common` independently (avx512bw : avx512dq :
// avx512cd : avx512f : common). So avx512bw is NOT a descendant of sse2 or
// avx. Check all three roots to cover every x86 ISA variant.
//
// The AVX-512 variant targets avx512bw (the x86-64-v4 floor: F+CD+BW+DQ+VL).
// Compared to plain avx512f this unlocks vpmullq (AVX512DQ, used by Philox
// mulhilo) and byte/word permutes (AVX512BW, used by ChaCha shuffles).
// avx512bw derives from avx512f, so the is_base_of_v<avx512f, ...> root check
// below is still correct.
using dispatch_arch_list = std::conditional_t<
    std::is_base_of_v<xsimd::sse2, xsimd::best_arch> ||
    std::is_base_of_v<xsimd::avx, xsimd::best_arch> ||
    std::is_base_of_v<xsimd::avx512f, xsimd::best_arch>,
    xsimd::arch_list<xsimd::avx512bw, xsimd::avx2, xsimd::sse2>,
    std::conditional_t<
        std::is_base_of_v<xsimd::neon, xsimd::best_arch>,
        xsimd::arch_list<xsimd::neon64>,
        xsimd::arch_list<rvv128>
    >
>;

} // namespace simdrng

#endif // SIMDRNG_WITH_XSIMD
