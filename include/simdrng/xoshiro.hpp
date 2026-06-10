#pragma once

// Umbrella header: `Xoshiro` is the runtime-dispatched SIMD generator when the
// library is built with xsimd, and the portable scalar generator otherwise.
#if SIMDRNG_WITH_XSIMD

#include "xoshiro_simd.hpp"

namespace simdrng {
using Xoshiro = XoshiroSIMD;
}

#else

#include "xoshiro_scalar.hpp"

namespace simdrng {
using Xoshiro = XoshiroScalar;
}

#endif // SIMDRNG_WITH_XSIMD
