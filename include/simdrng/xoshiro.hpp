#pragma once

// Umbrella header: `Xoshiro` is the runtime-dispatched SIMD generator when the
// library is built with SIMDRNG_DISPATCH, otherwise the native (best_arch) one.
#include "macros.hpp"
#include "xoshiro_simd.hpp"

namespace simdrng {
#if SIMDRNG_DISPATCH
using Xoshiro = XoshiroSIMD;
#else
using Xoshiro = XoshiroNative;
#endif
}
