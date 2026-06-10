#pragma once

// Umbrella header for the simdrng public API. This is also the entry point the
// single-header amalgamation (tools/amalgamate.py) inlines from.
//
// The SIMD generators are gated on SIMDRNG_WITH_XSIMD (defined by the build, or
// definable by a consumer before inclusion). When it is 0/undefined, only the
// scalar generators are available and `Xoshiro` aliases the scalar engine.

#include "version.hpp"

#include "chacha.hpp"
#include "macros.hpp"
#include "philox.hpp"
#include "splitmix.hpp"
#include "xoshiro_scalar.hpp"

#include "chacha_simd.hpp" // self-guarded by SIMDRNG_WITH_XSIMD
#include "philox_simd.hpp" // self-guarded by SIMDRNG_WITH_XSIMD
#include "xoshiro.hpp"     // Xoshiro alias (SIMD when available, else scalar)
