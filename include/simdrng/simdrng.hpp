#pragma once

// Umbrella header for the simdrng public API. This is also the entry point the
// single-header amalgamation (tools/amalgamate.py) inlines from.

#include "version.hpp"

#include "chacha.hpp"
#include "macros.hpp"
#include "philox.hpp"
#include "splitmix.hpp"
#include "xoshiro_scalar.hpp"

#include "chacha_simd.hpp"
#include "philox_simd.hpp"
#include "xoshiro.hpp" // Xoshiro alias (runtime-dispatched SIMD)
