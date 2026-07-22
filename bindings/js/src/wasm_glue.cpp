// wasm_glue.cpp — root TU for the Emscripten/WASM backend (--no-entry library).
// The public surface is the simdrng C ABI, exported by name via
// -sEXPORTED_FUNCTIONS in CMakeLists.txt. The one addition is sr_create_split:
// ccall marshals i64 awkwardly, so the JS side passes the 64-bit seed as two
// 32-bit halves and this rejoins them.

#include <simdrng/capi.h>

#include <cstdint>

extern "C" simdrng_t sr_create_split(int kind, uint32_t seed_hi, uint32_t seed_lo) {
  return simdrng_create(static_cast<simdrng_kind>(kind), (static_cast<uint64_t>(seed_hi) << 32) | seed_lo);
}
