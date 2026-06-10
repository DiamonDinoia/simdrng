// xsimd single-header smoke. The SIMD amalgamation (simdrng.hpp) inlines poet
// but keeps <xsimd/...> external, so compile with the xsimd headers on the
// include path (locally installed, or the Compiler Explorer xsimd library).
//
// Only the *Native generators are exercised: they are compile-time-arch and
// fully header-only. The runtime-dispatch types (XoshiroSIMD/Philox*SIMD and the
// default simdrng::Xoshiro alias) require the compiled per-arch dispatch TUs in
// libsimdrng.a and are intentionally not used here.
#include "simdrng.hpp"

int main() {
  simdrng::XoshiroNative    x(42);
  simdrng::Philox4x64Native p(7);
  unsigned long long acc = x();
  for (int i = 0; i < 4; ++i) acc ^= p();
  return acc == 0 ? 1 : 0;
}
