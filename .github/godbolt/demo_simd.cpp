// Compiler Explorer demo (SIMD). Locally this includes the SIMD single header;
// the godbolt-links workflow strips the include and prepends the amalgamated
// simdrng.hpp, then compiles with Compiler Explorer's vendored xsimd library.
//
// Uses the *Native generators (compile-time arch, header-only). The
// runtime-dispatch types need libsimdrng.a and cannot run on a single-header
// godbolt link.
#include "simdrng.hpp"
#include <cstdio>

int main() {
  simdrng::XoshiroNative    x(42);
  simdrng::Philox4x64Native p(7);
  unsigned long long acc = x();
  for (int i = 0; i < 4; ++i) acc ^= p();
  std::printf("xoshiro_native=%llu philox_native_mix=%016llx\n",
              (unsigned long long)x(), acc);
}
