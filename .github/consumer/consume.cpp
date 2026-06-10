// Downstream consumer used by the Install & Consume workflow. Includes simdrng
// purely from its install tree and links the exported target. Works against both
// an xsimd build (simdrng::Xoshiro is the runtime-dispatch SIMD engine) and a
// scalar-only build (it aliases the scalar engine) -- the SIMDRNG_WITH_XSIMD
// macro is carried by the installed target, so this source needs no #define.
#include <cstdint>
#include <cstdio>

#include <simdrng/philox.hpp>
#include <simdrng/version.hpp>
#include <simdrng/xoshiro.hpp>

int main() {
  std::printf("simdrng %s\n", SIMDRNG_VERSION_STRING);

  simdrng::Xoshiro rng(0x1234567890abcdefULL);
  simdrng::Philox4x64 phi(7);
  std::uint64_t acc = 0;
  for (int i = 0; i < 8; ++i)
    acc ^= rng();
  for (int i = 0; i < 8; ++i)
    acc ^= phi();

  std::printf("acc = %016llx\n", static_cast<unsigned long long>(acc));
  return acc == 0 ? 1 : 0; // vanishingly unlikely to be zero; nonzero => OK
}
