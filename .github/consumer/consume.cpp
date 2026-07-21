// Downstream consumer used by the Install & Consume workflow. Includes simdrng
// purely from its install tree and links the exported target;
// simdrng::Xoshiro is the runtime-dispatch SIMD engine.
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
