// Philox is a counter-based PRNG: same key + same counter => same output.
#include <iostream>
#include <random/philox_simd.hpp>

int main() {
  using Rng = prng::PhiloxSIMD<4, 64, 10>;
  Rng a(42u);
  Rng b(42u);

  for (int i = 0; i < 4; ++i) {
    const auto x = a();
    const auto y = b();
    std::cout << "a=" << x << " b=" << y
              << (x == y ? " (match)\n" : " (MISMATCH)\n");
  }
  return 0;
}
