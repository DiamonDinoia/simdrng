// Use XoshiroNative when compiling with -march=native / -mcpu for best performance.
#include <iostream>
#include <random/xoshiro_simd.hpp>

int main() {
  prng::XoshiroNative rng(42u);
  for (int i = 0; i < 4; ++i) std::cout << rng() << '\n';
  std::cout << "uniform: " << rng.uniform() << '\n';
  return 0;
}
