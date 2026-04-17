// Minimal scalar Xoshiro256++ usage: raw uint64, uniform(), and C++ <random> interop.
#include <iostream>
#include <random>
#include <random/xoshiro.hpp>

int main() {
  const auto seed = 42u;
  prng::Xoshiro rng(seed);

  std::cout << "uint64: " << rng() << '\n';
  std::cout << "uniform double: " << rng.uniform() << '\n';

  std::uniform_real_distribution<double> dist(0.0, 1.0);
  std::cout << "std::uniform_real_distribution: " << dist(rng) << '\n';
  return 0;
}
