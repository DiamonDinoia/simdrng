// Fill a large std::vector<double> using XoshiroSIMD in a hot loop.
#include <chrono>
#include <iostream>
#include <vector>
#include <random/xoshiro_simd.hpp>

int main() {
  constexpr std::size_t N = 1 << 22;
  prng::XoshiroSIMD rng(1234u);

  std::vector<double> out(N);
  const auto t0 = std::chrono::steady_clock::now();
  for (auto &x : out) x = rng.uniform();
  const auto t1 = std::chrono::steady_clock::now();

  const auto ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "Filled " << N << " doubles in " << ms << " ms ("
            << (N / ms / 1e3) << " Msamples/s)\n";
  std::cout << "first=" << out.front() << " last=" << out.back() << '\n';
  return 0;
}
