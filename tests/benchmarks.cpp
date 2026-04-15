#include <iostream>
#include <nanobench.h>
#include <random>
#include <random/chacha.hpp>
#include <random/chacha_simd.hpp>
#include <random/xoshiro_simd.hpp>

#include "xoshiro256plusplus.c"

static constexpr auto iterations = 1;

namespace {

ankerl::nanobench::Bench make_bench(const char *title, const char *unit,
                                    double batch) {
  using namespace std::chrono_literals;

  return ankerl::nanobench::Bench()
    .title(title)
    .unit(unit)
    .batch(batch)
    .minEpochTime(50ms)
    .minEpochIterations(2000)
    .relative(true);
}

} // namespace

int main() {
  volatile const auto seed = 42;
  std::cout << "SEED: " << seed << std::endl;
  prng::XoshiroScalar reference(seed);
  prng::XoshiroSIMD dispatch(seed);
  using ScalarChaCha20 = prng::ChaCha<20>;
  using SimdChaCha20 = prng::ChaChaSIMD<20>;
  constexpr std::array<ScalarChaCha20::matrix_word, 8> chacha_key = {
    0x03020100u, 0x07060504u, 0x0b0a0908u, 0x0f0e0d0cu,
    0x13121110u, 0x17161514u, 0x1b1a1918u, 0x1f1e1d1cu,
  };
  constexpr ScalarChaCha20::input_word chacha_counter = 0x0706050403020100ULL;
  constexpr ScalarChaCha20::input_word chacha_nonce = 0x0f0e0d0c0b0a0908ULL;
  std::cout << "ChaCha SIMD width: " << SimdChaCha20({}, 0, 0).getSIMDSize() << std::endl;
  ScalarChaCha20 chacha_scalar_uint64(chacha_key, chacha_counter, chacha_nonce);
  SimdChaCha20 chacha_simd_uint64(chacha_key, chacha_counter, chacha_nonce);
  ScalarChaCha20 chacha_scalar_double(chacha_key, chacha_counter, chacha_nonce);
  SimdChaCha20 chacha_simd_double(chacha_key, chacha_counter, chacha_nonce);
  ScalarChaCha20 chacha_scalar_dist(chacha_key, chacha_counter, chacha_nonce);
  SimdChaCha20 chacha_simd_dist(chacha_key, chacha_counter, chacha_nonce);

  s[0] = reference.getState()[0];
  s[1] = reference.getState()[1];
  s[2] = reference.getState()[2];
  s[3] = reference.getState()[3];

  std::uniform_real_distribution<double> double_dist(0.0, 1.0);
  std::mt19937_64 mt(seed);
  using ankerl::nanobench::doNotOptimizeAway;

  // --- Benchmarks that are always available (scalar, dispatch, mt) ---
  make_bench("UINT64 generation", "sample", static_cast<double>(iterations))
    .run("Reference Xoshiro UINT64", [&] {
      for (int i = 0; i < iterations; ++i) {
        doNotOptimizeAway(next());
      }
    })
    .run("Scalar Xoshiro UINT64", [&] {
      for (int i = 0; i < iterations; ++i) {
        doNotOptimizeAway(reference());
      }
    })
    .run("Dispatch Xoshiro UINT64", [&] {
      for (int i = 0; i < iterations; ++i) {
        doNotOptimizeAway(dispatch());
      }
    })
    .run("MersenneTwister UINT64", [&] {
      for (int i = 0; i < iterations; ++i) {
        doNotOptimizeAway(mt());
      }
    })
    .run("ChaCha20 scalar UINT64", [&] {
      for (int i = 0; i < iterations; ++i) {
        doNotOptimizeAway(chacha_scalar_uint64());
      }
    })
    .run("ChaCha20 dispatch UINT64", [&] {
      for (int i = 0; i < iterations; ++i) {
        doNotOptimizeAway(chacha_simd_uint64());
      }
    });

  make_bench("Unit-interval doubles", "sample", static_cast<double>(iterations))
    .run("Scalar Xoshiro DOUBLE", [&] {
      for (int i = 0; i < iterations; ++i) {
        doNotOptimizeAway(reference.uniform());
      }
    })
    .run("Dispatch Xoshiro DOUBLE", [&] {
      for (int i = 0; i < iterations; ++i) {
        doNotOptimizeAway(dispatch.uniform());
      }
    })
    .run("ChaCha20 scalar DOUBLE", [&] {
      for (int i = 0; i < iterations; ++i) {
        doNotOptimizeAway(chacha_scalar_double.uniform());
      }
    })
    .run("ChaCha20 Dispatch DOUBLE", [&] {
      for (int i = 0; i < iterations; ++i) {
        doNotOptimizeAway(chacha_simd_double.uniform());
      }
    });

  make_bench("std::uniform_real_distribution<double>", "sample",
             static_cast<double>(iterations))
    .run("Scalar Xoshiro std::random<double>", [&] {
      for (int i = 0; i < iterations; ++i) {
        doNotOptimizeAway(double_dist(reference));
      }
    })
    .run("Dispatch Xoshiro std::random<double>", [&] {
      for (int i = 0; i < iterations; ++i) {
        doNotOptimizeAway(double_dist(dispatch));
      }
    })
    .run("MersenneTwister std::random<double>", [&] {
      for (int i = 0; i < iterations; ++i) {
        doNotOptimizeAway(double_dist(mt));
      }
    })
    .run("ChaCha20 scalar std::random<double>", [&] {
      for (int i = 0; i < iterations; ++i) {
        doNotOptimizeAway(double_dist(chacha_scalar_dist));
      }
    })
    .run("ChaCha20 Dispatch std::random<double>", [&] {
      for (int i = 0; i < iterations; ++i) {
        doNotOptimizeAway(double_dist(chacha_simd_dist));
      }
    });

#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE
  // --- Native (compile-time best arch) benchmarks ---
  prng::XoshiroNative rng(seed);
  using NativeChaCha20 = prng::ChaChaNative<20>;
  NativeChaCha20 chacha_native_uint64(chacha_key, chacha_counter, chacha_nonce);
  NativeChaCha20 chacha_native_double(chacha_key, chacha_counter, chacha_nonce);
  NativeChaCha20 chacha_native_dist(chacha_key, chacha_counter, chacha_nonce);

  make_bench("Native UINT64 generation", "sample", static_cast<double>(iterations))
    .run("XoshiroNative UINT64", [&] {
      for (int i = 0; i < iterations; ++i) {
        doNotOptimizeAway(rng());
      }
    })
    .run("ChaCha20 native UINT64", [&] {
      for (int i = 0; i < iterations; ++i) {
        doNotOptimizeAway(chacha_native_uint64());
      }
    });

  make_bench("Native doubles", "sample", static_cast<double>(iterations))
    .run("XoshiroNative DOUBLE", [&] {
      for (int i = 0; i < iterations; ++i) {
        doNotOptimizeAway(rng.uniform());
      }
    })
    .run("ChaCha20 native DOUBLE", [&] {
      for (int i = 0; i < iterations; ++i) {
        doNotOptimizeAway(chacha_native_double.uniform());
      }
    });

  make_bench("Native std::uniform_real_distribution<double>", "sample",
             static_cast<double>(iterations))
    .run("XoshiroNative std::random<double>", [&] {
      for (int i = 0; i < iterations; ++i) {
        doNotOptimizeAway(double_dist(rng));
      }
    })
    .run("ChaCha20 native std::random<double>", [&] {
      for (int i = 0; i < iterations; ++i) {
        doNotOptimizeAway(double_dist(chacha_native_dist));
      }
    });
#endif // XSIMD_NO_SUPPORTED_ARCHITECTURE

}
