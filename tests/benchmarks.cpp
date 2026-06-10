// Google Benchmark harness covering every generator in simdrng.
//
// Hardware counters (IPC, cache misses, branch misses) are available on
// Linux when Google Benchmark is built with BENCHMARK_ENABLE_LIBPFM=ON
// and libpfm4-dev is installed. Invoke:
//
//   ./benchmarks
//     --benchmark_perf_counters=CYCLES,INSTRUCTIONS,CACHE-MISSES,BRANCH-MISSES,BRANCHES
//     --benchmark_format=json --benchmark_out=bench.json
//
// IPC = INSTRUCTIONS / CYCLES is computed downstream in
// scripts/analyze_bench.py.

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>

#include <simdrng/chacha.hpp>
#include <simdrng/chacha_simd.hpp>
#include <simdrng/philox.hpp>
#include <simdrng/philox_simd.hpp>
#include <simdrng/xoshiro.hpp>
#include <simdrng/xoshiro_simd.hpp>

#include "xoshiro256plusplus.c"

namespace {

constexpr std::uint64_t kSeed = 42;

constexpr std::array<simdrng::ChaCha<20>::matrix_word, 8> kChaChaKey = {
    0x03020100u, 0x07060504u, 0x0b0a0908u, 0x0f0e0d0cu, 0x13121110u, 0x17161514u, 0x1b1a1918u, 0x1f1e1d1cu,
};
constexpr simdrng::ChaCha<20>::input_word kChaChaCounter = 0x0706050403020100ULL;
constexpr simdrng::ChaCha<20>::input_word kChaChaNonce = 0x0f0e0d0c0b0a0908ULL;

// ---------- scalar / dispatch uint64 generation ----------------------------

static void BM_ReferenceXoshiro_u64(benchmark::State &bench) {
  // The reference implementation keeps state in file-scope `s[4]`; seed it
  // from a scalar Xoshiro so every run is deterministic.
  simdrng::XoshiroScalar seeder(kSeed);
  const auto st = seeder.getState();
  s[0] = st[0];
  s[1] = st[1];
  s[2] = st[2];
  s[3] = st[3];
  for (auto _ : bench)
    benchmark::DoNotOptimize(next());
}
BENCHMARK(BM_ReferenceXoshiro_u64);

static void BM_XoshiroScalar_u64(benchmark::State &s) {
  simdrng::XoshiroScalar rng(kSeed);
  for (auto _ : s)
    benchmark::DoNotOptimize(rng());
}
BENCHMARK(BM_XoshiroScalar_u64);

static void BM_XoshiroSIMD_u64(benchmark::State &s) {
  simdrng::XoshiroSIMD rng(kSeed);
  for (auto _ : s)
    benchmark::DoNotOptimize(rng());
}
BENCHMARK(BM_XoshiroSIMD_u64);

// ---------- bulk fill: the SIMD-favorable workload -------------------------
// N chosen so the destination buffer stays L1-resident, isolating generation
// + store throughput from DRAM bandwidth.
namespace {
constexpr std::size_t kFillN = 4096; // 32 KiB of u64
// Output buffers are SIMD-aligned, matching how the bulk API is used in
// practice (numpy buffers are 64-byte aligned). A std::vector's 16-byte
// alignment splits ~half the 32-byte AVX2 stores across cache lines (~9%).
template <class T> using aligned_vec = std::vector<T, xsimd::aligned_allocator<T, 64>>;
} // namespace

static void BM_XoshiroScalar_fill_u64(benchmark::State &s) {
  simdrng::XoshiroScalar rng(kSeed);
  aligned_vec<std::uint64_t> buf(kFillN);
  for (auto _ : s) {
    for (std::size_t i = 0; i < kFillN; ++i)
      buf[i] = rng();
    benchmark::DoNotOptimize(buf.data());
    benchmark::ClobberMemory();
  }
  s.SetItemsProcessed(s.iterations() * kFillN);
}
BENCHMARK(BM_XoshiroScalar_fill_u64);

static void BM_XoshiroSIMD_fill_u64(benchmark::State &s) {
  simdrng::XoshiroSIMD rng(kSeed);
  aligned_vec<std::uint64_t> buf(kFillN);
  for (auto _ : s) {
    rng.generate(buf.data(), kFillN);
    benchmark::DoNotOptimize(buf.data());
    benchmark::ClobberMemory();
  }
  s.SetItemsProcessed(s.iterations() * kFillN);
}
BENCHMARK(BM_XoshiroSIMD_fill_u64);

static void BM_XoshiroScalar_fill_double(benchmark::State &s) {
  simdrng::XoshiroScalar rng(kSeed);
  aligned_vec<double> buf(kFillN);
  for (auto _ : s) {
    for (std::size_t i = 0; i < kFillN; ++i)
      buf[i] = rng.uniform();
    benchmark::DoNotOptimize(buf.data());
    benchmark::ClobberMemory();
  }
  s.SetItemsProcessed(s.iterations() * kFillN);
}
BENCHMARK(BM_XoshiroScalar_fill_double);

static void BM_XoshiroSIMD_fill_double(benchmark::State &s) {
  simdrng::XoshiroSIMD rng(kSeed);
  aligned_vec<double> buf(kFillN);
  for (auto _ : s) {
    rng.fill_uniform(buf.data(), kFillN);
    benchmark::DoNotOptimize(buf.data());
    benchmark::ClobberMemory();
  }
  s.SetItemsProcessed(s.iterations() * kFillN);
}
BENCHMARK(BM_XoshiroSIMD_fill_double);

static void BM_MT19937_64(benchmark::State &s) {
  std::mt19937_64 rng(kSeed);
  for (auto _ : s)
    benchmark::DoNotOptimize(rng());
}
BENCHMARK(BM_MT19937_64);

// ---------- ChaCha ---------------------------------------------------------

static void BM_ChaCha20Scalar_u64(benchmark::State &s) {
  simdrng::ChaCha<20> rng(kChaChaKey, kChaChaCounter, kChaChaNonce);
  for (auto _ : s)
    benchmark::DoNotOptimize(rng());
}
BENCHMARK(BM_ChaCha20Scalar_u64);

static void BM_ChaCha20SIMD_u64(benchmark::State &s) {
  simdrng::ChaChaSIMD<20> rng(kChaChaKey, kChaChaCounter, kChaChaNonce);
  for (auto _ : s)
    benchmark::DoNotOptimize(rng());
}
BENCHMARK(BM_ChaCha20SIMD_u64);

static void BM_ChaCha20Scalar_fill_u64(benchmark::State &s) {
  simdrng::ChaCha<20> rng(kChaChaKey, kChaChaCounter, kChaChaNonce);
  aligned_vec<std::uint64_t> buf(kFillN);
  for (auto _ : s) {
    for (std::size_t i = 0; i < kFillN; ++i)
      buf[i] = rng();
    benchmark::DoNotOptimize(buf.data());
    benchmark::ClobberMemory();
  }
  s.SetItemsProcessed(s.iterations() * kFillN);
}
BENCHMARK(BM_ChaCha20Scalar_fill_u64);

static void BM_ChaCha20SIMD_fill_u64(benchmark::State &s) {
  simdrng::ChaChaSIMD<20> rng(kChaChaKey, kChaChaCounter, kChaChaNonce);
  aligned_vec<std::uint64_t> buf(kFillN);
  for (auto _ : s) {
    rng.generate(buf.data(), kFillN);
    benchmark::DoNotOptimize(buf.data());
    benchmark::ClobberMemory();
  }
  s.SetItemsProcessed(s.iterations() * kFillN);
}
BENCHMARK(BM_ChaCha20SIMD_fill_u64);

static void BM_ChaCha20SIMD_fill_double(benchmark::State &s) {
  simdrng::ChaChaSIMD<20> rng(kChaChaKey, kChaChaCounter, kChaChaNonce);
  aligned_vec<double> buf(kFillN);
  for (auto _ : s) {
    rng.fill_uniform(buf.data(), kFillN);
    benchmark::DoNotOptimize(buf.data());
    benchmark::ClobberMemory();
  }
  s.SetItemsProcessed(s.iterations() * kFillN);
}
BENCHMARK(BM_ChaCha20SIMD_fill_double);

// ---------- Philox ---------------------------------------------------------

static void BM_Philox4x32Scalar_u64(benchmark::State &s) {
  simdrng::Philox<4, 32, 10> rng(kSeed);
  for (auto _ : s)
    benchmark::DoNotOptimize(rng());
}
BENCHMARK(BM_Philox4x32Scalar_u64);

static void BM_Philox4x32SIMD_u64(benchmark::State &s) {
  simdrng::PhiloxSIMD<4, 32, 10> rng(kSeed);
  for (auto _ : s)
    benchmark::DoNotOptimize(rng());
}
BENCHMARK(BM_Philox4x32SIMD_u64);

static void BM_Philox4x64Scalar_u64(benchmark::State &s) {
  simdrng::Philox<4, 64, 10> rng(kSeed);
  for (auto _ : s)
    benchmark::DoNotOptimize(rng());
}
BENCHMARK(BM_Philox4x64Scalar_u64);

static void BM_Philox4x64SIMD_u64(benchmark::State &s) {
  simdrng::PhiloxSIMD<4, 64, 10> rng(kSeed);
  for (auto _ : s)
    benchmark::DoNotOptimize(rng());
}
BENCHMARK(BM_Philox4x64SIMD_u64);

// ---------- Philox bulk fill -----------------------------------------------
static void BM_Philox4x32Scalar_fill_u64(benchmark::State &s) {
  simdrng::Philox<4, 32, 10> rng(kSeed);
  aligned_vec<std::uint64_t> buf(kFillN);
  for (auto _ : s) {
    for (std::size_t i = 0; i < kFillN; ++i)
      buf[i] = rng();
    benchmark::DoNotOptimize(buf.data());
    benchmark::ClobberMemory();
  }
  s.SetItemsProcessed(s.iterations() * kFillN);
}
BENCHMARK(BM_Philox4x32Scalar_fill_u64);

static void BM_Philox4x32SIMD_fill_u64(benchmark::State &s) {
  simdrng::PhiloxSIMD<4, 32, 10> rng(kSeed);
  aligned_vec<std::uint64_t> buf(kFillN);
  for (auto _ : s) {
    rng.generate(buf.data(), kFillN);
    benchmark::DoNotOptimize(buf.data());
    benchmark::ClobberMemory();
  }
  s.SetItemsProcessed(s.iterations() * kFillN);
}
BENCHMARK(BM_Philox4x32SIMD_fill_u64);

static void BM_Philox4x64Scalar_fill_u64(benchmark::State &s) {
  simdrng::Philox<4, 64, 10> rng(kSeed);
  aligned_vec<std::uint64_t> buf(kFillN);
  for (auto _ : s) {
    for (std::size_t i = 0; i < kFillN; ++i)
      buf[i] = rng();
    benchmark::DoNotOptimize(buf.data());
    benchmark::ClobberMemory();
  }
  s.SetItemsProcessed(s.iterations() * kFillN);
}
BENCHMARK(BM_Philox4x64Scalar_fill_u64);

static void BM_Philox4x64SIMD_fill_u64(benchmark::State &s) {
  simdrng::PhiloxSIMD<4, 64, 10> rng(kSeed);
  aligned_vec<std::uint64_t> buf(kFillN);
  for (auto _ : s) {
    rng.generate(buf.data(), kFillN);
    benchmark::DoNotOptimize(buf.data());
    benchmark::ClobberMemory();
  }
  s.SetItemsProcessed(s.iterations() * kFillN);
}
BENCHMARK(BM_Philox4x64SIMD_fill_u64);

static void BM_Philox4x32SIMD_fill_double(benchmark::State &s) {
  simdrng::PhiloxSIMD<4, 32, 10> rng(kSeed);
  aligned_vec<double> buf(kFillN);
  for (auto _ : s) {
    rng.fill_uniform(buf.data(), kFillN);
    benchmark::DoNotOptimize(buf.data());
    benchmark::ClobberMemory();
  }
  s.SetItemsProcessed(s.iterations() * kFillN);
}
BENCHMARK(BM_Philox4x32SIMD_fill_double);

// ---------- uniform<double> via generator.uniform() ------------------------

static void BM_XoshiroScalar_double(benchmark::State &s) {
  simdrng::XoshiroScalar rng(kSeed);
  for (auto _ : s)
    benchmark::DoNotOptimize(rng.uniform());
}
BENCHMARK(BM_XoshiroScalar_double);

static void BM_XoshiroSIMD_double(benchmark::State &s) {
  simdrng::XoshiroSIMD rng(kSeed);
  for (auto _ : s)
    benchmark::DoNotOptimize(rng.uniform());
}
BENCHMARK(BM_XoshiroSIMD_double);

// ---------- std::uniform_real_distribution<double> -------------------------

static void BM_XoshiroScalar_std_double(benchmark::State &s) {
  simdrng::XoshiroScalar rng(kSeed);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  for (auto _ : s)
    benchmark::DoNotOptimize(dist(rng));
}
BENCHMARK(BM_XoshiroScalar_std_double);

static void BM_XoshiroSIMD_std_double(benchmark::State &s) {
  simdrng::XoshiroSIMD rng(kSeed);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  for (auto _ : s)
    benchmark::DoNotOptimize(dist(rng));
}
BENCHMARK(BM_XoshiroSIMD_std_double);

static void BM_MT19937_64_std_double(benchmark::State &s) {
  std::mt19937_64 rng(kSeed);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  for (auto _ : s)
    benchmark::DoNotOptimize(dist(rng));
}
BENCHMARK(BM_MT19937_64_std_double);

// ---------- Native (-march=native) variants --------------------------------

#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE

static void BM_XoshiroNative_u64(benchmark::State &s) {
  simdrng::XoshiroNative rng(kSeed);
  for (auto _ : s)
    benchmark::DoNotOptimize(rng());
}
BENCHMARK(BM_XoshiroNative_u64);

static void BM_XoshiroNative_double(benchmark::State &s) {
  simdrng::XoshiroNative rng(kSeed);
  for (auto _ : s)
    benchmark::DoNotOptimize(rng.uniform());
}
BENCHMARK(BM_XoshiroNative_double);

static void BM_XoshiroNative_fill_u64(benchmark::State &s) {
  simdrng::XoshiroNative rng(kSeed);
  aligned_vec<std::uint64_t> buf(kFillN);
  for (auto _ : s) {
    rng.generate(buf.data(), kFillN);
    benchmark::DoNotOptimize(buf.data());
    benchmark::ClobberMemory();
  }
  s.SetItemsProcessed(s.iterations() * kFillN);
}
BENCHMARK(BM_XoshiroNative_fill_u64);

static void BM_XoshiroNative_fill_double(benchmark::State &s) {
  simdrng::XoshiroNative rng(kSeed);
  aligned_vec<double> buf(kFillN);
  for (auto _ : s) {
    rng.fill_uniform(buf.data(), kFillN);
    benchmark::DoNotOptimize(buf.data());
    benchmark::ClobberMemory();
  }
  s.SetItemsProcessed(s.iterations() * kFillN);
}
BENCHMARK(BM_XoshiroNative_fill_double);

static void BM_ChaCha20Native_u64(benchmark::State &s) {
  simdrng::ChaChaNative<20> rng(kChaChaKey, kChaChaCounter, kChaChaNonce);
  for (auto _ : s)
    benchmark::DoNotOptimize(rng());
}
BENCHMARK(BM_ChaCha20Native_u64);

static void BM_Philox4x32Native_u64(benchmark::State &s) {
  simdrng::PhiloxNative<4, 32, 10> rng(kSeed);
  for (auto _ : s)
    benchmark::DoNotOptimize(rng());
}
BENCHMARK(BM_Philox4x32Native_u64);

static void BM_Philox4x64Native_u64(benchmark::State &s) {
  simdrng::PhiloxNative<4, 64, 10> rng(kSeed);
  for (auto _ : s)
    benchmark::DoNotOptimize(rng());
}
BENCHMARK(BM_Philox4x64Native_u64);

static void BM_Philox4x32Native_fill_u64(benchmark::State &s) {
  simdrng::PhiloxNative<4, 32, 10> rng(kSeed);
  aligned_vec<std::uint64_t> buf(kFillN);
  for (auto _ : s) {
    rng.generate(buf.data(), kFillN);
    benchmark::DoNotOptimize(buf.data());
    benchmark::ClobberMemory();
  }
  s.SetItemsProcessed(s.iterations() * kFillN);
}
BENCHMARK(BM_Philox4x32Native_fill_u64);

static void BM_Philox4x64Native_fill_u64(benchmark::State &s) {
  simdrng::PhiloxNative<4, 64, 10> rng(kSeed);
  aligned_vec<std::uint64_t> buf(kFillN);
  for (auto _ : s) {
    rng.generate(buf.data(), kFillN);
    benchmark::DoNotOptimize(buf.data());
    benchmark::ClobberMemory();
  }
  s.SetItemsProcessed(s.iterations() * kFillN);
}
BENCHMARK(BM_Philox4x64Native_fill_u64);

static void BM_ChaCha20Native_fill_u64(benchmark::State &s) {
  simdrng::ChaChaNative<20> rng(kChaChaKey, kChaChaCounter, kChaChaNonce);
  aligned_vec<std::uint64_t> buf(kFillN);
  for (auto _ : s) {
    rng.generate(buf.data(), kFillN);
    benchmark::DoNotOptimize(buf.data());
    benchmark::ClobberMemory();
  }
  s.SetItemsProcessed(s.iterations() * kFillN);
}
BENCHMARK(BM_ChaCha20Native_fill_u64);

#endif // XSIMD_NO_SUPPORTED_ARCHITECTURE

} // namespace

BENCHMARK_MAIN();
