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
#include <cstdint>
#include <random>

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
    0x03020100u, 0x07060504u, 0x0b0a0908u, 0x0f0e0d0cu,
    0x13121110u, 0x17161514u, 0x1b1a1918u, 0x1f1e1d1cu,
};
constexpr simdrng::ChaCha<20>::input_word kChaChaCounter = 0x0706050403020100ULL;
constexpr simdrng::ChaCha<20>::input_word kChaChaNonce   = 0x0f0e0d0c0b0a0908ULL;

// ---------- scalar / dispatch uint64 generation ----------------------------

static void BM_ReferenceXoshiro_u64(benchmark::State& bench) {
  // The reference implementation keeps state in file-scope `s[4]`; seed it
  // from a scalar Xoshiro so every run is deterministic.
  simdrng::XoshiroScalar seeder(kSeed);
  const auto st = seeder.getState();
  s[0] = st[0]; s[1] = st[1]; s[2] = st[2]; s[3] = st[3];
  for (auto _ : bench) benchmark::DoNotOptimize(next());
}
BENCHMARK(BM_ReferenceXoshiro_u64);

static void BM_XoshiroScalar_u64(benchmark::State& s) {
  simdrng::XoshiroScalar rng(kSeed);
  for (auto _ : s) benchmark::DoNotOptimize(rng());
}
BENCHMARK(BM_XoshiroScalar_u64);

static void BM_XoshiroSIMD_u64(benchmark::State& s) {
  simdrng::XoshiroSIMD rng(kSeed);
  for (auto _ : s) benchmark::DoNotOptimize(rng());
}
BENCHMARK(BM_XoshiroSIMD_u64);

static void BM_MT19937_64(benchmark::State& s) {
  std::mt19937_64 rng(kSeed);
  for (auto _ : s) benchmark::DoNotOptimize(rng());
}
BENCHMARK(BM_MT19937_64);

// ---------- ChaCha ---------------------------------------------------------

static void BM_ChaCha20Scalar_u64(benchmark::State& s) {
  simdrng::ChaCha<20> rng(kChaChaKey, kChaChaCounter, kChaChaNonce);
  for (auto _ : s) benchmark::DoNotOptimize(rng());
}
BENCHMARK(BM_ChaCha20Scalar_u64);

static void BM_ChaCha20SIMD_u64(benchmark::State& s) {
  simdrng::ChaChaSIMD<20> rng(kChaChaKey, kChaChaCounter, kChaChaNonce);
  for (auto _ : s) benchmark::DoNotOptimize(rng());
}
BENCHMARK(BM_ChaCha20SIMD_u64);

// ---------- Philox ---------------------------------------------------------

static void BM_Philox4x32Scalar_u64(benchmark::State& s) {
  simdrng::Philox<4, 32, 10> rng(kSeed);
  for (auto _ : s) benchmark::DoNotOptimize(rng());
}
BENCHMARK(BM_Philox4x32Scalar_u64);

static void BM_Philox4x32SIMD_u64(benchmark::State& s) {
  simdrng::PhiloxSIMD<4, 32, 10> rng(kSeed);
  for (auto _ : s) benchmark::DoNotOptimize(rng());
}
BENCHMARK(BM_Philox4x32SIMD_u64);

static void BM_Philox4x64Scalar_u64(benchmark::State& s) {
  simdrng::Philox<4, 64, 10> rng(kSeed);
  for (auto _ : s) benchmark::DoNotOptimize(rng());
}
BENCHMARK(BM_Philox4x64Scalar_u64);

static void BM_Philox4x64SIMD_u64(benchmark::State& s) {
  simdrng::PhiloxSIMD<4, 64, 10> rng(kSeed);
  for (auto _ : s) benchmark::DoNotOptimize(rng());
}
BENCHMARK(BM_Philox4x64SIMD_u64);

// ---------- uniform<double> via generator.uniform() ------------------------

static void BM_XoshiroScalar_double(benchmark::State& s) {
  simdrng::XoshiroScalar rng(kSeed);
  for (auto _ : s) benchmark::DoNotOptimize(rng.uniform());
}
BENCHMARK(BM_XoshiroScalar_double);

static void BM_XoshiroSIMD_double(benchmark::State& s) {
  simdrng::XoshiroSIMD rng(kSeed);
  for (auto _ : s) benchmark::DoNotOptimize(rng.uniform());
}
BENCHMARK(BM_XoshiroSIMD_double);

// ---------- std::uniform_real_distribution<double> -------------------------

static void BM_XoshiroScalar_std_double(benchmark::State& s) {
  simdrng::XoshiroScalar rng(kSeed);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  for (auto _ : s) benchmark::DoNotOptimize(dist(rng));
}
BENCHMARK(BM_XoshiroScalar_std_double);

static void BM_XoshiroSIMD_std_double(benchmark::State& s) {
  simdrng::XoshiroSIMD rng(kSeed);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  for (auto _ : s) benchmark::DoNotOptimize(dist(rng));
}
BENCHMARK(BM_XoshiroSIMD_std_double);

static void BM_MT19937_64_std_double(benchmark::State& s) {
  std::mt19937_64 rng(kSeed);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  for (auto _ : s) benchmark::DoNotOptimize(dist(rng));
}
BENCHMARK(BM_MT19937_64_std_double);

// ---------- Native (-march=native) variants --------------------------------

#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE

static void BM_XoshiroNative_u64(benchmark::State& s) {
  simdrng::XoshiroNative rng(kSeed);
  for (auto _ : s) benchmark::DoNotOptimize(rng());
}
BENCHMARK(BM_XoshiroNative_u64);

static void BM_XoshiroNative_double(benchmark::State& s) {
  simdrng::XoshiroNative rng(kSeed);
  for (auto _ : s) benchmark::DoNotOptimize(rng.uniform());
}
BENCHMARK(BM_XoshiroNative_double);

static void BM_ChaCha20Native_u64(benchmark::State& s) {
  simdrng::ChaChaNative<20> rng(kChaChaKey, kChaChaCounter, kChaChaNonce);
  for (auto _ : s) benchmark::DoNotOptimize(rng());
}
BENCHMARK(BM_ChaCha20Native_u64);

static void BM_Philox4x32Native_u64(benchmark::State& s) {
  simdrng::PhiloxNative<4, 32, 10> rng(kSeed);
  for (auto _ : s) benchmark::DoNotOptimize(rng());
}
BENCHMARK(BM_Philox4x32Native_u64);

static void BM_Philox4x64Native_u64(benchmark::State& s) {
  simdrng::PhiloxNative<4, 64, 10> rng(kSeed);
  for (auto _ : s) benchmark::DoNotOptimize(rng());
}
BENCHMARK(BM_Philox4x64Native_u64);

#endif  // XSIMD_NO_SUPPORTED_ARCHITECTURE

}  // namespace

BENCHMARK_MAIN();
