// Thread-safety smoke test, used as the ThreadSanitizer target.
//
// simdrng generators carry no shared mutable state: each thread constructs its
// own generator, and the (seed, thread_id) constructor places each thread on a
// jump-separated, non-overlapping subsequence. This test runs many such
// generators concurrently to prove there is no data race (under TSan) and that
// the per-thread streams are distinct and deterministic.

#include <array>
#include <cstdint>
#include <set>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <simdrng/xoshiro.hpp>

TEST_CASE("per-thread Xoshiro streams are race-free and independent", "[thread]") {
  constexpr int kThreads = 8;
  constexpr int kDraws = 4096;

  std::array<std::vector<std::uint64_t>, kThreads> out;
  std::vector<std::thread> pool;
  pool.reserve(kThreads);

  for (int t = 0; t < kThreads; ++t) {
    pool.emplace_back([&out, t] {
      // Each thread seeds its own generator on a distinct jumped subsequence
      // and writes only into its own (disjoint) output vector.
      simdrng::Xoshiro rng(42u, static_cast<std::uint64_t>(t));
      auto &dst = out[static_cast<std::size_t>(t)];
      dst.resize(kDraws);
      for (int i = 0; i < kDraws; ++i) {
        dst[static_cast<std::size_t>(i)] = rng();
      }
    });
  }
  for (auto &th : pool) {
    th.join();
  }

  // Distinct streams: the leading output of every thread is unique.
  std::set<std::uint64_t> firsts;
  for (int t = 0; t < kThreads; ++t) {
    REQUIRE(firsts.insert(out[static_cast<std::size_t>(t)][0]).second);
  }

  // Determinism: re-running thread 0's recipe reproduces its stream exactly.
  simdrng::Xoshiro replay(42u, 0u);
  for (int i = 0; i < kDraws; ++i) {
    REQUIRE(replay() == out[0][static_cast<std::size_t>(i)]);
  }
}
