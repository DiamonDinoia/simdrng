// Per-thread independent streams using the (seed, thread_id, cluster_id) constructor.
#include <cstdio>
#include <random/xoshiro.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

int main() {
#ifdef _OPENMP
#pragma omp parallel
  {
    prng::Xoshiro rng(42u, omp_get_thread_num());
    const auto x = rng();
    std::printf("thread %d: %llu\n", omp_get_thread_num(),
                static_cast<unsigned long long>(x));
  }
#else
  prng::Xoshiro rng(42u, 0);
  std::printf("single thread: %llu\n",
              static_cast<unsigned long long>(rng()));
#endif
  return 0;
}
