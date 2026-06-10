// Compiler Explorer demo (scalar). Locally this includes the scalar single
// header; the godbolt-links workflow strips the include and prepends the
// amalgamated simdrng-scalar.hpp so the link is self-contained (no libraries).
#include "simdrng-scalar.hpp"
#include <cstdio>

int main() {
  simdrng::Xoshiro x(42); // scalar engine in the scalar single header
  simdrng::SplitMix s(7);
  std::printf("xoshiro=%llu splitmix=%llu\n", (unsigned long long)x(), (unsigned long long)s());
}
