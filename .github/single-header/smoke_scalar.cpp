// Scalar single-header smoke. Compiled with NO -I: the scalar amalgamation
// (simdrng-scalar.hpp) must be fully self-contained -- no xsimd, no poet, no
// project include path. Catches un-inlined project #includes.
#include "simdrng-scalar.hpp"

int main() {
  simdrng::SplitMix s(42);
  simdrng::Xoshiro  x(42);          // scalar engine in this variant
  return (s() == 0 || x() == 0) ? 1 : 0;
}
