#include "random/xoshiro_simd.hpp"

namespace prng {

XoshiroSIMD::XoshiroSIMD(const result_type seed, const result_type thread_id, const result_type cluster_id) noexcept
    : m_cache{}, m_state{}, m_index{0} {
  auto result =
      xsimd::dispatch<dispatch_arch_list>(
          internal::XoshiroSIMDInitFunctor{m_state.data, seed, thread_id, cluster_id})();
  m_populate_cache = result.populate_cache;
  m_jump = result.jump;
  m_mid_jump = result.mid_jump;
  m_long_jump = result.long_jump;
}

} // namespace prng
