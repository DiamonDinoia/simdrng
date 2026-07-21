#include "simdrng/xoshiro_simd.hpp"

namespace simdrng {

XoshiroSIMD::XoshiroSIMD(const result_type seed, const result_type thread_id, const result_type cluster_id) noexcept {
  auto result = xsimd::dispatch<dispatch_arch_list>(internal::XoshiroSIMDInitFunctor{
      .state_storage = m_state.data, .seed = seed, .thread_id = thread_id, .cluster_id = cluster_id})();
  m_populate_cache = result.populate_cache;
  m_generate = result.generate_blocks;
  m_jump = result.jump;
  m_mid_jump = result.mid_jump;
  m_long_jump = result.long_jump;
  m_jump_n = result.jump_n;
  m_get_state = result.get_state;
  m_set_state = result.set_state;
  m_simd_width = result.simd_width;
}

} // namespace simdrng
