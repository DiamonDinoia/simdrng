/*  Written in 2014 originally by Guy Steele

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide.

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR
IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

Ported to C++ by Marco Barbone. Original implementation by Guy Steele.
*/

#pragma once

#include <cstdint>
#include <limits>

#include "macros.hpp"

namespace simdrng {

class SplitMix {
public:
  SIMDRNG_ALWAYS_INLINE constexpr explicit SplitMix(const std::uint64_t state) noexcept : m_state(state) {}

  SIMDRNG_ALWAYS_INLINE constexpr std::uint64_t operator()() noexcept {
    std::uint64_t z = (m_state += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
  }

  static constexpr SIMDRNG_ALWAYS_INLINE std::uint64_t(min)() noexcept {
    return std::numeric_limits<std::uint64_t>::lowest();
  }

  static constexpr SIMDRNG_ALWAYS_INLINE std::uint64_t(max)() noexcept {
    return std::numeric_limits<std::uint64_t>::max();
  }

  SIMDRNG_ALWAYS_INLINE constexpr std::uint64_t getState() const noexcept { return m_state; }

  SIMDRNG_ALWAYS_INLINE constexpr void setState(std::uint64_t state) noexcept { m_state = state; }

private:
  std::uint64_t m_state;
};

} // namespace simdrng
