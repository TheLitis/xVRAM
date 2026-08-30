#include "gemm_bench/sizing.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace xvram::gemm_bench {
namespace {

[[nodiscard]] std::optional<std::uint64_t> checked_add(const std::uint64_t left,
                                                       const std::uint64_t right) noexcept {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    return std::nullopt;
  }
  return left + right;
}

[[nodiscard]] std::optional<std::uint64_t> checked_multiply(const std::uint64_t left,
                                                            const std::uint64_t right) noexcept {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
    return std::nullopt;
  }
  return left * right;
}

} // namespace

std::optional<std::uint64_t> safe_host_logical_limit(const HostSizingInput& input) noexcept {
  if (input.physical_memory_bytes == 0 || input.available_memory_bytes == 0 ||
      input.chunk_bytes == 0 || input.staging_slots == 0) {
    return std::nullopt;
  }
  constexpr std::uint64_t minimum_headroom = 4ULL * 1024ULL * 1024ULL * 1024ULL;
  const std::uint64_t headroom = std::max(minimum_headroom, input.physical_memory_bytes / 4ULL);
  const auto pinned = checked_multiply(input.chunk_bytes, input.staging_slots);
  const auto with_pinned = pinned.has_value() ? checked_add(headroom, *pinned) : std::nullopt;
  const auto reserved = with_pinned.has_value()
                            ? checked_add(*with_pinned, input.service_reserve_bytes)
                            : std::nullopt;
  if (!reserved.has_value() || *reserved >= input.available_memory_bytes) {
    return std::nullopt;
  }
  return input.available_memory_bytes - *reserved;
}

std::optional<AutoGemmShape>
choose_auto_gemm_shape(const std::uint64_t total_vram_bytes, const std::uint64_t element_bytes,
                       const std::uint64_t safe_host_limit_bytes) noexcept {
  if (total_vram_bytes == 0 || element_bytes == 0 || safe_host_limit_bytes == 0) {
    return std::nullopt;
  }
  const auto half_rounded_up = checked_add(total_vram_bytes / 2ULL, total_vram_bytes % 2ULL);
  const auto automatic_target =
      half_rounded_up.has_value() ? checked_add(total_vram_bytes, *half_rounded_up) : std::nullopt;
  if (!automatic_target.has_value()) {
    return std::nullopt;
  }
  const std::uint64_t target = std::min(*automatic_target, safe_host_limit_bytes);
  constexpr std::uint64_t k = 256;
  const long double element_capacity = static_cast<long double>(target / element_bytes);
  const long double root =
      std::sqrt(static_cast<long double>(k) * static_cast<long double>(k) + element_capacity) -
      static_cast<long double>(k);
  if (!std::isfinite(root) || root < 256.0L ||
      root > static_cast<long double>(std::numeric_limits<std::uint64_t>::max())) {
    return std::nullopt;
  }
  std::uint64_t side = static_cast<std::uint64_t>(std::floor(root));
  side -= side % 256ULL;
  if (side == 0) {
    return std::nullopt;
  }
  const auto square = checked_multiply(side, side);
  const auto side_k = checked_multiply(side, k);
  const auto rectangular = side_k.has_value() ? checked_multiply(*side_k, 2ULL) : std::nullopt;
  const auto elements = square.has_value() && rectangular.has_value()
                            ? checked_add(*square, *rectangular)
                            : std::nullopt;
  const auto logical =
      elements.has_value() ? checked_multiply(*elements, element_bytes) : std::nullopt;
  if (!logical.has_value() || *logical > target || *logical > safe_host_limit_bytes ||
      *logical <= total_vram_bytes) {
    return std::nullopt;
  }
  return AutoGemmShape{side, side, k, *logical};
}

} // namespace xvram::gemm_bench
