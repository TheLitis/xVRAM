#include "residency/budget.hpp"

#include <algorithm>
#include <limits>

namespace xvram::residency {
namespace {

[[nodiscard]] std::uint64_t saturating_add(const std::uint64_t left,
                                           const std::uint64_t right) noexcept {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    return std::numeric_limits<std::uint64_t>::max();
  }
  return left + right;
}

[[nodiscard]] std::uint64_t saturating_subtract(const std::uint64_t left,
                                                const std::uint64_t right) noexcept {
  return right >= left ? 0 : left - right;
}

[[nodiscard]] std::optional<std::uint64_t> checked_multiply(const std::uint64_t left,
                                                            const std::uint64_t right) noexcept {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
    return std::nullopt;
  }
  return left * right;
}

[[nodiscard]] std::optional<std::uint64_t>
checked_align_up(const std::uint64_t value, const std::uint64_t alignment) noexcept {
  if (alignment == 0) {
    return std::nullopt;
  }
  const std::uint64_t remainder = value % alignment;
  if (remainder == 0) {
    return value;
  }
  const std::uint64_t increment = alignment - remainder;
  if (increment > std::numeric_limits<std::uint64_t>::max() - value) {
    return std::nullopt;
  }
  return value + increment;
}

} // namespace

std::string_view budget_target_error_name(const BudgetTargetError error) noexcept {
  switch (error) {
  case BudgetTargetError::none:
    return "none";
  case BudgetTargetError::invalid_configuration:
    return "invalid_configuration";
  case BudgetTargetError::arithmetic_overflow:
    return "arithmetic_overflow";
  case BudgetTargetError::below_minimum:
    return "below_minimum";
  }
  return "invalid";
}

BudgetTarget calculate_budget_target(const BudgetTargetInput& input) noexcept {
  BudgetTarget result;
  if (input.configured_cap_bytes == 0 || input.chunk_bytes == 0) {
    result.error = BudgetTargetError::invalid_configuration;
    return result;
  }

  const std::optional<std::uint64_t> two_chunks = checked_multiply(input.chunk_bytes, 2);
  const std::optional<std::uint64_t> aligned_working_set =
      checked_align_up(input.maximum_working_set_bytes, input.chunk_bytes);
  if (!two_chunks.has_value() || !aligned_working_set.has_value()) {
    result.error = BudgetTargetError::arithmetic_overflow;
    return result;
  }
  result.required_minimum_bytes = std::max(*two_chunks, *aligned_working_set);

  result.cuda_reclaimable_bytes = saturating_add(input.cuda_free_bytes, input.managed_frame_bytes);
  result.limiting_reclaimable_bytes =
      std::min(input.configured_cap_bytes, result.cuda_reclaimable_bytes);

  if (input.wddm.has_value()) {
    result.wddm_available_bytes =
        saturating_subtract(input.wddm->budget_bytes, input.wddm->current_usage_bytes);
    result.wddm_reclaimable_bytes =
        saturating_add(*result.wddm_available_bytes, input.managed_frame_bytes);
    result.limiting_reclaimable_bytes =
        std::min(result.limiting_reclaimable_bytes, *result.wddm_reclaimable_bytes);
  }

  result.limited_by_configured_cap =
      input.configured_cap_bytes == result.limiting_reclaimable_bytes;
  result.limited_by_cuda = result.cuda_reclaimable_bytes == result.limiting_reclaimable_bytes;
  result.limited_by_wddm =
      input.wddm.has_value() && *result.wddm_reclaimable_bytes == result.limiting_reclaimable_bytes;
  result.bytes_after_headroom =
      saturating_subtract(result.limiting_reclaimable_bytes, input.device_headroom_bytes);
  result.target_bytes = (result.bytes_after_headroom / input.chunk_bytes) * input.chunk_bytes;
  if (result.target_bytes < result.required_minimum_bytes) {
    result.error = BudgetTargetError::below_minimum;
  }
  return result;
}

std::string_view target_action_name(const TargetAction action) noexcept {
  switch (action) {
  case TargetAction::unchanged:
    return "unchanged";
  case TargetAction::shrink:
    return "shrink";
  case TargetAction::grow:
    return "grow";
  case TargetAction::budget_pressure:
    return "budget_pressure";
  case TargetAction::invalid_configuration:
    return "invalid_configuration";
  }
  return "invalid";
}

TargetDecision observe_safe_target(TargetHysteresisState& state,
                                   const std::uint64_t observed_safe_target_bytes,
                                   const std::uint64_t required_minimum_bytes,
                                   const TargetHysteresisConfig& config) noexcept {
  TargetDecision result;
  result.previous_target_bytes = state.current_target_bytes;
  result.observed_safe_target_bytes = observed_safe_target_bytes;
  result.effective_target_bytes = state.current_target_bytes;

  const bool aligned = config.chunk_bytes != 0 &&
                       state.current_target_bytes % config.chunk_bytes == 0 &&
                       observed_safe_target_bytes % config.chunk_bytes == 0 &&
                       required_minimum_bytes % config.chunk_bytes == 0;
  const std::optional<std::uint64_t> growth_step = checked_multiply(
      config.chunk_bytes, static_cast<std::uint64_t>(config.growth_chunks_per_cycle));
  const std::optional<std::uint64_t> growth_reserve = checked_multiply(
      config.chunk_bytes, static_cast<std::uint64_t>(config.growth_reserve_chunks));
  if (!aligned || required_minimum_bytes == 0 || config.safe_samples_before_growth == 0 ||
      config.growth_chunks_per_cycle == 0 || !growth_step.has_value() ||
      !growth_reserve.has_value()) {
    result.action = TargetAction::invalid_configuration;
    result.consecutive_safe_samples = state.consecutive_safe_samples;
    return result;
  }

  if (observed_safe_target_bytes < required_minimum_bytes) {
    state.current_target_bytes = observed_safe_target_bytes;
    state.consecutive_safe_samples = 0;
    result.action = TargetAction::budget_pressure;
    result.effective_target_bytes = state.current_target_bytes;
    return result;
  }
  if (observed_safe_target_bytes < state.current_target_bytes) {
    state.current_target_bytes = observed_safe_target_bytes;
    state.consecutive_safe_samples = 0;
    result.action = TargetAction::shrink;
    result.effective_target_bytes = state.current_target_bytes;
    return result;
  }

  const std::uint64_t available_growth = observed_safe_target_bytes - state.current_target_bytes;
  if (available_growth < *growth_reserve) {
    state.consecutive_safe_samples = 0;
    result.consecutive_safe_samples = 0;
    return result;
  }

  if (state.consecutive_safe_samples != std::numeric_limits<std::uint32_t>::max()) {
    ++state.consecutive_safe_samples;
  }
  if (state.consecutive_safe_samples < config.safe_samples_before_growth) {
    result.consecutive_safe_samples = state.consecutive_safe_samples;
    return result;
  }

  state.current_target_bytes += std::min(available_growth, *growth_step);
  state.consecutive_safe_samples = 0;
  result.action = TargetAction::grow;
  result.effective_target_bytes = state.current_target_bytes;
  result.consecutive_safe_samples = 0;
  return result;
}

} // namespace xvram::residency
