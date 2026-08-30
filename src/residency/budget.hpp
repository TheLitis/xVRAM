#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

namespace xvram::residency {

struct WddmBudgetObservation {
  std::uint64_t budget_bytes = 0;
  std::uint64_t current_usage_bytes = 0;
};

struct BudgetTargetInput {
  std::uint64_t configured_cap_bytes = 0;
  std::uint64_t cuda_free_bytes = 0;
  std::uint64_t managed_frame_bytes = 0;
  std::optional<WddmBudgetObservation> wddm;
  std::uint64_t device_headroom_bytes = 0;
  std::uint64_t chunk_bytes = 0;
  std::uint64_t maximum_working_set_bytes = 0;
};

enum class BudgetTargetError {
  none,
  invalid_configuration,
  arithmetic_overflow,
  below_minimum,
};

[[nodiscard]] std::string_view budget_target_error_name(BudgetTargetError error) noexcept;

struct BudgetTarget {
  BudgetTargetError error = BudgetTargetError::none;
  std::uint64_t cuda_reclaimable_bytes = 0;
  std::optional<std::uint64_t> wddm_available_bytes;
  std::optional<std::uint64_t> wddm_reclaimable_bytes;
  std::uint64_t limiting_reclaimable_bytes = 0;
  std::uint64_t bytes_after_headroom = 0;
  std::uint64_t target_bytes = 0;
  std::uint64_t required_minimum_bytes = 0;
  bool limited_by_configured_cap = false;
  bool limited_by_cuda = false;
  bool limited_by_wddm = false;

  [[nodiscard]] explicit operator bool() const noexcept {
    return error == BudgetTargetError::none;
  }
};

// Addition and subtraction of externally observed byte counters are saturating. Overflow while
// deriving mandatory aligned sizes is reported instead of silently weakening the minimum.
[[nodiscard]] BudgetTarget calculate_budget_target(const BudgetTargetInput& input) noexcept;

struct TargetHysteresisConfig {
  std::uint64_t chunk_bytes = 0;
  std::uint32_t safe_samples_before_growth = 10;
  std::uint32_t growth_chunks_per_cycle = 2;
  std::uint32_t growth_reserve_chunks = 2;
};

struct TargetHysteresisState {
  std::uint64_t current_target_bytes = 0;
  std::uint32_t consecutive_safe_samples = 0;
};

enum class TargetAction {
  unchanged,
  shrink,
  grow,
  budget_pressure,
  invalid_configuration,
};

[[nodiscard]] std::string_view target_action_name(TargetAction action) noexcept;

struct TargetDecision {
  TargetAction action = TargetAction::unchanged;
  std::uint64_t previous_target_bytes = 0;
  std::uint64_t observed_safe_target_bytes = 0;
  std::uint64_t effective_target_bytes = 0;
  std::uint32_t consecutive_safe_samples = 0;
};

// Shrink is immediate. Growth requires a reserve above the current target for every sample in the
// configured run and is rate limited per decision.
[[nodiscard]] TargetDecision observe_safe_target(TargetHysteresisState& state,
                                                 std::uint64_t observed_safe_target_bytes,
                                                 std::uint64_t required_minimum_bytes,
                                                 const TargetHysteresisConfig& config) noexcept;

} // namespace xvram::residency
