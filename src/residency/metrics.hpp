#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace xvram::residency {

struct CacheMetrics {
  std::uint64_t demand_accesses = 0;
  std::uint64_t cache_hits = 0;
  std::uint64_t cache_misses = 0;

  std::uint64_t prefetch_issued = 0;
  std::uint64_t prefetch_useful = 0;
  std::uint64_t prefetch_wasted = 0;
  std::uint64_t prefetch_cancelled = 0;
  std::uint64_t prefetch_in_flight = 0;
  std::uint64_t prefetch_promoted = 0;

  std::uint64_t clean_evictions = 0;
  std::uint64_t dirty_evictions = 0;
  std::uint64_t eviction_writebacks_completed = 0;
  std::uint64_t drain_writebacks_completed = 0;
  std::uint64_t writebacks_completed = 0;

  std::uint64_t mappings_completed = 0;
  std::uint64_t set_access_completed = 0;
  std::uint64_t unmaps_completed = 0;
  std::uint64_t active_mappings = 0;

  std::uint64_t handles_created = 0;
  std::uint64_t handles_released = 0;
  std::uint64_t live_handles = 0;
  std::uint64_t handle_reuses = 0;

  std::uint64_t h2d_bytes = 0;
  std::uint64_t d2h_bytes = 0;
  std::uint64_t resident_peak_bytes = 0;
  std::uint64_t target_peak_bytes = 0;
  std::uint32_t staging_slots_peak = 0;
  std::uint32_t staging_slots_capacity = 0;
  std::uint64_t unsafe_remaps = 0;
};

enum class MetricIssue {
  demand_accounting,
  prefetch_accounting,
  dirty_eviction_writeback_accounting,
  total_writeback_accounting,
  mapping_access_accounting,
  mapping_lifecycle_accounting,
  handle_lifecycle_accounting,
  mappings_exceed_live_handles,
  residency_exceeds_target,
  staging_exceeds_capacity,
  unsafe_remap,
};

[[nodiscard]] std::string_view metric_issue_name(MetricIssue issue) noexcept;

struct MetricsReconciliation {
  std::vector<MetricIssue> issues;

  [[nodiscard]] explicit operator bool() const noexcept {
    return issues.empty();
  }

  [[nodiscard]] bool contains(MetricIssue issue) const noexcept;
};

[[nodiscard]] MetricsReconciliation reconcile_metrics(const CacheMetrics& metrics);

// Combines counters that are additive across workloads. High-water marks are maximized. This
// helper is overflow-checked; failure leaves the accumulator unchanged.
[[nodiscard]] bool accumulate_metrics(CacheMetrics& accumulator,
                                      const CacheMetrics& delta) noexcept;

} // namespace xvram::residency
