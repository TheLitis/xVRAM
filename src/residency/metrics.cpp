#include "residency/metrics.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <optional>

namespace xvram::residency {
namespace {

[[nodiscard]] std::optional<std::uint64_t> checked_add(const std::uint64_t left,
                                                       const std::uint64_t right) noexcept {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    return std::nullopt;
  }
  return left + right;
}

template <std::size_t Size>
[[nodiscard]] std::optional<std::uint64_t>
checked_sum(const std::array<std::uint64_t, Size>& values) noexcept {
  std::uint64_t total = 0;
  for (const std::uint64_t value : values) {
    const std::optional<std::uint64_t> next = checked_add(total, value);
    if (!next.has_value()) {
      return std::nullopt;
    }
    total = *next;
  }
  return total;
}

} // namespace

std::string_view metric_issue_name(const MetricIssue issue) noexcept {
  switch (issue) {
  case MetricIssue::demand_accounting:
    return "demand_accounting";
  case MetricIssue::prefetch_accounting:
    return "prefetch_accounting";
  case MetricIssue::dirty_eviction_writeback_accounting:
    return "dirty_eviction_writeback_accounting";
  case MetricIssue::total_writeback_accounting:
    return "total_writeback_accounting";
  case MetricIssue::mapping_access_accounting:
    return "mapping_access_accounting";
  case MetricIssue::mapping_lifecycle_accounting:
    return "mapping_lifecycle_accounting";
  case MetricIssue::handle_lifecycle_accounting:
    return "handle_lifecycle_accounting";
  case MetricIssue::mappings_exceed_live_handles:
    return "mappings_exceed_live_handles";
  case MetricIssue::residency_exceeds_target:
    return "residency_exceeds_target";
  case MetricIssue::staging_exceeds_capacity:
    return "staging_exceeds_capacity";
  case MetricIssue::unsafe_remap:
    return "unsafe_remap";
  }
  return "invalid";
}

bool MetricsReconciliation::contains(const MetricIssue issue) const noexcept {
  return std::find(issues.begin(), issues.end(), issue) != issues.end();
}

MetricsReconciliation reconcile_metrics(const CacheMetrics& metrics) {
  MetricsReconciliation result;
  const auto record_if = [&](const bool condition, const MetricIssue issue) {
    if (condition) {
      result.issues.push_back(issue);
    }
  };

  const std::optional<std::uint64_t> demands =
      checked_sum<2>({metrics.cache_hits, metrics.cache_misses});
  record_if(!demands.has_value() || *demands != metrics.demand_accesses,
            MetricIssue::demand_accounting);

  const std::optional<std::uint64_t> prefetch_terminal =
      checked_sum<4>({metrics.prefetch_useful, metrics.prefetch_wasted, metrics.prefetch_cancelled,
                      metrics.prefetch_in_flight});
  record_if(!prefetch_terminal.has_value() || *prefetch_terminal != metrics.prefetch_issued,
            MetricIssue::prefetch_accounting);

  record_if(metrics.dirty_evictions != metrics.eviction_writebacks_completed,
            MetricIssue::dirty_eviction_writeback_accounting);
  const std::optional<std::uint64_t> writebacks =
      checked_sum<2>({metrics.eviction_writebacks_completed, metrics.drain_writebacks_completed});
  record_if(!writebacks.has_value() || *writebacks != metrics.writebacks_completed,
            MetricIssue::total_writeback_accounting);

  record_if(metrics.mappings_completed != metrics.set_access_completed,
            MetricIssue::mapping_access_accounting);
  const std::optional<std::uint64_t> mapping_terminal =
      checked_sum<2>({metrics.unmaps_completed, metrics.active_mappings});
  record_if(!mapping_terminal.has_value() || *mapping_terminal != metrics.mappings_completed,
            MetricIssue::mapping_lifecycle_accounting);

  const std::optional<std::uint64_t> handle_terminal =
      checked_sum<2>({metrics.handles_released, metrics.live_handles});
  record_if(!handle_terminal.has_value() || *handle_terminal != metrics.handles_created,
            MetricIssue::handle_lifecycle_accounting);
  record_if(metrics.active_mappings > metrics.live_handles,
            MetricIssue::mappings_exceed_live_handles);
  record_if(metrics.resident_peak_bytes > metrics.target_peak_bytes,
            MetricIssue::residency_exceeds_target);
  record_if(metrics.staging_slots_peak > metrics.staging_slots_capacity,
            MetricIssue::staging_exceeds_capacity);
  record_if(metrics.unsafe_remaps != 0, MetricIssue::unsafe_remap);
  return result;
}

bool accumulate_metrics(CacheMetrics& accumulator, const CacheMetrics& delta) noexcept {
  CacheMetrics combined = accumulator;
  const auto add = [](std::uint64_t& target, const std::uint64_t increment) {
    const std::optional<std::uint64_t> value = checked_add(target, increment);
    if (!value.has_value()) {
      return false;
    }
    target = *value;
    return true;
  };

  if (!add(combined.demand_accesses, delta.demand_accesses) ||
      !add(combined.cache_hits, delta.cache_hits) ||
      !add(combined.cache_misses, delta.cache_misses) ||
      !add(combined.prefetch_issued, delta.prefetch_issued) ||
      !add(combined.prefetch_useful, delta.prefetch_useful) ||
      !add(combined.prefetch_wasted, delta.prefetch_wasted) ||
      !add(combined.prefetch_cancelled, delta.prefetch_cancelled) ||
      !add(combined.prefetch_in_flight, delta.prefetch_in_flight) ||
      !add(combined.clean_evictions, delta.clean_evictions) ||
      !add(combined.dirty_evictions, delta.dirty_evictions) ||
      !add(combined.eviction_writebacks_completed, delta.eviction_writebacks_completed) ||
      !add(combined.drain_writebacks_completed, delta.drain_writebacks_completed) ||
      !add(combined.writebacks_completed, delta.writebacks_completed) ||
      !add(combined.mappings_completed, delta.mappings_completed) ||
      !add(combined.set_access_completed, delta.set_access_completed) ||
      !add(combined.unmaps_completed, delta.unmaps_completed) ||
      !add(combined.active_mappings, delta.active_mappings) ||
      !add(combined.handles_created, delta.handles_created) ||
      !add(combined.handles_released, delta.handles_released) ||
      !add(combined.live_handles, delta.live_handles) ||
      !add(combined.handle_reuses, delta.handle_reuses) ||
      !add(combined.h2d_bytes, delta.h2d_bytes) || !add(combined.d2h_bytes, delta.d2h_bytes) ||
      !add(combined.unsafe_remaps, delta.unsafe_remaps)) {
    return false;
  }

  combined.resident_peak_bytes = std::max(combined.resident_peak_bytes, delta.resident_peak_bytes);
  combined.target_peak_bytes = std::max(combined.target_peak_bytes, delta.target_peak_bytes);
  combined.staging_slots_peak = std::max(combined.staging_slots_peak, delta.staging_slots_peak);
  combined.staging_slots_capacity =
      std::max(combined.staging_slots_capacity, delta.staging_slots_capacity);
  accumulator = combined;
  return true;
}

} // namespace xvram::residency
