#include "compat_bench/report.hpp"
#include "platform/system_info.hpp"
#include "xvram/base/json_writer.hpp"
#include "xvram/version.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <numeric>
#include <sstream>

namespace xvram::compat_bench {
namespace {
template <class T> void field(JsonWriter& j, const char* name, const T& value) {
  j.key(name);
  j.value(value);
}
void field(JsonWriter& j, const char* name, const std::uint32_t value) {
  field(j, name, static_cast<std::uint64_t>(value));
}
void field(JsonWriter& j, const char* name, const int value) {
  field(j, name, static_cast<std::int64_t>(value));
}
template <class T> void field(JsonWriter& j, const char* name, const std::optional<T>& value) {
  j.key(name);
  if (value)
    j.value(*value);
  else
    j.null_value();
}
const char* outcome(const int code) {
  switch (code) {
  case 0:
    return "completed";
  case 23:
    return "skipped";
  case 24:
    return "corruption";
  case 25:
    return "oom";
  case 26:
    return "timeout";
  default:
    return "failed";
  }
}
void timings(JsonWriter& j, const std::vector<double>& values) {
  j.begin_object();
  field(j, "sample_count", static_cast<std::uint64_t>(values.size()));
  std::vector<double> sorted = values;
  std::sort(sorted.begin(), sorted.end());
  const auto statistic = [&](const char* name, const double fraction) {
    j.key(name);
    if (sorted.empty())
      j.null_value();
    else
      j.value(
          sorted[std::min(sorted.size() - 1U, static_cast<std::size_t>(std::ceil(
                                                  fraction * static_cast<double>(sorted.size()))) -
                                                  1U)]);
  };
  statistic("minimum_ms", 0.00001);
  statistic("median_ms", .5);
  statistic("p95_ms", .95);
  statistic("maximum_ms", 1.0);
  field(j, "total_ms", std::accumulate(values.begin(), values.end(), 0.0));
  j.end_object();
}
} // namespace
Report base_report(const Options& options) {
  Report report;
  report.options = options;
  report.system = platform::collect_system_info();
  report.telemetry.struct_size = sizeof(report.telemetry);
  const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::tm utc{};
#ifdef _WIN32
  gmtime_s(&utc, &now);
#else
  gmtime_r(&now, &utc);
#endif
  std::ostringstream date;
  date << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
  report.generated_at_utc = date.str();
  for (const char* key :
       {"supported_calls_routed", "reference_equal", "padding_preserved",
        "rejections_before_submission", "stable_addresses", "no_physical_aliases",
        "mapping_access_balanced", "mapping_unmap_balanced", "event_safe", "zero_unsafe_activity",
        "bounded_cache", "real_oversubscription_reuse", "cleanup_complete"})
    report.proof[key] = std::nullopt;
  for (const char* key :
       {"operations_drained", "events_drained", "allocations_released", "handles_destroyed",
        "reservations_freed", "adapter_closed", "worker_terminated", "trace_closed"})
    report.cleanup[key] = std::nullopt;
  return report;
}
void complete_proof(Report& r) {
  const auto& t = r.telemetry;
  const auto& c = t.runtime;
  if (!r.telemetry_observed)
    return;
  const bool computed = !r.workloads.empty();
  const bool verified =
      computed && std::all_of(r.workloads.begin(), r.workloads.end(), [](const auto& w) {
        return w.status == "completed" && w.mismatches == 0 && w.output_elements_checked > 0;
      });
  r.proof["supported_calls_routed"] =
      t.calls_completed > 0 && (r.options.scenario == "rejection" || t.gemm_calls > 0);
  r.proof["reference_equal"] =
      r.options.scenario == "rejection" ? r.rejections_failed == 0 : verified;
  r.proof["padding_preserved"] = r.options.scenario == "rejection" || verified;
  r.proof["rejections_before_submission"] = r.rejections_failed == 0;
  r.proof["stable_addresses"] = (c.flags & XVRAM_TELEMETRY_STABLE_VIRTUAL_ADDRESSES) != 0;
  r.proof["no_physical_aliases"] = (c.flags & XVRAM_TELEMETRY_NO_PHYSICAL_ALIASES) != 0;
  r.proof["mapping_access_balanced"] = c.mappings == c.set_access_calls;
  r.proof["mapping_unmap_balanced"] = c.mappings == c.unmaps;
  // event_boundaries counts completed boundaries consumed by unmap, not every compute
  // transaction. Resident cache hits retire transactions without another mapping/unmap.
  r.proof["event_safe"] = t.tiles_submitted == t.tiles_retired &&
                          c.transactions_completed == t.tiles_retired &&
                          c.event_boundaries == c.unmaps && t.cleanup_events_drained != 0;
  r.proof["zero_unsafe_activity"] =
      c.unsafe_remaps == 0 && c.unsafe_transitions == 0 && t.quarantined == 0;
  r.proof["bounded_cache"] = c.resident_bytes_peak <= c.cache_target_maximum_bytes;
  r.proof["real_oversubscription_reuse"] =
      std::all_of(r.workloads.begin(), r.workloads.end(), [&](const auto& w) {
        return w.logical_bytes <= t.total_vram_bytes || (w.evictions > 0 && w.handle_reuses > 0);
      });
  r.cleanup["operations_drained"] = t.cleanup_operations_drained != 0;
  r.cleanup["events_drained"] = t.cleanup_events_drained != 0;
  r.cleanup["allocations_released"] =
      t.allocations_created == t.allocations_released && t.live_allocations == 0;
  r.cleanup["handles_destroyed"] = t.handles_created == t.handles_destroyed && t.live_handles == 0;
  r.cleanup["reservations_freed"] = t.retired_va_reservations == 0 && t.retired_va_bytes == 0;
  r.cleanup["adapter_closed"] = t.cleanup_completed != 0;
  r.proof["cleanup_complete"] = t.cleanup_completed != 0 && t.quarantined == 0;
  if (r.exit_code == 0 && std::any_of(r.proof.begin(), r.proof.end(),
                                      [](const auto& flag) { return flag.second != true; })) {
    r.exit_code = 27;
    r.reason = "proof_incomplete";
    r.message = "observed telemetry did not satisfy all selected proof invariants";
  }
}
std::string report_json(const Report& r, const bool pretty) {
  std::ostringstream out;
  JsonWriter j(out, pretty);
  const auto& o = r.options;
  const auto& t = r.telemetry;
  const auto& c = t.runtime;
  j.begin_object();
  field(j, "schema_version", 1);
  field(j, "report_type", "xvram.cuda_compat");
  field(j, "generated_at_utc", r.generated_at_utc);
  j.key("build");
  j.begin_object();
  field(j, "version", XVRAM_VERSION);
  field(j, "git_commit", XVRAM_GIT_COMMIT);
  field(j, "cuda_headers_version", XVRAM_CUDA_HEADERS_VERSION);
#ifdef NDEBUG
  field(j, "build_type", "Release");
#else
  field(j, "build_type", "Debug");
#endif
  j.end_object();
  j.key("system");
  j.begin_object();
  field(j, "os_name", r.system.os_name);
  field(j, "os_version", r.system.os_version);
  field(j, "architecture", r.system.architecture);
  field(j, "logical_processor_count", r.system.logical_processor_count);
  field(j, "physical_memory_bytes", r.system.physical_memory_bytes);
  field(j, "available_memory_bytes", r.system.available_memory_bytes);
  j.end_object();
  j.key("device");
  j.begin_object();
  field(j, "ordinal", o.device);
  field(j, "name", r.telemetry_observed ? std::string(t.device_name) : std::string{});
  field(j, "total_memory_bytes", t.total_vram_bytes);
  field(j, "cuda_driver_version", c.cuda_driver_version);
  field(j, "cublas_version", c.cublas_version);
  field(j, "cublas_lt_version", c.cublas_lt_version);
  field(j, "cublas_library_source", c.cublas_library_source);
  field(j, "uuid", o.identifiers ? r.uuid : std::nullopt);
  field(j, "pci_bus_id", o.identifiers ? r.pci_bus_id : std::nullopt);
  j.end_object();
  j.key("configuration");
  j.begin_object();
  field(j, "scenario", o.scenario);
  field(j, "policy", o.policy);
  field(j, "requested_logical_bytes", o.logical_bytes);
  field(j, "requested_m", o.m);
  field(j, "requested_n", o.n);
  field(j, "requested_k", o.k);
  field(j, "chunk_bytes", o.chunk_bytes);
  field(j, "effective_chunk_bytes", t.effective_chunk_bytes);
  field(j, "host_store_cap_bytes", t.host_store_cap_bytes);
  field(j, "host_headroom_bytes", t.host_headroom_bytes);
  field(j, "host_budget_bytes", t.host_budget_bytes);
  field(j, "cache_target_bytes", o.cache_bytes);
  field(j, "device_headroom_bytes", o.headroom_bytes);
  field(j, "workspace_bytes", o.workspace_bytes);
  field(j, "staging_slots", o.staging_slots);
  field(j, "prefetch_distance", o.prefetch_distance);
  field(j, "budget_poll_ms", o.budget_poll_ms);
  field(j, "stall_timeout_ms", o.stall_timeout_ms);
  field(j, "timeout_ms", static_cast<std::uint64_t>(o.timeout.count()));
  field(j, "passes", o.passes);
  field(j, "alpha", static_cast<double>(o.alpha));
  field(j, "beta", static_cast<double>(o.beta));
  field(j, "padding_elements", o.padding);
  field(j, "offset_elements", o.offset_elements);
  field(j, "op_a", o.transpose_a ? "t" : "n");
  field(j, "op_b", o.transpose_b ? "t" : "n");
  field(j, "seed", o.seed);
  field(j, "trace_enabled", o.trace_path.has_value());
  field(j, "identifiers_included", o.identifiers);
  field(j, "compression", "disabled");
  j.end_object();
  j.key("compatibility");
  j.begin_object();
  field(j, "abi_version", 1);
  field(j, "profile", "synchronous_fp32_sgemm");
  field(j, "native_fallback", false);
  field(j, "calls_attempted", t.calls_attempted);
  field(j, "calls_submitted", t.calls_submitted);
  field(j, "calls_completed", t.calls_completed);
  field(j, "calls_rejected", t.calls_rejected);
  field(j, "rejections_checked", r.rejections_checked);
  field(j, "rejections_failed", r.rejections_failed);
  field(j, "allocations_created", t.allocations_created);
  field(j, "allocations_released", t.allocations_released);
  field(j, "live_allocations", t.live_allocations);
  field(j, "handles_created", t.handles_created);
  field(j, "handles_destroyed", t.handles_destroyed);
  field(j, "live_handles", t.live_handles);
  field(j, "retired_va_reservations", t.retired_va_reservations);
  field(j, "retired_va_reservations_freed", t.retired_va_reservations_freed);
  field(j, "retired_va_bytes", t.retired_va_bytes);
  field(j, "retired_va_bytes_freed", t.retired_va_bytes_freed);
  j.end_object();
  j.key("workloads");
  j.begin_array();
  for (const auto& w : r.workloads) {
    j.begin_object();
    field(j, "name", w.name);
    field(j, "status", w.status);
    field(j, "m", w.shape.m);
    field(j, "n", w.shape.n);
    field(j, "k", w.shape.k);
    field(j, "op_a", w.shape.transpose_a ? "t" : "n");
    field(j, "op_b", w.shape.transpose_b ? "t" : "n");
    field(j, "padding_elements", w.shape.padding);
    field(j, "offset_elements", w.shape.offset);
    field(j, "logical_bytes", w.logical_bytes);
    field(j, "storage_bytes", w.storage_bytes);
    field(j, "passes_completed", w.passes_completed);
    field(j, "output_elements_checked", w.output_elements_checked);
    field(j, "padding_elements_checked", w.padding_elements_checked);
    field(j, "mismatches", w.mismatches);
    field(j, "mismatch_offset", w.mismatch_offset);
    field(j, "max_absolute_error", w.max_absolute_error);
    field(j, "max_relative_error", w.max_relative_error);
    field(j, "digest", w.digest);
    field(j, "reference_digest", w.reference_digest);
    field(j, "reference_kind", w.reference_kind);
    field(j, "native_baseline_equal", w.native_baseline_equal);
    field(j, "native_baseline_ms", w.native_baseline_ms);
    field(j, "tiles_retired", w.tiles_retired);
    field(j, "mappings", w.mappings);
    field(j, "unmaps", w.unmaps);
    field(j, "h2d_bytes", w.h2d_bytes);
    field(j, "d2h_bytes", w.d2h_bytes);
    field(j, "evictions", w.evictions);
    field(j, "handle_reuses", w.handle_reuses);
    j.key("pass_timings");
    timings(j, w.pass_timings_ms);
    j.end_object();
  }
  j.end_array();
  j.key("execution");
  j.begin_object();
  field(j, "telemetry_observed", r.telemetry_observed);
  field(j, "gemm_calls", t.gemm_calls);
  field(j, "tiles_submitted", t.tiles_submitted);
  field(j, "tiles_retired", t.tiles_retired);
  field(j, "progress_sequence", t.progress_sequence);
  field(j, "host_h2d_bytes", t.h2d_bytes);
  field(j, "host_d2h_bytes", t.d2h_bytes);
  field(j, "logical_bytes_peak", t.logical_bytes_peak);
  field(j, "host_backing_peak_bytes", t.host_backing_peak_bytes);
  field(j, "trace_records", r.trace_records);
  field(j, "trace_complete", r.trace_complete);
  j.end_object();
  j.key("cache");
  j.begin_object();
#define CACHE_FIELD(name) field(j, #name, c.name)
  CACHE_FIELD(cache_target_bytes);
  CACHE_FIELD(cache_target_minimum_bytes);
  CACHE_FIELD(cache_target_maximum_bytes);
  CACHE_FIELD(resident_bytes);
  CACHE_FIELD(resident_bytes_peak);
  CACHE_FIELD(workspace_bytes);
  CACHE_FIELD(bytes_h2d);
  CACHE_FIELD(bytes_d2h);
  CACHE_FIELD(cache_hits);
  CACHE_FIELD(cache_misses);
  CACHE_FIELD(clean_evictions);
  CACHE_FIELD(dirty_evictions);
  CACHE_FIELD(writebacks_completed);
  CACHE_FIELD(mappings);
  CACHE_FIELD(unmaps);
  CACHE_FIELD(set_access_calls);
  CACHE_FIELD(handle_reuses);
  CACHE_FIELD(unsafe_remaps);
  CACHE_FIELD(unsafe_transitions);
  CACHE_FIELD(budget_shrinks);
  CACHE_FIELD(budget_grows);
  CACHE_FIELD(physical_handles_created);
  CACHE_FIELD(physical_handles_released);
  CACHE_FIELD(event_boundaries);
  CACHE_FIELD(pinned_staging_bytes);
  CACHE_FIELD(transactions_completed);
  CACHE_FIELD(algorithm_selections);
  CACHE_FIELD(algorithm_cache_hits);
  CACHE_FIELD(last_transaction_ms);
  CACHE_FIELD(budget_sample_count);
  CACHE_FIELD(cuda_free_bytes_minimum);
  CACHE_FIELD(cuda_free_bytes_end);
  CACHE_FIELD(wddm_available_bytes_minimum);
  CACHE_FIELD(wddm_available_bytes_end);
  CACHE_FIELD(wddm_budget_observed);
  CACHE_FIELD(watchdog_rejections);
  CACHE_FIELD(target_oom_retries);
#undef CACHE_FIELD
  j.end_object();
  std::uint64_t mismatches = 0, checked = 0;
  double maximum = 0;
  for (const auto& w : r.workloads) {
    mismatches += w.mismatches;
    checked += w.output_elements_checked;
    maximum = std::max(maximum, w.max_absolute_error);
  }
  j.key("verification");
  j.begin_object();
  field(j, "absolute_tolerance", 1e-4);
  field(j, "relative_tolerance", 2e-5);
  field(j, "mismatches", mismatches);
  field(j, "output_elements_checked", checked);
  field(j, "max_absolute_error", maximum);
  j.end_object();
  j.key("proof");
  j.begin_object();
  for (const auto& [key, value] : r.proof)
    field(j, key.c_str(), value);
  j.end_object();
  j.key("outcome");
  j.begin_object();
  field(j, "status", outcome(r.exit_code));
  field(j, "exit_code", r.exit_code);
  field(j, "reason", r.reason);
  field(j, "message", r.message);
  j.end_object();
  j.key("cleanup");
  j.begin_object();
  for (const auto& [key, value] : r.cleanup)
    field(j, key.c_str(), value);
  j.end_object();
  j.key("diagnostics");
  j.begin_array();
  for (const auto& diagnostic : r.diagnostics)
    j.value(diagnostic);
  j.end_array();
  j.end_object();
  return out.str();
}
std::string report_text(const Report& r) {
  std::ostringstream output;
  output << "xVRAM CUDA compatibility: " << outcome(r.exit_code) << " (exit " << r.exit_code
         << ")\n"
         << r.message << '\n';
  for (const auto& w : r.workloads)
    output << w.name << ": " << w.shape.m << 'x' << w.shape.n << 'x' << w.shape.k << ", "
           << w.logical_bytes << " operand bytes, " << w.tiles_retired
           << " retired tiles, mismatches " << w.mismatches << '\n';
  return output.str();
}
std::string trace_record(const std::uint64_t sequence, const std::uint64_t timestamp,
                         const std::string_view transition, const std::uint64_t operation,
                         const std::optional<std::uint64_t> allocation, const std::uint64_t bytes,
                         const std::uint64_t tiles, const std::string_view reason) {
  std::ostringstream out;
  JsonWriter j(out, false);
  j.begin_object();
  field(j, "schema_version", 1);
  field(j, "report_type", "xvram.cuda_compat_trace");
  field(j, "sequence", sequence);
  field(j, "monotonic_timestamp_ns", timestamp);
  field(j, "operation_id", operation);
  field(j, "allocation_id", allocation);
  field(j, "transition", transition);
  field(j, "bytes", bytes);
  field(j, "tiles_retired", tiles);
  field(j, "reason", reason);
  j.end_object();
  return out.str() + '\n';
}
} // namespace xvram::compat_bench
