#include "xvram/residency/report.hpp"

#include "xvram/base/json_writer.hpp"

#include <optional>
#include <ostream>
#include <string_view>
#include <type_traits>

namespace xvram::residency {
namespace {

template <typename T> void write_optional(JsonWriter& writer, const std::optional<T>& value) {
  if (!value.has_value()) {
    writer.null_value();
  } else if constexpr (std::is_same_v<T, std::string>) {
    writer.value(*value);
  } else if constexpr (std::is_same_v<T, bool>) {
    writer.value(*value);
  } else if constexpr (std::is_floating_point_v<T>) {
    writer.value(static_cast<double>(*value));
  } else if constexpr (std::is_signed_v<T>) {
    writer.value(static_cast<std::int64_t>(*value));
  } else {
    writer.value(static_cast<std::uint64_t>(*value));
  }
}

[[nodiscard]] std::string_view diagnostic_level_name(const probe::DiagnosticLevel level) {
  switch (level) {
  case probe::DiagnosticLevel::info:
    return "info";
  case probe::DiagnosticLevel::warning:
    return "warning";
  case probe::DiagnosticLevel::error:
    return "error";
  }
  return "unknown";
}

void write_build(JsonWriter& writer, const probe::BuildInfo& build) {
  writer.begin_object();
  writer.key("version");
  writer.value(build.version);
  writer.key("git_commit");
  writer.value(build.git_commit);
  writer.key("compiler");
  writer.value(build.compiler);
  writer.key("build_type");
  writer.value(build.build_type);
  writer.key("cuda_headers_version");
  writer.value(static_cast<std::int64_t>(build.cuda_headers_version));
  writer.end_object();
}

void write_system(JsonWriter& writer, const probe::SystemInfo& system) {
  writer.begin_object();
  writer.key("os_name");
  writer.value(system.os_name);
  writer.key("os_version");
  writer.value(system.os_version);
  writer.key("architecture");
  writer.value(system.architecture);
  writer.key("logical_processor_count");
  writer.value(static_cast<std::uint64_t>(system.logical_processor_count));
  writer.key("physical_memory_bytes");
  write_optional(writer, system.physical_memory_bytes);
  writer.key("available_memory_bytes");
  write_optional(writer, system.available_memory_bytes);
  writer.end_object();
}

void write_device(JsonWriter& writer, const DeviceInfo& device, const bool include_identifiers) {
  writer.begin_object();
  writer.key("ordinal");
  writer.value(static_cast<std::int64_t>(device.ordinal));
  writer.key("name");
  writer.value(device.name);
  writer.key("uuid");
  write_optional(writer, include_identifiers ? device.uuid : std::optional<std::string>{});
  writer.key("luid");
  write_optional(writer, include_identifiers ? device.luid : std::optional<std::string>{});
  writer.key("pci_bus_id");
  write_optional(writer,
                 include_identifiers ? device.pci_bus_id : std::optional<std::string>{});
  writer.key("driver_model");
  write_optional(writer, device.driver_model);
  writer.key("total_memory_bytes");
  writer.value(device.total_memory_bytes);
  writer.key("free_memory_bytes_start");
  write_optional(writer, device.free_memory_bytes_start);
  writer.key("free_memory_bytes_end");
  write_optional(writer, device.free_memory_bytes_end);
  writer.key("safe_device_budget_bytes");
  write_optional(writer, device.safe_device_budget_bytes);
  writer.key("wddm_budget_bytes_start");
  write_optional(writer, device.wddm_budget_bytes_start);
  writer.key("wddm_usage_bytes_start");
  write_optional(writer, device.wddm_usage_bytes_start);
  writer.key("wddm_available_bytes_start");
  write_optional(writer, device.wddm_available_bytes_start);
  writer.key("wddm_available_bytes_minimum");
  write_optional(writer, device.wddm_available_bytes_minimum);
  writer.key("wddm_available_bytes_end");
  write_optional(writer, device.wddm_available_bytes_end);
  writer.key("vmm_supported");
  writer.value(device.vmm_supported);
  writer.key("uva_supported");
  writer.value(device.uva_supported);
  writer.key("minimum_granularity_bytes");
  write_optional(writer, device.minimum_granularity_bytes);
  writer.key("recommended_granularity_bytes");
  write_optional(writer, device.recommended_granularity_bytes);
  writer.end_object();
}

void write_configuration(JsonWriter& writer, const Configuration& configuration) {
  writer.begin_object();
  writer.key("requested_device_ordinal");
  write_optional(writer, configuration.requested_device_ordinal);
  writer.key("requested_logical_bytes");
  write_optional(writer, configuration.requested_logical_bytes);
  writer.key("effective_logical_bytes");
  write_optional(writer, configuration.effective_logical_bytes);
  writer.key("requested_chunk_bytes");
  write_optional(writer, configuration.requested_chunk_bytes);
  writer.key("effective_chunk_bytes");
  write_optional(writer, configuration.effective_chunk_bytes);
  writer.key("requested_cache_target_bytes");
  write_optional(writer, configuration.requested_cache_target_bytes);
  writer.key("initial_cache_target_bytes");
  write_optional(writer, configuration.initial_cache_target_bytes);
  writer.key("staging_slots");
  writer.value(static_cast<std::uint64_t>(configuration.staging_slots));
  writer.key("policy");
  writer.value(configuration.policy);
  writer.key("prefetch_distance");
  writer.value(static_cast<std::uint64_t>(configuration.prefetch_distance));
  writer.key("scenario");
  writer.value(configuration.scenario);
  writer.key("passes");
  writer.value(static_cast<std::uint64_t>(configuration.passes));
  writer.key("pressure_bytes");
  write_optional(writer, configuration.pressure_bytes);
  writer.key("host_headroom_bytes");
  writer.value(configuration.host_headroom_bytes);
  writer.key("device_headroom_bytes");
  writer.value(configuration.device_headroom_bytes);
  writer.key("budget_poll_ms");
  writer.value(configuration.budget_poll_ms);
  writer.key("stall_timeout_ms");
  writer.value(configuration.stall_timeout_ms);
  writer.key("timeout_ms");
  writer.value(configuration.timeout_ms);
  writer.key("seed_hex");
  writer.value(configuration.seed_hex);
  writer.key("sizing_mode");
  writer.value(configuration.sizing_mode);
  writer.key("trace_enabled");
  writer.value(configuration.trace_enabled);
  writer.key("identifiers_included");
  writer.value(configuration.identifiers_included);
  writer.end_object();
}

void write_timing(JsonWriter& writer, const TimingSummary& timing) {
  writer.begin_object();
  writer.key("sample_count");
  writer.value(timing.sample_count);
  writer.key("total_ms");
  write_optional(writer, timing.total_ms);
  writer.key("minimum_ms");
  write_optional(writer, timing.minimum_ms);
  writer.key("median_ms");
  write_optional(writer, timing.median_ms);
  writer.key("p95_ms");
  write_optional(writer, timing.p95_ms);
  writer.key("maximum_ms");
  write_optional(writer, timing.maximum_ms);
  writer.end_object();
}

void write_workload(JsonWriter& writer, const WorkloadResult& workload) {
  writer.begin_object();
  writer.key("scenario");
  writer.value(workload.scenario);
  writer.key("policy");
  writer.value(workload.policy);
  writer.key("status");
  writer.value(workload.status);
  writer.key("operations_retired");
  write_optional(writer, workload.operations_retired);
  writer.key("passes_completed");
  write_optional(writer, workload.passes_completed);
  writer.key("logical_bytes");
  write_optional(writer, workload.logical_bytes);
  writer.key("logical_chunk_count");
  write_optional(writer, workload.logical_chunk_count);
  writer.key("maximum_working_set_bytes");
  write_optional(writer, workload.maximum_working_set_bytes);
  writer.key("read_operations");
  write_optional(writer, workload.read_operations);
  writer.key("read_write_operations");
  write_optional(writer, workload.read_write_operations);
  writer.key("write_only_operations");
  write_optional(writer, workload.write_only_operations);
  writer.key("cache_hits");
  write_optional(writer, workload.cache_hits);
  writer.key("cache_misses");
  write_optional(writer, workload.cache_misses);
  writer.key("cache_hit_rate");
  write_optional(writer, workload.cache_hit_rate);
  writer.key("bytes_h2d");
  write_optional(writer, workload.bytes_h2d);
  writer.key("bytes_d2h");
  write_optional(writer, workload.bytes_d2h);
  writer.key("clean_evictions");
  write_optional(writer, workload.clean_evictions);
  writer.key("dirty_evictions");
  write_optional(writer, workload.dirty_evictions);
  writer.key("writebacks_completed");
  write_optional(writer, workload.writebacks_completed);
  writer.key("prefetch_issued");
  write_optional(writer, workload.prefetch_issued);
  writer.key("prefetch_useful");
  write_optional(writer, workload.prefetch_useful);
  writer.key("prefetch_wasted");
  write_optional(writer, workload.prefetch_wasted);
  writer.key("prefetch_cancelled");
  write_optional(writer, workload.prefetch_cancelled);
  writer.key("prefetch_promoted");
  write_optional(writer, workload.prefetch_promoted);
  writer.key("sequential_bypasses");
  write_optional(writer, workload.sequential_bypasses);
  writer.key("target_shrink_count");
  write_optional(writer, workload.target_shrink_count);
  writer.key("target_grow_count");
  write_optional(writer, workload.target_grow_count);
  writer.key("mapping_count");
  write_optional(writer, workload.mapping_count);
  writer.key("unmap_count");
  write_optional(writer, workload.unmap_count);
  writer.key("set_access_count");
  write_optional(writer, workload.set_access_count);
  writer.key("handle_reuse_count");
  write_optional(writer, workload.handle_reuse_count);
  writer.key("event_boundary_count");
  write_optional(writer, workload.event_boundary_count);
  writer.key("unsafe_remap_count");
  write_optional(writer, workload.unsafe_remap_count);
  writer.key("unsafe_transition_count");
  write_optional(writer, workload.unsafe_transition_count);
  writer.key("stable_addresses_verified");
  write_optional(writer, workload.stable_addresses_verified);
  writer.key("full_verification_completed");
  write_optional(writer, workload.full_verification_completed);
  writer.key("matches_cpu");
  write_optional(writer, workload.matches_cpu);
  writer.key("elapsed_ms");
  write_optional(writer, workload.elapsed_ms);
  writer.key("throughput_gib_per_second");
  write_optional(writer, workload.throughput_gib_per_second);
  writer.key("expected_digest128");
  write_optional(writer, workload.expected_digest128);
  writer.key("output_digest128");
  write_optional(writer, workload.output_digest128);
  writer.key("mismatch_count");
  write_optional(writer, workload.mismatch_count);
  writer.key("first_mismatch_byte_offset");
  write_optional(writer, workload.first_mismatch_byte_offset);
  writer.end_object();
}

void write_cache(JsonWriter& writer, const CacheStatistics& cache) {
  writer.begin_object();
  writer.key("target_bytes_initial");
  write_optional(writer, cache.target_bytes_initial);
  writer.key("target_bytes_minimum");
  write_optional(writer, cache.target_bytes_minimum);
  writer.key("target_bytes_maximum");
  write_optional(writer, cache.target_bytes_maximum);
  writer.key("target_bytes_end");
  write_optional(writer, cache.target_bytes_end);
  writer.key("resident_bytes_peak");
  write_optional(writer, cache.resident_bytes_peak);
  writer.key("pinned_staging_bytes");
  write_optional(writer, cache.pinned_staging_bytes);
  writer.key("physical_frame_count_peak");
  write_optional(writer, cache.physical_frame_count_peak);
  writer.key("physical_handle_create_count");
  write_optional(writer, cache.physical_handle_create_count);
  writer.key("physical_handle_release_count");
  write_optional(writer, cache.physical_handle_release_count);
  writer.key("handle_reuse_count");
  write_optional(writer, cache.handle_reuse_count);
  writer.key("mapping_count");
  write_optional(writer, cache.mapping_count);
  writer.key("unmap_count");
  write_optional(writer, cache.unmap_count);
  writer.key("set_access_count");
  write_optional(writer, cache.set_access_count);
  writer.key("event_boundary_count");
  write_optional(writer, cache.event_boundary_count);
  writer.key("unsafe_remap_count");
  write_optional(writer, cache.unsafe_remap_count);
  writer.key("unsafe_transition_count");
  write_optional(writer, cache.unsafe_transition_count);
  writer.key("cache_hits");
  write_optional(writer, cache.cache_hits);
  writer.key("cache_misses");
  write_optional(writer, cache.cache_misses);
  writer.key("cache_hit_rate");
  write_optional(writer, cache.cache_hit_rate);
  writer.key("clean_evictions");
  write_optional(writer, cache.clean_evictions);
  writer.key("dirty_evictions");
  write_optional(writer, cache.dirty_evictions);
  writer.key("writebacks_completed");
  write_optional(writer, cache.writebacks_completed);
  writer.key("prefetch_issued");
  write_optional(writer, cache.prefetch_issued);
  writer.key("prefetch_useful");
  write_optional(writer, cache.prefetch_useful);
  writer.key("prefetch_wasted");
  write_optional(writer, cache.prefetch_wasted);
  writer.key("prefetch_cancelled");
  write_optional(writer, cache.prefetch_cancelled);
  writer.key("prefetch_promoted");
  write_optional(writer, cache.prefetch_promoted);
  writer.key("sequential_bypasses");
  write_optional(writer, cache.sequential_bypasses);
  writer.key("target_shrink_count");
  write_optional(writer, cache.target_shrink_count);
  writer.key("target_grow_count");
  write_optional(writer, cache.target_grow_count);
  writer.key("target_oom_retry_count");
  write_optional(writer, cache.target_oom_retry_count);
  writer.key("maximum_working_set_bytes");
  write_optional(writer, cache.maximum_working_set_bytes);
  writer.end_object();
}

void write_telemetry(JsonWriter& writer, const Telemetry& telemetry) {
  writer.begin_object();
  writer.key("total_elapsed_ms");
  write_optional(writer, telemetry.total_elapsed_ms);
  writer.key("bytes_h2d");
  write_optional(writer, telemetry.bytes_h2d);
  writer.key("bytes_d2h");
  write_optional(writer, telemetry.bytes_d2h);
  writer.key("budget_sample_count");
  write_optional(writer, telemetry.budget_sample_count);
  writer.key("cuda_free_bytes_minimum");
  write_optional(writer, telemetry.cuda_free_bytes_minimum);
  writer.key("cuda_free_bytes_end");
  write_optional(writer, telemetry.cuda_free_bytes_end);
  writer.key("wddm_available_bytes_minimum");
  write_optional(writer, telemetry.wddm_available_bytes_minimum);
  writer.key("wddm_available_bytes_end");
  write_optional(writer, telemetry.wddm_available_bytes_end);
  writer.key("resident_occupancy_peak");
  write_optional(writer, telemetry.resident_occupancy_peak);
  writer.key("resident_occupancy_mean");
  write_optional(writer, telemetry.resident_occupancy_mean);
  writer.key("remap_timing");
  write_timing(writer, telemetry.remap_timing);
  writer.key("h2d_timing");
  write_timing(writer, telemetry.h2d_timing);
  writer.key("kernel_timing");
  write_timing(writer, telemetry.kernel_timing);
  writer.key("d2h_timing");
  write_timing(writer, telemetry.d2h_timing);
  writer.key("writeback_timing");
  write_timing(writer, telemetry.writeback_timing);
  writer.key("trace_records_emitted");
  write_optional(writer, telemetry.trace_records_emitted);
  writer.key("trace_records_dropped");
  write_optional(writer, telemetry.trace_records_dropped);
  writer.key("trace_complete");
  write_optional(writer, telemetry.trace_complete);
  writer.end_object();
}

void write_proof(JsonWriter& writer, const Proof& proof) {
  writer.begin_object();
  writer.key("logical_bytes");
  write_optional(writer, proof.logical_bytes);
  writer.key("chunk_bytes");
  write_optional(writer, proof.chunk_bytes);
  writer.key("logical_chunk_count");
  write_optional(writer, proof.logical_chunk_count);
  writer.key("maximum_cache_target_bytes");
  write_optional(writer, proof.maximum_cache_target_bytes);
  writer.key("pinned_staging_bytes");
  write_optional(writer, proof.pinned_staging_bytes);
  writer.key("kernel_module_version");
  writer.value(static_cast<std::uint64_t>(proof.kernel_module_version));
  writer.key("kernel_module_sha256");
  writer.value(proof.kernel_module_sha256);
  writer.key("pattern_version");
  writer.value(proof.pattern_version);
  writer.key("cpu_reference_digest128");
  write_optional(writer, proof.cpu_reference_digest128);
  writer.key("logical_data_exceeds_vram");
  write_optional(writer, proof.logical_data_exceeds_vram);
  writer.key("cache_smaller_than_logical");
  write_optional(writer, proof.cache_smaller_than_logical);
  writer.key("handles_reused");
  write_optional(writer, proof.handles_reused);
  writer.key("stable_virtual_addresses_verified");
  write_optional(writer, proof.stable_virtual_addresses_verified);
  writer.key("set_access_after_map_verified");
  write_optional(writer, proof.set_access_after_map_verified);
  writer.key("event_boundaries_verified");
  write_optional(writer, proof.event_boundaries_verified);
  writer.key("no_physical_aliases_verified");
  write_optional(writer, proof.no_physical_aliases_verified);
  writer.key("dirty_writeback_verified");
  write_optional(writer, proof.dirty_writeback_verified);
  writer.key("staging_pool_bounded");
  write_optional(writer, proof.staging_pool_bounded);
  writer.key("cache_target_respected");
  write_optional(writer, proof.cache_target_respected);
  writer.key("all_workloads_match_cpu");
  write_optional(writer, proof.all_workloads_match_cpu);
  writer.key("policies_match");
  write_optional(writer, proof.policies_match);
  writer.key("raw_virtual_addresses_omitted");
  write_optional(writer, proof.raw_virtual_addresses_omitted);
  writer.end_object();
}

void write_outcome(JsonWriter& writer, const Outcome& outcome) {
  writer.begin_object();
  writer.key("status");
  writer.value(outcome.status);
  writer.key("reason");
  write_optional(writer, outcome.reason);
  writer.key("exit_code");
  writer.value(static_cast<std::int64_t>(outcome.exit_code));
  writer.key("stage");
  write_optional(writer, outcome.stage);
  writer.key("operation");
  write_optional(writer, outcome.operation);
  writer.key("native_code");
  write_optional(writer, outcome.native_code);
  writer.key("native_name");
  write_optional(writer, outcome.native_name);
  writer.key("message");
  write_optional(writer, outcome.message);
  writer.key("policy");
  write_optional(writer, outcome.policy);
  writer.key("scenario");
  write_optional(writer, outcome.scenario);
  writer.key("allocation_id");
  write_optional(writer, outcome.allocation_id);
  writer.key("chunk_index");
  write_optional(writer, outcome.chunk_index);
  writer.key("logical_byte_offset");
  write_optional(writer, outcome.logical_byte_offset);
  writer.end_object();
}

void write_cleanup(JsonWriter& writer, const Cleanup& cleanup) {
  writer.begin_object();
  writer.key("complete");
  write_optional(writer, cleanup.complete);
  writer.key("transactions_drained");
  write_optional(writer, cleanup.transactions_drained);
  writer.key("prefetch_drained");
  write_optional(writer, cleanup.prefetch_drained);
  writer.key("writebacks_completed");
  write_optional(writer, cleanup.writebacks_completed);
  writer.key("events_drained");
  write_optional(writer, cleanup.events_drained);
  writer.key("events_destroyed");
  write_optional(writer, cleanup.events_destroyed);
  writer.key("streams_destroyed");
  write_optional(writer, cleanup.streams_destroyed);
  writer.key("module_unloaded");
  write_optional(writer, cleanup.module_unloaded);
  writer.key("mappings_removed");
  write_optional(writer, cleanup.mappings_removed);
  writer.key("physical_handles_released");
  write_optional(writer, cleanup.physical_handles_released);
  writer.key("virtual_reservations_released");
  write_optional(writer, cleanup.virtual_reservations_released);
  writer.key("pinned_staging_released");
  write_optional(writer, cleanup.pinned_staging_released);
  writer.key("host_backing_released");
  write_optional(writer, cleanup.host_backing_released);
  writer.key("context_destroyed");
  write_optional(writer, cleanup.context_destroyed);
  writer.key("trace_closed");
  write_optional(writer, cleanup.trace_closed);
  writer.key("worker_terminated");
  write_optional(writer, cleanup.worker_terminated);
  writer.end_object();
}

void write_diagnostics(JsonWriter& writer, const std::vector<probe::Diagnostic>& diagnostics) {
  writer.begin_array();
  for (const probe::Diagnostic& diagnostic : diagnostics) {
    writer.begin_object();
    writer.key("level");
    writer.value(diagnostic_level_name(diagnostic.level));
    writer.key("component");
    writer.value(diagnostic.component);
    writer.key("operation");
    writer.value(diagnostic.operation);
    writer.key("message");
    writer.value(diagnostic.message);
    writer.key("code");
    write_optional(writer, diagnostic.code);
    writer.key("device_ordinal");
    write_optional(writer, diagnostic.device_ordinal);
    writer.end_object();
  }
  writer.end_array();
}

} // namespace

void write_json(const Report& report, std::ostream& output, const bool pretty) {
  JsonWriter writer(output, pretty);
  writer.begin_object();
  writer.key("schema_version");
  writer.value(static_cast<std::uint64_t>(report.schema_version));
  writer.key("report_type");
  writer.value(report.report_type);
  writer.key("generated_at_utc");
  writer.value(report.generated_at_utc);
  writer.key("build");
  write_build(writer, report.build);
  writer.key("system");
  write_system(writer, report.system);
  writer.key("device");
  if (report.device.has_value()) {
    write_device(writer, *report.device, report.configuration.identifiers_included);
  } else {
    writer.null_value();
  }
  writer.key("configuration");
  write_configuration(writer, report.configuration);
  writer.key("workloads");
  writer.begin_array();
  for (const WorkloadResult& workload : report.workloads) {
    write_workload(writer, workload);
  }
  writer.end_array();
  writer.key("cache");
  write_cache(writer, report.cache);
  writer.key("telemetry");
  write_telemetry(writer, report.telemetry);
  writer.key("proof");
  write_proof(writer, report.proof);
  writer.key("outcome");
  write_outcome(writer, report.outcome);
  writer.key("cleanup");
  write_cleanup(writer, report.cleanup);
  writer.key("diagnostics");
  write_diagnostics(writer, report.diagnostics);
  writer.end_object();
  output << '\n';
}

void write_trace_json(const TraceRecord& record, std::ostream& output) {
  JsonWriter writer(output, false);
  writer.begin_object();
  writer.key("schema_version");
  writer.value(static_cast<std::uint64_t>(record.schema_version));
  writer.key("report_type");
  writer.value(record.report_type);
  writer.key("sequence");
  writer.value(record.sequence);
  writer.key("monotonic_time_ns");
  writer.value(record.monotonic_time_ns);
  writer.key("event");
  writer.value(record.event);
  writer.key("allocation_id");
  write_optional(writer, record.allocation_id);
  writer.key("chunk_index");
  write_optional(writer, record.chunk_index);
  writer.key("operation_id");
  write_optional(writer, record.operation_id);
  writer.key("transaction_id");
  write_optional(writer, record.transaction_id);
  writer.key("from_state");
  write_optional(writer, record.from_state);
  writer.key("to_state");
  write_optional(writer, record.to_state);
  writer.key("bytes");
  write_optional(writer, record.bytes);
  writer.key("reason");
  write_optional(writer, record.reason);
  writer.key("policy");
  write_optional(writer, record.policy);
  writer.key("speculative");
  write_optional(writer, record.speculative);
  writer.key("generation");
  write_optional(writer, record.generation);
  writer.end_object();
  output << '\n';
}

} // namespace xvram::residency
