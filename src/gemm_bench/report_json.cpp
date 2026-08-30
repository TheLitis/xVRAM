#include "xvram/gemm/report.hpp"

#include "xvram/base/json_writer.hpp"

#include <optional>
#include <ostream>
#include <string_view>
#include <type_traits>

namespace xvram::gemm_bench {
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
  write_optional(writer, include_identifiers ? device.pci_bus_id : std::optional<std::string>{});
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
  writer.key("compute_capability_major");
  write_optional(writer, device.compute_capability_major);
  writer.key("compute_capability_minor");
  write_optional(writer, device.compute_capability_minor);
  writer.end_object();
}

void write_configuration(JsonWriter& writer, const Configuration& configuration) {
  writer.begin_object();
  writer.key("requested_device_ordinal");
  write_optional(writer, configuration.requested_device_ordinal);
  writer.key("requested_m");
  write_optional(writer, configuration.requested_m);
  writer.key("requested_n");
  write_optional(writer, configuration.requested_n);
  writer.key("requested_k");
  write_optional(writer, configuration.requested_k);
  writer.key("scenario");
  writer.value(configuration.scenario);
  writer.key("a_data_type");
  writer.value(configuration.a_data_type);
  writer.key("b_data_type");
  writer.value(configuration.b_data_type);
  writer.key("c_data_type");
  writer.value(configuration.c_data_type);
  writer.key("compute_mode");
  writer.value(configuration.compute_mode);
  writer.key("a_layout");
  writer.value(configuration.a_layout);
  writer.key("b_layout");
  writer.value(configuration.b_layout);
  writer.key("c_layout");
  writer.value(configuration.c_layout);
  writer.key("a_operation");
  writer.value(configuration.a_operation);
  writer.key("b_operation");
  writer.value(configuration.b_operation);
  writer.key("alpha");
  writer.value(configuration.alpha);
  writer.key("beta");
  writer.value(configuration.beta);
  writer.key("requested_chunk_bytes");
  write_optional(writer, configuration.requested_chunk_bytes);
  writer.key("effective_chunk_bytes");
  write_optional(writer, configuration.effective_chunk_bytes);
  writer.key("requested_cache_target_bytes");
  write_optional(writer, configuration.requested_cache_target_bytes);
  writer.key("initial_cache_target_bytes");
  write_optional(writer, configuration.initial_cache_target_bytes);
  writer.key("requested_workspace_cap_bytes");
  write_optional(writer, configuration.requested_workspace_cap_bytes);
  writer.key("effective_workspace_cap_bytes");
  write_optional(writer, configuration.effective_workspace_cap_bytes);
  writer.key("context_mode");
  writer.value(configuration.context_mode);
  writer.key("policy");
  writer.value(configuration.policy);
  writer.key("staging_slots");
  writer.value(static_cast<std::uint64_t>(configuration.staging_slots));
  writer.key("prefetch_distance");
  writer.value(static_cast<std::uint64_t>(configuration.prefetch_distance));
  writer.key("passes");
  writer.value(static_cast<std::uint64_t>(configuration.passes));
  writer.key("budget_poll_ms");
  writer.value(configuration.budget_poll_ms);
  writer.key("stall_timeout_ms");
  writer.value(configuration.stall_timeout_ms);
  writer.key("timeout_ms");
  writer.value(configuration.timeout_ms);
  writer.key("max_tile_ms");
  writer.value(configuration.max_tile_ms);
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

void write_plan(JsonWriter& writer, const PlanResult& plan) {
  writer.begin_object();
  writer.key("plan_id");
  writer.value(plan.plan_id);
  writer.key("status");
  writer.value(plan.status);
  writer.key("m");
  writer.value(plan.m);
  writer.key("n");
  writer.value(plan.n);
  writer.key("k");
  writer.value(plan.k);
  writer.key("a_data_type");
  writer.value(plan.a_data_type);
  writer.key("b_data_type");
  writer.value(plan.b_data_type);
  writer.key("c_data_type");
  writer.value(plan.c_data_type);
  writer.key("effective_compute_mode");
  writer.value(plan.effective_compute_mode);
  writer.key("tile_m");
  writer.value(plan.tile_m);
  writer.key("tile_n");
  writer.value(plan.tile_n);
  writer.key("tile_k");
  writer.value(plan.tile_k);
  writer.key("tile_count");
  writer.value(plan.tile_count);
  writer.key("maximum_working_set_bytes");
  writer.value(plan.maximum_working_set_bytes);
  writer.key("workspace_bytes");
  writer.value(plan.workspace_bytes);
  writer.key("uses_cublas_lt");
  writer.value(plan.uses_cublas_lt);
  writer.key("heuristic_algorithm_count");
  write_optional(writer, plan.heuristic_algorithm_count);
  writer.key("replan_count");
  write_optional(writer, plan.replan_count);
  writer.end_object();
}

void write_workload(JsonWriter& writer, const WorkloadResult& workload) {
  writer.begin_object();
  writer.key("name");
  writer.value(workload.name);
  writer.key("plan_id");
  writer.value(workload.plan_id);
  writer.key("status");
  writer.value(workload.status);
  writer.key("passes_completed");
  writer.value(static_cast<std::uint64_t>(workload.passes_completed));
  writer.key("tiles_total");
  writer.value(workload.tiles_total);
  writer.key("tiles_retired");
  writer.value(workload.tiles_retired);
  writer.key("logical_allocation_bytes");
  writer.value(workload.logical_allocation_bytes);
  writer.key("oversubscribed");
  writer.value(workload.oversubscribed);
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
  writer.key("elapsed_ms");
  write_optional(writer, workload.elapsed_ms);
  writer.key("achieved_tflops");
  write_optional(writer, workload.achieved_tflops);
  writer.key("output_digest128");
  write_optional(writer, workload.output_digest128);
  writer.end_object();
}

void write_numerics(JsonWriter& writer, const Numerics& numerics) {
  writer.begin_object();
  writer.key("validation_mode");
  writer.value(numerics.validation_mode);
  writer.key("reference_precision");
  writer.value(numerics.reference_precision);
  writer.key("absolute_tolerance");
  write_optional(writer, numerics.absolute_tolerance);
  writer.key("relative_tolerance");
  write_optional(writer, numerics.relative_tolerance);
  writer.key("maximum_absolute_error");
  write_optional(writer, numerics.maximum_absolute_error);
  writer.key("maximum_relative_error");
  write_optional(writer, numerics.maximum_relative_error);
  writer.key("elements_verified");
  write_optional(writer, numerics.elements_verified);
  writer.key("mismatch_count");
  write_optional(writer, numerics.mismatch_count);
  writer.key("first_mismatch_byte_offset");
  write_optional(writer, numerics.first_mismatch_byte_offset);
  writer.key("expected_digest128");
  write_optional(writer, numerics.expected_digest128);
  writer.key("output_digest128");
  write_optional(writer, numerics.output_digest128);
  writer.key("all_results_match");
  write_optional(writer, numerics.all_results_match);
  writer.end_object();
}

void write_cache(JsonWriter& writer, const CacheStatistics& cache) {
  writer.begin_object();
#define XVRAM_WRITE_CACHE_OPTIONAL(name)                                                           \
  writer.key(#name);                                                                               \
  write_optional(writer, cache.name)
  XVRAM_WRITE_CACHE_OPTIONAL(target_bytes_initial);
  XVRAM_WRITE_CACHE_OPTIONAL(target_bytes_minimum);
  XVRAM_WRITE_CACHE_OPTIONAL(target_bytes_maximum);
  XVRAM_WRITE_CACHE_OPTIONAL(target_bytes_end);
  XVRAM_WRITE_CACHE_OPTIONAL(resident_bytes_peak);
  XVRAM_WRITE_CACHE_OPTIONAL(pinned_staging_bytes);
  XVRAM_WRITE_CACHE_OPTIONAL(workspace_bytes_peak);
  XVRAM_WRITE_CACHE_OPTIONAL(physical_frame_count_peak);
  XVRAM_WRITE_CACHE_OPTIONAL(physical_handle_create_count);
  XVRAM_WRITE_CACHE_OPTIONAL(physical_handle_release_count);
  XVRAM_WRITE_CACHE_OPTIONAL(handle_reuse_count);
  XVRAM_WRITE_CACHE_OPTIONAL(mapping_count);
  XVRAM_WRITE_CACHE_OPTIONAL(unmap_count);
  XVRAM_WRITE_CACHE_OPTIONAL(set_access_count);
  XVRAM_WRITE_CACHE_OPTIONAL(event_boundary_count);
  XVRAM_WRITE_CACHE_OPTIONAL(unsafe_remap_count);
  XVRAM_WRITE_CACHE_OPTIONAL(unsafe_transition_count);
  XVRAM_WRITE_CACHE_OPTIONAL(cache_hits);
  XVRAM_WRITE_CACHE_OPTIONAL(cache_misses);
  XVRAM_WRITE_CACHE_OPTIONAL(cache_hit_rate);
  XVRAM_WRITE_CACHE_OPTIONAL(clean_evictions);
  XVRAM_WRITE_CACHE_OPTIONAL(dirty_evictions);
  XVRAM_WRITE_CACHE_OPTIONAL(writebacks_completed);
  XVRAM_WRITE_CACHE_OPTIONAL(target_shrink_count);
  XVRAM_WRITE_CACHE_OPTIONAL(target_grow_count);
  XVRAM_WRITE_CACHE_OPTIONAL(target_oom_retry_count);
#undef XVRAM_WRITE_CACHE_OPTIONAL
  writer.end_object();
}

void write_telemetry(JsonWriter& writer, const Telemetry& telemetry) {
  writer.begin_object();
#define XVRAM_WRITE_TELEMETRY_OPTIONAL(name)                                                       \
  writer.key(#name);                                                                               \
  write_optional(writer, telemetry.name)
  XVRAM_WRITE_TELEMETRY_OPTIONAL(cuda_driver_version);
  XVRAM_WRITE_TELEMETRY_OPTIONAL(cublas_version);
  XVRAM_WRITE_TELEMETRY_OPTIONAL(cublas_lt_version);
  XVRAM_WRITE_TELEMETRY_OPTIONAL(cublas_lt_available);
  XVRAM_WRITE_TELEMETRY_OPTIONAL(cublas_library_source);
  XVRAM_WRITE_TELEMETRY_OPTIONAL(algorithms_selected);
  XVRAM_WRITE_TELEMETRY_OPTIONAL(algorithm_cache_hits);
  XVRAM_WRITE_TELEMETRY_OPTIONAL(budget_sample_count);
  XVRAM_WRITE_TELEMETRY_OPTIONAL(cuda_free_bytes_minimum);
  XVRAM_WRITE_TELEMETRY_OPTIONAL(cuda_free_bytes_end);
  XVRAM_WRITE_TELEMETRY_OPTIONAL(wddm_available_bytes_minimum);
  XVRAM_WRITE_TELEMETRY_OPTIONAL(wddm_available_bytes_end);
  XVRAM_WRITE_TELEMETRY_OPTIONAL(replans);
  XVRAM_WRITE_TELEMETRY_OPTIONAL(watchdog_rejections);
  XVRAM_WRITE_TELEMETRY_OPTIONAL(total_elapsed_ms);
#undef XVRAM_WRITE_TELEMETRY_OPTIONAL
  writer.key("plan_timing");
  write_timing(writer, telemetry.plan_timing);
  writer.key("tile_timing");
  write_timing(writer, telemetry.tile_timing);
  writer.key("gemm_timing");
  write_timing(writer, telemetry.gemm_timing);
  writer.key("h2d_timing");
  write_timing(writer, telemetry.h2d_timing);
  writer.key("d2h_timing");
  write_timing(writer, telemetry.d2h_timing);
  writer.key("writeback_timing");
  write_timing(writer, telemetry.writeback_timing);
  writer.end_object();
}

void write_proof(JsonWriter& writer, const Proof& proof) {
  writer.begin_object();
#define XVRAM_WRITE_PROOF_OPTIONAL(name)                                                           \
  writer.key(#name);                                                                               \
  write_optional(writer, proof.name)
  XVRAM_WRITE_PROOF_OPTIONAL(logical_allocation_bytes);
  XVRAM_WRITE_PROOF_OPTIONAL(total_vram_bytes);
  XVRAM_WRITE_PROOF_OPTIONAL(maximum_cache_target_bytes);
  XVRAM_WRITE_PROOF_OPTIONAL(maximum_working_set_bytes);
  XVRAM_WRITE_PROOF_OPTIONAL(workspace_cap_bytes);
  XVRAM_WRITE_PROOF_OPTIONAL(logical_data_exceeds_vram);
  XVRAM_WRITE_PROOF_OPTIONAL(cache_smaller_than_logical);
  XVRAM_WRITE_PROOF_OPTIONAL(tiled_execution_verified);
  XVRAM_WRITE_PROOF_OPTIONAL(handles_reused);
  XVRAM_WRITE_PROOF_OPTIONAL(stable_virtual_addresses_verified);
  XVRAM_WRITE_PROOF_OPTIONAL(set_access_after_map_verified);
  XVRAM_WRITE_PROOF_OPTIONAL(event_boundaries_verified);
  XVRAM_WRITE_PROOF_OPTIONAL(no_physical_aliases_verified);
  XVRAM_WRITE_PROOF_OPTIONAL(dirty_writeback_verified);
  XVRAM_WRITE_PROOF_OPTIONAL(cache_target_respected);
  XVRAM_WRITE_PROOF_OPTIONAL(workspace_bounded);
  XVRAM_WRITE_PROOF_OPTIONAL(all_workloads_match_reference);
  XVRAM_WRITE_PROOF_OPTIONAL(raw_virtual_addresses_omitted);
#undef XVRAM_WRITE_PROOF_OPTIONAL
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
  writer.key("plan_id");
  write_optional(writer, outcome.plan_id);
  writer.key("tile_sequence");
  write_optional(writer, outcome.tile_sequence);
  writer.key("logical_byte_offset");
  write_optional(writer, outcome.logical_byte_offset);
  writer.end_object();
}

void write_cleanup(JsonWriter& writer, const Cleanup& cleanup) {
  writer.begin_object();
#define XVRAM_WRITE_CLEANUP_OPTIONAL(name)                                                         \
  writer.key(#name);                                                                               \
  write_optional(writer, cleanup.name)
  XVRAM_WRITE_CLEANUP_OPTIONAL(complete);
  XVRAM_WRITE_CLEANUP_OPTIONAL(operations_drained);
  XVRAM_WRITE_CLEANUP_OPTIONAL(events_drained);
  XVRAM_WRITE_CLEANUP_OPTIONAL(events_destroyed);
  XVRAM_WRITE_CLEANUP_OPTIONAL(streams_destroyed);
  XVRAM_WRITE_CLEANUP_OPTIONAL(cublas_handles_destroyed);
  XVRAM_WRITE_CLEANUP_OPTIONAL(mappings_removed);
  XVRAM_WRITE_CLEANUP_OPTIONAL(physical_handles_released);
  XVRAM_WRITE_CLEANUP_OPTIONAL(workspace_released);
  XVRAM_WRITE_CLEANUP_OPTIONAL(virtual_reservations_released);
  XVRAM_WRITE_CLEANUP_OPTIONAL(pinned_staging_released);
  XVRAM_WRITE_CLEANUP_OPTIONAL(host_backing_released);
  XVRAM_WRITE_CLEANUP_OPTIONAL(context_released);
  XVRAM_WRITE_CLEANUP_OPTIONAL(worker_terminated);
#undef XVRAM_WRITE_CLEANUP_OPTIONAL
  writer.end_object();
}

void write_diagnostic(JsonWriter& writer, const probe::Diagnostic& diagnostic) {
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
  writer.key("plans");
  writer.begin_array();
  for (const PlanResult& plan : report.plans) {
    write_plan(writer, plan);
  }
  writer.end_array();
  writer.key("workloads");
  writer.begin_array();
  for (const WorkloadResult& workload : report.workloads) {
    write_workload(writer, workload);
  }
  writer.end_array();
  writer.key("numerics");
  write_numerics(writer, report.numerics);
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
  writer.begin_array();
  for (const probe::Diagnostic& diagnostic : report.diagnostics) {
    write_diagnostic(writer, diagnostic);
  }
  writer.end_array();
  writer.end_object();
}

} // namespace xvram::gemm_bench
