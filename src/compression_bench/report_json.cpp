#include "xvram/compression/report.hpp"

#include "xvram/base/json_writer.hpp"

#include <optional>
#include <ostream>
#include <string_view>
#include <type_traits>

namespace xvram::compression {
namespace {

[[nodiscard]] std::string_view safe_text(const std::string_view value) noexcept {
  return contains_forbidden_runtime_identity(value)
             ? std::string_view{"[redacted runtime identity]"}
             : value;
}

template <typename T> void write_optional(JsonWriter& writer, const std::optional<T>& value) {
  if (!value.has_value()) {
    writer.null_value();
  } else if constexpr (std::is_same_v<T, std::string>) {
    writer.value(safe_text(*value));
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

template <typename T>
void optional_field(JsonWriter& writer, const std::string_view name,
                    const std::optional<T>& value) {
  writer.key(name);
  write_optional(writer, value);
}

template <typename T>
void unsigned_field(JsonWriter& writer, const std::string_view name, const T value) {
  writer.key(name);
  writer.value(static_cast<std::uint64_t>(value));
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
  unsigned_field(writer, "logical_processor_count", system.logical_processor_count);
  optional_field(writer, "physical_memory_bytes", system.physical_memory_bytes);
  optional_field(writer, "available_memory_bytes", system.available_memory_bytes);
  writer.end_object();
}

void write_device(JsonWriter& writer, const DeviceInfo& device, const bool include_identifiers) {
  writer.begin_object();
  writer.key("ordinal");
  writer.value(static_cast<std::int64_t>(device.ordinal));
  writer.key("name");
  writer.value(device.name);
  optional_field(writer, "uuid", include_identifiers ? device.uuid : std::optional<std::string>{});
  optional_field(writer, "luid", include_identifiers ? device.luid : std::optional<std::string>{});
  optional_field(writer, "pci_bus_id",
                 include_identifiers ? device.pci_bus_id : std::optional<std::string>{});
  optional_field(writer, "driver_model", device.driver_model);
  unsigned_field(writer, "total_memory_bytes", device.total_memory_bytes);
  optional_field(writer, "free_memory_bytes_start", device.free_memory_bytes_start);
  optional_field(writer, "free_memory_bytes_end", device.free_memory_bytes_end);
  optional_field(writer, "safe_device_budget_bytes", device.safe_device_budget_bytes);
  optional_field(writer, "wddm_budget_bytes_start", device.wddm_budget_bytes_start);
  optional_field(writer, "wddm_usage_bytes_start", device.wddm_usage_bytes_start);
  optional_field(writer, "wddm_available_bytes_minimum", device.wddm_available_bytes_minimum);
  optional_field(writer, "compute_capability_major", device.compute_capability_major);
  optional_field(writer, "compute_capability_minor", device.compute_capability_minor);
  writer.key("vmm_supported");
  writer.value(device.vmm_supported);
  writer.key("uva_supported");
  writer.value(device.uva_supported);
  writer.end_object();
}

void write_configuration(JsonWriter& writer, const Configuration& value) {
  writer.begin_object();
  optional_field(writer, "requested_device_ordinal", value.requested_device_ordinal);
  optional_field(writer, "requested_logical_bytes", value.requested_logical_bytes);
  optional_field(writer, "effective_logical_bytes", value.effective_logical_bytes);
  optional_field(writer, "requested_chunk_bytes", value.requested_chunk_bytes);
  optional_field(writer, "effective_chunk_bytes", value.effective_chunk_bytes);
  optional_field(writer, "requested_cache_target_bytes", value.requested_cache_target_bytes);
  optional_field(writer, "initial_cache_target_bytes", value.initial_cache_target_bytes);
  writer.key("compression_policy");
  writer.value(value.compression_policy);
  writer.key("path");
  writer.value(value.path);
  writer.key("codec");
  writer.value(value.codec);
  writer.key("replacement_policy");
  writer.value(value.replacement_policy);
  writer.key("scenario");
  writer.value(value.scenario);
  unsigned_field(writer, "passes", value.passes);
  unsigned_field(writer, "warmup_passes", value.warmup_passes);
  unsigned_field(writer, "measurement_passes", value.measurement_passes);
  optional_field(writer, "host_store_cap_bytes", value.host_store_cap_bytes);
  unsigned_field(writer, "host_headroom_bytes", value.host_headroom_bytes);
  unsigned_field(writer, "device_headroom_bytes", value.device_headroom_bytes);
  unsigned_field(writer, "compression_scratch_cap_bytes", value.compression_scratch_cap_bytes);
  unsigned_field(writer, "codec_slots", value.codec_slots);
  unsigned_field(writer, "codec_workers", value.codec_workers);
  unsigned_field(writer, "staging_slots", value.staging_slots);
  unsigned_field(writer, "prefetch_distance", value.prefetch_distance);
  unsigned_field(writer, "budget_poll_ms", value.budget_poll_ms);
  unsigned_field(writer, "stall_timeout_ms", value.stall_timeout_ms);
  unsigned_field(writer, "timeout_ms", value.timeout_ms);
  writer.key("seed_hex");
  writer.value(value.seed_hex);
  writer.key("sizing_mode");
  writer.value(value.sizing_mode);
  writer.key("trace_enabled");
  writer.value(value.trace_enabled);
  writer.key("identifiers_included");
  writer.value(value.identifiers_included);
  writer.end_object();
}

void write_timing(JsonWriter& writer, const TimingSummary& value) {
  writer.begin_object();
  unsigned_field(writer, "sample_count", value.sample_count);
  optional_field(writer, "total_ms", value.total_ms);
  optional_field(writer, "minimum_ms", value.minimum_ms);
  optional_field(writer, "median_ms", value.median_ms);
  optional_field(writer, "p95_ms", value.p95_ms);
  optional_field(writer, "maximum_ms", value.maximum_ms);
  writer.end_object();
}

void write_workload(JsonWriter& writer, const WorkloadResult& value) {
  writer.begin_object();
  writer.key("scenario");
  writer.value(value.scenario);
  writer.key("compression_policy");
  writer.value(value.compression_policy);
  writer.key("path");
  writer.value(value.path);
  writer.key("replacement_policy");
  writer.value(value.replacement_policy);
  writer.key("status");
  writer.value(value.status);
  optional_field(writer, "logical_bytes", value.logical_bytes);
  optional_field(writer, "logical_chunk_count", value.logical_chunk_count);
  optional_field(writer, "operations_retired", value.operations_retired);
  optional_field(writer, "warmup_passes_completed", value.warmup_passes_completed);
  optional_field(writer, "measurement_passes_completed", value.measurement_passes_completed);
  optional_field(writer, "raw_input_bytes", value.raw_input_bytes);
  optional_field(writer, "stored_bytes", value.stored_bytes);
  optional_field(writer, "stored_ratio", value.stored_ratio);
  optional_field(writer, "logical_h2d_bytes", value.logical_h2d_bytes);
  optional_field(writer, "pcie_h2d_bytes", value.pcie_h2d_bytes);
  optional_field(writer, "logical_d2h_bytes", value.logical_d2h_bytes);
  optional_field(writer, "pcie_d2h_bytes", value.pcie_d2h_bytes);
  optional_field(writer, "raw_path_decisions", value.raw_path_decisions);
  optional_field(writer, "cpu_lz4_gpu_decisions", value.cpu_lz4_gpu_decisions);
  optional_field(writer, "gpu_lz4_decisions", value.gpu_lz4_decisions);
  optional_field(writer, "never_compress_decisions", value.never_compress_decisions);
  optional_field(writer, "fallback_count", value.fallback_count);
  optional_field(writer, "mappings", value.mappings);
  optional_field(writer, "set_access", value.set_access);
  optional_field(writer, "unmaps", value.unmaps);
  optional_field(writer, "events_recorded", value.events_recorded);
  optional_field(writer, "events_retired", value.events_retired);
  optional_field(writer, "unsafe_remaps", value.unsafe_remaps);
  optional_field(writer, "unsafe_transitions", value.unsafe_transitions);
  optional_field(writer, "elapsed_ms", value.elapsed_ms);
  optional_field(writer, "expected_digest128", value.expected_digest128);
  optional_field(writer, "output_digest128", value.output_digest128);
  optional_field(writer, "mismatch_count", value.mismatch_count);
  optional_field(writer, "first_mismatch_byte_offset", value.first_mismatch_byte_offset);
  writer.end_object();
}

void write_backing(JsonWriter& writer, const BackingStatistics& value) {
  writer.begin_object();
  optional_field(writer, "logical_bytes", value.logical_bytes);
  optional_field(writer, "host_store_cap_bytes", value.host_store_cap_bytes);
  optional_field(writer, "host_bytes_current", value.host_bytes_current);
  optional_field(writer, "host_bytes_peak", value.host_bytes_peak);
  optional_field(writer, "host_budget_bytes_current", value.host_budget_bytes_current);
  optional_field(writer, "host_budget_bytes_peak", value.host_budget_bytes_peak);
  optional_field(writer, "raw_bytes_current", value.raw_bytes_current);
  optional_field(writer, "raw_bytes_peak", value.raw_bytes_peak);
  optional_field(writer, "compressed_bytes_current", value.compressed_bytes_current);
  optional_field(writer, "compressed_bytes_peak", value.compressed_bytes_peak);
  optional_field(writer, "spill_reserved_bytes_peak", value.spill_reserved_bytes_peak);
  optional_field(writer, "conversion_scratch_bytes_peak", value.conversion_scratch_bytes_peak);
  optional_field(writer, "invalid_chunks", value.invalid_chunks);
  optional_field(writer, "implicit_zero_chunks", value.implicit_zero_chunks);
  optional_field(writer, "raw_chunks", value.raw_chunks);
  optional_field(writer, "lz4_chunks", value.lz4_chunks);
  optional_field(writer, "generations_created", value.generations_created);
  optional_field(writer, "generations_committed", value.generations_committed);
  optional_field(writer, "generations_discarded", value.generations_discarded);
  optional_field(writer, "atomic_commit_failures", value.atomic_commit_failures);
  optional_field(writer, "expansion_rejections", value.expansion_rejections);
  optional_field(writer, "effective_stored_ratio", value.effective_stored_ratio);
  writer.end_object();
}

void write_codec(JsonWriter& writer, const CodecStatistics& value) {
  writer.begin_object();
  writer.key("container_version");
  writer.value(value.container_version);
  unsigned_field(writer, "block_bytes", value.block_bytes);
  writer.key("cpu_codec");
  writer.value(value.cpu_codec);
  optional_field(writer, "cpu_codec_version", value.cpu_codec_version);
  optional_field(writer, "nvcomp_available", value.nvcomp_available);
  optional_field(writer, "nvcomp_version", value.nvcomp_version);
  optional_field(writer, "nvcomp_library_source", value.nvcomp_library_source);
  optional_field(writer, "nvcomp_library_sha256", value.nvcomp_library_sha256);
  optional_field(writer, "cpu_encode_operations", value.cpu_encode_operations);
  optional_field(writer, "cpu_decode_operations", value.cpu_decode_operations);
  optional_field(writer, "gpu_encode_operations", value.gpu_encode_operations);
  optional_field(writer, "gpu_decode_operations", value.gpu_decode_operations);
  optional_field(writer, "raw_path_decisions", value.raw_path_decisions);
  optional_field(writer, "cpu_lz4_gpu_decisions", value.cpu_lz4_gpu_decisions);
  optional_field(writer, "gpu_lz4_decisions", value.gpu_lz4_decisions);
  optional_field(writer, "never_compress_decisions", value.never_compress_decisions);
  optional_field(writer, "calibration_samples", value.calibration_samples);
  optional_field(writer, "fallback_count", value.fallback_count);
  optional_field(writer, "verification_failures", value.verification_failures);
  optional_field(writer, "codec_slots_peak", value.codec_slots_peak);
  optional_field(writer, "workspace_bytes_peak", value.workspace_bytes_peak);
  optional_field(writer, "device_slot_bytes_peak", value.device_slot_bytes_peak);
  optional_field(writer, "device_slot_capacity_bytes", value.device_slot_capacity_bytes);
  writer.key("cpu_encode_timing");
  write_timing(writer, value.cpu_encode_timing);
  writer.key("cpu_decode_timing");
  write_timing(writer, value.cpu_decode_timing);
  writer.key("gpu_encode_timing");
  write_timing(writer, value.gpu_encode_timing);
  writer.key("gpu_decode_timing");
  write_timing(writer, value.gpu_decode_timing);
  writer.key("verification_timing");
  write_timing(writer, value.verification_timing);
  writer.end_object();
}

void write_telemetry(JsonWriter& writer, const Telemetry& value) {
  writer.begin_object();
  optional_field(writer, "total_elapsed_ms", value.total_elapsed_ms);
  optional_field(writer, "logical_h2d_bytes", value.logical_h2d_bytes);
  optional_field(writer, "pcie_h2d_bytes", value.pcie_h2d_bytes);
  optional_field(writer, "pcie_h2d_payload_bytes", value.pcie_h2d_payload_bytes);
  optional_field(writer, "pcie_h2d_metadata_bytes", value.pcie_h2d_metadata_bytes);
  optional_field(writer, "logical_d2h_bytes", value.logical_d2h_bytes);
  optional_field(writer, "pcie_d2h_bytes", value.pcie_d2h_bytes);
  optional_field(writer, "pcie_d2h_payload_bytes", value.pcie_d2h_payload_bytes);
  optional_field(writer, "pcie_d2h_metadata_bytes", value.pcie_d2h_metadata_bytes);
  optional_field(writer, "rejected_candidate_logical_d2h_bytes",
                 value.rejected_candidate_logical_d2h_bytes);
  optional_field(writer, "mapping_count", value.mapping_count);
  optional_field(writer, "set_access_count", value.set_access_count);
  optional_field(writer, "unmap_count", value.unmap_count);
  optional_field(writer, "event_record_count", value.event_record_count);
  optional_field(writer, "event_retire_count", value.event_retire_count);
  optional_field(writer, "handle_reuse_count", value.handle_reuse_count);
  optional_field(writer, "unsafe_remap_count", value.unsafe_remap_count);
  optional_field(writer, "unsafe_transition_count", value.unsafe_transition_count);
  optional_field(writer, "writeback_count", value.writeback_count);
  optional_field(writer, "target_shrink_count", value.target_shrink_count);
  optional_field(writer, "target_grow_count", value.target_grow_count);
  optional_field(writer, "budget_sample_count", value.budget_sample_count);
  optional_field(writer, "cache_target_bytes_minimum", value.cache_target_bytes_minimum);
  optional_field(writer, "cache_target_bytes_maximum", value.cache_target_bytes_maximum);
  optional_field(writer, "safe_device_budget_bytes_minimum",
                 value.safe_device_budget_bytes_minimum);
  optional_field(writer, "managed_device_bytes_peak", value.managed_device_bytes_peak);
  optional_field(writer, "device_reserve_bytes_peak", value.device_reserve_bytes_peak);
  optional_field(writer, "device_budget_violation_count", value.device_budget_violation_count);
  optional_field(writer, "trace_records_emitted", value.trace_records_emitted);
  optional_field(writer, "trace_records_dropped", value.trace_records_dropped);
  optional_field(writer, "trace_complete", value.trace_complete);
  writer.key("raw_transfer_timing");
  write_timing(writer, value.raw_transfer_timing);
  writer.key("compressed_transfer_timing");
  write_timing(writer, value.compressed_transfer_timing);
  writer.key("writeback_timing");
  write_timing(writer, value.writeback_timing);
  writer.end_object();
}

void write_proof(JsonWriter& writer, const Proof& value) {
  writer.begin_object();
  optional_field(writer, "logical_data_exceeds_vram", value.logical_data_exceeds_vram);
  optional_field(writer, "cache_smaller_than_logical", value.cache_smaller_than_logical);
  optional_field(writer, "authoritative_compressed_backing_verified",
                 value.authoritative_compressed_backing_verified);
  optional_field(writer, "no_expansion_stored", value.no_expansion_stored);
  optional_field(writer, "generation_atomicity_verified", value.generation_atomicity_verified);
  optional_field(writer, "write_admission_verified", value.write_admission_verified);
  optional_field(writer, "gpu_encode_before_d2h_verified", value.gpu_encode_before_d2h_verified);
  optional_field(writer, "compressed_h2d_reduction_verified",
                 value.compressed_h2d_reduction_verified);
  optional_field(writer, "stable_virtual_addresses_verified",
                 value.stable_virtual_addresses_verified);
  optional_field(writer, "maps_match_set_access", value.maps_match_set_access);
  optional_field(writer, "event_boundaries_verified", value.event_boundaries_verified);
  optional_field(writer, "host_budget_respected", value.host_budget_respected);
  optional_field(writer, "device_budget_respected", value.device_budget_respected);
  optional_field(writer, "all_workloads_match_reference", value.all_workloads_match_reference);
  optional_field(writer, "path_digests_match", value.path_digests_match);
  optional_field(writer, "policy_digests_match", value.policy_digests_match);
  optional_field(writer, "raw_virtual_addresses_omitted", value.raw_virtual_addresses_omitted);
  writer.end_object();
}

void write_outcome(JsonWriter& writer, const Outcome& value) {
  writer.begin_object();
  writer.key("status");
  writer.value(value.status);
  optional_field(writer, "reason", value.reason);
  writer.key("exit_code");
  writer.value(static_cast<std::int64_t>(value.exit_code));
  optional_field(writer, "stage", value.stage);
  optional_field(writer, "operation", value.operation);
  optional_field(writer, "native_code", value.native_code);
  optional_field(writer, "native_name", value.native_name);
  optional_field(writer, "message", value.message);
  optional_field(writer, "scenario", value.scenario);
  optional_field(writer, "allocation_id", value.allocation_id);
  optional_field(writer, "chunk_index", value.chunk_index);
  optional_field(writer, "generation", value.generation);
  optional_field(writer, "logical_byte_offset", value.logical_byte_offset);
  writer.end_object();
}

void write_cleanup(JsonWriter& writer, const Cleanup& value) {
  writer.begin_object();
  optional_field(writer, "complete", value.complete);
  optional_field(writer, "operations_drained", value.operations_drained);
  optional_field(writer, "codec_slots_drained", value.codec_slots_drained);
  optional_field(writer, "events_drained", value.events_drained);
  optional_field(writer, "events_destroyed", value.events_destroyed);
  optional_field(writer, "streams_destroyed", value.streams_destroyed);
  optional_field(writer, "mappings_removed", value.mappings_removed);
  optional_field(writer, "physical_handles_released", value.physical_handles_released);
  optional_field(writer, "codec_workspace_released", value.codec_workspace_released);
  optional_field(writer, "virtual_reservations_released", value.virtual_reservations_released);
  optional_field(writer, "pinned_staging_released", value.pinned_staging_released);
  optional_field(writer, "spill_reservations_released", value.spill_reservations_released);
  optional_field(writer, "host_backing_released", value.host_backing_released);
  optional_field(writer, "context_released", value.context_released);
  optional_field(writer, "trace_closed", value.trace_closed);
  optional_field(writer, "worker_terminated", value.worker_terminated);
  writer.end_object();
}

void write_diagnostics(JsonWriter& writer, const std::vector<probe::Diagnostic>& values) {
  writer.begin_array();
  for (const probe::Diagnostic& value : values) {
    writer.begin_object();
    writer.key("level");
    writer.value(diagnostic_level_name(value.level));
    writer.key("component");
    writer.value(safe_text(value.component));
    writer.key("operation");
    writer.value(safe_text(value.operation));
    writer.key("message");
    writer.value(safe_text(value.message));
    optional_field(writer, "code", value.code);
    optional_field(writer, "device_ordinal", value.device_ordinal);
    writer.end_object();
  }
  writer.end_array();
}

} // namespace

void write_json(const Report& report, std::ostream& output, const bool pretty) {
  JsonWriter writer(output, pretty);
  writer.begin_object();
  unsigned_field(writer, "schema_version", report.schema_version);
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
  writer.key("backing");
  write_backing(writer, report.backing);
  writer.key("codec");
  write_codec(writer, report.codec);
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
  unsigned_field(writer, "schema_version", record.schema_version);
  writer.key("report_type");
  writer.value(record.report_type);
  unsigned_field(writer, "sequence", record.sequence);
  unsigned_field(writer, "monotonic_time_ns", record.monotonic_time_ns);
  writer.key("event");
  writer.value(safe_text(record.event));
  optional_field(writer, "allocation_id", record.allocation_id);
  optional_field(writer, "chunk_index", record.chunk_index);
  optional_field(writer, "operation_id", record.operation_id);
  optional_field(writer, "source_generation", record.source_generation);
  optional_field(writer, "target_generation", record.target_generation);
  optional_field(writer, "slot_generation", record.slot_generation);
  optional_field(writer, "from_representation", record.from_representation);
  optional_field(writer, "to_representation", record.to_representation);
  optional_field(writer, "path", record.path);
  optional_field(writer, "logical_bytes", record.logical_bytes);
  optional_field(writer, "physical_bytes", record.physical_bytes);
  optional_field(writer, "reason", record.reason);
  optional_field(writer, "speculative", record.speculative);
  writer.end_object();
  output << '\n';
}

} // namespace xvram::compression
