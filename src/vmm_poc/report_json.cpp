#include "xvram/vmm_poc/report.hpp"

#include "xvram/base/json_writer.hpp"

#include <optional>
#include <ostream>
#include <string_view>
#include <type_traits>

namespace xvram::vmm_poc {
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
  writer.value(device.free_memory_bytes_start);
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
  writer.key("requested_chunk_bytes");
  write_optional(writer, configuration.requested_chunk_bytes);
  writer.key("requested_window_slots");
  write_optional(writer, configuration.requested_window_slots);
  writer.key("effective_window_slots");
  write_optional(writer, configuration.effective_window_slots);
  writer.key("passes");
  writer.value(static_cast<std::uint64_t>(configuration.passes));
  writer.key("timeout_ms");
  writer.value(configuration.timeout_ms);
  writer.key("stall_timeout_ms");
  writer.value(configuration.stall_timeout_ms);
  writer.key("seed_hex");
  writer.value(configuration.seed_hex);
  writer.key("host_headroom_bytes");
  writer.value(configuration.host_headroom_bytes);
  writer.key("device_headroom_bytes");
  writer.value(configuration.device_headroom_bytes);
  writer.key("sizing_mode");
  writer.value(configuration.sizing_mode);
  writer.key("mode");
  writer.value(configuration.mode);
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

void write_mode(JsonWriter& writer, const ModeResult& mode) {
  writer.begin_object();
  writer.key("status");
  writer.value(mode.status);
  writer.key("elements_processed");
  write_optional(writer, mode.elements_processed);
  writer.key("tiles_processed");
  write_optional(writer, mode.tiles_processed);
  writer.key("passes_completed");
  write_optional(writer, mode.passes_completed);
  writer.key("bytes_h2d");
  write_optional(writer, mode.bytes_h2d);
  writer.key("bytes_d2h");
  write_optional(writer, mode.bytes_d2h);
  writer.key("mapping_count");
  write_optional(writer, mode.mapping_count);
  writer.key("unmap_count");
  write_optional(writer, mode.unmap_count);
  writer.key("remap_count");
  write_optional(writer, mode.remap_count);
  writer.key("event_boundary_count");
  write_optional(writer, mode.event_boundary_count);
  writer.key("unsafe_remap_count");
  write_optional(writer, mode.unsafe_remap_count);
  writer.key("physical_handle_count");
  write_optional(writer, mode.physical_handle_count);
  writer.key("max_concurrent_mappings");
  write_optional(writer, mode.max_concurrent_mappings);
  writer.key("slot_count");
  write_optional(writer, mode.slot_count);
  writer.key("set_access_count");
  write_optional(writer, mode.set_access_count);
  writer.key("handle_reuse_count");
  write_optional(writer, mode.handle_reuse_count);
  writer.key("stable_addresses_verified");
  write_optional(writer, mode.stable_addresses_verified);
  writer.key("full_verification_completed");
  write_optional(writer, mode.full_verification_completed);
  writer.key("matches_cpu");
  write_optional(writer, mode.matches_cpu);
  writer.key("elapsed_ms");
  write_optional(writer, mode.elapsed_ms);
  writer.key("setup_ms");
  write_optional(writer, mode.setup_ms);
  writer.key("h2d_ms");
  write_optional(writer, mode.h2d_ms);
  writer.key("kernel_ms");
  write_optional(writer, mode.kernel_ms);
  writer.key("d2h_ms");
  write_optional(writer, mode.d2h_ms);
  writer.key("verification_ms");
  write_optional(writer, mode.verification_ms);
  writer.key("remap_timing");
  write_timing(writer, mode.remap_timing);
  writer.key("expected_digest128");
  write_optional(writer, mode.expected_digest128);
  writer.key("output_digest128");
  write_optional(writer, mode.output_digest128);
  writer.key("mismatch_count");
  write_optional(writer, mode.mismatch_count);
  writer.key("first_mismatch_byte_offset");
  write_optional(writer, mode.first_mismatch_byte_offset);
  writer.end_object();
}

void write_modes(JsonWriter& writer, const Modes& modes) {
  writer.begin_object();
  writer.key("backing");
  writer.value(modes.backing);
  writer.key("traversal");
  writer.value(modes.traversal);
  writer.key("verification");
  writer.value(modes.verification);
  writer.key("timeout_enforcement");
  writer.value(modes.timeout_enforcement);
  writer.key("reference");
  write_mode(writer, modes.reference);
  writer.key("pipeline");
  write_mode(writer, modes.pipeline);
  writer.key("pipeline_speedup");
  write_optional(writer, modes.pipeline_speedup);
  writer.end_object();
}

void write_proof(JsonWriter& writer, const Proof& proof) {
  writer.begin_object();
  writer.key("effective_logical_bytes");
  write_optional(writer, proof.effective_logical_bytes);
  writer.key("effective_chunk_bytes");
  write_optional(writer, proof.effective_chunk_bytes);
  writer.key("logical_element_count");
  write_optional(writer, proof.logical_element_count);
  writer.key("logical_chunk_count");
  write_optional(writer, proof.logical_chunk_count);
  writer.key("tile_visit_count");
  write_optional(writer, proof.tile_visit_count);
  writer.key("address_revisit_count");
  write_optional(writer, proof.address_revisit_count);
  writer.key("resident_physical_bytes");
  write_optional(writer, proof.resident_physical_bytes);
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
  writer.key("handles_reused");
  write_optional(writer, proof.handles_reused);
  writer.key("physical_window_smaller");
  write_optional(writer, proof.physical_window_smaller);
  writer.key("stable_virtual_addresses_verified");
  write_optional(writer, proof.stable_virtual_addresses_verified);
  writer.key("event_boundaries_verified");
  write_optional(writer, proof.event_boundaries_verified);
  writer.key("reference_matches_cpu");
  write_optional(writer, proof.reference_matches_cpu);
  writer.key("pipeline_matches_cpu");
  write_optional(writer, proof.pipeline_matches_cpu);
  writer.key("modes_match");
  write_optional(writer, proof.modes_match);
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
  writer.key("mode");
  write_optional(writer, outcome.mode);
  writer.key("pass_index");
  write_optional(writer, outcome.pass_index);
  writer.key("tile_index");
  write_optional(writer, outcome.tile_index);
  writer.key("logical_byte_offset");
  write_optional(writer, outcome.logical_byte_offset);
  writer.end_object();
}

void write_cleanup(JsonWriter& writer, const Cleanup& cleanup) {
  writer.begin_object();
  writer.key("complete");
  write_optional(writer, cleanup.complete);
  writer.key("events_drained");
  write_optional(writer, cleanup.events_drained);
  writer.key("events_destroyed");
  write_optional(writer, cleanup.events_destroyed);
  writer.key("streams_destroyed");
  write_optional(writer, cleanup.streams_destroyed);
  writer.key("module_unloaded");
  write_optional(writer, cleanup.module_unloaded);
  writer.key("device_allocations_released");
  write_optional(writer, cleanup.device_allocations_released);
  writer.key("mappings_removed");
  write_optional(writer, cleanup.mappings_removed);
  writer.key("physical_handle_released");
  write_optional(writer, cleanup.physical_handle_released);
  writer.key("virtual_reservation_released");
  write_optional(writer, cleanup.virtual_reservation_released);
  writer.key("pinned_staging_released");
  write_optional(writer, cleanup.pinned_staging_released);
  writer.key("host_backing_released");
  write_optional(writer, cleanup.host_backing_released);
  writer.key("context_destroyed");
  write_optional(writer, cleanup.context_destroyed);
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
  writer.key("modes");
  write_modes(writer, report.modes);
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

} // namespace xvram::vmm_poc
