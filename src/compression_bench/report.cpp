#include "xvram/compression/report.hpp"

#include <algorithm>
#include <string>

namespace xvram::compression {
namespace {

template <typename T>
[[nodiscard]] bool equal_present(const std::optional<T>& left, const std::optional<T>& right) {
  return left.has_value() && right.has_value() && *left == *right;
}

void require_true(const std::optional<bool>& value, const char* name,
                  std::vector<std::string>& errors) {
  if (!value.value_or(false)) {
    errors.emplace_back(std::string("proof.") + name + " must be true");
  }
}

void require_cleanup(const std::optional<bool>& value, const char* name,
                     std::vector<std::string>& errors) {
  if (!value.value_or(false)) {
    errors.emplace_back(std::string("cleanup.") + name + " must be true");
  }
}

} // namespace

void finalize_proof(Report& report) {
  if (report.device.has_value() && report.configuration.effective_logical_bytes.has_value()) {
    report.proof.logical_data_exceeds_vram =
        *report.configuration.effective_logical_bytes > report.device->total_memory_bytes;
  }
  if (report.configuration.effective_logical_bytes.has_value() &&
      report.configuration.initial_cache_target_bytes.has_value()) {
    report.proof.cache_smaller_than_logical = *report.configuration.initial_cache_target_bytes <
                                              *report.configuration.effective_logical_bytes;
  }
  if (report.backing.lz4_chunks.has_value() &&
      report.backing.compressed_bytes_current.has_value()) {
    report.proof.authoritative_compressed_backing_verified =
        *report.backing.lz4_chunks > 0 && *report.backing.compressed_bytes_current > 0;
  }
  if (report.backing.logical_bytes.has_value() && report.backing.host_bytes_current.has_value()) {
    report.proof.no_expansion_stored =
        *report.backing.host_bytes_current <= *report.backing.logical_bytes;
  }
  if (report.backing.generations_created.has_value() &&
      report.backing.generations_committed.has_value() &&
      report.backing.generations_discarded.has_value() &&
      report.backing.atomic_commit_failures.has_value()) {
    report.proof.generation_atomicity_verified =
        *report.backing.generations_created ==
            *report.backing.generations_committed + *report.backing.generations_discarded &&
        *report.backing.atomic_commit_failures == 0;
  }
  if (report.telemetry.logical_h2d_bytes.has_value() &&
      report.telemetry.pcie_h2d_bytes.has_value()) {
    report.proof.compressed_h2d_reduction_verified =
        *report.telemetry.pcie_h2d_bytes < *report.telemetry.logical_h2d_bytes;
  }
  if (report.telemetry.mapping_count.has_value() && report.telemetry.set_access_count.has_value()) {
    report.proof.maps_match_set_access =
        *report.telemetry.mapping_count == *report.telemetry.set_access_count;
  }
  if (report.telemetry.event_record_count.has_value() &&
      report.telemetry.event_retire_count.has_value() && report.telemetry.unmap_count.has_value()) {
    report.proof.event_boundaries_verified =
        *report.telemetry.event_record_count == *report.telemetry.event_retire_count &&
        *report.telemetry.unmap_count <= *report.telemetry.event_retire_count;
  }
  if (report.backing.host_store_cap_bytes.has_value() &&
      report.backing.host_bytes_peak.has_value()) {
    report.proof.host_budget_respected =
        *report.backing.host_bytes_peak <= *report.backing.host_store_cap_bytes;
  }
  if (report.configuration.initial_cache_target_bytes.has_value() &&
      report.telemetry.cache_target_bytes_maximum.has_value()) {
    report.proof.device_budget_respected = *report.telemetry.cache_target_bytes_maximum <=
                                           *report.configuration.initial_cache_target_bytes;
  }

  if (!report.workloads.empty()) {
    report.proof.all_workloads_match_reference = std::all_of(
        report.workloads.begin(), report.workloads.end(), [](const WorkloadResult& item) {
          return item.status == "completed" && item.mismatch_count.value_or(1) == 0 &&
                 equal_present(item.expected_digest128, item.output_digest128);
        });
  }

  // The public model and both serializers intentionally have no address or stream-handle fields.
  report.proof.raw_virtual_addresses_omitted = true;
}

std::vector<std::string> validate_success_semantics(const Report& report) {
  std::vector<std::string> errors;
  if (report.schema_version != 1 || report.report_type != "xvram.adaptive_compression") {
    errors.emplace_back("report identity is not xvram.adaptive_compression v1");
  }
  if (report.outcome.status != "completed" || report.outcome.exit_code != 0) {
    errors.emplace_back("successful report must have completed outcome and exit code 0");
  }
  if (!report.device.has_value()) {
    errors.emplace_back("successful report must identify a device");
  }
  if (report.workloads.empty()) {
    errors.emplace_back("successful report must contain at least one workload");
  }
  for (const WorkloadResult& workload : report.workloads) {
    if (workload.status != "completed") {
      errors.emplace_back("every workload must be completed");
    }
    if (!equal_present(workload.expected_digest128, workload.output_digest128) ||
        workload.mismatch_count.value_or(1) != 0) {
      errors.emplace_back("every workload must match its reference digest");
    }
    if (workload.unsafe_remaps.value_or(1) != 0 || workload.unsafe_transitions.value_or(1) != 0) {
      errors.emplace_back("workload unsafe counters must be zero");
    }
    if (!equal_present(workload.mappings, workload.set_access) ||
        !equal_present(workload.mappings, workload.unmaps)) {
      errors.emplace_back("workload map, SetAccess, and unmap counters must reconcile");
    }
    if (!equal_present(workload.events_recorded, workload.events_retired)) {
      errors.emplace_back("workload event counters must reconcile");
    }
  }

  require_true(report.proof.logical_data_exceeds_vram, "logical_data_exceeds_vram", errors);
  require_true(report.proof.cache_smaller_than_logical, "cache_smaller_than_logical", errors);
  require_true(report.proof.generation_atomicity_verified, "generation_atomicity_verified", errors);
  require_true(report.proof.stable_virtual_addresses_verified, "stable_virtual_addresses_verified",
               errors);
  require_true(report.proof.maps_match_set_access, "maps_match_set_access", errors);
  require_true(report.proof.event_boundaries_verified, "event_boundaries_verified", errors);
  require_true(report.proof.host_budget_respected, "host_budget_respected", errors);
  require_true(report.proof.device_budget_respected, "device_budget_respected", errors);
  require_true(report.proof.all_workloads_match_reference, "all_workloads_match_reference", errors);
  require_true(report.proof.raw_virtual_addresses_omitted, "raw_virtual_addresses_omitted", errors);

  if (report.telemetry.unsafe_remap_count.value_or(1) != 0 ||
      report.telemetry.unsafe_transition_count.value_or(1) != 0) {
    errors.emplace_back("global unsafe counters must be zero");
  }
  if (!equal_present(report.telemetry.mapping_count, report.telemetry.set_access_count) ||
      !equal_present(report.telemetry.mapping_count, report.telemetry.unmap_count)) {
    errors.emplace_back("global map, SetAccess, and unmap counters must reconcile");
  }
  if (!equal_present(report.telemetry.event_record_count, report.telemetry.event_retire_count)) {
    errors.emplace_back("global event counters must reconcile");
  }
  if (report.backing.generations_created.has_value() &&
      report.backing.generations_committed.has_value() &&
      report.backing.generations_discarded.has_value() &&
      *report.backing.generations_created !=
          *report.backing.generations_committed + *report.backing.generations_discarded) {
    errors.emplace_back("backing generation counters must reconcile");
  }

  require_cleanup(report.cleanup.complete, "complete", errors);
  require_cleanup(report.cleanup.operations_drained, "operations_drained", errors);
  require_cleanup(report.cleanup.codec_slots_drained, "codec_slots_drained", errors);
  require_cleanup(report.cleanup.events_drained, "events_drained", errors);
  require_cleanup(report.cleanup.events_destroyed, "events_destroyed", errors);
  require_cleanup(report.cleanup.streams_destroyed, "streams_destroyed", errors);
  require_cleanup(report.cleanup.mappings_removed, "mappings_removed", errors);
  require_cleanup(report.cleanup.physical_handles_released, "physical_handles_released", errors);
  require_cleanup(report.cleanup.codec_workspace_released, "codec_workspace_released", errors);
  require_cleanup(report.cleanup.virtual_reservations_released, "virtual_reservations_released",
                  errors);
  require_cleanup(report.cleanup.pinned_staging_released, "pinned_staging_released", errors);
  require_cleanup(report.cleanup.spill_reservations_released, "spill_reservations_released",
                  errors);
  require_cleanup(report.cleanup.host_backing_released, "host_backing_released", errors);
  require_cleanup(report.cleanup.context_released, "context_released", errors);
  require_cleanup(report.cleanup.trace_closed, "trace_closed", errors);
  require_cleanup(report.cleanup.worker_terminated, "worker_terminated", errors);
  return errors;
}

} // namespace xvram::compression
