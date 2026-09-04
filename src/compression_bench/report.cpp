#include "xvram/compression/report.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <limits>
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

bool contains_forbidden_runtime_identity(const std::string_view text) noexcept {
  try {
    std::string lowered(text);
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](const unsigned char value) {
      return static_cast<char>(std::tolower(value));
    });
    constexpr std::array forbidden{"cuda_va",        "raw_va",       "virtual_address",
                                   "device_pointer", "host_pointer", "stream_handle",
                                   "device address", "host address"};
    if (std::any_of(forbidden.begin(), forbidden.end(), [&](const std::string_view token) {
          return lowered.find(token) != std::string::npos;
        })) {
      return true;
    }
    for (std::size_t index = 0; index + 2U < lowered.size(); ++index) {
      if (lowered[index] != '0' || lowered[index + 1U] != 'x') {
        continue;
      }
      std::size_t digits = 0;
      for (std::size_t cursor = index + 2U;
           cursor < lowered.size() && std::isxdigit(static_cast<unsigned char>(lowered[cursor]));
           ++cursor) {
        ++digits;
      }
      if (digits >= 8U) {
        return true;
      }
    }
    return false;
  } catch (...) {
    return true;
  }
}

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
  bool compressed_h2d_case_seen = false;
  bool compressed_h2d_cases_reduced = true;
  for (const WorkloadResult& workload : report.workloads) {
    const bool compressed_path =
        workload.path == "cpu_lz4_gpu_decode" || workload.path == "nvcomp_gpu_codec";
    const bool stored_compressed =
        workload.stored_ratio.has_value() && *workload.stored_ratio < 1.0;
    if (!compressed_path || !stored_compressed || workload.logical_h2d_bytes.value_or(0U) == 0U ||
        !workload.pcie_h2d_bytes.has_value()) {
      continue;
    }
    compressed_h2d_case_seen = true;
    compressed_h2d_cases_reduced =
        compressed_h2d_cases_reduced &&
        *workload.pcie_h2d_bytes < *workload.logical_h2d_bytes;
  }
  if (compressed_h2d_case_seen) {
    report.proof.compressed_h2d_reduction_verified = compressed_h2d_cases_reduced;
  } else {
    report.proof.compressed_h2d_reduction_verified.reset();
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
      report.backing.host_budget_bytes_peak.has_value()) {
    report.proof.host_budget_respected =
        *report.backing.host_budget_bytes_peak <= *report.backing.host_store_cap_bytes;
  }
  if (report.configuration.initial_cache_target_bytes.has_value() &&
      report.telemetry.cache_target_bytes_maximum.has_value() &&
      report.telemetry.safe_device_budget_bytes_minimum.has_value() &&
      report.telemetry.managed_device_bytes_peak.has_value() &&
      report.telemetry.device_reserve_bytes_peak.has_value() &&
      report.telemetry.device_budget_violation_count.has_value()) {
    report.proof.device_budget_respected = *report.telemetry.cache_target_bytes_maximum <=
                                               *report.configuration.initial_cache_target_bytes &&
                                           *report.telemetry.managed_device_bytes_peak <=
                                               *report.telemetry.cache_target_bytes_maximum &&
                                           *report.telemetry.device_reserve_bytes_peak <=
                                               *report.telemetry.cache_target_bytes_maximum &&
                                           *report.telemetry.device_budget_violation_count == 0U;
  }

  if (!report.workloads.empty()) {
    const bool all_match_reference = std::all_of(
        report.workloads.begin(), report.workloads.end(), [](const WorkloadResult& item) {
          return item.status == "completed" && item.mismatch_count.value_or(1) == 0 &&
                 equal_present(item.expected_digest128, item.output_digest128);
        });
    report.proof.all_workloads_match_reference = all_match_reference;

    // A report containing multiple paths/policies proves parity by comparing like-for-like
    // cases. A single-case report cannot make that comparison, so its deterministic CPU
    // reference is the canonical parity anchor instead of an unconditional true flag.
    bool path_comparison_seen = false;
    bool policy_comparison_seen = false;
    bool path_comparisons_match = true;
    bool policy_comparisons_match = true;
    for (std::size_t left = 0; left < report.workloads.size(); ++left) {
      const WorkloadResult& a = report.workloads[left];
      for (std::size_t right = left + 1U; right < report.workloads.size(); ++right) {
        const WorkloadResult& b = report.workloads[right];
        const bool same_case = a.scenario == b.scenario &&
                               a.compression_policy == b.compression_policy &&
                               a.logical_bytes == b.logical_bytes;
        if (!same_case || !a.output_digest128.has_value() || !b.output_digest128.has_value()) {
          continue;
        }
        if (a.path != b.path) {
          path_comparison_seen = true;
          path_comparisons_match =
              path_comparisons_match && a.output_digest128 == b.output_digest128;
        }
        if (a.replacement_policy != b.replacement_policy) {
          policy_comparison_seen = true;
          policy_comparisons_match =
              policy_comparisons_match && a.output_digest128 == b.output_digest128;
        }
      }
    }
    report.proof.path_digests_match =
        all_match_reference && (!path_comparison_seen || path_comparisons_match);
    report.proof.policy_digests_match =
        all_match_reference && (!policy_comparison_seen || policy_comparisons_match);
  }

  bool identities_omitted = true;
  const auto inspect = [&](const std::optional<std::string>& value) {
    identities_omitted =
        identities_omitted && (!value.has_value() || !contains_forbidden_runtime_identity(*value));
  };
  inspect(report.outcome.reason);
  inspect(report.outcome.stage);
  inspect(report.outcome.operation);
  inspect(report.outcome.native_name);
  inspect(report.outcome.message);
  inspect(report.outcome.scenario);
  for (const probe::Diagnostic& diagnostic : report.diagnostics) {
    identities_omitted = identities_omitted &&
                         !contains_forbidden_runtime_identity(diagnostic.component) &&
                         !contains_forbidden_runtime_identity(diagnostic.operation) &&
                         !contains_forbidden_runtime_identity(diagnostic.message);
  }
  report.proof.raw_virtual_addresses_omitted = identities_omitted;
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
  std::uint64_t workload_logical_h2d = 0;
  std::uint64_t workload_pcie_h2d = 0;
  std::uint64_t workload_logical_d2h = 0;
  std::uint64_t workload_pcie_d2h = 0;
  std::uint64_t workload_mappings = 0;
  std::uint64_t workload_set_access = 0;
  std::uint64_t workload_unmaps = 0;
  std::uint64_t workload_events_recorded = 0;
  std::uint64_t workload_events_retired = 0;
  std::uint64_t workload_raw_decisions = 0;
  std::uint64_t workload_cpu_decisions = 0;
  std::uint64_t workload_gpu_decisions = 0;
  std::uint64_t workload_never_compress = 0;
  std::uint64_t workload_fallbacks = 0;
  bool workload_counters_complete = true;
  const auto add_counter = [&](const std::optional<std::uint64_t>& value, std::uint64_t& total,
                               const char* name) {
    if (!value.has_value()) {
      errors.emplace_back(std::string("completed workload is missing ") + name);
      workload_counters_complete = false;
      return;
    }
    if (*value > std::numeric_limits<std::uint64_t>::max() - total) {
      errors.emplace_back(std::string("workload counter overflow for ") + name);
      workload_counters_complete = false;
      return;
    }
    total += *value;
  };
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
    const std::uint64_t decisions = workload.raw_path_decisions.value_or(0U) +
                                    workload.cpu_lz4_gpu_decisions.value_or(0U) +
                                    workload.gpu_lz4_decisions.value_or(0U);
    if (workload.operations_retired.value_or(0U) != 0U && decisions == 0U) {
      errors.emplace_back("completed workload did not record a transfer-path decision");
    }
    if (workload.never_compress_decisions.value_or(0U) > workload.raw_path_decisions.value_or(0U)) {
      errors.emplace_back("workload never-compress decisions exceed raw decisions");
    }
    add_counter(workload.logical_h2d_bytes, workload_logical_h2d, "logical_h2d_bytes");
    add_counter(workload.pcie_h2d_bytes, workload_pcie_h2d, "pcie_h2d_bytes");
    add_counter(workload.logical_d2h_bytes, workload_logical_d2h, "logical_d2h_bytes");
    add_counter(workload.pcie_d2h_bytes, workload_pcie_d2h, "pcie_d2h_bytes");
    add_counter(workload.mappings, workload_mappings, "mappings");
    add_counter(workload.set_access, workload_set_access, "set_access");
    add_counter(workload.unmaps, workload_unmaps, "unmaps");
    add_counter(workload.events_recorded, workload_events_recorded, "events_recorded");
    add_counter(workload.events_retired, workload_events_retired, "events_retired");
    add_counter(workload.raw_path_decisions, workload_raw_decisions, "raw_path_decisions");
    add_counter(workload.cpu_lz4_gpu_decisions, workload_cpu_decisions, "cpu_lz4_gpu_decisions");
    add_counter(workload.gpu_lz4_decisions, workload_gpu_decisions, "gpu_lz4_decisions");
    add_counter(workload.never_compress_decisions, workload_never_compress,
                "never_compress_decisions");
    add_counter(workload.fallback_count, workload_fallbacks, "fallback_count");
  }

  require_true(report.proof.logical_data_exceeds_vram, "logical_data_exceeds_vram", errors);
  require_true(report.proof.cache_smaller_than_logical, "cache_smaller_than_logical", errors);
  require_true(report.proof.no_expansion_stored, "no_expansion_stored", errors);
  require_true(report.proof.generation_atomicity_verified, "generation_atomicity_verified", errors);
  require_true(report.proof.write_admission_verified, "write_admission_verified", errors);
  require_true(report.proof.stable_virtual_addresses_verified, "stable_virtual_addresses_verified",
               errors);
  require_true(report.proof.maps_match_set_access, "maps_match_set_access", errors);
  require_true(report.proof.event_boundaries_verified, "event_boundaries_verified", errors);
  require_true(report.proof.host_budget_respected, "host_budget_respected", errors);
  require_true(report.proof.device_budget_respected, "device_budget_respected", errors);
  require_true(report.proof.all_workloads_match_reference, "all_workloads_match_reference", errors);
  require_true(report.proof.path_digests_match, "path_digests_match", errors);
  require_true(report.proof.policy_digests_match, "policy_digests_match", errors);
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
  const auto validate_transport_split =
      [&](const std::optional<std::uint64_t>& total,
          const std::optional<std::uint64_t>& payload,
          const std::optional<std::uint64_t>& metadata,
          const std::optional<std::uint64_t>& logical_attempts, const char* direction) {
        if (!total.has_value() || !payload.has_value() || !metadata.has_value() ||
            !logical_attempts.has_value()) {
          errors.emplace_back(std::string("completed report is missing ") + direction +
                              " transport accounting");
          return;
        }
        if (*payload > std::numeric_limits<std::uint64_t>::max() - *metadata ||
            *payload + *metadata != *total) {
          errors.emplace_back(std::string(direction) +
                              " PCIe bytes do not reconcile with payload and metadata");
        }
        if (*payload > *logical_attempts) {
          errors.emplace_back(std::string(direction) +
                              " PCIe payload exceeds logical attempt bytes");
        }
      };
  validate_transport_split(report.telemetry.pcie_h2d_bytes,
                           report.telemetry.pcie_h2d_payload_bytes,
                           report.telemetry.pcie_h2d_metadata_bytes,
                           report.telemetry.logical_h2d_bytes, "H2D");
  validate_transport_split(report.telemetry.pcie_d2h_bytes,
                           report.telemetry.pcie_d2h_payload_bytes,
                           report.telemetry.pcie_d2h_metadata_bytes,
                           report.telemetry.logical_d2h_bytes, "D2H");
  if (!report.telemetry.rejected_candidate_logical_d2h_bytes.has_value() ||
      !report.telemetry.logical_d2h_bytes.has_value()) {
    errors.emplace_back("completed report is missing rejected-candidate D2H accounting");
  } else if (*report.telemetry.rejected_candidate_logical_d2h_bytes >
             *report.telemetry.logical_d2h_bytes) {
    errors.emplace_back("rejected-candidate logical D2H exceeds all logical D2H attempts");
  }
  if (report.telemetry.handle_reuse_count.value_or(0U) == 0U) {
    errors.emplace_back("oversubscribed workload did not reuse physical handles");
  }
  if (report.telemetry.device_budget_violation_count.value_or(1U) != 0U) {
    errors.emplace_back("device budget violation counter must be zero");
  }
  if (report.backing.host_store_cap_bytes.has_value() &&
      report.backing.host_budget_bytes_peak.has_value() &&
      *report.backing.host_budget_bytes_peak > *report.backing.host_store_cap_bytes) {
    errors.emplace_back("host budget peak exceeds the effective host-store cap");
  }
  if (report.backing.host_store_cap_bytes.has_value() &&
      report.backing.host_budget_bytes_current.has_value() &&
      *report.backing.host_budget_bytes_current > *report.backing.host_store_cap_bytes) {
    errors.emplace_back("current host budget exceeds the effective host-store cap");
  }
  if (!report.backing.host_bytes_current.has_value() ||
      !report.backing.raw_bytes_current.has_value() ||
      !report.backing.compressed_bytes_current.has_value() ||
      *report.backing.host_bytes_current !=
          *report.backing.raw_bytes_current + *report.backing.compressed_bytes_current) {
    errors.emplace_back("authoritative host representation bytes do not reconcile");
  }
  if (report.backing.generations_created.has_value() &&
      report.backing.generations_committed.has_value() &&
      report.backing.generations_discarded.has_value() &&
      *report.backing.generations_created !=
          *report.backing.generations_committed + *report.backing.generations_discarded) {
    errors.emplace_back("backing generation counters must reconcile");
  }
  if (report.codec.never_compress_decisions.value_or(0U) >
      report.codec.raw_path_decisions.value_or(0U)) {
    errors.emplace_back("never-compress decisions exceed raw decisions");
  }
  if (report.codec.workspace_bytes_peak.value_or(std::numeric_limits<std::uint64_t>::max()) >
      report.configuration.compression_scratch_cap_bytes) {
    errors.emplace_back("codec workspace exceeds the configured scratch cap");
  }
  if (!report.configuration.effective_chunk_bytes.has_value()) {
    errors.emplace_back("successful report is missing effective chunk bytes");
  }
  if (report.codec.codec_slots_peak.value_or(std::numeric_limits<std::uint64_t>::max()) >
      report.configuration.codec_slots) {
    errors.emplace_back("codec slot count exceeds the configured bound");
  }
  const std::uint64_t slot_peak =
      report.codec.device_slot_bytes_peak.value_or(std::numeric_limits<std::uint64_t>::max());
  const std::uint64_t slot_capacity =
      report.codec.device_slot_capacity_bytes.value_or(0U);
  if (slot_peak > slot_capacity) {
    errors.emplace_back("codec device slot bytes exceed the runtime-derived capacity");
  }
  const std::uint64_t workspace_peak =
      report.codec.workspace_bytes_peak.value_or(std::numeric_limits<std::uint64_t>::max());
  const std::uint64_t device_reserve_peak =
      report.telemetry.device_reserve_bytes_peak.value_or(0U);
  if (slot_capacity > std::numeric_limits<std::uint64_t>::max() - workspace_peak ||
      slot_capacity + workspace_peak > device_reserve_peak) {
    errors.emplace_back("codec slot and workspace capacity is not charged to the device reserve");
  }
  if (report.telemetry.trace_records_dropped.value_or(1U) != 0U ||
      !report.telemetry.trace_complete.value_or(false) ||
      !report.telemetry.trace_records_emitted.has_value()) {
    errors.emplace_back("successful trace accounting is incomplete");
  }

  if (workload_counters_complete) {
    const auto require_total = [&](const std::optional<std::uint64_t>& actual,
                                   const std::uint64_t expected, const char* name) {
      if (!actual.has_value() || *actual != expected) {
        errors.emplace_back(std::string("global ") + name + " does not equal the workload sum");
      }
    };
    require_total(report.telemetry.logical_h2d_bytes, workload_logical_h2d, "logical_h2d_bytes");
    require_total(report.telemetry.pcie_h2d_bytes, workload_pcie_h2d, "pcie_h2d_bytes");
    require_total(report.telemetry.logical_d2h_bytes, workload_logical_d2h, "logical_d2h_bytes");
    require_total(report.telemetry.pcie_d2h_bytes, workload_pcie_d2h, "pcie_d2h_bytes");
    require_total(report.telemetry.mapping_count, workload_mappings, "mapping_count");
    require_total(report.telemetry.set_access_count, workload_set_access, "set_access_count");
    require_total(report.telemetry.unmap_count, workload_unmaps, "unmap_count");
    require_total(report.telemetry.event_record_count, workload_events_recorded,
                  "event_record_count");
    require_total(report.telemetry.event_retire_count, workload_events_retired,
                  "event_retire_count");
    require_total(report.codec.raw_path_decisions, workload_raw_decisions, "raw_path_decisions");
    require_total(report.codec.cpu_lz4_gpu_decisions, workload_cpu_decisions,
                  "cpu_lz4_gpu_decisions");
    require_total(report.codec.gpu_lz4_decisions, workload_gpu_decisions, "gpu_lz4_decisions");
    require_total(report.codec.never_compress_decisions, workload_never_compress,
                  "never_compress_decisions");
    require_total(report.codec.fallback_count, workload_fallbacks, "fallback_count");
  }

  const bool used_compressed_path = std::any_of(
      report.workloads.begin(), report.workloads.end(), [](const WorkloadResult& workload) {
        return workload.path == "cpu_lz4_gpu_decode" || workload.path == "nvcomp_gpu_codec";
      });
  if (used_compressed_path) {
    require_true(report.proof.authoritative_compressed_backing_verified,
                 "authoritative_compressed_backing_verified", errors);
    const bool compression_reduction_expected = std::any_of(
        report.workloads.begin(), report.workloads.end(), [](const WorkloadResult& workload) {
          const bool compressed_path = workload.path == "cpu_lz4_gpu_decode" ||
                                       workload.path == "nvcomp_gpu_codec";
          return compressed_path && workload.stored_ratio.has_value() &&
                 *workload.stored_ratio < 1.0 && workload.logical_h2d_bytes.value_or(0U) != 0U;
        });
    if (compression_reduction_expected) {
      require_true(report.proof.compressed_h2d_reduction_verified,
                   "compressed_h2d_reduction_verified", errors);
    }
  }
  if (report.codec.gpu_encode_operations.value_or(0U) != 0U) {
    require_true(report.proof.gpu_encode_before_d2h_verified, "gpu_encode_before_d2h_verified",
                 errors);
  }
  if (report.codec.verification_failures.value_or(1U) != 0U) {
    errors.emplace_back("codec verification failures must be zero");
  }
  if (report.codec.gpu_encode_operations.has_value() &&
      report.codec.gpu_decode_operations.has_value()) {
    if (*report.codec.gpu_encode_operations >
        std::numeric_limits<std::uint64_t>::max() - *report.codec.gpu_decode_operations) {
      errors.emplace_back("codec verification sample count overflowed");
    } else {
      const std::uint64_t expected_verification_samples =
          *report.codec.gpu_encode_operations + *report.codec.gpu_decode_operations;
      if (report.codec.verification_timing.sample_count != expected_verification_samples) {
        errors.emplace_back("codec verification timing samples do not match GPU codec operations");
      }
      if (expected_verification_samples != 0U &&
          !report.codec.verification_timing.total_ms.has_value()) {
        errors.emplace_back("GPU codec verification timing is missing");
      }
    }
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
