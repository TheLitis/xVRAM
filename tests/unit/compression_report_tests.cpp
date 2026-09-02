#include "xvram/compression/report.hpp"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

namespace {

int failures = 0;

void expect(const bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

xvram::compression::Report completed_report() {
  constexpr std::uint64_t gib = 1024ULL * 1024ULL * 1024ULL;
  xvram::compression::Report report;
  report.generated_at_utc = "2026-09-03T00:00:00Z";
  report.build = {"0.1.0-dev", "fixture", "fixture", "Release", 13030};
  report.system = {"Fixture OS", "1.0", "x86_64", 8, 64 * gib, 48 * gib};
  xvram::compression::DeviceInfo device;
  device.name = "Fixture GPU";
  device.total_memory_bytes = 8 * gib;
  device.vmm_supported = true;
  device.uva_supported = true;
  report.device = device;
  report.configuration.effective_logical_bytes = 24 * gib;
  report.configuration.initial_cache_target_bytes = 7 * gib;
  report.configuration.host_store_cap_bytes = 16 * gib;

  xvram::compression::WorkloadResult workload;
  workload.scenario = "compressible-read";
  workload.compression_policy = "adaptive";
  workload.path = "cpu_lz4_gpu_decode";
  workload.replacement_policy = "clock";
  workload.status = "completed";
  workload.mappings = 4;
  workload.set_access = 4;
  workload.unmaps = 4;
  workload.events_recorded = 8;
  workload.events_retired = 8;
  workload.unsafe_remaps = 0;
  workload.unsafe_transitions = 0;
  workload.expected_digest128 = std::string(32, 'a');
  workload.output_digest128 = std::string(32, 'a');
  workload.mismatch_count = 0;
  report.workloads.push_back(workload);

  report.backing.logical_bytes = 24 * gib;
  report.backing.host_store_cap_bytes = 16 * gib;
  report.backing.host_bytes_current = 6 * gib;
  report.backing.host_bytes_peak = 8 * gib;
  report.backing.compressed_bytes_current = 6 * gib;
  report.backing.lz4_chunks = 384;
  report.backing.generations_created = 10;
  report.backing.generations_committed = 8;
  report.backing.generations_discarded = 2;
  report.backing.atomic_commit_failures = 0;
  report.telemetry.logical_h2d_bytes = 24 * gib;
  report.telemetry.pcie_h2d_bytes = 6 * gib;
  report.telemetry.mapping_count = 4;
  report.telemetry.set_access_count = 4;
  report.telemetry.unmap_count = 4;
  report.telemetry.event_record_count = 8;
  report.telemetry.event_retire_count = 8;
  report.telemetry.unsafe_remap_count = 0;
  report.telemetry.unsafe_transition_count = 0;
  report.telemetry.cache_target_bytes_maximum = 7 * gib;
  report.proof.write_admission_verified = true;
  report.proof.gpu_encode_before_d2h_verified = true;
  report.proof.stable_virtual_addresses_verified = true;
  report.proof.path_digests_match = true;
  report.proof.policy_digests_match = true;
  report.outcome.status = "completed";
  report.outcome.exit_code = 0;

  report.cleanup.complete = true;
  report.cleanup.operations_drained = true;
  report.cleanup.codec_slots_drained = true;
  report.cleanup.events_drained = true;
  report.cleanup.events_destroyed = true;
  report.cleanup.streams_destroyed = true;
  report.cleanup.mappings_removed = true;
  report.cleanup.physical_handles_released = true;
  report.cleanup.codec_workspace_released = true;
  report.cleanup.virtual_reservations_released = true;
  report.cleanup.pinned_staging_released = true;
  report.cleanup.spill_reservations_released = true;
  report.cleanup.host_backing_released = true;
  report.cleanup.context_released = true;
  report.cleanup.trace_closed = true;
  report.cleanup.worker_terminated = true;
  xvram::compression::finalize_proof(report);
  return report;
}

void test_proof_and_semantics() {
  xvram::compression::Report report = completed_report();
  expect(report.proof.logical_data_exceeds_vram.value_or(false),
         "oversubscription should be derived");
  expect(report.proof.authoritative_compressed_backing_verified.value_or(false),
         "compressed authority should be derived");
  expect(report.proof.generation_atomicity_verified.value_or(false),
         "generation atomicity should be derived");
  expect(report.proof.compressed_h2d_reduction_verified.value_or(false),
         "H2D reduction should be derived");
  expect(xvram::compression::validate_success_semantics(report).empty(),
         "completed fixture should satisfy semantics");

  report.telemetry.set_access_count = 3;
  expect(!xvram::compression::validate_success_semantics(report).empty(),
         "counter mismatch should fail semantics");
}

void test_serializers_omit_runtime_handles() {
  const xvram::compression::Report report = completed_report();
  std::ostringstream json;
  xvram::compression::write_json(report, json, false);
  const std::string serialized = json.str();
  expect(serialized.find("cuda_va") == std::string::npos, "JSON must omit raw CUDA VA");
  expect(serialized.find("stream_handle") == std::string::npos, "JSON must omit stream handles");

  xvram::compression::TraceRecord trace;
  trace.sequence = 1;
  trace.event = "codec_decision";
  std::ostringstream trace_json;
  xvram::compression::write_trace_json(trace, trace_json);
  expect(trace_json.str().find("cuda_va") == std::string::npos, "trace must omit raw CUDA VA");
}

} // namespace

int main() {
  test_proof_and_semantics();
  test_serializers_omit_runtime_handles();
  if (failures != 0) {
    std::cerr << failures << " compression report test(s) failed\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
