#include "xvram/compression/report.hpp"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

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
  report.configuration.effective_chunk_bytes = 64ULL * 1024ULL * 1024ULL;
  report.configuration.initial_cache_target_bytes = 7 * gib;
  report.configuration.host_store_cap_bytes = 16 * gib;

  xvram::compression::WorkloadResult workload;
  workload.scenario = "compressible-read";
  workload.compression_policy = "adaptive";
  workload.path = "cpu_lz4_gpu_decode";
  workload.replacement_policy = "clock";
  workload.status = "completed";
  workload.logical_bytes = 24 * gib;
  workload.logical_chunk_count = 384;
  workload.operations_retired = 4;
  workload.warmup_passes_completed = 2;
  workload.measurement_passes_completed = 5;
  workload.raw_input_bytes = 24 * gib;
  workload.stored_bytes = 6 * gib;
  workload.stored_ratio = 0.25;
  workload.logical_h2d_bytes = 24 * gib;
  workload.pcie_h2d_bytes = 6 * gib;
  workload.logical_d2h_bytes = 0;
  workload.pcie_d2h_bytes = 0;
  workload.raw_path_decisions = 0;
  workload.cpu_lz4_gpu_decisions = 4;
  workload.gpu_lz4_decisions = 0;
  workload.never_compress_decisions = 0;
  workload.fallback_count = 0;
  workload.mappings = 4;
  workload.set_access = 4;
  workload.unmaps = 4;
  workload.events_recorded = 8;
  workload.events_retired = 8;
  workload.unsafe_remaps = 0;
  workload.unsafe_transitions = 0;
  workload.elapsed_ms = 1000.0;
  workload.expected_digest128 = std::string(32, 'a');
  workload.output_digest128 = std::string(32, 'a');
  workload.mismatch_count = 0;
  report.workloads.push_back(workload);

  report.backing.logical_bytes = 24 * gib;
  report.backing.host_store_cap_bytes = 16 * gib;
  report.backing.host_bytes_current = 6 * gib;
  report.backing.host_bytes_peak = 8 * gib;
  report.backing.host_budget_bytes_current = 7 * gib;
  report.backing.host_budget_bytes_peak = 9 * gib;
  report.backing.raw_bytes_current = 0;
  report.backing.compressed_bytes_current = 6 * gib;
  report.backing.lz4_chunks = 384;
  report.backing.generations_created = 10;
  report.backing.generations_committed = 8;
  report.backing.generations_discarded = 2;
  report.backing.atomic_commit_failures = 0;
  report.telemetry.logical_h2d_bytes = 24 * gib;
  report.telemetry.pcie_h2d_bytes = 6 * gib;
  report.telemetry.pcie_h2d_payload_bytes = 6 * gib;
  report.telemetry.pcie_h2d_metadata_bytes = 0;
  report.telemetry.logical_d2h_bytes = 0;
  report.telemetry.pcie_d2h_bytes = 0;
  report.telemetry.pcie_d2h_payload_bytes = 0;
  report.telemetry.pcie_d2h_metadata_bytes = 0;
  report.telemetry.rejected_candidate_logical_d2h_bytes = 0;
  report.telemetry.mapping_count = 4;
  report.telemetry.set_access_count = 4;
  report.telemetry.unmap_count = 4;
  report.telemetry.event_record_count = 8;
  report.telemetry.event_retire_count = 8;
  report.telemetry.handle_reuse_count = 1;
  report.telemetry.unsafe_remap_count = 0;
  report.telemetry.unsafe_transition_count = 0;
  report.telemetry.cache_target_bytes_maximum = 7 * gib;
  report.telemetry.safe_device_budget_bytes_minimum = 6 * gib;
  report.telemetry.managed_device_bytes_peak = 6 * gib;
  report.telemetry.device_reserve_bytes_peak = gib;
  report.telemetry.device_budget_violation_count = 0;
  report.telemetry.trace_records_emitted = 0;
  report.telemetry.trace_records_dropped = 0;
  report.telemetry.trace_complete = true;
  report.codec.raw_path_decisions = 0;
  report.codec.cpu_lz4_gpu_decisions = 4;
  report.codec.gpu_lz4_decisions = 0;
  report.codec.cpu_encode_operations = 0;
  report.codec.cpu_decode_operations = 0;
  report.codec.gpu_encode_operations = 0;
  report.codec.gpu_decode_operations = 0;
  report.codec.never_compress_decisions = 0;
  report.codec.fallback_count = 0;
  report.codec.codec_slots_peak = 0;
  report.codec.workspace_bytes_peak = 0;
  report.codec.device_slot_bytes_peak = 0;
  report.codec.device_slot_capacity_bytes = 0;
  report.codec.verification_failures = 0;
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

  report.codec.codec_slots_peak = report.configuration.codec_slots + 1U;
  expect(!xvram::compression::validate_success_semantics(report).empty(),
         "codec slot count above the configured bound should fail semantics");
  report.codec.codec_slots_peak = 0;

  report.codec.device_slot_bytes_peak = 65ULL * 1024ULL * 1024ULL;
  report.codec.device_slot_capacity_bytes = 64ULL * 1024ULL * 1024ULL;
  expect(!xvram::compression::validate_success_semantics(report).empty(),
         "codec bytes above the runtime-derived capacity should fail semantics");
  report.codec.device_slot_bytes_peak = 0;
  report.codec.device_slot_capacity_bytes = 0;

  report.codec.verification_timing.sample_count = 1;
  expect(!xvram::compression::validate_success_semantics(report).empty(),
         "verification timing samples must reconcile with GPU codec operations");
  report.codec.verification_timing = {};

  report.telemetry.pcie_h2d_metadata_bytes = 1;
  expect(!xvram::compression::validate_success_semantics(report).empty(),
         "PCIe totals must reconcile with payload and metadata bytes");
  report.telemetry.pcie_h2d_metadata_bytes = 0;

  report.telemetry.set_access_count = 3;
  expect(!xvram::compression::validate_success_semantics(report).empty(),
         "counter mismatch should fail semantics");
}

void test_digest_parity_uses_comparable_evidence() {
  xvram::compression::Report single = completed_report();
  single.proof.path_digests_match = true;
  single.proof.policy_digests_match = true;
  single.workloads.front().output_digest128 = std::string(32, 'b');
  xvram::compression::finalize_proof(single);
  expect(!single.proof.path_digests_match.value_or(true),
         "single-case path parity must depend on the CPU reference digest");
  expect(!single.proof.policy_digests_match.value_or(true),
         "single-case policy parity must depend on the CPU reference digest");

  xvram::compression::Report paths = completed_report();
  xvram::compression::WorkloadResult second_path = paths.workloads.front();
  second_path.path = "nvcomp_gpu_codec";
  second_path.expected_digest128 = std::string(32, 'b');
  second_path.output_digest128 = std::string(32, 'b');
  paths.workloads.push_back(second_path);
  xvram::compression::finalize_proof(paths);
  expect(paths.proof.all_workloads_match_reference.value_or(false),
         "each comparable path may independently match its stated reference");
  expect(!paths.proof.path_digests_match.value_or(true),
         "different comparable path digests must fail path parity");

  xvram::compression::Report policies = completed_report();
  xvram::compression::WorkloadResult second_policy = policies.workloads.front();
  second_policy.replacement_policy = "lru";
  second_policy.expected_digest128 = std::string(32, 'c');
  second_policy.output_digest128 = std::string(32, 'c');
  policies.workloads.push_back(second_policy);
  xvram::compression::finalize_proof(policies);
  expect(!policies.proof.policy_digests_match.value_or(true),
         "different comparable policy digests must fail policy parity");
}

void test_codec_metadata_overhead_is_valid_transport() {
  xvram::compression::Report report = completed_report();
  xvram::compression::WorkloadResult& workload = report.workloads.front();
  workload.stored_ratio = 1.0;
  workload.pcie_h2d_bytes = workload.logical_h2d_bytes.value() + 4096U;
  workload.pcie_d2h_bytes = workload.logical_d2h_bytes.value() + 4096U;
  report.telemetry.pcie_h2d_bytes = workload.pcie_h2d_bytes;
  report.telemetry.pcie_d2h_bytes = workload.pcie_d2h_bytes;
  report.telemetry.pcie_h2d_payload_bytes = workload.logical_h2d_bytes;
  report.telemetry.pcie_h2d_metadata_bytes = 4096U;
  report.telemetry.pcie_d2h_payload_bytes = workload.logical_d2h_bytes;
  report.telemetry.pcie_d2h_metadata_bytes = 4096U;
  xvram::compression::finalize_proof(report);

  expect(!report.proof.compressed_h2d_reduction_verified.has_value(),
         "an incompressible codec attempt must not claim compressed H2D reduction");
  expect(xvram::compression::validate_success_semantics(report).empty(),
         "codec metadata may make physical transport exceed logical payload accounting");
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

  xvram::compression::Report unsafe = completed_report();
  unsafe.outcome.message = "device_pointer=0x12345678";
  xvram::compression::finalize_proof(unsafe);
  expect(!unsafe.proof.raw_virtual_addresses_omitted.value_or(true),
         "proof must reject pointer-bearing diagnostics");
  expect(!xvram::compression::validate_success_semantics(unsafe).empty(),
         "successful semantics must reject pointer-bearing diagnostics");
  std::ostringstream unsafe_json;
  xvram::compression::write_json(unsafe, unsafe_json, false);
  expect(unsafe_json.str().find("0x12345678") == std::string::npos,
         "report serializer must redact pointer-bearing text");

  trace.reason = "host_pointer=0xabcdef0123456789";
  std::ostringstream unsafe_trace;
  xvram::compression::write_trace_json(trace, unsafe_trace);
  expect(unsafe_trace.str().find("0xabcdef0123456789") == std::string::npos,
         "trace serializer must redact pointer-bearing text");
}

xvram::compression::TraceRecord completed_trace_record() {
  xvram::compression::TraceRecord trace;
  trace.sequence = 1;
  trace.monotonic_time_ns = 123;
  trace.event = "backing_transition";
  trace.allocation_id = 7;
  trace.chunk_index = 11;
  trace.operation_id = 19;
  trace.source_generation = 3;
  trace.target_generation = 4;
  trace.from_representation = "raw";
  trace.to_representation = "lz4_blocks";
  trace.path = "cpu_lz4_gpu_decode";
  trace.logical_bytes = 64ULL * 1024ULL * 1024ULL;
  trace.physical_bytes = 16ULL * 1024ULL * 1024ULL;
  trace.reason = "adaptive_cost_win";
  trace.speculative = false;
  return trace;
}

} // namespace

int main(const int argc, char** argv) {
  if (argc == 2) {
    const std::string_view mode(argv[1]);
    if (mode == "--emit-json-fixture") {
      const xvram::compression::Report report = completed_report();
      if (!xvram::compression::validate_success_semantics(report).empty()) {
        return EXIT_FAILURE;
      }
      xvram::compression::write_json(report, std::cout, false);
      return std::cout ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (mode == "--emit-trace-fixture") {
      xvram::compression::write_trace_json(completed_trace_record(), std::cout);
      return std::cout ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    return 64;
  }
  if (argc != 1) {
    return 64;
  }
  test_proof_and_semantics();
  test_digest_parity_uses_comparable_evidence();
  test_codec_metadata_overhead_is_valid_transport();
  test_serializers_omit_runtime_handles();
  if (failures != 0) {
    std::cerr << failures << " compression report test(s) failed\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
