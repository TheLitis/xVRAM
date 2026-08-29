#include "xvram/probe/report.hpp"
#include "xvram/vmm_poc/report.hpp"

#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace {

constexpr std::uint64_t mib = 1024ULL * 1024ULL;
constexpr std::uint64_t gib = 1024ULL * mib;

[[nodiscard]] xvram::vmm_poc::ModeResult completed_mode(const bool pipeline) {
  xvram::vmm_poc::ModeResult mode;
  mode.status = "completed";
  mode.elements_processed = 6ULL * 1024ULL * 1024ULL * 1024ULL;
  mode.tiles_processed = 384;
  mode.passes_completed = 2;
  mode.bytes_h2d = 24ULL * gib;
  mode.bytes_d2h = 24ULL * gib;
  mode.mapping_count = 384;
  mode.unmap_count = 384;
  mode.remap_count = pipeline ? 382 : 383;
  mode.event_boundary_count = 384;
  mode.unsafe_remap_count = 0;
  mode.physical_handle_count = pipeline ? 2 : 1;
  mode.max_concurrent_mappings = pipeline ? 2 : 1;
  mode.slot_count = pipeline ? 2 : 1;
  mode.set_access_count = 384;
  mode.handle_reuse_count = pipeline ? 382 : 383;
  mode.stable_addresses_verified = true;
  mode.full_verification_completed = true;
  mode.matches_cpu = true;
  mode.elapsed_ms = pipeline ? 3100.0 : 3400.0;
  mode.setup_ms = 25.0;
  mode.h2d_ms = 1200.0;
  mode.kernel_ms = 450.0;
  mode.d2h_ms = 1250.0;
  mode.verification_ms = 175.0;
  mode.remap_timing = pipeline ? xvram::vmm_poc::TimingSummary{382, 42.0, 0.05, 0.10, 0.18, 0.25}
                               : xvram::vmm_poc::TimingSummary{383, 44.0, 0.06, 0.11, 0.19, 0.27};
  mode.expected_digest128 = std::string(32, 'b');
  mode.output_digest128 = std::string(32, 'b');
  mode.mismatch_count = 0;
  return mode;
}

[[nodiscard]] xvram::vmm_poc::Report vmm_poc_report(const std::string_view scenario,
                                                    const bool include_identifiers = false) {
  using namespace xvram::vmm_poc;

  Report report;
  report.generated_at_utc = "2026-01-02T03:04:05Z";
  report.build = {"0.1.0-dev", "fixture", "fixture compiler", "Release", 13030};
  report.system = {"Fixture OS", "1.0", "x86_64", 8, 32ULL * gib, 24ULL * gib};
  report.device =
      DeviceInfo{0,          "Fixture GPU", std::nullopt, std::nullopt, std::nullopt, "wddm",
                 8ULL * gib, 6ULL * gib,    4ULL * gib,   7ULL * gib,   1ULL * gib,   6ULL * gib,
                 5ULL * gib, 6500ULL * mib, true,         2ULL * mib,   2ULL * mib};
  report.configuration.requested_device_ordinal = 0;
  report.configuration.requested_logical_bytes = 12ULL * gib;
  report.configuration.requested_chunk_bytes = 64ULL * mib;
  report.configuration.requested_window_slots = 2;
  report.configuration.effective_window_slots = 2;
  report.configuration.passes = 2;
  report.configuration.timeout_ms = 120'000;
  report.configuration.stall_timeout_ms = 5'000;
  report.configuration.seed_hex = "0123456789abcdef";
  report.configuration.host_headroom_bytes = 4ULL * gib;
  report.configuration.device_headroom_bytes = 512ULL * mib;
  report.configuration.sizing_mode = "explicit";
  report.configuration.mode = "compare";
  report.configuration.identifiers_included = include_identifiers;
  if (include_identifiers) {
    report.device->uuid = "GPU-00112233-4455-6677-8899-aabbccddeeff";
    report.device->luid = "0102030405060708";
    report.device->pci_bus_id = "00000000:65:00.0";
  }

  report.proof.effective_logical_bytes = 12ULL * gib;
  report.proof.effective_chunk_bytes = 64ULL * mib;
  report.proof.logical_element_count = 3ULL * 1024ULL * 1024ULL * 1024ULL;
  report.proof.logical_chunk_count = 192;
  report.proof.tile_visit_count = 384;
  report.proof.address_revisit_count = 192;
  report.proof.resident_physical_bytes = 128ULL * mib;
  report.proof.pinned_staging_bytes = 128ULL * mib;
  report.proof.kernel_module_version = 1;
  report.proof.kernel_module_sha256 = std::string(64, 'a');
  report.proof.pattern_version = "uint32-index-v1";
  report.proof.cpu_reference_digest128 = std::string(32, 'b');

  if (scenario == "skipped") {
    report.device.reset();
    report.proof = {};
    report.proof.kernel_module_version = 1;
    report.proof.kernel_module_sha256 = std::string(64, 'a');
    report.proof.pattern_version = "uint32-index-v1";
    report.outcome.status = "skipped";
    report.outcome.reason = "device_unavailable";
    report.outcome.exit_code = 23;
    report.outcome.stage = "planning";
    report.outcome.operation = "device_selection";
    report.outcome.message = "Fixture device is unavailable";
    report.cleanup = {true, true, true, true, true, true, true, true, true, true, true, true, true};
    report.diagnostics.push_back({xvram::probe::DiagnosticLevel::warning, "vmm_poc",
                                  "device_selection", "fixture device is unavailable", 23, 0});
    return report;
  }

  if (scenario == "oom") {
    report.outcome.status = "failed";
    report.outcome.reason = "host_oom";
    report.outcome.exit_code = 25;
    report.outcome.stage = "host_backing";
    report.outcome.operation = "allocate_pageable_backing";
    report.outcome.message = "Fixture host allocation failed";
    report.cleanup = {true, true, true, true, true, true, true, true, true, true, true, true, true};
    report.diagnostics.push_back({xvram::probe::DiagnosticLevel::error, "vmm_poc",
                                  "allocate_pageable_backing", "fixture host OOM", 25, 0});
    return report;
  }

  if (scenario == "timeout") {
    report.modes.pipeline.status = "timed_out";
    report.modes.pipeline.elements_processed = 16ULL * 1024ULL * 1024ULL;
    report.modes.pipeline.tiles_processed = 1;
    report.modes.pipeline.passes_completed = 0;
    report.modes.pipeline.mapping_count = 1;
    report.modes.pipeline.event_boundary_count = 0;
    report.modes.pipeline.physical_handle_count = 1;
    report.modes.pipeline.max_concurrent_mappings = 1;
    report.outcome.status = "failed";
    report.outcome.reason = "timeout";
    report.outcome.exit_code = 26;
    report.outcome.stage = "watchdog";
    report.outcome.operation = "worker_deadline";
    report.outcome.message = "Fixture worker exceeded its hard deadline";
    report.outcome.mode = "pipeline";
    report.outcome.pass_index = 0;
    report.outcome.tile_index = 0;
    report.cleanup.worker_terminated = true;
    report.diagnostics.push_back(
        {xvram::probe::DiagnosticLevel::error, "vmm_poc", "watchdog", "fixture timeout", 26, 0});
    return report;
  }

  report.modes.reference = completed_mode(false);
  report.modes.pipeline = completed_mode(true);
  report.modes.pipeline_speedup = 3400.0 / 3100.0;
  report.proof.handles_reused = true;
  report.proof.physical_window_smaller = true;
  report.proof.stable_virtual_addresses_verified = true;
  report.proof.event_boundaries_verified = true;
  report.proof.reference_matches_cpu = true;
  report.proof.pipeline_matches_cpu = true;
  report.proof.modes_match = true;
  report.outcome.status = "completed";
  report.outcome.reason.reset();
  report.outcome.exit_code = 0;
  report.outcome.message = "Fixture VMM proof completed";
  report.cleanup = {true, true, true, true, true, true, true, true, true, true, true, true, true};

  if (scenario == "corruption") {
    report.modes.pipeline.status = "failed";
    report.modes.pipeline.matches_cpu = false;
    report.modes.pipeline.output_digest128 = std::string(32, 'c');
    report.modes.pipeline.mismatch_count = 1;
    report.modes.pipeline.first_mismatch_byte_offset = 4096;
    report.proof.pipeline_matches_cpu = false;
    report.proof.modes_match = false;
    report.outcome.status = "failed";
    report.outcome.reason = "data_mismatch";
    report.outcome.exit_code = 24;
    report.outcome.stage = "verification";
    report.outcome.operation = "compare_tile";
    report.outcome.message = "Fixture CPU reference mismatch";
    report.outcome.mode = "pipeline";
    report.outcome.pass_index = 1;
    report.outcome.tile_index = 192;
    report.outcome.logical_byte_offset = 4096;
    report.diagnostics.push_back({xvram::probe::DiagnosticLevel::error, "vmm_poc", "compare_tile",
                                  "fixture data mismatch", 24, 0});
    return report;
  }

  if (scenario == "cleanup-failure") {
    report.outcome.status = "failed";
    report.outcome.reason = "cleanup_error";
    report.outcome.exit_code = 27;
    report.outcome.stage = "cleanup";
    report.outcome.operation = "cuMemUnmap";
    report.outcome.message = "Fixture cleanup failed";
    report.cleanup.complete = false;
    report.cleanup.mappings_removed = false;
    report.cleanup.virtual_reservation_released = false;
    report.diagnostics.push_back({xvram::probe::DiagnosticLevel::error, "vmm_poc", "cuMemUnmap",
                                  "fixture cleanup failure", 27, 0});
    return report;
  }

  report.diagnostics.push_back({xvram::probe::DiagnosticLevel::info, "vmm_poc", "fixture",
                                "synthetic VMM proof report", 0, 0});
  return report;
}

} // namespace

int main(const int argc, char** argv) {
  if (argc == 2) {
    const std::string_view mode = argv[1];
    if (mode == "vmm-poc-completed" || mode == "vmm-poc-skipped" || mode == "vmm-poc-corruption" ||
        mode == "vmm-poc-oom" || mode == "vmm-poc-timeout" || mode == "vmm-poc-cleanup-failure" ||
        mode == "vmm-poc-text" || mode == "vmm-poc-text-identifiers" ||
        mode == "vmm-poc-identifiers-included" || mode == "vmm-poc-identifiers-redacted") {
      std::string_view scenario = "completed";
      if (mode == "vmm-poc-skipped") {
        scenario = "skipped";
      } else if (mode == "vmm-poc-corruption") {
        scenario = "corruption";
      } else if (mode == "vmm-poc-oom") {
        scenario = "oom";
      } else if (mode == "vmm-poc-timeout") {
        scenario = "timeout";
      } else if (mode == "vmm-poc-cleanup-failure") {
        scenario = "cleanup-failure";
      }
      xvram::vmm_poc::Report report = vmm_poc_report(
          scenario, mode == "vmm-poc-text-identifiers" || mode == "vmm-poc-identifiers-included" ||
                        mode == "vmm-poc-identifiers-redacted");
      if (mode == "vmm-poc-identifiers-redacted") {
        report.configuration.identifiers_included = false;
      }
      if (mode == "vmm-poc-text" || mode == "vmm-poc-text-identifiers") {
        xvram::vmm_poc::write_text(report, std::cout);
      } else {
        xvram::vmm_poc::write_json(report, std::cout, false);
      }
      return 0;
    }
  }

  using namespace xvram::probe;

  ProbeReport report;
  report.generated_at_utc = "2026-01-02T03:04:05Z";
  report.build = {"0.1.0-dev", "fixture", "fixture compiler", "Release", 13030};
  report.system = {"Fixture OS",
                   "1.0",
                   "x86_64",
                   8,
                   16ULL * 1024ULL * 1024ULL * 1024ULL,
                   8ULL * 1024ULL * 1024ULL * 1024ULL};
  report.cuda.library_loaded = true;
  report.cuda.library_name = "fixture-driver";
  report.cuda.initialization_code = 0;
  report.cuda.initialization_name = "CUDA_SUCCESS";
  report.cuda.initialization_message = "no error";
  report.cuda.driver_version = DriverVersion{13030, 13, 3};
  report.cuda.nvml_library_loaded = true;
  report.cuda.display_driver_version = "999.99";

  DeviceReport device;
  device.ordinal = 0;
  device.name = "Fixture GPU";
  device.node_mask = 1;
  device.driver_model = "wddm";
  device.pending_driver_model = "wddm";
  device.total_memory_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
  device.free_memory_bytes = 6ULL * 1024ULL * 1024ULL * 1024ULL;

  for (const std::string key : {
           "can_map_host_memory",
           "compute_preemption",
           "concurrent_kernels",
           "concurrent_managed_access",
           "device_memory_pools",
           "direct_managed_access_from_host",
           "generic_compression",
           "gpudirect_rdma",
           "gpudirect_rdma_with_vmm",
           "host_memory_pools",
           "host_native_atomic",
           "host_numa_memory_pools",
           "host_numa_vmm",
           "host_register",
           "host_vmm",
           "managed_memory",
           "pageable_access_uses_host_page_tables",
           "pageable_memory_access",
           "same_host_registered_pointer",
           "unified_addressing",
           "virtual_memory_management",
           "win32_kmt_shareable_handle",
           "win32_shareable_handle",
       }) {
    device.capabilities.emplace(key, true);
  }

  std::int64_t attribute_value = 1;
  for (const std::string key : {
           "async_engine_count",
           "clock_rate_khz",
           "compute_capability_major",
           "compute_capability_minor",
           "compute_mode",
           "host_numa_id",
           "integrated",
           "l2_cache_bytes",
           "max_persisting_l2_cache_bytes",
           "max_threads_per_block",
           "memory_bus_width_bits",
           "memory_clock_rate_khz",
           "mempool_supported_handle_types",
           "multiprocessor_count",
           "tcc_driver",
           "warp_size",
       }) {
    device.attributes.emplace(key, attribute_value++);
  }

  device.allocation_granularity.emplace(
      "device", GranularityInfo{2ULL * 1024ULL * 1024ULL, 2ULL * 1024ULL * 1024ULL, std::nullopt});
  device.vmm_smoke = VmmSmokeResult{"completed", 2ULL * 1024ULL * 1024ULL,     true, true,
                                    true,        "fixture verification passed"};

  DxgiAdapterInfo dxgi;
  dxgi.name = "Fixture GPU";
  dxgi.node_index = 0;
  dxgi.dedicated_video_memory_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
  dxgi.shared_system_memory_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
  dxgi.local = VideoMemoryInfo{7ULL * 1024ULL * 1024ULL * 1024ULL, 1ULL * 1024ULL * 1024ULL,
                               2ULL * 1024ULL * 1024ULL, 0};
  dxgi.non_local =
      VideoMemoryInfo{8ULL * 1024ULL * 1024ULL * 1024ULL, 0, 4ULL * 1024ULL * 1024ULL * 1024ULL, 0};
  device.dxgi = dxgi;
  report.cuda.devices.push_back(std::move(device));

  report.transfer_benchmark = TransferMeasurement{
      "completed", 0, 64ULL * 1024ULL * 1024ULL, 20, 12.0, 11.0, 15.0, "fixture benchmark"};

  OverlapDirectionMeasurement h2d_overlap;
  h2d_overlap.copy_repetitions = 8;
  h2d_overlap.copy_only_ms = 40.0;
  h2d_overlap.concurrent_ms = 43.0;
  h2d_overlap.serial_ms = 80.0;
  h2d_overlap.ideal_ms = 40.0;
  h2d_overlap.speedup = 80.0 / 43.0;
  h2d_overlap.overlap_efficiency = 0.925;

  OverlapDirectionMeasurement d2h_overlap = h2d_overlap;
  d2h_overlap.copy_repetitions = 7;
  d2h_overlap.copy_only_ms = 39.0;
  d2h_overlap.concurrent_ms = 44.0;
  d2h_overlap.serial_ms = 79.0;
  d2h_overlap.speedup = 79.0 / 44.0;
  d2h_overlap.overlap_efficiency = 35.0 / 39.0;

  OverlapMeasurement overlap;
  overlap.status = "completed";
  overlap.device_ordinal = 0;
  overlap.module_version = 1;
  overlap.module_sha256 = std::string(64, 'a');
  overlap.bytes_per_copy = 64ULL * 1024ULL * 1024ULL;
  overlap.samples = 7;
  overlap.target_compute_ms = 40;
  overlap.grid_blocks = 184;
  overlap.block_threads = 256;
  overlap.kernel_iterations = 1'000'000;
  overlap.compute_only_ms = 40.0;
  overlap.h2d = h2d_overlap;
  overlap.d2h = d2h_overlap;
  overlap.compute_verified = true;
  overlap.transfer_verified = true;
  overlap.cleanup_complete = true;
  overlap.message = "fixture overlap benchmark";
  report.overlap_benchmark = std::move(overlap);
  report.diagnostics.push_back(
      {DiagnosticLevel::info, "fixture", "generate", "synthetic report", 0});

  write_json(report, std::cout, false);
  return 0;
}
