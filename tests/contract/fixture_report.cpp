#include "xvram/probe/report.hpp"

#include <iostream>
#include <optional>
#include <string>
#include <utility>

int main() {
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
  report.diagnostics.push_back(
      {DiagnosticLevel::info, "fixture", "generate", "synthetic report", 0});

  write_json(report, std::cout, false);
  return 0;
}
