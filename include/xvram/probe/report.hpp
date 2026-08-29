#pragma once

#include <cstdint>
#include <iosfwd>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace xvram::probe {

enum class DiagnosticLevel { info, warning, error };

struct Diagnostic {
  Diagnostic() = default;
  Diagnostic(DiagnosticLevel diagnostic_level, std::string diagnostic_component,
             std::string diagnostic_operation, std::string diagnostic_message,
             std::optional<std::int64_t> native_code,
             std::optional<std::int32_t> cuda_device_ordinal = std::nullopt)
      : level(diagnostic_level), component(std::move(diagnostic_component)),
        operation(std::move(diagnostic_operation)), message(std::move(diagnostic_message)),
        code(native_code), device_ordinal(cuda_device_ordinal) {}

  DiagnosticLevel level = DiagnosticLevel::info;
  std::string component;
  std::string operation;
  std::string message;
  std::optional<std::int64_t> code;
  std::optional<std::int32_t> device_ordinal;
};

struct BuildInfo {
  std::string version;
  std::string git_commit;
  std::string compiler;
  std::string build_type;
  std::int32_t cuda_headers_version = 0;
};

struct SystemInfo {
  std::string os_name;
  std::string os_version;
  std::string architecture;
  std::uint32_t logical_processor_count = 0;
  std::optional<std::uint64_t> physical_memory_bytes;
  std::optional<std::uint64_t> available_memory_bytes;
};

struct VideoMemoryInfo {
  std::uint64_t budget_bytes = 0;
  std::uint64_t current_usage_bytes = 0;
  std::uint64_t available_for_reservation_bytes = 0;
  std::uint64_t current_reservation_bytes = 0;
};

struct DxgiAdapterInfo {
  std::string name;
  std::string luid;
  std::uint32_t node_index = 0;
  std::uint64_t dedicated_video_memory_bytes = 0;
  std::uint64_t dedicated_system_memory_bytes = 0;
  std::uint64_t shared_system_memory_bytes = 0;
  bool software_adapter = false;
  std::optional<VideoMemoryInfo> local;
  std::optional<VideoMemoryInfo> non_local;
};

struct GranularityInfo {
  std::optional<std::uint64_t> minimum_bytes;
  std::optional<std::uint64_t> recommended_bytes;
  std::optional<std::string> error;
};

struct DriverVersion {
  std::int32_t raw = 0;
  std::int32_t major = 0;
  std::int32_t minor = 0;
};

struct VmmSmokeResult {
  std::string status = "not_requested";
  std::uint64_t allocation_bytes = 0;
  std::optional<bool> initial_copy_verified;
  std::optional<bool> remap_copy_verified;
  std::optional<bool> cleanup_complete;
  std::optional<std::string> message;
};

struct DeviceReport {
  std::int32_t ordinal = 0;
  std::string name;
  std::optional<std::string> uuid;
  std::optional<std::string> luid;
  std::optional<std::uint32_t> node_mask;
  std::optional<std::string> pci_bus_id;
  std::optional<std::string> driver_model;
  std::optional<std::string> pending_driver_model;
  std::optional<std::uint64_t> total_memory_bytes;
  std::optional<std::uint64_t> free_memory_bytes;
  std::map<std::string, std::optional<bool>> capabilities;
  std::map<std::string, std::optional<std::int64_t>> attributes;
  std::map<std::string, GranularityInfo> allocation_granularity;
  std::optional<VmmSmokeResult> vmm_smoke;
  std::optional<DxgiAdapterInfo> dxgi;
};

struct CudaReport {
  bool library_loaded = false;
  std::string library_name;
  std::optional<std::int32_t> initialization_code;
  std::optional<std::string> initialization_name;
  std::optional<std::string> initialization_message;
  std::optional<DriverVersion> driver_version;
  bool nvml_library_loaded = false;
  std::optional<std::string> display_driver_version;
  std::vector<DeviceReport> devices;
};

struct TransferMeasurement {
  std::string status = "not_requested";
  std::optional<std::int32_t> device_ordinal;
  std::uint64_t bytes_per_direction = 0;
  std::uint32_t iterations = 0;
  std::optional<double> h2d_gib_per_second;
  std::optional<double> d2h_gib_per_second;
  std::optional<double> full_duplex_aggregate_gib_per_second;
  std::optional<std::string> message;
};

struct ProbeReport {
  std::uint32_t schema_version = 1;
  std::string report_type = "xvram.capability_probe";
  std::string generated_at_utc;
  BuildInfo build;
  SystemInfo system;
  CudaReport cuda;
  std::optional<TransferMeasurement> transfer_benchmark;
  std::vector<Diagnostic> diagnostics;
};

struct ProbeOptions {
  std::optional<std::int32_t> device_ordinal;
  bool run_transfer_benchmark = false;
  bool run_vmm_smoke = true;
  bool include_stable_identifiers = false;
  std::uint64_t benchmark_bytes = 64ULL * 1024ULL * 1024ULL;
  std::uint32_t benchmark_iterations = 20;
};

[[nodiscard]] ProbeReport collect(const ProbeOptions& options);
void write_json(const ProbeReport& report, std::ostream& output, bool pretty);
void write_text(const ProbeReport& report, std::ostream& output);

} // namespace xvram::probe
