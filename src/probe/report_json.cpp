#include "xvram/probe/report.hpp"

#include "xvram/base/json_writer.hpp"

#include <optional>
#include <ostream>
#include <string_view>

namespace xvram::probe {
namespace {

template <typename T> void optional_value(JsonWriter& writer, const std::optional<T>& value) {
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

[[nodiscard]] std::string_view level_name(const DiagnosticLevel level) {
  switch (level) {
  case DiagnosticLevel::info:
    return "info";
  case DiagnosticLevel::warning:
    return "warning";
  case DiagnosticLevel::error:
    return "error";
  }
  return "unknown";
}

void write_memory_info(JsonWriter& writer, const VideoMemoryInfo& info) {
  writer.begin_object();
  writer.key("budget_bytes");
  writer.value(info.budget_bytes);
  writer.key("current_usage_bytes");
  writer.value(info.current_usage_bytes);
  writer.key("available_for_reservation_bytes");
  writer.value(info.available_for_reservation_bytes);
  writer.key("current_reservation_bytes");
  writer.value(info.current_reservation_bytes);
  writer.end_object();
}

void write_dxgi(JsonWriter& writer, const DxgiAdapterInfo& adapter) {
  writer.begin_object();
  writer.key("name");
  writer.value(adapter.name);
  writer.key("luid");
  if (adapter.luid.empty()) {
    writer.null_value();
  } else {
    writer.value(adapter.luid);
  }
  writer.key("node_index");
  writer.value(static_cast<std::uint64_t>(adapter.node_index));
  writer.key("dedicated_video_memory_bytes");
  writer.value(adapter.dedicated_video_memory_bytes);
  writer.key("dedicated_system_memory_bytes");
  writer.value(adapter.dedicated_system_memory_bytes);
  writer.key("shared_system_memory_bytes");
  writer.value(adapter.shared_system_memory_bytes);
  writer.key("software_adapter");
  writer.value(adapter.software_adapter);
  writer.key("local");
  if (adapter.local.has_value()) {
    write_memory_info(writer, *adapter.local);
  } else {
    writer.null_value();
  }
  writer.key("non_local");
  if (adapter.non_local.has_value()) {
    write_memory_info(writer, *adapter.non_local);
  } else {
    writer.null_value();
  }
  writer.end_object();
}

void write_device(JsonWriter& writer, const DeviceReport& device) {
  writer.begin_object();
  writer.key("ordinal");
  writer.value(static_cast<std::int64_t>(device.ordinal));
  writer.key("name");
  writer.value(device.name);
  writer.key("uuid");
  optional_value(writer, device.uuid);
  writer.key("luid");
  optional_value(writer, device.luid);
  writer.key("node_mask");
  optional_value(writer, device.node_mask);
  writer.key("pci_bus_id");
  optional_value(writer, device.pci_bus_id);
  writer.key("driver_model");
  optional_value(writer, device.driver_model);
  writer.key("pending_driver_model");
  optional_value(writer, device.pending_driver_model);
  writer.key("total_memory_bytes");
  optional_value(writer, device.total_memory_bytes);
  writer.key("free_memory_bytes");
  optional_value(writer, device.free_memory_bytes);

  writer.key("capabilities");
  writer.begin_object();
  for (const auto& [name, value] : device.capabilities) {
    writer.key(name);
    optional_value(writer, value);
  }
  writer.end_object();

  writer.key("attributes");
  writer.begin_object();
  for (const auto& [name, value] : device.attributes) {
    writer.key(name);
    optional_value(writer, value);
  }
  writer.end_object();

  writer.key("allocation_granularity");
  writer.begin_object();
  for (const auto& [name, granularity] : device.allocation_granularity) {
    writer.key(name);
    writer.begin_object();
    writer.key("minimum_bytes");
    optional_value(writer, granularity.minimum_bytes);
    writer.key("recommended_bytes");
    optional_value(writer, granularity.recommended_bytes);
    writer.key("error");
    optional_value(writer, granularity.error);
    writer.end_object();
  }
  writer.end_object();

  writer.key("vmm_smoke");
  if (device.vmm_smoke.has_value()) {
    const VmmSmokeResult& smoke = *device.vmm_smoke;
    writer.begin_object();
    writer.key("status");
    writer.value(smoke.status);
    writer.key("allocation_bytes");
    writer.value(smoke.allocation_bytes);
    writer.key("initial_copy_verified");
    optional_value(writer, smoke.initial_copy_verified);
    writer.key("remap_copy_verified");
    optional_value(writer, smoke.remap_copy_verified);
    writer.key("cleanup_complete");
    optional_value(writer, smoke.cleanup_complete);
    writer.key("message");
    optional_value(writer, smoke.message);
    writer.end_object();
  } else {
    writer.null_value();
  }

  writer.key("dxgi");
  if (device.dxgi.has_value()) {
    write_dxgi(writer, *device.dxgi);
  } else {
    writer.null_value();
  }
  writer.end_object();
}

void write_transfer(JsonWriter& writer, const TransferMeasurement& measurement) {
  writer.begin_object();
  writer.key("status");
  writer.value(measurement.status);
  writer.key("device_ordinal");
  optional_value(writer, measurement.device_ordinal);
  writer.key("bytes_per_direction");
  writer.value(measurement.bytes_per_direction);
  writer.key("iterations");
  writer.value(static_cast<std::uint64_t>(measurement.iterations));
  writer.key("h2d_gib_per_second");
  optional_value(writer, measurement.h2d_gib_per_second);
  writer.key("d2h_gib_per_second");
  optional_value(writer, measurement.d2h_gib_per_second);
  writer.key("full_duplex_aggregate_gib_per_second");
  optional_value(writer, measurement.full_duplex_aggregate_gib_per_second);
  writer.key("message");
  optional_value(writer, measurement.message);
  writer.end_object();
}

} // namespace

void write_json(const ProbeReport& report, std::ostream& output, const bool pretty) {
  JsonWriter writer(output, pretty);
  writer.begin_object();
  writer.key("schema_version");
  writer.value(static_cast<std::uint64_t>(report.schema_version));
  writer.key("report_type");
  writer.value(report.report_type);
  writer.key("generated_at_utc");
  writer.value(report.generated_at_utc);

  writer.key("build");
  writer.begin_object();
  writer.key("version");
  writer.value(report.build.version);
  writer.key("git_commit");
  writer.value(report.build.git_commit);
  writer.key("compiler");
  writer.value(report.build.compiler);
  writer.key("build_type");
  writer.value(report.build.build_type);
  writer.key("cuda_headers_version");
  writer.value(static_cast<std::int64_t>(report.build.cuda_headers_version));
  writer.end_object();

  writer.key("system");
  writer.begin_object();
  writer.key("os_name");
  writer.value(report.system.os_name);
  writer.key("os_version");
  writer.value(report.system.os_version);
  writer.key("architecture");
  writer.value(report.system.architecture);
  writer.key("logical_processor_count");
  writer.value(static_cast<std::uint64_t>(report.system.logical_processor_count));
  writer.key("physical_memory_bytes");
  optional_value(writer, report.system.physical_memory_bytes);
  writer.key("available_memory_bytes");
  optional_value(writer, report.system.available_memory_bytes);
  writer.end_object();

  writer.key("cuda");
  writer.begin_object();
  writer.key("library_loaded");
  writer.value(report.cuda.library_loaded);
  writer.key("library_name");
  writer.value(report.cuda.library_name);
  writer.key("initialization");
  writer.begin_object();
  writer.key("code");
  optional_value(writer, report.cuda.initialization_code);
  writer.key("name");
  optional_value(writer, report.cuda.initialization_name);
  writer.key("message");
  optional_value(writer, report.cuda.initialization_message);
  writer.end_object();
  writer.key("driver_api_version");
  if (report.cuda.driver_version.has_value()) {
    writer.begin_object();
    writer.key("raw");
    writer.value(static_cast<std::int64_t>(report.cuda.driver_version->raw));
    writer.key("major");
    writer.value(static_cast<std::int64_t>(report.cuda.driver_version->major));
    writer.key("minor");
    writer.value(static_cast<std::int64_t>(report.cuda.driver_version->minor));
    writer.end_object();
  } else {
    writer.null_value();
  }
  writer.key("nvml_library_loaded");
  writer.value(report.cuda.nvml_library_loaded);
  writer.key("display_driver_version");
  optional_value(writer, report.cuda.display_driver_version);
  writer.key("devices");
  writer.begin_array();
  for (const DeviceReport& device : report.cuda.devices) {
    write_device(writer, device);
  }
  writer.end_array();
  writer.end_object();

  writer.key("transfer_benchmark");
  if (report.transfer_benchmark.has_value()) {
    write_transfer(writer, *report.transfer_benchmark);
  } else {
    writer.null_value();
  }

  writer.key("diagnostics");
  writer.begin_array();
  for (const Diagnostic& diagnostic : report.diagnostics) {
    writer.begin_object();
    writer.key("level");
    writer.value(level_name(diagnostic.level));
    writer.key("component");
    writer.value(diagnostic.component);
    writer.key("operation");
    writer.value(diagnostic.operation);
    writer.key("message");
    writer.value(diagnostic.message);
    writer.key("code");
    optional_value(writer, diagnostic.code);
    writer.key("device_ordinal");
    optional_value(writer, diagnostic.device_ordinal);
    writer.end_object();
  }
  writer.end_array();
  writer.end_object();
  output << '\n';
}

} // namespace xvram::probe
