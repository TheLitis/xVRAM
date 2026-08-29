#include "xvram/probe/report.hpp"

#include "xvram/base/size_parser.hpp"

#include <iomanip>
#include <ostream>
#include <string_view>

namespace xvram::probe {
namespace {

[[nodiscard]] std::string_view bool_text(const std::optional<bool>& value) {
  if (!value.has_value()) {
    return "unavailable";
  }
  return *value ? "yes" : "no";
}

[[nodiscard]] std::string_view diagnostic_text(const DiagnosticLevel level) {
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

void write_budget(std::ostream& output, const char* label,
                  const std::optional<VideoMemoryInfo>& memory) {
  output << "    " << label << ": ";
  if (!memory.has_value()) {
    output << "unavailable\n";
    return;
  }
  output << "budget " << format_bytes(memory->budget_bytes) << ", process usage "
         << format_bytes(memory->current_usage_bytes) << ", reservable "
         << format_bytes(memory->available_for_reservation_bytes) << '\n';
}

} // namespace

void write_text(const ProbeReport& report, std::ostream& output) {
  output << "xVRAM capability probe " << report.build.version << " (" << report.build.git_commit
         << ")\n";
  output << "Built with CUDA headers: " << report.build.cuda_headers_version << '\n';
  output << "Generated: " << report.generated_at_utc << '\n';
  output << "System:    " << report.system.os_name << ' ' << report.system.os_version << " / "
         << report.system.architecture << ", " << report.system.logical_processor_count
         << " logical CPUs";
  if (report.system.physical_memory_bytes.has_value()) {
    output << ", " << format_bytes(*report.system.physical_memory_bytes) << " RAM";
  }
  output << "\n\n";

  output << "CUDA driver: ";
  if (!report.cuda.library_loaded) {
    output << "not available\n";
  } else if (!report.cuda.driver_version.has_value()) {
    output << "loaded, version unavailable\n";
  } else {
    output << report.cuda.driver_version->major << '.' << report.cuda.driver_version->minor
           << " (raw " << report.cuda.driver_version->raw << ")\n";
  }
  if (report.cuda.display_driver_version.has_value()) {
    output << "NVIDIA display driver: " << *report.cuda.display_driver_version << '\n';
  }

  if (report.cuda.devices.empty()) {
    output << "CUDA devices: none selected or available\n";
  }

  for (const DeviceReport& device : report.cuda.devices) {
    output << "\nGPU " << device.ordinal << ": " << device.name << '\n';
    output << "  PCI:     " << device.pci_bus_id.value_or("unavailable") << '\n';
    output << "  Driver:  " << device.driver_model.value_or("unknown");
    if (device.pending_driver_model.has_value() &&
        device.pending_driver_model != device.driver_model) {
      output << " (pending " << *device.pending_driver_model << ')';
    }
    output << '\n';
    output << "  Memory:  ";
    if (device.free_memory_bytes.has_value() && device.total_memory_bytes.has_value()) {
      output << format_bytes(*device.free_memory_bytes) << " free / "
             << format_bytes(*device.total_memory_bytes) << " total\n";
    } else if (device.total_memory_bytes.has_value()) {
      output << format_bytes(*device.total_memory_bytes) << " total\n";
    } else {
      output << "unavailable\n";
    }

    const auto major = device.attributes.find("compute_capability_major");
    const auto minor = device.attributes.find("compute_capability_minor");
    if (major != device.attributes.end() && minor != device.attributes.end() &&
        major->second.has_value() && minor->second.has_value()) {
      output << "  Compute: " << *major->second << '.' << *minor->second << '\n';
    }

    output << "  Capabilities:\n";
    for (const auto& [name, value] : device.capabilities) {
      output << "    " << std::left << std::setw(39) << name << bool_text(value) << '\n';
    }

    if (!device.allocation_granularity.empty()) {
      output << "  VMM allocation granularity:\n";
      for (const auto& [name, granularity] : device.allocation_granularity) {
        output << "    " << std::left << std::setw(14) << name;
        if (granularity.minimum_bytes.has_value()) {
          output << "minimum " << format_bytes(*granularity.minimum_bytes);
        } else {
          output << "minimum unavailable";
        }
        if (granularity.recommended_bytes.has_value()) {
          output << ", recommended " << format_bytes(*granularity.recommended_bytes);
        }
        if (granularity.error.has_value()) {
          output << " (" << *granularity.error << ')';
        }
        output << '\n';
      }
    }

    if (device.vmm_smoke.has_value()) {
      output << "  VMM map/remap smoke: " << device.vmm_smoke->status;
      if (device.vmm_smoke->allocation_bytes != 0) {
        output << " (" << format_bytes(device.vmm_smoke->allocation_bytes) << ')';
      }
      output << '\n';
      if (device.vmm_smoke->message.has_value()) {
        output << "    " << *device.vmm_smoke->message << '\n';
      }
    }

    if (device.dxgi.has_value()) {
      output << "  DXGI/WDDM adapter: " << device.dxgi->name << " (node " << device.dxgi->node_index
             << ")\n";
      write_budget(output, "local", device.dxgi->local);
      write_budget(output, "non-local", device.dxgi->non_local);
    } else {
      output << "  DXGI/WDDM adapter: unavailable\n";
    }
  }

  if (report.transfer_benchmark.has_value()) {
    const TransferMeasurement& benchmark = *report.transfer_benchmark;
    output << "\nPinned transfer benchmark: " << benchmark.status << '\n';
    if (benchmark.device_ordinal.has_value()) {
      output << "  GPU ordinal:      " << *benchmark.device_ordinal << '\n';
    }
    if (benchmark.h2d_gib_per_second.has_value()) {
      output << "  H2D:              " << std::fixed << std::setprecision(2)
             << *benchmark.h2d_gib_per_second << " GiB/s\n";
    }
    if (benchmark.d2h_gib_per_second.has_value()) {
      output << "  D2H:              " << std::fixed << std::setprecision(2)
             << *benchmark.d2h_gib_per_second << " GiB/s\n";
    }
    if (benchmark.full_duplex_aggregate_gib_per_second.has_value()) {
      output << "  Full duplex total:" << std::fixed << std::setprecision(2) << ' '
             << *benchmark.full_duplex_aggregate_gib_per_second << " GiB/s\n";
    }
    if (benchmark.message.has_value()) {
      output << "  " << *benchmark.message << '\n';
    }
  }

  if (!report.diagnostics.empty()) {
    output << "\nDiagnostics:\n";
    for (const Diagnostic& diagnostic : report.diagnostics) {
      output << "  [" << diagnostic_text(diagnostic.level) << "] " << diagnostic.component << '/'
             << diagnostic.operation << ": " << diagnostic.message;
      if (diagnostic.device_ordinal.has_value()) {
        output << " [GPU " << *diagnostic.device_ordinal << ']';
      }
      if (diagnostic.code.has_value()) {
        output << " (code " << *diagnostic.code << ')';
      }
      output << '\n';
    }
  }
}

} // namespace xvram::probe
