#include "xvram/vmm_poc/report.hpp"

#include "xvram/base/size_parser.hpp"

#include <iomanip>
#include <ostream>
#include <string_view>

namespace xvram::vmm_poc {
namespace {

[[nodiscard]] std::string_view bool_text(const std::optional<bool>& value) {
  if (!value.has_value()) {
    return "unavailable";
  }
  return *value ? "yes" : "no";
}

[[nodiscard]] std::string_view diagnostic_text(const probe::DiagnosticLevel level) {
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

void write_bytes(std::ostream& output, const std::optional<std::uint64_t>& value) {
  if (value.has_value()) {
    output << format_bytes(*value);
  } else {
    output << "unavailable";
  }
}

void write_count(std::ostream& output, const std::optional<std::uint64_t>& value) {
  if (value.has_value()) {
    output << *value;
  } else {
    output << "unavailable";
  }
}

void write_mode(std::ostream& output, const char* label, const ModeResult& mode) {
  output << "  " << label << ": " << mode.status << '\n';
  output << "    tiles / passes:       ";
  write_count(output, mode.tiles_processed);
  output << " / ";
  if (mode.passes_completed.has_value()) {
    output << *mode.passes_completed;
  } else {
    output << "unavailable";
  }
  output << '\n';
  output << "    maps/unmaps/remaps:   ";
  write_count(output, mode.mapping_count);
  output << '/';
  write_count(output, mode.unmap_count);
  output << '/';
  write_count(output, mode.remap_count);
  output << '\n';
  output << "    slots/access/reuses:  ";
  if (mode.slot_count.has_value()) {
    output << *mode.slot_count;
  } else {
    output << "unavailable";
  }
  output << '/';
  write_count(output, mode.set_access_count);
  output << '/';
  write_count(output, mode.handle_reuse_count);
  output << '\n';
  output << "    stable/full/matches:  " << bool_text(mode.stable_addresses_verified) << '/'
         << bool_text(mode.full_verification_completed) << '/' << bool_text(mode.matches_cpu)
         << '\n';
  output << "    mismatches:           ";
  write_count(output, mode.mismatch_count);
  output << '\n';
  if (mode.output_digest128.has_value()) {
    output << "    digest128:            " << *mode.output_digest128;
    if (mode.expected_digest128.has_value()) {
      output << " (expected " << *mode.expected_digest128 << ')';
    }
    output << '\n';
  }
  if (mode.elapsed_ms.has_value()) {
    output << "    elapsed:              " << std::fixed << std::setprecision(2) << *mode.elapsed_ms
           << " ms\n";
  }
  if (mode.remap_timing.total_ms.has_value()) {
    output << "    remap:                " << std::fixed << std::setprecision(2)
           << *mode.remap_timing.total_ms << " ms / " << mode.remap_timing.sample_count
           << " samples\n";
  }
}

} // namespace

void write_text(const Report& report, std::ostream& output) {
  output << "xVRAM VMM proof of concept " << report.build.version << " (" << report.build.git_commit
         << ")\n";
  output << "Generated: " << report.generated_at_utc << '\n';
  output << "Outcome:   " << report.outcome.status;
  if (report.outcome.reason.has_value()) {
    output << " (" << *report.outcome.reason << ')';
  }
  output << ", exit " << report.outcome.exit_code << '\n';
  if (report.outcome.message.has_value()) {
    output << "Message:   " << *report.outcome.message << '\n';
  }
  output << "System:    " << report.system.os_name << ' ' << report.system.os_version << " / "
         << report.system.architecture << '\n';
  if (report.device.has_value()) {
    output << "GPU:       " << report.device->ordinal << ": " << report.device->name << '\n';
    output << "VRAM:      " << format_bytes(report.device->free_memory_bytes_start) << " free / "
           << format_bytes(report.device->total_memory_bytes) << " total\n";
    if (report.configuration.identifiers_included) {
      if (report.device->uuid.has_value()) {
        output << "GPU UUID:  " << *report.device->uuid << '\n';
      }
      if (report.device->luid.has_value()) {
        output << "GPU LUID:  " << *report.device->luid << '\n';
      }
      if (report.device->pci_bus_id.has_value()) {
        output << "GPU PCI:   " << *report.device->pci_bus_id << '\n';
      }
    }
  } else {
    output << "GPU:       unavailable\n";
  }

  output << "\nConfiguration:\n";
  output << "  Requested logical:      ";
  write_bytes(output, report.configuration.requested_logical_bytes);
  output << '\n';
  output << "  Requested chunk:        ";
  write_bytes(output, report.configuration.requested_chunk_bytes);
  output << '\n';
  output << "  Passes / timeout:       " << report.configuration.passes << " / "
         << report.configuration.timeout_ms << " ms\n";
  output << "  Window slots / stall:   ";
  if (report.configuration.effective_window_slots.has_value()) {
    output << *report.configuration.effective_window_slots;
  } else {
    output << "unavailable";
  }
  output << " / " << report.configuration.stall_timeout_ms << " ms\n";
  output << "  Mode:                   " << report.configuration.mode << '\n';

  output << "\nModes: " << report.modes.backing << ", " << report.modes.traversal << ", "
         << report.modes.verification << ", " << report.modes.timeout_enforcement << '\n';
  write_mode(output, "reference", report.modes.reference);
  write_mode(output, "pipeline", report.modes.pipeline);
  if (report.modes.pipeline_speedup.has_value()) {
    output << "  pipeline speedup: " << std::fixed << std::setprecision(3)
           << *report.modes.pipeline_speedup << "x\n";
  }

  output << "\nProof:\n";
  output << "  Effective logical:      ";
  write_bytes(output, report.proof.effective_logical_bytes);
  output << '\n';
  output << "  Effective chunk:        ";
  write_bytes(output, report.proof.effective_chunk_bytes);
  output << '\n';
  output << "  Handles reused:         " << bool_text(report.proof.handles_reused) << '\n';
  output << "  Physical window smaller:" << ' ' << bool_text(report.proof.physical_window_smaller)
         << '\n';
  output << "  Stable addresses:       "
         << bool_text(report.proof.stable_virtual_addresses_verified) << '\n';
  output << "  Event boundaries:       " << bool_text(report.proof.event_boundaries_verified)
         << '\n';
  output << "  Reference matches CPU:  " << bool_text(report.proof.reference_matches_cpu) << '\n';
  output << "  Pipeline matches CPU:   " << bool_text(report.proof.pipeline_matches_cpu) << '\n';
  output << "  Modes match:            " << bool_text(report.proof.modes_match) << '\n';

  output << "\nCleanup: " << bool_text(report.cleanup.complete) << '\n';
  output << "  drained/events/streams/module: " << bool_text(report.cleanup.events_drained) << '/'
         << bool_text(report.cleanup.events_destroyed) << '/'
         << bool_text(report.cleanup.streams_destroyed) << '/'
         << bool_text(report.cleanup.module_unloaded) << '\n';
  output << "  allocations/mappings/handle:   "
         << bool_text(report.cleanup.device_allocations_released) << '/'
         << bool_text(report.cleanup.mappings_removed) << '/'
         << bool_text(report.cleanup.physical_handle_released) << '\n';
  output << "  reservation/staging/backing:   "
         << bool_text(report.cleanup.virtual_reservation_released) << '/'
         << bool_text(report.cleanup.pinned_staging_released) << '/'
         << bool_text(report.cleanup.host_backing_released) << '\n';
  output << "  context/worker:                " << bool_text(report.cleanup.context_destroyed)
         << '/' << bool_text(report.cleanup.worker_terminated) << '\n';

  if (!report.diagnostics.empty()) {
    output << "\nDiagnostics:\n";
    for (const probe::Diagnostic& diagnostic : report.diagnostics) {
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

} // namespace xvram::vmm_poc
