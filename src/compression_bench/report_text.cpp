#include "xvram/compression/report.hpp"

#include "xvram/base/size_parser.hpp"

#include <iomanip>
#include <ostream>
#include <string_view>

namespace xvram::compression {
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

void write_ratio(std::ostream& output, const std::optional<double>& value) {
  if (value.has_value()) {
    output << std::fixed << std::setprecision(3) << *value;
  } else {
    output << "unavailable";
  }
}

void write_workload(std::ostream& output, const WorkloadResult& workload) {
  output << "  " << workload.compression_policy << '/' << workload.path << '/'
         << workload.replacement_policy << '/' << workload.scenario << ": " << workload.status
         << '\n';
  output << "    logical / stored / ratio: ";
  write_bytes(output, workload.raw_input_bytes);
  output << " / ";
  write_bytes(output, workload.stored_bytes);
  output << " / ";
  write_ratio(output, workload.stored_ratio);
  output << '\n';
  output << "    H2D logical / PCIe:       ";
  write_bytes(output, workload.logical_h2d_bytes);
  output << " / ";
  write_bytes(output, workload.pcie_h2d_bytes);
  output << '\n';
  output << "    D2H logical / PCIe:       ";
  write_bytes(output, workload.logical_d2h_bytes);
  output << " / ";
  write_bytes(output, workload.pcie_d2h_bytes);
  output << '\n';
  output << "    raw/cpu/gpu/never/fallback: ";
  write_count(output, workload.raw_path_decisions);
  output << '/';
  write_count(output, workload.cpu_lz4_gpu_decisions);
  output << '/';
  write_count(output, workload.gpu_lz4_decisions);
  output << '/';
  write_count(output, workload.never_compress_decisions);
  output << '/';
  write_count(output, workload.fallback_count);
  output << '\n';
  output << "    map/access/unmap:         ";
  write_count(output, workload.mappings);
  output << '/';
  write_count(output, workload.set_access);
  output << '/';
  write_count(output, workload.unmaps);
  output << '\n';
  output << "    events / unsafe r/t:      ";
  write_count(output, workload.events_recorded);
  output << '/';
  write_count(output, workload.events_retired);
  output << " / ";
  write_count(output, workload.unsafe_remaps);
  output << '/';
  write_count(output, workload.unsafe_transitions);
  output << '\n';
  if (workload.output_digest128.has_value()) {
    output << "    digest128:                 " << *workload.output_digest128;
    if (workload.expected_digest128.has_value()) {
      output << " (expected " << *workload.expected_digest128 << ')';
    }
    output << '\n';
  }
}

} // namespace

void write_text(const Report& report, std::ostream& output) {
  output << "xVRAM adaptive compression " << report.build.version << " (" << report.build.git_commit
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
    output << "VRAM:      " << format_bytes(report.device->total_memory_bytes) << '\n';
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
  output << "  Logical / chunk:            ";
  write_bytes(output, report.configuration.effective_logical_bytes);
  output << " / ";
  write_bytes(output, report.configuration.effective_chunk_bytes);
  output << '\n';
  output << "  Compression / path / codec: " << report.configuration.compression_policy << " / "
         << report.configuration.path << " / " << report.configuration.codec << '\n';
  output << "  Replacement / scenario:     " << report.configuration.replacement_policy << " / "
         << report.configuration.scenario << '\n';
  output << "  Codec slots / workers:       " << report.configuration.codec_slots << " / "
         << report.configuration.codec_workers << '\n';
  output << "  Host cap / scratch cap:      ";
  write_bytes(output, report.configuration.host_store_cap_bytes);
  output << " / " << format_bytes(report.configuration.compression_scratch_cap_bytes) << '\n';

  output << "\nWorkloads:\n";
  if (report.workloads.empty()) {
    output << "  none\n";
  }
  for (const WorkloadResult& workload : report.workloads) {
    write_workload(output, workload);
  }

  output << "\nBacking:\n";
  output << "  Logical / current / peak:    ";
  write_bytes(output, report.backing.logical_bytes);
  output << " / ";
  write_bytes(output, report.backing.host_bytes_current);
  output << " / ";
  write_bytes(output, report.backing.host_bytes_peak);
  output << '\n';
  output << "  Raw / compressed current:    ";
  write_bytes(output, report.backing.raw_bytes_current);
  output << " / ";
  write_bytes(output, report.backing.compressed_bytes_current);
  output << '\n';
  output << "  Generations c/c/d/f:         ";
  write_count(output, report.backing.generations_created);
  output << '/';
  write_count(output, report.backing.generations_committed);
  output << '/';
  write_count(output, report.backing.generations_discarded);
  output << '/';
  write_count(output, report.backing.atomic_commit_failures);
  output << '\n';

  output << "\nCodec:\n";
  output << "  Container / block:           " << report.codec.container_version << " / "
         << format_bytes(report.codec.block_bytes) << '\n';
  output << "  nvCOMP available / version:  " << bool_text(report.codec.nvcomp_available) << " / "
         << report.codec.nvcomp_version.value_or("unavailable") << '\n';
  output << "  CPU e/d / GPU e/d:           ";
  write_count(output, report.codec.cpu_encode_operations);
  output << '/';
  write_count(output, report.codec.cpu_decode_operations);
  output << " / ";
  write_count(output, report.codec.gpu_encode_operations);
  output << '/';
  write_count(output, report.codec.gpu_decode_operations);
  output << '\n';
  output << "  GPU verification samples/ms: ";
  output << report.codec.verification_timing.sample_count;
  output << " / ";
  if (report.codec.verification_timing.total_ms.has_value()) {
    output << *report.codec.verification_timing.total_ms;
  } else {
    output << "n/a";
  }
  output << '\n';

  output << "\nTransport:\n";
  output << "  H2D logical / PCIe:          ";
  write_bytes(output, report.telemetry.logical_h2d_bytes);
  output << " / ";
  write_bytes(output, report.telemetry.pcie_h2d_bytes);
  output << '\n';
  output << "  H2D payload / metadata:      ";
  write_bytes(output, report.telemetry.pcie_h2d_payload_bytes);
  output << " / ";
  write_bytes(output, report.telemetry.pcie_h2d_metadata_bytes);
  output << '\n';
  output << "  D2H logical / PCIe:          ";
  write_bytes(output, report.telemetry.logical_d2h_bytes);
  output << " / ";
  write_bytes(output, report.telemetry.pcie_d2h_bytes);
  output << '\n';
  output << "  D2H payload / metadata / rejected logical: ";
  write_bytes(output, report.telemetry.pcie_d2h_payload_bytes);
  output << " / ";
  write_bytes(output, report.telemetry.pcie_d2h_metadata_bytes);
  output << " / ";
  write_bytes(output, report.telemetry.rejected_candidate_logical_d2h_bytes);
  output << '\n';
  output << "  map/access/unmap/events:     ";
  write_count(output, report.telemetry.mapping_count);
  output << '/';
  write_count(output, report.telemetry.set_access_count);
  output << '/';
  write_count(output, report.telemetry.unmap_count);
  output << " / ";
  write_count(output, report.telemetry.event_record_count);
  output << '/';
  write_count(output, report.telemetry.event_retire_count);
  output << '\n';

  output << "\nProof:\n";
  output << "  Oversubscribed / cache smaller: "
         << bool_text(report.proof.logical_data_exceeds_vram) << '/'
         << bool_text(report.proof.cache_smaller_than_logical) << '\n';
  output << "  Compressed authority / no expansion: "
         << bool_text(report.proof.authoritative_compressed_backing_verified) << '/'
         << bool_text(report.proof.no_expansion_stored) << '\n';
  output << "  Atomic generations / write admission: "
         << bool_text(report.proof.generation_atomicity_verified) << '/'
         << bool_text(report.proof.write_admission_verified) << '\n';
  output << "  GPU encode before D2H / H2D reduction: "
         << bool_text(report.proof.gpu_encode_before_d2h_verified) << '/'
         << bool_text(report.proof.compressed_h2d_reduction_verified) << '\n';
  output << "  Stable VA / SetAccess / event-safe: "
         << bool_text(report.proof.stable_virtual_addresses_verified) << '/'
         << bool_text(report.proof.maps_match_set_access) << '/'
         << bool_text(report.proof.event_boundaries_verified) << '\n';
  output << "  Host/device budget / reference: " << bool_text(report.proof.host_budget_respected)
         << '/' << bool_text(report.proof.device_budget_respected) << '/'
         << bool_text(report.proof.all_workloads_match_reference) << '\n';
  output << "  Raw VA omitted:              "
         << bool_text(report.proof.raw_virtual_addresses_omitted) << '\n';

  output << "\nCleanup: " << bool_text(report.cleanup.complete) << '\n';
  output << "  operations/codec/events:     " << bool_text(report.cleanup.operations_drained) << '/'
         << bool_text(report.cleanup.codec_slots_drained) << '/'
         << bool_text(report.cleanup.events_drained) << '\n';
  output << "  mappings/handles/workspace:  " << bool_text(report.cleanup.mappings_removed) << '/'
         << bool_text(report.cleanup.physical_handles_released) << '/'
         << bool_text(report.cleanup.codec_workspace_released) << '\n';
  output << "  backing/context/worker:      " << bool_text(report.cleanup.host_backing_released)
         << '/' << bool_text(report.cleanup.context_released) << '/'
         << bool_text(report.cleanup.worker_terminated) << '\n';

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

} // namespace xvram::compression
