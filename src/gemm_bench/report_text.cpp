#include "xvram/gemm/report.hpp"

#include "xvram/base/size_parser.hpp"

#include <iomanip>
#include <ostream>
#include <string_view>

namespace xvram::gemm_bench {
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

void write_number(std::ostream& output, const std::optional<double>& value,
                  const int precision = 3) {
  if (value.has_value()) {
    output << std::fixed << std::setprecision(precision) << *value;
  } else {
    output << "unavailable";
  }
}

} // namespace

void write_text(const Report& report, std::ostream& output) {
  output << "xVRAM tiled GEMM benchmark " << report.build.version << " (" << report.build.git_commit
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
    output << "VRAM:      " << format_bytes(report.device->total_memory_bytes) << " total";
    if (report.device->free_memory_bytes_start.has_value()) {
      output << ", " << format_bytes(*report.device->free_memory_bytes_start) << " free at start";
    }
    output << '\n';
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
  output << "  Shape M/N/K:              ";
  write_count(output, report.configuration.requested_m);
  output << '/';
  write_count(output, report.configuration.requested_n);
  output << '/';
  write_count(output, report.configuration.requested_k);
  output << '\n';
  output << "  A/B/C data types:         " << report.configuration.a_data_type << '/'
         << report.configuration.b_data_type << '/' << report.configuration.c_data_type << '\n';
  output << "  Compute / context:        " << report.configuration.compute_mode << " / "
         << report.configuration.context_mode << '\n';
  output << "  A/B/C layouts:            " << report.configuration.a_layout << '/'
         << report.configuration.b_layout << '/' << report.configuration.c_layout << '\n';
  output << "  A/B operations:           " << report.configuration.a_operation << '/'
         << report.configuration.b_operation << '\n';
  output << "  Chunk / cache / workspace:" << ' ';
  write_bytes(output, report.configuration.effective_chunk_bytes);
  output << " / ";
  write_bytes(output, report.configuration.initial_cache_target_bytes);
  output << " / ";
  write_bytes(output, report.configuration.effective_workspace_cap_bytes);
  output << '\n';
  output << "  Policy / prefetch:        " << report.configuration.policy << " / "
         << report.configuration.prefetch_distance << '\n';
  output << "  Passes / timeout:         " << report.configuration.passes << " / "
         << report.configuration.timeout_ms << " ms\n";

  output << "\nPlans:\n";
  if (report.plans.empty()) {
    output << "  none\n";
  }
  for (const PlanResult& plan : report.plans) {
    output << "  #" << plan.plan_id << ' ' << plan.m << 'x' << plan.n << 'x' << plan.k << ": "
           << plan.status << '\n';
    output << "    tile / count:            " << plan.tile_m << 'x' << plan.tile_n << 'x'
           << plan.tile_k << " / " << plan.tile_count << '\n';
    output << "    working set / workspace: " << format_bytes(plan.maximum_working_set_bytes)
           << " / " << format_bytes(plan.workspace_bytes) << '\n';
    output << "    compute / cuBLASLt:      " << plan.effective_compute_mode << " / "
           << (plan.uses_cublas_lt ? "yes" : "no") << '\n';
  }

  output << "\nWorkloads:\n";
  if (report.workloads.empty()) {
    output << "  none\n";
  }
  for (const WorkloadResult& workload : report.workloads) {
    output << "  " << workload.name << " (#" << workload.plan_id << "): " << workload.status
           << '\n';
    output << "    tiles retired / total:   " << workload.tiles_retired << " / "
           << workload.tiles_total << '\n';
    output << "    logical allocation:      " << format_bytes(workload.logical_allocation_bytes)
           << (workload.oversubscribed ? " (oversubscribed)" : "") << '\n';
    output << "    H2D / D2H:               ";
    write_bytes(output, workload.bytes_h2d);
    output << " / ";
    write_bytes(output, workload.bytes_d2h);
    output << '\n';
    output << "    hits / misses / ratio:   ";
    write_count(output, workload.cache_hits);
    output << " / ";
    write_count(output, workload.cache_misses);
    output << " / ";
    write_number(output, workload.cache_hit_rate);
    output << '\n';
    output << "    elapsed / TFLOP/s:       ";
    write_number(output, workload.elapsed_ms, 2);
    output << " ms / ";
    write_number(output, workload.achieved_tflops, 3);
    output << '\n';
  }

  output << "\nNumerics:\n";
  output << "  Mode / reference:         " << report.numerics.validation_mode << " / "
         << report.numerics.reference_precision << '\n';
  output << "  Absolute / relative max:  ";
  write_number(output, report.numerics.maximum_absolute_error, 8);
  output << " / ";
  write_number(output, report.numerics.maximum_relative_error, 8);
  output << '\n';
  output << "  Verified / mismatches:    ";
  write_count(output, report.numerics.elements_verified);
  output << " / ";
  write_count(output, report.numerics.mismatch_count);
  output << '\n';
  output << "  All results match:        " << bool_text(report.numerics.all_results_match) << '\n';

  output << "\nCache totals:\n";
  output << "  Target initial/min/max/end:" << ' ';
  write_bytes(output, report.cache.target_bytes_initial);
  output << " / ";
  write_bytes(output, report.cache.target_bytes_minimum);
  output << " / ";
  write_bytes(output, report.cache.target_bytes_maximum);
  output << " / ";
  write_bytes(output, report.cache.target_bytes_end);
  output << '\n';
  output << "  Resident/staging/workspace:" << ' ';
  write_bytes(output, report.cache.resident_bytes_peak);
  output << " / ";
  write_bytes(output, report.cache.pinned_staging_bytes);
  output << " / ";
  write_bytes(output, report.cache.workspace_bytes_peak);
  output << '\n';
  output << "  Map/unmap/access/reuse:   ";
  write_count(output, report.cache.mapping_count);
  output << '/';
  write_count(output, report.cache.unmap_count);
  output << '/';
  write_count(output, report.cache.set_access_count);
  output << '/';
  write_count(output, report.cache.handle_reuse_count);
  output << '\n';

  output << "\nLibraries:\n";
  output << "  CUDA / cuBLAS / cuBLASLt: ";
  if (report.telemetry.cuda_driver_version.has_value()) {
    output << *report.telemetry.cuda_driver_version;
  } else {
    output << "unavailable";
  }
  output << " / ";
  if (report.telemetry.cublas_version.has_value()) {
    output << *report.telemetry.cublas_version;
  } else {
    output << "unavailable";
  }
  output << " / " << bool_text(report.telemetry.cublas_lt_available) << '\n';

  output << "\nProof:\n";
  output << "  Logical exceeds VRAM:     " << bool_text(report.proof.logical_data_exceeds_vram)
         << '\n';
  output << "  Cache smaller / tiled:    " << bool_text(report.proof.cache_smaller_than_logical)
         << '/' << bool_text(report.proof.tiled_execution_verified) << '\n';
  output << "  Stable VA / SetAccess:    "
         << bool_text(report.proof.stable_virtual_addresses_verified) << '/'
         << bool_text(report.proof.set_access_after_map_verified) << '\n';
  output << "  Events / no aliases:      " << bool_text(report.proof.event_boundaries_verified)
         << '/' << bool_text(report.proof.no_physical_aliases_verified) << '\n';
  output << "  Handles / writeback:      " << bool_text(report.proof.handles_reused) << '/'
         << bool_text(report.proof.dirty_writeback_verified) << '\n';
  output << "  Target / workspace:       " << bool_text(report.proof.cache_target_respected) << '/'
         << bool_text(report.proof.workspace_bounded) << '\n';
  output << "  Reference / raw VA omitted: "
         << bool_text(report.proof.all_workloads_match_reference) << '/'
         << bool_text(report.proof.raw_virtual_addresses_omitted) << '\n';

  output << "\nCleanup: " << bool_text(report.cleanup.complete) << '\n';
  output << "  operations/events/streams: " << bool_text(report.cleanup.operations_drained) << '/'
         << bool_text(report.cleanup.events_destroyed) << '/'
         << bool_text(report.cleanup.streams_destroyed) << '\n';
  output << "  cuBLAS/mappings/handles:    " << bool_text(report.cleanup.cublas_handles_destroyed)
         << '/' << bool_text(report.cleanup.mappings_removed) << '/'
         << bool_text(report.cleanup.physical_handles_released) << '\n';
  output << "  workspace/backing/context:  " << bool_text(report.cleanup.workspace_released) << '/'
         << bool_text(report.cleanup.host_backing_released) << '/'
         << bool_text(report.cleanup.context_released) << '\n';
  output << "  worker:                     " << bool_text(report.cleanup.worker_terminated) << '\n';

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

} // namespace xvram::gemm_bench
