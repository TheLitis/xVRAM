#include "xvram/residency/report.hpp"

#include "xvram/base/size_parser.hpp"

#include <iomanip>
#include <ostream>
#include <string_view>

namespace xvram::residency {
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
  output << "  " << workload.policy << '/' << workload.scenario << ": " << workload.status
         << '\n';
  output << "    operations / passes:      ";
  write_count(output, workload.operations_retired);
  output << " / ";
  if (workload.passes_completed.has_value()) {
    output << *workload.passes_completed;
  } else {
    output << "unavailable";
  }
  output << '\n';
  output << "    hits / misses / ratio:    ";
  write_count(output, workload.cache_hits);
  output << " / ";
  write_count(output, workload.cache_misses);
  output << " / ";
  write_ratio(output, workload.cache_hit_rate);
  output << '\n';
  output << "    H2D / D2H:                ";
  write_bytes(output, workload.bytes_h2d);
  output << " / ";
  write_bytes(output, workload.bytes_d2h);
  output << '\n';
  output << "    clean/dirty/writeback:    ";
  write_count(output, workload.clean_evictions);
  output << '/';
  write_count(output, workload.dirty_evictions);
  output << '/';
  write_count(output, workload.writebacks_completed);
  output << '\n';
  output << "    prefetch i/u/w/c/p:       ";
  write_count(output, workload.prefetch_issued);
  output << '/';
  write_count(output, workload.prefetch_useful);
  output << '/';
  write_count(output, workload.prefetch_wasted);
  output << '/';
  write_count(output, workload.prefetch_cancelled);
  output << '/';
  write_count(output, workload.prefetch_promoted);
  output << '\n';
  output << "    map/unmap/access/reuse:   ";
  write_count(output, workload.mapping_count);
  output << '/';
  write_count(output, workload.unmap_count);
  output << '/';
  write_count(output, workload.set_access_count);
  output << '/';
  write_count(output, workload.handle_reuse_count);
  output << '\n';
  output << "    stable/full/matches:      " << bool_text(workload.stable_addresses_verified)
         << '/' << bool_text(workload.full_verification_completed) << '/'
         << bool_text(workload.matches_cpu) << '\n';
  output << "    unsafe remap/transition:  ";
  write_count(output, workload.unsafe_remap_count);
  output << '/';
  write_count(output, workload.unsafe_transition_count);
  output << '\n';
  if (workload.output_digest128.has_value()) {
    output << "    digest128:                " << *workload.output_digest128;
    if (workload.expected_digest128.has_value()) {
      output << " (expected " << *workload.expected_digest128 << ')';
    }
    output << '\n';
  }
  if (workload.elapsed_ms.has_value()) {
    output << "    elapsed / GiB/s:          " << std::fixed << std::setprecision(2)
           << *workload.elapsed_ms << " ms / ";
    write_ratio(output, workload.throughput_gib_per_second);
    output << '\n';
  }
}

} // namespace

void write_text(const Report& report, std::ostream& output) {
  output << "xVRAM residency cache " << report.build.version << " (" << report.build.git_commit
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
  output << "  Logical / chunk:           ";
  write_bytes(output, report.configuration.effective_logical_bytes);
  output << " / ";
  write_bytes(output, report.configuration.effective_chunk_bytes);
  output << '\n';
  output << "  Initial cache target:      ";
  write_bytes(output, report.configuration.initial_cache_target_bytes);
  output << '\n';
  output << "  Policy / scenario:         " << report.configuration.policy << " / "
         << report.configuration.scenario << '\n';
  output << "  Staging / prefetch:        " << report.configuration.staging_slots << " / "
         << report.configuration.prefetch_distance << '\n';
  output << "  Passes / timeout:          " << report.configuration.passes << " / "
         << report.configuration.timeout_ms << " ms\n";
  output << "  Budget poll / stall:       " << report.configuration.budget_poll_ms << " / "
         << report.configuration.stall_timeout_ms << " ms\n";
  output << "  Trace:                     "
         << (report.configuration.trace_enabled ? "enabled" : "disabled") << '\n';

  output << "\nWorkloads:\n";
  if (report.workloads.empty()) {
    output << "  none\n";
  }
  for (const WorkloadResult& workload : report.workloads) {
    write_workload(output, workload);
  }

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
  output << "  Resident peak / staging:  ";
  write_bytes(output, report.cache.resident_bytes_peak);
  output << " / ";
  write_bytes(output, report.cache.pinned_staging_bytes);
  output << '\n';
  output << "  Hits / misses / ratio:    ";
  write_count(output, report.cache.cache_hits);
  output << " / ";
  write_count(output, report.cache.cache_misses);
  output << " / ";
  write_ratio(output, report.cache.cache_hit_rate);
  output << '\n';
  output << "  Shrinks / grows / retries:" << ' ';
  write_count(output, report.cache.target_shrink_count);
  output << " / ";
  write_count(output, report.cache.target_grow_count);
  output << " / ";
  write_count(output, report.cache.target_oom_retry_count);
  output << '\n';

  output << "\nProof:\n";
  output << "  Logical exceeds VRAM:     " << bool_text(report.proof.logical_data_exceeds_vram)
         << '\n';
  output << "  Cache smaller:            " << bool_text(report.proof.cache_smaller_than_logical)
         << '\n';
  output << "  Handles reused:           " << bool_text(report.proof.handles_reused) << '\n';
  output << "  Stable VA / SetAccess:    "
         << bool_text(report.proof.stable_virtual_addresses_verified) << '/'
         << bool_text(report.proof.set_access_after_map_verified) << '\n';
  output << "  Events / no aliases:      " << bool_text(report.proof.event_boundaries_verified)
         << '/' << bool_text(report.proof.no_physical_aliases_verified) << '\n';
  output << "  Writeback / bounded pool: " << bool_text(report.proof.dirty_writeback_verified)
         << '/' << bool_text(report.proof.staging_pool_bounded) << '\n';
  output << "  Target / CPU match:       " << bool_text(report.proof.cache_target_respected) << '/'
         << bool_text(report.proof.all_workloads_match_cpu) << '\n';
  output << "  Policies match:           " << bool_text(report.proof.policies_match) << '\n';
  output << "  Raw VA omitted:           " << bool_text(report.proof.raw_virtual_addresses_omitted)
         << '\n';

  output << "\nCleanup: " << bool_text(report.cleanup.complete) << '\n';
  output << "  transactions/prefetch/writeback: "
         << bool_text(report.cleanup.transactions_drained) << '/'
         << bool_text(report.cleanup.prefetch_drained) << '/'
         << bool_text(report.cleanup.writebacks_completed) << '\n';
  output << "  events/streams/module:            " << bool_text(report.cleanup.events_destroyed)
         << '/' << bool_text(report.cleanup.streams_destroyed) << '/'
         << bool_text(report.cleanup.module_unloaded) << '\n';
  output << "  mappings/handles/device/reservations: "
         << bool_text(report.cleanup.mappings_removed)
         << '/' << bool_text(report.cleanup.physical_handles_released) << '/'
         << bool_text(report.cleanup.device_allocations_released) << '/'
         << bool_text(report.cleanup.virtual_reservations_released) << '\n';
  output << "  staging/backing/context:          "
         << bool_text(report.cleanup.pinned_staging_released) << '/'
         << bool_text(report.cleanup.host_backing_released) << '/'
         << bool_text(report.cleanup.context_destroyed) << '\n';
  output << "  trace/worker:                     " << bool_text(report.cleanup.trace_closed) << '/'
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

} // namespace xvram::residency
