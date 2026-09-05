#pragma once
#include "compat_bench/options.hpp"
#include "compat_bench/pattern.hpp"
#include "xvram/cuda_compat.h"
#include "xvram/probe/report.hpp"
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace xvram::compat_bench {
struct Workload {
  std::string name, status = "not_run";
  Shape shape;
  std::uint64_t logical_bytes = 0, storage_bytes = 0, output_elements_checked = 0;
  std::uint64_t padding_elements_checked = 0, mismatches = 0, tiles_retired = 0;
  std::uint64_t mappings = 0, unmaps = 0, h2d_bytes = 0, d2h_bytes = 0, evictions = 0,
                handle_reuses = 0;
  std::uint32_t passes_completed = 0;
  double max_absolute_error = 0.0, max_relative_error = 0.0;
  std::optional<std::uint64_t> mismatch_offset;
  std::string digest, reference_digest, reference_kind;
  std::optional<bool> native_baseline_equal;
  std::optional<double> native_baseline_ms;
  std::vector<double> pass_timings_ms;
};
struct Report {
  Options options;
  probe::SystemInfo system;
  xvram_cuda_compat_telemetry_v1 telemetry{};
  bool telemetry_observed = false;
  std::optional<std::string> uuid, pci_bus_id;
  std::vector<Workload> workloads;
  int exit_code = 23;
  std::string reason = "not_started", message;
  std::string generated_at_utc;
  std::uint64_t rejections_checked = 0, rejections_failed = 0, trace_records = 0;
  std::map<std::string, std::optional<bool>> proof;
  std::map<std::string, std::optional<bool>> cleanup;
  std::vector<std::string> diagnostics;
  bool trace_complete = false;
};
[[nodiscard]] Report base_report(const Options& options);
[[nodiscard]] std::string report_json(const Report& report, bool pretty);
[[nodiscard]] std::string report_text(const Report& report);
void complete_proof(Report& report);
[[nodiscard]] std::string trace_record(std::uint64_t sequence, std::uint64_t timestamp_ns,
                                       std::string_view transition, std::uint64_t operation_id,
                                       std::optional<std::uint64_t> allocation_id,
                                       std::uint64_t bytes, std::uint64_t tiles,
                                       std::string_view reason);
} // namespace xvram::compat_bench
