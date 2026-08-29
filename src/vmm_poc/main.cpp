#include "platform/cuda/cuda_api.hpp"
#include "platform/system_info.hpp"
#include "vmm_poc/executor.hpp"
#include "vmm_poc/outcome.hpp"
#include "vmm_poc/worker_controller.hpp"
#include "vmm_poc/worker_protocol.hpp"
#include "vmm_poc/workload.hpp"
#include "xvram/base/size_parser.hpp"
#include "xvram/version.hpp"
#include "xvram/vmm_poc/report.hpp"
#include "xvram/vmm_transform_ptx.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#include <fcntl.h>
#include <io.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

namespace {

constexpr int exit_completed = 0;
constexpr int exit_prerequisite = 23;
constexpr int exit_oom = 25;
constexpr int exit_timeout = 26;
constexpr int exit_failure = 27;
constexpr int exit_usage = 64;
constexpr int exit_internal = 70;
constexpr int exit_io = 74;

enum class CliAction { run, help, version };

struct CliOptions {
  xvram::vmm_poc::ExecutorOptions executor;
  std::chrono::seconds timeout{120};
  std::optional<std::string> json_path;
  bool pretty_json = true;
  bool print_text = true;
  bool include_identifiers = false;
  bool worker = false;
  CliAction action = CliAction::run;
};

void print_help(std::ostream& output) {
  output << R"(xvram-vmm-poc - explicit CUDA VMM oversubscription proof

Usage:
  xvram-vmm-poc [options]

Options:
  --device <ordinal>             CUDA device ordinal (default: 0)
  --logical-size <auto|size>     Logical uint32 array size (default: auto)
  --chunk-size <size>            Requested tile size (default: 64 MiB)
  --window-slots <1..8>          Physical handles and pinned buffers (default: 2)
  --passes <2..8>                Alternating forward/reverse passes (default: 2)
  --mode <reference|pipeline|both>
                                 Execution mode (default: both)
  --stall-timeout-ms <n>         Worker GPU polling deadline (default: 5000)
  --timeout-seconds <n>          Controller overall deadline (default: 120)
  --device-headroom <size>       Required free device headroom (default: 512 MiB)
  --seed <hex-u64>               Deterministic workload seed
  --json <path|->                Write the strict xvram.vmm_poc v1 report
  --compact-json                 Disable JSON indentation
  --no-text                      Suppress the human-readable report
  --include-identifiers          Include stable UUID/LUID/PCI identifiers
  --version                      Print xVRAM version
  -h, --help                     Show this help

Success means the complete resident-only oversubscription proof passed. The process is an
isolated controller; it never changes TDR, WDDM, or NVIDIA system settings.
)";
}

[[nodiscard]] bool parse_u32(const std::string_view text, std::uint32_t& output) {
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), output, 10);
  return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

[[nodiscard]] bool parse_i32(const std::string_view text, std::int32_t& output) {
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), output, 10);
  return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

[[nodiscard]] bool parse_seed(std::string_view text, std::uint64_t& output) {
  if (text.starts_with("0x") || text.starts_with("0X")) {
    text.remove_prefix(2);
  }
  if (text.empty() || text.size() > 16U) {
    return false;
  }
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), output, 16);
  return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

[[nodiscard]] bool parse_cli(const int argc, char** argv, CliOptions& options, std::string& error) {
  std::vector<std::string_view> arguments;
  arguments.reserve(static_cast<std::size_t>(argc > 0 ? argc - 1 : 0));
  for (int index = 1; index < argc; ++index) {
    arguments.emplace_back(argv[index]);
  }
  const auto value_after = [&](std::size_t& index,
                               const std::string_view option) -> std::optional<std::string_view> {
    if (index + 1U >= arguments.size()) {
      error = std::string(option) + " requires a value";
      return std::nullopt;
    }
    ++index;
    return arguments[index];
  };

  for (std::size_t index = 0; index < arguments.size(); ++index) {
    const std::string_view argument = arguments[index];
    if (argument == "-h" || argument == "--help") {
      options.action = CliAction::help;
      continue;
    }
    if (argument == "--version") {
      options.action = CliAction::version;
      continue;
    }
    if (argument == "--worker") {
      options.worker = true;
      continue;
    }
    if (argument == "--device") {
      const auto value = value_after(index, argument);
      if (!value.has_value() || !parse_i32(*value, options.executor.device_ordinal) ||
          options.executor.device_ordinal < 0) {
        if (error.empty()) {
          error = "--device must be a non-negative integer";
        }
        return false;
      }
      continue;
    }
    if (argument == "--logical-size") {
      const auto value = value_after(index, argument);
      if (!value.has_value()) {
        return false;
      }
      if (*value == "auto") {
        options.executor.logical_bytes.reset();
      } else {
        const xvram::ParsedSize parsed = xvram::parse_size(*value);
        if (!parsed || *parsed.bytes == 0 || *parsed.bytes % xvram::vmm_poc::word_bytes != 0) {
          error = "--logical-size must be auto or a positive uint32-aligned size";
          return false;
        }
        options.executor.logical_bytes = *parsed.bytes;
      }
      continue;
    }
    if (argument == "--chunk-size" || argument == "--device-headroom") {
      const auto value = value_after(index, argument);
      if (!value.has_value()) {
        return false;
      }
      const xvram::ParsedSize parsed = xvram::parse_size(*value);
      if (!parsed || *parsed.bytes == 0) {
        error = std::string(argument) + " must be a positive size";
        return false;
      }
      if (argument == "--chunk-size") {
        options.executor.chunk_bytes = *parsed.bytes;
      } else {
        options.executor.device_headroom_bytes = *parsed.bytes;
      }
      continue;
    }
    if (argument == "--window-slots" || argument == "--passes" ||
        argument == "--stall-timeout-ms" || argument == "--timeout-seconds") {
      const auto value = value_after(index, argument);
      std::uint32_t parsed = 0;
      if (!value.has_value() || !parse_u32(*value, parsed) || parsed == 0) {
        if (error.empty()) {
          error = std::string(argument) + " must be a positive integer";
        }
        return false;
      }
      if (argument == "--window-slots") {
        if (parsed > 8U) {
          error = "--window-slots must be between 1 and 8";
          return false;
        }
        options.executor.window_slots = parsed;
      } else if (argument == "--passes") {
        if (parsed < 2U || parsed > 8U) {
          error = "--passes must be between 2 and 8";
          return false;
        }
        options.executor.passes = parsed;
      } else if (argument == "--stall-timeout-ms") {
        options.executor.stall_timeout = std::chrono::milliseconds(parsed);
      } else {
        options.timeout = std::chrono::seconds(parsed);
      }
      continue;
    }
    if (argument == "--mode") {
      const auto value = value_after(index, argument);
      if (!value.has_value()) {
        return false;
      }
      if (*value == "reference") {
        options.executor.mode = xvram::vmm_poc::RequestedMode::reference;
      } else if (*value == "pipeline") {
        options.executor.mode = xvram::vmm_poc::RequestedMode::pipeline;
      } else if (*value == "both") {
        options.executor.mode = xvram::vmm_poc::RequestedMode::both;
      } else {
        error = "--mode must be reference, pipeline, or both";
        return false;
      }
      continue;
    }
    if (argument == "--seed") {
      const auto value = value_after(index, argument);
      if (!value.has_value() || !parse_seed(*value, options.executor.seed)) {
        if (error.empty()) {
          error = "--seed must be a hexadecimal uint64 value";
        }
        return false;
      }
      continue;
    }
    if (argument == "--json") {
      const auto value = value_after(index, argument);
      if (!value.has_value()) {
        return false;
      }
      options.json_path = std::string(*value);
      continue;
    }
    if (argument == "--compact-json") {
      options.pretty_json = false;
      continue;
    }
    if (argument == "--no-text") {
      options.print_text = false;
      continue;
    }
    if (argument == "--include-identifiers") {
      options.include_identifiers = true;
      options.executor.include_identifiers = true;
      continue;
    }
    error = "unknown argument: " + std::string(argument);
    return false;
  }

  if (options.action != CliAction::run && arguments.size() > 1U) {
    error = "--help and --version cannot be combined with execution options";
    return false;
  }
  if (!options.print_text && !options.json_path.has_value() && !options.worker) {
    error = "--no-text requires --json";
    return false;
  }
  if (options.executor.mode != xvram::vmm_poc::RequestedMode::reference &&
      options.executor.window_slots < 2U) {
    error = "--window-slots 1 is allowed only with --mode reference";
    return false;
  }
  return true;
}

[[nodiscard]] std::string utc_timestamp() {
  const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::tm utc{};
#ifdef _WIN32
  gmtime_s(&utc, &now);
#else
  gmtime_r(&now, &utc);
#endif
  std::ostringstream output;
  output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
  return output.str();
}

[[nodiscard]] std::string compiler_description() {
#if defined(_MSC_VER)
  return "MSVC " + std::to_string(_MSC_VER);
#elif defined(__clang__)
  return "Clang " + std::to_string(__clang_major__) + "." + std::to_string(__clang_minor__) + "." +
         std::to_string(__clang_patchlevel__);
#elif defined(__GNUC__)
  return "GCC " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__) + "." +
         std::to_string(__GNUC_PATCHLEVEL__);
#else
  return "unknown";
#endif
}

[[nodiscard]] std::string build_type() {
#ifdef NDEBUG
  return "Release";
#else
  return "Debug";
#endif
}

[[nodiscard]] std::string seed_string(const std::uint64_t seed) {
  std::ostringstream output;
  output << std::hex << std::nouppercase << std::setfill('0') << std::setw(16) << seed;
  return output.str();
}

[[nodiscard]] std::string requested_mode_name(const xvram::vmm_poc::RequestedMode mode) {
  switch (mode) {
  case xvram::vmm_poc::RequestedMode::reference:
    return "reference";
  case xvram::vmm_poc::RequestedMode::pipeline:
    return "pipeline";
  case xvram::vmm_poc::RequestedMode::both:
    return "compare";
  }
  return "compare";
}

[[nodiscard]] xvram::vmm_poc::Report make_base_report(const CliOptions& options) {
  xvram::vmm_poc::Report report;
  report.generated_at_utc = utc_timestamp();
  report.build = {XVRAM_VERSION, XVRAM_GIT_COMMIT, compiler_description(), build_type(),
                  XVRAM_CUDA_HEADERS_VERSION};
  report.system = xvram::platform::collect_system_info();
  report.configuration.requested_device_ordinal = options.executor.device_ordinal;
  report.configuration.requested_logical_bytes = options.executor.logical_bytes;
  report.configuration.requested_chunk_bytes = options.executor.chunk_bytes;
  report.configuration.requested_window_slots = options.executor.window_slots;
  report.configuration.passes = options.executor.passes;
  report.configuration.timeout_ms = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(options.timeout).count());
  report.configuration.stall_timeout_ms =
      static_cast<std::uint64_t>(options.executor.stall_timeout.count());
  report.configuration.seed_hex = seed_string(options.executor.seed);
  report.configuration.device_headroom_bytes = options.executor.device_headroom_bytes;
  report.configuration.sizing_mode =
      options.executor.logical_bytes.has_value() ? "explicit" : "auto";
  report.configuration.mode = requested_mode_name(options.executor.mode);
  report.configuration.identifiers_included = options.include_identifiers;
  report.proof.kernel_module_version = xvram::vmm_poc::vmm_transform_module_version;
  report.proof.kernel_module_sha256 = xvram::vmm_poc::vmm_transform_module_sha256;
  report.proof.pattern_version = "xvram.uint32_transform.v1";
  report.outcome.status = "skipped";
  report.outcome.reason.reset();
  report.outcome.exit_code = exit_prerequisite;
  return report;
}

[[nodiscard]] xvram::vmm_poc::TimingSummary timing_summary(const std::vector<double>& samples) {
  xvram::vmm_poc::TimingSummary output;
  const auto summary = xvram::vmm_poc::summarize_samples(samples);
  if (!summary.has_value()) {
    return output;
  }
  output.sample_count = summary->count;
  output.total_ms = summary->total;
  output.minimum_ms = summary->minimum;
  output.median_ms = summary->median;
  output.p95_ms = summary->percentile_95;
  output.maximum_ms = summary->maximum;
  return output;
}

[[nodiscard]] xvram::vmm_poc::ModeResult
mode_report(const xvram::vmm_poc::ModeStatistics& statistics, const bool timeout_snapshot = false) {
  xvram::vmm_poc::ModeResult output;
  if (statistics.status == "not_requested") {
    return output;
  }
  if (statistics.status == "running") {
    output.status = timeout_snapshot ? "timed_out" : "failed";
  } else {
    output.status = statistics.status;
  }
  output.elements_processed = statistics.h2d_bytes / xvram::vmm_poc::word_bytes;
  output.tiles_processed = statistics.tiles_retired;
  output.passes_completed = statistics.passes_completed;
  output.bytes_h2d = statistics.h2d_bytes;
  output.bytes_d2h = statistics.d2h_bytes;
  output.mapping_count = statistics.mappings;
  output.unmap_count = statistics.unmappings;
  output.remap_count = statistics.remappings;
  output.event_boundary_count = statistics.event_boundaries;
  output.unsafe_remap_count = statistics.unsafe_remaps;
  output.physical_handle_count = statistics.slots;
  output.max_concurrent_mappings = statistics.slots;
  output.slot_count = statistics.slots;
  output.set_access_count = statistics.set_access_calls;
  output.handle_reuse_count = statistics.handle_reuses;
  output.stable_addresses_verified = statistics.stable_addresses;
  output.full_verification_completed = statistics.full_verification;
  output.matches_cpu = statistics.matches_cpu;
  output.elapsed_ms = statistics.timings.total_ms;
  output.setup_ms = statistics.timings.setup_ms;
  output.h2d_ms = statistics.timings.h2d_ms;
  output.kernel_ms = statistics.timings.kernel_ms;
  output.d2h_ms = statistics.timings.d2h_ms;
  output.verification_ms = statistics.timings.verification_ms;
  output.remap_timing = timing_summary(statistics.remap_samples_ms);
  if (!statistics.expected_digest128.empty()) {
    output.expected_digest128 = statistics.expected_digest128;
  }
  if (!statistics.digest128.empty()) {
    output.output_digest128 = statistics.digest128;
  }
  output.mismatch_count = statistics.mismatch_count;
  output.first_mismatch_byte_offset = statistics.first_mismatch_byte_offset;
  return output;
}

[[nodiscard]] bool mode_event_boundaries(const xvram::vmm_poc::ModeStatistics& mode) {
  return mode.status == "completed" && mode.unsafe_remaps == 0 &&
         mode.mappings == mode.set_access_calls && mode.mappings == mode.unmappings &&
         mode.unmappings == mode.event_boundaries;
}

void apply_executor_result(xvram::vmm_poc::Report& report,
                           const xvram::vmm_poc::ExecutorResult& execution,
                           const CliOptions& options) {
  if (execution.device.has_value()) {
    const auto& source = *execution.device;
    xvram::vmm_poc::DeviceInfo device;
    device.ordinal = source.ordinal;
    device.name = source.name;
    device.uuid = source.uuid;
    device.luid = source.luid;
    device.pci_bus_id = source.pci_bus_id;
    device.driver_model = source.driver_model;
    device.total_memory_bytes = source.total_memory_bytes;
    device.free_memory_bytes_start = source.free_memory_bytes_start;
    device.safe_device_budget_bytes = source.free_memory_bytes_start;
    if (source.wddm_available_bytes_start.has_value()) {
      device.safe_device_budget_bytes =
          std::min(*device.safe_device_budget_bytes, *source.wddm_available_bytes_start);
    }
    device.wddm_budget_bytes_start = source.wddm_budget_bytes_start;
    device.wddm_usage_bytes_start = source.wddm_usage_bytes_start;
    device.wddm_available_bytes_start = source.wddm_available_bytes_start;
    device.wddm_available_bytes_minimum = source.wddm_available_bytes_minimum;
    device.wddm_available_bytes_end = source.wddm_available_bytes_end;
    device.vmm_supported = source.vmm_supported;
    if (source.minimum_granularity_bytes != 0) {
      device.minimum_granularity_bytes = source.minimum_granularity_bytes;
    }
    if (source.recommended_granularity_bytes != 0) {
      device.recommended_granularity_bytes = source.recommended_granularity_bytes;
    }
    report.device = std::move(device);
  }

  report.modes.reference = mode_report(execution.reference);
  report.modes.pipeline = mode_report(execution.pipeline);
  if (execution.reference.status == "completed" && execution.pipeline.status == "completed" &&
      execution.pipeline.timings.total_ms > 0.0) {
    report.modes.pipeline_speedup =
        execution.reference.timings.total_ms / execution.pipeline.timings.total_ms;
  }

  if (execution.plan.has_value()) {
    const auto& plan = *execution.plan;
    report.configuration.effective_window_slots = plan.window_slots;
    report.configuration.host_headroom_bytes = plan.host_headroom_bytes;
    report.proof.effective_logical_bytes = plan.effective_logical_bytes;
    report.proof.effective_chunk_bytes = plan.effective_chunk_bytes;
    report.proof.logical_element_count = plan.logical_element_count;
    report.proof.logical_chunk_count = plan.logical_chunk_count;
    report.proof.tile_visit_count = plan.tile_visit_count;
    report.proof.address_revisit_count = plan.address_revisit_count;
    report.proof.resident_physical_bytes = plan.resident_physical_bytes;
    report.proof.pinned_staging_bytes = plan.pinned_staging_bytes;
    if (execution.device.has_value()) {
      report.proof.physical_window_smaller =
          plan.resident_physical_bytes < plan.effective_logical_bytes;
    }
  }

  const bool reference_requested =
      options.executor.mode == xvram::vmm_poc::RequestedMode::reference ||
      options.executor.mode == xvram::vmm_poc::RequestedMode::both;
  const bool pipeline_requested =
      options.executor.mode == xvram::vmm_poc::RequestedMode::pipeline ||
      options.executor.mode == xvram::vmm_poc::RequestedMode::both;
  const bool reference_reused =
      !reference_requested ||
      (execution.reference.status == "completed" && execution.reference.handle_reuses > 0);
  const bool pipeline_reused = !pipeline_requested || (execution.pipeline.status == "completed" &&
                                                       execution.pipeline.handle_reuses > 0);
  report.proof.handles_reused = reference_reused && pipeline_reused;
  report.proof.stable_virtual_addresses_verified =
      (!reference_requested ||
       (execution.reference.status == "completed" && execution.reference.stable_addresses)) &&
      (!pipeline_requested ||
       (execution.pipeline.status == "completed" && execution.pipeline.stable_addresses));
  report.proof.event_boundaries_verified =
      (!reference_requested || mode_event_boundaries(execution.reference)) &&
      (!pipeline_requested || mode_event_boundaries(execution.pipeline));
  if (reference_requested) {
    report.proof.reference_matches_cpu = execution.reference.matches_cpu;
  }
  if (pipeline_requested) {
    report.proof.pipeline_matches_cpu = execution.pipeline.matches_cpu;
  }
  if (reference_requested && pipeline_requested) {
    report.proof.modes_match = execution.reference.status == "completed" &&
                               execution.pipeline.status == "completed" &&
                               execution.reference.digest128 == execution.pipeline.digest128;
  }
  if (!execution.reference.expected_digest128.empty()) {
    report.proof.cpu_reference_digest128 = execution.reference.expected_digest128;
  } else if (!execution.pipeline.expected_digest128.empty()) {
    report.proof.cpu_reference_digest128 = execution.pipeline.expected_digest128;
  }

  report.outcome.status = execution.status;
  report.outcome.reason =
      execution.exit_code == exit_completed
          ? std::optional<std::string>{}
          : std::optional<std::string>{xvram::vmm_poc::normalize_outcome_reason(execution)};
  report.outcome.exit_code = execution.exit_code;
  if (execution.failure.has_value()) {
    const auto& failure = *execution.failure;
    report.outcome.stage = xvram::vmm_poc::normalize_outcome_stage(failure);
    report.outcome.operation = failure.operation;
    report.outcome.native_code = failure.native_code;
    report.outcome.native_name = failure.native_name;
    report.outcome.message = failure.message;
    report.outcome.pass_index = failure.pass_index;
    report.outcome.tile_index = failure.tile_index;
    report.outcome.logical_byte_offset = failure.logical_byte_offset;
    if (execution.pipeline.status == "failed") {
      report.outcome.mode = "pipeline";
    } else if (execution.reference.status == "failed") {
      report.outcome.mode = "reference";
    }
  }
  report.cleanup.complete = execution.cleanup.complete;
  report.cleanup.events_drained = execution.cleanup.events_drained;
  report.cleanup.events_destroyed = execution.cleanup.events_destroyed;
  report.cleanup.streams_destroyed = execution.cleanup.streams_destroyed;
  report.cleanup.module_unloaded = execution.cleanup.module_unloaded;
  report.cleanup.device_allocations_released = execution.cleanup.handles_released;
  report.cleanup.mappings_removed = execution.cleanup.mappings_removed;
  report.cleanup.physical_handle_released = execution.cleanup.handles_released;
  report.cleanup.virtual_reservation_released = execution.cleanup.reservation_released;
  report.cleanup.pinned_staging_released = execution.cleanup.pinned_buffers_released;
  report.cleanup.host_backing_released = execution.cleanup.host_backing_released;
  report.cleanup.context_destroyed = execution.cleanup.context_destroyed;
  report.cleanup.worker_terminated = true;
  for (const auto& message : execution.diagnostics) {
    report.diagnostics.push_back({xvram::probe::DiagnosticLevel::error, "vmm_poc", "executor",
                                  message, std::nullopt, options.executor.device_ordinal});
  }
}

void mark_timeout_snapshot(xvram::vmm_poc::Report& report, const CliOptions& options,
                           const xvram::vmm_poc::ModeStatistics* reference = nullptr,
                           const xvram::vmm_poc::ModeStatistics* pipeline = nullptr) {
  if (reference != nullptr) {
    report.modes.reference = mode_report(*reference, true);
  } else if (options.executor.mode != xvram::vmm_poc::RequestedMode::pipeline) {
    report.modes.reference.status = "timed_out";
  }
  if (pipeline != nullptr) {
    report.modes.pipeline = mode_report(*pipeline, true);
  } else if (options.executor.mode != xvram::vmm_poc::RequestedMode::reference) {
    report.modes.pipeline.status = "timed_out";
  }
  report.outcome.status = "failed";
  report.outcome.reason = "timeout";
  report.outcome.exit_code = exit_timeout;
  report.outcome.stage = "watchdog";
  report.outcome.operation = "watchdog";
  report.outcome.message = "the isolated worker exceeded its progress or overall deadline";
  report.cleanup = {};
  report.cleanup.worker_terminated = true;
}

[[nodiscard]] std::string report_json(const xvram::vmm_poc::Report& report, const bool pretty) {
  std::ostringstream output;
  xvram::vmm_poc::write_json(report, output, pretty);
  return output.str();
}

[[nodiscard]] std::string report_text(const xvram::vmm_poc::Report& report) {
  std::ostringstream output;
  xvram::vmm_poc::write_text(report, output);
  return output.str();
}

[[nodiscard]] std::string mode_argument(const xvram::vmm_poc::RequestedMode mode) {
  switch (mode) {
  case xvram::vmm_poc::RequestedMode::reference:
    return "reference";
  case xvram::vmm_poc::RequestedMode::pipeline:
    return "pipeline";
  case xvram::vmm_poc::RequestedMode::both:
    return "both";
  }
  return "both";
}

[[nodiscard]] std::vector<std::string> worker_arguments(const CliOptions& options) {
  std::vector<std::string> arguments{"--worker",
                                     "--device",
                                     std::to_string(options.executor.device_ordinal),
                                     "--logical-size",
                                     options.executor.logical_bytes.has_value()
                                         ? std::to_string(*options.executor.logical_bytes)
                                         : "auto",
                                     "--chunk-size",
                                     std::to_string(options.executor.chunk_bytes),
                                     "--window-slots",
                                     std::to_string(options.executor.window_slots),
                                     "--passes",
                                     std::to_string(options.executor.passes),
                                     "--mode",
                                     mode_argument(options.executor.mode),
                                     "--stall-timeout-ms",
                                     std::to_string(options.executor.stall_timeout.count()),
                                     "--timeout-seconds",
                                     std::to_string(options.timeout.count()),
                                     "--device-headroom",
                                     std::to_string(options.executor.device_headroom_bytes),
                                     "--seed",
                                     seed_string(options.executor.seed)};
  if (options.include_identifiers) {
    arguments.emplace_back("--include-identifiers");
  }
  if (!options.pretty_json) {
    arguments.emplace_back("--compact-json");
  }
  return arguments;
}

[[nodiscard]] bool emit_outputs(const CliOptions& options, const std::string& json,
                                const std::string& text) {
  if (options.json_path.has_value()) {
    if (*options.json_path == "-") {
      std::cout << json;
      if (!json.empty() && json.back() != '\n') {
        std::cout << '\n';
      }
      if (!std::cout) {
        return false;
      }
      if (options.print_text) {
        std::cerr << text;
      }
      return static_cast<bool>(std::cerr);
    }
    const std::filesystem::path path(*options.json_path);
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
      return false;
    }
    output << json;
    if (!json.empty() && json.back() != '\n') {
      output << '\n';
    }
    if (!output) {
      return false;
    }
    if (options.print_text) {
      std::cout << text;
    }
    return static_cast<bool>(std::cout);
  }
  if (options.print_text) {
    std::cout << text;
  }
  return static_cast<bool>(std::cout);
}

[[nodiscard]] int worker_main(const CliOptions& options) {
#ifdef _WIN32
  if (_setmode(_fileno(stdout), _O_BINARY) == -1) {
    return exit_failure;
  }
#endif
  std::uint64_t sequence = 1;
  xvram::vmm_poc::Report initial = make_base_report(options);
  mark_timeout_snapshot(initial, options);
  if (!xvram::vmm_poc::write_worker_frame(std::cout, xvram::vmm_poc::WorkerFrameType::plan,
                                          sequence++, report_json(initial, options.pretty_json))) {
    return exit_failure;
  }

  xvram::vmm_poc::ModeStatistics reference_progress;
  xvram::vmm_poc::ModeStatistics pipeline_progress;
  bool protocol_output_ok = true;
  const auto progress = [&](const std::string_view mode,
                            const xvram::vmm_poc::ModeStatistics& statistics, const std::uint64_t) {
    if (mode == "reference") {
      reference_progress = statistics;
    } else {
      pipeline_progress = statistics;
    }
    xvram::vmm_poc::Report snapshot = make_base_report(options);
    mark_timeout_snapshot(snapshot, options, &reference_progress, &pipeline_progress);
    if (!xvram::vmm_poc::write_worker_frame(std::cout, xvram::vmm_poc::WorkerFrameType::progress,
                                            sequence++,
                                            report_json(snapshot, options.pretty_json))) {
      protocol_output_ok = false;
    }
  };

  xvram::cuda::CudaApi api;
  const xvram::vmm_poc::ExecutorResult execution =
      xvram::vmm_poc::run_executor(api, options.executor, progress);
  xvram::vmm_poc::Report final_report = make_base_report(options);
  apply_executor_result(final_report, execution, options);
  xvram::vmm_poc::FinalWorkerPayload payload;
  payload.exit_code = execution.exit_code;
  payload.json = report_json(final_report, options.pretty_json);
  payload.text = report_text(final_report);
  const std::string encoded = xvram::vmm_poc::encode_final_worker_payload(payload);
  if (encoded.empty() || !protocol_output_ok ||
      !xvram::vmm_poc::write_worker_frame(std::cout, xvram::vmm_poc::WorkerFrameType::final,
                                          sequence, encoded)) {
    return exit_failure;
  }
  return execution.exit_code;
}

[[nodiscard]] xvram::vmm_poc::Report controller_failure_report(const CliOptions& options,
                                                               const int exit_code,
                                                               std::string reason,
                                                               std::string message) {
  xvram::vmm_poc::Report report = make_base_report(options);
  report.outcome.status = "failed";
  report.outcome.reason = std::move(reason);
  report.outcome.exit_code = exit_code;
  report.outcome.stage = "watchdog";
  report.outcome.operation = exit_code == exit_timeout ? "watchdog" : "worker_protocol";
  report.outcome.message = std::move(message);
  report.cleanup.worker_terminated = true;
  return report;
}

[[nodiscard]] int controller_main(const CliOptions& options,
                                  const std::filesystem::path& executable) {
  xvram::vmm_poc::WorkerControllerOptions controller;
  controller.executable = executable;
  controller.arguments = worker_arguments(options);
  controller.no_progress_timeout = std::chrono::seconds(15);
  controller.overall_timeout = options.timeout;
  const xvram::vmm_poc::WorkerControllerResult worker =
      xvram::vmm_poc::run_isolated_worker(controller);

  int exit_code = exit_failure;
  std::string json;
  std::string text;
  if (worker.timed_out) {
    exit_code = exit_timeout;
    const xvram::vmm_poc::Report timeout = controller_failure_report(
        options, exit_timeout, "timeout",
        "the isolated worker exceeded its progress or overall deadline and was terminated");
    // Progress payloads are deliberately opaque to the transport layer. A worker that hangs
    // may have emitted a truncated, stale, or non-report payload, so only the controller-owned
    // report is safe to expose as the timeout contract.
    json = report_json(timeout, options.pretty_json);
    text = report_text(timeout);
  } else if (worker.protocol_error || !worker.started || !worker.final.has_value()) {
    const xvram::vmm_poc::Report failure = controller_failure_report(
        options, exit_failure, "platform_error",
        worker.error.empty() ? "the isolated worker failed without a valid final report"
                             : worker.error);
    json = report_json(failure, options.pretty_json);
    text = report_text(failure);
  } else if (worker.process_exit_code != worker.final->exit_code) {
    const xvram::vmm_poc::Report failure = controller_failure_report(
        options, exit_failure, "platform_error",
        "the worker process exit code does not match its final protocol report");
    json = report_json(failure, options.pretty_json);
    text = report_text(failure);
  } else {
    exit_code = worker.final->exit_code;
    json = worker.final->json;
    text = worker.final->text;
  }
  if (!emit_outputs(options, json, text)) {
    std::cerr << "error: failed while writing report output\n";
    return exit_io;
  }
  return exit_code;
}

[[nodiscard]] std::filesystem::path current_executable_path(const char* argv_zero) {
#ifdef _WIN32
  std::vector<wchar_t> module_path(32'768U, L'\0');
  const DWORD length =
      GetModuleFileNameW(nullptr, module_path.data(), static_cast<DWORD>(module_path.size()));
  if (length != 0 && length < module_path.size()) {
    return std::filesystem::path(std::wstring(module_path.data(), length));
  }
#elif defined(__linux__)
  std::vector<char> module_path(4'096U, '\0');
  while (module_path.size() <= 1024U * 1024U) {
    const ssize_t length = readlink("/proc/self/exe", module_path.data(), module_path.size());
    if (length >= 0 && static_cast<std::size_t>(length) < module_path.size()) {
      return std::filesystem::path(
          std::string(module_path.data(), static_cast<std::size_t>(length)));
    }
    if (length < 0) {
      break;
    }
    module_path.resize(module_path.size() * 2U);
  }
#endif

  if (argv_zero == nullptr || *argv_zero == '\0') {
    return {};
  }
  const std::filesystem::path requested(argv_zero);
  std::error_code path_error;
  if (requested.is_absolute() || requested.has_parent_path()) {
    const std::filesystem::path absolute = std::filesystem::absolute(requested, path_error);
    return path_error ? std::filesystem::path{} : absolute;
  }

#ifdef _WIN32
  char* path_value = nullptr;
  std::size_t path_value_size = 0;
  if (_dupenv_s(&path_value, &path_value_size, "PATH") != 0 || path_value == nullptr) {
    return {};
  }
  const std::string owned_search_path(path_value, path_value_size > 0 ? path_value_size - 1U : 0U);
  std::free(path_value);
  constexpr char path_separator = ';';
#else
  const char* path_value = std::getenv("PATH");
  if (path_value == nullptr) {
    return {};
  }
  const std::string owned_search_path(path_value);
  constexpr char path_separator = ':';
#endif
  const std::string_view search_path(owned_search_path);
  std::size_t begin = 0;
  while (begin <= search_path.size()) {
    const std::size_t end = search_path.find(path_separator, begin);
    const std::string_view entry = search_path.substr(
        begin, end == std::string_view::npos ? search_path.size() - begin : end - begin);
    const std::filesystem::path directory = entry.empty()
                                                ? std::filesystem::current_path(path_error)
                                                : std::filesystem::path(std::string(entry));
    if (!path_error) {
      const std::filesystem::path candidate = directory / requested;
      if (std::filesystem::is_regular_file(candidate, path_error) && !path_error) {
        const std::filesystem::path absolute = std::filesystem::absolute(candidate, path_error);
        if (!path_error) {
          return absolute;
        }
      }
    }
    path_error.clear();
    if (end == std::string_view::npos) {
      break;
    }
    begin = end + 1U;
  }
  return {};
}

} // namespace

int main(const int argc, char** argv) {
  CliOptions options;
  std::string parse_error;
  if (!parse_cli(argc, argv, options, parse_error)) {
    std::cerr << "error: " << parse_error << "\n\n";
    print_help(std::cerr);
    return exit_usage;
  }
  if (options.action == CliAction::help) {
    print_help(std::cout);
    return exit_completed;
  }
  if (options.action == CliAction::version) {
    std::cout << XVRAM_VERSION << '\n';
    return exit_completed;
  }
  try {
    if (options.worker) {
      return worker_main(options);
    }
    const std::filesystem::path executable = current_executable_path(argc > 0 ? argv[0] : nullptr);
    return controller_main(options, executable);
  } catch (const std::bad_alloc&) {
    if (options.worker) {
      return exit_oom;
    }
    std::cerr << "fatal: memory allocation failed\n";
    return exit_internal;
  } catch (const std::exception& exception) {
    if (options.worker) {
      return exit_failure;
    }
    std::cerr << "fatal: " << exception.what() << '\n';
    return exit_internal;
  } catch (...) {
    if (options.worker) {
      return exit_failure;
    }
    std::cerr << "fatal: unknown internal error\n";
    return exit_internal;
  }
}
