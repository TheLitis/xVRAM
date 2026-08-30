#include "platform/cuda/cuda_api.hpp"
#include "platform/system_info.hpp"
#include "residency/executor.hpp"
#include "residency/scenario.hpp"
#include "residency/worker_controller.hpp"
#include "residency/worker_protocol.hpp"
#include "xvram/base/size_parser.hpp"
#include "xvram/residency/report.hpp"
#include "xvram/version.hpp"

#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <new>
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
constexpr int exit_corruption = 24;
constexpr int exit_oom = 25;
constexpr int exit_timeout = 26;
constexpr int exit_failure = 27;
constexpr int exit_usage = 64;
constexpr int exit_internal = 70;
constexpr int exit_io = 74;
constexpr std::uint64_t word_bytes = sizeof(std::uint32_t);

enum class CliAction { run, help, version };

struct CliOptions {
  xvram::residency::ExecutorOptions executor;
  std::chrono::seconds timeout{300};
  std::optional<std::string> json_path;
  std::optional<std::string> trace_path;
  bool pretty_json = true;
  bool print_text = true;
  bool include_identifiers = false;
  bool worker = false;
  CliAction action = CliAction::run;
};

void print_help(std::ostream& output) {
  output << R"(xvram-cache-bench - event-safe CUDA VMM residency-cache benchmark

Usage:
  xvram-cache-bench [options]

Options:
  --device <ordinal>                       CUDA device ordinal (default: 0)
  --logical-size <auto|size>               Logical uint32 heap size (default: auto)
  --chunk-size <size>                      Requested cache chunk size (default: 64 MiB)
  --cache-target <auto|size>               Maximum VRAM cache target (default: auto)
  --staging-slots <2..8>                   Total pinned staging buffers (default: 4)
  --policy <clock|lru|both>                Eviction policy (default: clock)
  --prefetch-distance <0..8>               Sequential prefetch distance (default: 2)
  --scenario <suite|sequential|reuse|random|read-only|write-heavy|budget-pressure>
                                             Workload scenario (default: suite)
  --passes <2..8>                          Scenario passes (default: 2)
  --pressure-size <auto|size>              Budget-pressure allocation (default: auto)
  --device-headroom <size>                 Required device headroom (default: 512 MiB)
  --budget-poll-ms <n>                     Live-budget polling interval (default: 100)
  --stall-timeout-ms <n>                   Worker polling deadline (default: 5000)
  --timeout-seconds <n>                    Controller overall deadline (default: 300)
  --seed <hex-u64>                         Deterministic workload seed
  --trace <path>                           Write xvram.residency_trace v1 JSONL
  --json <path|->                          Write xvram.residency_cache v1 JSON
  --compact-json                           Disable JSON indentation
  --no-text                                Suppress the human-readable report
  --include-identifiers                    Include stable UUID/LUID/PCI identifiers
  --version                                Print xVRAM version
  -h, --help                               Show this help

The controller isolates the CUDA worker and owns all report and trace files. It never changes
TDR, WDDM, registry, or NVIDIA system settings.
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

[[nodiscard]] std::optional<std::uint64_t> parse_positive_size(const std::string_view text) {
  const xvram::ParsedSize parsed = xvram::parse_size(text);
  if (!parsed || *parsed.bytes == 0) {
    return std::nullopt;
  }
  return *parsed.bytes;
}

[[nodiscard]] bool parse_scenario(const std::string_view text,
                                  xvram::residency::ScenarioKind& output) {
  if (text == "suite") {
    output = xvram::residency::ScenarioKind::suite;
  } else if (text == "sequential") {
    output = xvram::residency::ScenarioKind::sequential;
  } else if (text == "reuse") {
    output = xvram::residency::ScenarioKind::reuse;
  } else if (text == "random") {
    output = xvram::residency::ScenarioKind::random;
  } else if (text == "read-only") {
    output = xvram::residency::ScenarioKind::read_only;
  } else if (text == "write-heavy") {
    output = xvram::residency::ScenarioKind::write_heavy;
  } else if (text == "budget-pressure") {
    output = xvram::residency::ScenarioKind::budget_pressure;
  } else {
    return false;
  }
  return true;
}

[[nodiscard]] bool paths_collide(const std::string& left, const std::string& right) {
  std::error_code error;
  const std::filesystem::path left_absolute =
      std::filesystem::absolute(std::filesystem::path(left), error).lexically_normal();
  if (error) {
    return left == right;
  }
  const std::filesystem::path right_absolute =
      std::filesystem::absolute(std::filesystem::path(right), error).lexically_normal();
  return error ? left == right : left_absolute == right_absolute;
}

[[nodiscard]] bool parse_cli(const int argc, char** argv, CliOptions& options,
                             std::string& error) {
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
    if (argument == "--trace-enabled") {
      options.executor.trace_enabled = true;
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
        const auto parsed = parse_positive_size(*value);
        if (!parsed.has_value() || *parsed % word_bytes != 0) {
          error = "--logical-size must be auto or a positive uint32-aligned size";
          return false;
        }
        options.executor.logical_bytes = *parsed;
      }
      continue;
    }
    if (argument == "--cache-target" || argument == "--pressure-size") {
      const auto value = value_after(index, argument);
      if (!value.has_value()) {
        return false;
      }
      std::optional<std::uint64_t>& destination =
          argument == "--cache-target" ? options.executor.cache_target_bytes
                                       : options.executor.pressure_bytes;
      if (*value == "auto") {
        destination.reset();
      } else {
        const auto parsed = parse_positive_size(*value);
        if (!parsed.has_value()) {
          error = std::string(argument) + " must be auto or a positive size";
          return false;
        }
        destination = *parsed;
      }
      continue;
    }
    if (argument == "--chunk-size" || argument == "--device-headroom") {
      const auto value = value_after(index, argument);
      const auto parsed = value.has_value() ? parse_positive_size(*value) : std::nullopt;
      if (!parsed.has_value()) {
        if (error.empty()) {
          error = std::string(argument) + " must be a positive size";
        }
        return false;
      }
      if (argument == "--chunk-size") {
        if (*parsed % word_bytes != 0) {
          error = "--chunk-size must be uint32-aligned";
          return false;
        }
        options.executor.chunk_bytes = *parsed;
      } else {
        options.executor.device_headroom_bytes = *parsed;
      }
      continue;
    }
    if (argument == "--staging-slots" || argument == "--prefetch-distance" ||
        argument == "--passes" || argument == "--budget-poll-ms" ||
        argument == "--stall-timeout-ms" || argument == "--timeout-seconds") {
      const auto value = value_after(index, argument);
      std::uint32_t parsed = 0;
      if (!value.has_value() || !parse_u32(*value, parsed)) {
        if (error.empty()) {
          error = std::string(argument) + " must be an integer";
        }
        return false;
      }
      if (argument == "--staging-slots") {
        if (parsed < 2U || parsed > 8U) {
          error = "--staging-slots must be between 2 and 8";
          return false;
        }
        options.executor.staging_slots = parsed;
      } else if (argument == "--prefetch-distance") {
        if (parsed > 8U) {
          error = "--prefetch-distance must be between 0 and 8";
          return false;
        }
        options.executor.prefetch_distance = parsed;
      } else if (argument == "--passes") {
        if (parsed < 2U || parsed > 8U) {
          error = "--passes must be between 2 and 8";
          return false;
        }
        options.executor.passes = parsed;
      } else if (argument == "--budget-poll-ms") {
        if (parsed == 0) {
          error = "--budget-poll-ms must be positive";
          return false;
        }
        options.executor.budget_poll_interval = std::chrono::milliseconds(parsed);
      } else if (argument == "--stall-timeout-ms") {
        if (parsed == 0) {
          error = "--stall-timeout-ms must be positive";
          return false;
        }
        options.executor.stall_timeout = std::chrono::milliseconds(parsed);
      } else {
        if (parsed == 0) {
          error = "--timeout-seconds must be positive";
          return false;
        }
        options.timeout = std::chrono::seconds(parsed);
      }
      continue;
    }
    if (argument == "--policy") {
      const auto value = value_after(index, argument);
      if (!value.has_value()) {
        return false;
      }
      if (*value == "clock") {
        options.executor.policy = xvram::residency::RequestedPolicy::clock;
      } else if (*value == "lru") {
        options.executor.policy = xvram::residency::RequestedPolicy::lru;
      } else if (*value == "both") {
        options.executor.policy = xvram::residency::RequestedPolicy::both;
      } else {
        error = "--policy must be clock, lru, or both";
        return false;
      }
      continue;
    }
    if (argument == "--scenario") {
      const auto value = value_after(index, argument);
      if (!value.has_value() || !parse_scenario(*value, options.executor.scenario)) {
        if (error.empty()) {
          error = "--scenario must be suite, sequential, reuse, random, read-only, "
                  "write-heavy, or budget-pressure";
        }
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
    if (argument == "--trace" || argument == "--json") {
      const auto value = value_after(index, argument);
      if (!value.has_value() || value->empty()) {
        if (error.empty()) {
          error = std::string(argument) + " requires a non-empty path";
        }
        return false;
      }
      if (argument == "--trace") {
        if (*value == "-") {
          error = "--trace requires a file path; stdout is reserved for report output";
          return false;
        }
        options.trace_path = std::string(*value);
        options.executor.trace_enabled = true;
      } else {
        options.json_path = std::string(*value);
      }
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
  if (options.executor.trace_enabled && !options.trace_path.has_value() && !options.worker) {
    error = "--trace-enabled is reserved for the isolated worker";
    return false;
  }
  if (options.trace_path.has_value() && options.json_path.has_value() &&
      *options.json_path != "-" && paths_collide(*options.trace_path, *options.json_path)) {
    error = "--trace and --json must name different files";
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
  return "Clang " + std::to_string(__clang_major__) + "." +
         std::to_string(__clang_minor__) + "." + std::to_string(__clang_patchlevel__);
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

[[nodiscard]] std::string policy_name(const xvram::residency::RequestedPolicy policy) {
  switch (policy) {
  case xvram::residency::RequestedPolicy::clock:
    return "clock";
  case xvram::residency::RequestedPolicy::lru:
    return "lru";
  case xvram::residency::RequestedPolicy::both:
    return "both";
  }
  return "clock";
}

[[nodiscard]] std::string sizing_mode(const CliOptions& options) {
  if (!options.executor.logical_bytes.has_value() &&
      !options.executor.cache_target_bytes.has_value()) {
    return "auto";
  }
  if (options.executor.logical_bytes.has_value() &&
      options.executor.cache_target_bytes.has_value()) {
    return "explicit";
  }
  return "mixed";
}

void apply_cli_configuration(xvram::residency::Configuration& configuration,
                             const CliOptions& options) {
  configuration.requested_device_ordinal = options.executor.device_ordinal;
  configuration.requested_logical_bytes = options.executor.logical_bytes;
  configuration.requested_chunk_bytes = options.executor.chunk_bytes;
  configuration.requested_cache_target_bytes = options.executor.cache_target_bytes;
  configuration.staging_slots = options.executor.staging_slots;
  configuration.policy = policy_name(options.executor.policy);
  configuration.prefetch_distance = options.executor.prefetch_distance;
  configuration.scenario =
      std::string(xvram::residency::scenario_kind_name(options.executor.scenario));
  configuration.passes = options.executor.passes;
  configuration.pressure_bytes = options.executor.pressure_bytes;
  configuration.device_headroom_bytes = options.executor.device_headroom_bytes;
  configuration.budget_poll_ms =
      static_cast<std::uint64_t>(options.executor.budget_poll_interval.count());
  configuration.stall_timeout_ms =
      static_cast<std::uint64_t>(options.executor.stall_timeout.count());
  configuration.timeout_ms = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(options.timeout).count());
  configuration.seed_hex = seed_string(options.executor.seed);
  configuration.sizing_mode = sizing_mode(options);
  configuration.trace_enabled = options.executor.trace_enabled;
  configuration.identifiers_included = options.include_identifiers;
}

[[nodiscard]] xvram::residency::Report make_base_report(const CliOptions& options) {
  xvram::residency::Report report;
  report.generated_at_utc = utc_timestamp();
  report.build = {XVRAM_VERSION, XVRAM_GIT_COMMIT, compiler_description(), build_type(),
                  XVRAM_CUDA_HEADERS_VERSION};
  report.system = xvram::platform::collect_system_info();
  apply_cli_configuration(report.configuration, options);
  report.outcome.status = "skipped";
  report.outcome.reason = "device_unavailable";
  report.outcome.exit_code = exit_prerequisite;
  report.outcome.stage = "planning";
  report.outcome.operation = "device_preflight";
  return report;
}

[[nodiscard]] bool valid_outcome_reason(const std::string_view reason) {
  return reason == "device_unavailable" || reason == "vmm_unsupported" ||
         reason == "uva_unsupported" || reason == "invalid_configuration" ||
         reason == "insufficient_host_memory" || reason == "insufficient_device_budget" ||
         reason == "working_set_too_large" || reason == "host_oom" ||
         reason == "device_oom" || reason == "budget_pressure" || reason == "timeout" ||
         reason == "data_mismatch" || reason == "cuda_error" ||
         reason == "platform_error" || reason == "protocol_error" ||
         reason == "trace_io_error" || reason == "cleanup_error" ||
         reason == "internal_error";
}

[[nodiscard]] std::string default_reason(const int exit_code) {
  switch (exit_code) {
  case exit_prerequisite:
    return "invalid_configuration";
  case exit_corruption:
    return "data_mismatch";
  case exit_oom:
    return "device_oom";
  case exit_timeout:
    return "timeout";
  case exit_io:
    return "trace_io_error";
  default:
    return "internal_error";
  }
}

[[nodiscard]] bool valid_outcome_stage(const std::string_view stage) {
  return stage == "planning" || stage == "reserve" || stage == "host_backing" ||
         stage == "physical_allocation" || stage == "mapping" ||
         stage == "cache_lookup" || stage == "prefetch" || stage == "transfer_h2d" ||
         stage == "kernel" || stage == "writeback" || stage == "transfer_d2h" ||
         stage == "verification" || stage == "budget" || stage == "unmapping" ||
         stage == "cleanup" || stage == "watchdog" || stage == "protocol" ||
         stage == "trace";
}

[[nodiscard]] bool valid_outcome_policy(const std::string_view policy) {
  return policy == "clock" || policy == "lru";
}

[[nodiscard]] bool valid_outcome_scenario(const std::string_view scenario) {
  return scenario == "sequential" || scenario == "reuse" || scenario == "random" ||
         scenario == "read-only" || scenario == "write-heavy" ||
         scenario == "budget-pressure";
}

void apply_executor_result(xvram::residency::Report& report,
                           const xvram::residency::ExecutorResult& execution,
                           const CliOptions& options) {
  report.configuration = execution.configuration;
  apply_cli_configuration(report.configuration, options);
  report.device = execution.device;
  report.workloads = execution.workloads;
  report.cache = execution.cache;
  report.telemetry = execution.telemetry;
  report.proof = execution.proof;
  report.cleanup = execution.cleanup;
  report.diagnostics = execution.diagnostics;

  report.outcome.exit_code = execution.exit_code;
  if (execution.exit_code == exit_completed) {
    report.outcome = {};
    report.outcome.status = "completed";
    report.outcome.exit_code = exit_completed;
  } else {
    report.outcome.status = execution.status == "skipped" ? "skipped" : "failed";
    if (execution.reason.has_value() && valid_outcome_reason(*execution.reason)) {
      report.outcome.reason = execution.reason;
    } else {
      report.outcome.reason = default_reason(execution.exit_code);
    }
  }

  if (execution.exit_code != exit_completed && execution.failure.has_value()) {
    const xvram::residency::ExecutionFailure& failure = *execution.failure;
    report.outcome.stage =
        valid_outcome_stage(failure.stage) ? failure.stage : std::string("planning");
    report.outcome.operation = failure.operation;
    report.outcome.native_code = failure.native_code;
    report.outcome.native_name = failure.native_name;
    report.outcome.message = failure.message;
    if (failure.policy.has_value() && valid_outcome_policy(*failure.policy)) {
      report.outcome.policy = failure.policy;
    }
    if (failure.scenario.has_value() && valid_outcome_scenario(*failure.scenario)) {
      report.outcome.scenario = failure.scenario;
    }
    report.outcome.allocation_id = failure.allocation_id;
    report.outcome.chunk_index = failure.chunk_index;
    report.outcome.logical_byte_offset = failure.logical_byte_offset;
  }
  report.cleanup.worker_terminated = true;
}

void mark_progress_snapshot(xvram::residency::Report& report) {
  for (xvram::residency::WorkloadResult& workload : report.workloads) {
    if (workload.status != "not_run" && workload.status != "completed" &&
        workload.status != "failed" && workload.status != "timed_out") {
      workload.status = "timed_out";
    }
  }
  report.outcome.status = "failed";
  report.outcome.reason = "timeout";
  report.outcome.exit_code = exit_timeout;
  report.outcome.stage = "watchdog";
  report.outcome.operation = "progress_snapshot";
  report.outcome.message = "the worker had not completed when this progress report was emitted";
  report.cleanup = {};
  report.cleanup.worker_terminated = true;
}

[[nodiscard]] std::string report_json(const xvram::residency::Report& report,
                                      const bool pretty) {
  std::ostringstream output;
  xvram::residency::write_json(report, output, pretty);
  return output.str();
}

[[nodiscard]] std::string report_text(const xvram::residency::Report& report) {
  std::ostringstream output;
  xvram::residency::write_text(report, output);
  return output.str();
}

[[nodiscard]] std::string trace_json(const xvram::residency::TraceRecord& record) {
  std::ostringstream output;
  xvram::residency::write_trace_json(record, output);
  return output.str();
}

[[nodiscard]] std::vector<std::string> worker_arguments(const CliOptions& options) {
  std::vector<std::string> arguments{
      "--worker",
      "--device",
      std::to_string(options.executor.device_ordinal),
      "--logical-size",
      options.executor.logical_bytes.has_value() ? std::to_string(*options.executor.logical_bytes)
                                                 : "auto",
      "--chunk-size",
      std::to_string(options.executor.chunk_bytes),
      "--cache-target",
      options.executor.cache_target_bytes.has_value()
          ? std::to_string(*options.executor.cache_target_bytes)
          : "auto",
      "--staging-slots",
      std::to_string(options.executor.staging_slots),
      "--policy",
      policy_name(options.executor.policy),
      "--prefetch-distance",
      std::to_string(options.executor.prefetch_distance),
      "--scenario",
      std::string(xvram::residency::scenario_kind_name(options.executor.scenario)),
      "--passes",
      std::to_string(options.executor.passes),
      "--pressure-size",
      options.executor.pressure_bytes.has_value() ? std::to_string(*options.executor.pressure_bytes)
                                                  : "auto",
      "--device-headroom",
      std::to_string(options.executor.device_headroom_bytes),
      "--budget-poll-ms",
      std::to_string(options.executor.budget_poll_interval.count()),
      "--stall-timeout-ms",
      std::to_string(options.executor.stall_timeout.count()),
      "--timeout-seconds",
      std::to_string(options.timeout.count()),
      "--seed",
      seed_string(options.executor.seed)};
  if (options.executor.trace_enabled) {
    arguments.emplace_back("--trace-enabled");
  }
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
    std::ofstream output(std::filesystem::path(*options.json_path),
                         std::ios::binary | std::ios::trunc);
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
  std::uint64_t frame_sequence = 1;
  std::mutex protocol_mutex;
  bool protocol_output_ok = true;
  const auto send_frame = [&](const xvram::residency::WorkerFrameType type,
                              const std::string_view payload) {
    std::scoped_lock lock(protocol_mutex);
    if (!protocol_output_ok) {
      return false;
    }
    protocol_output_ok = xvram::residency::write_worker_frame(
        std::cout, type, frame_sequence++, payload);
    return protocol_output_ok;
  };

  const xvram::residency::Report report_template = make_base_report(options);
  xvram::residency::Report initial = report_template;
  initial.outcome.message = "worker plan is awaiting CUDA and memory preflight";
  if (!send_frame(xvram::residency::WorkerFrameType::plan,
                  report_json(initial, options.pretty_json))) {
    return exit_failure;
  }

  const auto progress = [&](const xvram::residency::ExecutorResult& execution,
                            const std::uint64_t, const std::uint64_t) {
    xvram::residency::Report snapshot = report_template;
    apply_executor_result(snapshot, execution, options);
    mark_progress_snapshot(snapshot);
    static_cast<void>(send_frame(xvram::residency::WorkerFrameType::progress,
                                 report_json(snapshot, options.pretty_json)));
  };
  const auto trace = [&](const xvram::residency::TraceRecord& record) {
    const std::string json = trace_json(record);
    if (json.size() > xvram::residency::maximum_cache_trace_batch_bytes) {
      std::scoped_lock lock(protocol_mutex);
      protocol_output_ok = false;
      return;
    }
    static_cast<void>(send_frame(xvram::residency::WorkerFrameType::trace, json));
  };

  xvram::cuda::CudaApi api;
  xvram::residency::TraceCallback trace_callback;
  if (options.executor.trace_enabled) {
    trace_callback = trace;
  }
  const xvram::residency::ExecutorResult execution = xvram::residency::run_executor(
      api, options.executor, progress, trace_callback);
  xvram::residency::Report final_report = report_template;
  apply_executor_result(final_report, execution, options);
  {
    std::scoped_lock lock(protocol_mutex);
    final_report.cleanup.trace_closed = protocol_output_ok;
  }
  xvram::residency::FinalWorkerPayload payload;
  payload.exit_code = execution.exit_code;
  payload.json = report_json(final_report, options.pretty_json);
  payload.text = report_text(final_report);
  const std::string encoded = xvram::residency::encode_final_worker_payload(payload);
  if (encoded.empty() || !send_frame(xvram::residency::WorkerFrameType::final, encoded)) {
    return exit_failure;
  }
  return execution.exit_code;
}

[[nodiscard]] xvram::residency::Report
controller_failure_report(const CliOptions& options, const int exit_code, std::string reason,
                          std::string stage, std::string operation, std::string message,
                          const std::optional<bool> trace_closed = std::nullopt) {
  xvram::residency::Report report = make_base_report(options);
  report.outcome.status = "failed";
  report.outcome.reason = std::move(reason);
  report.outcome.exit_code = exit_code;
  report.outcome.stage = std::move(stage);
  report.outcome.operation = std::move(operation);
  report.outcome.message = std::move(message);
  report.cleanup = {};
  report.cleanup.trace_closed = trace_closed;
  report.cleanup.worker_terminated = true;
  return report;
}

[[nodiscard]] int controller_main(const CliOptions& options,
                                  const std::filesystem::path& executable) {
  std::ofstream trace_output;
  bool trace_io_failed = false;
  std::string trace_error;
  if (options.trace_path.has_value()) {
    trace_output.open(std::filesystem::path(*options.trace_path),
                      std::ios::binary | std::ios::trunc);
    if (!trace_output) {
      const xvram::residency::Report failure = controller_failure_report(
          options, exit_io, "trace_io_error", "trace", "open_trace",
          "the controller could not open the trace output file", false);
      if (!emit_outputs(options, report_json(failure, options.pretty_json),
                        report_text(failure))) {
        std::cerr << "error: failed while writing report output\n";
      }
      return exit_io;
    }
  }

  xvram::residency::WorkerControllerOptions controller;
  controller.executable = executable;
  controller.arguments = worker_arguments(options);
  controller.no_progress_timeout = std::chrono::seconds(15);
  controller.overall_timeout = options.timeout;
  if (options.trace_path.has_value()) {
    controller.trace_sink = [&](const std::string_view batch, std::string& error) {
      trace_output.write(batch.data(), static_cast<std::streamsize>(batch.size()));
      trace_output.flush();
      if (trace_output) {
        return true;
      }
      trace_io_failed = true;
      trace_error = "the controller failed while writing the trace output file";
      error = trace_error;
      return false;
    };
  }
  const xvram::residency::WorkerControllerResult worker =
      xvram::residency::run_isolated_worker(controller);

  if (trace_output.is_open()) {
    trace_output.flush();
    if (!trace_output) {
      trace_io_failed = true;
      trace_error = "the controller failed while flushing the trace output file";
    }
    trace_output.close();
    if (!trace_output && !trace_io_failed) {
      trace_io_failed = true;
      trace_error = "the controller failed while closing the trace output file";
    }
  }

  int exit_code = exit_failure;
  std::string json;
  std::string text;
  if (trace_io_failed) {
    exit_code = exit_io;
    const xvram::residency::Report failure = controller_failure_report(
        options, exit_io, "trace_io_error", "trace", "write_trace", trace_error, false);
    json = report_json(failure, options.pretty_json);
    text = report_text(failure);
  } else if (worker.timed_out) {
    exit_code = exit_timeout;
    const xvram::residency::Report timeout = controller_failure_report(
        options, exit_timeout, "timeout", "watchdog", "watchdog",
        "the isolated worker exceeded its progress or overall deadline and was terminated",
        true);
    json = report_json(timeout, options.pretty_json);
    text = report_text(timeout);
  } else if (!worker.started) {
    const xvram::residency::Report failure = controller_failure_report(
        options, exit_failure, "platform_error", "protocol", "start_worker",
        worker.error.empty() ? "the controller could not start the isolated worker"
                             : worker.error,
        true);
    json = report_json(failure, options.pretty_json);
    text = report_text(failure);
  } else if (worker.protocol_error || !worker.final.has_value()) {
    const xvram::residency::Report failure = controller_failure_report(
        options, exit_failure, "protocol_error", "protocol", "worker_protocol",
        worker.error.empty() ? "the isolated worker failed without a valid final report"
                             : worker.error,
        true);
    json = report_json(failure, options.pretty_json);
    text = report_text(failure);
  } else if (worker.process_exit_code != worker.final->exit_code) {
    const xvram::residency::Report failure = controller_failure_report(
        options, exit_failure, "protocol_error", "protocol", "worker_exit_code",
        "the worker process exit code does not match its final protocol report", true);
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
