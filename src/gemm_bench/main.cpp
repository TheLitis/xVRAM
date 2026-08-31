#include "gemm_bench/executor.hpp"
#include "gemm_bench/worker_controller.hpp"
#include "gemm_bench/worker_protocol.hpp"
#include "platform/system_info.hpp"
#include "xvram/base/size_parser.hpp"
#include "xvram/gemm/report.hpp"
#include "xvram/version.hpp"

#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <new>
#include <optional>
#include <sstream>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
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

enum class CliAction { run, help, version };

struct CliOptions {
  xvram::gemm_bench::ExecutorOptions executor;
  std::optional<std::string> json_path;
  bool pretty_json = true;
  bool print_text = true;
  bool worker = false;
  CliAction action = CliAction::run;
};

void print_help(std::ostream& output) {
  output << R"(xvram-gemm-bench - isolated tiled GEMM correctness and residency benchmark

Usage:
  xvram-gemm-bench [options]

Options:
  --device <ordinal>                       CUDA device ordinal (default: 0)
  --m <auto|n>                             GEMM M dimension (default: auto)
  --n <auto|n>                             GEMM N dimension (default: auto)
  --k <auto|n>                             GEMM K dimension (default: auto)
                                             M/N/K must be all auto or all explicit
  --suite                                  Run the FP16/BF16/FP32/FP64 suite (default)
  --data-type <suite|fp16|bf16|fp32|fp64>  Requested data type or full suite
  --compute-mode <auto|strict|tf32|fp64>   Requested accumulation mode (default: auto)
  --layout-a <row|column>                  A storage layout (default: row)
  --layout-b <row|column>                  B storage layout (default: row)
  --layout-c <row|column>                  C storage layout (default: row)
  --op-a <n|t>                             A matrix operation (default: n)
  --op-b <n|t>                             B matrix operation (default: n)
  --chunk-size <size>                      Residency chunk size (default: 64 MiB)
  --cache-target <auto|size>               Maximum VRAM cache target (default: auto)
  --workspace <size>                       cuBLAS workspace cap (default: 4 MiB)
  --staging-slots <2..8>                   Pinned staging buffers (default: 4)
  --passes <n>                             Passes per requested data type (default: 1)
  --device-headroom <size>                 Reserved device headroom (default: 512 MiB)
  --timeout-seconds <n>                    Controller and executor deadline (default: 300)
  --json <path|->                          Write xvram.gemm_bench v1 JSON
  --compact-json                           Disable JSON indentation
  --no-text                                Suppress the human-readable report
  --include-identifiers                    Include stable device identifiers
  --version                                Print xVRAM version
  -h, --help                               Show this help

The controller runs CUDA work in an isolated child process, enforces a 15-second progress
watchdog, and is the only process that writes user-selected report output.
)";
}

[[nodiscard]] bool parse_u32(const std::string_view text, std::uint32_t& output) {
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), output, 10);
  return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

[[nodiscard]] bool parse_u64(const std::string_view text, std::uint64_t& output) {
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), output, 10);
  return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

[[nodiscard]] bool parse_i32(const std::string_view text, std::int32_t& output) {
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), output, 10);
  return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

[[nodiscard]] std::optional<std::uint64_t> parse_size_value(const std::string_view text,
                                                            const bool allow_zero) {
  const xvram::ParsedSize parsed = xvram::parse_size(text);
  if (!parsed || (!allow_zero && *parsed.bytes == 0)) {
    return std::nullopt;
  }
  return *parsed.bytes;
}

[[nodiscard]] bool parse_data_type(const std::string_view text,
                                   xvram::gemm_bench::RequestedDataType& output) {
  if (text == "suite") {
    output = xvram::gemm_bench::RequestedDataType::suite;
  } else if (text == "fp16") {
    output = xvram::gemm_bench::RequestedDataType::fp16;
  } else if (text == "bf16") {
    output = xvram::gemm_bench::RequestedDataType::bf16;
  } else if (text == "fp32") {
    output = xvram::gemm_bench::RequestedDataType::fp32;
  } else if (text == "fp64") {
    output = xvram::gemm_bench::RequestedDataType::fp64;
  } else {
    return false;
  }
  return true;
}

[[nodiscard]] bool parse_compute_mode(const std::string_view text,
                                      xvram::gemm_bench::RequestedComputeMode& output) {
  if (text == "auto") {
    output = xvram::gemm_bench::RequestedComputeMode::automatic;
  } else if (text == "strict") {
    output = xvram::gemm_bench::RequestedComputeMode::strict_fp32;
  } else if (text == "tf32") {
    output = xvram::gemm_bench::RequestedComputeMode::tf32;
  } else if (text == "fp64") {
    output = xvram::gemm_bench::RequestedComputeMode::fp64;
  } else {
    return false;
  }
  return true;
}

[[nodiscard]] bool parse_layout(const std::string_view text,
                                xvram::gemm_bench::RequestedLayout& output) {
  if (text == "row") {
    output = xvram::gemm_bench::RequestedLayout::row_major;
  } else if (text == "column") {
    output = xvram::gemm_bench::RequestedLayout::column_major;
  } else {
    return false;
  }
  return true;
}

[[nodiscard]] bool parse_operation(const std::string_view text,
                                   xvram::gemm_bench::RequestedOperation& output) {
  if (text == "n") {
    output = xvram::gemm_bench::RequestedOperation::none;
  } else if (text == "t") {
    output = xvram::gemm_bench::RequestedOperation::transpose;
  } else {
    return false;
  }
  return true;
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
    if (argument == "--suite") {
      options.executor.data_type = xvram::gemm_bench::RequestedDataType::suite;
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
    if (argument == "--m" || argument == "--n" || argument == "--k") {
      const auto value = value_after(index, argument);
      if (!value.has_value()) {
        return false;
      }
      std::optional<std::uint64_t>* destination =
          argument == "--m" ? &options.executor.m
                            : (argument == "--n" ? &options.executor.n : &options.executor.k);
      if (*value == "auto") {
        destination->reset();
      } else {
        std::uint64_t parsed = 0;
        if (!parse_u64(*value, parsed) || parsed == 0) {
          error = std::string(argument) + " must be auto or a positive integer";
          return false;
        }
        *destination = parsed;
      }
      continue;
    }
    if (argument == "--data-type") {
      const auto value = value_after(index, argument);
      if (!value.has_value() || !parse_data_type(*value, options.executor.data_type)) {
        if (error.empty()) {
          error = "--data-type must be suite, fp16, bf16, fp32, or fp64";
        }
        return false;
      }
      continue;
    }
    if (argument == "--compute-mode") {
      const auto value = value_after(index, argument);
      if (!value.has_value() || !parse_compute_mode(*value, options.executor.compute_mode)) {
        if (error.empty()) {
          error = "--compute-mode must be auto, strict, tf32, or fp64";
        }
        return false;
      }
      continue;
    }
    if (argument == "--layout-a" || argument == "--layout-b" || argument == "--layout-c") {
      const auto value = value_after(index, argument);
      xvram::gemm_bench::RequestedLayout* destination =
          argument == "--layout-a" ? &options.executor.a_layout
                                   : (argument == "--layout-b" ? &options.executor.b_layout
                                                               : &options.executor.c_layout);
      if (!value.has_value() || !parse_layout(*value, *destination)) {
        if (error.empty()) {
          error = std::string(argument) + " must be row or column";
        }
        return false;
      }
      continue;
    }
    if (argument == "--op-a" || argument == "--op-b") {
      const auto value = value_after(index, argument);
      xvram::gemm_bench::RequestedOperation& destination =
          argument == "--op-a" ? options.executor.a_operation : options.executor.b_operation;
      if (!value.has_value() || !parse_operation(*value, destination)) {
        if (error.empty()) {
          error = std::string(argument) + " must be n or t";
        }
        return false;
      }
      continue;
    }
    if (argument == "--cache-target") {
      const auto value = value_after(index, argument);
      if (!value.has_value()) {
        return false;
      }
      if (*value == "auto") {
        options.executor.cache_target_bytes.reset();
      } else {
        const auto parsed = parse_size_value(*value, false);
        if (!parsed.has_value()) {
          error = "--cache-target must be auto or a positive size";
          return false;
        }
        options.executor.cache_target_bytes = *parsed;
      }
      continue;
    }
    if (argument == "--chunk-size" || argument == "--workspace" ||
        argument == "--device-headroom") {
      const auto value = value_after(index, argument);
      const bool allow_zero = argument != "--chunk-size";
      const auto parsed = value.has_value() ? parse_size_value(*value, allow_zero) : std::nullopt;
      if (!parsed.has_value()) {
        if (error.empty()) {
          error = std::string(argument) +
                  (allow_zero ? " must be a non-negative size" : " must be a positive size");
        }
        return false;
      }
      if (argument == "--chunk-size") {
        options.executor.chunk_bytes = *parsed;
      } else if (argument == "--workspace") {
        options.executor.workspace_bytes = *parsed;
      } else {
        options.executor.device_headroom_bytes = *parsed;
      }
      continue;
    }
    if (argument == "--staging-slots" || argument == "--passes" ||
        argument == "--timeout-seconds") {
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
      } else if (argument == "--passes") {
        if (parsed == 0) {
          error = "--passes must be at least 1";
          return false;
        }
        options.executor.passes = parsed;
      } else {
        if (parsed == 0) {
          error = "--timeout-seconds must be positive";
          return false;
        }
        options.executor.timeout = std::chrono::seconds(parsed);
      }
      continue;
    }
    if (argument == "--json") {
      const auto value = value_after(index, argument);
      if (!value.has_value() || value->empty()) {
        if (error.empty()) {
          error = "--json requires a non-empty path";
        }
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
  const unsigned explicit_dimensions = static_cast<unsigned>(options.executor.m.has_value()) +
                                       static_cast<unsigned>(options.executor.n.has_value()) +
                                       static_cast<unsigned>(options.executor.k.has_value());
  if (explicit_dimensions != 0U && explicit_dimensions != 3U) {
    error = "--m, --n, and --k must be all auto or all explicit";
    return false;
  }
  if (!options.print_text && !options.json_path.has_value() && !options.worker) {
    error = "--no-text requires --json";
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

[[nodiscard]] const char*
cli_compute_mode_name(const xvram::gemm_bench::RequestedComputeMode mode) noexcept {
  switch (mode) {
  case xvram::gemm_bench::RequestedComputeMode::automatic:
    return "auto";
  case xvram::gemm_bench::RequestedComputeMode::strict_fp32:
    return "strict";
  case xvram::gemm_bench::RequestedComputeMode::tf32:
    return "tf32";
  case xvram::gemm_bench::RequestedComputeMode::fp64:
    return "fp64";
  }
  return "auto";
}

[[nodiscard]] const char*
cli_layout_name(const xvram::gemm_bench::RequestedLayout layout) noexcept {
  return layout == xvram::gemm_bench::RequestedLayout::column_major ? "column" : "row";
}

void apply_cli_configuration(xvram::gemm_bench::Configuration& configuration,
                             const CliOptions& options) {
  configuration.requested_device_ordinal = options.executor.device_ordinal;
  configuration.requested_m = options.executor.m;
  configuration.requested_n = options.executor.n;
  configuration.requested_k = options.executor.k;
  configuration.scenario =
      options.executor.data_type == xvram::gemm_bench::RequestedDataType::suite ? "suite" : "gemm";
  const xvram::gemm_bench::RequestedDataType configuration_type =
      options.executor.data_type == xvram::gemm_bench::RequestedDataType::suite
          ? xvram::gemm_bench::RequestedDataType::fp32
          : options.executor.data_type;
  configuration.a_data_type = xvram::gemm_bench::requested_data_type_name(configuration_type);
  configuration.b_data_type = configuration.a_data_type;
  configuration.c_data_type = configuration.a_data_type;
  configuration.compute_mode =
      xvram::gemm_bench::requested_compute_mode_name(options.executor.compute_mode);
  configuration.a_layout = xvram::gemm_bench::requested_layout_name(options.executor.a_layout);
  configuration.b_layout = xvram::gemm_bench::requested_layout_name(options.executor.b_layout);
  configuration.c_layout = xvram::gemm_bench::requested_layout_name(options.executor.c_layout);
  configuration.a_operation =
      xvram::gemm_bench::requested_operation_name(options.executor.a_operation);
  configuration.b_operation =
      xvram::gemm_bench::requested_operation_name(options.executor.b_operation);
  configuration.requested_chunk_bytes = options.executor.chunk_bytes;
  configuration.requested_cache_target_bytes = options.executor.cache_target_bytes;
  configuration.requested_workspace_cap_bytes = options.executor.workspace_bytes;
  configuration.staging_slots = options.executor.staging_slots;
  configuration.passes = options.executor.passes;
  configuration.timeout_ms = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(options.executor.timeout).count());
  configuration.identifiers_included = options.executor.include_identifiers;
}

[[nodiscard]] xvram::gemm_bench::Report make_base_report(const CliOptions& options) {
  xvram::gemm_bench::Report report;
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

void mark_progress_snapshot(xvram::gemm_bench::Report& report) {
  for (xvram::gemm_bench::WorkloadResult& workload : report.workloads) {
    if (workload.status == "running") {
      workload.status = "timed_out";
    }
  }
  report.outcome.status = "timed_out";
  report.outcome.reason = "timeout";
  report.outcome.exit_code = exit_timeout;
  report.outcome.stage = "watchdog";
  report.outcome.operation = "progress_snapshot";
  report.outcome.message = "the worker had not completed when this progress report was emitted";
  report.cleanup = {};
  report.cleanup.worker_terminated = true;
}

[[nodiscard]] std::string report_json(const xvram::gemm_bench::Report& report, const bool pretty) {
  std::ostringstream output;
  xvram::gemm_bench::write_json(report, output, pretty);
  return output.str();
}

[[nodiscard]] std::string report_text(const xvram::gemm_bench::Report& report) {
  std::ostringstream output;
  xvram::gemm_bench::write_text(report, output);
  return output.str();
}

[[nodiscard]] std::string dimension_argument(const std::optional<std::uint64_t>& value) {
  return value.has_value() ? std::to_string(*value) : "auto";
}

[[nodiscard]] std::vector<std::string> worker_arguments(const CliOptions& options) {
  std::vector<std::string> arguments{
      "--worker",
      "--device",
      std::to_string(options.executor.device_ordinal),
      "--m",
      dimension_argument(options.executor.m),
      "--n",
      dimension_argument(options.executor.n),
      "--k",
      dimension_argument(options.executor.k),
      "--data-type",
      xvram::gemm_bench::requested_data_type_name(options.executor.data_type),
      "--compute-mode",
      cli_compute_mode_name(options.executor.compute_mode),
      "--layout-a",
      cli_layout_name(options.executor.a_layout),
      "--layout-b",
      cli_layout_name(options.executor.b_layout),
      "--layout-c",
      cli_layout_name(options.executor.c_layout),
      "--op-a",
      xvram::gemm_bench::requested_operation_name(options.executor.a_operation),
      "--op-b",
      xvram::gemm_bench::requested_operation_name(options.executor.b_operation),
      "--chunk-size",
      std::to_string(options.executor.chunk_bytes),
      "--cache-target",
      options.executor.cache_target_bytes.has_value()
          ? std::to_string(*options.executor.cache_target_bytes)
          : "auto",
      "--workspace",
      std::to_string(options.executor.workspace_bytes),
      "--staging-slots",
      std::to_string(options.executor.staging_slots),
      "--passes",
      std::to_string(options.executor.passes),
      "--device-headroom",
      std::to_string(options.executor.device_headroom_bytes),
      "--timeout-seconds",
      std::to_string(options.executor.timeout.count())};
  if (options.executor.include_identifiers) {
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
  const auto send_frame = [&](const xvram::gemm_bench::WorkerFrameType type,
                              const std::string_view payload) {
    std::scoped_lock lock(protocol_mutex);
    if (!protocol_output_ok) {
      return false;
    }
    protocol_output_ok =
        xvram::gemm_bench::write_worker_frame(std::cout, type, frame_sequence++, payload);
    return protocol_output_ok;
  };

  xvram::gemm_bench::Report initial = make_base_report(options);
  initial.outcome.message = "worker plan is awaiting CUDA and memory preflight";
  if (!send_frame(xvram::gemm_bench::WorkerFrameType::plan,
                  report_json(initial, options.pretty_json))) {
    return exit_failure;
  }

  xvram::gemm_bench::Report latest_snapshot = initial;
  mark_progress_snapshot(latest_snapshot);
  std::mutex snapshot_mutex;
  std::mutex heartbeat_wait_mutex;
  std::condition_variable heartbeat_wakeup;
  std::jthread heartbeat([&](const std::stop_token stop) {
    const std::stop_callback wake_on_stop(stop, [&] { heartbeat_wakeup.notify_all(); });
    std::unique_lock wait_lock(heartbeat_wait_mutex);
    while (!stop.stop_requested()) {
      if (heartbeat_wakeup.wait_for(wait_lock, std::chrono::seconds(1)) !=
          std::cv_status::timeout) {
        continue;
      }
      wait_lock.unlock();
      try {
        xvram::gemm_bench::Report snapshot;
        {
          std::scoped_lock snapshot_lock(snapshot_mutex);
          snapshot = latest_snapshot;
        }
        if (!send_frame(xvram::gemm_bench::WorkerFrameType::progress,
                        report_json(snapshot, options.pretty_json))) {
          return;
        }
      } catch (...) {
        std::scoped_lock protocol_lock(protocol_mutex);
        protocol_output_ok = false;
        return;
      }
      wait_lock.lock();
    }
  });

  const auto progress = [&](const xvram::gemm_bench::Report& execution,
                            const std::uint64_t completed, const std::uint64_t total) {
    xvram::gemm_bench::Report snapshot = execution;
    mark_progress_snapshot(snapshot);
    snapshot.outcome.message = "worker made measurable progress: " + std::to_string(completed) +
                               "/" + std::to_string(total);
    std::scoped_lock snapshot_lock(snapshot_mutex);
    latest_snapshot = std::move(snapshot);
  };
  xvram::gemm_bench::Report final_report;
  try {
    final_report = xvram::gemm_bench::run_executor(options.executor, progress);
  } catch (const std::bad_alloc&) {
    final_report = make_base_report(options);
    final_report.outcome.status = "failed";
    final_report.outcome.reason = "host_oom";
    final_report.outcome.exit_code = exit_oom;
    final_report.outcome.stage = "worker";
    final_report.outcome.operation = "run_executor";
    final_report.outcome.message = "the worker exhausted host memory outside the SDK boundary";
  } catch (const std::exception& exception) {
    final_report = make_base_report(options);
    final_report.outcome.status = "failed";
    final_report.outcome.reason = "internal_error";
    final_report.outcome.exit_code = exit_internal;
    final_report.outcome.stage = "worker";
    final_report.outcome.operation = "run_executor";
    final_report.outcome.message = exception.what();
  } catch (...) {
    final_report = make_base_report(options);
    final_report.outcome.status = "failed";
    final_report.outcome.reason = "internal_error";
    final_report.outcome.exit_code = exit_internal;
    final_report.outcome.stage = "worker";
    final_report.outcome.operation = "run_executor";
    final_report.outcome.message = "the worker caught an unknown internal exception";
  }
  heartbeat.request_stop();
  heartbeat.join();
  final_report.cleanup.worker_terminated = true;
  xvram::gemm_bench::FinalWorkerPayload payload;
  payload.exit_code = final_report.outcome.exit_code;
  payload.json = report_json(final_report, options.pretty_json);
  payload.text = report_text(final_report);
  const std::string encoded = xvram::gemm_bench::encode_final_worker_payload(payload);
  if (encoded.empty() || !send_frame(xvram::gemm_bench::WorkerFrameType::final, encoded)) {
    return exit_failure;
  }
  return payload.exit_code;
}

[[nodiscard]] xvram::gemm_bench::Report
controller_failure_report(const CliOptions& options, const int exit_code, std::string reason,
                          std::string stage, std::string operation, std::string message) {
  xvram::gemm_bench::Report report = make_base_report(options);
  report.outcome.status = exit_code == exit_timeout ? "timed_out" : "failed";
  report.outcome.reason = std::move(reason);
  report.outcome.exit_code = exit_code;
  report.outcome.stage = std::move(stage);
  report.outcome.operation = std::move(operation);
  report.outcome.message = std::move(message);
  report.cleanup = {};
  report.cleanup.worker_terminated = true;
  return report;
}

[[nodiscard]] bool valid_worker_exit_code(const int exit_code) noexcept {
  return exit_code == exit_completed || exit_code == exit_prerequisite ||
         exit_code == exit_corruption || exit_code == exit_oom || exit_code == exit_timeout ||
         exit_code == exit_failure || exit_code == exit_internal;
}

[[nodiscard]] int controller_main(const CliOptions& options,
                                  const std::filesystem::path& executable) {
  xvram::gemm_bench::WorkerControllerOptions controller;
  controller.executable = executable;
  controller.arguments = worker_arguments(options);
  controller.no_progress_timeout = std::chrono::seconds(15);
  controller.overall_timeout = options.executor.timeout;
  const xvram::gemm_bench::WorkerControllerResult worker =
      xvram::gemm_bench::run_isolated_worker(controller);

  int exit_code = exit_failure;
  std::string json;
  std::string text;
  if (worker.timed_out) {
    exit_code = exit_timeout;
    const xvram::gemm_bench::Report timeout = controller_failure_report(
        options, exit_timeout, "timeout", "watchdog", "watchdog",
        "the isolated worker exceeded its progress or overall deadline and was terminated");
    json = report_json(timeout, options.pretty_json);
    text = report_text(timeout);
  } else if (!worker.started) {
    const xvram::gemm_bench::Report failure = controller_failure_report(
        options, exit_failure, "platform_error", "protocol", "start_worker",
        worker.error.empty() ? "the controller could not start the isolated worker" : worker.error);
    json = report_json(failure, options.pretty_json);
    text = report_text(failure);
  } else if (worker.protocol_error || !worker.final.has_value()) {
    const xvram::gemm_bench::Report failure = controller_failure_report(
        options, exit_failure, "protocol_error", "protocol", "worker_protocol",
        worker.error.empty() ? "the isolated worker failed without a valid final report"
                             : worker.error);
    json = report_json(failure, options.pretty_json);
    text = report_text(failure);
  } else if (worker.process_exit_code != worker.final->exit_code ||
             !valid_worker_exit_code(worker.final->exit_code)) {
    const xvram::gemm_bench::Report failure = controller_failure_report(
        options, exit_failure, "protocol_error", "protocol", "worker_exit_code",
        "the worker process exit code is invalid or does not match its final protocol report");
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
