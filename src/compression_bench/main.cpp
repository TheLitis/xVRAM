#include "compression_bench/executor.hpp"
#include "compression_bench/options.hpp"
#include "compression_bench/worker_controller.hpp"
#include "compression_bench/worker_protocol.hpp"
#include "platform/system_info.hpp"
#include "xvram/compression/report.hpp"
#include "xvram/version.hpp"

#include <chrono>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <new>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
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

[[nodiscard]] std::string seed_string(const std::uint64_t value) {
  std::ostringstream output;
  output << std::hex << std::nouppercase << std::setfill('0') << std::setw(16) << value;
  return output.str();
}

void apply_cli_configuration(xvram::compression::Configuration& configuration,
                             const xvram::compression::CliOptions& options) {
  const xvram::compression::ExecutorOptions& value = options.executor;
  configuration.requested_device_ordinal = value.device_ordinal;
  configuration.requested_logical_bytes = value.logical_bytes;
  configuration.requested_chunk_bytes = value.chunk_bytes;
  configuration.requested_cache_target_bytes = value.cache_target_bytes;
  configuration.compression_policy =
      std::string(xvram::compression::compression_policy_name(value.compression_policy));
  configuration.path = std::string(xvram::compression::path_name(value.path));
  configuration.codec = std::string(xvram::residency::compression_codec_name(value.codec));
  configuration.replacement_policy =
      std::string(xvram::compression::replacement_policy_name(value.replacement_policy));
  configuration.scenario = std::string(xvram::compression::scenario_name(value.scenario));
  configuration.passes = value.passes;
  configuration.warmup_passes = value.warmup_passes;
  configuration.measurement_passes = value.measurement_passes;
  configuration.host_store_cap_bytes = value.host_store_cap_bytes;
  configuration.host_headroom_bytes = value.host_headroom_bytes.value_or(0U);
  configuration.device_headroom_bytes = value.device_headroom_bytes;
  configuration.compression_scratch_cap_bytes = value.compression_scratch_cap_bytes;
  configuration.codec_slots = value.codec_slots;
  configuration.codec_workers = value.codec_workers;
  configuration.staging_slots = value.staging_slots;
  configuration.prefetch_distance = value.prefetch_distance;
  configuration.budget_poll_ms = static_cast<std::uint64_t>(value.budget_poll_interval.count());
  configuration.stall_timeout_ms = static_cast<std::uint64_t>(value.stall_timeout.count());
  configuration.timeout_ms = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(options.timeout).count());
  configuration.seed_hex = seed_string(value.seed);
  configuration.sizing_mode = value.logical_bytes.has_value() ? "explicit" : "auto";
  configuration.trace_enabled = value.trace_enabled;
  configuration.identifiers_included = value.include_identifiers;
}

[[nodiscard]] xvram::compression::Report
make_base_report(const xvram::compression::CliOptions& options) {
  xvram::compression::Report report;
  report.generated_at_utc = utc_timestamp();
  report.build = {XVRAM_VERSION, XVRAM_GIT_COMMIT, compiler_description(), build_type(),
                  XVRAM_CUDA_HEADERS_VERSION};
  report.system = xvram::platform::collect_system_info();
  apply_cli_configuration(report.configuration, options);
  report.outcome.status = "skipped";
  report.outcome.reason = "worker_not_started";
  report.outcome.exit_code = exit_prerequisite;
  report.outcome.stage = "planning";
  report.outcome.operation = "worker_preflight";
  return report;
}

[[nodiscard]] std::string outcome_status(const int exit_code) {
  switch (exit_code) {
  case exit_completed:
    return "completed";
  case exit_prerequisite:
    return "skipped";
  case exit_corruption:
    return "corruption";
  case exit_oom:
    return "oom";
  case exit_timeout:
    return "timeout";
  default:
    return "failed";
  }
}

void apply_executor_result(xvram::compression::Report& report,
                           const xvram::compression::ExecutorResult& execution,
                           const xvram::compression::CliOptions& options) {
  report.configuration = execution.configuration;
  report.configuration.timeout_ms = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(options.timeout).count());
  report.configuration.trace_enabled = options.executor.trace_enabled;
  report.configuration.identifiers_included = options.executor.include_identifiers;
  report.device = execution.device;
  report.workloads = execution.workloads;
  report.backing = execution.backing;
  report.codec = execution.codec;
  report.telemetry = execution.telemetry;
  report.proof = execution.proof;
  report.cleanup = execution.cleanup;
  report.diagnostics = execution.diagnostics;
  report.outcome.status = outcome_status(execution.exit_code);
  report.outcome.reason = execution.reason;
  report.outcome.exit_code = execution.exit_code;
  if (execution.failure.has_value()) {
    const xvram::compression::ExecutionFailure& failure = *execution.failure;
    report.outcome.stage = failure.stage;
    report.outcome.operation = failure.operation;
    report.outcome.native_code = failure.native_code;
    report.outcome.native_name = failure.native_name;
    report.outcome.message = failure.message;
    report.outcome.scenario = failure.scenario;
    report.outcome.allocation_id = failure.allocation_id;
    report.outcome.chunk_index = failure.chunk_index;
    report.outcome.generation = failure.generation;
    report.outcome.logical_byte_offset = failure.logical_byte_offset;
  } else {
    report.outcome.stage.reset();
    report.outcome.operation.reset();
    report.outcome.native_code.reset();
    report.outcome.native_name.reset();
    report.outcome.message.reset();
    report.outcome.scenario.reset();
    report.outcome.allocation_id.reset();
    report.outcome.chunk_index.reset();
    report.outcome.generation.reset();
    report.outcome.logical_byte_offset.reset();
  }
  xvram::compression::finalize_proof(report);
}

void mark_progress_snapshot(xvram::compression::Report& report) {
  report.outcome.status = "failed";
  report.outcome.reason = "in_progress";
  report.outcome.exit_code = exit_failure;
  report.outcome.stage = "execution";
  report.outcome.operation = "worker_progress";
  report.outcome.message = "isolated compression worker is still executing";
  report.cleanup.complete = false;
  report.cleanup.worker_terminated = false;
}

[[nodiscard]] std::string report_json(const xvram::compression::Report& report, const bool pretty) {
  std::ostringstream output;
  xvram::compression::write_json(report, output, pretty);
  return output.str();
}

[[nodiscard]] std::string report_text(const xvram::compression::Report& report) {
  std::ostringstream output;
  xvram::compression::write_text(report, output);
  return output.str();
}

[[nodiscard]] std::string trace_json(const xvram::compression::TraceRecord& record) {
  std::ostringstream output;
  xvram::compression::write_trace_json(record, output);
  return output.str();
}

[[nodiscard]] bool emit_outputs(const xvram::compression::CliOptions& options,
                                const std::string& json, const std::string& text) {
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

[[nodiscard]] int worker_main(const xvram::compression::CliOptions& options) {
#ifdef _WIN32
  if (_setmode(_fileno(stdout), _O_BINARY) == -1) {
    return exit_failure;
  }
#endif
  std::uint64_t frame_sequence = 1;
  std::mutex protocol_mutex;
  bool protocol_output_ok = true;
  const auto send_frame = [&](const xvram::compression::WorkerFrameType type,
                              const std::string_view payload) {
    std::scoped_lock lock(protocol_mutex);
    if (!protocol_output_ok) {
      return false;
    }
    protocol_output_ok =
        xvram::compression::write_worker_frame(std::cout, type, frame_sequence++, payload);
    return protocol_output_ok;
  };

  const xvram::compression::Report report_template = make_base_report(options);
  xvram::compression::Report initial = report_template;
  initial.outcome.reason = "awaiting_preflight";
  initial.outcome.message = "worker plan is awaiting CUDA, codec, and memory preflight";
  if (!send_frame(xvram::compression::WorkerFrameType::plan,
                  report_json(initial, options.pretty_json))) {
    return exit_failure;
  }

  std::jthread heartbeat([&](const std::stop_token stop) {
    while (!stop.stop_requested()) {
      for (unsigned int tick = 0; tick < 10U && !stop.stop_requested(); ++tick) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      if (!stop.stop_requested()) {
        static_cast<void>(send_frame(xvram::compression::WorkerFrameType::heartbeat, "{}"));
      }
    }
  });

  const auto progress = [&](const xvram::compression::ExecutorResult& execution,
                            const std::uint64_t, const std::uint64_t) {
    xvram::compression::Report snapshot = report_template;
    apply_executor_result(snapshot, execution, options);
    mark_progress_snapshot(snapshot);
    static_cast<void>(send_frame(xvram::compression::WorkerFrameType::progress,
                                 report_json(snapshot, options.pretty_json)));
  };
  const auto trace = [&](const xvram::compression::TraceRecord& record) {
    const std::string json = trace_json(record);
    if (json.size() > xvram::compression::maximum_trace_batch_bytes) {
      std::scoped_lock lock(protocol_mutex);
      protocol_output_ok = false;
      throw std::runtime_error("compression trace record exceeds the protocol limit");
    }
    if (!send_frame(xvram::compression::WorkerFrameType::trace, json)) {
      throw std::runtime_error("compression trace frame could not be delivered");
    }
  };

  xvram::compression::TraceCallback trace_callback;
  if (options.executor.trace_enabled) {
    trace_callback = trace;
  }
  const xvram::compression::ExecutorResult execution =
      xvram::compression::run_executor(options.executor, progress, trace_callback);
  heartbeat.request_stop();
  heartbeat.join();

  xvram::compression::Report final_report = report_template;
  apply_executor_result(final_report, execution, options);
  {
    std::scoped_lock lock(protocol_mutex);
    final_report.cleanup.trace_closed = protocol_output_ok;
  }
  if (final_report.outcome.exit_code == exit_completed) {
    const std::vector<std::string> semantic_errors =
        xvram::compression::validate_success_semantics(final_report);
    if (!semantic_errors.empty()) {
      final_report.outcome.status = "failed";
      final_report.outcome.reason = "semantic_contract_failure";
      final_report.outcome.exit_code = exit_failure;
      final_report.outcome.stage = "report";
      final_report.outcome.operation = "validate_success_semantics";
      final_report.outcome.message = semantic_errors.front();
    }
  }

  xvram::compression::FinalWorkerPayload payload;
  payload.exit_code = final_report.outcome.exit_code;
  payload.json = report_json(final_report, options.pretty_json);
  payload.text = report_text(final_report);
  const std::string encoded = xvram::compression::encode_final_worker_payload(payload);
  if (encoded.empty() || !send_frame(xvram::compression::WorkerFrameType::final, encoded)) {
    return exit_failure;
  }
  return final_report.outcome.exit_code;
}

[[nodiscard]] xvram::compression::Report
controller_failure_report(const xvram::compression::CliOptions& options, const int exit_code,
                          std::string reason, std::string stage, std::string operation,
                          std::string message,
                          const std::optional<bool> trace_closed = std::nullopt,
                          const std::uint64_t trace_records_dropped = 0U) {
  xvram::compression::Report report = make_base_report(options);
  report.outcome.status = outcome_status(exit_code);
  report.outcome.reason = std::move(reason);
  report.outcome.exit_code = exit_code;
  report.outcome.stage = std::move(stage);
  report.outcome.operation = std::move(operation);
  report.outcome.message = std::move(message);
  report.cleanup = {};
  report.cleanup.trace_closed = trace_closed;
  report.cleanup.worker_terminated = true;
  if (!options.executor.trace_enabled) {
    report.telemetry.trace_records_emitted = 0U;
  }
  report.telemetry.trace_records_dropped = trace_records_dropped;
  report.telemetry.trace_complete = !options.executor.trace_enabled;
  return report;
}

[[nodiscard]] int controller_main(const xvram::compression::CliOptions& options,
                                  const std::filesystem::path& executable) {
  std::ofstream trace_output;
  bool trace_io_failed = false;
  std::string trace_error;
  if (options.trace_path.has_value()) {
    trace_output.open(std::filesystem::path(*options.trace_path),
                      std::ios::binary | std::ios::trunc);
    if (!trace_output) {
      const xvram::compression::Report failure = controller_failure_report(
          options, exit_io, "trace_io_error", "trace", "open_trace",
          "the controller could not open the trace output file", false, 1U);
      if (!emit_outputs(options, report_json(failure, options.pretty_json), report_text(failure))) {
        std::cerr << "error: failed while writing report output\n";
      }
      return exit_io;
    }
  }

  xvram::compression::WorkerControllerOptions controller;
  controller.executable = executable;
  controller.arguments = xvram::compression::worker_arguments(options);
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
  const xvram::compression::WorkerControllerResult worker =
      xvram::compression::run_isolated_worker(controller);
  if (trace_output.is_open()) {
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
    const auto failure = controller_failure_report(options, exit_io, "trace_io_error", "trace",
                                                   "write_trace", trace_error, false, 1U);
    json = report_json(failure, options.pretty_json);
    text = report_text(failure);
  } else if (worker.timed_out) {
    exit_code = exit_timeout;
    const auto timeout = controller_failure_report(
        options, exit_timeout, "timeout", "watchdog", "worker_watchdog",
        "the isolated worker exceeded its no-progress or overall deadline and was terminated",
        true);
    json = report_json(timeout, options.pretty_json);
    text = report_text(timeout);
  } else if (!worker.started) {
    const auto failure = controller_failure_report(
        options, exit_failure, "platform_error", "protocol", "start_worker",
        worker.error.empty() ? "the controller could not start the isolated worker" : worker.error,
        true);
    json = report_json(failure, options.pretty_json);
    text = report_text(failure);
  } else if (worker.protocol_error || !worker.final.has_value()) {
    const auto failure = controller_failure_report(
        options, exit_failure, "protocol_error", "protocol", "worker_protocol",
        worker.error.empty() ? "the isolated worker failed without a valid final report"
                             : worker.error,
        true);
    json = report_json(failure, options.pretty_json);
    text = report_text(failure);
  } else if (worker.process_exit_code != worker.final->exit_code) {
    const auto failure = controller_failure_report(
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

} // namespace

int main(const int argc, char** argv) {
  xvram::compression::CliOptions options;
  std::string parse_error;
  if (!xvram::compression::parse_cli(argc, argv, options, parse_error)) {
    std::cerr << "error: " << parse_error << "\n\n";
    xvram::compression::print_help(std::cerr);
    return exit_usage;
  }
  if (options.action == xvram::compression::CliAction::help) {
    xvram::compression::print_help(std::cout);
    return exit_completed;
  }
  if (options.action == xvram::compression::CliAction::version) {
    std::cout << XVRAM_VERSION << '\n';
    return exit_completed;
  }
  try {
    if (options.worker) {
      return worker_main(options);
    }
    const std::filesystem::path executable =
        xvram::compression::current_executable_path(argc > 0 ? argv[0] : nullptr);
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
