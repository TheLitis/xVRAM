#include "compat_bench/executor.hpp"
#include "compat_bench/final_validation.hpp"
#include "compat_bench/worker_controller.hpp"
#include "xvram/version.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <fcntl.h>
#include <io.h>
#else
#include <unistd.h>
#endif

namespace xvram::compat_bench {
namespace {
using Clock = std::chrono::steady_clock;
std::filesystem::path executable_path(const char* argv_zero) {
#ifdef _WIN32
  std::array<wchar_t, 32768> path{};
  const auto count = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
  if (count > 0 && count < path.size())
    return std::filesystem::path(std::wstring(path.data(), count));
#else
  std::array<char, 4096> path{};
  const auto count = readlink("/proc/self/exe", path.data(), path.size());
  if (count > 0 && static_cast<std::size_t>(count) < path.size())
    return std::filesystem::path(std::string(path.data(), static_cast<std::size_t>(count)));
#endif
  return argv_zero ? std::filesystem::absolute(argv_zero) : std::filesystem::path{};
}
int test_worker(const Options& options) {
  const auto& mode = options.test_worker;
  if (mode == "oversized") {
    const std::array<char, 20> header{'X', 'V', 'I', '1', 1, 0, 1, 0, 1, 0,
                                      16,  0,   1,   0,   0, 0, 0, 0, 0, 0};
    std::cout.write(header.data(), header.size());
    std::cout.flush();
    return 0;
  }
  if (mode == "truncated") {
    std::cout << "XVI1";
    std::cout.flush();
    return 0;
  }
  auto report = base_report(options);
  report.reason = "no_device_fixture";
  report.message = "isolated protocol test worker; no CUDA work was requested";
  if (!write_worker_frame(std::cout, WorkerFrameType::plan, 1, report_json(report, false)))
    return 74;
  if (mode == "crash")
    std::_Exit(42);
  if (mode == "hang" || mode == "heartbeat-hang") {
    std::uint64_t sequence = 2;
    for (;;) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      if (mode == "heartbeat-hang" &&
          !write_worker_frame(std::cout, WorkerFrameType::heartbeat, sequence++, "{}"))
        return 74;
    }
  }
  if (mode == "non-monotonic") {
    static_cast<void>(write_worker_frame(std::cout, WorkerFrameType::progress, 1, "1"));
    return 0;
  }
  if (mode == "stale-progress") {
    static_cast<void>(write_worker_frame(std::cout, WorkerFrameType::progress, 2, "1"));
    static_cast<void>(write_worker_frame(std::cout, WorkerFrameType::progress, 3, "1"));
    return 0;
  }
  if (mode == "trace") {
    static_cast<void>(
        write_worker_frame(std::cout, WorkerFrameType::trace, 2,
                           trace_record(1, 1, "call", 1, std::nullopt, 0, 0, "fixture")));
  } else if (mode != "success" && mode != "final-exit-mismatch" && mode != "contradictory-report" &&
             mode != "malformed-json" && mode != "duplicate-json-key")
    return 64;
  std::string final_json = report_json(report, false);
  if (mode == "contradictory-report") {
    report.exit_code = 27;
    final_json = report_json(report, false);
  } else if (mode == "malformed-json") {
    final_json = "{broken";
  } else if (mode == "duplicate-json-key") {
    final_json.insert(1, "\"schema_version\":1,");
  }
  const auto payload = encode_final_worker_payload({23, final_json, report_text(report)});
  if (!write_worker_frame(std::cout, WorkerFrameType::final, 3, payload))
    return 74;
  return mode == "final-exit-mismatch" ? 0 : 23;
}
int worker_main(const Options& options) {
#ifdef _WIN32
  if (_setmode(_fileno(stdout), _O_BINARY) == -1)
    return 27;
#endif
  if (!options.test_worker.empty())
    return test_worker(options);
  std::mutex output_mutex;
  std::uint64_t frame_sequence = 1, trace_sequence = 0, last_tiles = 0, progress = 0,
                last_runtime_progress = 0;
  std::uint64_t active_operation = 0;
  bool output_ok = true;
  const auto start = Clock::now();
  const auto send_unlocked = [&](WorkerFrameType type, std::string_view payload) {
    if (output_ok)
      output_ok = write_worker_frame(std::cout, type, frame_sequence++, payload);
    return output_ok;
  };
  const auto emit_unlocked = [&](std::string_view transition, std::uint64_t operation,
                                 std::optional<std::uint64_t> allocation, std::uint64_t bytes,
                                 std::uint64_t tiles, std::string_view reason) {
    if (!options.trace_path)
      return;
    const auto timestamp = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
    static_cast<void>(send_unlocked(WorkerFrameType::trace,
                                    trace_record(++trace_sequence, timestamp, transition, operation,
                                                 allocation, bytes, tiles, reason)));
  };
  const auto initial = base_report(options);
  if (!send_unlocked(WorkerFrameType::plan, report_json(initial, false)))
    return 27;
  xvram_cuda_compat_api_v1 api{};
  const bool have_api = xvram_cuda_compat_get_api(1, sizeof(api), &api) == XVRAM_STATUS_SUCCESS;
  const auto observe_unlocked = [&] {
    if (!have_api)
      return;
    xvram_cuda_compat_telemetry_v1 telemetry{};
    telemetry.struct_size = sizeof(telemetry);
    if (api.get_telemetry(&telemetry) != XVRAM_STATUS_SUCCESS)
      return;
    if (telemetry.progress_sequence > last_runtime_progress) {
      last_runtime_progress = telemetry.progress_sequence;
      static_cast<void>(send_unlocked(WorkerFrameType::progress, std::to_string(++progress)));
    }
    // A coherent snapshot may retire multiple short tiles between polls. Emit each monotonic
    // tile retirement with its observation timestamp, never fabricate CUDA elapsed timings.
    for (; last_tiles < telemetry.tiles_retired;)
      emit_unlocked("retired_tile", active_operation, std::nullopt, 0, ++last_tiles,
                    "observed_event_retirement");
  };
  std::jthread monitor([&](const std::stop_token stop) {
    auto heartbeat = Clock::now();
    while (!stop.stop_requested()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      std::scoped_lock lock(output_mutex);
      observe_unlocked();
      if (Clock::now() - heartbeat >= std::chrono::seconds(1)) {
        static_cast<void>(send_unlocked(WorkerFrameType::heartbeat, "{}"));
        heartbeat = Clock::now();
      }
    }
  });
  const auto trace = [&](std::string_view transition, std::uint64_t operation,
                         std::optional<std::uint64_t> allocation, std::uint64_t bytes,
                         std::string_view reason) {
    std::scoped_lock lock(output_mutex);
    if (transition == "call")
      active_operation = operation;
    if (transition == "return") {
      observe_unlocked();
      // A returned API call is actual work, including host backing copies that have no GPU tile.
      static_cast<void>(send_unlocked(WorkerFrameType::progress, std::to_string(++progress)));
    }
    emit_unlocked(transition, operation, allocation, bytes, last_tiles, reason);
    if (transition == "return")
      active_operation = 0;
    if (!output_ok)
      throw std::runtime_error("worker protocol output failed");
  };
  auto report = run_executor(options, trace);
  monitor.request_stop();
  monitor.join();
  observe_unlocked();
  emit_unlocked("final", 0, std::nullopt, 0, last_tiles,
                report.exit_code == 0 ? "completed" : "failed");
  report.trace_records = trace_sequence;
  report.trace_complete = output_ok;
  report.cleanup["trace_closed"] = output_ok;
  // Only the controller claims process termination; the wire report is amended after reap.
  report.cleanup["worker_terminated"] = std::nullopt;
  const FinalWorkerPayload final{report.exit_code, report_json(report, false), report_text(report)};
  if (!send_unlocked(WorkerFrameType::final, encode_final_worker_payload(final)))
    return 27;
  return report.exit_code;
}
bool emit_outputs(const Options& options, const std::string& json, const std::string& text) {
  if (options.json_path) {
    if (*options.json_path == "-") {
      std::cout << json << '\n';
      if (options.text)
        std::cerr << text;
    } else {
      std::ofstream file(utf8_path(*options.json_path), std::ios::binary | std::ios::trunc);
      file << json << '\n';
      file.flush();
      if (!file)
        return false;
      if (options.text)
        std::cout << text;
    }
  } else if (options.text)
    std::cout << text;
  return static_cast<bool>(std::cout) && static_cast<bool>(std::cerr);
}
int controller_main(const Options& options, const std::filesystem::path& executable) {
  auto report = base_report(options);
  WorkerControllerOptions config;
  config.executable = executable;
  config.arguments = worker_arguments(options);
  config.overall_timeout = options.timeout;
  std::ofstream trace;
  bool trace_ok = true;
  if (options.trace_path) {
    trace.open(utf8_path(*options.trace_path), std::ios::binary | std::ios::trunc);
    trace_ok = static_cast<bool>(trace);
    config.trace_sink = [&](const std::string_view batch, std::string& error) {
      trace.write(batch.data(), static_cast<std::streamsize>(batch.size()));
      trace.flush();
      if (!trace) {
        trace_ok = false;
        error = "controller failed to write trace";
      }
      return trace_ok;
    };
  }
  WorkerControllerResult worker;
  if (trace_ok)
    worker = run_isolated_worker(config);
  if (trace.is_open()) {
    trace.flush();
    trace_ok = static_cast<bool>(trace) && trace_ok;
    trace.close();
    trace_ok = !trace.fail() && trace_ok;
  }
  std::string json, text;
  int code = 27;
  if (trace_ok && !worker.trace_error && worker.started && !worker.timed_out &&
      !worker.protocol_error && worker.final &&
      worker.process_exit_code == worker.final->exit_code) {
    code = worker.final->exit_code;
    json = worker.final->json;
    text = worker.final->text;
    std::string validated;
    if (!finalize_json(json, code, options.pretty, validated, report.message)) {
      code = 27;
      json.clear();
      report.reason = "protocol_error";
    } else {
      json = std::move(validated);
    }
  }
  if (json.empty()) {
    if (!trace_ok || worker.trace_error) {
      code = 74;
      report.reason = "trace_io_error";
      report.message = "controller-owned trace output failed";
    } else if (worker.timed_out) {
      code = 26;
      report.reason = "timeout";
      report.message = "worker exceeded the 15-second no-progress watchdog or overall deadline; it "
                       "was terminated and reaped";
    } else if (!worker.started) {
      code = 27;
      report.reason = "platform_error";
      report.message = worker.error;
    } else if (report.message.empty()) {
      code = 27;
      report.reason = "protocol_error";
      report.message = worker.error.empty()
                           ? "worker report, protocol, or process exit status is inconsistent"
                           : worker.error;
    }
    report.exit_code = code;
    report.cleanup["worker_terminated"] = true;
    report.cleanup["trace_closed"] = trace_ok;
    report.trace_complete = false;
    json = report_json(report, options.pretty);
    text = report_text(report);
  }
  if (!emit_outputs(options, json, text)) {
    std::cerr << "report output I/O failure\n";
    return 74;
  }
  return code;
}
} // namespace
} // namespace xvram::compat_bench

int main(const int argc, char** argv) {
  xvram::compat_bench::Options options;
  std::string error;
  if (!xvram::compat_bench::parse_options(argc, argv, options, error)) {
    std::cerr << error << '\n' << xvram::compat_bench::help_text();
    return 64;
  }
  if (options.help) {
    std::cout << xvram::compat_bench::help_text();
    return 0;
  }
  if (options.version) {
    std::cout << XVRAM_VERSION << '\n';
    return 0;
  }
  try {
    if (options.worker)
      return xvram::compat_bench::worker_main(options);
    return xvram::compat_bench::controller_main(
        options, xvram::compat_bench::executable_path(argc > 0 ? argv[0] : nullptr));
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << '\n';
    return 70;
  } catch (...) {
    std::cerr << "unknown internal failure\n";
    return 70;
  }
}
