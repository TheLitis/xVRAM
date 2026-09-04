#include "compression_bench/worker_controller.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#else
#include <cerrno>
#include <csignal>
#include <unistd.h>
#endif

namespace {

int failures = 0;
bool emit_failure_reports = false;

void check(const bool condition, const char* expression, const int line) {
  if (!condition) {
    std::cerr << "line " << line << ": check failed: " << expression << '\n';
    ++failures;
  }
}

#define CHECK(expression) check(static_cast<bool>(expression), #expression, __LINE__)

[[nodiscard]] xvram::compression::WorkerControllerOptions
options_for(const std::filesystem::path& helper, const std::string& mode) {
  xvram::compression::WorkerControllerOptions options;
  options.executable = helper;
  options.arguments = {mode};
  options.no_progress_timeout = std::chrono::seconds(2);
  options.overall_timeout = std::chrono::seconds(5);
  return options;
}

[[nodiscard]] std::uint64_t current_process_id() {
#ifdef _WIN32
  return static_cast<std::uint64_t>(GetCurrentProcessId());
#else
  return static_cast<std::uint64_t>(getpid());
#endif
}

[[nodiscard]] bool process_running(const std::uint64_t process_id) {
#ifdef _WIN32
  if (process_id > static_cast<std::uint64_t>(std::numeric_limits<DWORD>::max())) {
    return false;
  }
  HANDLE process = OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, FALSE,
                               static_cast<DWORD>(process_id));
  if (process == nullptr) {
    return GetLastError() != ERROR_INVALID_PARAMETER;
  }
  const DWORD wait = WaitForSingleObject(process, 0U);
  CloseHandle(process);
  return wait == WAIT_TIMEOUT;
#else
  errno = 0;
  const int code = kill(static_cast<pid_t>(process_id), 0);
  return code == 0 || errno == EPERM;
#endif
}

[[nodiscard]] bool wait_until_process_is_gone(const std::uint64_t process_id) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  do {
    if (!process_running(process_id)) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  } while (std::chrono::steady_clock::now() < deadline);
  return !process_running(process_id);
}

[[nodiscard]] std::filesystem::path marker_path(const std::string_view suffix) {
  const auto unique = std::to_string(current_process_id()) + "-" +
                      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  return std::filesystem::temp_directory_path() /
         ("xvram-compression-worker-" + unique + "-" + std::string(suffix) + ".pid");
}

[[nodiscard]] std::uint64_t read_worker_pid(const std::filesystem::path& marker) {
  std::ifstream input(marker);
  std::uint64_t process_id = 0;
  input >> process_id;
  CHECK(input.good() || input.eof());
  return process_id;
}

void remove_marker(const std::filesystem::path& marker) {
  std::error_code error;
  const bool removed = std::filesystem::remove(marker, error);
  CHECK(removed);
  CHECK(!error);
}

template <typename Configure>
[[nodiscard]] xvram::compression::WorkerControllerResult
run_marked_mode(const std::filesystem::path& helper, const std::string& mode,
                const std::string_view marker_suffix, Configure&& configure,
                std::uint64_t& worker_pid) {
  const std::filesystem::path marker = marker_path(marker_suffix);
  std::error_code ignored;
  std::filesystem::remove(marker, ignored);
  auto options = options_for(helper, mode);
  options.arguments.push_back(marker.string());
  configure(options);
  const auto result = xvram::compression::run_isolated_worker(options);
  worker_pid = read_worker_pid(marker);
  remove_marker(marker);
  return result;
}

void check_worker_reaped(const std::uint64_t worker_pid) {
  CHECK(worker_pid != 0U);
  if (worker_pid != 0U) {
    CHECK(wait_until_process_is_gone(worker_pid));
  }
}

[[nodiscard]] xvram::compression::Report base_report() {
  xvram::compression::Report report;
  report.generated_at_utc = "2026-09-04T00:00:00Z";
  report.build = {"0.1.0-dev", "fixture", "fixture", "Release", 13030};
  report.system = {"Fixture OS", "1.0", "x86_64", 8, 64ULL << 30U, 48ULL << 30U};
  report.configuration.trace_enabled = true;
  return report;
}

void check_failure_report(const xvram::compression::WorkerControllerResult& result,
                          const int expected_exit_code) {
  const auto output = xvram::compression::finalize_controller_result(base_report(), result, false);
  CHECK(output.exit_code == expected_exit_code);
  CHECK(output.json.find("\"report_type\":\"xvram.adaptive_compression\"") != std::string::npos);
  CHECK(output.json.find("\"complete\":null") != std::string::npos);
  CHECK(output.json.find("\"worker_terminated\":true") != std::string::npos);
  CHECK(output.json.find("\"trace_complete\":false") != std::string::npos);
  CHECK(output.json.find("\"status\":\"completed\"") == std::string::npos);
  if (emit_failure_reports) {
    std::cout << output.json;
    if (output.json.empty() || output.json.back() != '\n') {
      std::cout << '\n';
    }
  }
}

void success_test(const std::filesystem::path& helper) {
  const auto result = xvram::compression::run_isolated_worker(options_for(helper, "success"));
  CHECK(result.started);
  CHECK(!result.timed_out);
  CHECK(!result.protocol_error);
  CHECK(result.final.has_value());
  CHECK(result.process_exit_code == 0);
  CHECK(result.heartbeat_frames == 1U);
  CHECK(result.progress_frames == 1U);
  const auto output = xvram::compression::finalize_controller_result(base_report(), result, false);
  CHECK(output.exit_code == 0);
  if (result.final.has_value()) {
    CHECK(output.json == result.final->json);
    CHECK(output.text == result.final->text);
  }
}

void malformed_tests(const std::filesystem::path& helper) {
  std::uint64_t worker_pid = 0;
  const auto truncated =
      run_marked_mode(helper, "truncated", "truncated", [](auto&) {}, worker_pid);
  CHECK(truncated.started);
  CHECK(truncated.protocol_error);
  CHECK(truncated.error.find("truncated") != std::string::npos);
  check_worker_reaped(worker_pid);
  check_failure_report(truncated, 27);

  const auto oversized =
      run_marked_mode(helper, "oversized", "oversized", [](auto&) {}, worker_pid);
  CHECK(oversized.started);
  CHECK(oversized.protocol_error);
  CHECK(oversized.error.find("exceeds") != std::string::npos);
  check_worker_reaped(worker_pid);
  check_failure_report(oversized, 27);
}

void crash_and_reap_test(const std::filesystem::path& helper) {
  std::uint64_t worker_pid = 0;
  const auto result = run_marked_mode(helper, "crash", "crash", [](auto&) {}, worker_pid);
  CHECK(result.started);
  CHECK(!result.timed_out);
  CHECK(result.protocol_error);
  CHECK(result.process_exit_code == 42);
  CHECK(!result.final.has_value());
  CHECK(result.latest_report_json.find("before_crash") != std::string::npos);
  CHECK(result.error.find("before the final") != std::string::npos ||
        result.error.find("without a final") != std::string::npos);
  check_worker_reaped(worker_pid);
  check_failure_report(result, 27);
}

void silent_hang_and_reap_test(const std::filesystem::path& helper) {
  std::uint64_t worker_pid = 0;
  const auto began = std::chrono::steady_clock::now();
  const auto result = run_marked_mode(
      helper, "silent-hang", "silent-hang",
      [](auto& options) {
        options.no_progress_timeout = std::chrono::milliseconds(350);
        options.overall_timeout = std::chrono::seconds(3);
      },
      worker_pid);
  const auto elapsed = std::chrono::steady_clock::now() - began;
  CHECK(result.started);
  CHECK(result.timed_out);
  CHECK(result.process_exit_code == 26);
  CHECK(!result.final.has_value());
  CHECK(result.latest_report_json.find("silent_wait") != std::string::npos);
  CHECK(elapsed < std::chrono::seconds(5));
  check_worker_reaped(worker_pid);
  check_failure_report(result, 26);
}

void non_monotonic_protocol_and_reap_test(const std::filesystem::path& helper) {
  std::uint64_t worker_pid = 0;
  const auto result = run_marked_mode(
      helper, "non-monotonic", "non-monotonic",
      [](auto& options) { options.overall_timeout = std::chrono::seconds(3); }, worker_pid);
  CHECK(result.started);
  CHECK(!result.timed_out);
  CHECK(result.protocol_error);
  CHECK(!result.final.has_value());
  CHECK(result.error.find("sequence did not increase") != std::string::npos);
  check_worker_reaped(worker_pid);
  check_failure_report(result, 27);
}

void trace_output_sink_failure_and_reap_test(const std::filesystem::path& helper) {
  std::uint64_t sink_calls = 0;
  std::uint64_t worker_pid = 0;
  const auto result = run_marked_mode(
      helper, "trace-sink-hang", "trace-sink",
      [&](auto& options) {
        options.overall_timeout = std::chrono::seconds(3);
        options.trace_sink = [&](const std::string_view batch, std::string& error) {
          ++sink_calls;
          CHECK(batch.find("xvram.compression_trace") != std::string_view::npos);
          error = "injected compression trace output failure";
          return false;
        };
      },
      worker_pid);
  CHECK(result.started);
  CHECK(!result.timed_out);
  CHECK(result.protocol_error);
  CHECK(result.trace_error);
  CHECK(!result.final.has_value());
  CHECK(result.error.find("injected compression trace output failure") != std::string::npos);
  CHECK(sink_calls == 1U);
  CHECK(result.trace_frames == 0U);
  CHECK(result.trace_bytes == 0U);
  check_worker_reaped(worker_pid);
  check_failure_report(result, 74);
}

void throwing_trace_sink_and_reap_test(const std::filesystem::path& helper) {
  std::uint64_t worker_pid = 0;
  const auto result = run_marked_mode(
      helper, "trace-sink-hang", "throwing-trace-sink",
      [](auto& options) {
        options.trace_sink = [](const std::string_view, std::string&) -> bool {
          throw std::runtime_error("injected trace callback exception");
        };
      },
      worker_pid);
  CHECK(result.started);
  CHECK(!result.timed_out);
  CHECK(result.protocol_error);
  CHECK(result.trace_error);
  CHECK(!result.final.has_value());
  CHECK(result.error.find("trace sink threw") != std::string::npos);
  CHECK(result.trace_frames == 0U);
  CHECK(result.trace_bytes == 0U);
  check_worker_reaped(worker_pid);
  check_failure_report(result, 74);
}

void heartbeat_only_watchdog_test(const std::filesystem::path& helper) {
  const std::filesystem::path marker = marker_path("heartbeat");
  std::error_code error;
  std::filesystem::remove(marker, error);
  auto options = options_for(helper, "heartbeat-hang");
  options.arguments.push_back(marker.string());
  options.no_progress_timeout = std::chrono::milliseconds(350);
  options.overall_timeout = std::chrono::seconds(3);
  const auto result = xvram::compression::run_isolated_worker(options);
  CHECK(result.started);
  CHECK(result.timed_out);
  CHECK(!result.final.has_value());
  CHECK(result.heartbeat_frames > 0U);

  std::ifstream input(marker);
  std::uint64_t worker_pid = 0;
  input >> worker_pid;
  input.close();
  CHECK(worker_pid != 0U);
  if (worker_pid != 0U) {
    CHECK(wait_until_process_is_gone(worker_pid));
  }
  std::filesystem::remove(marker, error);
  check_failure_report(result, 26);
}

} // namespace

int main(const int argc, char** argv) {
  if (argc != 2 && (argc != 3 || std::string_view(argv[2]) != "--emit-failure-reports")) {
    return 64;
  }
  const std::filesystem::path helper(argv[1]);
  emit_failure_reports = argc == 3;
  success_test(helper);
  crash_and_reap_test(helper);
  malformed_tests(helper);
  silent_hang_and_reap_test(helper);
  non_monotonic_protocol_and_reap_test(helper);
  trace_output_sink_failure_and_reap_test(helper);
  throwing_trace_sink_and_reap_test(helper);
  heartbeat_only_watchdog_test(helper);
  if (failures != 0) {
    std::cerr << failures << " compression worker controller test(s) failed\n";
    return 1;
  }
  return 0;
}
