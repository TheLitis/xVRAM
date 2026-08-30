#include "residency/worker_controller.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

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

void check(const bool condition, const char* expression, const int line) {
  if (!condition) {
    std::cerr << "line " << line << ": check failed: " << expression << '\n';
    ++failures;
  }
}

#define CHECK(expression) check(static_cast<bool>(expression), #expression, __LINE__)

[[nodiscard]] xvram::residency::WorkerControllerOptions
base_options(const std::filesystem::path& helper, const std::string& mode) {
  xvram::residency::WorkerControllerOptions options;
  options.executable = helper;
  options.arguments = {mode};
  options.no_progress_timeout = std::chrono::seconds(2);
  options.overall_timeout = std::chrono::seconds(5);
  return options;
}

[[nodiscard]] xvram::residency::WorkerControllerResult
run_mode(const std::filesystem::path& helper, const std::string& mode) {
  return xvram::residency::run_isolated_worker(base_options(helper, mode));
}

void success_test(const std::filesystem::path& helper) {
  const auto result = run_mode(helper, "success");
  CHECK(result.started);
  CHECK(!result.timed_out);
  CHECK(!result.protocol_error);
  CHECK(result.error.empty());
  CHECK(result.process_exit_code == 0);
  CHECK(result.final.has_value());
  CHECK(result.latest_report_json == R"({"state":"completed","operations_retired":2})");
  CHECK(result.trace_frames == 0);
  CHECK(result.trace_bytes == 0);
  if (result.final.has_value()) {
    CHECK(result.final->exit_code == 0);
    CHECK(result.final->json == result.latest_report_json);
    CHECK(result.final->text == "cache worker completed\n");
  }
}

void crash_test(const std::filesystem::path& helper) {
  const auto result = run_mode(helper, "crash");
  CHECK(result.started);
  CHECK(!result.timed_out);
  CHECK(result.protocol_error);
  CHECK(result.process_exit_code == 42);
  CHECK(!result.final.has_value());
  CHECK(result.latest_report_json.find("before_crash") != std::string::npos);
  CHECK(result.error.find("without a final") != std::string::npos);
}

void malformed_stream_tests(const std::filesystem::path& helper) {
  const auto truncated = run_mode(helper, "truncated");
  CHECK(truncated.started);
  CHECK(!truncated.timed_out);
  CHECK(truncated.protocol_error);
  CHECK(!truncated.final.has_value());
  CHECK(truncated.error.find("truncated") != std::string::npos);

  const auto oversized = run_mode(helper, "oversized");
  CHECK(oversized.started);
  CHECK(!oversized.timed_out);
  CHECK(oversized.protocol_error);
  CHECK(!oversized.final.has_value());
  CHECK(oversized.error.find("exceeds") != std::string::npos);
}

void trace_success_test(const std::filesystem::path& helper) {
  std::vector<std::string> batches;
  auto options = base_options(helper, "trace-success");
  options.trace_sink = [&](const std::string_view batch, std::string&) {
    batches.emplace_back(batch);
    return true;
  };
  const auto result = xvram::residency::run_isolated_worker(options);

  CHECK(result.started);
  CHECK(!result.timed_out);
  CHECK(!result.protocol_error);
  CHECK(result.error.empty());
  CHECK(result.process_exit_code == 0);
  CHECK(result.final.has_value());
  CHECK(result.trace_frames == 2);
  CHECK(batches.size() == 2);
  std::uint64_t expected_bytes = 0;
  for (const std::string& batch : batches) {
    expected_bytes += batch.size();
    CHECK(batch.ends_with('\n'));
    CHECK(batch.find("xvram.residency_trace") != std::string::npos);
  }
  CHECK(result.trace_bytes == expected_bytes);
  if (result.final.has_value()) {
    CHECK(result.final->json == R"({"state":"completed","trace_records":2})");
    CHECK(result.final->text == "cache trace completed\n");
  }
}

[[nodiscard]] std::uint64_t controller_process_id() {
#ifdef _WIN32
  return static_cast<std::uint64_t>(GetCurrentProcessId());
#else
  return static_cast<std::uint64_t>(getpid());
#endif
}

[[nodiscard]] bool process_is_running(const std::uint64_t process_id) {
#ifdef _WIN32
  if (process_id > static_cast<std::uint64_t>(std::numeric_limits<DWORD>::max())) {
    return false;
  }
  HANDLE process = OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, FALSE,
                               static_cast<DWORD>(process_id));
  if (process == nullptr) {
    return GetLastError() != ERROR_INVALID_PARAMETER;
  }
  const DWORD wait_result = WaitForSingleObject(process, 0);
  CloseHandle(process);
  return wait_result == WAIT_TIMEOUT;
#else
  if (process_id > static_cast<std::uint64_t>(std::numeric_limits<pid_t>::max())) {
    return false;
  }
  errno = 0;
  const int result = kill(static_cast<pid_t>(process_id), 0);
  return result == 0 || errno == EPERM;
#endif
}

[[nodiscard]] bool wait_until_process_is_gone(const std::uint64_t process_id) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  do {
    if (!process_is_running(process_id)) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  } while (std::chrono::steady_clock::now() < deadline);
  return !process_is_running(process_id);
}

[[nodiscard]] std::filesystem::path marker_path(const std::string_view suffix) {
  const auto unique = std::to_string(controller_process_id()) + "-" +
                      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  return std::filesystem::temp_directory_path() /
         ("xvram-cache-worker-" + unique + "-" + std::string(suffix) + ".pid");
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

void watchdog_and_reap_test(const std::filesystem::path& helper) {
  const std::filesystem::path marker = marker_path("watchdog");
  std::error_code ignored;
  std::filesystem::remove(marker, ignored);

  auto options = base_options(helper, "hang");
  options.arguments.push_back(marker.string());
  options.no_progress_timeout = std::chrono::milliseconds(350);
  options.overall_timeout = std::chrono::seconds(3);
  const auto began = std::chrono::steady_clock::now();
  const auto result = xvram::residency::run_isolated_worker(options);
  const auto elapsed = std::chrono::steady_clock::now() - began;

  CHECK(result.started);
  CHECK(result.timed_out);
  CHECK(!result.protocol_error);
  CHECK(!result.final.has_value());
  CHECK(result.latest_report_json.find("waiting_forever") != std::string::npos);
  CHECK(elapsed < std::chrono::seconds(5));

  const std::uint64_t worker_pid = read_worker_pid(marker);
  CHECK(worker_pid != 0);
  if (worker_pid != 0) {
    CHECK(wait_until_process_is_gone(worker_pid));
  }
  remove_marker(marker);
}

void trace_sink_failure_and_reap_test(const std::filesystem::path& helper) {
  const std::filesystem::path marker = marker_path("trace-sink");
  std::error_code ignored;
  std::filesystem::remove(marker, ignored);

  auto options = base_options(helper, "trace-hang");
  options.arguments.push_back(marker.string());
  options.no_progress_timeout = std::chrono::seconds(2);
  options.overall_timeout = std::chrono::seconds(3);
  std::uint64_t sink_calls = 0;
  options.trace_sink = [&](const std::string_view, std::string& error) {
    ++sink_calls;
    error = "injected trace sink failure";
    return false;
  };
  const auto began = std::chrono::steady_clock::now();
  const auto result = xvram::residency::run_isolated_worker(options);
  const auto elapsed = std::chrono::steady_clock::now() - began;

  CHECK(result.started);
  CHECK(!result.timed_out);
  CHECK(result.protocol_error);
  CHECK(!result.final.has_value());
  CHECK(result.error.find("injected trace sink failure") != std::string::npos);
  CHECK(sink_calls == 1);
  CHECK(result.trace_frames == 0);
  CHECK(result.trace_bytes == 0);
  CHECK(elapsed < std::chrono::seconds(5));

  const std::uint64_t worker_pid = read_worker_pid(marker);
  CHECK(worker_pid != 0);
  if (worker_pid != 0) {
    CHECK(wait_until_process_is_gone(worker_pid));
  }
  remove_marker(marker);
}

void invalid_options_test() {
  xvram::residency::WorkerControllerOptions options;
  const auto result = xvram::residency::run_isolated_worker(options);
  CHECK(!result.started);
  CHECK(!result.timed_out);
  CHECK(!result.protocol_error);
  CHECK(!result.error.empty());
}

} // namespace

int main(const int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: cache_worker_controller_tests <cache-worker-test-helper>\n";
    return 64;
  }

  const std::filesystem::path helper(argv[1]);
  success_test(helper);
  crash_test(helper);
  malformed_stream_tests(helper);
  trace_success_test(helper);
  watchdog_and_reap_test(helper);
  trace_sink_failure_and_reap_test(helper);
  invalid_options_test();

  if (failures != 0) {
    std::cerr << failures << " cache worker controller test(s) failed\n";
    return 1;
  }
  std::cout << "all cache worker controller tests passed\n";
  return 0;
}
