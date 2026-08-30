#include "gemm_bench/worker_controller.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
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

void check(const bool condition, const char* expression, const int line) {
  if (!condition) {
    std::cerr << "line " << line << ": check failed: " << expression << '\n';
    ++failures;
  }
}

#define CHECK(expression) check(static_cast<bool>(expression), #expression, __LINE__)

[[nodiscard]] xvram::gemm_bench::WorkerControllerOptions
base_options(const std::filesystem::path& helper, const std::string& mode) {
  xvram::gemm_bench::WorkerControllerOptions options;
  options.executable = helper;
  options.arguments = {mode};
  options.no_progress_timeout = std::chrono::seconds(2);
  options.overall_timeout = std::chrono::seconds(5);
  return options;
}

[[nodiscard]] xvram::gemm_bench::WorkerControllerResult
run_mode(const std::filesystem::path& helper, const std::string& mode) {
  return xvram::gemm_bench::run_isolated_worker(base_options(helper, mode));
}

void success_test(const std::filesystem::path& helper) {
  const auto result = run_mode(helper, "success");
  CHECK(result.started);
  CHECK(!result.timed_out);
  CHECK(!result.protocol_error);
  CHECK(result.error.empty());
  CHECK(result.process_exit_code == 0);
  CHECK(result.final.has_value());
  CHECK(result.latest_report_json == R"({"state":"completed","tiles_retired":2})");
  if (result.final.has_value()) {
    CHECK(result.final->exit_code == 0);
    CHECK(result.final->json == result.latest_report_json);
    CHECK(result.final->text == "GEMM worker completed\n");
  }
}

void failure_tests(const std::filesystem::path& helper) {
  const auto crashed = run_mode(helper, "crash");
  CHECK(crashed.started);
  CHECK(!crashed.timed_out);
  CHECK(crashed.protocol_error);
  CHECK(crashed.process_exit_code == 42);
  CHECK(!crashed.final.has_value());
  CHECK(crashed.latest_report_json.find("before_crash") != std::string::npos);

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

void timeout_and_reap_test(const std::filesystem::path& helper) {
  const std::string unique =
      std::to_string(controller_process_id()) + "-" +
      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  const std::filesystem::path marker =
      std::filesystem::temp_directory_path() / ("xvram-gemm-worker-" + unique + ".pid");
  std::error_code ignored;
  std::filesystem::remove(marker, ignored);

  auto options = base_options(helper, "hang");
  options.arguments.push_back(marker.string());
  options.no_progress_timeout = std::chrono::milliseconds(350);
  options.overall_timeout = std::chrono::seconds(3);
  const auto began = std::chrono::steady_clock::now();
  const auto result = xvram::gemm_bench::run_isolated_worker(options);
  const auto elapsed = std::chrono::steady_clock::now() - began;

  CHECK(result.started);
  CHECK(result.timed_out);
  CHECK(!result.protocol_error);
  CHECK(result.error.empty());
  CHECK(result.process_exit_code == 26);
  CHECK(!result.final.has_value());
  CHECK(result.latest_report_json.find("heartbeat") != std::string::npos);
  CHECK(elapsed < std::chrono::seconds(2));

  std::ifstream marker_input(marker);
  std::uint64_t worker_pid = 0;
  marker_input >> worker_pid;
  CHECK(marker_input.good() || marker_input.eof());
  CHECK(worker_pid != 0);
  if (worker_pid != 0) {
    CHECK(wait_until_process_is_gone(worker_pid));
  }
  marker_input.close();
  std::error_code remove_error;
  CHECK(std::filesystem::remove(marker, remove_error));
  CHECK(!remove_error);
}

void invalid_options_test() {
  xvram::gemm_bench::WorkerControllerOptions options;
  const auto result = xvram::gemm_bench::run_isolated_worker(options);
  CHECK(!result.started);
  CHECK(!result.timed_out);
  CHECK(!result.protocol_error);
  CHECK(!result.error.empty());
}

} // namespace

int main(const int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: gemm_worker_controller_tests <gemm-worker-test-helper>\n";
    return 64;
  }
  const std::filesystem::path helper(argv[1]);
  success_test(helper);
  failure_tests(helper);
  timeout_and_reap_test(helper);
  invalid_options_test();
  if (failures != 0) {
    std::cerr << failures << " GEMM worker controller test(s) failed\n";
    return 1;
  }
  std::cout << "all GEMM worker controller tests passed\n";
  return 0;
}
