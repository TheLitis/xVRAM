#include "compat_bench/worker_controller.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#else
#include <cerrno>
#include <csignal>
#include <unistd.h>
#endif
namespace {
int failures = 0;
void check(bool condition, const char* text) {
  if (!condition) {
    ++failures;
    std::cerr << text << '\n';
  }
}
bool process_alive(std::uint64_t pid) {
#ifdef _WIN32
  HANDLE process = OpenProcess(SYNCHRONIZE, FALSE, static_cast<DWORD>(pid));
  if (!process)
    return GetLastError() != ERROR_INVALID_PARAMETER;
  const auto code = WaitForSingleObject(process, 0);
  CloseHandle(process);
  return code == WAIT_TIMEOUT;
#else
  errno = 0;
  return kill(static_cast<pid_t>(pid), 0) == 0 || errno == EPERM;
#endif
}
} // namespace
int main(int argc, char** argv) {
  if (argc != 2)
    return 64;
  using namespace xvram::compat_bench;
  WorkerControllerOptions options;
  options.executable = argv[1];
  options.arguments = {"success"};
  options.no_progress_timeout = std::chrono::seconds(3);
  options.overall_timeout = std::chrono::seconds(8);
  const auto success = run_isolated_worker(options);
  check(success.started && !success.timed_out && !success.protocol_error &&
            success.final.has_value() && success.process_exit_code == 0,
        "success protocol failed");
  check(success.heartbeat_frames == 1 && success.progress_frames == 1,
        "heartbeat/progress counters failed");
  for (const char* mode : {"crash", "truncated", "oversized", "non-monotonic", "silent-hang",
                           "heartbeat-hang", "trace-sink-hang"}) {
    const auto marker =
        std::filesystem::temp_directory_path() /
        ("xvram-compat-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".pid");
    options.arguments = {mode, marker.string()};
    options.no_progress_timeout = std::chrono::milliseconds(750);
    options.trace_sink = {};
    if (std::string(mode) == "trace-sink-hang")
      options.trace_sink = [](std::string_view, std::string& error) {
        error = "injected trace failure";
        return false;
      };
    const auto result = run_isolated_worker(options);
    const bool hang = std::string(mode) == "silent-hang" || std::string(mode) == "heartbeat-hang";
    check(result.started, "fault worker did not start");
    check(hang ? result.timed_out : result.protocol_error,
          "fault worker did not produce expected controller result");
    if (std::string(mode) == "heartbeat-hang")
      check(result.heartbeat_frames > 0, "heartbeat fixture did not emit heartbeat");
    if (std::string(mode) == "trace-sink-hang")
      check(result.trace_error, "trace failure was not marked");
    std::ifstream input(marker);
    std::uint64_t pid = 0;
    input >> pid;
    input.close();
    check(pid != 0, "missing worker PID marker");
    if (pid != 0)
      check(!process_alive(pid), "residual worker survived controller return");
    std::error_code error;
    std::filesystem::remove(marker, error);
    check(!error, "marker cleanup failed");
  }
  options.arguments = {"success"};
  options.executable = std::filesystem::path(argv[1]).parent_path() / "missing-compat-worker";
  check(!run_isolated_worker(options).started, "nonexistent worker should fail to start");
  return failures == 0 ? 0 : 1;
}
