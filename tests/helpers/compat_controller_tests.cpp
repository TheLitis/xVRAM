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
#include <sys/wait.h>
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
  const auto missing = run_isolated_worker(options);
  check(!missing.timed_out && !missing.trace_error && !missing.final.has_value(),
        "missing executable must fail without timeout, trace error, or a final report");
  check(missing.latest_report_json.empty() && missing.heartbeat_frames == 0 &&
            missing.progress_frames == 0 && missing.trace_frames == 0 && missing.trace_bytes == 0,
        "missing executable must not report worker activity");
  check(!missing.error.empty(), "missing executable must retain a failure diagnostic");
#ifdef _WIN32
  // CreateProcessW reports an absent image before a process is created. The public
  // controller converts this platform failure to its schema-valid exit-27 report.
  check(!missing.started && !missing.protocol_error && missing.process_exit_code == 27,
        "Windows missing executable must fail during process creation");
#else
  // fork succeeds before execv can report ENOENT. The short-lived child exits 127;
  // EOF without a final frame is a protocol failure, mapped to public exit 27.
  check(missing.started && missing.protocol_error && missing.process_exit_code == 127,
        "POSIX missing executable must preserve the reaped exec-failure result");
  // This dedicated test process owns only the already-drained helper children. Check
  // that the exec-failure child is neither still running nor left as an unreaped zombie.
  int child_status = 0;
  errno = 0;
  const auto child = waitpid(-1, &child_status, WNOHANG);
  check(child == -1 && errno == ECHILD,
        "POSIX missing executable left a live or unreaped child after controller return");
#endif
  return failures == 0 ? 0 : 1;
}
