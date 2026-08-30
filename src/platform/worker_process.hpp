#pragma once

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace xvram::platform {

class WorkerStreamConsumer {
public:
  virtual ~WorkerStreamConsumer() = default;

  WorkerStreamConsumer(const WorkerStreamConsumer&) = delete;
  WorkerStreamConsumer& operator=(const WorkerStreamConsumer&) = delete;
  WorkerStreamConsumer(WorkerStreamConsumer&&) = delete;
  WorkerStreamConsumer& operator=(WorkerStreamConsumer&&) = delete;

  [[nodiscard]] virtual bool consume(const char* data, std::size_t size, bool& made_progress,
                                     std::string& error) = 0;
  [[nodiscard]] virtual bool finish(std::string& error) = 0;

protected:
  WorkerStreamConsumer() = default;
};

struct IsolatedWorkerOptions {
  std::filesystem::path executable;
  std::vector<std::string> arguments;
  std::chrono::milliseconds no_progress_timeout{15000};
  std::chrono::milliseconds overall_timeout{120000};
  int timeout_exit_code = 26;
  int failure_exit_code = 27;
};

struct IsolatedWorkerResult {
  bool started = false;
  bool timed_out = false;
  bool stream_error = false;
  int process_exit_code = 27;
  std::string error;
};

[[nodiscard]] IsolatedWorkerResult
run_isolated_worker_process(const IsolatedWorkerOptions& options,
                            WorkerStreamConsumer& consumer);

} // namespace xvram::platform
