#pragma once

#include "vmm_poc/worker_protocol.hpp"

#include <chrono>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace xvram::vmm_poc {

struct WorkerControllerOptions {
  std::filesystem::path executable;
  std::vector<std::string> arguments;
  std::chrono::milliseconds no_progress_timeout{15000};
  std::chrono::milliseconds overall_timeout{120000};
};

struct WorkerControllerResult {
  bool started = false;
  bool timed_out = false;
  bool protocol_error = false;
  int process_exit_code = 27;
  std::string latest_report_json;
  std::optional<FinalWorkerPayload> final;
  std::string error;
};

[[nodiscard]] WorkerControllerResult run_isolated_worker(const WorkerControllerOptions& options);

} // namespace xvram::vmm_poc
