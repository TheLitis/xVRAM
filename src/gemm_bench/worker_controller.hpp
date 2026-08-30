#pragma once

#include "gemm_bench/worker_protocol.hpp"

#include <chrono>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace xvram::gemm_bench {

struct WorkerControllerOptions {
  std::filesystem::path executable;
  std::vector<std::string> arguments;
  std::chrono::milliseconds no_progress_timeout{15000};
  std::chrono::milliseconds overall_timeout{300000};
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

} // namespace xvram::gemm_bench
