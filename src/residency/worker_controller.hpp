#pragma once

#include "residency/worker_protocol.hpp"

#include <chrono>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace xvram::residency {

using TraceSink = std::function<bool(std::string_view batch, std::string& error)>;

struct WorkerControllerOptions {
  std::filesystem::path executable;
  std::vector<std::string> arguments;
  std::chrono::milliseconds no_progress_timeout{15000};
  std::chrono::milliseconds overall_timeout{300000};
  TraceSink trace_sink;
};

struct WorkerControllerResult {
  bool started = false;
  bool timed_out = false;
  bool protocol_error = false;
  int process_exit_code = 27;
  std::string latest_report_json;
  std::optional<FinalWorkerPayload> final;
  std::uint64_t trace_frames = 0;
  std::uint64_t trace_bytes = 0;
  std::string error;
};

[[nodiscard]] WorkerControllerResult run_isolated_worker(const WorkerControllerOptions& options);

} // namespace xvram::residency
