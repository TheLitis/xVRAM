#include "compression_bench/worker_controller.hpp"

#include "platform/worker_process.hpp"

#include <cstddef>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace xvram::compression {
namespace {

class ProtocolConsumer final : public platform::WorkerStreamConsumer {
public:
  explicit ProtocolConsumer(TraceSink trace_sink) : trace_sink_(std::move(trace_sink)) {}

  [[nodiscard]] bool consume(const char* data, const std::size_t size, bool& made_progress,
                             std::string& error) override {
    std::vector<WorkerFrame> frames;
    if (!decoder_.append(data, size, frames, error)) {
      return false;
    }
    for (auto& frame : frames) {
      if (frame.type == WorkerFrameType::plan) {
        latest_report_json_ = std::move(frame.payload);
        made_progress = true;
        continue;
      }
      if (frame.type == WorkerFrameType::heartbeat) {
        ++heartbeat_frames_;
        // Heartbeats prove that protocol I/O is alive but deliberately do not reset the
        // no-progress watchdog. A worker that only emits heartbeats is still hung.
        continue;
      }
      if (frame.type == WorkerFrameType::progress) {
        latest_report_json_ = std::move(frame.payload);
        ++progress_frames_;
        made_progress = true;
        continue;
      }
      if (frame.type == WorkerFrameType::trace) {
        bool accepted = true;
        try {
          accepted = !trace_sink_ || trace_sink_(frame.payload, error);
        } catch (...) {
          accepted = false;
          error = "compression trace sink threw while delivering a batch";
        }
        if (!accepted) {
          trace_error_ = true;
          if (error.empty()) {
            error = "compression trace sink rejected a batch";
          }
          return false;
        }
        ++trace_frames_;
        trace_bytes_ += frame.payload.size();
        continue;
      }
      std::string decode_error;
      auto decoded = decode_final_worker_payload(frame.payload, decode_error);
      if (!decoded.has_value()) {
        error = std::move(decode_error);
        return false;
      }
      if (final_.has_value()) {
        error = "compression worker protocol sent more than one final frame";
        return false;
      }
      latest_report_json_ = decoded->json;
      final_ = std::move(decoded);
      made_progress = true;
    }
    return true;
  }

  [[nodiscard]] bool finish(std::string& error) override {
    return decoder_.finish(error);
  }

  WorkerFrameDecoder decoder_;
  TraceSink trace_sink_;
  std::string latest_report_json_;
  std::optional<FinalWorkerPayload> final_;
  std::uint64_t heartbeat_frames_ = 0;
  std::uint64_t progress_frames_ = 0;
  std::uint64_t trace_frames_ = 0;
  std::uint64_t trace_bytes_ = 0;
  bool trace_error_ = false;
};

} // namespace

WorkerControllerResult run_isolated_worker(const WorkerControllerOptions& options) {
  platform::IsolatedWorkerOptions process_options;
  process_options.executable = options.executable;
  process_options.arguments = options.arguments;
  process_options.no_progress_timeout = options.no_progress_timeout;
  process_options.overall_timeout = options.overall_timeout;
  process_options.timeout_exit_code = 26;
  process_options.failure_exit_code = 27;

  ProtocolConsumer consumer(options.trace_sink);
  const platform::IsolatedWorkerResult process =
      platform::run_isolated_worker_process(process_options, consumer);

  WorkerControllerResult result;
  result.started = process.started;
  result.timed_out = process.timed_out;
  result.protocol_error = process.stream_error;
  result.trace_error = consumer.trace_error_;
  result.process_exit_code = process.process_exit_code;
  result.latest_report_json = std::move(consumer.latest_report_json_);
  result.final = std::move(consumer.final_);
  result.heartbeat_frames = consumer.heartbeat_frames_;
  result.progress_frames = consumer.progress_frames_;
  result.trace_frames = consumer.trace_frames_;
  result.trace_bytes = consumer.trace_bytes_;
  result.error = process.error;
  if (!result.timed_out && !result.protocol_error && result.started && !result.final.has_value()) {
    result.protocol_error = true;
    result.error = "compression worker exited without a final protocol frame";
  }
  return result;
}

FinalWorkerPayload finalize_controller_result(Report report, const WorkerControllerResult& worker,
                                              const bool pretty,
                                              const std::optional<std::string>& trace_io_error) {
  const bool trace_failed = trace_io_error.has_value() || worker.trace_error;
  if (!trace_failed && !worker.timed_out && worker.started && !worker.protocol_error &&
      worker.final.has_value() && worker.process_exit_code == worker.final->exit_code) {
    return *worker.final;
  }

  report.outcome = {};
  report.outcome.status = "failed";
  report.outcome.exit_code = 27;
  report.outcome.stage = "protocol";
  report.outcome.reason = "protocol_error";
  report.outcome.operation = "worker_protocol";
  report.outcome.message = worker.error.empty()
                               ? "the isolated worker failed without a valid final report"
                               : worker.error;
  if (trace_failed) {
    report.outcome.exit_code = 74;
    report.outcome.reason = "trace_io_error";
    report.outcome.stage = "trace";
    report.outcome.operation = "write_trace";
    report.outcome.message =
        trace_io_error.has_value()
            ? *trace_io_error
            : (worker.error.empty() ? "the controller trace sink failed" : worker.error);
  } else if (worker.timed_out) {
    report.outcome.exit_code = 26;
    report.outcome.status = "timeout";
    report.outcome.reason = "timeout";
    report.outcome.stage = "watchdog";
    report.outcome.operation = "worker_watchdog";
    report.outcome.message =
        "the isolated worker exceeded its no-progress or overall deadline and was terminated";
  } else if (!worker.started) {
    report.outcome.reason = "platform_error";
    report.outcome.operation = "start_worker";
    report.outcome.message =
        worker.error.empty() ? "the controller could not start the isolated worker" : worker.error;
  } else if (!worker.protocol_error && worker.final.has_value()) {
    report.outcome.operation = "worker_exit_code";
    report.outcome.message =
        "the worker process exit code does not match its final protocol report";
  }
  report.cleanup = {};
  report.cleanup.trace_closed = !trace_failed;
  report.cleanup.worker_terminated = true;
  report.proof = {};
  report.telemetry.trace_records_emitted =
      report.configuration.trace_enabled ? std::nullopt : std::optional<std::uint64_t>{0U};
  report.telemetry.trace_records_dropped = trace_failed ? 1U : 0U;
  report.telemetry.trace_complete = !report.configuration.trace_enabled && !trace_failed;

  FinalWorkerPayload result;
  result.exit_code = report.outcome.exit_code;
  std::ostringstream json;
  write_json(report, json, pretty);
  result.json = json.str();
  std::ostringstream text;
  write_text(report, text);
  result.text = text.str();
  return result;
}

} // namespace xvram::compression
