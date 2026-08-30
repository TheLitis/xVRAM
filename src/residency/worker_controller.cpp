#include "residency/worker_controller.hpp"

#include "platform/worker_process.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace xvram::residency {
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
      if (frame.type == WorkerFrameType::plan || frame.type == WorkerFrameType::progress) {
        latest_report_json_ = std::move(frame.payload);
        made_progress = true;
        continue;
      }
      if (frame.type == WorkerFrameType::trace) {
        if (trace_sink_ && !trace_sink_(frame.payload, error)) {
          if (error.empty()) {
            error = "cache trace sink rejected a batch";
          }
          return false;
        }
        ++trace_frames_;
        trace_bytes_ += frame.payload.size();
        continue;
      }
      std::string decode_error;
      auto final = decode_final_worker_payload(frame.payload, decode_error);
      if (!final.has_value()) {
        error = std::move(decode_error);
        return false;
      }
      if (final_.has_value()) {
        error = "cache worker protocol sent more than one final frame";
        return false;
      }
      latest_report_json_ = final->json;
      final_ = std::move(final);
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
  std::uint64_t trace_frames_ = 0;
  std::uint64_t trace_bytes_ = 0;
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
  result.process_exit_code = process.process_exit_code;
  result.latest_report_json = std::move(consumer.latest_report_json_);
  result.final = std::move(consumer.final_);
  result.trace_frames = consumer.trace_frames_;
  result.trace_bytes = consumer.trace_bytes_;
  result.error = process.error;
  if (!result.timed_out && !result.protocol_error && result.started && !result.final.has_value()) {
    result.protocol_error = true;
    result.error = "cache worker exited without a final protocol frame";
  }
  return result;
}

} // namespace xvram::residency
