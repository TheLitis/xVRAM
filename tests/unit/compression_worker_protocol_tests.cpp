#include "compression_bench/worker_protocol.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(const bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

void test_round_trip() {
  using xvram::compression::WorkerFrameType;
  std::ostringstream encoded(std::ios::binary);
  expect(xvram::compression::write_worker_frame(encoded, WorkerFrameType::plan, 1, "{}"),
         "plan should encode");
  expect(xvram::compression::write_worker_frame(encoded, WorkerFrameType::heartbeat, 2, "{}"),
         "heartbeat should encode");
  expect(xvram::compression::write_worker_frame(encoded, WorkerFrameType::progress, 3, "{}"),
         "progress should encode");
  expect(xvram::compression::write_worker_frame(encoded, WorkerFrameType::trace, 4,
                                                "{\"sequence\":1}\n"),
         "trace should encode");

  xvram::compression::FinalWorkerPayload final;
  final.exit_code = 0;
  final.json = "{\"outcome\":{\"status\":\"completed\"}}";
  final.text = "completed\n";
  const std::string final_payload = xvram::compression::encode_final_worker_payload(final);
  expect(!final_payload.empty(), "final payload should encode");
  expect(xvram::compression::write_worker_frame(encoded, WorkerFrameType::final, 5, final_payload),
         "final frame should encode");

  const std::string bytes = encoded.str();
  xvram::compression::WorkerFrameDecoder decoder;
  std::vector<xvram::compression::WorkerFrame> frames;
  std::string error;
  for (std::size_t offset = 0; offset < bytes.size();) {
    const std::size_t fragment = std::min<std::size_t>(7, bytes.size() - offset);
    expect(decoder.append(bytes.data() + offset, fragment, frames, error),
           "fragment should decode");
    offset += fragment;
  }
  expect(decoder.finish(error), "complete stream should finish");
  expect(frames.size() == 5, "all XVZ1 frame types should round-trip");
  if (frames.size() == 5) {
    std::string decode_error;
    const auto decoded =
        xvram::compression::decode_final_worker_payload(frames.back().payload, decode_error);
    expect(decoded.has_value(), "final payload should decode");
    if (decoded.has_value()) {
      expect(decoded->exit_code == final.exit_code, "exit code should round-trip");
      expect(decoded->json == final.json, "JSON should round-trip");
      expect(decoded->text == final.text, "text should round-trip");
    }
  }
}

void test_rejections() {
  using xvram::compression::WorkerFrameType;
  std::ostringstream output(std::ios::binary);
  expect(!xvram::compression::write_worker_frame(output, WorkerFrameType::plan, 0, "{}"),
         "sequence zero should be rejected");
  const std::string oversized_trace(
      static_cast<std::size_t>(xvram::compression::maximum_trace_batch_bytes) + 1, 'x');
  expect(
      !xvram::compression::write_worker_frame(output, WorkerFrameType::trace, 1, oversized_trace),
      "oversized trace should be rejected");

  std::ostringstream no_plan(std::ios::binary);
  expect(xvram::compression::write_worker_frame(no_plan, WorkerFrameType::heartbeat, 1, "{}"),
         "invalid ordering can be encoded for decoder test");
  xvram::compression::WorkerFrameDecoder decoder;
  std::vector<xvram::compression::WorkerFrame> frames;
  std::string error;
  const std::string no_plan_bytes = no_plan.str();
  expect(!decoder.append(no_plan_bytes.data(), no_plan_bytes.size(), frames, error),
         "stream must begin with plan");

  std::ostringstream duplicate(std::ios::binary);
  expect(xvram::compression::write_worker_frame(duplicate, WorkerFrameType::plan, 1, "{}"),
         "first plan should encode");
  expect(xvram::compression::write_worker_frame(duplicate, WorkerFrameType::progress, 1, "{}"),
         "duplicate sequence can be encoded for decoder test");
  xvram::compression::WorkerFrameDecoder duplicate_decoder;
  frames.clear();
  error.clear();
  const std::string duplicate_bytes = duplicate.str();
  expect(!duplicate_decoder.append(duplicate_bytes.data(), duplicate_bytes.size(), frames, error),
         "duplicate sequence should be rejected");

  std::ostringstream incomplete(std::ios::binary);
  expect(xvram::compression::write_worker_frame(incomplete, WorkerFrameType::plan, 1, "{}"),
         "incomplete plan should encode");
  xvram::compression::WorkerFrameDecoder incomplete_decoder;
  frames.clear();
  error.clear();
  const std::string incomplete_bytes = incomplete.str();
  expect(incomplete_decoder.append(incomplete_bytes.data(), incomplete_bytes.size(), frames, error),
         "plan should decode");
  expect(!incomplete_decoder.finish(error), "stream without final should fail finish");

  xvram::compression::FinalWorkerPayload invalid_exit;
  invalid_exit.exit_code = 63;
  invalid_exit.json = "{}";
  expect(xvram::compression::encode_final_worker_payload(invalid_exit).empty(),
         "final payload should reject exit codes outside the public contract");

  xvram::compression::FinalWorkerPayload empty_report;
  empty_report.exit_code = 27;
  expect(xvram::compression::encode_final_worker_payload(empty_report).empty(),
         "final payload should reject an empty report");
}

void test_per_case_progress_payload_reset() {
  using xvram::compression::WorkerFrameType;
  std::ostringstream encoded(std::ios::binary);
  expect(xvram::compression::write_worker_frame(encoded, WorkerFrameType::plan, 1, "{}"),
         "multi-case plan should encode");
  expect(xvram::compression::write_worker_frame(
             encoded, WorkerFrameType::progress, 2,
             R"({"workloads":[{"scenario":"reuse","operations_retired":400}]})"),
         "completed first-case progress should encode");
  expect(
      xvram::compression::write_worker_frame(
          encoded, WorkerFrameType::progress, 3,
          R"({"workloads":[{"scenario":"reuse","operations_retired":400},{"scenario":"mixed","operations_retired":0}]})"),
      "new case may restart its operation counter at zero");
  expect(
      xvram::compression::write_worker_frame(
          encoded, WorkerFrameType::progress, 4,
          R"({"workloads":[{"scenario":"reuse","operations_retired":400},{"scenario":"mixed","operations_retired":1}]})"),
      "new-case progress should continue on the global frame sequence");
  xvram::compression::FinalWorkerPayload final;
  final.exit_code = 0;
  final.json = R"({"outcome":{"status":"completed"}})";
  expect(xvram::compression::write_worker_frame(
             encoded, WorkerFrameType::final, 5,
             xvram::compression::encode_final_worker_payload(final)),
         "multi-case final should encode");

  const std::string bytes = encoded.str();
  xvram::compression::WorkerFrameDecoder decoder;
  std::vector<xvram::compression::WorkerFrame> frames;
  std::string error;
  expect(decoder.append(bytes.data(), bytes.size(), frames, error),
         "per-case progress reset must not be mistaken for a protocol sequence reset");
  expect(decoder.finish(error), "multi-case stream should finish");
  expect(frames.size() == 5U, "all multi-case frames should be accepted");
}

} // namespace

int main() {
  test_round_trip();
  test_rejections();
  test_per_case_progress_payload_reset();
  if (failures != 0) {
    std::cerr << failures << " compression worker protocol test(s) failed\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
