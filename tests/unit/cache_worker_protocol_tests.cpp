#include "residency/worker_protocol.hpp"

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
  std::ostringstream encoded(std::ios::binary);
  expect(xvram::residency::write_worker_frame(encoded,
                                               xvram::residency::WorkerFrameType::plan, 1,
                                               R"({"status":"planning"})"),
         "plan frame should encode");
  expect(xvram::residency::write_worker_frame(encoded,
                                               xvram::residency::WorkerFrameType::trace, 2,
                                               "{\"sequence\":1}\n"),
         "trace frame should encode");
  xvram::residency::FinalWorkerPayload final;
  final.exit_code = 0;
  final.json = R"({"outcome":{"status":"completed"}})";
  final.text = "completed\n";
  const std::string payload = xvram::residency::encode_final_worker_payload(final);
  expect(!payload.empty(), "final payload should encode");
  expect(xvram::residency::write_worker_frame(
             encoded, xvram::residency::WorkerFrameType::final, 3, payload),
         "final frame should encode");

  const std::string bytes = encoded.str();
  xvram::residency::WorkerFrameDecoder decoder;
  std::vector<xvram::residency::WorkerFrame> frames;
  std::string error;
  const std::size_t split = bytes.size() / 2U;
  expect(decoder.append(bytes.data(), split, frames, error), "first fragment should decode");
  expect(decoder.append(bytes.data() + split, bytes.size() - split, frames, error),
         "second fragment should decode");
  expect(decoder.finish(error), "complete stream should finish");
  expect(frames.size() == 3U, "three frames should round-trip");
  if (frames.size() == 3U) {
    expect(frames[1].type == xvram::residency::WorkerFrameType::trace,
           "trace type should round-trip");
    std::string decode_error;
    const auto decoded =
        xvram::residency::decode_final_worker_payload(frames[2].payload, decode_error);
    expect(decoded.has_value(), "final payload should decode");
    if (decoded.has_value()) {
      expect(decoded->exit_code == final.exit_code, "final exit code should round-trip");
      expect(decoded->json == final.json, "final JSON should round-trip");
      expect(decoded->text == final.text, "final text should round-trip");
    }
  }
}

void test_rejections() {
  std::ostringstream output(std::ios::binary);
  const std::string oversized_trace(
      static_cast<std::size_t>(xvram::residency::maximum_cache_trace_batch_bytes) + 1U, 'x');
  expect(!xvram::residency::write_worker_frame(
             output, xvram::residency::WorkerFrameType::trace, 1, oversized_trace),
         "oversized trace should be rejected");

  std::ostringstream encoded(std::ios::binary);
  expect(xvram::residency::write_worker_frame(
             encoded, xvram::residency::WorkerFrameType::progress, 2, "{}"),
         "progress should encode");
  expect(xvram::residency::write_worker_frame(
             encoded, xvram::residency::WorkerFrameType::progress, 2, "{}"),
         "duplicate sequence can be encoded for decoder test");
  xvram::residency::WorkerFrameDecoder decoder;
  std::vector<xvram::residency::WorkerFrame> frames;
  std::string error;
  const std::string bytes = encoded.str();
  expect(!decoder.append(bytes.data(), bytes.size(), frames, error),
         "duplicate sequence should be rejected");

  xvram::residency::WorkerFrameDecoder truncated;
  frames.clear();
  error.clear();
  expect(truncated.append(bytes.data(), 5U, frames, error),
         "truncated prefix should wait for more bytes");
  expect(!truncated.finish(error), "truncated stream should fail finish");
}

} // namespace

int main() {
  test_round_trip();
  test_rejections();
  if (failures != 0) {
    std::cerr << failures << " cache worker protocol test(s) failed\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
