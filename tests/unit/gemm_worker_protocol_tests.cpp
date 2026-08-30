#include "gemm_bench/worker_protocol.hpp"

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

[[nodiscard]] std::string frame(const xvram::gemm_bench::WorkerFrameType type,
                                const std::uint64_t sequence,
                                const std::string_view payload = "{}") {
  std::ostringstream output(std::ios::binary);
  if (!xvram::gemm_bench::write_worker_frame(output, type, sequence, payload)) {
    return {};
  }
  return output.str();
}

void test_round_trip() {
  using namespace xvram::gemm_bench;
  std::string encoded = frame(WorkerFrameType::plan, 1, R"({"tiles":3})");
  encoded += frame(WorkerFrameType::progress, 2, R"({"tile":1})");
  encoded += frame(WorkerFrameType::progress, 3, R"({"tile":2})");

  FinalWorkerPayload final;
  final.exit_code = 0;
  final.json = R"({"outcome":{"status":"completed"}})";
  final.text = "completed\n";
  const std::string final_payload = encode_final_worker_payload(final);
  expect(!final_payload.empty(), "final payload should encode");
  encoded += frame(WorkerFrameType::final, 4, final_payload);

  WorkerFrameDecoder decoder;
  std::vector<WorkerFrame> frames;
  std::string error;
  for (const char byte : encoded) {
    expect(decoder.append(&byte, 1U, frames, error), "single-byte fragments should decode");
  }
  expect(decoder.finish(error), "complete stream should finish");
  expect(frames.size() == 4U, "four frames should round-trip");
  if (frames.size() == 4U) {
    expect(frames.front().type == WorkerFrameType::plan, "first frame should be plan");
    expect(frames[1].type == WorkerFrameType::progress, "middle frame should be progress");
    expect(frames.back().type == WorkerFrameType::final, "last frame should be final");
    std::string decode_error;
    const auto decoded = decode_final_worker_payload(frames.back().payload, decode_error);
    expect(decoded.has_value(), "final payload should decode");
    if (decoded.has_value()) {
      expect(decoded->exit_code == final.exit_code, "exit code should round-trip");
      expect(decoded->json == final.json, "JSON should round-trip");
      expect(decoded->text == final.text, "text should round-trip");
    }
  }
}

void test_limits_and_truncation() {
  using namespace xvram::gemm_bench;
  std::ostringstream output(std::ios::binary);
  const std::string oversized(static_cast<std::size_t>(maximum_worker_payload_bytes) + 1U, 'x');
  expect(!write_worker_frame(output, WorkerFrameType::progress, 1, oversized),
         "oversized frame should be rejected");
  expect(!write_worker_frame(output, WorkerFrameType::progress, 0, "{}"),
         "zero sequence should be rejected");
  expect(!write_worker_frame(output, static_cast<WorkerFrameType>(99), 1, "{}"),
         "unknown frame type should be rejected");

  FinalWorkerPayload oversized_final;
  oversized_final.json = oversized;
  expect(encode_final_worker_payload(oversized_final).empty(),
         "oversized final payload should be rejected");

  FinalWorkerPayload boundary_final;
  boundary_final.json.assign(static_cast<std::size_t>(maximum_worker_payload_bytes) - 13U, 'j');
  boundary_final.text = "t";
  expect(encode_final_worker_payload(boundary_final).size() == maximum_worker_payload_bytes,
         "final payload exactly at the frame limit should encode");
  boundary_final.text = "tt";
  expect(encode_final_worker_payload(boundary_final).empty(),
         "combined final payload above the frame limit should be rejected before encoding");

  std::string error;
  expect(!decode_final_worker_payload("short", error).has_value(),
         "truncated final payload should be rejected");
  std::string malformed(12, '\0');
  malformed[4] = 4;
  error.clear();
  expect(!decode_final_worker_payload(malformed, error).has_value(),
         "invalid final lengths should be rejected");

  const std::string encoded = frame(WorkerFrameType::plan, 1);
  WorkerFrameDecoder truncated;
  std::vector<WorkerFrame> frames;
  error.clear();
  expect(truncated.append(encoded.data(), 7U, frames, error),
         "truncated header should wait for bytes");
  expect(!truncated.finish(error), "truncated stream should fail finish");

  WorkerFrameDecoder missing_final;
  frames.clear();
  error.clear();
  expect(missing_final.append(encoded.data(), encoded.size(), frames, error),
         "plan-only stream should decode so far");
  expect(!missing_final.finish(error), "plan-only stream should fail finish");
}

void test_order_and_header_rejections() {
  using namespace xvram::gemm_bench;
  std::vector<WorkerFrame> frames;
  std::string error;

  const std::string progress_first = frame(WorkerFrameType::progress, 1);
  WorkerFrameDecoder first_decoder;
  expect(!first_decoder.append(progress_first.data(), progress_first.size(), frames, error),
         "progress before plan should be rejected");

  const std::string duplicate_plan =
      frame(WorkerFrameType::plan, 1) + frame(WorkerFrameType::plan, 2);
  WorkerFrameDecoder duplicate_decoder;
  frames.clear();
  error.clear();
  expect(!duplicate_decoder.append(duplicate_plan.data(), duplicate_plan.size(), frames, error),
         "duplicate plan should be rejected");

  FinalWorkerPayload final_payload;
  final_payload.exit_code = 0;
  final_payload.json = "{}";
  const std::string after_final =
      frame(WorkerFrameType::plan, 1) +
      frame(WorkerFrameType::final, 2, encode_final_worker_payload(final_payload)) +
      frame(WorkerFrameType::progress, 3);
  WorkerFrameDecoder after_decoder;
  frames.clear();
  error.clear();
  expect(!after_decoder.append(after_final.data(), after_final.size(), frames, error),
         "data after final should be rejected");

  const std::string duplicate_sequence =
      frame(WorkerFrameType::plan, 4) + frame(WorkerFrameType::progress, 4);
  WorkerFrameDecoder sequence_decoder;
  frames.clear();
  error.clear();
  expect(
      !sequence_decoder.append(duplicate_sequence.data(), duplicate_sequence.size(), frames, error),
      "duplicate sequence should be rejected");

  std::string bad_magic = frame(WorkerFrameType::plan, 1);
  bad_magic[0] = 'Y';
  WorkerFrameDecoder magic_decoder;
  frames.clear();
  error.clear();
  expect(!magic_decoder.append(bad_magic.data(), bad_magic.size(), frames, error),
         "bad magic should be rejected");

  std::string bad_version = frame(WorkerFrameType::plan, 1);
  bad_version[4] = 2;
  WorkerFrameDecoder version_decoder;
  frames.clear();
  error.clear();
  expect(!version_decoder.append(bad_version.data(), bad_version.size(), frames, error),
         "unsupported version should be rejected");

  std::string bad_type = frame(WorkerFrameType::plan, 1);
  bad_type[6] = 99;
  WorkerFrameDecoder type_decoder;
  frames.clear();
  error.clear();
  expect(!type_decoder.append(bad_type.data(), bad_type.size(), frames, error),
         "invalid type should be rejected");

  std::string bad_length = frame(WorkerFrameType::plan, 1);
  bad_length[8] = static_cast<char>(0x01);
  bad_length[9] = static_cast<char>(0x00);
  bad_length[10] = static_cast<char>(0x10);
  bad_length[11] = static_cast<char>(0x00);
  WorkerFrameDecoder length_decoder;
  frames.clear();
  error.clear();
  expect(!length_decoder.append(bad_length.data(), bad_length.size(), frames, error),
         "payload above one MiB should be rejected from header");
}

} // namespace

int main() {
  test_round_trip();
  test_limits_and_truncation();
  test_order_and_header_rejections();
  if (failures != 0) {
    std::cerr << failures << " GEMM worker protocol test(s) failed\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
