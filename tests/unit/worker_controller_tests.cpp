#include "vmm_poc/worker_controller.hpp"
#include "vmm_poc/worker_protocol.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#else
#include <cerrno>
#include <csignal>
#include <unistd.h>
#endif

namespace {

int failures = 0;

void check(const bool condition, const char* expression, const int line) {
  if (!condition) {
    std::cerr << "line " << line << ": check failed: " << expression << '\n';
    ++failures;
  }
}

#define CHECK(expression) check(static_cast<bool>(expression), #expression, __LINE__)

[[nodiscard]] std::string encode_frame(const xvram::vmm_poc::WorkerFrameType type,
                                       const std::uint64_t sequence,
                                       const std::string_view payload) {
  std::ostringstream output(std::ios::out | std::ios::binary);
  if (!xvram::vmm_poc::write_worker_frame(output, type, sequence, payload)) {
    return {};
  }
  return output.str();
}

void final_payload_tests() {
  using namespace xvram::vmm_poc;

  FinalWorkerPayload source;
  source.exit_code = 24;
  source.json = R"({"outcome":"corruption"})";
  source.text = "mismatch at byte 4096\n";
  const std::string encoded = encode_final_worker_payload(source);
  CHECK(!encoded.empty());

  std::string error;
  const auto decoded = decode_final_worker_payload(encoded, error);
  CHECK(decoded.has_value());
  CHECK(error.empty());
  if (decoded.has_value()) {
    CHECK(decoded->exit_code == source.exit_code);
    CHECK(decoded->json == source.json);
    CHECK(decoded->text == source.text);
  }

  std::string truncated = encoded;
  truncated.pop_back();
  error.clear();
  CHECK(!decode_final_worker_payload(truncated, error).has_value());
  CHECK(error.find("lengths") != std::string::npos);

  error.clear();
  CHECK(!decode_final_worker_payload("short", error).has_value());
  CHECK(error.find("truncated") != std::string::npos);
}

void frame_decoder_tests() {
  using namespace xvram::vmm_poc;

  const std::string plan = encode_frame(WorkerFrameType::plan, 1, R"({"plan":true})");
  const std::string progress = encode_frame(WorkerFrameType::progress, 2, R"({"retired":1})");
  FinalWorkerPayload final_payload;
  final_payload.exit_code = 0;
  final_payload.json = R"({"outcome":"completed"})";
  final_payload.text = "done\n";
  const std::string final =
      encode_frame(WorkerFrameType::final, 3, encode_final_worker_payload(final_payload));
  const std::string wire = plan + progress + final;
  CHECK(!wire.empty());

  WorkerFrameDecoder decoder;
  std::vector<WorkerFrame> frames;
  std::string error;
  std::size_t offset = 0;
  constexpr std::size_t chunk_sizes[] = {1, 2, 7, 3, 29};
  std::size_t chunk_index = 0;
  while (offset < wire.size()) {
    const std::size_t requested = chunk_sizes[chunk_index % std::size(chunk_sizes)];
    const std::size_t remaining = wire.size() - offset;
    const std::size_t size = requested < remaining ? requested : remaining;
    CHECK(decoder.append(wire.data() + offset, size, frames, error));
    offset += size;
    ++chunk_index;
  }
  CHECK(decoder.finish(error));
  CHECK(error.empty());
  CHECK(frames.size() == 3);
  if (frames.size() == 3) {
    CHECK(frames[0].type == WorkerFrameType::plan);
    CHECK(frames[0].sequence == 1);
    CHECK(frames[1].type == WorkerFrameType::progress);
    CHECK(frames[1].sequence == 2);
    CHECK(frames[2].type == WorkerFrameType::final);
    CHECK(frames[2].sequence == 3);
    std::string final_error;
    const auto decoded = decode_final_worker_payload(frames[2].payload, final_error);
    CHECK(decoded.has_value());
    CHECK(final_error.empty());
    if (decoded.has_value()) {
      CHECK(decoded->json == final_payload.json);
    }
  }

  WorkerFrameDecoder empty_append;
  frames.clear();
  error.clear();
  CHECK(empty_append.append(nullptr, 0, frames, error));
  CHECK(!empty_append.append(nullptr, 1, frames, error));
  CHECK(error.find("null") != std::string::npos);

  WorkerFrameDecoder bad_magic;
  std::string corrupt = plan;
  corrupt[0] = 'B';
  frames.clear();
  error.clear();
  CHECK(!bad_magic.append(corrupt.data(), corrupt.size(), frames, error));
  CHECK(error.find("magic") != std::string::npos);

  WorkerFrameDecoder truncated;
  frames.clear();
  error.clear();
  CHECK(truncated.append(plan.data(), plan.size() - 1U, frames, error));
  CHECK(!truncated.finish(error));
  CHECK(error.find("truncated") != std::string::npos);

  WorkerFrameDecoder repeated_sequence;
  const std::string duplicate = plan + encode_frame(WorkerFrameType::progress, 1, "{}");
  frames.clear();
  error.clear();
  CHECK(!repeated_sequence.append(duplicate.data(), duplicate.size(), frames, error));
  CHECK(error.find("sequence") != std::string::npos);

  std::ostringstream oversized(std::ios::out | std::ios::binary);
  const std::string excessive(maximum_worker_payload_bytes + 1ULL, 'x');
  CHECK(!write_worker_frame(oversized, WorkerFrameType::progress, 1, excessive));
}

[[nodiscard]] xvram::vmm_poc::WorkerControllerResult run_mode(const std::filesystem::path& helper,
                                                              const std::string& mode) {
  xvram::vmm_poc::WorkerControllerOptions options;
  options.executable = helper;
  options.arguments = {mode};
  options.no_progress_timeout = std::chrono::seconds(2);
  options.overall_timeout = std::chrono::seconds(5);
  return xvram::vmm_poc::run_isolated_worker(options);
}

void controller_success_test(const std::filesystem::path& helper) {
  const auto result = run_mode(helper, "success");
  CHECK(result.started);
  CHECK(!result.timed_out);
  CHECK(!result.protocol_error);
  CHECK(result.error.empty());
  CHECK(result.process_exit_code == 0);
  CHECK(result.final.has_value());
  CHECK(result.latest_report_json == R"({"state":"completed","retired":2})");
  if (result.final.has_value()) {
    CHECK(result.final->exit_code == 0);
    CHECK(result.final->json == result.latest_report_json);
    CHECK(result.final->text == "worker completed\n");
  }
}

void controller_failure_tests(const std::filesystem::path& helper) {
  const auto crashed = run_mode(helper, "crash");
  CHECK(crashed.started);
  CHECK(!crashed.timed_out);
  CHECK(crashed.protocol_error);
  CHECK(crashed.process_exit_code == 42);
  CHECK(!crashed.final.has_value());
  CHECK(crashed.latest_report_json.find("before_crash") != std::string::npos);

  const auto truncated = run_mode(helper, "truncated");
  CHECK(truncated.started);
  CHECK(!truncated.timed_out);
  CHECK(truncated.protocol_error);
  CHECK(!truncated.final.has_value());
  CHECK(truncated.error.find("truncated") != std::string::npos);

  const auto bad_magic = run_mode(helper, "bad-magic");
  CHECK(bad_magic.started);
  CHECK(!bad_magic.timed_out);
  CHECK(bad_magic.protocol_error);
  CHECK(!bad_magic.final.has_value());
  CHECK(bad_magic.error.find("magic") != std::string::npos);
}

[[nodiscard]] std::uint64_t controller_test_process_id() {
#ifdef _WIN32
  return static_cast<std::uint64_t>(GetCurrentProcessId());
#else
  return static_cast<std::uint64_t>(getpid());
#endif
}

[[nodiscard]] bool process_is_running(const std::uint64_t process_id) {
#ifdef _WIN32
  if (process_id > static_cast<std::uint64_t>(std::numeric_limits<DWORD>::max())) {
    return false;
  }
  HANDLE process = OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, FALSE,
                               static_cast<DWORD>(process_id));
  if (process == nullptr) {
    return GetLastError() != ERROR_INVALID_PARAMETER;
  }
  const DWORD wait_result = WaitForSingleObject(process, 0);
  CloseHandle(process);
  return wait_result == WAIT_TIMEOUT;
#else
  if (process_id > static_cast<std::uint64_t>(std::numeric_limits<pid_t>::max())) {
    return false;
  }
  errno = 0;
  const int result = kill(static_cast<pid_t>(process_id), 0);
  return result == 0 || errno == EPERM;
#endif
}

[[nodiscard]] bool wait_until_process_is_gone(const std::uint64_t process_id) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  do {
    if (!process_is_running(process_id)) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  } while (std::chrono::steady_clock::now() < deadline);
  return !process_is_running(process_id);
}

void controller_timeout_and_reap_test(const std::filesystem::path& helper) {
  const auto unique = std::to_string(controller_test_process_id()) + "-" +
                      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  const std::filesystem::path marker =
      std::filesystem::temp_directory_path() / ("xvram-worker-test-" + unique + ".pid");
  std::error_code remove_error;
  std::filesystem::remove(marker, remove_error);

  xvram::vmm_poc::WorkerControllerOptions options;
  options.executable = helper;
  options.arguments = {"hang", marker.string()};
  options.no_progress_timeout = std::chrono::milliseconds(350);
  options.overall_timeout = std::chrono::seconds(3);
  const auto began = std::chrono::steady_clock::now();
  const auto result = xvram::vmm_poc::run_isolated_worker(options);
  const auto elapsed = std::chrono::steady_clock::now() - began;

  CHECK(result.started);
  CHECK(result.timed_out);
  CHECK(!result.protocol_error);
  CHECK(!result.final.has_value());
  CHECK(result.latest_report_json.find("waiting_forever") != std::string::npos);
  CHECK(elapsed < std::chrono::seconds(5));

  std::ifstream marker_input(marker);
  std::uint64_t worker_process_id = 0;
  marker_input >> worker_process_id;
  CHECK(marker_input.good() || marker_input.eof());
  CHECK(worker_process_id != 0);
  if (worker_process_id != 0) {
    CHECK(wait_until_process_is_gone(worker_process_id));
  }
  marker_input.close();
  remove_error.clear();
  std::filesystem::remove(marker, remove_error);
  CHECK(!remove_error);
}

void invalid_options_test() {
  xvram::vmm_poc::WorkerControllerOptions options;
  const auto result = xvram::vmm_poc::run_isolated_worker(options);
  CHECK(!result.started);
  CHECK(!result.timed_out);
  CHECK(!result.protocol_error);
  CHECK(!result.error.empty());
}

} // namespace

int main(const int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: worker_controller_tests <worker-test-helper>\n";
    return 64;
  }

  final_payload_tests();
  frame_decoder_tests();
  controller_success_test(argv[1]);
  controller_failure_tests(argv[1]);
  controller_timeout_and_reap_test(argv[1]);
  invalid_options_test();

  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
    return 1;
  }
  std::cout << "all worker protocol/controller tests passed\n";
  return 0;
}
