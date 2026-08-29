#include "vmm_poc/worker_protocol.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#include <fcntl.h>
#include <io.h>
#else
#include <unistd.h>
#endif

namespace {

using xvram::vmm_poc::FinalWorkerPayload;
using xvram::vmm_poc::WorkerFrameType;

[[nodiscard]] bool enable_binary_stdout() {
#ifdef _WIN32
  return _setmode(_fileno(stdout), _O_BINARY) != -1;
#else
  return true;
#endif
}

[[nodiscard]] std::uint64_t current_process_id() {
#ifdef _WIN32
  return static_cast<std::uint64_t>(GetCurrentProcessId());
#else
  return static_cast<std::uint64_t>(getpid());
#endif
}

[[nodiscard]] bool emit_frame(const WorkerFrameType type, const std::uint64_t sequence,
                              const std::string_view payload) {
  return xvram::vmm_poc::write_worker_frame(std::cout, type, sequence, payload);
}

int success_mode() {
  if (!emit_frame(WorkerFrameType::plan, 1, R"({"state":"planned"})") ||
      !emit_frame(WorkerFrameType::progress, 2, R"({"state":"running","retired":1})")) {
    return 74;
  }

  FinalWorkerPayload final;
  final.exit_code = 0;
  final.json = R"({"state":"completed","retired":2})";
  final.text = "worker completed\n";
  const std::string payload = xvram::vmm_poc::encode_final_worker_payload(final);
  if (payload.empty() || !emit_frame(WorkerFrameType::final, 3, payload)) {
    return 74;
  }
  return 0;
}

[[noreturn]] void crash_mode() {
  static_cast<void>(
      emit_frame(WorkerFrameType::progress, 1, R"({"state":"running","before_crash":true})"));
  std::_Exit(42);
}

int truncated_mode() {
  std::ostringstream encoded(std::ios::out | std::ios::binary);
  if (!xvram::vmm_poc::write_worker_frame(encoded, WorkerFrameType::progress, 1,
                                          R"({"state":"truncated"})")) {
    return 70;
  }
  const std::string bytes = encoded.str();
  if (bytes.empty()) {
    return 70;
  }
  std::cout.write(bytes.data(), static_cast<std::streamsize>(bytes.size() - 1U));
  std::cout.flush();
  return std::cout ? 0 : 74;
}

int bad_magic_mode() {
  std::array<char, 20> frame{};
  frame[0] = 'B';
  frame[1] = 'A';
  frame[2] = 'D';
  frame[3] = '!';
  std::cout.write(frame.data(), static_cast<std::streamsize>(frame.size()));
  std::cout.flush();
  return std::cout ? 0 : 74;
}

[[noreturn]] void hang_mode(const std::string& marker_path) {
  {
    std::ofstream marker(marker_path, std::ios::out | std::ios::trunc);
    marker << current_process_id() << '\n';
    marker.flush();
    if (!marker) {
      std::_Exit(74);
    }
  }
  if (!emit_frame(WorkerFrameType::plan, 1, R"({"state":"planned","waiting_forever":true})")) {
    std::_Exit(74);
  }
  for (;;) {
    std::this_thread::sleep_for(std::chrono::hours(1));
  }
}

} // namespace

int main(const int argc, char** argv) {
  if (!enable_binary_stdout() || argc < 2) {
    return 64;
  }

  const std::string_view mode = argv[1];
  if (mode == "success") {
    return success_mode();
  }
  if (mode == "crash") {
    crash_mode();
  }
  if (mode == "truncated") {
    return truncated_mode();
  }
  if (mode == "bad-magic") {
    return bad_magic_mode();
  }
  if (mode == "hang" && argc == 3) {
    hang_mode(argv[2]);
  }
  return 64;
}
