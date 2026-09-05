#include "compat_bench/worker_protocol.hpp"

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

using xvram::compat_bench::FinalWorkerPayload;
using xvram::compat_bench::WorkerFrameType;

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

[[nodiscard]] bool write_marker(const std::string& marker_path) {
  std::ofstream marker(marker_path, std::ios::out | std::ios::trunc);
  marker << current_process_id() << '\n';
  marker.flush();
  return static_cast<bool>(marker);
}

[[nodiscard]] bool emit(const WorkerFrameType type, const std::uint64_t sequence,
                        const std::string_view payload) {
  return xvram::compat_bench::write_worker_frame(std::cout, type, sequence, payload);
}

int success_mode() {
  if (!emit(WorkerFrameType::plan, 1U, R"({"state":"planned"})") ||
      !emit(WorkerFrameType::heartbeat, 2U, "{}") || !emit(WorkerFrameType::progress, 3U, "1")) {
    return 74;
  }
  FinalWorkerPayload final;
  final.exit_code = 0;
  final.json = R"({"state":"completed"})";
  final.text = "compatibility worker completed\n";
  const std::string encoded = xvram::compat_bench::encode_final_worker_payload(final);
  return !encoded.empty() && emit(WorkerFrameType::final, 4U, encoded) ? 0 : 74;
}

[[noreturn]] void crash_mode(const std::string& marker_path) {
  if (!write_marker(marker_path) || !emit(WorkerFrameType::plan, 1U, R"({"state":"planned"})") ||
      !emit(WorkerFrameType::progress, 2U, "1")) {
    std::_Exit(74);
  }
  std::_Exit(42);
}

[[noreturn]] void hang_mode(const std::string& marker_path, const bool emit_trace) {
  if (!write_marker(marker_path) ||
      !emit(WorkerFrameType::plan, 1U,
            emit_trace ? R"({"state":"trace_sink_wait"})" : R"({"state":"silent_wait"})")) {
    std::_Exit(74);
  }
  if (emit_trace && !emit(WorkerFrameType::trace, 2U,
                          R"({"report_type":"xvram.cuda_compat_trace","sequence":1})"
                          "\n")) {
    std::_Exit(74);
  }
  for (;;) {
    std::this_thread::sleep_for(std::chrono::hours(1));
  }
}

[[noreturn]] void non_monotonic_mode(const std::string& marker_path) {
  if (!write_marker(marker_path) || !emit(WorkerFrameType::plan, 1U, R"({"state":"planned"})") ||
      !emit(WorkerFrameType::progress, 2U, "1") ||
      !emit(WorkerFrameType::heartbeat, 2U, "{}")) {
    std::_Exit(74);
  }
  for (;;) {
    std::this_thread::sleep_for(std::chrono::hours(1));
  }
}

[[noreturn]] void heartbeat_hang_mode(const std::string& marker_path) {
  if (!write_marker(marker_path) || !emit(WorkerFrameType::plan, 1U, R"({"state":"waiting"})")) {
    std::_Exit(74);
  }
  std::uint64_t sequence = 2U;
  for (;;) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    if (!emit(WorkerFrameType::heartbeat, sequence++, "{}")) {
      std::_Exit(74);
    }
  }
}

int truncated_mode() {
  std::ostringstream stream(std::ios::binary);
  if (!xvram::compat_bench::write_worker_frame(stream, WorkerFrameType::plan, 1U, "{}")) {
    return 70;
  }
  const std::string encoded = stream.str();
  std::cout.write(encoded.data(), static_cast<std::streamsize>(encoded.size() - 1U));
  std::cout.flush();
  return std::cout ? 0 : 74;
}

void write_u16(std::array<char, 20>& output, const std::size_t offset, const std::uint16_t value) {
  output[offset] = static_cast<char>(value & 0xFFU);
  output[offset + 1U] = static_cast<char>((value >> 8U) & 0xFFU);
}

void write_u32(std::array<char, 20>& output, const std::size_t offset, const std::uint32_t value) {
  for (unsigned int index = 0; index < 4U; ++index) {
    output[offset + index] = static_cast<char>((value >> (index * 8U)) & 0xFFU);
  }
}

void write_u64(std::array<char, 20>& output, const std::size_t offset, const std::uint64_t value) {
  for (unsigned int index = 0; index < 8U; ++index) {
    output[offset + index] = static_cast<char>((value >> (index * 8U)) & 0xFFU);
  }
}

int oversized_mode() {
  std::array<char, 20> header{};
  header[0] = 'X';
  header[1] = 'V';
  header[2] = 'I';
  header[3] = '1';
  write_u16(header, 4U, xvram::compat_bench::worker_protocol_version);
  write_u16(header, 6U, static_cast<std::uint16_t>(WorkerFrameType::progress));
  write_u32(header, 8U, xvram::compat_bench::maximum_worker_payload_bytes + 1U);
  write_u64(header, 12U, 1U);
  std::cout.write(header.data(), static_cast<std::streamsize>(header.size()));
  std::cout.flush();
  return std::cout ? 0 : 74;
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
  if (mode == "crash" && argc == 3) {
    crash_mode(argv[2]);
  }
  if (mode == "silent-hang" && argc == 3) {
    hang_mode(argv[2], false);
  }
  if (mode == "trace-sink-hang" && argc == 3) {
    hang_mode(argv[2], true);
  }
  if (mode == "non-monotonic" && argc == 3) {
    non_monotonic_mode(argv[2]);
  }
  if (mode == "heartbeat-hang" && argc == 3) {
    heartbeat_hang_mode(argv[2]);
  }
  if (mode == "truncated") {
    if (argc == 3 && !write_marker(argv[2])) {
      return 74;
    }
    return truncated_mode();
  }
  if (mode == "oversized") {
    if (argc == 3 && !write_marker(argv[2])) {
      return 74;
    }
    return oversized_mode();
  }
  return 64;
}
