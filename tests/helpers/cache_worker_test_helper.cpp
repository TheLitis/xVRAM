#include "residency/worker_protocol.hpp"

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

using xvram::residency::FinalWorkerPayload;
using xvram::residency::WorkerFrameType;

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

[[nodiscard]] bool write_marker(const std::string& path) {
  std::ofstream marker(path, std::ios::out | std::ios::trunc);
  marker << current_process_id() << '\n';
  marker.flush();
  return static_cast<bool>(marker);
}

[[nodiscard]] bool emit_frame(const WorkerFrameType type, const std::uint64_t sequence,
                              const std::string_view payload) {
  return xvram::residency::write_worker_frame(std::cout, type, sequence, payload);
}

int emit_final(const std::uint64_t sequence, const std::string& json, const std::string& text) {
  FinalWorkerPayload final;
  final.exit_code = 0;
  final.json = json;
  final.text = text;
  const std::string payload = xvram::residency::encode_final_worker_payload(final);
  if (payload.empty() || !emit_frame(WorkerFrameType::final, sequence, payload)) {
    return 74;
  }
  return 0;
}

int success_mode() {
  if (!emit_frame(WorkerFrameType::plan, 1, R"({"state":"planned"})") ||
      !emit_frame(WorkerFrameType::progress, 2,
                  R"({"state":"running","operations_retired":1})")) {
    return 74;
  }
  return emit_final(3, R"({"state":"completed","operations_retired":2})",
                    "cache worker completed\n");
}

[[noreturn]] void crash_mode() {
  static_cast<void>(emit_frame(WorkerFrameType::progress, 1,
                               R"({"state":"running","before_crash":true})"));
  std::_Exit(42);
}

int truncated_mode() {
  std::ostringstream encoded(std::ios::out | std::ios::binary);
  if (!xvram::residency::write_worker_frame(encoded, WorkerFrameType::progress, 1,
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

void write_u16(std::array<char, 20>& header, const std::size_t offset,
               const std::uint16_t value) {
  header[offset] = static_cast<char>(value & 0xFFU);
  header[offset + 1U] = static_cast<char>((value >> 8U) & 0xFFU);
}

void write_u32(std::array<char, 20>& header, const std::size_t offset,
               const std::uint32_t value) {
  for (unsigned int index = 0; index < 4U; ++index) {
    header[offset + index] = static_cast<char>((value >> (index * 8U)) & 0xFFU);
  }
}

void write_u64(std::array<char, 20>& header, const std::size_t offset,
               const std::uint64_t value) {
  for (unsigned int index = 0; index < 8U; ++index) {
    header[offset + index] = static_cast<char>((value >> (index * 8U)) & 0xFFU);
  }
}

int oversized_mode() {
  std::array<char, 20> header{};
  header[0] = 'X';
  header[1] = 'V';
  header[2] = 'C';
  header[3] = '1';
  write_u16(header, 4, xvram::residency::cache_worker_protocol_version);
  write_u16(header, 6, static_cast<std::uint16_t>(WorkerFrameType::progress));
  write_u32(header, 8, xvram::residency::maximum_cache_worker_payload_bytes + 1U);
  write_u64(header, 12, 1);
  std::cout.write(header.data(), static_cast<std::streamsize>(header.size()));
  std::cout.flush();
  return std::cout ? 0 : 74;
}

int trace_success_mode() {
  constexpr std::string_view first =
      R"({"report_type":"xvram.residency_trace","sequence":1})"
      "\n";
  constexpr std::string_view second =
      R"({"report_type":"xvram.residency_trace","sequence":2})"
      "\n";
  if (!emit_frame(WorkerFrameType::plan, 1, R"({"state":"planned"})") ||
      !emit_frame(WorkerFrameType::trace, 2, first) ||
      !emit_frame(WorkerFrameType::trace, 3, second) ||
      !emit_frame(WorkerFrameType::progress, 4,
                  R"({"state":"running","operations_retired":1})")) {
    return 74;
  }
  return emit_final(5, R"({"state":"completed","trace_records":2})",
                    "cache trace completed\n");
}

[[noreturn]] void hang_mode(const std::string& marker_path, const bool emit_trace) {
  if (!write_marker(marker_path)) {
    std::_Exit(74);
  }
  if (!emit_frame(WorkerFrameType::plan, 1,
                  emit_trace ? R"({"state":"trace-hang"})"
                             : R"({"state":"waiting_forever"})")) {
    std::_Exit(74);
  }
  if (emit_trace &&
      !emit_frame(WorkerFrameType::trace, 2,
                  R"({"report_type":"xvram.residency_trace","sequence":1})"
                  "\n")) {
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
  if (mode == "oversized") {
    return oversized_mode();
  }
  if (mode == "trace-success") {
    return trace_success_mode();
  }
  if (mode == "hang" && argc == 3) {
    hang_mode(argv[2], false);
  }
  if (mode == "trace-hang" && argc == 3) {
    hang_mode(argv[2], true);
  }
  return 64;
}
