#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace xvram::residency {

inline constexpr std::uint16_t cache_worker_protocol_version = 1;
inline constexpr std::uint32_t maximum_cache_worker_payload_bytes = 1024U * 1024U;
inline constexpr std::uint32_t maximum_cache_trace_batch_bytes = 64U * 1024U;

enum class WorkerFrameType : std::uint16_t { plan = 1, progress = 2, trace = 3, final = 4 };

struct WorkerFrame {
  WorkerFrameType type = WorkerFrameType::progress;
  std::uint64_t sequence = 0;
  std::string payload;
};

struct FinalWorkerPayload {
  std::int32_t exit_code = 70;
  std::string json;
  std::string text;
};

[[nodiscard]] bool write_worker_frame(std::ostream& output, WorkerFrameType type,
                                      std::uint64_t sequence, std::string_view payload);
[[nodiscard]] std::string encode_final_worker_payload(const FinalWorkerPayload& payload);
[[nodiscard]] std::optional<FinalWorkerPayload>
decode_final_worker_payload(std::string_view payload, std::string& error);

class WorkerFrameDecoder {
public:
  [[nodiscard]] bool append(const char* data, std::size_t size, std::vector<WorkerFrame>& frames,
                            std::string& error);
  [[nodiscard]] bool finish(std::string& error) const;

private:
  std::vector<std::uint8_t> buffer_;
  std::uint64_t last_sequence_ = 0;
  bool have_sequence_ = false;
};

} // namespace xvram::residency
