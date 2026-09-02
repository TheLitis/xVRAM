#include "compression_bench/worker_protocol.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <ostream>
#include <utility>

namespace xvram::compression {
namespace {

constexpr std::array<std::uint8_t, 4> magic{'X', 'V', 'Z', '1'};
constexpr std::size_t header_bytes = 20;

void append_u16(std::string& output, const std::uint16_t value) {
  output.push_back(static_cast<char>(value & 0xFFU));
  output.push_back(static_cast<char>((value >> 8U) & 0xFFU));
}

void append_u32(std::string& output, const std::uint32_t value) {
  for (unsigned int shift = 0; shift < 32U; shift += 8U) {
    output.push_back(static_cast<char>((value >> shift) & 0xFFU));
  }
}

void append_u64(std::string& output, const std::uint64_t value) {
  for (unsigned int shift = 0; shift < 64U; shift += 8U) {
    output.push_back(static_cast<char>((value >> shift) & 0xFFU));
  }
}

[[nodiscard]] std::uint16_t read_u16(const std::uint8_t* input) {
  return static_cast<std::uint16_t>(input[0]) |
         static_cast<std::uint16_t>(static_cast<std::uint16_t>(input[1]) << 8U);
}

[[nodiscard]] std::uint32_t read_u32(const std::uint8_t* input) {
  std::uint32_t value = 0;
  for (unsigned int index = 0; index < 4U; ++index) {
    value |= static_cast<std::uint32_t>(input[index]) << (index * 8U);
  }
  return value;
}

[[nodiscard]] std::uint64_t read_u64(const std::uint8_t* input) {
  std::uint64_t value = 0;
  for (unsigned int index = 0; index < 8U; ++index) {
    value |= static_cast<std::uint64_t>(input[index]) << (index * 8U);
  }
  return value;
}

[[nodiscard]] bool valid_type(const std::uint16_t raw) {
  return raw >= static_cast<std::uint16_t>(WorkerFrameType::plan) &&
         raw <= static_cast<std::uint16_t>(WorkerFrameType::final);
}

[[nodiscard]] std::uint32_t payload_limit(const WorkerFrameType type) {
  return type == WorkerFrameType::trace ? maximum_trace_batch_bytes : maximum_worker_payload_bytes;
}

} // namespace

bool write_worker_frame(std::ostream& output, const WorkerFrameType type,
                        const std::uint64_t sequence, const std::string_view payload) {
  if (!valid_type(static_cast<std::uint16_t>(type)) || sequence == 0 ||
      payload.size() > payload_limit(type) ||
      payload.size() > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  std::string header;
  header.reserve(header_bytes);
  header.append(reinterpret_cast<const char*>(magic.data()), magic.size());
  append_u16(header, worker_protocol_version);
  append_u16(header, static_cast<std::uint16_t>(type));
  append_u32(header, static_cast<std::uint32_t>(payload.size()));
  append_u64(header, sequence);
  output.write(header.data(), static_cast<std::streamsize>(header.size()));
  output.write(payload.data(), static_cast<std::streamsize>(payload.size()));
  output.flush();
  return static_cast<bool>(output);
}

std::string encode_final_worker_payload(const FinalWorkerPayload& payload) {
  constexpr std::size_t final_header_bytes = 12U;
  if (payload.json.size() > maximum_worker_payload_bytes - final_header_bytes ||
      payload.text.size() >
          maximum_worker_payload_bytes - final_header_bytes - payload.json.size() ||
      payload.json.size() > std::numeric_limits<std::uint32_t>::max() ||
      payload.text.size() > std::numeric_limits<std::uint32_t>::max()) {
    return {};
  }
  std::string output;
  output.reserve(final_header_bytes + payload.json.size() + payload.text.size());
  append_u32(output, static_cast<std::uint32_t>(payload.exit_code));
  append_u32(output, static_cast<std::uint32_t>(payload.json.size()));
  append_u32(output, static_cast<std::uint32_t>(payload.text.size()));
  output.append(payload.json);
  output.append(payload.text);
  return output;
}

std::optional<FinalWorkerPayload> decode_final_worker_payload(const std::string_view payload,
                                                              std::string& error) {
  if (payload.size() > maximum_worker_payload_bytes) {
    error = "final compression worker payload exceeds the frame limit";
    return std::nullopt;
  }
  if (payload.size() < 12U) {
    error = "final compression worker payload is truncated";
    return std::nullopt;
  }
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(payload.data());
  const std::uint32_t exit_code = read_u32(bytes);
  const std::uint32_t json_size = read_u32(bytes + 4U);
  const std::uint32_t text_size = read_u32(bytes + 8U);
  const std::uint64_t expected_size = 12ULL + json_size + text_size;
  if (expected_size != payload.size()) {
    error = "final compression worker payload lengths are invalid";
    return std::nullopt;
  }
  FinalWorkerPayload output;
  output.exit_code = static_cast<std::int32_t>(exit_code);
  output.json.assign(payload.data() + 12, json_size);
  output.text.assign(payload.data() + 12 + json_size, text_size);
  return output;
}

bool WorkerFrameDecoder::append(const char* data, const std::size_t size,
                                std::vector<WorkerFrame>& frames, std::string& error) {
  if (size > 0 && data == nullptr) {
    error = "compression worker protocol received a null buffer";
    return false;
  }
  if (size > 0) {
    const auto* begin = reinterpret_cast<const std::uint8_t*>(data);
    buffer_.insert(buffer_.end(), begin, begin + size);
  }
  for (;;) {
    if (buffer_.size() < header_bytes) {
      return true;
    }
    if (!std::equal(magic.begin(), magic.end(), buffer_.begin())) {
      error = "compression worker protocol magic does not match";
      return false;
    }
    const std::uint16_t version = read_u16(buffer_.data() + 4U);
    const std::uint16_t raw_type = read_u16(buffer_.data() + 6U);
    const std::uint32_t payload_size = read_u32(buffer_.data() + 8U);
    const std::uint64_t sequence = read_u64(buffer_.data() + 12U);
    if (version != worker_protocol_version) {
      error = "compression worker protocol version is unsupported";
      return false;
    }
    if (!valid_type(raw_type)) {
      error = "compression worker protocol frame type is invalid";
      return false;
    }
    const auto type = static_cast<WorkerFrameType>(raw_type);
    if (payload_size > payload_limit(type)) {
      error = "compression worker protocol payload exceeds the frame limit";
      return false;
    }
    const std::size_t frame_size = header_bytes + payload_size;
    if (buffer_.size() < frame_size) {
      return true;
    }
    if (sequence == 0 || (have_sequence_ && sequence <= last_sequence_)) {
      error = "compression worker protocol sequence did not increase";
      return false;
    }
    if (state_ == State::awaiting_plan && type != WorkerFrameType::plan) {
      error = "compression worker protocol must begin with a plan frame";
      return false;
    }
    if (state_ == State::running && type == WorkerFrameType::plan) {
      error = "compression worker protocol contains a duplicate plan frame";
      return false;
    }
    if (state_ == State::finished) {
      error = "compression worker protocol contains data after the final frame";
      return false;
    }

    WorkerFrame frame;
    frame.type = type;
    frame.sequence = sequence;
    frame.payload.assign(reinterpret_cast<const char*>(buffer_.data() + header_bytes),
                         payload_size);
    frames.push_back(std::move(frame));
    buffer_.erase(buffer_.begin(), buffer_.begin() + static_cast<std::ptrdiff_t>(frame_size));
    last_sequence_ = sequence;
    have_sequence_ = true;
    if (type == WorkerFrameType::plan) {
      state_ = State::running;
    } else if (type == WorkerFrameType::final) {
      state_ = State::finished;
    }
  }
}

bool WorkerFrameDecoder::finish(std::string& error) const {
  if (!buffer_.empty()) {
    error = "compression worker protocol ended with a truncated frame";
    return false;
  }
  if (state_ != State::finished) {
    error = "compression worker protocol ended before the final frame";
    return false;
  }
  return true;
}

} // namespace xvram::compression
