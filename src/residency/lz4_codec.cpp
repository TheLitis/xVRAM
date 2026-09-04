#include "residency/lz4_codec.hpp"

#include <lz4.h>

#include <limits>

namespace xvram::residency {

CompressionCodec Lz4BlockCodec::codec() const noexcept {
  return CompressionCodec::lz4;
}

std::string_view Lz4BlockCodec::implementation_name() const noexcept {
  return "lz4-1.10.0";
}

std::optional<std::size_t>
Lz4BlockCodec::maximum_compressed_bytes(const std::size_t input_bytes) const noexcept {
  if (input_bytes > static_cast<std::size_t>(LZ4_MAX_INPUT_SIZE) ||
      input_bytes > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return std::nullopt;
  }
  const int bound = LZ4_compressBound(static_cast<int>(input_bytes));
  return bound > 0 ? std::optional<std::size_t>{static_cast<std::size_t>(bound)} : std::nullopt;
}

CodecStatus Lz4BlockCodec::compress(const std::span<const std::byte> input,
                                    const std::span<std::byte> output,
                                    std::size_t& written_bytes) const noexcept {
  written_bytes = 0;
  if (input.empty() || input.size() > static_cast<std::size_t>(LZ4_MAX_INPUT_SIZE) ||
      input.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      output.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return CodecStatus::invalid_argument;
  }
  const auto bound = maximum_compressed_bytes(input.size());
  if (!bound.has_value()) {
    return CodecStatus::invalid_argument;
  }
  if (output.size() < *bound) {
    return CodecStatus::output_too_small;
  }
  const int compressed = LZ4_compress_default(
      reinterpret_cast<const char*>(input.data()), reinterpret_cast<char*>(output.data()),
      static_cast<int>(input.size()), static_cast<int>(output.size()));
  if (compressed <= 0) {
    return CodecStatus::internal_failure;
  }
  written_bytes = static_cast<std::size_t>(compressed);
  return CodecStatus::success;
}

CodecStatus Lz4BlockCodec::decompress(const std::span<const std::byte> input,
                                      const std::span<std::byte> output) const noexcept {
  if (input.empty() || output.empty() ||
      input.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      output.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return CodecStatus::invalid_argument;
  }
  const int decoded = LZ4_decompress_safe(
      reinterpret_cast<const char*>(input.data()), reinterpret_cast<char*>(output.data()),
      static_cast<int>(input.size()), static_cast<int>(output.size()));
  if (decoded < 0) {
    return CodecStatus::corrupt_input;
  }
  return decoded == static_cast<int>(output.size()) ? CodecStatus::success
                                                    : CodecStatus::corrupt_input;
}

} // namespace xvram::residency
