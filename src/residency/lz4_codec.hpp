#pragma once

#include "residency/compression.hpp"

namespace xvram::residency {

// Upstream LZ4 1.10 raw-block adapter. The on-disk/host container is owned by
// HostBackingStore; this class deliberately exposes no frame-format behavior.
class Lz4BlockCodec final : public BlockCodec {
public:
  [[nodiscard]] CompressionCodec codec() const noexcept override;
  [[nodiscard]] std::string_view implementation_name() const noexcept override;
  [[nodiscard]] std::optional<std::size_t>
  maximum_compressed_bytes(std::size_t input_bytes) const noexcept override;
  [[nodiscard]] CodecStatus compress(std::span<const std::byte> input, std::span<std::byte> output,
                                     std::size_t& written_bytes) const noexcept override;
  [[nodiscard]] CodecStatus decompress(std::span<const std::byte> input,
                                       std::span<std::byte> output) const noexcept override;
};

} // namespace xvram::residency
