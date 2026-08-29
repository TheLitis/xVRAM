#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace xvram {

struct ParsedSize {
  std::optional<std::uint64_t> bytes;
  std::string error;

  [[nodiscard]] explicit operator bool() const noexcept {
    return bytes.has_value();
  }
};

[[nodiscard]] ParsedSize parse_size(std::string_view text);
[[nodiscard]] std::string format_bytes(std::uint64_t bytes);

} // namespace xvram
