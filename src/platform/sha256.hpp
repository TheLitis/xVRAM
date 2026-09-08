#pragma once

#include <filesystem>
#include <span>
#include <cstddef>
#include <string>

namespace xvram::platform {

struct FileSha256Result {
  std::string digest;
  std::string error;

  [[nodiscard]] explicit operator bool() const noexcept {
    return !digest.empty();
  }
};

// Hash the bytes currently stored at an exact filesystem path. The lowercase hexadecimal digest
// is returned only after the whole file has been read successfully.
[[nodiscard]] FileSha256Result sha256_file(const std::filesystem::path& path);
// Internal CPU utility; no native SDK export.
[[nodiscard]] std::string sha256_bytes(std::span<const std::byte> bytes);

} // namespace xvram::platform
