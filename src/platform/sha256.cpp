#include "platform/sha256.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <span>
#include <string>

namespace xvram::platform {
namespace {

constexpr std::array<std::uint32_t, 64> round_constants{
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U,
    0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU,
    0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU,
    0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
    0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
    0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U,
    0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U,
    0xc67178f2U,
};

class Sha256 final {
public:
  [[nodiscard]] bool update(const std::span<const std::byte> input) noexcept {
    if (input.size() > (std::numeric_limits<std::uint64_t>::max() - total_bytes_)) {
      return false;
    }
    total_bytes_ += static_cast<std::uint64_t>(input.size());

    std::size_t cursor = 0;
    if (buffer_size_ != 0U) {
      const std::size_t copied = std::min(input.size(), buffer_.size() - buffer_size_);
      std::copy_n(input.data(), copied, buffer_.data() + buffer_size_);
      buffer_size_ += copied;
      cursor += copied;
      if (buffer_size_ == buffer_.size()) {
        transform(buffer_);
        buffer_size_ = 0U;
      }
    }

    while (input.size() - cursor >= buffer_.size()) {
      std::array<std::byte, 64> block{};
      std::copy_n(input.data() + cursor, block.size(), block.data());
      transform(block);
      cursor += block.size();
    }
    if (cursor != input.size()) {
      buffer_size_ = input.size() - cursor;
      std::copy_n(input.data() + cursor, buffer_size_, buffer_.data());
    }
    return true;
  }

  [[nodiscard]] std::string finish() noexcept {
    const std::uint64_t bit_count = total_bytes_ * 8U;
    buffer_[buffer_size_++] = std::byte{0x80U};
    if (buffer_size_ > 56U) {
      std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(buffer_size_), buffer_.end(),
                std::byte{0});
      transform(buffer_);
      buffer_size_ = 0U;
    }
    std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(buffer_size_), buffer_.begin() + 56,
              std::byte{0});
    for (std::size_t index = 0; index < sizeof(bit_count); ++index) {
      buffer_[63U - index] = static_cast<std::byte>((bit_count >> (index * 8U)) & 0xffU);
    }
    transform(buffer_);

    constexpr char hex[] = "0123456789abcdef";
    std::string output(64U, '0');
    std::size_t cursor = 0;
    for (const std::uint32_t word : state_) {
      for (int shift = 28; shift >= 0; shift -= 4) {
        output[cursor++] = hex[(word >> static_cast<unsigned>(shift)) & 0xfU];
      }
    }
    return output;
  }

private:
  void transform(const std::array<std::byte, 64>& block) noexcept {
    std::array<std::uint32_t, 64> words{};
    for (std::size_t index = 0; index < 16U; ++index) {
      const std::size_t offset = index * 4U;
      words[index] = (std::to_integer<std::uint32_t>(block[offset]) << 24U) |
                     (std::to_integer<std::uint32_t>(block[offset + 1U]) << 16U) |
                     (std::to_integer<std::uint32_t>(block[offset + 2U]) << 8U) |
                     std::to_integer<std::uint32_t>(block[offset + 3U]);
    }
    for (std::size_t index = 16U; index < words.size(); ++index) {
      const std::uint32_t sigma0 = std::rotr(words[index - 15U], 7) ^
                                   std::rotr(words[index - 15U], 18) ^ (words[index - 15U] >> 3U);
      const std::uint32_t sigma1 = std::rotr(words[index - 2U], 17) ^
                                   std::rotr(words[index - 2U], 19) ^ (words[index - 2U] >> 10U);
      words[index] = words[index - 16U] + sigma0 + words[index - 7U] + sigma1;
    }

    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];
    for (std::size_t index = 0; index < words.size(); ++index) {
      const std::uint32_t sum1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
      const std::uint32_t choice = (e & f) ^ (~e & g);
      const std::uint32_t temporary1 = h + sum1 + choice + round_constants[index] + words[index];
      const std::uint32_t sum0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temporary2 = sum0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temporary1;
      d = c;
      c = b;
      b = a;
      a = temporary1 + temporary2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                                      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
  std::array<std::byte, 64> buffer_{};
  std::size_t buffer_size_ = 0;
  std::uint64_t total_bytes_ = 0;
};

} // namespace

std::string sha256_bytes(const std::span<const std::byte> bytes) {
  Sha256 hash;
  if (!hash.update(bytes)) return {};
  return hash.finish();
}

FileSha256Result sha256_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return {{}, "unable to open the file for SHA-256 verification"};
  }

  Sha256 hash;
  std::array<std::byte, 64U * 1024U> buffer{};
  while (input) {
    input.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize count = input.gcount();
    if (count > 0 &&
        !hash.update(std::span<const std::byte>(buffer.data(), static_cast<std::size_t>(count)))) {
      return {{}, "file is too large for SHA-256 length accounting"};
    }
  }
  if (!input.eof()) {
    return {{}, "failed while reading the file for SHA-256 verification"};
  }
  return {hash.finish(), {}};
}

} // namespace xvram::platform
