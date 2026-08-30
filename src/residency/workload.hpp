#pragma once

#include "residency/core.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <string>

namespace xvram::residency {

inline constexpr std::uint64_t workload_word_bytes = sizeof(std::uint32_t);
inline constexpr std::uint32_t write_only_initial_value = 0xA5A5A5A5U;

[[nodiscard]] std::optional<std::uint64_t> checked_add(std::uint64_t left,
                                                       std::uint64_t right) noexcept;
[[nodiscard]] std::optional<std::uint64_t> checked_multiply(std::uint64_t left,
                                                            std::uint64_t right) noexcept;
[[nodiscard]] std::optional<std::uint64_t> checked_align_up(std::uint64_t value,
                                                            std::uint64_t alignment) noexcept;
[[nodiscard]] std::optional<std::uint64_t>
checked_least_common_multiple(std::uint64_t left, std::uint64_t right) noexcept;

static_assert(static_cast<std::uint32_t>(AccessMode::read) == 0U);
static_assert(static_cast<std::uint32_t>(AccessMode::read_write) == 1U);
static_assert(static_cast<std::uint32_t>(AccessMode::write_only) == 2U);

// Canonical pageable-backing pattern for Phase 2. These word primitives are inline because full
// verification applies them to every logical element.
[[nodiscard]] inline std::uint32_t initial_word(const std::uint64_t global_index,
                                                const std::uint64_t seed) noexcept {
  const std::uint32_t low = static_cast<std::uint32_t>(global_index);
  const std::uint32_t high = static_cast<std::uint32_t>(global_index >> 32U);
  const std::uint32_t seed_low = static_cast<std::uint32_t>(seed);
  const std::uint32_t seed_high = static_cast<std::uint32_t>(seed >> 32U);
  std::uint32_t value = low ^ (high * 0x9E3779B9U) ^ seed_low ^ (seed_high * 0x85EBCA6BU);
  value ^= value >> 16U;
  value *= 0x7FEB352DU;
  value ^= value >> 15U;
  value *= 0x846CA68BU;
  return value ^ (value >> 16U);
}

// Bit-for-bit equivalent to %r20 in residency_workload_v1.ptx.
[[nodiscard]] inline std::uint32_t operation_mix(const std::uint64_t global_index,
                                                 const std::uint32_t operation_index,
                                                 const std::uint64_t seed) noexcept {
  const std::uint32_t low = static_cast<std::uint32_t>(global_index);
  const std::uint32_t high = static_cast<std::uint32_t>(global_index >> 32U);
  const std::uint32_t seed_low = static_cast<std::uint32_t>(seed);
  const std::uint32_t seed_high = static_cast<std::uint32_t>(seed >> 32U);
  return (low * 0x9E3779B9U) ^ (high * 0x85EBCA6BU) ^ seed_low ^ (seed_high * 0x165667B1U) ^
         ((operation_index + std::uint32_t{1}) * 0xC2B2AE35U);
}

// Read leaves the loaded value unchanged. Read-write transforms it in place. Write-only ignores
// the prior contents and transforms the PTX constant 0xa5a5a5a5.
[[nodiscard]] inline std::uint32_t transform_word(std::uint32_t value,
                                                  const std::uint64_t global_index,
                                                  const std::uint32_t operation_index,
                                                  const std::uint64_t seed,
                                                  const AccessMode mode) noexcept {
  if (mode == AccessMode::read) {
    return value;
  }
  if (mode == AccessMode::write_only) {
    value = write_only_initial_value;
  }
  value ^= operation_mix(global_index, operation_index, seed);
  value = value * 0x0019660DU + 0x3C6EF35FU;
  value ^= value >> 16U;
  return value * 0x27D4EB2DU;
}

// Computes the token produced by the PTX kernel from pre-operation words, including the transform
// for write modes. Empty spans have token zero. A span crossing UINT64_MAX is rejected.
[[nodiscard]] std::optional<std::uint32_t>
expected_verification_token(std::span<const std::uint32_t> input_words, std::uint64_t global_start,
                            std::uint32_t operation_index, std::uint64_t seed,
                            AccessMode mode) noexcept;

// Computes the same XOR token from already transformed/output words.
[[nodiscard]] std::optional<std::uint32_t>
verification_token_for_output(std::span<const std::uint32_t> output_words,
                              std::uint64_t global_start, std::uint32_t operation_index,
                              std::uint64_t seed) noexcept;

struct Digest128Value {
  std::uint64_t low = 0;
  std::uint64_t high = 0;

  [[nodiscard]] bool operator==(const Digest128Value&) const noexcept = default;
};

[[nodiscard]] std::string format_digest128(Digest128Value digest);

// Deterministic, order-sensitive integrity checksum. It is deliberately not cryptographic.
class Digest128Accumulator {
public:
  void update(std::uint32_t value, std::uint64_t global_index) noexcept;
  [[nodiscard]] bool update_words(std::span<const std::uint32_t> values,
                                  std::uint64_t first_global_index) noexcept;
  [[nodiscard]] Digest128Value value() const noexcept;
  [[nodiscard]] std::uint64_t count() const noexcept {
    return count_;
  }

private:
  std::uint64_t low_ = 0x243F6A8885A308D3ULL;
  std::uint64_t high_ = 0x13198A2E03707344ULL;
  std::uint64_t count_ = 0;
};

struct PercentileSummary {
  std::uint64_t count = 0;
  double minimum = 0.0;
  double median = 0.0;
  double percentile_95 = 0.0;
  double maximum = 0.0;
  double total = 0.0;
  double mean = 0.0;
};

// Uses linear interpolation at q * (n - 1), matching the common R-7 definition.
[[nodiscard]] std::optional<PercentileSummary> summarize_timings(std::span<const double> samples);

} // namespace xvram::residency
