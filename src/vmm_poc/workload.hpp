#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace xvram::vmm_poc {

inline constexpr std::uint64_t word_bytes = sizeof(std::uint32_t);

[[nodiscard]] std::optional<std::uint64_t> checked_add(std::uint64_t left,
                                                       std::uint64_t right) noexcept;
[[nodiscard]] std::optional<std::uint64_t> checked_multiply(std::uint64_t left,
                                                            std::uint64_t right) noexcept;
[[nodiscard]] std::optional<std::uint64_t> checked_align_up(std::uint64_t value,
                                                            std::uint64_t alignment) noexcept;
[[nodiscard]] std::optional<std::uint64_t>
checked_least_common_multiple(std::uint64_t left, std::uint64_t right) noexcept;

enum class PlanError {
  none,
  invalid_configuration,
  arithmetic_overflow,
  logical_size_not_oversubscribed,
  insufficient_host_memory,
  insufficient_device_budget,
};

[[nodiscard]] std::string_view plan_error_name(PlanError error) noexcept;

struct PlanningInput {
  std::optional<std::uint64_t> requested_logical_bytes;
  std::uint64_t requested_chunk_bytes = 64ULL * 1024ULL * 1024ULL;
  std::uint32_t passes = 2;
  std::uint32_t window_slots = 1;

  std::uint64_t total_device_bytes = 0;
  std::uint64_t free_device_bytes = 0;
  std::optional<std::uint64_t> available_device_budget_bytes;
  std::uint64_t minimum_granularity_bytes = 0;
  std::optional<std::uint64_t> recommended_granularity_bytes;

  std::uint64_t physical_host_bytes = 0;
  std::uint64_t available_host_bytes = 0;
  std::uint64_t service_host_bytes = 256ULL * 1024ULL * 1024ULL;
  std::uint64_t device_headroom_bytes = 256ULL * 1024ULL * 1024ULL;
};

struct WorkloadPlan {
  bool logical_size_automatic = false;
  std::uint64_t requested_logical_bytes = 0;
  std::uint64_t effective_logical_bytes = 0;
  std::uint64_t requested_chunk_bytes = 0;
  std::uint64_t effective_chunk_bytes = 0;
  std::uint64_t effective_alignment_bytes = 0;
  std::uint32_t passes = 0;
  std::uint32_t window_slots = 0;

  std::uint64_t logical_element_count = 0;
  std::uint64_t logical_chunk_count = 0;
  std::uint64_t tail_chunk_bytes = 0;
  std::uint64_t tile_visit_count = 0;
  std::uint64_t address_revisit_count = 0;

  std::uint64_t backing_bytes = 0;
  std::uint64_t pinned_staging_bytes = 0;
  std::uint64_t resident_physical_bytes = 0;
  std::uint64_t host_headroom_bytes = 0;
  std::uint64_t safe_logical_limit_bytes = 0;
};

struct PlanningResult {
  PlanError error = PlanError::none;
  std::optional<WorkloadPlan> plan;

  [[nodiscard]] explicit operator bool() const noexcept {
    return error == PlanError::none && plan.has_value();
  }
};

[[nodiscard]] PlanningResult make_workload_plan(const PlanningInput& input) noexcept;

struct TileVisit {
  std::uint64_t ordinal = 0;
  std::uint32_t pass_index = 0;
  std::uint64_t tile_index = 0;
  bool forward = true;
};

[[nodiscard]] std::optional<TileVisit> tile_visit_at(std::uint64_t logical_chunk_count,
                                                     std::uint32_t passes,
                                                     std::uint64_t ordinal) noexcept;

// These word-level primitives deliberately live in the header: the proof applies them to
// billions of elements, and keeping them visible to the optimizer avoids a function call per
// word while preserving one canonical CPU definition for tests and verification.
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

[[nodiscard]] inline std::uint32_t transform_word(std::uint32_t value,
                                                  const std::uint64_t global_index,
                                                  const std::uint32_t pass_index,
                                                  const std::uint64_t seed) noexcept {
  const std::uint32_t low = static_cast<std::uint32_t>(global_index);
  const std::uint32_t high = static_cast<std::uint32_t>(global_index >> 32U);
  const std::uint32_t seed_low = static_cast<std::uint32_t>(seed);
  const std::uint32_t seed_high = static_cast<std::uint32_t>(seed >> 32U);
  const std::uint32_t mix = (low * 0x9E3779B9U) ^ (high * 0x85EBCA6BU) ^ seed_low ^
                            (seed_high * 0x165667B1U) ^ ((pass_index + 1U) * 0xC2B2AE35U);
  value ^= mix;
  value = value * 1'664'525U + 1'013'904'223U;
  value ^= value >> 16U;
  return value * 0x27D4EB2DU;
}

[[nodiscard]] inline std::uint32_t expected_word(const std::uint64_t global_index,
                                                 const std::uint32_t passes,
                                                 const std::uint64_t seed) noexcept {
  std::uint32_t value = initial_word(global_index, seed);
  for (std::uint32_t pass = 0; pass < passes; ++pass) {
    value = transform_word(value, global_index, pass, seed);
  }
  return value;
}

struct Digest128Value {
  std::uint64_t low = 0;
  std::uint64_t high = 0;

  [[nodiscard]] bool operator==(const Digest128Value&) const noexcept = default;
};

[[nodiscard]] std::string format_digest128(Digest128Value digest);

// A deterministic, order-sensitive pair of independent 64-bit lanes. This is an
// integrity checksum for the synthetic proof workload, not a cryptographic hash.
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

struct SampleSummary {
  std::uint64_t count = 0;
  double minimum = 0.0;
  double median = 0.0;
  double percentile_95 = 0.0;
  double maximum = 0.0;
  double total = 0.0;
  double mean = 0.0;
};

// Uses linear interpolation at q * (n - 1), matching the common R-7 definition.
[[nodiscard]] std::optional<SampleSummary> summarize_samples(std::span<const double> samples);

} // namespace xvram::vmm_poc
