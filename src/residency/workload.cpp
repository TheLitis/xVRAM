#include "residency/workload.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

namespace xvram::residency {
namespace {

[[nodiscard]] std::uint64_t avalanche64(std::uint64_t value) noexcept {
  value ^= value >> 30U;
  value *= 0xBF58476D1CE4E5B9ULL;
  value ^= value >> 27U;
  value *= 0x94D049BB133111EBULL;
  return value ^ (value >> 31U);
}

[[nodiscard]] bool span_indices_fit(const std::size_t size,
                                    const std::uint64_t global_start) noexcept {
  if (size == 0) {
    return true;
  }
  const std::uint64_t count = static_cast<std::uint64_t>(size);
  return count - 1U <= std::numeric_limits<std::uint64_t>::max() - global_start;
}

[[nodiscard]] double interpolated_percentile(const std::vector<double>& sorted,
                                             const double quantile) {
  const double position = quantile * static_cast<double>(sorted.size() - 1U);
  const auto lower_index = static_cast<std::size_t>(std::floor(position));
  const auto upper_index = static_cast<std::size_t>(std::ceil(position));
  const double fraction = position - static_cast<double>(lower_index);
  return sorted[lower_index] + (sorted[upper_index] - sorted[lower_index]) * fraction;
}

} // namespace

std::optional<std::uint64_t> checked_add(const std::uint64_t left,
                                         const std::uint64_t right) noexcept {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    return std::nullopt;
  }
  return left + right;
}

std::optional<std::uint64_t> checked_multiply(const std::uint64_t left,
                                              const std::uint64_t right) noexcept {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
    return std::nullopt;
  }
  return left * right;
}

std::optional<std::uint64_t> checked_align_up(const std::uint64_t value,
                                              const std::uint64_t alignment) noexcept {
  if (alignment == 0) {
    return std::nullopt;
  }
  const std::uint64_t remainder = value % alignment;
  if (remainder == 0) {
    return value;
  }
  return checked_add(value, alignment - remainder);
}

std::optional<std::uint64_t> checked_least_common_multiple(const std::uint64_t left,
                                                           const std::uint64_t right) noexcept {
  if (left == 0 || right == 0) {
    return 0;
  }
  return checked_multiply(left / std::gcd(left, right), right);
}

std::optional<std::uint32_t>
expected_verification_token(const std::span<const std::uint32_t> input_words,
                            const std::uint64_t global_start, const std::uint32_t operation_index,
                            const std::uint64_t seed, const AccessMode mode) noexcept {
  if (!span_indices_fit(input_words.size(), global_start)) {
    return std::nullopt;
  }
  std::uint32_t token = 0;
  for (std::size_t index = 0; index < input_words.size(); ++index) {
    const std::uint64_t global_index = global_start + static_cast<std::uint64_t>(index);
    const std::uint32_t transformed =
        transform_word(input_words[index], global_index, operation_index, seed, mode);
    token ^= transformed ^ operation_mix(global_index, operation_index, seed);
  }
  return token;
}

std::optional<std::uint32_t>
verification_token_for_output(const std::span<const std::uint32_t> output_words,
                              const std::uint64_t global_start, const std::uint32_t operation_index,
                              const std::uint64_t seed) noexcept {
  if (!span_indices_fit(output_words.size(), global_start)) {
    return std::nullopt;
  }
  std::uint32_t token = 0;
  for (std::size_t index = 0; index < output_words.size(); ++index) {
    const std::uint64_t global_index = global_start + static_cast<std::uint64_t>(index);
    token ^= output_words[index] ^ operation_mix(global_index, operation_index, seed);
  }
  return token;
}

void Digest128Accumulator::update(const std::uint32_t value,
                                  const std::uint64_t global_index) noexcept {
  const std::uint64_t key =
      avalanche64(global_index ^ (static_cast<std::uint64_t>(value) << 32U) ^ count_);
  low_ ^= key;
  low_ = std::rotl(low_, 27) * 0x3C79AC492BA7B653ULL + 0x1C69B3F74AC4AE35ULL;
  high_ += key ^ std::rotl(global_index, 17);
  high_ = std::rotl(high_, 31) * 0x9E3779B185EBCA87ULL + 0xD1B54A32D192ED03ULL;
  ++count_;
}

bool Digest128Accumulator::update_words(const std::span<const std::uint32_t> values,
                                        const std::uint64_t first_global_index) noexcept {
  const std::uint64_t size = static_cast<std::uint64_t>(values.size());
  if (size > std::numeric_limits<std::uint64_t>::max() - count_ ||
      !span_indices_fit(values.size(), first_global_index)) {
    return false;
  }
  for (std::size_t index = 0; index < values.size(); ++index) {
    update(values[index], first_global_index + static_cast<std::uint64_t>(index));
  }
  return true;
}

Digest128Value Digest128Accumulator::value() const noexcept {
  const std::uint64_t low = avalanche64(low_ ^ count_ ^ 0xA0761D6478BD642FULL);
  const std::uint64_t high =
      avalanche64(high_ ^ std::rotl(count_, 32) ^ low ^ 0xE7037ED1A0B428DBULL);
  return {low, high};
}

std::string format_digest128(const Digest128Value digest) {
  static constexpr char hex[] = "0123456789abcdef";
  std::string result(32, '0');
  const auto write_lane = [&](const std::uint64_t lane, const std::size_t offset) {
    for (std::size_t index = 0; index < 16U; ++index) {
      const std::size_t shift = (15U - index) * 4U;
      result[offset + index] = hex[(lane >> shift) & 0xFU];
    }
  };
  write_lane(digest.high, 0);
  write_lane(digest.low, 16);
  return result;
}

std::optional<PercentileSummary> summarize_timings(const std::span<const double> samples) {
  if (samples.empty()) {
    return std::nullopt;
  }

  std::vector<double> sorted;
  sorted.reserve(samples.size());
  double total = 0.0;
  for (const double sample : samples) {
    if (!std::isfinite(sample) || sample < 0.0) {
      return std::nullopt;
    }
    total += sample;
    if (!std::isfinite(total)) {
      return std::nullopt;
    }
    sorted.push_back(sample);
  }
  std::sort(sorted.begin(), sorted.end());

  PercentileSummary summary;
  summary.count = static_cast<std::uint64_t>(sorted.size());
  summary.minimum = sorted.front();
  summary.median = interpolated_percentile(sorted, 0.5);
  summary.percentile_95 = interpolated_percentile(sorted, 0.95);
  summary.maximum = sorted.back();
  summary.total = total;
  summary.mean = total / static_cast<double>(sorted.size());
  return summary;
}

} // namespace xvram::residency
