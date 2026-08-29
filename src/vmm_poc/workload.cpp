#include "vmm_poc/workload.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

namespace xvram::vmm_poc {
namespace {

[[nodiscard]] std::uint64_t avalanche64(std::uint64_t value) noexcept {
  value ^= value >> 30U;
  value *= 0xBF58476D1CE4E5B9ULL;
  value ^= value >> 27U;
  value *= 0x94D049BB133111EBULL;
  return value ^ (value >> 31U);
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
  const std::uint64_t reduced = left / std::gcd(left, right);
  return checked_multiply(reduced, right);
}

std::string_view plan_error_name(const PlanError error) noexcept {
  switch (error) {
  case PlanError::none:
    return "none";
  case PlanError::invalid_configuration:
    return "invalid_configuration";
  case PlanError::arithmetic_overflow:
    return "arithmetic_overflow";
  case PlanError::logical_size_not_oversubscribed:
    return "logical_size_not_oversubscribed";
  case PlanError::insufficient_host_memory:
    return "insufficient_host_memory";
  case PlanError::insufficient_device_budget:
    return "insufficient_device_budget";
  }
  return "invalid_configuration";
}

PlanningResult make_workload_plan(const PlanningInput& input) noexcept {
  const auto fail = [](const PlanError error) { return PlanningResult{error, std::nullopt}; };

  if (input.requested_chunk_bytes == 0 || input.total_device_bytes == 0 ||
      input.minimum_granularity_bytes == 0 || input.physical_host_bytes == 0 ||
      input.available_host_bytes == 0 || input.passes < 2U || input.passes > 8U ||
      input.window_slots == 0U || input.window_slots > 8U ||
      (input.requested_logical_bytes.has_value() &&
       (*input.requested_logical_bytes == 0 || *input.requested_logical_bytes % word_bytes != 0))) {
    return fail(PlanError::invalid_configuration);
  }

  const std::uint64_t recommended =
      input.recommended_granularity_bytes.value_or(input.minimum_granularity_bytes);
  if (recommended == 0) {
    return fail(PlanError::invalid_configuration);
  }
  const std::optional<std::uint64_t> device_alignment =
      checked_least_common_multiple(input.minimum_granularity_bytes, recommended);
  if (!device_alignment.has_value()) {
    return fail(PlanError::arithmetic_overflow);
  }
  const std::optional<std::uint64_t> effective_alignment =
      checked_least_common_multiple(*device_alignment, word_bytes);
  if (!effective_alignment.has_value() || *effective_alignment == 0) {
    return fail(effective_alignment.has_value() ? PlanError::invalid_configuration
                                                : PlanError::arithmetic_overflow);
  }

  const std::optional<std::uint64_t> effective_chunk =
      checked_align_up(input.requested_chunk_bytes, *effective_alignment);
  if (!effective_chunk.has_value() || *effective_chunk == 0) {
    return fail(effective_chunk.has_value() ? PlanError::invalid_configuration
                                            : PlanError::arithmetic_overflow);
  }

  const std::optional<std::uint64_t> window_bytes =
      checked_multiply(*effective_chunk, static_cast<std::uint64_t>(input.window_slots));
  if (!window_bytes.has_value()) {
    return fail(PlanError::arithmetic_overflow);
  }

  constexpr std::uint64_t minimum_host_headroom = 4ULL * 1024ULL * 1024ULL * 1024ULL;
  const std::uint64_t host_headroom =
      std::max<std::uint64_t>(minimum_host_headroom, input.physical_host_bytes / std::uint64_t{4});
  const std::optional<std::uint64_t> host_reserved_with_staging =
      checked_add(host_headroom, *window_bytes);
  const std::optional<std::uint64_t> host_reserved =
      host_reserved_with_staging.has_value()
          ? checked_add(*host_reserved_with_staging, input.service_host_bytes)
          : std::nullopt;
  if (!host_reserved.has_value()) {
    return fail(PlanError::arithmetic_overflow);
  }
  if (*host_reserved >= input.available_host_bytes) {
    return fail(PlanError::insufficient_host_memory);
  }
  const std::uint64_t safe_logical_limit =
      ((input.available_host_bytes - *host_reserved) / word_bytes) * word_bytes;

  const bool automatic = !input.requested_logical_bytes.has_value();
  std::uint64_t requested_logical = 0;
  if (automatic) {
    const std::optional<std::uint64_t> half_device_rounded_up =
        checked_add(input.total_device_bytes / 2ULL, input.total_device_bytes % 2ULL);
    const std::optional<std::uint64_t> automatic_target =
        half_device_rounded_up.has_value()
            ? checked_add(input.total_device_bytes, *half_device_rounded_up)
            : std::nullopt;
    if (!automatic_target.has_value()) {
      return fail(PlanError::arithmetic_overflow);
    }
    requested_logical = (std::min(*automatic_target, safe_logical_limit) / word_bytes) * word_bytes;
  } else {
    requested_logical = *input.requested_logical_bytes;
  }
  if (requested_logical <= input.total_device_bytes) {
    return fail(automatic ? PlanError::insufficient_host_memory
                          : PlanError::logical_size_not_oversubscribed);
  }
  if (requested_logical > safe_logical_limit) {
    return fail(PlanError::insufficient_host_memory);
  }
  if (requested_logical > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return fail(PlanError::invalid_configuration);
  }

  if (*window_bytes > input.free_device_bytes ||
      input.free_device_bytes - *window_bytes < input.device_headroom_bytes) {
    return fail(PlanError::insufficient_device_budget);
  }
  if (input.available_device_budget_bytes.has_value() &&
      (*window_bytes > *input.available_device_budget_bytes ||
       *input.available_device_budget_bytes - *window_bytes < input.device_headroom_bytes)) {
    return fail(PlanError::insufficient_device_budget);
  }

  const std::uint64_t chunk_count = ((requested_logical - 1ULL) / *effective_chunk) + 1ULL;
  const std::optional<std::uint64_t> bytes_before_tail =
      checked_multiply(chunk_count - 1ULL, *effective_chunk);
  if (!bytes_before_tail.has_value()) {
    return fail(PlanError::arithmetic_overflow);
  }
  const std::uint64_t tail_chunk_bytes = requested_logical - *bytes_before_tail;
  const std::optional<std::uint64_t> visit_count =
      checked_multiply(chunk_count, static_cast<std::uint64_t>(input.passes));
  const std::optional<std::uint64_t> revisit_count =
      checked_multiply(chunk_count, static_cast<std::uint64_t>(input.passes - 1U));
  if (!visit_count.has_value() || !revisit_count.has_value()) {
    return fail(PlanError::arithmetic_overflow);
  }

  WorkloadPlan plan;
  plan.logical_size_automatic = automatic;
  plan.requested_logical_bytes = requested_logical;
  plan.effective_logical_bytes = requested_logical;
  plan.requested_chunk_bytes = input.requested_chunk_bytes;
  plan.effective_chunk_bytes = *effective_chunk;
  plan.effective_alignment_bytes = *effective_alignment;
  plan.passes = input.passes;
  plan.window_slots = input.window_slots;
  plan.logical_element_count = requested_logical / word_bytes;
  plan.logical_chunk_count = chunk_count;
  plan.tail_chunk_bytes = tail_chunk_bytes;
  plan.tile_visit_count = *visit_count;
  plan.address_revisit_count = *revisit_count;
  plan.backing_bytes = requested_logical;
  plan.pinned_staging_bytes = *window_bytes;
  plan.resident_physical_bytes = *window_bytes;
  plan.host_headroom_bytes = host_headroom;
  plan.safe_logical_limit_bytes = safe_logical_limit;
  return {PlanError::none, plan};
}

std::optional<TileVisit> tile_visit_at(const std::uint64_t logical_chunk_count,
                                       const std::uint32_t passes,
                                       const std::uint64_t ordinal) noexcept {
  if (logical_chunk_count == 0 || passes == 0) {
    return std::nullopt;
  }
  const std::optional<std::uint64_t> total =
      checked_multiply(logical_chunk_count, static_cast<std::uint64_t>(passes));
  if (!total.has_value() || ordinal >= *total) {
    return std::nullopt;
  }

  const std::uint64_t pass = ordinal / logical_chunk_count;
  const std::uint64_t within_pass = ordinal % logical_chunk_count;
  const bool forward = (pass & 1ULL) == 0;
  const std::uint64_t tile_index = forward ? within_pass : logical_chunk_count - 1ULL - within_pass;
  return TileVisit{ordinal, static_cast<std::uint32_t>(pass), tile_index, forward};
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
  if (size > std::numeric_limits<std::uint64_t>::max() - count_) {
    return false;
  }
  if (size != 0 && size - 1ULL > std::numeric_limits<std::uint64_t>::max() - first_global_index) {
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

std::optional<SampleSummary> summarize_samples(const std::span<const double> samples) {
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

  SampleSummary summary;
  summary.count = static_cast<std::uint64_t>(sorted.size());
  summary.minimum = sorted.front();
  summary.median = interpolated_percentile(sorted, 0.5);
  summary.percentile_95 = interpolated_percentile(sorted, 0.95);
  summary.maximum = sorted.back();
  summary.total = total;
  summary.mean = total / static_cast<double>(sorted.size());
  return summary;
}

} // namespace xvram::vmm_poc
