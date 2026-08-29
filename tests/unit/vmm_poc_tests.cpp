#include "vmm_poc/workload.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>

namespace {

using xvram::vmm_poc::PlanError;

int failures = 0;

void check(const bool condition, const char* expression, const int line) {
  if (!condition) {
    std::cerr << "line " << line << ": check failed: " << expression << '\n';
    ++failures;
  }
}

#define CHECK(expression) check(static_cast<bool>(expression), #expression, __LINE__)

[[nodiscard]] bool near(const double left, const double right) {
  return std::abs(left - right) < 1.0e-12;
}

void arithmetic_tests() {
  using namespace xvram::vmm_poc;
  constexpr std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();

  CHECK(checked_add(7, 9) == 16);
  CHECK(!checked_add(maximum, 1).has_value());
  CHECK(checked_multiply(7, 9) == 63);
  CHECK(checked_multiply(0, maximum) == 0);
  CHECK(!checked_multiply(maximum, 2).has_value());

  CHECK(checked_align_up(0, 8) == 0);
  CHECK(checked_align_up(1, 8) == 8);
  CHECK(checked_align_up(16, 8) == 16);
  CHECK(!checked_align_up(1, 0).has_value());
  CHECK(!checked_align_up(maximum - 3, 8).has_value());

  CHECK(checked_least_common_multiple(6, 8) == 24);
  CHECK(checked_least_common_multiple(0, 8) == 0);
  CHECK(!checked_least_common_multiple(maximum, 2).has_value());
}

[[nodiscard]] xvram::vmm_poc::PlanningInput valid_planning_input() {
  using namespace xvram::vmm_poc;
  PlanningInput input;
  input.requested_logical_bytes = 12ULL * 1024ULL * 1024ULL * 1024ULL;
  input.requested_chunk_bytes = 64ULL * 1024ULL * 1024ULL;
  input.passes = 2;
  input.window_slots = 1;
  input.total_device_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
  input.free_device_bytes = 7ULL * 1024ULL * 1024ULL * 1024ULL;
  input.available_device_budget_bytes = 6ULL * 1024ULL * 1024ULL * 1024ULL;
  input.minimum_granularity_bytes = 2ULL * 1024ULL * 1024ULL;
  input.recommended_granularity_bytes = 2ULL * 1024ULL * 1024ULL;
  input.physical_host_bytes = 64ULL * 1024ULL * 1024ULL * 1024ULL;
  input.available_host_bytes = 40ULL * 1024ULL * 1024ULL * 1024ULL;
  return input;
}

void planning_tests() {
  using namespace xvram::vmm_poc;
  constexpr std::uint64_t gib = 1024ULL * 1024ULL * 1024ULL;
  constexpr std::uint64_t mib = 1024ULL * 1024ULL;

  PlanningInput input = valid_planning_input();
  const PlanningResult result = make_workload_plan(input);
  CHECK(result);
  CHECK(result.plan->effective_logical_bytes == 12ULL * gib);
  CHECK(result.plan->effective_chunk_bytes == 64ULL * mib);
  CHECK(result.plan->effective_alignment_bytes == 2ULL * mib);
  CHECK(result.plan->logical_element_count == 3'221'225'472ULL);
  CHECK(result.plan->logical_chunk_count == 192);
  CHECK(result.plan->tail_chunk_bytes == 64ULL * mib);
  CHECK(result.plan->tile_visit_count == 384);
  CHECK(result.plan->address_revisit_count == 192);
  CHECK(result.plan->backing_bytes == 12ULL * gib);
  CHECK(result.plan->pinned_staging_bytes == 64ULL * mib);
  CHECK(result.plan->resident_physical_bytes == 64ULL * mib);
  CHECK(result.plan->host_headroom_bytes == 16ULL * gib);
  CHECK(!result.plan->logical_size_automatic);

  input.requested_logical_bytes.reset();
  const PlanningResult automatic = make_workload_plan(input);
  CHECK(automatic);
  CHECK(automatic.plan->logical_size_automatic);
  CHECK(automatic.plan->effective_logical_bytes == 12ULL * gib);
  CHECK(automatic.plan->logical_chunk_count == 192);

  input.available_host_bytes = 26ULL * gib;
  const PlanningResult host_limited_auto = make_workload_plan(input);
  CHECK(host_limited_auto);
  CHECK(host_limited_auto.plan->effective_logical_bytes == 9ULL * gib + 704ULL * mib);

  input = valid_planning_input();
  input.requested_logical_bytes = 12ULL * gib + 4ULL;
  const PlanningResult tail = make_workload_plan(input);
  CHECK(tail);
  CHECK(tail.plan->effective_logical_bytes == 12ULL * gib + 4ULL);
  CHECK(tail.plan->logical_chunk_count == 193);
  CHECK(tail.plan->tail_chunk_bytes == 4ULL);

  input = valid_planning_input();
  input.window_slots = 3;
  const PlanningResult window = make_workload_plan(input);
  CHECK(window);
  CHECK(window.plan->pinned_staging_bytes == 192ULL * mib);
  CHECK(window.plan->resident_physical_bytes == 192ULL * mib);

  input = valid_planning_input();
  input.requested_chunk_bytes = 25;
  input.minimum_granularity_bytes = 6;
  input.recommended_granularity_bytes = 8;
  const PlanningResult common_alignment = make_workload_plan(input);
  CHECK(common_alignment);
  CHECK(common_alignment.plan->effective_alignment_bytes == 24);
  CHECK(common_alignment.plan->effective_chunk_bytes == 48);

  input = valid_planning_input();
  input.requested_logical_bytes = 4ULL * gib;
  CHECK(make_workload_plan(input).error == PlanError::logical_size_not_oversubscribed);

  input = valid_planning_input();
  input.available_host_bytes = 1ULL * gib;
  CHECK(make_workload_plan(input).error == PlanError::insufficient_host_memory);

  input = valid_planning_input();
  input.available_host_bytes = 20ULL * gib;
  CHECK(make_workload_plan(input).error == PlanError::insufficient_host_memory);

  input = valid_planning_input();
  input.available_device_budget_bytes = 128ULL * mib;
  CHECK(make_workload_plan(input).error == PlanError::insufficient_device_budget);

  input = valid_planning_input();
  input.requested_logical_bytes.reset();
  input.total_device_bytes = std::numeric_limits<std::uint64_t>::max() - 1ULL;
  CHECK(make_workload_plan(input).error == PlanError::arithmetic_overflow);

  input = valid_planning_input();
  input.requested_logical_bytes = 12ULL * gib + 1ULL;
  CHECK(make_workload_plan(input).error == PlanError::invalid_configuration);

  input = valid_planning_input();
  input.window_slots = 9;
  CHECK(make_workload_plan(input).error == PlanError::invalid_configuration);

  CHECK(plan_error_name(PlanError::insufficient_host_memory) == "insufficient_host_memory");
}

void schedule_tests() {
  using namespace xvram::vmm_poc;
  constexpr std::array<std::uint64_t, 9> expected{0, 1, 2, 2, 1, 0, 0, 1, 2};
  for (std::uint64_t ordinal = 0; ordinal < expected.size(); ++ordinal) {
    const std::optional<TileVisit> visit = tile_visit_at(3, 3, ordinal);
    CHECK(visit.has_value());
    CHECK(visit->ordinal == ordinal);
    CHECK(visit->pass_index == ordinal / 3ULL);
    CHECK(visit->tile_index == expected[static_cast<std::size_t>(ordinal)]);
    CHECK(visit->forward == (visit->pass_index % 2U == 0U));
  }
  CHECK(!tile_visit_at(0, 2, 0).has_value());
  CHECK(!tile_visit_at(3, 0, 0).has_value());
  CHECK(!tile_visit_at(3, 3, 9).has_value());
}

void workload_pattern_tests() {
  using namespace xvram::vmm_poc;
  constexpr std::uint64_t seed = 0xC001D00D5EED1234ULL;

  CHECK(initial_word(0, seed) == 1'959'270'207U);
  CHECK(initial_word(5, seed) == 661'088'027U);
  CHECK(initial_word((1ULL << 32U) + 5ULL, seed) == 3'792'126'075U);

  const std::uint32_t initial = initial_word(123'456'789ULL, seed);
  const std::uint32_t pass_zero = transform_word(initial, 123'456'789ULL, 0, seed);
  const std::uint32_t pass_one = transform_word(pass_zero, 123'456'789ULL, 1, seed);
  CHECK(initial == 1'312'001'775U);
  CHECK(pass_zero == 615'769'163U);
  CHECK(pass_one == 3'516'258'272U);
  CHECK(expected_word(123'456'789ULL, 0, seed) == initial);
  CHECK(expected_word(123'456'789ULL, 1, seed) == pass_zero);
  CHECK(expected_word(123'456'789ULL, 2, seed) == pass_one);
  CHECK(pass_zero != pass_one);
}

void digest_tests() {
  using namespace xvram::vmm_poc;
  constexpr std::array<std::uint32_t, 4> words{0, 1, 2, 3};

  Digest128Accumulator whole;
  CHECK(whole.update_words(words, 10));
  CHECK(whole.count() == words.size());
  CHECK(whole.value().low == 0xB731F8279A618194ULL);
  CHECK(whole.value().high == 0x9AE5FEB0B6F0F830ULL);
  CHECK(format_digest128(whole.value()) == "9ae5feb0b6f0f830b731f8279a618194");

  Digest128Accumulator split;
  CHECK(split.update_words(std::span<const std::uint32_t>(words).first(2), 10));
  CHECK(split.update_words(std::span<const std::uint32_t>(words).subspan(2), 12));
  CHECK(split.value() == whole.value());

  Digest128Accumulator reordered;
  constexpr std::array<std::uint32_t, 4> reverse{3, 2, 1, 0};
  CHECK(reordered.update_words(reverse, 10));
  CHECK(reordered.value() != whole.value());

  Digest128Accumulator shifted;
  CHECK(shifted.update_words(words, 11));
  CHECK(shifted.value() != whole.value());

  const Digest128Value before_rejected_update = whole.value();
  CHECK(!whole.update_words(words, std::numeric_limits<std::uint64_t>::max() - 2ULL));
  CHECK(whole.value() == before_rejected_update);
}

void sample_summary_tests() {
  using namespace xvram::vmm_poc;
  CHECK(!summarize_samples(std::span<const double>{}).has_value());

  constexpr std::array<double, 4> samples{4.0, 1.0, 3.0, 2.0};
  const std::optional<SampleSummary> summary = summarize_samples(samples);
  CHECK(summary.has_value());
  CHECK(summary->count == 4);
  CHECK(near(summary->minimum, 1.0));
  CHECK(near(summary->median, 2.5));
  CHECK(near(summary->percentile_95, 3.85));
  CHECK(near(summary->maximum, 4.0));
  CHECK(near(summary->total, 10.0));
  CHECK(near(summary->mean, 2.5));

  constexpr std::array<double, 1> single{5.0};
  const std::optional<SampleSummary> single_summary = summarize_samples(single);
  CHECK(single_summary.has_value());
  CHECK(near(single_summary->median, 5.0));
  CHECK(near(single_summary->percentile_95, 5.0));

  const std::array<double, 1> invalid{std::numeric_limits<double>::quiet_NaN()};
  CHECK(!summarize_samples(invalid).has_value());
  constexpr std::array<double, 1> negative{-1.0};
  CHECK(!summarize_samples(negative).has_value());
}

} // namespace

int main() {
  arithmetic_tests();
  planning_tests();
  schedule_tests();
  workload_pattern_tests();
  digest_tests();
  sample_summary_tests();
  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
    return 1;
  }
  std::cout << "all VMM proof utility tests passed\n";
  return 0;
}
