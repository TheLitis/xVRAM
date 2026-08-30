#include "residency/workload.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <span>

namespace {

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
  using namespace xvram::residency;
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

void ptx_vector_tests() {
  using namespace xvram::residency;
  constexpr std::uint64_t seed = 0x585652414D503032ULL;

  CHECK(initial_word(0, seed) == 0x7F887FF6U);
  CHECK(initial_word(5, seed) == 0xBA52E10BU);
  CHECK(initial_word((1ULL << 32U) + 5ULL, seed) == 0x987FEB92U);
  CHECK(initial_word(123'456'789ULL, seed) == 0xD1A3ECBAU);

  CHECK(operation_mix(0, 0, seed) == 0x897F9BF6U);
  CHECK(operation_mix(5, 7, seed) == 0x494D24F6U);
  CHECK(operation_mix((1ULL << 32U) + 5ULL, 0, seed) == 0x1B813100U);
  CHECK(operation_mix(123'456'789ULL, std::numeric_limits<std::uint32_t>::max(), seed) ==
        0xF10414EEU);

  constexpr std::uint32_t input = 0x12345678U;
  CHECK(transform_word(input, 123'456'789ULL, 7, seed, AccessMode::read) == input);
  CHECK(transform_word(input, 123'456'789ULL, 7, seed, AccessMode::read_write) == 0x23070936U);
  CHECK(transform_word(input, 123'456'789ULL, 7, seed, AccessMode::write_only) == 0x31EA5974U);
  CHECK(transform_word(0, 123'456'789ULL, 7, seed, AccessMode::write_only) ==
        transform_word(std::numeric_limits<std::uint32_t>::max(), 123'456'789ULL, 7, seed,
                       AccessMode::write_only));
}

void verification_token_tests() {
  using namespace xvram::residency;
  constexpr std::uint64_t seed = 0x585652414D503032ULL;
  constexpr std::array<std::uint32_t, 5> input{0x1B992391U, 0xE9300271U, 0xD8D5C0C0U, 0x38228A16U,
                                               0x5F51CE73U};
  constexpr std::array<std::uint32_t, 5> read_write_output{0xF946E459U, 0x65F052B4U, 0x421950BDU,
                                                           0xE4B61092U, 0x4B61EF5FU};
  constexpr std::array<std::uint32_t, 5> write_only_output{0x7066FC50U, 0x9D1543FDU, 0xDCC5CE72U,
                                                           0x4EA51905U, 0x784124F1U};

  CHECK(expected_verification_token(input, 100, 7, seed, AccessMode::read) == 0x51DD9D66U);
  CHECK(expected_verification_token(input, 100, 7, seed, AccessMode::read_write) == 0x6DAA21BEU);
  CHECK(expected_verification_token(input, 100, 7, seed, AccessMode::write_only) == 0x1B807408U);
  CHECK(verification_token_for_output(input, 100, 7, seed) == 0x51DD9D66U);
  CHECK(verification_token_for_output(read_write_output, 100, 7, seed) == 0x6DAA21BEU);
  CHECK(verification_token_for_output(write_only_output, 100, 7, seed) == 0x1B807408U);

  const std::span<const std::uint32_t> tail = std::span<const std::uint32_t>(input).subspan(3);
  CHECK(expected_verification_token(tail, 103, 7, seed, AccessMode::read_write) == 0x410F758AU);
  CHECK(expected_verification_token(tail, 103, 7, seed, AccessMode::write_only) == 0xD83CB7B3U);

  const std::span<const std::uint32_t> empty;
  CHECK(expected_verification_token(empty, std::numeric_limits<std::uint64_t>::max(), 7, seed,
                                    AccessMode::read) == 0U);
  constexpr std::array<std::uint32_t, 2> two_words{1, 2};
  CHECK(!expected_verification_token(two_words, std::numeric_limits<std::uint64_t>::max(), 7, seed,
                                     AccessMode::read)
             .has_value());
  CHECK(
      !verification_token_for_output(two_words, std::numeric_limits<std::uint64_t>::max(), 7, seed)
           .has_value());
}

void digest_tests() {
  using namespace xvram::residency;
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

void timing_tests() {
  using namespace xvram::residency;
  CHECK(!summarize_timings(std::span<const double>{}).has_value());

  constexpr std::array<double, 4> samples{4.0, 1.0, 3.0, 2.0};
  const std::optional<PercentileSummary> summary = summarize_timings(samples);
  CHECK(summary.has_value());
  CHECK(summary->count == 4);
  CHECK(near(summary->minimum, 1.0));
  CHECK(near(summary->median, 2.5));
  CHECK(near(summary->percentile_95, 3.85));
  CHECK(near(summary->maximum, 4.0));
  CHECK(near(summary->total, 10.0));
  CHECK(near(summary->mean, 2.5));

  constexpr std::array<double, 1> single{5.0};
  const std::optional<PercentileSummary> single_summary = summarize_timings(single);
  CHECK(single_summary.has_value());
  CHECK(near(single_summary->median, 5.0));
  CHECK(near(single_summary->percentile_95, 5.0));

  const std::array<double, 1> nan{std::numeric_limits<double>::quiet_NaN()};
  CHECK(!summarize_timings(nan).has_value());
  constexpr std::array<double, 1> negative{-1.0};
  CHECK(!summarize_timings(negative).has_value());
  const std::array<double, 2> overflowing{std::numeric_limits<double>::max(),
                                          std::numeric_limits<double>::max()};
  CHECK(!summarize_timings(overflowing).has_value());
}

} // namespace

int main() {
  arithmetic_tests();
  ptx_vector_tests();
  verification_token_tests();
  digest_tests();
  timing_tests();
  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
    return 1;
  }
  std::cout << "all residency workload tests passed\n";
  return 0;
}
