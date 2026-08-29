#include "xvram/base/size_parser.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <iomanip>
#include <limits>
#include <sstream>

namespace xvram {
namespace {

[[nodiscard]] std::string lowercase(std::string_view input) {
  std::string result(input);
  std::transform(result.begin(), result.end(), result.begin(),
                 [](const unsigned char value) { return static_cast<char>(std::tolower(value)); });
  return result;
}

} // namespace

ParsedSize parse_size(const std::string_view text) {
  if (text.empty()) {
    return {std::nullopt, "size must not be empty"};
  }

  std::size_t digit_count = 0;
  while (digit_count < text.size() && text[digit_count] >= '0' && text[digit_count] <= '9') {
    ++digit_count;
  }
  if (digit_count == 0) {
    return {std::nullopt, "size must start with a non-negative integer"};
  }

  std::uint64_t number = 0;
  const auto parse_result = std::from_chars(text.data(), text.data() + digit_count, number, 10);
  if (parse_result.ec != std::errc{}) {
    return {std::nullopt, "size integer is out of range"};
  }

  const std::string suffix = lowercase(text.substr(digit_count));
  std::uint64_t multiplier = 0;
  if (suffix.empty() || suffix == "b") {
    multiplier = 1;
  } else if (suffix == "kib" || suffix == "ki" || suffix == "k") {
    multiplier = 1024ULL;
  } else if (suffix == "mib" || suffix == "mi" || suffix == "m") {
    multiplier = 1024ULL * 1024ULL;
  } else if (suffix == "gib" || suffix == "gi" || suffix == "g") {
    multiplier = 1024ULL * 1024ULL * 1024ULL;
  } else {
    return {std::nullopt, "unknown size suffix; use B, KiB, MiB, or GiB"};
  }

  if (number > std::numeric_limits<std::uint64_t>::max() / multiplier) {
    return {std::nullopt, "size is out of range"};
  }
  return {number * multiplier, {}};
}

std::string format_bytes(const std::uint64_t bytes) {
  constexpr std::array<const char*, 4> units = {"B", "KiB", "MiB", "GiB"};
  double value = static_cast<double>(bytes);
  std::size_t unit = 0;
  while (value >= 1024.0 && unit + 1 < units.size()) {
    value /= 1024.0;
    ++unit;
  }

  std::ostringstream output;
  if (unit == 0) {
    output << bytes << ' ' << units[unit];
  } else {
    output << std::fixed << std::setprecision(value >= 100.0 ? 1 : 2) << value << ' '
           << units[unit];
  }
  return output.str();
}

} // namespace xvram
