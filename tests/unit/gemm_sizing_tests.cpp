#include "gemm_bench/sizing.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>

namespace {

constexpr std::uint64_t mib = 1024ULL * 1024ULL;
constexpr std::uint64_t gib = 1024ULL * mib;
int failures = 0;

void check(const bool condition, const char* expression, const int line) {
  if (!condition) {
    std::cerr << "line " << line << ": check failed: " << expression << '\n';
    ++failures;
  }
}

#define CHECK(expression) check(static_cast<bool>(expression), #expression, __LINE__)

void safe_host_limit_tests() {
  using xvram::gemm_bench::HostSizingInput;
  using xvram::gemm_bench::safe_host_logical_limit;

  const auto ordinary =
      safe_host_logical_limit(HostSizingInput{64ULL * gib, 40ULL * gib, 64ULL * mib, 4U});
  CHECK(ordinary == 23ULL * gib + 512ULL * mib);

  const auto minimum_headroom =
      safe_host_logical_limit(HostSizingInput{8ULL * gib, 5ULL * gib, 64ULL * mib, 4U});
  CHECK(minimum_headroom == 512ULL * mib);
  CHECK(!safe_host_logical_limit(
             HostSizingInput{8ULL * gib, 4ULL * gib + 512ULL * mib, 64ULL * mib, 4U})
             .has_value());
  CHECK(!safe_host_logical_limit(HostSizingInput{}).has_value());
  CHECK(!safe_host_logical_limit(HostSizingInput{std::numeric_limits<std::uint64_t>::max(),
                                                 std::numeric_limits<std::uint64_t>::max(),
                                                 std::numeric_limits<std::uint64_t>::max(), 8U})
             .has_value());
}

void auto_shape_tests() {
  using xvram::gemm_bench::choose_auto_gemm_shape;
  for (const std::uint64_t element_bytes : std::array<std::uint64_t, 4>{2, 4, 8, 16}) {
    const auto shape = choose_auto_gemm_shape(8ULL * gib, element_bytes, 24ULL * gib);
    CHECK(shape.has_value());
    if (shape.has_value()) {
      CHECK(shape->m == shape->n);
      CHECK(shape->m % 256ULL == 0);
      CHECK(shape->k == 256ULL);
      CHECK(shape->logical_bytes > 8ULL * gib);
      CHECK(shape->logical_bytes <= 12ULL * gib);
    }
  }

  const auto host_limited = choose_auto_gemm_shape(8ULL * gib, 4, 9ULL * gib);
  CHECK(host_limited.has_value());
  if (host_limited.has_value()) {
    CHECK(host_limited->logical_bytes > 8ULL * gib);
    CHECK(host_limited->logical_bytes <= 9ULL * gib);
  }
  CHECK(!choose_auto_gemm_shape(8ULL * gib, 4, 8ULL * gib).has_value());
  CHECK(!choose_auto_gemm_shape(std::numeric_limits<std::uint64_t>::max(), 4,
                                std::numeric_limits<std::uint64_t>::max())
             .has_value());
  CHECK(!choose_auto_gemm_shape(0, 4, 12ULL * gib).has_value());
}

} // namespace

int main() {
  safe_host_limit_tests();
  auto_shape_tests();
  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
    return 1;
  }
  std::cout << "all GEMM sizing tests passed\n";
  return 0;
}
