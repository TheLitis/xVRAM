#include "compat_bench/pattern.hpp"
#include <array>
#include <bit>
#include <iomanip>
#include <limits>
#include <sstream>

namespace xvram::compat_bench {
namespace {
std::optional<std::uint64_t> mul(const std::uint64_t a, const std::uint64_t b) noexcept {
  if (a != 0 && b > UINT64_MAX / a)
    return std::nullopt;
  return a * b;
}
std::optional<std::uint64_t> sum(const std::uint64_t a, const std::uint64_t b) noexcept {
  if (b > UINT64_MAX - a)
    return std::nullopt;
  return a + b;
}
} // namespace
std::optional<Matrix> matrix_storage(const Shape& s, const Operand op) noexcept {
  if (s.m == 0 || s.n == 0 || s.k == 0 || s.m > INT32_MAX || s.n > INT32_MAX || s.k > INT32_MAX)
    return std::nullopt;
  Matrix matrix;
  matrix.offset = s.offset;
  if (op == Operand::a) {
    matrix.rows = s.transpose_a ? s.k : s.m;
    matrix.columns = s.transpose_a ? s.m : s.k;
  } else if (op == Operand::b) {
    matrix.rows = s.transpose_b ? s.n : s.k;
    matrix.columns = s.transpose_b ? s.k : s.n;
  } else {
    matrix.rows = s.m;
    matrix.columns = s.n;
  }
  const auto ld = sum(matrix.rows, s.padding);
  if (!ld || *ld > INT32_MAX)
    return std::nullopt;
  matrix.ld = *ld;
  const auto storage = mul(matrix.ld, matrix.columns);
  const auto prefix = storage ? sum(*storage, matrix.offset) : std::nullopt;
  const auto total = prefix ? sum(*prefix, s.offset) : std::nullopt;
  if (!total || !mul(*total, sizeof(float)))
    return std::nullopt;
  matrix.elements = *total;
  return matrix;
}
std::optional<std::uint64_t> operand_bytes(const Shape& s) noexcept {
  const auto a = mul(s.m, s.k), b = mul(s.k, s.n), c = mul(s.m, s.n);
  if (!a || !b || !c)
    return std::nullopt;
  const auto ab = sum(*a, *b);
  const auto abc = ab ? sum(*ab, *c) : std::nullopt;
  return abc ? mul(*abc, sizeof(float)) : std::nullopt;
}
float pattern_a(const std::uint64_t row, const std::uint64_t k, const std::uint64_t seed) noexcept {
  constexpr std::array<float, 4> pattern{1.0F, -1.0F, 0.5F, 0.5F};
  return static_cast<float>(1U + (row + seed % 3U) % 3U) * pattern[k % 4U];
}
float pattern_b(const std::uint64_t k, const std::uint64_t column,
                const std::uint64_t seed) noexcept {
  constexpr std::array<float, 4> pattern{1.0F, 1.0F, 0.5F, 0.5F};
  return static_cast<float>(1U + (column + (seed >> 8U) % 3U) % 3U) * pattern[k % 4U];
}
float pattern_c(const std::uint64_t row, const std::uint64_t column) noexcept {
  return static_cast<float>(1U + (row + column) % 7U) * 0.25F;
}
double reference_dot(const std::uint64_t row, const std::uint64_t column, const std::uint64_t k,
                     const std::uint64_t seed, const bool full) noexcept {
  double total = 0.0;
  if (full) {
    for (std::uint64_t p = 0; p < k; ++p)
      total += static_cast<double>(pattern_a(row, p, seed)) * pattern_b(p, column, seed);
    return total;
  }
  const double scale = static_cast<double>(pattern_a(row, 0, seed)) * pattern_b(0, column, seed);
  total = static_cast<double>(k / 4U) * 0.5 * scale;
  for (std::uint64_t p = 0; p < k % 4U; ++p)
    total += static_cast<double>(pattern_a(row, p, seed)) * pattern_b(p, column, seed);
  return total;
}
double reference_result(const Shape& s, const std::uint64_t row, const std::uint64_t column,
                        const std::uint64_t seed, const float alpha, const float beta,
                        const std::uint32_t passes, const bool full) noexcept {
  const auto dot = reference_dot(row, column, s.k, seed, full);
  double result = pattern_c(row, column);
  for (std::uint32_t p = 0; p < passes; ++p)
    result = static_cast<double>(alpha) * dot + static_cast<double>(beta) * result;
  return result;
}
void fill_pattern(const std::span<float> output, const std::uint64_t start, const Shape& s,
                  const Operand op, const std::uint64_t seed) {
  const auto matrix = matrix_storage(s, op);
  if (!matrix)
    return;
  for (std::size_t i = 0; i < output.size(); ++i) {
    const auto absolute = start + i;
    float value = padding_value;
    if (absolute >= matrix->offset) {
      const auto index = absolute - matrix->offset;
      const auto row = index % matrix->ld, column = index / matrix->ld;
      if (row < matrix->rows && column < matrix->columns) {
        if (op == Operand::a)
          value = pattern_a(s.transpose_a ? column : row, s.transpose_a ? row : column, seed);
        else if (op == Operand::b)
          value = pattern_b(s.transpose_b ? column : row, s.transpose_b ? row : column, seed);
        else
          value = pattern_c(row, column);
      }
    }
    output[i] = value;
  }
}
void Digest::add(const float value) noexcept {
  const auto bits = std::bit_cast<std::uint32_t>(value);
  for (unsigned shift = 0; shift < 32; shift += 8) {
    const auto byte = (bits >> shift) & 255U;
    a = (a ^ byte) * 1099511628211ULL;
    b = (b ^ (byte + 0x9eU)) * 0x100000001b3ULL;
    b ^= b >> 29U;
  }
}
std::string Digest::string() const {
  std::ostringstream stream;
  stream << std::hex << std::setfill('0') << std::setw(16) << b << std::setw(16) << a;
  return stream.str();
}
} // namespace xvram::compat_bench
