#include "gemm_bench/pattern.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <limits>
#include <utility>

namespace xvram::gemm_bench {
namespace {

[[nodiscard]] std::optional<std::uint64_t> checked_add(const std::uint64_t left,
                                                       const std::uint64_t right) noexcept {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    return std::nullopt;
  }
  return left + right;
}

[[nodiscard]] std::optional<std::uint64_t> checked_multiply(const std::uint64_t left,
                                                            const std::uint64_t right) noexcept {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
    return std::nullopt;
  }
  return left * right;
}

[[nodiscard]] std::uint16_t float_to_half(const float input) noexcept {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(input);
  const std::uint32_t sign = (bits >> 16U) & 0x8000U;
  const std::uint32_t exponent = (bits >> 23U) & 0xFFU;
  const std::uint32_t mantissa = bits & 0x007FFFFFU;
  if (exponent == 0xFFU) {
    return static_cast<std::uint16_t>(sign | 0x7C00U | (mantissa == 0 ? 0U : 0x0200U));
  }
  const int half_exponent = static_cast<int>(exponent) - 127 + 15;
  if (half_exponent >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7C00U);
  }
  if (half_exponent <= 0) {
    if (half_exponent < -10) {
      return static_cast<std::uint16_t>(sign);
    }
    const std::uint32_t normalized = mantissa | 0x00800000U;
    const unsigned shift = static_cast<unsigned>(14 - half_exponent);
    std::uint32_t rounded = normalized >> shift;
    const std::uint32_t remainder = normalized & ((UINT32_C(1) << shift) - 1U);
    const std::uint32_t halfway = UINT32_C(1) << (shift - 1U);
    if (remainder > halfway || (remainder == halfway && (rounded & 1U) != 0U)) {
      ++rounded;
    }
    return static_cast<std::uint16_t>(sign | rounded);
  }
  std::uint32_t rounded_mantissa = mantissa >> 13U;
  const std::uint32_t remainder = mantissa & 0x1FFFU;
  if (remainder > 0x1000U || (remainder == 0x1000U && (rounded_mantissa & 1U) != 0U)) {
    ++rounded_mantissa;
    if (rounded_mantissa == 0x400U) {
      rounded_mantissa = 0;
      if (half_exponent + 1 >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7C00U);
      }
      return static_cast<std::uint16_t>(sign |
                                        (static_cast<std::uint32_t>(half_exponent + 1) << 10U));
    }
  }
  return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(half_exponent) << 10U) |
                                    rounded_mantissa);
}

[[nodiscard]] float half_to_float(const std::uint16_t input) noexcept {
  const std::uint32_t sign = static_cast<std::uint32_t>(input & 0x8000U) << 16U;
  std::uint32_t exponent = (input >> 10U) & 0x1FU;
  std::uint32_t mantissa = input & 0x03FFU;
  std::uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      exponent = 113U;
      while ((mantissa & 0x0400U) == 0) {
        mantissa <<= 1U;
        --exponent;
      }
      bits = sign | (exponent << 23U) | ((mantissa & 0x03FFU) << 13U);
    }
  } else if (exponent == 31U) {
    bits = sign | 0x7F800000U | (mantissa << 13U);
  } else {
    bits = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
  }
  return std::bit_cast<float>(bits);
}

[[nodiscard]] std::uint16_t float_to_bf16(const float input) noexcept {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(input);
  const std::uint32_t rounding_bias = 0x7FFFU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>((bits + rounding_bias) >> 16U);
}

[[nodiscard]] bool valid_matrix(const PatternMatrix& matrix) noexcept {
  if (matrix.rows == 0 || matrix.columns == 0 || pattern_element_bytes(matrix.data_type) == 0) {
    return false;
  }
  return matrix.layout == RequestedLayout::row_major ? matrix.leading_dimension >= matrix.columns
                                                     : matrix.leading_dimension >= matrix.rows;
}

[[nodiscard]] std::optional<std::uint64_t> storage_elements(const PatternMatrix& matrix) noexcept {
  if (!valid_matrix(matrix)) {
    return std::nullopt;
  }
  const std::uint64_t major =
      matrix.layout == RequestedLayout::row_major ? matrix.rows : matrix.columns;
  return checked_multiply(major, matrix.leading_dimension);
}

struct PhysicalCoordinate {
  std::uint64_t row = 0;
  std::uint64_t column = 0;
  bool padding = false;
};

[[nodiscard]] std::optional<PhysicalCoordinate>
physical_coordinate(const PatternMatrix& matrix, const std::uint64_t index) noexcept {
  const std::optional<std::uint64_t> elements = storage_elements(matrix);
  if (!elements.has_value() || index >= *elements) {
    return std::nullopt;
  }
  if (matrix.layout == RequestedLayout::row_major) {
    const std::uint64_t row = index / matrix.leading_dimension;
    const std::uint64_t column = index % matrix.leading_dimension;
    return PhysicalCoordinate{row, column, column >= matrix.columns};
  }
  const std::uint64_t column = index / matrix.leading_dimension;
  const std::uint64_t row = index % matrix.leading_dimension;
  return PhysicalCoordinate{row, column, row >= matrix.rows};
}

[[nodiscard]] bool matrix_matches_problem(const PatternMatrix& matrix, const PatternOperand operand,
                                          const PatternProblem& problem) noexcept {
  if (problem.m == 0 || problem.n == 0 || problem.k == 0 || !valid_matrix(matrix)) {
    return false;
  }
  switch (operand) {
  case PatternOperand::a:
    return matrix.rows ==
               (problem.operation_a == RequestedOperation::none ? problem.m : problem.k) &&
           matrix.columns ==
               (problem.operation_a == RequestedOperation::none ? problem.k : problem.m);
  case PatternOperand::b:
    return matrix.rows ==
               (problem.operation_b == RequestedOperation::none ? problem.k : problem.n) &&
           matrix.columns ==
               (problem.operation_b == RequestedOperation::none ? problem.n : problem.k);
  case PatternOperand::expected_c:
    return matrix.rows == problem.m && matrix.columns == problem.n;
  }
  return false;
}

[[nodiscard]] std::optional<std::uint64_t>
value_for_coordinate(const PhysicalCoordinate& coordinate, const PatternOperand operand,
                     const PatternProblem& problem) noexcept {
  if (coordinate.padding) {
    return 0;
  }
  switch (operand) {
  case PatternOperand::a: {
    const std::uint64_t logical_row =
        problem.operation_a == RequestedOperation::none ? coordinate.row : coordinate.column;
    const std::uint64_t k_index =
        problem.operation_a == RequestedOperation::none ? coordinate.column : coordinate.row;
    return pattern_a_value(logical_row, k_index);
  }
  case PatternOperand::b: {
    const std::uint64_t k_index =
        problem.operation_b == RequestedOperation::none ? coordinate.row : coordinate.column;
    const std::uint64_t logical_column =
        problem.operation_b == RequestedOperation::none ? coordinate.column : coordinate.row;
    return pattern_b_value(k_index, logical_column);
  }
  case PatternOperand::expected_c:
    return pattern_expected_value(coordinate.row, coordinate.column, problem.k);
  }
  return std::nullopt;
}

} // namespace

std::uint64_t pattern_element_bytes(const RequestedDataType type) noexcept {
  switch (type) {
  case RequestedDataType::fp16:
  case RequestedDataType::bf16:
    return 2;
  case RequestedDataType::fp32:
    return 4;
  case RequestedDataType::fp64:
    return 8;
  case RequestedDataType::suite:
    return 0;
  }
  return 0;
}

std::uint64_t pattern_a_value(const std::uint64_t row, const std::uint64_t k_index) noexcept {
  return 1ULL + row % 3ULL + k_index % 2ULL;
}

std::uint64_t pattern_b_value(const std::uint64_t k_index, const std::uint64_t column) noexcept {
  return 1ULL + column % 5ULL + k_index % 3ULL;
}

PatternSums pattern_closed_form_sums(const std::uint64_t k) noexcept {
  static constexpr std::array<std::uint64_t, 3> q_remainder{0, 0, 1};
  static constexpr std::array<std::uint64_t, 6> pq_remainder{0, 0, 1, 1, 1, 1};
  return PatternSums{k / 2ULL, 3ULL * (k / 3ULL) + q_remainder[k % 3ULL],
                     3ULL * (k / 6ULL) + pq_remainder[k % 6ULL]};
}

std::optional<std::uint64_t> pattern_expected_value(const std::uint64_t row,
                                                    const std::uint64_t column,
                                                    const std::uint64_t k) noexcept {
  const std::uint64_t a = 1ULL + row % 3ULL;
  const std::uint64_t b = 1ULL + column % 5ULL;
  const PatternSums sums = pattern_closed_form_sums(k);
  const auto ab = checked_multiply(a, b);
  const auto term_ab = ab.has_value() ? checked_multiply(k, *ab) : std::nullopt;
  const auto term_aq = checked_multiply(a, sums.sum_q);
  const auto term_bp = checked_multiply(b, sums.sum_p);
  const auto first =
      term_ab.has_value() && term_aq.has_value() ? checked_add(*term_ab, *term_aq) : std::nullopt;
  const auto second =
      first.has_value() && term_bp.has_value() ? checked_add(*first, *term_bp) : std::nullopt;
  return second.has_value() ? checked_add(*second, sums.sum_pq) : std::nullopt;
}

std::optional<std::uint64_t> pattern_physical_index(const PatternMatrix& matrix,
                                                    const std::uint64_t row,
                                                    const std::uint64_t column) noexcept {
  if (!valid_matrix(matrix) || row >= matrix.rows || column >= matrix.columns) {
    return std::nullopt;
  }
  const auto major = matrix.layout == RequestedLayout::row_major
                         ? checked_multiply(row, matrix.leading_dimension)
                         : checked_multiply(column, matrix.leading_dimension);
  const std::uint64_t minor = matrix.layout == RequestedLayout::row_major ? column : row;
  return major.has_value() ? checked_add(*major, minor) : std::nullopt;
}

std::optional<std::uint64_t> pattern_storage_bytes(const PatternMatrix& matrix) noexcept {
  const std::optional<std::uint64_t> elements = storage_elements(matrix);
  return elements.has_value() ? checked_multiply(*elements, pattern_element_bytes(matrix.data_type))
                              : std::nullopt;
}

bool encode_pattern_value(const std::uint64_t value, const RequestedDataType type,
                          const std::span<std::byte> output) noexcept {
  const std::uint64_t bytes = pattern_element_bytes(type);
  if (bytes == 0 || bytes > static_cast<std::uint64_t>(output.size())) {
    return false;
  }
  if (type == RequestedDataType::fp16) {
    const std::uint16_t encoded = float_to_half(static_cast<float>(value));
    std::memcpy(output.data(), &encoded, sizeof(encoded));
  } else if (type == RequestedDataType::bf16) {
    const std::uint16_t encoded = float_to_bf16(static_cast<float>(value));
    std::memcpy(output.data(), &encoded, sizeof(encoded));
  } else if (type == RequestedDataType::fp32) {
    const float encoded = static_cast<float>(value);
    std::memcpy(output.data(), &encoded, sizeof(encoded));
  } else if (type == RequestedDataType::fp64) {
    const double encoded = static_cast<double>(value);
    std::memcpy(output.data(), &encoded, sizeof(encoded));
  } else {
    return false;
  }
  return true;
}

std::optional<double> decode_pattern_value(const std::span<const std::byte> input,
                                           const RequestedDataType type) noexcept {
  const std::uint64_t bytes = pattern_element_bytes(type);
  if (bytes == 0 || bytes > static_cast<std::uint64_t>(input.size())) {
    return std::nullopt;
  }
  if (type == RequestedDataType::fp16) {
    std::uint16_t value = 0;
    std::memcpy(&value, input.data(), sizeof(value));
    return static_cast<double>(half_to_float(value));
  }
  if (type == RequestedDataType::bf16) {
    std::uint16_t value = 0;
    std::memcpy(&value, input.data(), sizeof(value));
    return static_cast<double>(std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U));
  }
  if (type == RequestedDataType::fp32) {
    float value = 0.0F;
    std::memcpy(&value, input.data(), sizeof(value));
    return static_cast<double>(value);
  }
  if (type == RequestedDataType::fp64) {
    double value = 0.0;
    std::memcpy(&value, input.data(), sizeof(value));
    return value;
  }
  return std::nullopt;
}

bool fill_pattern_bytes(const std::span<std::byte> output, const std::uint64_t absolute_byte_offset,
                        const PatternMatrix& matrix, const PatternOperand operand,
                        const PatternProblem& problem) noexcept {
  if (!matrix_matches_problem(matrix, operand, problem)) {
    return false;
  }
  const std::optional<std::uint64_t> total_bytes = pattern_storage_bytes(matrix);
  const std::uint64_t output_bytes = static_cast<std::uint64_t>(output.size());
  if (!total_bytes.has_value() || absolute_byte_offset > *total_bytes ||
      output_bytes > *total_bytes - absolute_byte_offset) {
    return false;
  }
  if (output.empty()) {
    return true;
  }

  const std::uint64_t item_bytes = pattern_element_bytes(matrix.data_type);
  const std::uint64_t end_offset = absolute_byte_offset + output_bytes;
  const std::uint64_t first_element = absolute_byte_offset / item_bytes;
  const std::uint64_t last_element = (end_offset - 1ULL) / item_bytes;
  std::array<std::byte, sizeof(double)> encoded{};
  for (std::uint64_t element = first_element; element <= last_element; ++element) {
    const std::optional<PhysicalCoordinate> coordinate = physical_coordinate(matrix, element);
    if (!coordinate.has_value()) {
      return false;
    }
    const std::optional<std::uint64_t> value = value_for_coordinate(*coordinate, operand, problem);
    if (!value.has_value() ||
        !encode_pattern_value(
            *value, matrix.data_type,
            std::span<std::byte>(encoded.data(), static_cast<std::size_t>(item_bytes)))) {
      return false;
    }

    const std::uint64_t element_begin = element * item_bytes;
    const std::uint64_t copy_begin = std::max(element_begin, absolute_byte_offset);
    const std::uint64_t copy_end = std::min(element_begin + item_bytes, end_offset);
    const auto source_offset = static_cast<std::size_t>(copy_begin - element_begin);
    const auto destination_offset = static_cast<std::size_t>(copy_begin - absolute_byte_offset);
    const auto copy_bytes = static_cast<std::size_t>(copy_end - copy_begin);
    std::memcpy(output.data() + destination_offset, encoded.data() + source_offset, copy_bytes);
  }
  return true;
}

} // namespace xvram::gemm_bench
