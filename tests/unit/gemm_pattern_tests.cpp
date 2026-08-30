#include "gemm_bench/pattern.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace {

int failures = 0;

void check(const bool condition, const char* expression, const int line) {
  if (!condition) {
    std::cerr << "line " << line << ": check failed: " << expression << '\n';
    ++failures;
  }
}

#define CHECK(expression) check(static_cast<bool>(expression), #expression, __LINE__)

using xvram::gemm_bench::PatternMatrix;
using xvram::gemm_bench::PatternOperand;
using xvram::gemm_bench::PatternProblem;
using xvram::gemm_bench::RequestedDataType;
using xvram::gemm_bench::RequestedLayout;
using xvram::gemm_bench::RequestedOperation;

[[nodiscard]] PatternMatrix matrix_for(const PatternProblem& problem, const PatternOperand operand,
                                       const RequestedLayout layout, const RequestedDataType type) {
  std::uint64_t rows = 0;
  std::uint64_t columns = 0;
  if (operand == PatternOperand::a) {
    rows = problem.operation_a == RequestedOperation::none ? problem.m : problem.k;
    columns = problem.operation_a == RequestedOperation::none ? problem.k : problem.m;
  } else if (operand == PatternOperand::b) {
    rows = problem.operation_b == RequestedOperation::none ? problem.k : problem.n;
    columns = problem.operation_b == RequestedOperation::none ? problem.n : problem.k;
  } else {
    rows = problem.m;
    columns = problem.n;
  }
  const std::uint64_t leading = (layout == RequestedLayout::row_major ? columns : rows) + 2ULL;
  return PatternMatrix{rows, columns, leading, layout, type};
}

[[nodiscard]] std::vector<std::byte>
filled(const PatternMatrix& matrix, const PatternOperand operand, const PatternProblem& problem) {
  const auto bytes = xvram::gemm_bench::pattern_storage_bytes(matrix);
  CHECK(bytes.has_value());
  std::vector<std::byte> output(static_cast<std::size_t>(bytes.value_or(0)),
                                static_cast<std::byte>(0xA5));
  CHECK(xvram::gemm_bench::fill_pattern_bytes(output, 0, matrix, operand, problem));
  return output;
}

[[nodiscard]] std::optional<double> value_at(const std::vector<std::byte>& storage,
                                             const PatternMatrix& matrix, const std::uint64_t row,
                                             const std::uint64_t column) {
  const auto index = xvram::gemm_bench::pattern_physical_index(matrix, row, column);
  const std::uint64_t item_bytes = xvram::gemm_bench::pattern_element_bytes(matrix.data_type);
  if (!index.has_value() || *index > std::numeric_limits<std::uint64_t>::max() / item_bytes) {
    return std::nullopt;
  }
  const std::uint64_t offset = *index * item_bytes;
  if (offset > static_cast<std::uint64_t>(storage.size()) ||
      item_bytes > static_cast<std::uint64_t>(storage.size()) - offset) {
    return std::nullopt;
  }
  return xvram::gemm_bench::decode_pattern_value(
      std::span<const std::byte>(storage.data() + static_cast<std::size_t>(offset),
                                 static_cast<std::size_t>(item_bytes)),
      matrix.data_type);
}

[[nodiscard]] std::vector<std::byte> filled_in_tails(const PatternMatrix& matrix,
                                                     const PatternOperand operand,
                                                     const PatternProblem& problem) {
  static constexpr std::array<std::uint64_t, 9> batches{1, 7, 2, 13, 3, 5, 17, 4, 11};
  const auto bytes = xvram::gemm_bench::pattern_storage_bytes(matrix);
  CHECK(bytes.has_value());
  std::vector<std::byte> output(static_cast<std::size_t>(bytes.value_or(0)),
                                static_cast<std::byte>(0x5A));
  std::uint64_t offset = 0;
  std::size_t batch_index = 0;
  while (bytes.has_value() && offset < *bytes) {
    const std::uint64_t batch = std::min(batches[batch_index % batches.size()], *bytes - offset);
    CHECK(xvram::gemm_bench::fill_pattern_bytes(
        std::span<std::byte>(output.data() + static_cast<std::size_t>(offset),
                             static_cast<std::size_t>(batch)),
        offset, matrix, operand, problem));
    offset += batch;
    ++batch_index;
  }
  return output;
}

void value_and_sum_tests() {
  using namespace xvram::gemm_bench;
  CHECK(pattern_element_bytes(RequestedDataType::suite) == 0);
  CHECK(pattern_element_bytes(RequestedDataType::fp16) == 2);
  CHECK(pattern_element_bytes(RequestedDataType::bf16) == 2);
  CHECK(pattern_element_bytes(RequestedDataType::fp32) == 4);
  CHECK(pattern_element_bytes(RequestedDataType::fp64) == 8);

  CHECK(pattern_a_value(0, 0) == 1);
  CHECK(pattern_a_value(2, 1) == 4);
  CHECK(pattern_a_value(4, 7) == 3);
  CHECK(pattern_b_value(0, 0) == 1);
  CHECK(pattern_b_value(5, 4) == 7);
  CHECK(pattern_b_value(8, 6) == 4);

  for (std::uint64_t k = 0; k <= 300; ++k) {
    std::uint64_t sum_p = 0;
    std::uint64_t sum_q = 0;
    std::uint64_t sum_pq = 0;
    for (std::uint64_t index = 0; index < k; ++index) {
      const std::uint64_t p = index % 2ULL;
      const std::uint64_t q = index % 3ULL;
      sum_p += p;
      sum_q += q;
      sum_pq += p * q;
    }
    const auto sums = pattern_closed_form_sums(k);
    CHECK(sums.sum_p == sum_p);
    CHECK(sums.sum_q == sum_q);
    CHECK(sums.sum_pq == sum_pq);
  }
  const auto split_sums = pattern_closed_form_sums(32769);
  CHECK(split_sums.sum_p == 16384);
  CHECK(split_sums.sum_q == 32769);
  CHECK(split_sums.sum_pq == 16384);

  for (std::uint64_t row = 0; row < 7; ++row) {
    for (std::uint64_t column = 0; column < 9; ++column) {
      for (std::uint64_t k = 1; k < 24; ++k) {
        std::uint64_t naive = 0;
        for (std::uint64_t index = 0; index < k; ++index) {
          naive += pattern_a_value(row, index) * pattern_b_value(index, column);
        }
        CHECK(pattern_expected_value(row, column, k) == naive);
      }
    }
  }
  CHECK(pattern_expected_value(4, 6, 11) == 78);
  CHECK(!pattern_expected_value(2, 4, std::numeric_limits<std::uint64_t>::max()).has_value());
}

void encoding_tests() {
  using namespace xvram::gemm_bench;
  constexpr std::array types{RequestedDataType::fp16, RequestedDataType::bf16,
                             RequestedDataType::fp32, RequestedDataType::fp64};
  constexpr std::array<std::uint64_t, 6> values{0, 1, 7, 127, 257, 5390};
  for (const RequestedDataType type : types) {
    const std::uint64_t bytes = pattern_element_bytes(type);
    for (const std::uint64_t value : values) {
      std::array<std::byte, sizeof(double)> encoded{};
      CHECK(encode_pattern_value(
          value, type, std::span<std::byte>(encoded.data(), static_cast<std::size_t>(bytes))));
      const auto decoded = decode_pattern_value(
          std::span<const std::byte>(encoded.data(), static_cast<std::size_t>(bytes)), type);
      CHECK(decoded.has_value());
      CHECK(decoded.value_or(-1.0) >= 0.0);
      CHECK(decoded.value_or(-1.0) <= 5400.0);
    }
  }

  std::array<std::byte, 2> half_one{};
  std::array<std::byte, 2> bf16_one{};
  CHECK(encode_pattern_value(1, RequestedDataType::fp16, half_one));
  CHECK(encode_pattern_value(1, RequestedDataType::bf16, bf16_one));
  std::uint16_t half_raw = 0;
  std::uint16_t bf16_raw = 0;
  std::memcpy(&half_raw, half_one.data(), sizeof(half_raw));
  std::memcpy(&bf16_raw, bf16_one.data(), sizeof(bf16_raw));
  CHECK(half_raw == 0x3C00U);
  CHECK(bf16_raw == 0x3F80U);
  CHECK(!encode_pattern_value(1, RequestedDataType::suite, half_one));
  CHECK(!decode_pattern_value(half_one, RequestedDataType::suite).has_value());
  CHECK(!decode_pattern_value(std::span<const std::byte>{}, RequestedDataType::fp64).has_value());
}

void physical_index_tests() {
  using namespace xvram::gemm_bench;
  const PatternMatrix row{3, 4, 6, RequestedLayout::row_major, RequestedDataType::fp32};
  const PatternMatrix column{3, 4, 5, RequestedLayout::column_major, RequestedDataType::fp32};
  CHECK(pattern_physical_index(row, 0, 0) == 0);
  CHECK(pattern_physical_index(row, 2, 3) == 15);
  CHECK(pattern_storage_bytes(row) == 72);
  CHECK(pattern_physical_index(column, 0, 0) == 0);
  CHECK(pattern_physical_index(column, 2, 3) == 17);
  CHECK(pattern_storage_bytes(column) == 80);
  CHECK(!pattern_physical_index(row, 3, 0).has_value());
  CHECK(!pattern_physical_index(column, 0, 4).has_value());

  PatternMatrix bad_leading = row;
  bad_leading.leading_dimension = 3;
  CHECK(!pattern_storage_bytes(bad_leading).has_value());
  PatternMatrix bad_type = row;
  bad_type.data_type = RequestedDataType::suite;
  CHECK(!pattern_storage_bytes(bad_type).has_value());
  const PatternMatrix overflow{std::numeric_limits<std::uint64_t>::max(), 2,
                               std::numeric_limits<std::uint64_t>::max(),
                               RequestedLayout::row_major, RequestedDataType::fp64};
  CHECK(!pattern_storage_bytes(overflow).has_value());
}

void layout_transpose_type_and_tail_tests() {
  using namespace xvram::gemm_bench;
  constexpr std::array types{RequestedDataType::fp16, RequestedDataType::bf16,
                             RequestedDataType::fp32, RequestedDataType::fp64};
  constexpr std::array layouts{RequestedLayout::row_major, RequestedLayout::column_major};
  constexpr std::array operations{RequestedOperation::none, RequestedOperation::transpose};

  for (const RequestedDataType type : types) {
    for (const RequestedLayout a_layout : layouts) {
      for (const RequestedLayout b_layout : layouts) {
        for (const RequestedLayout c_layout : layouts) {
          for (const RequestedOperation operation_a : operations) {
            for (const RequestedOperation operation_b : operations) {
              const PatternProblem problem{5, 7, 11, operation_a, operation_b};
              const PatternMatrix a = matrix_for(problem, PatternOperand::a, a_layout, type);
              const PatternMatrix b = matrix_for(problem, PatternOperand::b, b_layout, type);
              const PatternMatrix c =
                  matrix_for(problem, PatternOperand::expected_c, c_layout, type);
              const std::vector<std::byte> a_storage = filled(a, PatternOperand::a, problem);
              const std::vector<std::byte> b_storage = filled(b, PatternOperand::b, problem);
              const std::vector<std::byte> c_storage =
                  filled(c, PatternOperand::expected_c, problem);
              CHECK(filled_in_tails(a, PatternOperand::a, problem) == a_storage);
              CHECK(filled_in_tails(b, PatternOperand::b, problem) == b_storage);
              CHECK(filled_in_tails(c, PatternOperand::expected_c, problem) == c_storage);

              for (std::uint64_t row = 0; row < problem.m; ++row) {
                for (std::uint64_t k_index = 0; k_index < problem.k; ++k_index) {
                  const std::uint64_t stored_row =
                      operation_a == RequestedOperation::none ? row : k_index;
                  const std::uint64_t stored_column =
                      operation_a == RequestedOperation::none ? k_index : row;
                  CHECK(value_at(a_storage, a, stored_row, stored_column) ==
                        static_cast<double>(pattern_a_value(row, k_index)));
                }
              }
              for (std::uint64_t k_index = 0; k_index < problem.k; ++k_index) {
                for (std::uint64_t column = 0; column < problem.n; ++column) {
                  const std::uint64_t stored_row =
                      operation_b == RequestedOperation::none ? k_index : column;
                  const std::uint64_t stored_column =
                      operation_b == RequestedOperation::none ? column : k_index;
                  CHECK(value_at(b_storage, b, stored_row, stored_column) ==
                        static_cast<double>(pattern_b_value(k_index, column)));
                }
              }

              for (std::uint64_t row = 0; row < problem.m; ++row) {
                for (std::uint64_t column = 0; column < problem.n; ++column) {
                  std::uint64_t naive = 0;
                  for (std::uint64_t k_index = 0; k_index < problem.k; ++k_index) {
                    naive += pattern_a_value(row, k_index) * pattern_b_value(k_index, column);
                  }
                  CHECK(pattern_expected_value(row, column, problem.k) == naive);
                  std::array<std::byte, sizeof(double)> encoded{};
                  const std::uint64_t item_bytes = pattern_element_bytes(type);
                  CHECK(encode_pattern_value(
                      naive, type,
                      std::span<std::byte>(encoded.data(), static_cast<std::size_t>(item_bytes))));
                  const auto encoded_value = decode_pattern_value(
                      std::span<const std::byte>(encoded.data(),
                                                 static_cast<std::size_t>(item_bytes)),
                      type);
                  CHECK(value_at(c_storage, c, row, column) == encoded_value);
                }
              }
            }
          }
        }
      }
    }
  }
}

void range_validation_tests() {
  using namespace xvram::gemm_bench;
  const PatternProblem problem{3, 4, 5, RequestedOperation::none, RequestedOperation::none};
  const PatternMatrix matrix = matrix_for(problem, PatternOperand::expected_c,
                                          RequestedLayout::row_major, RequestedDataType::fp32);
  const auto bytes = pattern_storage_bytes(matrix);
  CHECK(bytes.has_value());
  std::array<std::byte, 3> tail{};
  CHECK(
      fill_pattern_bytes(tail, *bytes - tail.size(), matrix, PatternOperand::expected_c, problem));
  CHECK(!fill_pattern_bytes(tail, *bytes - tail.size() + 1ULL, matrix, PatternOperand::expected_c,
                            problem));
  CHECK(!fill_pattern_bytes(tail, *bytes + 1ULL, matrix, PatternOperand::expected_c, problem));
  CHECK(fill_pattern_bytes(std::span<std::byte>{}, *bytes, matrix, PatternOperand::expected_c,
                           problem));

  PatternProblem wrong = problem;
  ++wrong.m;
  CHECK(!fill_pattern_bytes(tail, 0, matrix, PatternOperand::expected_c, wrong));
}

} // namespace

int main() {
  value_and_sum_tests();
  encoding_tests();
  physical_index_tests();
  layout_transpose_type_and_tail_tests();
  range_validation_tests();
  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
    return 1;
  }
  std::cout << "all GEMM pattern tests passed\n";
  return 0;
}
