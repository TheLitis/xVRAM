#pragma once

#include "gemm_bench/executor.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace xvram::gemm_bench {

enum class PatternOperand { a, b, expected_c };

struct PatternProblem {
  std::uint64_t m = 0;
  std::uint64_t n = 0;
  std::uint64_t k = 0;
  RequestedOperation operation_a = RequestedOperation::none;
  RequestedOperation operation_b = RequestedOperation::none;
};

struct PatternMatrix {
  std::uint64_t rows = 0;
  std::uint64_t columns = 0;
  std::uint64_t leading_dimension = 0;
  RequestedLayout layout = RequestedLayout::row_major;
  RequestedDataType data_type = RequestedDataType::fp32;
};

struct PatternSums {
  std::uint64_t sum_p = 0;
  std::uint64_t sum_q = 0;
  std::uint64_t sum_pq = 0;
};

[[nodiscard]] std::uint64_t pattern_element_bytes(RequestedDataType type) noexcept;

[[nodiscard]] std::uint64_t pattern_a_value(std::uint64_t row, std::uint64_t k_index) noexcept;
[[nodiscard]] std::uint64_t pattern_b_value(std::uint64_t k_index, std::uint64_t column) noexcept;
[[nodiscard]] PatternSums pattern_closed_form_sums(std::uint64_t k) noexcept;
[[nodiscard]] std::optional<std::uint64_t>
pattern_expected_value(std::uint64_t row, std::uint64_t column, std::uint64_t k) noexcept;

[[nodiscard]] std::optional<std::uint64_t> pattern_physical_index(const PatternMatrix& matrix,
                                                                  std::uint64_t row,
                                                                  std::uint64_t column) noexcept;
[[nodiscard]] std::optional<std::uint64_t>
pattern_storage_bytes(const PatternMatrix& matrix) noexcept;

[[nodiscard]] bool encode_pattern_value(std::uint64_t value, RequestedDataType type,
                                        std::span<std::byte> output) noexcept;
[[nodiscard]] std::optional<double> decode_pattern_value(std::span<const std::byte> input,
                                                         RequestedDataType type) noexcept;

// Generates an arbitrary byte range of one physical matrix allocation. This deliberately
// supports ranges that begin or end in the middle of an element so streaming I/O batch tails do
// not need special handling. Padding implied by a non-tight leading dimension is zero-filled.
[[nodiscard]] bool fill_pattern_bytes(std::span<std::byte> output,
                                      std::uint64_t absolute_byte_offset,
                                      const PatternMatrix& matrix, PatternOperand operand,
                                      const PatternProblem& problem) noexcept;

} // namespace xvram::gemm_bench
