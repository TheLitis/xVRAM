#pragma once
#include <cstdint>
#include <optional>
#include <span>
#include <string>

namespace xvram::compat_bench {
struct Shape {
  std::uint64_t m = 0, n = 0, k = 0, padding = 0, offset = 0;
  bool transpose_a = false, transpose_b = false;
};
struct Matrix {
  std::uint64_t rows = 0, columns = 0, ld = 0, offset = 0, elements = 0;
};
enum class Operand { a, b, c };
[[nodiscard]] std::optional<Matrix> matrix_storage(const Shape& shape, Operand operand) noexcept;
[[nodiscard]] std::optional<std::uint64_t> operand_bytes(const Shape& shape) noexcept;
[[nodiscard]] float pattern_a(std::uint64_t row, std::uint64_t k, std::uint64_t seed) noexcept;
[[nodiscard]] float pattern_b(std::uint64_t k, std::uint64_t column, std::uint64_t seed) noexcept;
[[nodiscard]] float pattern_c(std::uint64_t row, std::uint64_t column) noexcept;
[[nodiscard]] double reference_dot(std::uint64_t row, std::uint64_t column, std::uint64_t k,
                                   std::uint64_t seed, bool full_reference) noexcept;
[[nodiscard]] double reference_result(const Shape& shape, std::uint64_t row, std::uint64_t column,
                                      std::uint64_t seed, float alpha, float beta,
                                      std::uint32_t passes, bool full_reference) noexcept;
void fill_pattern(std::span<float> output, std::uint64_t element_offset, const Shape& shape,
                  Operand operand, std::uint64_t seed);
inline constexpr float padding_value = -1234.5F;
struct Digest {
  std::uint64_t a = 1469598103934665603ULL, b = 0x9e3779b97f4a7c15ULL;
  void add(float value) noexcept;
  [[nodiscard]] std::string string() const;
};
} // namespace xvram::compat_bench
