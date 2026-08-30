#pragma once

#include "residency/core.hpp"

#include <compare>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace xvram::gemm {

enum class MatrixLayout {
  row_major,
  column_major,
};

enum class MatrixOperation {
  none,
  transpose,
};

enum class ElementType {
  fp16,
  bf16,
  fp32,
  fp64,
};

enum class ComputeMode {
  strict_fp32,
  fast_tf32,
  fp64,
};

[[nodiscard]] std::string_view matrix_layout_name(MatrixLayout layout) noexcept;
[[nodiscard]] std::string_view matrix_operation_name(MatrixOperation operation) noexcept;
[[nodiscard]] std::string_view element_type_name(ElementType type) noexcept;
[[nodiscard]] std::string_view compute_mode_name(ComputeMode mode) noexcept;
[[nodiscard]] std::optional<std::uint64_t> element_size_bytes(ElementType type) noexcept;

struct MatrixView {
  residency::AllocationId allocation_id;
  std::uint64_t allocation_bytes = 0;
  std::uint64_t offset_bytes = 0;
  std::uint64_t rows = 0;
  std::uint64_t columns = 0;
  std::uint64_t leading_dimension = 0;
  MatrixLayout layout = MatrixLayout::row_major;
  ElementType element_type = ElementType::fp32;
};

struct GemmProblem {
  std::uint64_t m = 0;
  std::uint64_t n = 0;
  std::uint64_t k = 0;
  MatrixView a;
  MatrixView b;
  MatrixView c;
  MatrixOperation a_operation = MatrixOperation::none;
  MatrixOperation b_operation = MatrixOperation::none;
  ComputeMode compute_mode = ComputeMode::fast_tf32;
  double alpha = 1.0;
  double beta = 0.0;
};

enum class ProblemError {
  none,
  zero_dimension,
  invalid_allocation,
  invalid_layout,
  invalid_operation,
  invalid_element_type,
  invalid_compute_mode,
  dimension_mismatch,
  leading_dimension_too_small,
  storage_overflow,
  storage_out_of_bounds,
  unsupported_type_combination,
  output_aliases_input,
  integer_limit_exceeded,
};

struct ProblemValidation {
  ProblemError error = ProblemError::none;
  char operand = '\0';

  [[nodiscard]] explicit operator bool() const noexcept {
    return error == ProblemError::none;
  }
};

[[nodiscard]] std::string_view problem_error_name(ProblemError error) noexcept;
[[nodiscard]] ProblemValidation validate_problem(const GemmProblem& problem) noexcept;

enum class BetaMode {
  user_beta,
  accumulate_one,
};

struct WorkspaceSignature {
  ElementType a_type = ElementType::fp32;
  ElementType b_type = ElementType::fp32;
  ElementType c_type = ElementType::fp32;
  ComputeMode compute_mode = ComputeMode::fast_tf32;
  std::uint64_t tile_m = 0;
  std::uint64_t tile_n = 0;
  std::uint64_t tile_k = 0;
  std::uint64_t workspace_limit_bytes = 0;

  [[nodiscard]] auto operator<=>(const WorkspaceSignature&) const noexcept = default;
};

struct AlgorithmSignature {
  MatrixLayout a_layout = MatrixLayout::row_major;
  MatrixLayout b_layout = MatrixLayout::row_major;
  MatrixLayout c_layout = MatrixLayout::row_major;
  MatrixOperation a_operation = MatrixOperation::none;
  MatrixOperation b_operation = MatrixOperation::none;
  std::uint64_t a_leading_dimension = 0;
  std::uint64_t b_leading_dimension = 0;
  std::uint64_t c_leading_dimension = 0;
  std::uint16_t a_alignment_bytes = 0;
  std::uint16_t b_alignment_bytes = 0;
  std::uint16_t c_alignment_bytes = 0;
  WorkspaceSignature workspace;

  [[nodiscard]] auto operator<=>(const AlgorithmSignature&) const noexcept = default;
};

struct GemmTile {
  std::uint64_t sequence = 0;
  std::uint64_t m_begin = 0;
  std::uint64_t n_begin = 0;
  std::uint64_t k_begin = 0;
  std::uint64_t m_count = 0;
  std::uint64_t n_count = 0;
  std::uint64_t k_count = 0;
  BetaMode beta_mode = BetaMode::user_beta;
  residency::AccessMode c_access_mode = residency::AccessMode::read_write;
  std::uint64_t resident_bytes = 0;
  std::uint64_t chunk_count = 0;
  std::uint64_t access_range_count = 0;
  AlgorithmSignature algorithm_signature;
};

enum class WorkingSetError {
  none,
  invalid_problem,
  invalid_tile,
  invalid_chunk_size,
  range_overflow,
  too_many_ranges,
  access_normalization_failed,
  resident_size_overflow,
};

struct WorkingSetResult {
  WorkingSetError error = WorkingSetError::none;
  std::vector<residency::AccessRange> accesses;
  std::vector<residency::ChunkKey> chunks;
  std::uint64_t resident_bytes = 0;

  [[nodiscard]] explicit operator bool() const noexcept {
    return error == WorkingSetError::none;
  }
};

[[nodiscard]] std::string_view working_set_error_name(WorkingSetError error) noexcept;

// Returns disjoint, allocation-relative access ranges and the exact unique chunk working set for
// one tile. Transposed and strided operands are enumerated as physical row/column segments rather
// than a conservative bounding interval.
[[nodiscard]] WorkingSetResult
enumerate_tile_working_set(const GemmProblem& problem, const GemmTile& tile,
                           std::uint64_t chunk_bytes, std::uint64_t maximum_access_ranges = 131072);

struct TileGeometry {
  std::uint64_t m = 0;
  std::uint64_t n = 0;
  std::uint64_t k = 0;

  [[nodiscard]] auto operator<=>(const TileGeometry&) const noexcept = default;
};

struct PlannerConfig {
  std::uint64_t chunk_bytes = 64ULL * 1024ULL * 1024ULL;
  std::uint64_t cache_target_bytes = 0;
  std::uint64_t workspace_bytes = 4ULL * 1024ULL * 1024ULL;
  TileGeometry preferred_geometry{4096, 4096, 1024};
  std::uint64_t maximum_tiles = 1'000'000;
  std::uint64_t maximum_access_ranges_per_tile = 131072;
};

// Deterministic work counters used to guard planner-complexity regressions without relying on
// machine-dependent wall-clock thresholds.
struct PlannerStatistics {
  std::uint64_t geometry_assessments = 0;
  std::uint64_t matrix_rectangle_summaries = 0;
  std::uint64_t tile_combinations_assessed = 0;
  std::uint64_t legacy_tile_enumerations = 0;
};

enum class PlanError {
  none,
  invalid_problem,
  invalid_chunk_size,
  invalid_cache_target,
  workspace_exceeds_cache,
  invalid_preferred_geometry,
  tile_count_overflow,
  too_many_tiles,
  working_set_too_large,
  working_set_enumeration_failed,
};

struct GemmPlan {
  PlanError error = PlanError::none;
  ProblemValidation problem_validation;
  WorkingSetError working_set_error = WorkingSetError::none;
  TileGeometry geometry;
  std::uint64_t cache_bytes_available_to_chunks = 0;
  std::uint64_t maximum_resident_bytes = 0;
  PlannerStatistics statistics;
  std::vector<GemmTile> tiles;

  [[nodiscard]] explicit operator bool() const noexcept {
    return error == PlanError::none;
  }
};

[[nodiscard]] std::string_view plan_error_name(PlanError error) noexcept;
[[nodiscard]] GemmPlan make_plan(const GemmProblem& problem, const PlannerConfig& config);

[[nodiscard]] AlgorithmSignature
make_algorithm_signature(const GemmProblem& problem, const GemmTile& tile,
                         std::uint64_t workspace_limit_bytes) noexcept;

} // namespace xvram::gemm
