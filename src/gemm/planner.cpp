#include "gemm/planner.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <tuple>
#include <utility>

namespace xvram::gemm {
namespace {

constexpr std::uint64_t maximum_algorithm_alignment = 256;

[[nodiscard]] bool checked_add(const std::uint64_t left, const std::uint64_t right,
                               std::uint64_t& output) noexcept {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    return false;
  }
  output = left + right;
  return true;
}

[[nodiscard]] bool checked_multiply(const std::uint64_t left, const std::uint64_t right,
                                    std::uint64_t& output) noexcept {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
    return false;
  }
  output = left * right;
  return true;
}

[[nodiscard]] bool is_valid_layout(const MatrixLayout layout) noexcept {
  return layout == MatrixLayout::row_major || layout == MatrixLayout::column_major;
}

[[nodiscard]] bool is_valid_operation(const MatrixOperation operation) noexcept {
  return operation == MatrixOperation::none || operation == MatrixOperation::transpose;
}

[[nodiscard]] bool is_valid_compute_mode(const ComputeMode mode) noexcept {
  return mode == ComputeMode::strict_fp32 || mode == ComputeMode::fast_tf32 ||
         mode == ComputeMode::fp64;
}

struct MatrixBounds {
  std::uint64_t begin = 0;
  std::uint64_t end = 0;
};

[[nodiscard]] ProblemValidation validate_matrix(const MatrixView& matrix,
                                                const char operand) noexcept {
  const auto fail = [operand](const ProblemError error) {
    return ProblemValidation{error, operand};
  };
  if (!matrix.allocation_id || matrix.allocation_bytes == 0) {
    return fail(ProblemError::invalid_allocation);
  }
  if (!is_valid_layout(matrix.layout)) {
    return fail(ProblemError::invalid_layout);
  }
  const std::optional<std::uint64_t> element_bytes = element_size_bytes(matrix.element_type);
  if (!element_bytes.has_value()) {
    return fail(ProblemError::invalid_element_type);
  }
  if (matrix.rows == 0 || matrix.columns == 0) {
    return fail(ProblemError::zero_dimension);
  }
  const std::uint64_t required_leading_dimension =
      matrix.layout == MatrixLayout::row_major ? matrix.columns : matrix.rows;
  if (matrix.leading_dimension < required_leading_dimension) {
    return fail(ProblemError::leading_dimension_too_small);
  }
  if (matrix.leading_dimension > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    return fail(ProblemError::integer_limit_exceeded);
  }

  const std::uint64_t major_count =
      matrix.layout == MatrixLayout::row_major ? matrix.rows : matrix.columns;
  const std::uint64_t minor_count =
      matrix.layout == MatrixLayout::row_major ? matrix.columns : matrix.rows;
  std::uint64_t last_major_offset = 0;
  if (!checked_multiply(major_count - 1U, matrix.leading_dimension, last_major_offset)) {
    return fail(ProblemError::storage_overflow);
  }
  std::uint64_t last_element = 0;
  if (!checked_add(last_major_offset, minor_count - 1U, last_element) ||
      !checked_add(last_element, 1, last_element)) {
    return fail(ProblemError::storage_overflow);
  }
  std::uint64_t storage_bytes = 0;
  std::uint64_t storage_end = 0;
  if (!checked_multiply(last_element, *element_bytes, storage_bytes) ||
      !checked_add(matrix.offset_bytes, storage_bytes, storage_end)) {
    return fail(ProblemError::storage_overflow);
  }
  if (matrix.offset_bytes >= matrix.allocation_bytes || storage_end > matrix.allocation_bytes) {
    return fail(ProblemError::storage_out_of_bounds);
  }
  return {};
}

[[nodiscard]] std::optional<MatrixBounds> matrix_bounds(const MatrixView& matrix) noexcept {
  const std::optional<std::uint64_t> element_bytes = element_size_bytes(matrix.element_type);
  if (!element_bytes.has_value() || matrix.rows == 0 || matrix.columns == 0) {
    return std::nullopt;
  }
  const std::uint64_t major_count =
      matrix.layout == MatrixLayout::row_major ? matrix.rows : matrix.columns;
  const std::uint64_t minor_count =
      matrix.layout == MatrixLayout::row_major ? matrix.columns : matrix.rows;
  std::uint64_t elements = 0;
  std::uint64_t bytes = 0;
  std::uint64_t end = 0;
  if (!checked_multiply(major_count - 1U, matrix.leading_dimension, elements) ||
      !checked_add(elements, minor_count, elements) ||
      !checked_multiply(elements, *element_bytes, bytes) ||
      !checked_add(matrix.offset_bytes, bytes, end)) {
    return std::nullopt;
  }
  return MatrixBounds{matrix.offset_bytes, end};
}

[[nodiscard]] std::pair<std::uint64_t, std::uint64_t>
operated_shape(const MatrixView& matrix, const MatrixOperation operation) noexcept {
  if (operation == MatrixOperation::none) {
    return {matrix.rows, matrix.columns};
  }
  return {matrix.columns, matrix.rows};
}

[[nodiscard]] std::uint64_t tile_alignment(const GemmProblem& problem) noexcept {
  if (problem.a.element_type == ElementType::fp16 || problem.a.element_type == ElementType::bf16) {
    return 16;
  }
  if (problem.compute_mode == ComputeMode::fp64) {
    return 4;
  }
  return 8;
}

[[nodiscard]] std::uint64_t reduced_dimension(const std::uint64_t value,
                                              const std::uint64_t alignment) noexcept {
  if (value <= 1) {
    return value;
  }
  std::uint64_t reduced = value / 2U;
  if (value > alignment) {
    reduced = std::max(reduced, alignment);
    reduced -= reduced % alignment;
  }
  reduced = std::max<std::uint64_t>(1, reduced);
  if (reduced >= value) {
    reduced = value - 1U;
  }
  return reduced;
}

[[nodiscard]] std::uint64_t ceil_divide(const std::uint64_t value,
                                        const std::uint64_t divisor) noexcept {
  return value / divisor + static_cast<std::uint64_t>((value % divisor) != 0);
}

[[nodiscard]] std::optional<std::uint64_t> tile_count(const GemmProblem& problem,
                                                      const TileGeometry& geometry) noexcept {
  if (geometry.m == 0 || geometry.n == 0 || geometry.k == 0) {
    return std::nullopt;
  }
  const std::uint64_t m_tiles = ceil_divide(problem.m, geometry.m);
  const std::uint64_t n_tiles = ceil_divide(problem.n, geometry.n);
  const std::uint64_t k_tiles = ceil_divide(problem.k, geometry.k);
  std::uint64_t count = 0;
  if (!checked_multiply(m_tiles, n_tiles, count) || !checked_multiply(count, k_tiles, count)) {
    return std::nullopt;
  }
  return count;
}

[[nodiscard]] bool append_access(std::vector<residency::AccessRange>& ranges,
                                 const residency::AccessRange& access,
                                 const std::uint64_t maximum_ranges) {
  if (!ranges.empty()) {
    residency::AccessRange& previous = ranges.back();
    std::uint64_t previous_end = 0;
    if (previous.allocation_id == access.allocation_id && previous.mode == access.mode &&
        checked_add(previous.offset_bytes, previous.length_bytes, previous_end) &&
        previous_end == access.offset_bytes &&
        checked_add(previous.length_bytes, access.length_bytes, previous.length_bytes)) {
      return true;
    }
  }
  if (ranges.size() >= maximum_ranges) {
    return false;
  }
  ranges.push_back(access);
  return true;
}

[[nodiscard]] WorkingSetError
append_matrix_rectangle(const MatrixView& matrix, const MatrixOperation operation,
                        const std::uint64_t logical_row, const std::uint64_t logical_rows,
                        const std::uint64_t logical_column, const std::uint64_t logical_columns,
                        const residency::AccessMode mode, const std::uint64_t maximum_ranges,
                        std::vector<residency::AccessRange>& ranges) {
  const auto [operated_rows, operated_columns] = operated_shape(matrix, operation);
  std::uint64_t logical_row_end = 0;
  std::uint64_t logical_column_end = 0;
  if (logical_rows == 0 || logical_columns == 0 ||
      !checked_add(logical_row, logical_rows, logical_row_end) ||
      !checked_add(logical_column, logical_columns, logical_column_end) ||
      logical_row_end > operated_rows || logical_column_end > operated_columns) {
    return WorkingSetError::invalid_tile;
  }

  const std::uint64_t physical_row =
      operation == MatrixOperation::none ? logical_row : logical_column;
  const std::uint64_t physical_rows =
      operation == MatrixOperation::none ? logical_rows : logical_columns;
  const std::uint64_t physical_column =
      operation == MatrixOperation::none ? logical_column : logical_row;
  const std::uint64_t physical_columns =
      operation == MatrixOperation::none ? logical_columns : logical_rows;
  const std::uint64_t element_bytes = *element_size_bytes(matrix.element_type);

  const std::uint64_t segments =
      matrix.layout == MatrixLayout::row_major ? physical_rows : physical_columns;
  const std::uint64_t segment_elements =
      matrix.layout == MatrixLayout::row_major ? physical_columns : physical_rows;
  std::uint64_t segment_bytes = 0;
  if (!checked_multiply(segment_elements, element_bytes, segment_bytes)) {
    return WorkingSetError::range_overflow;
  }

  for (std::uint64_t segment = 0; segment < segments; ++segment) {
    std::uint64_t major = 0;
    std::uint64_t element_index = 0;
    std::uint64_t byte_offset = 0;
    const std::uint64_t minor =
        matrix.layout == MatrixLayout::row_major ? physical_column : physical_row;
    if (!checked_add(matrix.layout == MatrixLayout::row_major ? physical_row : physical_column,
                     segment, major) ||
        !checked_multiply(major, matrix.leading_dimension, element_index) ||
        !checked_add(element_index, minor, element_index) ||
        !checked_multiply(element_index, element_bytes, byte_offset) ||
        !checked_add(matrix.offset_bytes, byte_offset, byte_offset)) {
      return WorkingSetError::range_overflow;
    }
    if (!append_access(
            ranges, residency::AccessRange{matrix.allocation_id, byte_offset, segment_bytes, mode},
            maximum_ranges)) {
      return WorkingSetError::too_many_ranges;
    }
  }
  return WorkingSetError::none;
}

[[nodiscard]] std::vector<residency::AllocationLayout>
allocation_layouts(const GemmProblem& problem) {
  std::array matrices{&problem.a, &problem.b, &problem.c};
  std::vector<residency::AllocationLayout> layouts;
  for (const MatrixView* matrix : matrices) {
    const auto found = std::find_if(layouts.begin(), layouts.end(), [&](const auto& layout) {
      return layout.id == matrix->allocation_id;
    });
    if (found == layouts.end()) {
      layouts.push_back({matrix->allocation_id, matrix->allocation_bytes});
    }
  }
  return layouts;
}

struct FastNormalizationResult {
  bool requires_general_normalizer = false;
  std::vector<residency::AccessRange> ranges;
};

[[nodiscard]] FastNormalizationResult
fast_normalize_accesses(std::vector<residency::AccessRange> accesses) {
  std::sort(accesses.begin(), accesses.end(),
            [](const residency::AccessRange& left, const residency::AccessRange& right) {
              return std::tuple{left.allocation_id, left.offset_bytes, left.length_bytes,
                                left.mode} < std::tuple{right.allocation_id, right.offset_bytes,
                                                        right.length_bytes, right.mode};
            });

  FastNormalizationResult result;
  result.ranges.reserve(accesses.size());
  for (const residency::AccessRange& access : accesses) {
    if (result.ranges.empty() || result.ranges.back().allocation_id != access.allocation_id) {
      result.ranges.push_back(access);
      continue;
    }

    residency::AccessRange& previous = result.ranges.back();
    const std::uint64_t previous_end = previous.offset_bytes + previous.length_bytes;
    const std::uint64_t access_end = access.offset_bytes + access.length_bytes;
    if (access.offset_bytes < previous_end && previous.mode != access.mode) {
      result.requires_general_normalizer = true;
      result.ranges.clear();
      return result;
    }
    if (previous.mode == access.mode && access.offset_bytes <= previous_end) {
      previous.length_bytes = std::max(previous_end, access_end) - previous.offset_bytes;
      continue;
    }
    result.ranges.push_back(access);
  }
  return result;
}

[[nodiscard]] bool collect_chunks(const std::span<const residency::AccessRange> accesses,
                                  const std::uint64_t chunk_bytes,
                                  std::vector<residency::ChunkKey>& chunks) {
  for (const residency::AccessRange& access : accesses) {
    const std::uint64_t first_chunk = access.offset_bytes / chunk_bytes;
    const std::uint64_t last_chunk = (access.offset_bytes + access.length_bytes - 1U) / chunk_bytes;
    for (std::uint64_t chunk = first_chunk;; ++chunk) {
      chunks.push_back({access.allocation_id, chunk});
      if (chunk == last_chunk) {
        break;
      }
      if (chunk == std::numeric_limits<std::uint64_t>::max()) {
        return false;
      }
    }
  }
  std::sort(chunks.begin(), chunks.end());
  chunks.erase(std::unique(chunks.begin(), chunks.end()), chunks.end());
  return true;
}

struct GeometryAssessment {
  PlanError error = PlanError::none;
  WorkingSetError working_set_error = WorkingSetError::none;
  std::uint64_t maximum_resident_bytes = 0;
  struct TileSummary {
    WorkingSetError error = WorkingSetError::none;
    std::uint64_t resident_bytes = 0;
    std::uint64_t chunk_count = 0;
    std::uint64_t access_range_count = 0;

    [[nodiscard]] explicit operator bool() const noexcept {
      return error == WorkingSetError::none;
    }
  };
  std::vector<TileSummary> tiles;
};

struct AxisTile {
  std::uint64_t begin = 0;
  std::uint64_t count = 0;
};

[[nodiscard]] std::vector<AxisTile> axis_tiles(const std::uint64_t total,
                                               const std::uint64_t tile) {
  std::vector<AxisTile> result;
  result.reserve(static_cast<std::size_t>(ceil_divide(total, tile)));
  for (std::uint64_t begin = 0; begin < total;) {
    const std::uint64_t count = std::min(tile, total - begin);
    result.push_back({begin, count});
    begin += count;
  }
  return result;
}

[[nodiscard]] GeometryAssessment::TileSummary
summarize_matrix_rectangle(const MatrixView& matrix, const MatrixOperation operation,
                           const std::uint64_t logical_row, const std::uint64_t logical_rows,
                           const std::uint64_t logical_column, const std::uint64_t logical_columns,
                           const std::uint64_t chunk_bytes,
                           const std::uint64_t maximum_ranges) noexcept {
  GeometryAssessment::TileSummary result;
  const auto [operated_rows, operated_columns] = operated_shape(matrix, operation);
  std::uint64_t logical_row_end = 0;
  std::uint64_t logical_column_end = 0;
  if (chunk_bytes == 0 || maximum_ranges == 0 || logical_rows == 0 || logical_columns == 0 ||
      !checked_add(logical_row, logical_rows, logical_row_end) ||
      !checked_add(logical_column, logical_columns, logical_column_end) ||
      logical_row_end > operated_rows || logical_column_end > operated_columns) {
    result.error = WorkingSetError::invalid_tile;
    return result;
  }

  const std::uint64_t physical_row =
      operation == MatrixOperation::none ? logical_row : logical_column;
  const std::uint64_t physical_rows =
      operation == MatrixOperation::none ? logical_rows : logical_columns;
  const std::uint64_t physical_column =
      operation == MatrixOperation::none ? logical_column : logical_row;
  const std::uint64_t physical_columns =
      operation == MatrixOperation::none ? logical_columns : logical_rows;
  const std::uint64_t element_bytes = *element_size_bytes(matrix.element_type);
  const std::uint64_t segments =
      matrix.layout == MatrixLayout::row_major ? physical_rows : physical_columns;
  const std::uint64_t segment_elements =
      matrix.layout == MatrixLayout::row_major ? physical_columns : physical_rows;

  std::uint64_t segment_bytes = 0;
  std::uint64_t segment_stride_bytes = 0;
  if (!checked_multiply(segment_elements, element_bytes, segment_bytes) ||
      !checked_multiply(matrix.leading_dimension, element_bytes, segment_stride_bytes)) {
    result.error = WorkingSetError::range_overflow;
    return result;
  }
  const bool contiguous = segment_bytes == segment_stride_bytes;
  result.access_range_count = contiguous ? 1 : segments;
  if (result.access_range_count > maximum_ranges) {
    result.error = WorkingSetError::too_many_ranges;
    return result;
  }

  const std::uint64_t major =
      matrix.layout == MatrixLayout::row_major ? physical_row : physical_column;
  const std::uint64_t minor =
      matrix.layout == MatrixLayout::row_major ? physical_column : physical_row;
  std::uint64_t element_index = 0;
  std::uint64_t first_byte = 0;
  if (!checked_multiply(major, matrix.leading_dimension, element_index) ||
      !checked_add(element_index, minor, element_index) ||
      !checked_multiply(element_index, element_bytes, first_byte) ||
      !checked_add(matrix.offset_bytes, first_byte, first_byte)) {
    result.error = WorkingSetError::range_overflow;
    return result;
  }

  std::uint64_t last_segment_delta = 0;
  std::uint64_t last_segment_begin = 0;
  std::uint64_t final_byte = 0;
  if (!checked_multiply(segments - 1U, segment_stride_bytes, last_segment_delta) ||
      !checked_add(first_byte, last_segment_delta, last_segment_begin) ||
      !checked_add(last_segment_begin, segment_bytes - 1U, final_byte) ||
      final_byte >= matrix.allocation_bytes) {
    result.error = WorkingSetError::range_overflow;
    return result;
  }

  if (contiguous) {
    result.chunk_count = final_byte / chunk_bytes - first_byte / chunk_bytes + 1U;
  } else {
    std::uint64_t segment_begin = first_byte;
    std::uint64_t covered_last_chunk = 0;
    bool has_covered_chunk = false;
    for (std::uint64_t segment = 0; segment < segments; ++segment) {
      std::uint64_t segment_end = 0;
      if (!checked_add(segment_begin, segment_bytes - 1U, segment_end)) {
        result.error = WorkingSetError::range_overflow;
        return result;
      }
      const std::uint64_t first_chunk = segment_begin / chunk_bytes;
      const std::uint64_t last_chunk = segment_end / chunk_bytes;
      std::uint64_t additional_chunks = 0;
      if (!has_covered_chunk || first_chunk > covered_last_chunk) {
        additional_chunks = last_chunk - first_chunk + 1U;
      } else if (last_chunk > covered_last_chunk) {
        additional_chunks = last_chunk - covered_last_chunk;
      }
      if (!checked_add(result.chunk_count, additional_chunks, result.chunk_count)) {
        result.error = WorkingSetError::resident_size_overflow;
        return result;
      }
      covered_last_chunk = std::max(covered_last_chunk, last_chunk);
      has_covered_chunk = true;
      if (segment + 1U != segments &&
          !checked_add(segment_begin, segment_stride_bytes, segment_begin)) {
        result.error = WorkingSetError::range_overflow;
        return result;
      }
    }
  }
  if (!checked_multiply(result.chunk_count, chunk_bytes, result.resident_bytes)) {
    result.error = WorkingSetError::resident_size_overflow;
  }
  return result;
}

[[nodiscard]] GemmTile make_tile(const GemmProblem& problem, const TileGeometry& geometry,
                                 const std::uint64_t m_begin, const std::uint64_t n_begin,
                                 const std::uint64_t k_begin, const std::uint64_t sequence) {
  GemmTile tile;
  tile.sequence = sequence;
  tile.m_begin = m_begin;
  tile.n_begin = n_begin;
  tile.k_begin = k_begin;
  tile.m_count = std::min(geometry.m, problem.m - m_begin);
  tile.n_count = std::min(geometry.n, problem.n - n_begin);
  tile.k_count = std::min(geometry.k, problem.k - k_begin);
  tile.beta_mode = k_begin == 0 ? BetaMode::user_beta : BetaMode::accumulate_one;
  tile.c_access_mode = k_begin == 0 && problem.beta == 0.0 ? residency::AccessMode::write_only
                                                           : residency::AccessMode::read_write;
  return tile;
}

template <typename Visitor>
[[nodiscard]] bool visit_tiles(const GemmProblem& problem, const TileGeometry& geometry,
                               Visitor&& visitor) {
  std::uint64_t sequence = 0;
  for (std::uint64_t m_begin = 0; m_begin < problem.m;) {
    const std::uint64_t m_count = std::min(geometry.m, problem.m - m_begin);
    for (std::uint64_t n_begin = 0; n_begin < problem.n;) {
      const std::uint64_t n_count = std::min(geometry.n, problem.n - n_begin);
      for (std::uint64_t k_begin = 0; k_begin < problem.k;) {
        const std::uint64_t k_count = std::min(geometry.k, problem.k - k_begin);
        if (!visitor(make_tile(problem, geometry, m_begin, n_begin, k_begin, sequence++))) {
          return false;
        }
        k_begin += k_count;
      }
      n_begin += n_count;
    }
    m_begin += m_count;
  }
  return true;
}

[[nodiscard]] GeometryAssessment assess_geometry(const GemmProblem& problem,
                                                 const PlannerConfig& config,
                                                 const TileGeometry& geometry,
                                                 PlannerStatistics& statistics) {
  ++statistics.geometry_assessments;
  const std::optional<std::uint64_t> count = tile_count(problem, geometry);
  if (!count.has_value()) {
    GeometryAssessment failure;
    failure.error = PlanError::tile_count_overflow;
    return failure;
  }
  if (*count > config.maximum_tiles) {
    GeometryAssessment failure;
    failure.error = PlanError::too_many_tiles;
    return failure;
  }
  if (*count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    GeometryAssessment failure;
    failure.error = PlanError::tile_count_overflow;
    return failure;
  }

  GeometryAssessment assessment;
  assessment.tiles.reserve(static_cast<std::size_t>(*count));
  const bool distinct_allocations = problem.a.allocation_id != problem.b.allocation_id &&
                                    problem.a.allocation_id != problem.c.allocation_id &&
                                    problem.b.allocation_id != problem.c.allocation_id;
  if (!distinct_allocations) {
    const bool completed = visit_tiles(problem, geometry, [&](const GemmTile& tile) {
      ++statistics.legacy_tile_enumerations;
      const WorkingSetResult working_set = enumerate_tile_working_set(
          problem, tile, config.chunk_bytes, config.maximum_access_ranges_per_tile);
      if (!working_set) {
        assessment.error = PlanError::working_set_enumeration_failed;
        assessment.working_set_error = working_set.error;
        return false;
      }
      GeometryAssessment::TileSummary summary;
      summary.resident_bytes = working_set.resident_bytes;
      summary.chunk_count = static_cast<std::uint64_t>(working_set.chunks.size());
      summary.access_range_count = static_cast<std::uint64_t>(working_set.accesses.size());
      assessment.maximum_resident_bytes =
          std::max(assessment.maximum_resident_bytes, summary.resident_bytes);
      assessment.tiles.push_back(summary);
      return true;
    });
    if (!completed && assessment.error == PlanError::none) {
      assessment.error = PlanError::working_set_enumeration_failed;
    }
    return assessment;
  }

  const std::vector<AxisTile> m_tiles = axis_tiles(problem.m, geometry.m);
  const std::vector<AxisTile> n_tiles = axis_tiles(problem.n, geometry.n);
  const std::vector<AxisTile> k_tiles = axis_tiles(problem.k, geometry.k);
  std::vector<GeometryAssessment::TileSummary> a_summaries(m_tiles.size() * k_tiles.size());
  std::vector<GeometryAssessment::TileSummary> b_summaries(k_tiles.size() * n_tiles.size());
  std::vector<GeometryAssessment::TileSummary> c_summaries(m_tiles.size() * n_tiles.size());
  const auto summarize = [&](const MatrixView& matrix, const MatrixOperation operation,
                             const AxisTile& rows, const AxisTile& columns) {
    ++statistics.matrix_rectangle_summaries;
    return summarize_matrix_rectangle(matrix, operation, rows.begin, rows.count, columns.begin,
                                      columns.count, config.chunk_bytes,
                                      config.maximum_access_ranges_per_tile);
  };
  const auto remember_error = [&](const GeometryAssessment::TileSummary& summary) {
    if (!summary && assessment.error == PlanError::none) {
      assessment.error = PlanError::working_set_enumeration_failed;
      assessment.working_set_error = summary.error;
    }
  };

  for (std::size_t m_index = 0; m_index < m_tiles.size() && assessment.error == PlanError::none;
       ++m_index) {
    for (std::size_t k_index = 0; k_index < k_tiles.size() && assessment.error == PlanError::none;
         ++k_index) {
      GeometryAssessment::TileSummary& summary = a_summaries[m_index * k_tiles.size() + k_index];
      summary = summarize(problem.a, problem.a_operation, m_tiles[m_index], k_tiles[k_index]);
      remember_error(summary);
    }
  }
  for (std::size_t k_index = 0; k_index < k_tiles.size() && assessment.error == PlanError::none;
       ++k_index) {
    for (std::size_t n_index = 0; n_index < n_tiles.size() && assessment.error == PlanError::none;
         ++n_index) {
      GeometryAssessment::TileSummary& summary = b_summaries[k_index * n_tiles.size() + n_index];
      summary = summarize(problem.b, problem.b_operation, k_tiles[k_index], n_tiles[n_index]);
      remember_error(summary);
    }
  }
  for (std::size_t m_index = 0; m_index < m_tiles.size() && assessment.error == PlanError::none;
       ++m_index) {
    for (std::size_t n_index = 0; n_index < n_tiles.size() && assessment.error == PlanError::none;
         ++n_index) {
      GeometryAssessment::TileSummary& summary = c_summaries[m_index * n_tiles.size() + n_index];
      summary = summarize(problem.c, MatrixOperation::none, m_tiles[m_index], n_tiles[n_index]);
      remember_error(summary);
    }
  }
  if (assessment.error != PlanError::none) {
    return assessment;
  }

  for (std::size_t m_index = 0; m_index < m_tiles.size(); ++m_index) {
    for (std::size_t n_index = 0; n_index < n_tiles.size(); ++n_index) {
      for (std::size_t k_index = 0; k_index < k_tiles.size(); ++k_index) {
        ++statistics.tile_combinations_assessed;
        const GeometryAssessment::TileSummary& a = a_summaries[m_index * k_tiles.size() + k_index];
        const GeometryAssessment::TileSummary& b = b_summaries[k_index * n_tiles.size() + n_index];
        const GeometryAssessment::TileSummary& c = c_summaries[m_index * n_tiles.size() + n_index];
        GeometryAssessment::TileSummary combined;
        std::uint64_t partial = 0;
        if (!checked_add(a.chunk_count, b.chunk_count, partial) ||
            !checked_add(partial, c.chunk_count, combined.chunk_count) ||
            !checked_add(a.access_range_count, b.access_range_count, partial) ||
            !checked_add(partial, c.access_range_count, combined.access_range_count) ||
            !checked_multiply(combined.chunk_count, config.chunk_bytes, combined.resident_bytes)) {
          assessment.error = PlanError::working_set_enumeration_failed;
          assessment.working_set_error = WorkingSetError::resident_size_overflow;
          assessment.tiles.clear();
          return assessment;
        }
        if (combined.access_range_count > config.maximum_access_ranges_per_tile) {
          assessment.error = PlanError::working_set_enumeration_failed;
          assessment.working_set_error = WorkingSetError::too_many_ranges;
          assessment.tiles.clear();
          return assessment;
        }
        assessment.maximum_resident_bytes =
            std::max(assessment.maximum_resident_bytes, combined.resident_bytes);
        assessment.tiles.push_back(combined);
      }
    }
  }
  if (assessment.tiles.size() != static_cast<std::size_t>(*count)) {
    assessment.error = PlanError::working_set_enumeration_failed;
    assessment.working_set_error = WorkingSetError::resident_size_overflow;
  }
  return assessment;
}

[[nodiscard]] std::uint64_t geometry_volume(const TileGeometry& geometry) noexcept {
  std::uint64_t volume = 0;
  std::uint64_t mn = 0;
  if (!checked_multiply(geometry.m, geometry.n, mn) || !checked_multiply(mn, geometry.k, volume)) {
    return std::numeric_limits<std::uint64_t>::max();
  }
  return volume;
}

[[nodiscard]] std::uint16_t pointer_alignment(const std::uint64_t offset) noexcept {
  if (offset == 0) {
    return static_cast<std::uint16_t>(maximum_algorithm_alignment);
  }
  std::uint64_t alignment = 1;
  while (alignment < maximum_algorithm_alignment && (offset % (alignment * 2U)) == 0) {
    alignment *= 2U;
  }
  return static_cast<std::uint16_t>(alignment);
}

[[nodiscard]] std::uint64_t tile_origin(const MatrixView& matrix, const MatrixOperation operation,
                                        const std::uint64_t logical_row,
                                        const std::uint64_t logical_column) noexcept {
  const std::uint64_t physical_row =
      operation == MatrixOperation::none ? logical_row : logical_column;
  const std::uint64_t physical_column =
      operation == MatrixOperation::none ? logical_column : logical_row;
  const std::uint64_t major =
      matrix.layout == MatrixLayout::row_major ? physical_row : physical_column;
  const std::uint64_t minor =
      matrix.layout == MatrixLayout::row_major ? physical_column : physical_row;
  const std::uint64_t element_index = major * matrix.leading_dimension + minor;
  return matrix.offset_bytes + element_index * *element_size_bytes(matrix.element_type);
}

} // namespace

std::string_view matrix_layout_name(const MatrixLayout layout) noexcept {
  switch (layout) {
  case MatrixLayout::row_major:
    return "row_major";
  case MatrixLayout::column_major:
    return "column_major";
  }
  return "invalid";
}

std::string_view matrix_operation_name(const MatrixOperation operation) noexcept {
  switch (operation) {
  case MatrixOperation::none:
    return "none";
  case MatrixOperation::transpose:
    return "transpose";
  }
  return "invalid";
}

std::string_view element_type_name(const ElementType type) noexcept {
  switch (type) {
  case ElementType::fp16:
    return "fp16";
  case ElementType::bf16:
    return "bf16";
  case ElementType::fp32:
    return "fp32";
  case ElementType::fp64:
    return "fp64";
  }
  return "invalid";
}

std::string_view compute_mode_name(const ComputeMode mode) noexcept {
  switch (mode) {
  case ComputeMode::strict_fp32:
    return "strict_fp32";
  case ComputeMode::fast_tf32:
    return "fast_tf32";
  case ComputeMode::fp64:
    return "fp64";
  }
  return "invalid";
}

std::optional<std::uint64_t> element_size_bytes(const ElementType type) noexcept {
  switch (type) {
  case ElementType::fp16:
  case ElementType::bf16:
    return 2;
  case ElementType::fp32:
    return 4;
  case ElementType::fp64:
    return 8;
  }
  return std::nullopt;
}

std::string_view problem_error_name(const ProblemError error) noexcept {
  switch (error) {
  case ProblemError::none:
    return "none";
  case ProblemError::zero_dimension:
    return "zero_dimension";
  case ProblemError::invalid_allocation:
    return "invalid_allocation";
  case ProblemError::invalid_layout:
    return "invalid_layout";
  case ProblemError::invalid_operation:
    return "invalid_operation";
  case ProblemError::invalid_element_type:
    return "invalid_element_type";
  case ProblemError::invalid_compute_mode:
    return "invalid_compute_mode";
  case ProblemError::dimension_mismatch:
    return "dimension_mismatch";
  case ProblemError::leading_dimension_too_small:
    return "leading_dimension_too_small";
  case ProblemError::storage_overflow:
    return "storage_overflow";
  case ProblemError::storage_out_of_bounds:
    return "storage_out_of_bounds";
  case ProblemError::unsupported_type_combination:
    return "unsupported_type_combination";
  case ProblemError::output_aliases_input:
    return "output_aliases_input";
  case ProblemError::integer_limit_exceeded:
    return "integer_limit_exceeded";
  }
  return "invalid";
}

ProblemValidation validate_problem(const GemmProblem& problem) noexcept {
  if (problem.m == 0 || problem.n == 0 || problem.k == 0) {
    return {ProblemError::zero_dimension};
  }
  if (!is_valid_operation(problem.a_operation)) {
    return {ProblemError::invalid_operation, 'A'};
  }
  if (!is_valid_operation(problem.b_operation)) {
    return {ProblemError::invalid_operation, 'B'};
  }
  if (!is_valid_compute_mode(problem.compute_mode)) {
    return {ProblemError::invalid_compute_mode};
  }
  for (const auto [matrix, operand] : std::array{
           std::pair{&problem.a, 'A'}, std::pair{&problem.b, 'B'}, std::pair{&problem.c, 'C'}}) {
    const ProblemValidation validation = validate_matrix(*matrix, operand);
    if (!validation) {
      return validation;
    }
  }
  if ((problem.a.allocation_id == problem.b.allocation_id &&
       problem.a.allocation_bytes != problem.b.allocation_bytes) ||
      (problem.a.allocation_id == problem.c.allocation_id &&
       problem.a.allocation_bytes != problem.c.allocation_bytes) ||
      (problem.b.allocation_id == problem.c.allocation_id &&
       problem.b.allocation_bytes != problem.c.allocation_bytes)) {
    return {ProblemError::invalid_allocation};
  }

  if (operated_shape(problem.a, problem.a_operation) != std::pair{problem.m, problem.k} ||
      operated_shape(problem.b, problem.b_operation) != std::pair{problem.k, problem.n} ||
      std::pair{problem.c.rows, problem.c.columns} != std::pair{problem.m, problem.n}) {
    return {ProblemError::dimension_mismatch};
  }
  if (problem.a.element_type != problem.b.element_type) {
    return {ProblemError::unsupported_type_combination};
  }

  bool supported_types = false;
  switch (problem.a.element_type) {
  case ElementType::fp16:
  case ElementType::bf16:
    supported_types = (problem.c.element_type == problem.a.element_type ||
                       problem.c.element_type == ElementType::fp32) &&
                      problem.compute_mode == ComputeMode::strict_fp32;
    break;
  case ElementType::fp32:
    supported_types = problem.c.element_type == ElementType::fp32 &&
                      (problem.compute_mode == ComputeMode::strict_fp32 ||
                       problem.compute_mode == ComputeMode::fast_tf32);
    break;
  case ElementType::fp64:
    supported_types =
        problem.c.element_type == ElementType::fp64 && problem.compute_mode == ComputeMode::fp64;
    break;
  }
  if (!supported_types) {
    return {ProblemError::unsupported_type_combination};
  }

  const std::optional<MatrixBounds> a_bounds = matrix_bounds(problem.a);
  const std::optional<MatrixBounds> b_bounds = matrix_bounds(problem.b);
  const std::optional<MatrixBounds> c_bounds = matrix_bounds(problem.c);
  if (!a_bounds.has_value() || !b_bounds.has_value() || !c_bounds.has_value()) {
    return {ProblemError::storage_overflow};
  }
  const auto overlaps = [](const MatrixBounds& left, const MatrixBounds& right) {
    return left.begin < right.end && right.begin < left.end;
  };
  if ((problem.a.allocation_id == problem.c.allocation_id && overlaps(*a_bounds, *c_bounds)) ||
      (problem.b.allocation_id == problem.c.allocation_id && overlaps(*b_bounds, *c_bounds))) {
    return {ProblemError::output_aliases_input, 'C'};
  }
  return {};
}

std::string_view working_set_error_name(const WorkingSetError error) noexcept {
  switch (error) {
  case WorkingSetError::none:
    return "none";
  case WorkingSetError::invalid_problem:
    return "invalid_problem";
  case WorkingSetError::invalid_tile:
    return "invalid_tile";
  case WorkingSetError::invalid_chunk_size:
    return "invalid_chunk_size";
  case WorkingSetError::range_overflow:
    return "range_overflow";
  case WorkingSetError::too_many_ranges:
    return "too_many_ranges";
  case WorkingSetError::access_normalization_failed:
    return "access_normalization_failed";
  case WorkingSetError::resident_size_overflow:
    return "resident_size_overflow";
  }
  return "invalid";
}

WorkingSetResult enumerate_tile_working_set(const GemmProblem& problem, const GemmTile& tile,
                                            const std::uint64_t chunk_bytes,
                                            const std::uint64_t maximum_access_ranges) {
  WorkingSetResult result;
  if (!validate_problem(problem)) {
    result.error = WorkingSetError::invalid_problem;
    return result;
  }
  if (chunk_bytes == 0) {
    result.error = WorkingSetError::invalid_chunk_size;
    return result;
  }
  if (maximum_access_ranges == 0 || tile.m_count == 0 || tile.n_count == 0 || tile.k_count == 0 ||
      tile.m_begin >= problem.m || tile.n_begin >= problem.n || tile.k_begin >= problem.k ||
      tile.m_count > problem.m - tile.m_begin || tile.n_count > problem.n - tile.n_begin ||
      tile.k_count > problem.k - tile.k_begin) {
    result.error = WorkingSetError::invalid_tile;
    return result;
  }

  std::vector<residency::AccessRange> raw_accesses;
  WorkingSetError error = append_matrix_rectangle(
      problem.a, problem.a_operation, tile.m_begin, tile.m_count, tile.k_begin, tile.k_count,
      residency::AccessMode::read, maximum_access_ranges, raw_accesses);
  if (error == WorkingSetError::none) {
    error = append_matrix_rectangle(problem.b, problem.b_operation, tile.k_begin, tile.k_count,
                                    tile.n_begin, tile.n_count, residency::AccessMode::read,
                                    maximum_access_ranges, raw_accesses);
  }
  if (error == WorkingSetError::none) {
    const residency::AccessMode c_access_mode = tile.k_begin == 0 && problem.beta == 0.0
                                                    ? residency::AccessMode::write_only
                                                    : residency::AccessMode::read_write;
    error = append_matrix_rectangle(problem.c, MatrixOperation::none, tile.m_begin, tile.m_count,
                                    tile.n_begin, tile.n_count, c_access_mode,
                                    maximum_access_ranges, raw_accesses);
  }
  if (error != WorkingSetError::none) {
    result.error = error;
    return result;
  }

  FastNormalizationResult fast = fast_normalize_accesses(raw_accesses);
  if (!fast.requires_general_normalizer) {
    result.accesses = std::move(fast.ranges);
    if (!collect_chunks(result.accesses, chunk_bytes, result.chunks)) {
      result.error = WorkingSetError::resident_size_overflow;
      result.accesses.clear();
      result.chunks.clear();
      return result;
    }
  } else {
    const residency::AccessPlanResult normalized = residency::normalize_and_split_accesses(
        allocation_layouts(problem), raw_accesses, chunk_bytes);
    if (!normalized) {
      result.error = WorkingSetError::access_normalization_failed;
      return result;
    }
    result.accesses = normalized.normalized_ranges;
    result.chunks.reserve(normalized.chunks.size());
    for (const residency::ChunkAccessPlan& chunk : normalized.chunks) {
      result.chunks.push_back(chunk.key);
    }
  }
  if (!checked_multiply(static_cast<std::uint64_t>(result.chunks.size()), chunk_bytes,
                        result.resident_bytes)) {
    result.error = WorkingSetError::resident_size_overflow;
    result.accesses.clear();
    result.chunks.clear();
  }
  return result;
}

AlgorithmSignature make_algorithm_signature(const GemmProblem& problem, const GemmTile& tile,
                                            const std::uint64_t workspace_limit_bytes) noexcept {
  AlgorithmSignature signature;
  signature.a_layout = problem.a.layout;
  signature.b_layout = problem.b.layout;
  signature.c_layout = problem.c.layout;
  signature.a_operation = problem.a_operation;
  signature.b_operation = problem.b_operation;
  signature.a_leading_dimension = problem.a.leading_dimension;
  signature.b_leading_dimension = problem.b.leading_dimension;
  signature.c_leading_dimension = problem.c.leading_dimension;
  signature.a_alignment_bytes =
      pointer_alignment(tile_origin(problem.a, problem.a_operation, tile.m_begin, tile.k_begin));
  signature.b_alignment_bytes =
      pointer_alignment(tile_origin(problem.b, problem.b_operation, tile.k_begin, tile.n_begin));
  signature.c_alignment_bytes =
      pointer_alignment(tile_origin(problem.c, MatrixOperation::none, tile.m_begin, tile.n_begin));
  signature.workspace = {problem.a.element_type, problem.b.element_type, problem.c.element_type,
                         problem.compute_mode,   tile.m_count,           tile.n_count,
                         tile.k_count,           workspace_limit_bytes};
  return signature;
}

std::string_view plan_error_name(const PlanError error) noexcept {
  switch (error) {
  case PlanError::none:
    return "none";
  case PlanError::invalid_problem:
    return "invalid_problem";
  case PlanError::invalid_chunk_size:
    return "invalid_chunk_size";
  case PlanError::invalid_cache_target:
    return "invalid_cache_target";
  case PlanError::workspace_exceeds_cache:
    return "workspace_exceeds_cache";
  case PlanError::invalid_preferred_geometry:
    return "invalid_preferred_geometry";
  case PlanError::tile_count_overflow:
    return "tile_count_overflow";
  case PlanError::too_many_tiles:
    return "too_many_tiles";
  case PlanError::working_set_too_large:
    return "working_set_too_large";
  case PlanError::working_set_enumeration_failed:
    return "working_set_enumeration_failed";
  }
  return "invalid";
}

GemmPlan make_plan(const GemmProblem& problem, const PlannerConfig& config) {
  GemmPlan plan;
  const auto fail = [&](const PlanError error) {
    plan.error = error;
    plan.tiles.clear();
    return plan;
  };

  plan.problem_validation = validate_problem(problem);
  if (!plan.problem_validation) {
    return fail(PlanError::invalid_problem);
  }
  if (config.chunk_bytes == 0) {
    return fail(PlanError::invalid_chunk_size);
  }
  if (config.cache_target_bytes < config.chunk_bytes) {
    return fail(PlanError::invalid_cache_target);
  }
  if (config.workspace_bytes >= config.cache_target_bytes) {
    return fail(PlanError::workspace_exceeds_cache);
  }
  if (config.preferred_geometry.m == 0 || config.preferred_geometry.n == 0 ||
      config.preferred_geometry.k == 0 || config.maximum_tiles == 0 ||
      config.maximum_access_ranges_per_tile == 0) {
    return fail(PlanError::invalid_preferred_geometry);
  }

  plan.cache_bytes_available_to_chunks =
      ((config.cache_target_bytes - config.workspace_bytes) / config.chunk_bytes) *
      config.chunk_bytes;
  if (plan.cache_bytes_available_to_chunks == 0) {
    return fail(PlanError::workspace_exceeds_cache);
  }

  TileGeometry geometry{std::min(problem.m, config.preferred_geometry.m),
                        std::min(problem.n, config.preferred_geometry.n),
                        std::min(problem.k, config.preferred_geometry.k)};
  const std::optional<std::uint64_t> initial_tile_count = tile_count(problem, geometry);
  if (!initial_tile_count.has_value()) {
    return fail(PlanError::tile_count_overflow);
  }
  if (*initial_tile_count > config.maximum_tiles) {
    return fail(PlanError::too_many_tiles);
  }

  GeometryAssessment assessment = assess_geometry(problem, config, geometry, plan.statistics);
  if (assessment.error != PlanError::none) {
    plan.working_set_error = assessment.working_set_error;
    return fail(assessment.error);
  }

  const std::uint64_t alignment = tile_alignment(problem);
  while (assessment.maximum_resident_bytes > plan.cache_bytes_available_to_chunks) {
    struct Candidate {
      TileGeometry geometry;
      GeometryAssessment assessment;
    };
    std::vector<Candidate> candidates;
    const std::array reduced{
        TileGeometry{reduced_dimension(geometry.m, alignment), geometry.n, geometry.k},
        TileGeometry{geometry.m, reduced_dimension(geometry.n, alignment), geometry.k},
        TileGeometry{geometry.m, geometry.n, reduced_dimension(geometry.k, alignment)},
    };
    for (const TileGeometry& candidate_geometry : reduced) {
      if (candidate_geometry == geometry ||
          std::find_if(candidates.begin(), candidates.end(), [&](const Candidate& candidate) {
            return candidate.geometry == candidate_geometry;
          }) != candidates.end()) {
        continue;
      }
      GeometryAssessment candidate_assessment =
          assess_geometry(problem, config, candidate_geometry, plan.statistics);
      if (candidate_assessment.error == PlanError::none) {
        candidates.push_back({candidate_geometry, candidate_assessment});
      }
    }
    if (candidates.empty()) {
      return fail(PlanError::working_set_too_large);
    }
    const bool has_fitting_candidate =
        std::any_of(candidates.begin(), candidates.end(), [&](const Candidate& candidate) {
          return candidate.assessment.maximum_resident_bytes <=
                 plan.cache_bytes_available_to_chunks;
        });
    const auto best = std::min_element(
        candidates.begin(), candidates.end(), [&](const Candidate& left, const Candidate& right) {
          if (has_fitting_candidate) {
            const bool left_fits =
                left.assessment.maximum_resident_bytes <= plan.cache_bytes_available_to_chunks;
            const bool right_fits =
                right.assessment.maximum_resident_bytes <= plan.cache_bytes_available_to_chunks;
            if (left_fits != right_fits) {
              return left_fits;
            }
            return std::tuple{std::numeric_limits<std::uint64_t>::max() -
                                  geometry_volume(left.geometry),
                              left.assessment.maximum_resident_bytes, left.geometry.m,
                              left.geometry.n, left.geometry.k} <
                   std::tuple{std::numeric_limits<std::uint64_t>::max() -
                                  geometry_volume(right.geometry),
                              right.assessment.maximum_resident_bytes, right.geometry.m,
                              right.geometry.n, right.geometry.k};
          }
          return std::tuple{left.assessment.maximum_resident_bytes,
                            std::numeric_limits<std::uint64_t>::max() -
                                geometry_volume(left.geometry),
                            left.geometry.m, left.geometry.n, left.geometry.k} <
                 std::tuple{right.assessment.maximum_resident_bytes,
                            std::numeric_limits<std::uint64_t>::max() -
                                geometry_volume(right.geometry),
                            right.geometry.m, right.geometry.n, right.geometry.k};
        });
    geometry = best->geometry;
    assessment = best->assessment;
  }

  const std::uint64_t count = *tile_count(problem, geometry);
  plan.tiles.reserve(static_cast<std::size_t>(count));
  const bool completed = visit_tiles(problem, geometry, [&](GemmTile tile) {
    if (tile.sequence >= assessment.tiles.size()) {
      plan.working_set_error = WorkingSetError::resident_size_overflow;
      return false;
    }
    const GeometryAssessment::TileSummary& working_set = assessment.tiles[tile.sequence];
    if (!working_set) {
      plan.working_set_error = working_set.error;
      return false;
    }
    tile.resident_bytes = working_set.resident_bytes;
    tile.chunk_count = working_set.chunk_count;
    tile.access_range_count = working_set.access_range_count;
    tile.algorithm_signature = make_algorithm_signature(problem, tile, config.workspace_bytes);
    plan.maximum_resident_bytes = std::max(plan.maximum_resident_bytes, tile.resident_bytes);
    plan.tiles.push_back(std::move(tile));
    return true;
  });
  if (!completed || plan.tiles.size() != static_cast<std::size_t>(count)) {
    return fail(PlanError::working_set_enumeration_failed);
  }
  plan.geometry = geometry;
  return plan;
}

} // namespace xvram::gemm
