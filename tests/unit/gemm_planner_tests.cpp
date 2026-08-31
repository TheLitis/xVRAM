#include "gemm/planner.hpp"
#include "platform/cublas/cublas_api.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string_view>

namespace {

int failures = 0;

void check(const bool condition, const char* expression, const int line) {
  if (!condition) {
    std::cerr << "line " << line << ": check failed: " << expression << '\n';
    ++failures;
  }
}

#define CHECK(expression) check(static_cast<bool>(expression), #expression, __LINE__)

using xvram::gemm::ComputeMode;
using xvram::gemm::ElementType;
using xvram::gemm::GemmProblem;
using xvram::gemm::MatrixLayout;
using xvram::gemm::MatrixOperation;
using xvram::gemm::MatrixView;
using xvram::residency::AllocationId;

[[nodiscard]] MatrixView view(const std::uint64_t id, const std::uint64_t rows,
                              const std::uint64_t columns, const ElementType type,
                              const MatrixLayout layout = MatrixLayout::row_major,
                              std::uint64_t leading_dimension = 0, const std::uint64_t offset = 0) {
  if (leading_dimension == 0) {
    leading_dimension = layout == MatrixLayout::row_major ? columns : rows;
  }
  return {AllocationId{id}, 1ULL << 30U, offset, rows, columns, leading_dimension, layout, type};
}

[[nodiscard]] GemmProblem problem(const std::uint64_t m = 8, const std::uint64_t n = 8,
                                  const std::uint64_t k = 8) {
  return {m,
          n,
          k,
          view(1, m, k, ElementType::fp32),
          view(2, k, n, ElementType::fp32),
          view(3, m, n, ElementType::fp32),
          MatrixOperation::none,
          MatrixOperation::none,
          ComputeMode::fast_tf32,
          1.0,
          0.0};
}

void validation_tests() {
  using namespace xvram::gemm;

  GemmProblem valid = problem();
  CHECK(validate_problem(valid));
  CHECK(element_size_bytes(ElementType::fp16) == 2);
  CHECK(element_size_bytes(ElementType::bf16) == 2);
  CHECK(element_size_bytes(ElementType::fp32) == 4);
  CHECK(element_size_bytes(ElementType::fp64) == 8);
  CHECK(matrix_layout_name(MatrixLayout::column_major) == std::string_view{"column_major"});
  CHECK(compute_mode_name(ComputeMode::fast_tf32) == std::string_view{"fast_tf32"});

  GemmProblem fp16 = problem();
  fp16.a.element_type = ElementType::fp16;
  fp16.b.element_type = ElementType::fp16;
  fp16.c.element_type = ElementType::fp16;
  fp16.compute_mode = ComputeMode::strict_fp32;
  CHECK(validate_problem(fp16));
  fp16.c.element_type = ElementType::fp32;
  CHECK(validate_problem(fp16));

  GemmProblem bf16 = fp16;
  bf16.a.element_type = ElementType::bf16;
  bf16.b.element_type = ElementType::bf16;
  bf16.c.element_type = ElementType::bf16;
  CHECK(validate_problem(bf16));
  bf16.c.element_type = ElementType::fp16;
  CHECK(validate_problem(bf16).error == ProblemError::unsupported_type_combination);

  GemmProblem fp64 = problem();
  fp64.a.element_type = ElementType::fp64;
  fp64.b.element_type = ElementType::fp64;
  fp64.c.element_type = ElementType::fp64;
  fp64.compute_mode = ComputeMode::fp64;
  CHECK(validate_problem(fp64));
  fp64.compute_mode = ComputeMode::fast_tf32;
  CHECK(validate_problem(fp64).error == ProblemError::unsupported_type_combination);

  GemmProblem transpose = problem(4, 5, 6);
  transpose.a = view(1, 6, 4, ElementType::fp32, MatrixLayout::column_major);
  transpose.b = view(2, 5, 6, ElementType::fp32, MatrixLayout::row_major);
  transpose.a_operation = MatrixOperation::transpose;
  transpose.b_operation = MatrixOperation::transpose;
  CHECK(validate_problem(transpose));

  GemmProblem bad_leading = problem();
  bad_leading.a.leading_dimension = bad_leading.a.columns - 1U;
  const ProblemValidation bad_leading_result = validate_problem(bad_leading);
  CHECK(bad_leading_result.error == ProblemError::leading_dimension_too_small);
  CHECK(bad_leading_result.operand == 'A');

  GemmProblem overflow = problem();
  overflow.a.rows = 2;
  overflow.a.columns = 1;
  overflow.a.leading_dimension = std::numeric_limits<std::uint64_t>::max();
  overflow.m = 2;
  CHECK(validate_problem(overflow).error == ProblemError::integer_limit_exceeded);

  GemmProblem cublas_integer_limit = problem(1, 1, 1);
  cublas_integer_limit.a.leading_dimension =
      static_cast<std::uint64_t>(std::numeric_limits<int>::max()) + 1U;
  CHECK(validate_problem(cublas_integer_limit).error == ProblemError::integer_limit_exceeded);

  GemmProblem out_of_bounds = problem();
  out_of_bounds.a.allocation_bytes = 16;
  CHECK(validate_problem(out_of_bounds).error == ProblemError::storage_out_of_bounds);

  GemmProblem alias = problem();
  alias.c.allocation_id = alias.a.allocation_id;
  CHECK(validate_problem(alias).error == ProblemError::output_aliases_input);
}

void strided_range_tests() {
  using namespace xvram::gemm;
  using xvram::residency::AccessMode;
  using xvram::residency::ChunkKey;

  GemmProblem strided = problem(4, 5, 6);
  strided.a = view(1, 4, 6, ElementType::fp32, MatrixLayout::row_major, 8);
  strided.b = view(2, 6, 5, ElementType::fp32, MatrixLayout::row_major, 5);
  strided.c = view(3, 4, 5, ElementType::fp32, MatrixLayout::row_major, 5);
  CHECK(validate_problem(strided));

  GemmTile tile;
  tile.m_begin = 1;
  tile.m_count = 2;
  tile.n_begin = 1;
  tile.n_count = 2;
  tile.k_begin = 2;
  tile.k_count = 3;
  tile.c_access_mode = AccessMode::write_only;
  const WorkingSetResult working = enumerate_tile_working_set(strided, tile, 32);
  CHECK(working);
  CHECK(working.accesses.size() == 7);
  CHECK(working.chunks.size() == 6);
  CHECK(working.resident_bytes == 192);
  CHECK((working.chunks.front() == ChunkKey{AllocationId{1}, 1}));
  CHECK(working.accesses[0].offset_bytes == 40);
  CHECK(working.accesses[0].length_bytes == 12);
  CHECK(working.accesses[1].offset_bytes == 72);

  GemmProblem transposed = problem(4, 5, 6);
  transposed.a = view(1, 6, 4, ElementType::fp32, MatrixLayout::column_major, 6);
  transposed.a_operation = MatrixOperation::transpose;
  CHECK(validate_problem(transposed));
  const WorkingSetResult transposed_working = enumerate_tile_working_set(transposed, tile, 32);
  CHECK(transposed_working);
  CHECK(transposed_working.accesses[0].offset_bytes == 32);
  CHECK(transposed_working.accesses[0].length_bytes == 12);
  CHECK(transposed_working.accesses[1].offset_bytes == 56);
  CHECK(transposed_working.accesses[1].length_bytes == 12);

  GemmTile invalid = tile;
  invalid.m_count = 99;
  CHECK(enumerate_tile_working_set(strided, invalid, 32).error == WorkingSetError::invalid_tile);
  CHECK(enumerate_tile_working_set(strided, tile, 0).error == WorkingSetError::invalid_chunk_size);
  CHECK(enumerate_tile_working_set(strided, tile, 32, 1).error == WorkingSetError::too_many_ranges);
}

void planner_tests() {
  using namespace xvram::gemm;
  using xvram::residency::AccessMode;

  GemmProblem split_k = problem(8, 8, 5);
  PlannerConfig split_config;
  split_config.chunk_bytes = 64;
  split_config.cache_target_bytes = 1ULL << 20U;
  split_config.workspace_bytes = 4096;
  split_config.preferred_geometry = {8, 8, 2};
  const GemmPlan split_plan = make_plan(split_k, split_config);
  CHECK(split_plan);
  CHECK((split_plan.geometry == TileGeometry{8, 8, 2}));
  CHECK(split_plan.tiles.size() == 3);
  CHECK(split_plan.tiles[0].beta_mode == BetaMode::user_beta);
  CHECK(split_plan.tiles[0].c_access_mode == AccessMode::write_only);
  CHECK(split_plan.tiles[1].beta_mode == BetaMode::accumulate_one);
  CHECK(split_plan.tiles[1].c_access_mode == AccessMode::read_write);
  CHECK(split_plan.tiles[2].k_count == 1);
  CHECK(split_plan.tiles[0].algorithm_signature.workspace.workspace_limit_bytes == 4096);

  GemmProblem low_precision_output = split_k;
  low_precision_output.a.element_type = ElementType::fp16;
  low_precision_output.b.element_type = ElementType::fp16;
  low_precision_output.c.element_type = ElementType::fp16;
  low_precision_output.compute_mode = ComputeMode::strict_fp32;
  const GemmPlan low_precision_plan = make_plan(low_precision_output, split_config);
  CHECK(low_precision_plan);
  CHECK(low_precision_plan.geometry.k == low_precision_output.k);
  CHECK(low_precision_plan.tiles.size() == 1);

  low_precision_output.c.element_type = ElementType::fp32;
  const GemmPlan fp32_accumulator_plan = make_plan(low_precision_output, split_config);
  CHECK(fp32_accumulator_plan);
  CHECK(fp32_accumulator_plan.geometry.k == split_config.preferred_geometry.k);
  CHECK(fp32_accumulator_plan.tiles.size() == split_plan.tiles.size());

  GemmProblem nonzero_beta = split_k;
  nonzero_beta.beta = 2.0;
  const GemmPlan beta_plan = make_plan(nonzero_beta, split_config);
  CHECK(beta_plan);
  CHECK(beta_plan.tiles.front().c_access_mode == AccessMode::read_write);

  GemmProblem tails = problem(10, 9, 7);
  PlannerConfig tail_config = split_config;
  tail_config.preferred_geometry = {4, 4, 4};
  const GemmPlan tail_plan = make_plan(tails, tail_config);
  CHECK(tail_plan);
  CHECK(tail_plan.tiles.size() == 18);
  CHECK(tail_plan.tiles.back().m_count == 2);
  CHECK(tail_plan.tiles.back().n_count == 1);
  CHECK(tail_plan.tiles.back().k_count == 3);
  const GemmPlan repeated = make_plan(tails, tail_config);
  CHECK(repeated);
  CHECK(repeated.geometry == tail_plan.geometry);
  CHECK(repeated.tiles.size() == tail_plan.tiles.size());
  for (std::size_t index = 0; index < repeated.tiles.size(); ++index) {
    CHECK(repeated.tiles[index].m_begin == tail_plan.tiles[index].m_begin);
    CHECK(repeated.tiles[index].n_begin == tail_plan.tiles[index].n_begin);
    CHECK(repeated.tiles[index].k_begin == tail_plan.tiles[index].k_begin);
  }

  GemmProblem shrink = problem(16, 16, 16);
  PlannerConfig shrink_config;
  shrink_config.chunk_bytes = 256;
  shrink_config.cache_target_bytes = 768;
  shrink_config.workspace_bytes = 0;
  shrink_config.preferred_geometry = {16, 16, 16};
  const GemmPlan shrink_plan = make_plan(shrink, shrink_config);
  CHECK(shrink_plan);
  CHECK(shrink_plan.maximum_resident_bytes <= shrink_plan.cache_bytes_available_to_chunks);
  CHECK(shrink_plan.geometry != shrink_config.preferred_geometry);
  for (const GemmTile& planned_tile : shrink_plan.tiles) {
    CHECK(planned_tile.resident_bytes <= shrink_plan.cache_bytes_available_to_chunks);
  }

  shrink_config.cache_target_bytes = 512;
  CHECK(make_plan(shrink, shrink_config).error == PlanError::working_set_too_large);

  PlannerConfig invalid_config = split_config;
  invalid_config.workspace_bytes = invalid_config.cache_target_bytes;
  CHECK(make_plan(split_k, invalid_config).error == PlanError::workspace_exceeds_cache);
  invalid_config = split_config;
  invalid_config.preferred_geometry.m = 0;
  CHECK(make_plan(split_k, invalid_config).error == PlanError::invalid_preferred_geometry);
}

void planner_large_k_regression_tests() {
  using namespace xvram::gemm;

  constexpr std::uint64_t m = 2049;
  constexpr std::uint64_t n = 2049;
  constexpr std::uint64_t k = 32769;
  constexpr std::uint64_t chunk = 64ULL * 1024ULL * 1024ULL;
  GemmProblem large;
  large.m = m;
  large.n = n;
  large.k = k;
  large.a = view(101, m, k, ElementType::fp32, MatrixLayout::row_major, k);
  large.b = view(102, k, n, ElementType::fp32, MatrixLayout::row_major, n);
  large.c = view(103, m, n, ElementType::fp32, MatrixLayout::row_major, n);
  large.compute_mode = ComputeMode::fast_tf32;
  CHECK(validate_problem(large));

  PlannerConfig config;
  config.chunk_bytes = chunk;
  config.cache_target_bytes = 256ULL * 1024ULL * 1024ULL;
  config.workspace_bytes = 4ULL * 1024ULL * 1024ULL;
  config.preferred_geometry = {4096, 4096, 1024};
  config.maximum_tiles = 1'000'000;
  const GemmPlan insufficient = make_plan(large, config);
  CHECK(insufficient.error == PlanError::working_set_too_large);
  CHECK(insufficient.statistics.legacy_tile_enumerations == 0);
  CHECK(insufficient.statistics.geometry_assessments < 128);
  CHECK(insufficient.statistics.matrix_rectangle_summaries < 2'000'000);
  CHECK(insufficient.statistics.tile_combinations_assessed < 50'000'000);

  // Five resident frames are the smallest safe target found for this row-major strided fixture:
  // floor((384 MiB - 4 MiB workspace) / 64 MiB) == 5.
  config.cache_target_bytes = 384ULL * 1024ULL * 1024ULL;
  const GemmPlan fitting = make_plan(large, config);
  CHECK(fitting);
  CHECK((fitting.geometry == TileGeometry{512, 2049, 1024}));
  CHECK(fitting.tiles.size() == 165);
  CHECK(fitting.maximum_resident_bytes == 5U * chunk);
  CHECK(fitting.geometry.k < large.k);
  CHECK(fitting.maximum_resident_bytes <= fitting.cache_bytes_available_to_chunks);
  CHECK(fitting.statistics.legacy_tile_enumerations == 0);
  CHECK(fitting.statistics.geometry_assessments < 32);
  CHECK(fitting.statistics.matrix_rectangle_summaries < 100'000);
  CHECK(fitting.statistics.tile_combinations_assessed < 10'000);

  const auto verify_summary = [&](const std::size_t index) {
    CHECK(index < fitting.tiles.size());
    if (index >= fitting.tiles.size()) {
      return;
    }
    const GemmTile& tile = fitting.tiles[index];
    const WorkingSetResult exact = enumerate_tile_working_set(large, tile, chunk);
    CHECK(exact);
    CHECK(tile.resident_bytes == exact.resident_bytes);
    CHECK(tile.chunk_count == exact.chunks.size());
    CHECK(tile.access_range_count == exact.accesses.size());
  };
  verify_summary(0);
  verify_summary(fitting.tiles.size() / 2U);
  verify_summary(fitting.tiles.size() - 1U);
}

void planner_summary_equivalence_tests() {
  using namespace xvram::gemm;

  constexpr std::array layouts{MatrixLayout::row_major, MatrixLayout::column_major};
  constexpr std::array operations{MatrixOperation::none, MatrixOperation::transpose};
  std::uint64_t next_id = 200;
  for (const MatrixLayout a_layout : layouts) {
    for (const MatrixLayout b_layout : layouts) {
      for (const MatrixLayout c_layout : layouts) {
        for (const MatrixOperation a_operation : operations) {
          for (const MatrixOperation b_operation : operations) {
            constexpr std::uint64_t m = 10;
            constexpr std::uint64_t n = 9;
            constexpr std::uint64_t k = 7;
            const std::uint64_t a_rows = a_operation == MatrixOperation::none ? m : k;
            const std::uint64_t a_columns = a_operation == MatrixOperation::none ? k : m;
            const std::uint64_t b_rows = b_operation == MatrixOperation::none ? k : n;
            const std::uint64_t b_columns = b_operation == MatrixOperation::none ? n : k;
            const auto padded_leading = [](const MatrixLayout layout, const std::uint64_t rows,
                                           const std::uint64_t columns) {
              return (layout == MatrixLayout::row_major ? columns : rows) + 3U;
            };
            GemmProblem input;
            input.m = m;
            input.n = n;
            input.k = k;
            input.a = view(next_id++, a_rows, a_columns, ElementType::fp32, a_layout,
                           padded_leading(a_layout, a_rows, a_columns), 12);
            input.b = view(next_id++, b_rows, b_columns, ElementType::fp32, b_layout,
                           padded_leading(b_layout, b_rows, b_columns), 20);
            input.c = view(next_id++, m, n, ElementType::fp32, c_layout,
                           padded_leading(c_layout, m, n), 28);
            input.a_operation = a_operation;
            input.b_operation = b_operation;
            input.compute_mode = ComputeMode::fast_tf32;
            CHECK(validate_problem(input));

            PlannerConfig config;
            config.chunk_bytes = 32;
            config.cache_target_bytes = 1ULL << 20U;
            config.workspace_bytes = 0;
            config.preferred_geometry = {4, 3, 3};
            const GemmPlan plan = make_plan(input, config);
            CHECK(plan);
            CHECK(plan.statistics.legacy_tile_enumerations == 0);
            for (const GemmTile& tile : plan.tiles) {
              const WorkingSetResult exact = enumerate_tile_working_set(input, tile, 32);
              CHECK(exact);
              CHECK(tile.resident_bytes == exact.resident_bytes);
              CHECK(tile.chunk_count == exact.chunks.size());
              CHECK(tile.access_range_count == exact.accesses.size());
            }
          }
        }
      }
    }
  }

  GemmProblem shared_inputs = problem(8, 8, 8);
  shared_inputs.b.allocation_id = shared_inputs.a.allocation_id;
  CHECK(validate_problem(shared_inputs));
  PlannerConfig shared_config;
  shared_config.chunk_bytes = 64;
  shared_config.cache_target_bytes = 1ULL << 20U;
  shared_config.workspace_bytes = 0;
  shared_config.preferred_geometry = {4, 4, 4};
  const GemmPlan shared_plan = make_plan(shared_inputs, shared_config);
  CHECK(shared_plan);
  CHECK(shared_plan.statistics.legacy_tile_enumerations == shared_plan.tiles.size());
}

void signature_tests() {
  using namespace xvram::gemm;

  GemmProblem input = problem();
  GemmTile tile;
  tile.m_count = input.m;
  tile.n_count = input.n;
  tile.k_count = input.k;
  const AlgorithmSignature first = make_algorithm_signature(input, tile, 4096);
  const AlgorithmSignature same = make_algorithm_signature(input, tile, 4096);
  CHECK(first == same);
  CHECK(first.a_alignment_bytes == 256);

  const AlgorithmSignature different_workspace = make_algorithm_signature(input, tile, 8192);
  CHECK(first != different_workspace);
  input.a.layout = MatrixLayout::column_major;
  input.a.leading_dimension = input.a.rows;
  const AlgorithmSignature different_layout = make_algorithm_signature(input, tile, 4096);
  CHECK(first != different_layout);
}

struct FakeLtState {
  int create_calls = 0;
  int destroy_calls = 0;
  int descriptor_creates = 0;
  int descriptor_destroys = 0;
  int layout_creates = 0;
  int layout_destroys = 0;
  int preference_creates = 0;
  int preference_destroys = 0;
  int heuristic_calls = 0;
  int matmul_calls = 0;
  int core_gemm_calls = 0;
  xvram::cublas::abi::MathMode last_math_mode = xvram::cublas::abi::default_math;
  xvram::cublas::abi::Status destroy_status = xvram::cublas::abi::success;
  std::uint64_t workspace_limit = 0;
  std::array<std::uint32_t, 4> alignments{};
  xvram::cublas::abi::Status heuristic_status = xvram::cublas::abi::success;
};

FakeLtState fake_lt;

xvram::cublas::abi::Status fake_core_create(xvram::cublas::abi::Handle* output) {
  *output = reinterpret_cast<xvram::cublas::abi::Handle>(&fake_lt);
  return xvram::cublas::abi::success;
}

xvram::cublas::abi::Status fake_core_destroy(xvram::cublas::abi::Handle) {
  return xvram::cublas::abi::success;
}

xvram::cublas::abi::Status fake_core_version(xvram::cublas::abi::Handle, int* output) {
  *output = 13'050'127;
  return xvram::cublas::abi::success;
}

xvram::cublas::abi::Status fake_lt_create(xvram::cublas::abi::LtHandle* output) {
  ++fake_lt.create_calls;
  *output = reinterpret_cast<xvram::cublas::abi::LtHandle>(&fake_lt);
  return xvram::cublas::abi::success;
}

xvram::cublas::abi::Status fake_lt_destroy(xvram::cublas::abi::LtHandle) {
  ++fake_lt.destroy_calls;
  return fake_lt.destroy_status;
}

std::size_t fake_lt_version() {
  return 13'050'127;
}

xvram::cublas::abi::Status fake_desc_create(xvram::cublas::abi::LtMatmulDesc* output,
                                            xvram::cublas::abi::ComputeType,
                                            xvram::cublas::abi::DataType) {
  ++fake_lt.descriptor_creates;
  *output = reinterpret_cast<xvram::cublas::abi::LtMatmulDesc>(&fake_lt);
  return xvram::cublas::abi::success;
}

xvram::cublas::abi::Status fake_desc_destroy(xvram::cublas::abi::LtMatmulDesc) {
  ++fake_lt.descriptor_destroys;
  return xvram::cublas::abi::success;
}

xvram::cublas::abi::Status fake_desc_set(xvram::cublas::abi::LtMatmulDesc,
                                         xvram::cublas::abi::LtAttribute, const void*,
                                         std::size_t) {
  return xvram::cublas::abi::success;
}

xvram::cublas::abi::Status fake_layout_create(xvram::cublas::abi::LtMatrixLayout* output,
                                              xvram::cublas::abi::DataType, std::uint64_t,
                                              std::uint64_t, std::int64_t) {
  ++fake_lt.layout_creates;
  *output = reinterpret_cast<xvram::cublas::abi::LtMatrixLayout>(&fake_lt);
  return xvram::cublas::abi::success;
}

xvram::cublas::abi::Status fake_layout_destroy(xvram::cublas::abi::LtMatrixLayout) {
  ++fake_lt.layout_destroys;
  return xvram::cublas::abi::success;
}

xvram::cublas::abi::Status fake_layout_set(xvram::cublas::abi::LtMatrixLayout,
                                           xvram::cublas::abi::LtAttribute attribute,
                                           const void* value, std::size_t size) {
  CHECK(attribute == xvram::cublas::abi::lt_matrix_layout_order);
  CHECK(size == sizeof(xvram::cublas::abi::LtOrder));
  CHECK(value != nullptr);
  return xvram::cublas::abi::success;
}

xvram::cublas::abi::Status fake_preference_create(xvram::cublas::abi::LtMatmulPreference* output) {
  ++fake_lt.preference_creates;
  *output = reinterpret_cast<xvram::cublas::abi::LtMatmulPreference>(&fake_lt);
  return xvram::cublas::abi::success;
}

xvram::cublas::abi::Status fake_preference_destroy(xvram::cublas::abi::LtMatmulPreference) {
  ++fake_lt.preference_destroys;
  return xvram::cublas::abi::success;
}

xvram::cublas::abi::Status fake_preference_set(xvram::cublas::abi::LtMatmulPreference,
                                               const xvram::cublas::abi::LtAttribute attribute,
                                               const void* value, const std::size_t size) {
  if (attribute == xvram::cublas::abi::lt_preference_max_workspace_bytes) {
    CHECK(size == sizeof(std::uint64_t));
    std::memcpy(&fake_lt.workspace_limit, value, size);
  } else {
    CHECK(size == sizeof(std::uint32_t));
    const std::size_t index = static_cast<std::size_t>(
        attribute - xvram::cublas::abi::lt_preference_min_alignment_a_bytes);
    CHECK(index < fake_lt.alignments.size());
    if (index < fake_lt.alignments.size()) {
      std::memcpy(&fake_lt.alignments[index], value, size);
    }
  }
  return xvram::cublas::abi::success;
}

xvram::cublas::abi::Status
fake_heuristic(xvram::cublas::abi::LtHandle, xvram::cublas::abi::LtMatmulDesc,
               xvram::cublas::abi::LtMatrixLayout, xvram::cublas::abi::LtMatrixLayout,
               xvram::cublas::abi::LtMatrixLayout, xvram::cublas::abi::LtMatrixLayout,
               xvram::cublas::abi::LtMatmulPreference, int,
               xvram::cublas::abi::LtMatmulHeuristicResult* results, int* returned,
               xvram::cublas::abi::Stream) {
  ++fake_lt.heuristic_calls;
  if (fake_lt.heuristic_status != xvram::cublas::abi::success) {
    return fake_lt.heuristic_status;
  }
  results[0].algorithm.data[0] = 42;
  results[0].workspace_size = 256;
  results[0].state = xvram::cublas::abi::success;
  results[0].waves_count = 1.0F;
  *returned = 1;
  return xvram::cublas::abi::success;
}

xvram::cublas::abi::Status fake_matmul(xvram::cublas::abi::LtHandle,
                                       xvram::cublas::abi::LtMatmulDesc, const void*, const void*,
                                       xvram::cublas::abi::LtMatrixLayout, const void*,
                                       xvram::cublas::abi::LtMatrixLayout, const void*, const void*,
                                       xvram::cublas::abi::LtMatrixLayout, void*,
                                       xvram::cublas::abi::LtMatrixLayout,
                                       const xvram::cublas::abi::LtMatmulAlgorithm* algorithm,
                                       void*, std::size_t, xvram::cublas::abi::Stream) {
  ++fake_lt.matmul_calls;
  CHECK(algorithm != nullptr);
  CHECK(algorithm != nullptr && algorithm->data[0] == 42);
  return xvram::cublas::abi::success;
}

xvram::cublas::abi::Status fake_set_stream(xvram::cublas::abi::Handle, xvram::cublas::abi::Stream) {
  return xvram::cublas::abi::success;
}

xvram::cublas::abi::Status fake_set_math(xvram::cublas::abi::Handle,
                                         const xvram::cublas::abi::MathMode mode) {
  fake_lt.last_math_mode = mode;
  return xvram::cublas::abi::success;
}

xvram::cublas::abi::Status fake_set_workspace(xvram::cublas::abi::Handle, void*, std::size_t) {
  return xvram::cublas::abi::success;
}

xvram::cublas::abi::Status fake_gemm_ex(xvram::cublas::abi::Handle, xvram::cublas::abi::Operation,
                                        xvram::cublas::abi::Operation, int, int, int, const void*,
                                        const void*, xvram::cublas::abi::DataType, int, const void*,
                                        xvram::cublas::abi::DataType, int, const void*, void*,
                                        xvram::cublas::abi::DataType, int,
                                        xvram::cublas::abi::ComputeType,
                                        xvram::cublas::abi::GemmAlgorithm) {
  ++fake_lt.core_gemm_calls;
  return xvram::cublas::abi::success;
}

xvram::cublas::abi::Status fake_dgemm(xvram::cublas::abi::Handle, xvram::cublas::abi::Operation,
                                      xvram::cublas::abi::Operation, int, int, int, const double*,
                                      const double*, int, const double*, int, const double*,
                                      double*, int) {
  ++fake_lt.core_gemm_calls;
  return xvram::cublas::abi::success;
}

void cublas_lt_executor_tests() {
  using namespace xvram::cublas;
  fake_lt = {};
  CublasDispatch dispatch;
  dispatch.set_stream = &fake_set_stream;
  dispatch.set_math_mode = &fake_set_math;
  dispatch.set_workspace = &fake_set_workspace;
  dispatch.gemm_ex = &fake_gemm_ex;
  dispatch.dgemm = &fake_dgemm;
  dispatch.lt_create = &fake_lt_create;
  dispatch.lt_destroy = &fake_lt_destroy;
  dispatch.lt_get_version = &fake_lt_version;
  dispatch.lt_matmul_desc_create = &fake_desc_create;
  dispatch.lt_matmul_desc_destroy = &fake_desc_destroy;
  dispatch.lt_matmul_desc_set_attribute = &fake_desc_set;
  dispatch.lt_matrix_layout_create = &fake_layout_create;
  dispatch.lt_matrix_layout_destroy = &fake_layout_destroy;
  dispatch.lt_matrix_layout_set_attribute = &fake_layout_set;
  dispatch.lt_matmul_preference_create = &fake_preference_create;
  dispatch.lt_matmul_preference_destroy = &fake_preference_destroy;
  dispatch.lt_matmul_preference_set_attribute = &fake_preference_set;
  dispatch.lt_matmul_algo_get_heuristic_for_stream = &fake_heuristic;
  dispatch.lt_matmul_algo_get_heuristic = nullptr;
  dispatch.lt_matmul = &fake_matmul;

  alignas(256) std::array<std::byte, 4096> a{};
  alignas(256) std::array<std::byte, 4096> b{};
  alignas(256) std::array<std::byte, 4096> c{};
  alignas(256) std::array<std::byte, 4096> workspace{};
  const float alpha = 1.0F;
  const float beta = 0.0F;
  LtMatmulRequest request;
  request.signature.compute_type = abi::compute_fast_tf32;
  request.signature.scale_type = abi::data_fp32;
  request.signature.a = {abi::data_fp32, abi::lt_order_row_major, 2, 3, 3, 0};
  request.signature.b = {abi::data_fp32, abi::lt_order_row_major, 3, 4, 4, 0};
  request.signature.c = {abi::data_fp32, abi::lt_order_row_major, 2, 4, 4, 0};
  request.signature.d = request.signature.c;
  request.signature.workspace_limit_bytes = 1024;
  request.alpha = &alpha;
  request.a = a.data();
  request.b = b.data();
  request.beta = &beta;
  request.c = c.data();
  request.d = c.data();
  request.workspace = workspace.data();
  request.workspace_bytes = workspace.size();

  LtMatmulExecutor executor(dispatch, 2);
  CHECK(executor.initialize() == abi::success);
  CHECK(executor.library_version() == 13'050'127);
  const LtExecutionResult first = executor.execute(request);
  CHECK(first);
  CHECK(!first.algorithm_cache_hit);
  CHECK(first.algorithm_workspace_bytes == 256);
  CHECK(fake_lt.heuristic_calls == 1);
  CHECK(fake_lt.matmul_calls == 1);
  CHECK(fake_lt.workspace_limit == 1024);
  CHECK(fake_lt.alignments[0] == 256);

  const LtExecutionResult second = executor.execute(request);
  CHECK(second);
  CHECK(second.algorithm_cache_hit);
  CHECK(fake_lt.heuristic_calls == 1);
  CHECK(fake_lt.matmul_calls == 2);
  CHECK(executor.cache().statistics().hits == 1);
  CHECK(executor.cache().statistics().misses == 1);

  request.signature.workspace_limit_bytes = 512;
  CHECK(executor.execute(request));
  CHECK(fake_lt.heuristic_calls == 2);
  CHECK(executor.cache().size() == 2);

  executor.cache().clear();
  request.signature.compute_type = abi::compute_fp32_pedantic;
  request.signature.a.data_type = abi::data_bf16;
  request.signature.b.data_type = abi::data_bf16;
  request.signature.c.data_type = abi::data_bf16;
  request.signature.d.data_type = abi::data_bf16;
  CHECK(executor.execute(request));

  CoreGemmRequest strict_low_precision;
  strict_low_precision.m = 2;
  strict_low_precision.n = 4;
  strict_low_precision.k = 1025;
  strict_low_precision.a = a.data();
  strict_low_precision.lda = 2;
  strict_low_precision.b = b.data();
  strict_low_precision.ldb = 1025;
  strict_low_precision.c = c.data();
  strict_low_precision.ldc = 2;
  strict_low_precision.compute_type = abi::compute_fp32_pedantic;
  strict_low_precision.math_mode = abi::pedantic_math;
  strict_low_precision.workspace = workspace.data();
  strict_low_precision.workspace_bytes = workspace.size();
  const auto core_handle = reinterpret_cast<abi::Handle>(&fake_lt);
  for (const abi::DataType low_precision_type : {abi::data_fp16, abi::data_bf16}) {
    request.signature.compute_type = abi::compute_fp32_pedantic;
    request.signature.a.data_type = low_precision_type;
    request.signature.b.data_type = low_precision_type;
    request.signature.c.data_type = low_precision_type;
    request.signature.d.data_type = low_precision_type;
    strict_low_precision.a_type = low_precision_type;
    strict_low_precision.b_type = low_precision_type;
    strict_low_precision.c_type = low_precision_type;
    const int matmul_calls_before = fake_lt.matmul_calls;
    const int heuristic_calls_before = fake_lt.heuristic_calls;
    const int core_calls_before = fake_lt.core_gemm_calls;
    const PreferredGemmResult exact =
        executor.execute_preferred(request, core_handle, strict_low_precision);
    CHECK(exact);
    CHECK(exact.path == PreferredGemmPath::cublas_core);
    CHECK(fake_lt.matmul_calls == matmul_calls_before);
    CHECK(fake_lt.heuristic_calls == heuristic_calls_before);
    CHECK(fake_lt.core_gemm_calls == core_calls_before + 1);
    CHECK(fake_lt.last_math_mode ==
          (abi::pedantic_math | abi::disallow_reduced_precision_reduction));
  }

  executor.cache().clear();
  request.signature.compute_type = abi::compute_fast_tf32;
  request.signature.a.data_type = abi::data_fp32;
  request.signature.b.data_type = abi::data_fp32;
  request.signature.c.data_type = abi::data_fp32;
  request.signature.d.data_type = abi::data_fp32;
  fake_lt.heuristic_status = abi::not_supported;
  CoreGemmRequest fallback;
  fallback.m = 2;
  fallback.n = 4;
  fallback.k = 3;
  fallback.a = a.data();
  fallback.lda = 2;
  fallback.b = b.data();
  fallback.ldb = 3;
  fallback.c = c.data();
  fallback.ldc = 2;
  fallback.compute_type = abi::compute_fast_tf32;
  fallback.math_mode = abi::tf32_tensor_op_math;
  fallback.workspace = workspace.data();
  fallback.workspace_bytes = workspace.size();
  const int fallback_core_calls_before = fake_lt.core_gemm_calls;
  const PreferredGemmResult preferred = executor.execute_preferred(request, core_handle, fallback);
  CHECK(preferred);
  CHECK(preferred.path == PreferredGemmPath::cublas_core);
  CHECK(preferred.lt.code == LtExecutionCode::fallback_required);
  CHECK(fake_lt.core_gemm_calls == fallback_core_calls_before + 1);
  fake_lt.destroy_status = abi::execution_failed;
  CHECK(executor.close() == abi::execution_failed);
  CHECK(fake_lt.destroy_calls == 1);
  CHECK(fake_lt.descriptor_creates == fake_lt.descriptor_destroys);
  CHECK(fake_lt.layout_creates == fake_lt.layout_destroys);
  CHECK(fake_lt.preference_creates == fake_lt.preference_destroys);

  fake_lt.destroy_status = abi::success;
  const int destroy_calls_before_abandon = fake_lt.destroy_calls;
  {
    LtMatmulExecutor abandoned(dispatch, 2);
    CHECK(abandoned.initialize() == abi::success);
    abandoned.abandon();
  }
  CHECK(fake_lt.destroy_calls == destroy_calls_before_abandon);
}

void cublas_optional_lt_loader_tests() {
  using namespace xvram::cublas;

  CublasDispatch dispatch;
  dispatch.create = &fake_core_create;
  dispatch.destroy = &fake_core_destroy;
  dispatch.set_stream = &fake_set_stream;
  dispatch.get_version = &fake_core_version;
  dispatch.set_math_mode = &fake_set_math;
  dispatch.gemm_ex = &fake_gemm_ex;
  dispatch.dgemm = &fake_dgemm;

  CublasApi api(dispatch, "injected-core", "");
  CHECK(api.status() == CublasLoadStatus::loaded);
  CHECK(api.has_complete_baseline());
  CHECK(!api.has_lt());
  CHECK(api.lt_status() == CublasLtLoadStatus::library_unavailable);
  CHECK(api.library_source() == CublasLibrarySource::unknown);
  CHECK(api.error().empty());
  CHECK(!api.lt_error().empty());
  CHECK(api.missing_symbols().empty());
  CHECK(!api.missing_lt_symbols().empty());
  CHECK(std::string_view{cublas_lt_load_status_name(api.lt_status())} == "library_unavailable");
  CHECK(std::string_view{cublas_library_source_name(CublasLibrarySource::system)} == "system");
  CHECK(std::string_view{cublas_library_source_name(CublasLibrarySource::app_local)} ==
        "app_local");
  CHECK(std::string_view{cublas_library_source_name(CublasLibrarySource::explicit_path)} ==
        "explicit");

  const CublasLoadResult repeated = api.load();
  CHECK(repeated.status == CublasLoadStatus::loaded);
  CHECK(!repeated.attempted_now);
  CHECK(repeated.lt_status == CublasLtLoadStatus::library_unavailable);

  dispatch.lt_create = &fake_lt_create;
  CublasApi partial_lt(dispatch, "injected-core", "injected-partial-lt");
  CHECK(partial_lt.status() == CublasLoadStatus::loaded);
  CHECK(!partial_lt.has_lt());
  CHECK(partial_lt.lt_status() == CublasLtLoadStatus::symbols_missing);
  CHECK(partial_lt.error().empty());
  CHECK(!partial_lt.missing_lt_symbols().empty());
}

} // namespace

int main() {
  validation_tests();
  strided_range_tests();
  planner_tests();
  planner_large_k_regression_tests();
  planner_summary_equivalence_tests();
  signature_tests();
  cublas_lt_executor_tests();
  cublas_optional_lt_loader_tests();
  return failures == 0 ? 0 : 1;
}
