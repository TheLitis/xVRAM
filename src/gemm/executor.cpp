#include "gemm/executor.hpp"

#include "residency/runtime.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>

namespace xvram::gemm {
namespace {

[[nodiscard]] cublas::abi::DataType to_cublas_type(const ElementType type) noexcept {
  switch (type) {
  case ElementType::fp16:
    return cublas::abi::data_fp16;
  case ElementType::bf16:
    return cublas::abi::data_bf16;
  case ElementType::fp64:
    return cublas::abi::data_fp64;
  case ElementType::fp32:
  default:
    return cublas::abi::data_fp32;
  }
}

[[nodiscard]] bool fits_cublas_int(const std::uint64_t value) noexcept {
  return value <= static_cast<std::uint64_t>(std::numeric_limits<int>::max());
}

[[nodiscard]] bool uses_low_precision_output(const GemmProblem& problem) noexcept {
  return problem.c.element_type == ElementType::fp16 ||
         problem.c.element_type == ElementType::bf16;
}

[[nodiscard]] std::uint64_t matrix_offset(const MatrixView& matrix, const std::uint64_t row,
                                          const std::uint64_t column) noexcept {
  const std::uint64_t index = matrix.layout == MatrixLayout::row_major
                                  ? row * matrix.leading_dimension + column
                                  : column * matrix.leading_dimension + row;
  return matrix.offset_bytes + index * *element_size_bytes(matrix.element_type);
}

[[nodiscard]] cublas::abi::Operation
physical_operation(const MatrixView& matrix, const MatrixOperation logical_operation,
                   const bool transpose_result) noexcept {
  const bool logical_transpose = logical_operation == MatrixOperation::transpose;
  const bool desired_transpose = transpose_result ? !logical_transpose : logical_transpose;
  const bool physical_transpose = matrix.layout == MatrixLayout::row_major;
  return desired_transpose != physical_transpose ? cublas::abi::operation_transpose
                                                 : cublas::abi::operation_none;
}

[[nodiscard]] std::optional<std::uint64_t>
resolved_base(const ResolvedGemmTile& transaction, const residency::AllocationId id) {
  for (const ResolvedGemmAllocation& allocation : transaction.allocations) {
    if (allocation.allocation_id == id) {
      return allocation.device_base;
    }
  }
  return std::nullopt;
}

[[nodiscard]] ExecutionResult failure(const ExecutionStatus status, std::string operation,
                                      std::string message) {
  ExecutionResult result;
  result.status = status;
  result.stage = "gemm";
  result.operation = std::move(operation);
  result.message = std::move(message);
  return result;
}

[[nodiscard]] ExecutionResult control_result(const ExecutionControl control,
                                             const ExecutionBoundary boundary) {
  if (control == ExecutionControl::proceed) {
    return {};
  }
  return failure(control == ExecutionControl::deadline_expired
                     ? ExecutionStatus::deadline_expired
                     : ExecutionStatus::cancelled,
                 execution_boundary_operation(boundary),
                 control == ExecutionControl::deadline_expired
                     ? "GEMM execution deadline expired at a safe tile boundary"
                     : "GEMM execution was cancelled at a safe tile boundary");
}

[[nodiscard]] ExecutionControl sample_boundary(const ExecutionHooks& hooks,
                                               const ExecutionBoundary boundary,
                                               const std::size_t tile_index) {
  return hooks.boundary ? hooks.boundary(boundary, tile_index) : ExecutionControl::proceed;
}

[[nodiscard]] ExecutionResult runtime_failure(const residency::RuntimeStatus status) {
  ExecutionResult result =
      failure(ExecutionStatus::runtime_failure, "runtime_execute", "residency runtime failed");
  result.runtime_status = status;
  return result;
}

} // namespace

const char* execution_status_name(const ExecutionStatus status) noexcept {
  switch (status) {
  case ExecutionStatus::success:
    return "success";
  case ExecutionStatus::invalid_argument:
    return "invalid_argument";
  case ExecutionStatus::unavailable:
    return "unavailable";
  case ExecutionStatus::unsupported:
    return "unsupported";
  case ExecutionStatus::runtime_failure:
    return "runtime_failure";
  case ExecutionStatus::cublas_failure:
    return "cublas_failure";
  case ExecutionStatus::deadline_expired:
    return "deadline_expired";
  case ExecutionStatus::cancelled:
    return "cancelled";
  case ExecutionStatus::internal_failure:
    return "internal_failure";
  }
  return "internal_failure";
}

const char* execution_boundary_operation(const ExecutionBoundary boundary) noexcept {
  switch (boundary) {
  case ExecutionBoundary::before_execution:
    return "pre_launch_deadline";
  case ExecutionBoundary::before_tile:
  case ExecutionBoundary::after_tile:
    return "tile_deadline";
  case ExecutionBoundary::after_residency:
    return "post_residency_deadline";
  }
  return "tile_deadline";
}

bool is_cublas_lt_exact_proof_eligible(const GemmProblem& problem) noexcept {
  return problem.compute_mode != ComputeMode::strict_fp32 || !uses_low_precision_output(problem);
}

ExecutionResult execute_resolved_gemm_tile(const NativeGemmResources& resources,
                                           const GemmProblem& problem, const GemmTile& tile,
                                           const ResolvedGemmTile& transaction,
                                           const ExecutionAlgorithmHook& algorithm) {
  if (resources.handle == nullptr) {
    return failure(ExecutionStatus::unavailable, "load_cublas",
                   "cuBLAS/cuBLASLt runtime is unavailable");
  }
  const std::optional<std::uint64_t> a_base =
      resolved_base(transaction, problem.a.allocation_id);
  const std::optional<std::uint64_t> b_base =
      resolved_base(transaction, problem.b.allocation_id);
  const std::optional<std::uint64_t> c_base =
      resolved_base(transaction, problem.c.allocation_id);
  if (!a_base.has_value() || !b_base.has_value() || !c_base.has_value()) {
    return failure(ExecutionStatus::internal_failure, "resolve_tile",
                   "runtime did not resolve every GEMM allocation");
  }

  const std::uint64_t a_row =
      problem.a_operation == MatrixOperation::none ? tile.m_begin : tile.k_begin;
  const std::uint64_t a_column =
      problem.a_operation == MatrixOperation::none ? tile.k_begin : tile.m_begin;
  const std::uint64_t b_row =
      problem.b_operation == MatrixOperation::none ? tile.k_begin : tile.n_begin;
  const std::uint64_t b_column =
      problem.b_operation == MatrixOperation::none ? tile.n_begin : tile.k_begin;
  const std::uint64_t a_address = *a_base + matrix_offset(problem.a, a_row, a_column);
  const std::uint64_t b_address = *b_base + matrix_offset(problem.b, b_row, b_column);
  const std::uint64_t c_address =
      *c_base + matrix_offset(problem.c, tile.m_begin, tile.n_begin);
  if (!fits_cublas_int(tile.m_count) || !fits_cublas_int(tile.n_count) ||
      !fits_cublas_int(tile.k_count) || !fits_cublas_int(problem.a.leading_dimension) ||
      !fits_cublas_int(problem.b.leading_dimension) ||
      !fits_cublas_int(problem.c.leading_dimension)) {
    return failure(ExecutionStatus::unsupported, "launch_tile",
                   "tile dimensions exceed the cuBLAS v1 integer ABI");
  }

  const cublas::abi::MathMode math_mode =
      problem.compute_mode == ComputeMode::fast_tf32
          ? cublas::abi::tf32_tensor_op_math
          : (problem.compute_mode == ComputeMode::strict_fp32 ? cublas::abi::pedantic_math
                                                               : cublas::abi::default_math);
  const bool transpose_result = problem.c.layout == MatrixLayout::row_major;
  const MatrixView& first_matrix = transpose_result ? problem.b : problem.a;
  const MatrixView& second_matrix = transpose_result ? problem.a : problem.b;
  const MatrixOperation first_logical =
      transpose_result ? problem.b_operation : problem.a_operation;
  const MatrixOperation second_logical =
      transpose_result ? problem.a_operation : problem.b_operation;
  const std::uint64_t first_address = transpose_result ? b_address : a_address;
  const std::uint64_t second_address = transpose_result ? a_address : b_address;
  const int m = static_cast<int>(transpose_result ? tile.n_count : tile.m_count);
  const int n = static_cast<int>(transpose_result ? tile.m_count : tile.n_count);
  const int k = static_cast<int>(tile.k_count);
  const cublas::abi::Operation op_first =
      physical_operation(first_matrix, first_logical, transpose_result);
  const cublas::abi::Operation op_second =
      physical_operation(second_matrix, second_logical, transpose_result);
  const int lda = static_cast<int>(first_matrix.leading_dimension);
  const int ldb = static_cast<int>(second_matrix.leading_dimension);
  const int ldc = static_cast<int>(problem.c.leading_dimension);
  const double beta_double = tile.beta_mode == BetaMode::user_beta ? problem.beta : 1.0;
  const cublas::abi::ComputeType compute_type =
      problem.compute_mode == ComputeMode::fp64
          ? cublas::abi::compute_fp64
          : (problem.compute_mode == ComputeMode::fast_tf32
                 ? cublas::abi::compute_fast_tf32
                 : (problem.compute_mode == ComputeMode::strict_fp32
                        ? cublas::abi::compute_fp32_pedantic
                        : cublas::abi::compute_fp32));
  void* const workspace =
      reinterpret_cast<void*>(static_cast<std::uintptr_t>(transaction.workspace_address));

  cublas::CoreGemmRequest core;
  core.operation_a = op_first;
  core.operation_b = op_second;
  core.m = m;
  core.n = n;
  core.k = k;
  core.a = reinterpret_cast<const void*>(static_cast<std::uintptr_t>(first_address));
  core.a_type = to_cublas_type(first_matrix.element_type);
  core.lda = lda;
  core.b = reinterpret_cast<const void*>(static_cast<std::uintptr_t>(second_address));
  core.b_type = to_cublas_type(second_matrix.element_type);
  core.ldb = ldb;
  core.c = reinterpret_cast<void*>(static_cast<std::uintptr_t>(c_address));
  core.c_type = to_cublas_type(problem.c.element_type);
  core.ldc = ldc;
  core.compute_type = compute_type;
  core.math_mode = math_mode;
  core.alpha = problem.alpha;
  core.beta = beta_double;
  core.workspace = workspace;
  core.workspace_bytes = transaction.workspace_bytes;
  core.stream = transaction.stream;

  const auto lt_order = [](const MatrixLayout layout) {
    return layout == MatrixLayout::row_major ? cublas::abi::lt_order_row_major
                                             : cublas::abi::lt_order_column_major;
  };
  const auto lt_matrix = [&](const MatrixView& matrix, const std::uint64_t rows,
                             const std::uint64_t columns) {
    return cublas::LtMatrixSignature{to_cublas_type(matrix.element_type),
                                     lt_order(matrix.layout),
                                     rows,
                                     columns,
                                     static_cast<std::int64_t>(matrix.leading_dimension),
                                     0U};
  };
  const std::uint64_t a_rows =
      problem.a_operation == MatrixOperation::none ? tile.m_count : tile.k_count;
  const std::uint64_t a_columns =
      problem.a_operation == MatrixOperation::none ? tile.k_count : tile.m_count;
  const std::uint64_t b_rows =
      problem.b_operation == MatrixOperation::none ? tile.k_count : tile.n_count;
  const std::uint64_t b_columns =
      problem.b_operation == MatrixOperation::none ? tile.n_count : tile.k_count;
  cublas::LtMatmulSignature lt_signature;
  lt_signature.compute_type = compute_type;
  lt_signature.scale_type =
      problem.compute_mode == ComputeMode::fp64 ? cublas::abi::data_fp64 : cublas::abi::data_fp32;
  lt_signature.operation_a = problem.a_operation == MatrixOperation::none
                                 ? cublas::abi::operation_none
                                 : cublas::abi::operation_transpose;
  lt_signature.operation_b = problem.b_operation == MatrixOperation::none
                                 ? cublas::abi::operation_none
                                 : cublas::abi::operation_transpose;
  lt_signature.a = lt_matrix(problem.a, a_rows, a_columns);
  lt_signature.b = lt_matrix(problem.b, b_rows, b_columns);
  lt_signature.c = lt_matrix(problem.c, tile.m_count, tile.n_count);
  lt_signature.d = lt_signature.c;
  lt_signature.workspace_limit_bytes = transaction.workspace_bytes;

  const float alpha_float = static_cast<float>(problem.alpha);
  const float beta_float = static_cast<float>(beta_double);
  const double alpha_double = problem.alpha;
  cublas::LtMatmulRequest lt_request;
  lt_request.signature = lt_signature;
  lt_request.alpha = problem.compute_mode == ComputeMode::fp64
                         ? static_cast<const void*>(&alpha_double)
                         : static_cast<const void*>(&alpha_float);
  lt_request.a = reinterpret_cast<const void*>(static_cast<std::uintptr_t>(a_address));
  lt_request.b = reinterpret_cast<const void*>(static_cast<std::uintptr_t>(b_address));
  lt_request.beta = problem.compute_mode == ComputeMode::fp64
                        ? static_cast<const void*>(&beta_double)
                        : static_cast<const void*>(&beta_float);
  lt_request.c = reinterpret_cast<const void*>(static_cast<std::uintptr_t>(c_address));
  lt_request.d = reinterpret_cast<void*>(static_cast<std::uintptr_t>(c_address));
  lt_request.workspace = workspace;
  lt_request.workspace_bytes = transaction.workspace_bytes;
  lt_request.stream = transaction.stream;

  cublas::PreferredGemmResult result;
  if (resources.lt != nullptr) {
    result = resources.lt->execute_preferred(lt_request, resources.handle, core);
  } else {
    result.path = cublas::PreferredGemmPath::cublas_core;
    result.core = cublas::execute_core_gemm(resources.dispatch, resources.handle, core);
  }
  if (result) {
    if (algorithm) {
      algorithm(result);
    }
    return {};
  }

  const cublas::abi::Status status = result.path == cublas::PreferredGemmPath::cublas_core
                                         ? result.core.status
                                         : result.lt.status;
  const std::string_view operation = result.path == cublas::PreferredGemmPath::cublas_core
                                         ? result.core.operation
                                         : result.lt.operation;
  const char* status_text = resources.dispatch.get_status_string != nullptr
                                ? resources.dispatch.get_status_string(status)
                                : "cuBLAS operation failed";
  ExecutionResult failure_result =
      failure(ExecutionStatus::cublas_failure, std::string(operation),
              status_text != nullptr ? status_text : "cuBLAS operation failed");
  failure_result.cublas_status = status;
  return failure_result;
}

ExecutionResult execute_tiled_gemm(const TiledGemmResources& resources,
                                   const TiledGemmRequest& request,
                                   const ExecutionHooks& hooks) {
  if (!request.plan) {
    return failure(ExecutionStatus::invalid_argument, "execute_plan",
                   "GEMM plan is not executable");
  }
  if (const ExecutionResult boundary =
          control_result(sample_boundary(hooks, ExecutionBoundary::before_execution, 0),
                         ExecutionBoundary::before_execution);
      !boundary) {
    return boundary;
  }
  if (resources.native.handle == nullptr) {
    return failure(ExecutionStatus::unavailable, "load_cublas",
                   "cuBLAS/cuBLASLt runtime is unavailable");
  }

  for (std::size_t tile_index = 0; tile_index < request.plan.tiles.size(); ++tile_index) {
    const GemmTile& tile = request.plan.tiles[tile_index];
    if (const ExecutionResult boundary =
            control_result(sample_boundary(hooks, ExecutionBoundary::before_tile, tile_index),
                           ExecutionBoundary::before_tile);
        !boundary) {
      return boundary;
    }

    const WorkingSetResult working_set =
        enumerate_tile_working_set(request.problem, tile, resources.runtime.chunk_bytes());
    if (!working_set) {
      ExecutionResult result =
          failure(ExecutionStatus::internal_failure, "enumerate_working_set",
                  std::string(working_set_error_name(working_set.error)));
      result.working_set_error = working_set.error;
      return result;
    }

    ExecutionResult launch_result;
    const residency::RuntimeStatus runtime_status = resources.runtime.execute(
        working_set.accesses, request.workspace_bytes,
        [&](const residency::TransactionContext& transaction) {
          launch_result = control_result(
              sample_boundary(hooks, ExecutionBoundary::after_residency, tile_index),
              ExecutionBoundary::after_residency);
          if (!launch_result) {
            return residency::RuntimeStatus::callback_skipped;
          }
          std::array<ResolvedGemmAllocation, 3> resolved{};
          std::size_t resolved_count = 0;
          const auto append_base = [&](const residency::AllocationId id) {
            for (std::size_t index = 0; index < resolved_count; ++index) {
              if (resolved[index].allocation_id == id) {
                return true;
              }
            }
            for (const residency::ResolvedRange& range : transaction.ranges) {
              if (range.allocation_id == id && range.device_address >= range.offset_bytes) {
                resolved[resolved_count++] = ResolvedGemmAllocation{
                    id, static_cast<std::uint64_t>(range.device_address) - range.offset_bytes};
                return true;
              }
            }
            return false;
          };
          if (!append_base(request.problem.a.allocation_id) ||
              !append_base(request.problem.b.allocation_id) ||
              !append_base(request.problem.c.allocation_id)) {
            launch_result = failure(ExecutionStatus::internal_failure, "resolve_tile",
                                    "runtime did not resolve every GEMM allocation");
            return residency::RuntimeStatus::callback_failed;
          }
          const ResolvedGemmTile resolved_tile{
              std::span<const ResolvedGemmAllocation>(resolved.data(), resolved_count),
              static_cast<std::uint64_t>(transaction.workspace_address),
              transaction.workspace_bytes,
              reinterpret_cast<cublas::abi::Stream>(transaction.stream)};
          launch_result = execute_resolved_gemm_tile(resources.native, request.problem, tile,
                                                     resolved_tile, hooks.algorithm);
          return launch_result ? residency::RuntimeStatus::success
                               : residency::RuntimeStatus::callback_failed;
        });
    if (!launch_result) {
      return launch_result;
    }
    if (runtime_status != residency::RuntimeStatus::success) {
      return runtime_failure(runtime_status);
    }

    if (hooks.progress) {
      hooks.progress(tile_index + 1U, request.plan.tiles.size());
    }
    if (const ExecutionResult boundary =
            control_result(sample_boundary(hooks, ExecutionBoundary::after_tile, tile_index),
                           ExecutionBoundary::after_tile);
        !boundary) {
      return boundary;
    }

    for (std::uint32_t distance = 1; distance <= request.prefetch_distance; ++distance) {
      if (distance > request.plan.tiles.size() - tile_index - 1U) {
        break;
      }
      const GemmTile& future = request.plan.tiles[tile_index + distance];
      const WorkingSetResult future_working_set =
          enumerate_tile_working_set(request.problem, future, resources.runtime.chunk_bytes());
      if (!future_working_set) {
        ExecutionResult result =
            failure(ExecutionStatus::internal_failure, "enumerate_prefetch_working_set",
                    std::string(working_set_error_name(future_working_set.error)));
        result.working_set_error = future_working_set.error;
        return result;
      }
      const residency::RuntimeStatus prefetch_status =
          resources.runtime.prefetch(future_working_set.accesses);
      if (prefetch_status == residency::RuntimeStatus::budget_pressure) {
        break;
      }
      if (prefetch_status != residency::RuntimeStatus::success) {
        return runtime_failure(prefetch_status);
      }
    }
  }
  return {};
}

} // namespace xvram::gemm
