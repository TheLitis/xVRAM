#pragma once

#include "gemm/planner.hpp"
#include "platform/cublas/cublas_api.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>

namespace xvram::residency {
class Runtime;
struct TransactionContext;
enum class RuntimeStatus;
} // namespace xvram::residency

namespace xvram::gemm {

enum class ExecutionStatus {
  success,
  invalid_argument,
  unavailable,
  unsupported,
  runtime_failure,
  cublas_failure,
  deadline_expired,
  cancelled,
  internal_failure,
};

enum class ExecutionBoundary {
  before_execution,
  before_tile,
  after_residency,
  after_tile,
};

enum class ExecutionControl {
  proceed,
  deadline_expired,
  cancelled,
};

struct ExecutionResult {
  ExecutionStatus status = ExecutionStatus::success;
  residency::RuntimeStatus runtime_status = static_cast<residency::RuntimeStatus>(0);
  cublas::abi::Status cublas_status = cublas::abi::success;
  WorkingSetError working_set_error = WorkingSetError::none;
  std::string stage;
  std::string operation;
  std::string message;

  [[nodiscard]] explicit operator bool() const noexcept {
    return status == ExecutionStatus::success;
  }
};

using ExecutionBoundaryHook =
    std::function<ExecutionControl(ExecutionBoundary boundary, std::size_t tile_index)>;
using ExecutionProgressHook =
    std::function<void(std::size_t completed_tiles, std::size_t total_tiles)>;
using ExecutionAlgorithmHook = std::function<void(const cublas::PreferredGemmResult& result)>;

struct ExecutionHooks {
  ExecutionBoundaryHook boundary;
  ExecutionProgressHook progress;
  ExecutionAlgorithmHook algorithm;
};

struct NativeGemmResources {
  const cublas::CublasDispatch& dispatch;
  cublas::abi::Handle handle = nullptr;
  cublas::LtMatmulExecutor* lt = nullptr;
};

struct TiledGemmResources {
  residency::Runtime& runtime;
  NativeGemmResources native;
};

struct TiledGemmRequest {
  const GemmProblem& problem;
  const GemmPlan& plan;
  std::uint64_t workspace_bytes = 0;
  std::uint32_t prefetch_distance = 0;
};

struct ResolvedGemmAllocation {
  residency::AllocationId allocation_id;
  std::uint64_t device_base = 0;
};

struct ResolvedGemmTile {
  std::span<const ResolvedGemmAllocation> allocations;
  std::uint64_t workspace_address = 0;
  std::uint64_t workspace_bytes = 0;
  cublas::abi::Stream stream = nullptr;
};

[[nodiscard]] const char* execution_status_name(ExecutionStatus status) noexcept;
[[nodiscard]] const char* execution_boundary_operation(ExecutionBoundary boundary) noexcept;

// cuBLASLt may choose reduction trees whose low-precision final conversion is not bit-identical
// to the strict Phase 3 scalar proof. Callers can use this predicate when reporting/planning the
// exact-proof path.
[[nodiscard]] bool is_cublas_lt_exact_proof_eligible(const GemmProblem& problem) noexcept;

// Executes one already-resident tile. This is exposed internally for focused tests and for
// adapters that already own an external residency lease.
[[nodiscard]] ExecutionResult
execute_resolved_gemm_tile(const NativeGemmResources& resources, const GemmProblem& problem,
                           const GemmTile& tile, const ResolvedGemmTile& transaction,
                           const ExecutionAlgorithmHook& algorithm = {});

// Executes the complete plan through residency transactions. Deadline/cancellation decisions are
// sampled only at event-safe boundaries; a submitted tile always retires before the next sample.
[[nodiscard]] ExecutionResult execute_tiled_gemm(const TiledGemmResources& resources,
                                                 const TiledGemmRequest& request,
                                                 const ExecutionHooks& hooks = {});

} // namespace xvram::gemm
