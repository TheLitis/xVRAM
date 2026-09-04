#pragma once

#include "compression_bench/options.hpp"
#include "xvram/compression/report.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace xvram::cuda {
class CudaApi;
}

namespace xvram::residency {
struct RuntimeTelemetry;
}

namespace xvram::compression {

struct ExecutionFailure {
  std::string stage;
  std::string operation;
  std::string message;
  std::optional<std::int64_t> native_code;
  std::optional<std::string> native_name;
  std::optional<std::string> scenario;
  std::optional<std::uint64_t> allocation_id;
  std::optional<std::uint64_t> chunk_index;
  std::optional<std::uint64_t> generation;
  std::optional<std::uint64_t> logical_byte_offset;
};

struct ExecutorResult {
  int exit_code = 27;
  std::string status = "failed";
  std::optional<std::string> reason;
  Configuration configuration;
  std::optional<DeviceInfo> device;
  std::vector<WorkloadResult> workloads;
  BackingStatistics backing;
  CodecStatistics codec;
  Telemetry telemetry;
  Proof proof;
  Cleanup cleanup;
  std::optional<ExecutionFailure> failure;
  std::vector<probe::Diagnostic> diagnostics;
};

using ProgressCallback = std::function<void(
    const ExecutorResult&, std::uint64_t operations_completed, std::uint64_t operations_total)>;
using TraceCallback = std::function<void(const TraceRecord&)>;

namespace detail {

// Callback failures are observable report failures, but must not unwind through CUDA callbacks
// or prevent the runtime from draining and cleaning up its resources.
[[nodiscard]] bool deliver_trace_record(const TraceCallback& callback,
                                        const TraceRecord& record) noexcept;
void apply_trace_delivery_result(ExecutorResult& result, std::uint64_t produced,
                                 std::uint64_t dropped);

// The mixed workload deliberately assigns one representation class to an entire residency chunk.
// Two classes remain stable to provide cost-history anchors, while two swap in opposite directions
// in the second content generation. This avoids the old fixture where every chunk had the same
// 50/50 block mixture in both generations.
[[nodiscard]] bool mixed_chunk_is_compressible(std::uint64_t chunk_index,
                                               bool phase_changed) noexcept;

inline constexpr std::uint32_t mixed_history_cycles = 64U;

[[nodiscard]] bool mixed_chunk_needs_history(std::uint64_t chunk_index) noexcept;
[[nodiscard]] bool mixed_chunk_is_write_capable(std::uint64_t chunk_index,
                                                std::uint32_t pass) noexcept;

// Derive stage-specific cleanup evidence from the runtime's post-close ledger. The few CUDA
// resources that currently lack individual counters remain tied to Runtime::close()'s status;
// mappings, handles, reservations, backing, codec resources, staging, and spill are not.
[[nodiscard]] Cleanup derive_runtime_cleanup(const residency::RuntimeTelemetry& telemetry,
                                             bool close_complete) noexcept;

[[nodiscard]] bool derive_write_admission_proof(const residency::RuntimeTelemetry& telemetry,
                                                RequestedPath requested_path) noexcept;

void fill_workload_pattern(ScenarioKind scenario, std::uint64_t seed, std::uint64_t absolute_offset,
                           std::uint64_t chunk_bytes, std::span<std::byte> output,
                           bool phase_changed = false) noexcept;

} // namespace detail

[[nodiscard]] ExecutorResult run_executor(cuda::CudaApi& api, const ExecutorOptions& options,
                                          const ProgressCallback& progress = {},
                                          const TraceCallback& trace = {});
[[nodiscard]] ExecutorResult run_executor(const ExecutorOptions& options,
                                          const ProgressCallback& progress = {},
                                          const TraceCallback& trace = {});

} // namespace xvram::compression
