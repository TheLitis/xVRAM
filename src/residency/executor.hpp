#pragma once

#include "platform/cuda/cuda_api.hpp"
#include "residency/budget.hpp"
#include "residency/scenario.hpp"
#include "xvram/residency/report.hpp"

#include <chrono>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace xvram::residency {

enum class RequestedPolicy { clock, lru, both };

struct ExecutorOptions {
  std::int32_t device_ordinal = 0;
  std::optional<std::uint64_t> logical_bytes;
  std::uint64_t chunk_bytes = 64ULL * 1024ULL * 1024ULL;
  std::optional<std::uint64_t> cache_target_bytes;
  std::uint32_t staging_slots = 4;
  RequestedPolicy policy = RequestedPolicy::clock;
  std::uint32_t prefetch_distance = 2;
  ScenarioKind scenario = ScenarioKind::suite;
  std::uint32_t passes = 2;
  std::optional<std::uint64_t> pressure_bytes;
  std::uint64_t device_headroom_bytes = 512ULL * 1024ULL * 1024ULL;
  std::chrono::milliseconds budget_poll_interval{100};
  std::chrono::milliseconds stall_timeout{5000};
  std::chrono::milliseconds progress_heartbeat{1000};
  std::uint64_t seed = 0x585652414D503032ULL;
  bool trace_enabled = false;
  bool include_identifiers = false;
};

struct BudgetSnapshot {
  std::uint64_t budget_bytes = 0;
  std::uint64_t usage_bytes = 0;

  [[nodiscard]] std::uint64_t available_bytes() const noexcept {
    return budget_bytes > usage_bytes ? budget_bytes - usage_bytes : 0;
  }
};

struct ExecutorEnvironment {
  std::uint64_t physical_host_bytes = 0;
  std::uint64_t available_host_bytes = 0;
  std::optional<BudgetSnapshot> initial_device_budget;
  std::function<std::optional<BudgetSnapshot>()> query_device_budget;
};

struct ExecutionFailure {
  std::string stage;
  std::string operation;
  std::string message;
  std::optional<std::int64_t> native_code;
  std::optional<std::string> native_name;
  std::optional<std::string> policy;
  std::optional<std::string> scenario;
  std::optional<std::uint64_t> allocation_id;
  std::optional<std::uint64_t> chunk_index;
  std::optional<std::uint64_t> logical_byte_offset;
};

struct ExecutorResult {
  int exit_code = 27;
  std::string status = "failed";
  std::optional<std::string> reason;
  Configuration configuration;
  std::optional<DeviceInfo> device;
  std::vector<WorkloadResult> workloads;
  CacheStatistics cache;
  Telemetry telemetry;
  Proof proof;
  Cleanup cleanup;
  std::optional<ExecutionFailure> failure;
  std::vector<probe::Diagnostic> diagnostics;
};

using ProgressCallback =
    std::function<void(const ExecutorResult& result, std::uint64_t operations_completed,
                       std::uint64_t operations_total)>;
using TraceCallback = std::function<void(const TraceRecord& record)>;

[[nodiscard]] ExecutorResult run_executor(cuda::CudaApi& api, const ExecutorOptions& options,
                                          const ProgressCallback& progress = {},
                                          const TraceCallback& trace = {},
                                          const ExecutorEnvironment* environment = nullptr);

} // namespace xvram::residency
