#pragma once

#include "platform/cuda/cuda_api.hpp"
#include "vmm_poc/workload.hpp"

#include <chrono>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace xvram::vmm_poc {

enum class RequestedMode { reference, pipeline, both };

struct ExecutorOptions {
  std::int32_t device_ordinal = 0;
  std::optional<std::uint64_t> logical_bytes;
  std::uint64_t chunk_bytes = 64ULL * 1024ULL * 1024ULL;
  std::uint32_t window_slots = 2;
  std::uint32_t passes = 2;
  RequestedMode mode = RequestedMode::both;
  std::chrono::milliseconds stall_timeout{5000};
  // Internal/testable CPU-scan heartbeat cadence. The CLI keeps the one-second default.
  std::chrono::milliseconds progress_heartbeat{1000};
  std::uint64_t device_headroom_bytes = 512ULL * 1024ULL * 1024ULL;
  std::uint64_t seed = 0x585652414D503031ULL;
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

struct DeviceSnapshot {
  std::int32_t ordinal = 0;
  std::string name;
  std::optional<std::string> uuid;
  std::optional<std::string> luid;
  std::optional<std::string> pci_bus_id;
  std::optional<std::string> driver_model;
  std::uint64_t total_memory_bytes = 0;
  std::uint64_t free_memory_bytes_start = 0;
  bool vmm_supported = false;
  bool unified_addressing = false;
  std::uint32_t multiprocessor_count = 0;
  std::uint32_t warp_size = 0;
  std::uint32_t maximum_threads_per_block = 0;
  std::uint64_t minimum_granularity_bytes = 0;
  std::uint64_t recommended_granularity_bytes = 0;
  std::optional<std::uint64_t> wddm_budget_bytes_start;
  std::optional<std::uint64_t> wddm_usage_bytes_start;
  std::optional<std::uint64_t> wddm_available_bytes_start;
  std::optional<std::uint64_t> wddm_available_bytes_minimum;
  std::optional<std::uint64_t> wddm_available_bytes_end;
};

struct StageDurations {
  double setup_ms = 0.0;
  double h2d_ms = 0.0;
  double kernel_ms = 0.0;
  double d2h_ms = 0.0;
  double verification_ms = 0.0;
  double total_ms = 0.0;
};

struct ModeStatistics {
  std::string status = "not_requested";
  std::uint32_t slots = 0;
  std::uint64_t mappings = 0;
  std::uint64_t set_access_calls = 0;
  std::uint64_t unmappings = 0;
  std::uint64_t remappings = 0;
  std::uint64_t event_boundaries = 0;
  std::uint64_t unsafe_remaps = 0;
  std::uint64_t handle_reuses = 0;
  std::uint64_t h2d_bytes = 0;
  std::uint64_t d2h_bytes = 0;
  std::uint64_t tiles_retired = 0;
  std::uint64_t words_verified = 0;
  std::uint32_t passes_completed = 0;
  std::uint64_t mismatch_count = 0;
  std::optional<std::uint64_t> first_mismatch_byte_offset;
  bool stable_addresses = true;
  bool full_verification = false;
  bool matches_cpu = false;
  std::string expected_digest128;
  std::string digest128;
  StageDurations timings;
  std::vector<double> remap_samples_ms;
};

struct CleanupLedger {
  std::optional<bool> complete;
  std::optional<bool> events_drained;
  std::optional<bool> mappings_removed;
  std::optional<bool> handles_released;
  std::optional<bool> reservation_released;
  std::optional<bool> events_destroyed;
  std::optional<bool> streams_destroyed;
  std::optional<bool> module_unloaded;
  std::optional<bool> pinned_buffers_released;
  std::optional<bool> host_backing_released;
  std::optional<bool> context_destroyed;
};

struct ExecutionFailure {
  std::string stage;
  std::string operation;
  std::string message;
  std::optional<std::int64_t> native_code;
  std::optional<std::string> native_name;
  std::optional<std::uint32_t> pass_index;
  std::optional<std::uint64_t> tile_index;
  std::optional<std::uint64_t> logical_byte_offset;
};

struct ExecutorResult {
  int exit_code = 27;
  std::string status = "failed";
  std::optional<std::string> reason;
  std::optional<DeviceSnapshot> device;
  std::optional<WorkloadPlan> plan;
  ModeStatistics reference;
  ModeStatistics pipeline;
  CleanupLedger cleanup;
  std::optional<ExecutionFailure> failure;
  std::vector<std::string> diagnostics;
};

using ProgressCallback = std::function<void(std::string_view mode, const ModeStatistics& statistics,
                                            std::uint64_t total_tile_visits)>;

[[nodiscard]] ExecutorResult run_executor(cuda::CudaApi& api, const ExecutorOptions& options,
                                          const ProgressCallback& progress = {},
                                          const ExecutorEnvironment* environment = nullptr);

} // namespace xvram::vmm_poc
