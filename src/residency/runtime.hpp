#pragma once

#include "platform/cuda/cuda_api.hpp"
#include "residency/core.hpp"

#include <chrono>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace xvram::residency {

enum class RuntimeStatus {
  success,
  invalid_argument,
  unavailable,
  unsupported,
  host_oom,
  device_oom,
  budget_pressure,
  timeout,
  callback_skipped,
  callback_failed,
  cuda_failure,
  poisoned,
  cleanup_failure,
  internal_failure,
};

[[nodiscard]] const char* runtime_status_name(RuntimeStatus status) noexcept;

enum class RuntimeContextMode { isolated, attach_current };
enum class RuntimePolicy { clock, lru };
enum class ResidencyHint { normal, hot, streaming };

struct RuntimeConfig {
  std::int32_t device_ordinal = 0;
  RuntimeContextMode context_mode = RuntimeContextMode::isolated;
  cuda::abi::Context attached_context = nullptr;
  std::uint64_t chunk_bytes = 64ULL * 1024ULL * 1024ULL;
  std::uint64_t cache_target_bytes = 0;
  std::uint64_t device_headroom_bytes = 512ULL * 1024ULL * 1024ULL;
  std::uint64_t workspace_reserve_bytes = 0;
  std::uint32_t staging_slots = 4;
  RuntimePolicy policy = RuntimePolicy::clock;
  std::chrono::milliseconds stall_timeout{5000};
  std::chrono::milliseconds budget_poll_interval{100};
  std::chrono::milliseconds maximum_transaction_duration{250};
};

struct RuntimeError {
  RuntimeStatus status = RuntimeStatus::success;
  std::string stage;
  std::string operation;
  std::string message;
  std::optional<std::int64_t> native_code;
};

struct RuntimeAllocation {
  AllocationId id;
  std::uint64_t bytes = 0;
};

struct ResolvedRange {
  AllocationId allocation_id;
  std::uint64_t offset_bytes = 0;
  std::uint64_t length_bytes = 0;
  AccessMode mode = AccessMode::read;
  cuda::abi::DevicePointer device_address = 0;
};

struct TransactionContext {
  std::span<const ResolvedRange> ranges;
  cuda::abi::DevicePointer workspace_address = 0;
  std::uint64_t workspace_bytes = 0;
  cuda::abi::Stream stream = nullptr;
};

using TransactionCallback = std::function<RuntimeStatus(const TransactionContext&)>;

struct ExternalLeaseId {
  std::uint64_t value = 0;

  [[nodiscard]] explicit operator bool() const noexcept {
    return value != 0;
  }

  [[nodiscard]] auto operator<=>(const ExternalLeaseId&) const noexcept = default;
};

enum class ExternalSealMode {
  success,
  cancelled_before_submission,
  failed_after_possible_submission,
};

enum class ExternalLeaseState {
  armed,
  submitted,
  completed,
  cancelled,
  failed,
  quarantined,
};

struct ExternalLeaseRequest {
  std::span<const AccessRange> ranges;
  std::uint64_t workspace_bytes = 0;
};

struct ExternalLease {
  ExternalLeaseId id;
  std::vector<ResolvedRange> ranges;
  cuda::abi::DevicePointer workspace_address = 0;
  std::uint64_t workspace_bytes = 0;
  // Runtime-owned and non-owning for the caller. All leased work must be submitted here.
  cuda::abi::Stream stream = nullptr;
};

struct ExternalLeasePoll {
  ExternalLeaseId id;
  ExternalLeaseState state = ExternalLeaseState::armed;
  RuntimeStatus result = RuntimeStatus::success;
  double elapsed_ms = 0.0;
};

struct RuntimeTelemetry {
  std::uint64_t allocations_created = 0;
  std::uint64_t allocations_released = 0;
  std::uint64_t handles_created = 0;
  std::uint64_t handles_reused = 0;
  std::uint64_t handles_released = 0;
  std::uint64_t maps = 0;
  std::uint64_t set_access = 0;
  std::uint64_t unmaps = 0;
  std::uint64_t event_boundaries = 0;
  std::uint64_t unsafe_remaps = 0;
  std::uint64_t unsafe_transitions = 0;
  std::uint64_t h2d_bytes = 0;
  std::uint64_t d2h_bytes = 0;
  std::uint64_t cache_hits = 0;
  std::uint64_t cache_misses = 0;
  std::uint64_t clean_evictions = 0;
  std::uint64_t dirty_evictions = 0;
  std::uint64_t dirty_writebacks = 0;
  std::uint64_t transactions_submitted = 0;
  std::uint64_t transactions_completed = 0;
  std::uint64_t prefetches = 0;
  std::uint64_t workspace_peak_bytes = 0;
  std::uint64_t pinned_staging_bytes = 0;
  std::uint64_t target_bytes = 0;
  std::uint64_t target_minimum_bytes = 0;
  std::uint64_t target_maximum_bytes = 0;
  std::uint64_t budget_samples = 0;
  std::uint64_t budget_shrinks = 0;
  std::uint64_t budget_grows = 0;
  std::uint64_t target_oom_retries = 0;
  std::uint64_t watchdog_rejections = 0;
  std::uint64_t cuda_free_minimum_bytes = 0;
  std::uint64_t cuda_free_end_bytes = 0;
  std::optional<std::uint64_t> wddm_available_minimum_bytes;
  std::optional<std::uint64_t> wddm_available_end_bytes;
  std::uint64_t resident_bytes = 0;
  std::uint64_t resident_peak_bytes = 0;
  double last_transaction_ms = 0.0;
  bool stable_addresses = true;
  bool no_physical_aliases = true;
  bool quarantined = false;
};

class Runtime {
public:
  explicit Runtime(cuda::CudaApi& api, RuntimeConfig config);
  ~Runtime();

  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;
  Runtime(Runtime&&) = delete;
  Runtime& operator=(Runtime&&) = delete;

  [[nodiscard]] RuntimeStatus setup();
  [[nodiscard]] RuntimeStatus allocate(std::uint64_t bytes, ResidencyHint hint,
                                       RuntimeAllocation& output);
  [[nodiscard]] RuntimeStatus release(AllocationId allocation_id);
  [[nodiscard]] RuntimeStatus discard_dead(AllocationId allocation_id);
  [[nodiscard]] RuntimeStatus discard_dead(AllocationId allocation_id,
                                           std::uint64_t offset_bytes,
                                           std::uint64_t length_bytes);
  [[nodiscard]] RuntimeStatus write(AllocationId allocation_id, std::uint64_t offset,
                                    const void* source, std::uint64_t bytes);
  [[nodiscard]] RuntimeStatus read(AllocationId allocation_id, std::uint64_t offset,
                                   void* destination, std::uint64_t bytes);
  [[nodiscard]] RuntimeStatus prefetch(std::span<const AccessRange> ranges);
  [[nodiscard]] RuntimeStatus acquire_external(const ExternalLeaseRequest& request,
                                               ExternalLease& output);
  [[nodiscard]] RuntimeStatus seal_external(ExternalLeaseId lease_id, ExternalSealMode mode);
  [[nodiscard]] RuntimeStatus poll_external(ExternalLeaseId lease_id,
                                            ExternalLeasePoll& output);
  [[nodiscard]] RuntimeStatus wait_external(ExternalLeaseId lease_id,
                                            std::chrono::milliseconds timeout,
                                            ExternalLeasePoll& output);
  [[nodiscard]] RuntimeStatus execute(std::span<const AccessRange> ranges,
                                      std::uint64_t workspace_bytes,
                                      const TransactionCallback& callback);
  [[nodiscard]] RuntimeStatus drain(bool release_mappings);
  [[nodiscard]] RuntimeStatus close() noexcept;

  [[nodiscard]] cuda::abi::Context context() const noexcept;
  [[nodiscard]] cuda::abi::Device device() const noexcept;
  [[nodiscard]] std::uint64_t chunk_bytes() const noexcept;
  [[nodiscard]] std::uint64_t target_bytes() const noexcept;
  [[nodiscard]] std::optional<std::uint64_t> allocation_size(AllocationId id) const noexcept;
  [[nodiscard]] const RuntimeTelemetry& telemetry() const noexcept;
  [[nodiscard]] const RuntimeError& error() const noexcept;
  [[nodiscard]] bool poisoned() const noexcept;
  [[nodiscard]] bool async_completion_unknown() const noexcept;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace xvram::residency
