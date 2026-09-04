#pragma once

#include "platform/cuda/cuda_api.hpp"
#include "residency/compression.hpp"
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

namespace xvram::nvcomp {
class NvcompApi;
}

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

struct HostMemorySample {
  std::optional<std::uint64_t> physical_bytes;
  std::optional<std::uint64_t> available_bytes;
};

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
  CompressionMode compression_mode = CompressionMode::disabled;
  CompressionCodec compression_codec = CompressionCodec::automatic;
  std::uint64_t host_store_cap_bytes = 0;
  std::uint64_t host_headroom_bytes = 0;
  // V2 frontends leave automatic values unresolved until setup has established the CUDA
  // context. Resolve and validate against one sample, not two racing available-RAM snapshots.
  bool resolve_host_budget = false;
  std::function<HostMemorySample()> host_memory_sample;
  std::uint64_t compression_workspace_cap_bytes = 256ULL * 1024ULL * 1024ULL;
  std::uint32_t codec_slots = 2;
  std::uint32_t codec_workers = 2;
  std::optional<CompressionPath> forced_compression_path;
  // Reports real, incremental host-generation lifecycle work during potentially long allocation
  // or cleanup. The callback is advisory, may be absent, and is isolated if it throws.
  std::function<void()> lifecycle_progress;
  // Reports exact codec and PCIe generation boundaries without exposing CUDA addresses or native
  // handles. The callback is advisory, may be absent, and is isolated if it throws.
  std::function<void(const CompressionTraceEvent&)> compression_trace;
  // Optional injected dispatch owner for fault/state-machine tests. Production runtimes own and
  // dynamically load an app-local nvCOMP instance when this is null.
  nvcomp::NvcompApi* nvcomp_api = nullptr;
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

  // Phase 5 compression/backing telemetry. Logical counters describe uncompressed user bytes;
  // PCIe counters describe bytes actually transferred across the bus.
  std::uint64_t logical_bytes = 0;
  std::uint64_t host_stored_bytes = 0;
  std::uint64_t host_stored_peak_bytes = 0;
  std::uint64_t host_raw_bytes = 0;
  std::uint64_t host_raw_peak_bytes = 0;
  std::uint64_t host_compressed_bytes = 0;
  std::uint64_t host_compressed_peak_bytes = 0;
  std::uint64_t host_implicit_zero_bytes = 0;
  std::uint64_t host_invalid_bytes = 0;
  std::uint64_t host_invalid_chunks = 0;
  std::uint64_t host_implicit_zero_chunks = 0;
  std::uint64_t host_raw_chunks = 0;
  std::uint64_t host_lz4_chunks = 0;
  std::uint64_t host_store_cap_bytes = 0;
  std::uint64_t host_headroom_bytes = 0;
  std::uint64_t host_authoritative_bytes = 0;
  std::uint64_t host_authoritative_peak_bytes = 0;
  std::uint64_t host_budget_bytes = 0;
  std::uint64_t host_budget_peak_bytes = 0;
  std::uint64_t conversion_scratch_peak_bytes = 0;
  std::uint64_t logical_h2d_bytes = 0;
  std::uint64_t pcie_h2d_bytes = 0;
  std::uint64_t pcie_h2d_payload_bytes = 0;
  std::uint64_t pcie_h2d_metadata_bytes = 0;
  std::uint64_t logical_d2h_bytes = 0;
  std::uint64_t pcie_d2h_bytes = 0;
  std::uint64_t pcie_d2h_payload_bytes = 0;
  std::uint64_t pcie_d2h_metadata_bytes = 0;
  // Logical bytes represented by a verified GPU-compressed candidate that was rejected before
  // the authoritative raw fallback. This is a subset of logical_d2h_bytes, not an additional
  // transfer total.
  std::uint64_t rejected_candidate_logical_d2h_bytes = 0;
  std::uint64_t hot_allocation_d2h_bytes = 0;
  std::uint64_t non_hot_allocation_d2h_bytes = 0;
  std::uint64_t raw_path_decisions = 0;
  std::uint64_t cpu_lz4_gpu_decode_decisions = 0;
  std::uint64_t gpu_lz4_decisions = 0;
  std::uint64_t never_compress_decisions = 0;
  std::uint64_t raw_fallbacks = 0;
  std::uint64_t cpu_codec_fallbacks = 0;
  std::uint64_t gpu_codec_fallbacks = 0;
  std::uint64_t compression_attempts = 0;
  std::uint64_t compression_commits = 0;
  std::uint64_t decompression_attempts = 0;
  std::uint64_t decompression_commits = 0;
  std::uint64_t cpu_encode_operations = 0;
  // Individual 64-KiB encode jobs accepted by the bounded CPU codec pool. This internal counter
  // is intentionally distinct from cpu_encode_operations, which counts whole chunk candidates.
  std::uint64_t cpu_codec_blocks_submitted = 0;
  std::uint64_t cpu_decode_operations = 0;
  std::uint64_t gpu_encode_operations = 0;
  std::uint64_t gpu_decode_operations = 0;
  std::uint64_t codec_calibration_samples = 0;
  std::uint64_t codec_verification_failures = 0;
  std::uint64_t codec_slots_created = 0;
  std::uint64_t codec_slots_reused = 0;
  // Maximum simultaneously provisioned codec slots. codec_slots_created is cumulative across
  // budget-driven pipeline suspend/restore epochs and must not be used as a concurrency peak.
  std::uint64_t codec_slots_peak = 0;
  std::uint64_t atomic_commit_failures = 0;
  std::uint64_t expansion_rejections = 0;
  std::uint64_t generations_created = 0;
  std::uint64_t generations_committed = 0;
  std::uint64_t generations_discarded = 0;
  std::uint64_t codec_workspace_bytes = 0;
  std::uint64_t codec_workspace_peak_bytes = 0;
  std::uint64_t codec_slot_bytes = 0;
  std::uint64_t codec_slot_peak_bytes = 0;
  std::uint64_t codec_slot_capacity_bytes = 0;
  std::uint64_t spill_reserved_bytes = 0;
  std::uint64_t spill_reserved_peak_bytes = 0;
  std::uint64_t cpu_encode_nanoseconds = 0;
  std::uint64_t cpu_decode_nanoseconds = 0;
  std::uint64_t gpu_encode_nanoseconds = 0;
  std::uint64_t gpu_decode_nanoseconds = 0;
  std::uint64_t verification_nanoseconds = 0;
  std::uint64_t codec_events_recorded = 0;
  std::uint64_t codec_events_retired = 0;
  std::uint64_t safe_device_budget_minimum_bytes = 0;
  std::uint64_t managed_device_bytes_peak = 0;
  std::uint64_t device_reserve_bytes_peak = 0;
  std::uint64_t device_budget_violation_count = 0;
  bool nvcomp_available = false;
  bool nvcomp_app_local = false;
  // Populated only for a dynamically loaded library whose on-disk bytes passed the embedded
  // SHA-256 pin and whose runtime semantic version passed the exact compatibility gate.
  std::string nvcomp_version;
  std::string nvcomp_library_sha256;
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
  [[nodiscard]] RuntimeStatus discard_dead(AllocationId allocation_id, std::uint64_t offset_bytes,
                                           std::uint64_t length_bytes);
  [[nodiscard]] RuntimeStatus write(AllocationId allocation_id, std::uint64_t offset,
                                    const void* source, std::uint64_t bytes);
  [[nodiscard]] RuntimeStatus read(AllocationId allocation_id, std::uint64_t offset,
                                   void* destination, std::uint64_t bytes);
  [[nodiscard]] RuntimeStatus prefetch(std::span<const AccessRange> ranges);
  [[nodiscard]] RuntimeStatus acquire_external(const ExternalLeaseRequest& request,
                                               ExternalLease& output);
  [[nodiscard]] RuntimeStatus seal_external(ExternalLeaseId lease_id, ExternalSealMode mode);
  [[nodiscard]] RuntimeStatus poll_external(ExternalLeaseId lease_id, ExternalLeasePoll& output);
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
  [[nodiscard]] std::optional<HostChunkInfo> backing_info(AllocationId id,
                                                          std::uint64_t chunk_index) const noexcept;
  [[nodiscard]] const RuntimeTelemetry& telemetry() const noexcept;
  [[nodiscard]] const RuntimeError& error() const noexcept;
  [[nodiscard]] bool poisoned() const noexcept;
  [[nodiscard]] bool async_completion_unknown() const noexcept;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace xvram::residency
