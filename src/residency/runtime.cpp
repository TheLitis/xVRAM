#include "residency/runtime.hpp"

#include "platform/pageable_memory.hpp"
#ifdef _WIN32
#include "platform/dxgi_memory.hpp"
#endif
#include "residency/policy.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <new>
#include <numeric>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace xvram::residency {
namespace {

using Clock = std::chrono::steady_clock;

[[nodiscard]] std::optional<std::uint64_t> checked_add(const std::uint64_t left,
                                                       const std::uint64_t right) noexcept {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    return std::nullopt;
  }
  return left + right;
}

[[nodiscard]] std::optional<std::uint64_t> checked_multiply(const std::uint64_t left,
                                                            const std::uint64_t right) noexcept {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
    return std::nullopt;
  }
  return left * right;
}

[[nodiscard]] std::optional<std::uint64_t> align_up(const std::uint64_t value,
                                                    const std::uint64_t alignment) noexcept {
  if (alignment == 0) {
    return std::nullopt;
  }
  const std::uint64_t remainder = value % alignment;
  return remainder == 0 ? std::optional<std::uint64_t>{value}
                        : checked_add(value, alignment - remainder);
}

[[nodiscard]] RuntimeStatus classify_cuda(const cuda::abi::Result result) noexcept {
  return result == CUDA_ERROR_OUT_OF_MEMORY ? RuntimeStatus::device_oom
                                            : RuntimeStatus::cuda_failure;
}

} // namespace

const char* runtime_status_name(const RuntimeStatus status) noexcept {
  switch (status) {
  case RuntimeStatus::success:
    return "success";
  case RuntimeStatus::invalid_argument:
    return "invalid_argument";
  case RuntimeStatus::unavailable:
    return "unavailable";
  case RuntimeStatus::unsupported:
    return "unsupported";
  case RuntimeStatus::host_oom:
    return "host_oom";
  case RuntimeStatus::device_oom:
    return "device_oom";
  case RuntimeStatus::budget_pressure:
    return "budget_pressure";
  case RuntimeStatus::timeout:
    return "timeout";
  case RuntimeStatus::callback_skipped:
    return "callback_skipped";
  case RuntimeStatus::callback_failed:
    return "callback_failed";
  case RuntimeStatus::cuda_failure:
    return "cuda_failure";
  case RuntimeStatus::poisoned:
    return "poisoned";
  case RuntimeStatus::cleanup_failure:
    return "cleanup_failure";
  case RuntimeStatus::internal_failure:
    return "internal_failure";
  }
  return "internal_failure";
}

class Runtime::Impl {
public:
  Impl(cuda::CudaApi& api, RuntimeConfig config) : api_(api), config_(std::move(config)) {}

  struct Allocation {
    AllocationId id;
    std::uint64_t logical_bytes = 0;
    std::uint64_t mapped_bytes = 0;
    cuda::abi::DevicePointer reservation = 0;
    std::uint64_t reservation_bytes = 0;
    cuda::abi::DevicePointer logical_base = 0;
    std::unique_ptr<platform::PageableMemory> backing;
    std::vector<ChunkRecord> chunks;
    ResidencyHint hint = ResidencyHint::normal;
    bool reservation_quarantined = false;
  };

  struct Frame {
    cuda::abi::GenericAllocationHandle handle{};
    std::optional<ChunkKey> key;
    cuda::abi::DevicePointer address = 0;
    bool mapped = false;
    bool quarantined = false;
    std::uint64_t map_generation = 0;
  };

  struct StagingSlot {
    void* memory = nullptr;
    cuda::abi::Event done = nullptr;
  };

  [[nodiscard]] RuntimeStatus setup() {
    if (setup_complete_) {
      return fail(RuntimeStatus::invalid_argument, "setup", "setup",
                  "runtime has already been initialized");
    }
    const auto load = api_.load();
    if (load.status != cuda::CudaApi::LoadStatus::loaded) {
      return fail(RuntimeStatus::unavailable, "setup", "load_cuda_driver",
                  api_.error().empty() ? "CUDA driver library is unavailable" : api_.error());
    }
    if (!have_required_symbols()) {
      return fail(RuntimeStatus::unsupported, "setup", "resolve_cuda_symbols",
                  "CUDA driver lacks a Phase 3 VMM symbol");
    }
    if (const cuda::abi::Result code = api_.init_(0); code != cuda::abi::success) {
      return fail_cuda("setup", "cuInit", code);
    }

    if (config_.context_mode == RuntimeContextMode::attach_current) {
      if (config_.attached_context == nullptr) {
        return fail(RuntimeStatus::invalid_argument, "setup", "attach_context",
                    "attach-current mode requires a captured CUDA context");
      }
      context_ = config_.attached_context;
      const cuda::abi::Result code = api_.context_push_current_(context_);
      if (code != cuda::abi::success) {
        return fail_cuda("setup", "cuCtxPushCurrent", code);
      }
      context_pushed_ = true;
      if (const cuda::abi::Result device_code = api_.context_get_device_(&device_);
          device_code != cuda::abi::success) {
        return fail_cuda("setup", "cuCtxGetDevice", device_code);
      }
    } else {
      if (config_.device_ordinal < 0) {
        return fail(RuntimeStatus::invalid_argument, "setup", "device_ordinal",
                    "device ordinal must be non-negative");
      }
      if (const cuda::abi::Result code = api_.device_get_(&device_, config_.device_ordinal);
          code != cuda::abi::success) {
        return fail_cuda("setup", "cuDeviceGet", code);
      }
      if (const cuda::abi::Result code =
              api_.context_create_(&context_, CU_CTX_SCHED_AUTO, device_);
          code != cuda::abi::success) {
        return fail_cuda("setup", "cuCtxCreate", code);
      }
      owns_context_ = true;
    }

    CUmemAllocationProp property{};
    property.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    property.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    property.location.id = device_;
    property.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
    std::size_t minimum = 0;
    std::size_t recommended = 0;
    if (const cuda::abi::Result code = api_.mem_get_allocation_granularity_(
            &minimum, &property, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        code != cuda::abi::success) {
      return fail_cuda("setup", "cuMemGetAllocationGranularity(minimum)", code);
    }
    if (const cuda::abi::Result code = api_.mem_get_allocation_granularity_(
            &recommended, &property, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
        code != cuda::abi::success) {
      return fail_cuda("setup", "cuMemGetAllocationGranularity(recommended)", code);
    }
    const std::uint64_t minimum_u64 = static_cast<std::uint64_t>(minimum);
    const std::uint64_t recommended_u64 = static_cast<std::uint64_t>(recommended);
    if (minimum_u64 == 0 || recommended_u64 == 0) {
      return fail(RuntimeStatus::unsupported, "setup", "allocation_granularity",
                  "CUDA returned zero VMM allocation granularity");
    }
    const std::uint64_t divisor = std::gcd(minimum_u64, recommended_u64);
    const auto common = checked_multiply(minimum_u64 / divisor, recommended_u64);
    const auto aligned_chunk =
        common.has_value() ? align_up(config_.chunk_bytes, *common) : std::nullopt;
    if (!aligned_chunk.has_value() || *aligned_chunk == 0 ||
        *aligned_chunk > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
      return fail(RuntimeStatus::invalid_argument, "setup", "chunk_size",
                  "chunk size cannot be aligned to CUDA VMM granularities");
    }
    config_.chunk_bytes = *aligned_chunk;

    std::size_t cuda_free = 0;
    std::size_t cuda_total = 0;
    if (const cuda::abi::Result code = api_.mem_get_info_(&cuda_free, &cuda_total);
        code != cuda::abi::success) {
      return fail_cuda("setup", "cuMemGetInfo", code);
    }
    const std::uint64_t free_u64 = static_cast<std::uint64_t>(cuda_free);
    initialize_wddm_identity();
    const std::optional<std::uint64_t> wddm_available = query_wddm_available();
    wddm_budget_active_ = wddm_available.has_value();
    const std::uint64_t externally_available =
        wddm_available.has_value() ? std::min(free_u64, *wddm_available) : free_u64;
    const std::uint64_t after_headroom = externally_available > config_.device_headroom_bytes
                                             ? externally_available - config_.device_headroom_bytes
                                             : 0;
    std::uint64_t target = config_.cache_target_bytes == 0
                               ? after_headroom
                               : std::min(config_.cache_target_bytes, after_headroom);
    target -= target % config_.chunk_bytes;
    if (target < config_.chunk_bytes * 2ULL) {
      return fail(RuntimeStatus::budget_pressure, "setup", "cache_target",
                  "live CUDA budget cannot safely hold two cache chunks");
    }
    telemetry_.target_bytes = target;
    telemetry_.target_minimum_bytes = target;
    telemetry_.target_maximum_bytes = target;
    telemetry_.cuda_free_minimum_bytes = free_u64;
    telemetry_.cuda_free_end_bytes = free_u64;
    telemetry_.wddm_available_minimum_bytes = wddm_available;
    telemetry_.wddm_available_end_bytes = wddm_available;
    telemetry_.budget_samples = 1;
    configured_target_cap_ = config_.cache_target_bytes == 0
                                 ? static_cast<std::uint64_t>(cuda_total)
                                 : config_.cache_target_bytes;
    const std::uint64_t frame_budget =
        config_.workspace_reserve_bytes < target ? target - config_.workspace_reserve_bytes : 0;
    frame_capacity_ = frame_budget / config_.chunk_bytes;
    if (frame_capacity_ < 2) {
      return fail(RuntimeStatus::budget_pressure, "setup", "workspace_reserve",
                  "workspace reserve leaves fewer than two cache frames");
    }
    if (frame_capacity_ > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
      return fail(RuntimeStatus::invalid_argument, "setup", "cache_target",
                  "cache frame count exceeds the platform size limit");
    }
    frames_.resize(static_cast<std::size_t>(frame_capacity_));
    maximum_frame_capacity_ = frame_capacity_;
    last_budget_sample_ = Clock::now();

    if (config_.staging_slots < 2U || config_.staging_slots > 8U) {
      return fail(RuntimeStatus::invalid_argument, "setup", "staging_slots",
                  "staging slot count must be in [2, 8]");
    }
    const auto pinned_staging_bytes = checked_multiply(config_.chunk_bytes, config_.staging_slots);
    if (!pinned_staging_bytes.has_value()) {
      return fail(RuntimeStatus::invalid_argument, "setup", "staging_slots",
                  "pinned staging pool size overflowed");
    }

    if (!create_stream(h2d_stream_, "cuStreamCreate(h2d)") ||
        !create_stream(compute_stream_, "cuStreamCreate(compute)") ||
        !create_stream(d2h_stream_, "cuStreamCreate(d2h)") ||
        !create_event(compute_started_, "cuEventCreate(compute_start)") ||
        !create_event(compute_done_, "cuEventCreate(compute_done)")) {
      return error_.status;
    }
    const std::size_t h2d_slots = (static_cast<std::size_t>(config_.staging_slots) + 1U) / 2U;
    const std::size_t d2h_slots = static_cast<std::size_t>(config_.staging_slots) - h2d_slots;
    h2d_staging_.resize(h2d_slots);
    d2h_staging_.resize(d2h_slots);
    if (!create_staging_pool(h2d_staging_, "cuEventCreate(h2d)", "cuMemHostAlloc(h2d)") ||
        !create_staging_pool(d2h_staging_, "cuEventCreate(d2h)", "cuMemHostAlloc(d2h)")) {
      return error_.status;
    }
    telemetry_.pinned_staging_bytes = *pinned_staging_bytes;
    policy_ = config_.policy == RuntimePolicy::lru
                  ? std::unique_ptr<VictimPolicy>(std::make_unique<LruPolicy>())
                  : std::unique_ptr<VictimPolicy>(std::make_unique<ClockPolicy>());
    setup_complete_ = true;
    return RuntimeStatus::success;
  }

  [[nodiscard]] RuntimeStatus allocate(const std::uint64_t bytes, const ResidencyHint hint,
                                       RuntimeAllocation& output) {
    if (!ready() || bytes == 0 || !next_allocation_id_) {
      return fail(RuntimeStatus::invalid_argument, "allocation", "allocate",
                  "allocation size or runtime state is invalid");
    }
    const auto chunk_count = checked_add((bytes - 1ULL) / config_.chunk_bytes, 1ULL);
    const auto mapped = chunk_count.has_value()
                            ? checked_multiply(*chunk_count, config_.chunk_bytes)
                            : std::nullopt;
    const auto reservation =
        mapped.has_value() ? checked_add(*mapped, config_.chunk_bytes) : std::nullopt;
    if (!chunk_count.has_value() || !mapped.has_value() || !reservation.has_value() ||
        *chunk_count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
        *reservation > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
      return fail(RuntimeStatus::invalid_argument, "allocation", "plan_reservation",
                  "allocation layout overflowed");
    }

    auto owner = std::make_unique<Allocation>();
    owner->id = next_allocation_id_;
    owner->logical_bytes = bytes;
    owner->mapped_bytes = *mapped;
    owner->reservation_bytes = *reservation;
    owner->hint = hint;
    owner->backing = std::make_unique<platform::PageableMemory>();
    if (!owner->backing->allocate(bytes)) {
      return fail(RuntimeStatus::host_oom, "allocation", "allocate_pageable_backing",
                  "pageable host backing allocation failed");
    }
    try {
      owner->chunks.resize(static_cast<std::size_t>(*chunk_count));
    } catch (const std::bad_alloc&) {
      return fail(RuntimeStatus::host_oom, "allocation", "allocate_chunk_metadata",
                  "chunk metadata allocation failed");
    } catch (...) {
      return fail(RuntimeStatus::internal_failure, "allocation", "allocate_chunk_metadata",
                  "chunk metadata allocation raised an unexpected exception");
    }
    for (std::uint64_t index = 0; index < *chunk_count; ++index) {
      owner->chunks[static_cast<std::size_t>(index)].key = ChunkKey{owner->id, index};
    }
    try {
      auto [position, inserted] = allocations_.try_emplace(owner->id.value, std::move(owner));
      if (!inserted) {
        return fail(RuntimeStatus::internal_failure, "allocation", "register_allocation",
                    "allocation identifier was already registered");
      }
    } catch (const std::bad_alloc&) {
      return fail(RuntimeStatus::host_oom, "allocation", "register_allocation",
                  "allocation registry growth failed");
    } catch (...) {
      return fail(RuntimeStatus::internal_failure, "allocation", "register_allocation",
                  "allocation registry raised an unexpected exception");
    }

    Allocation& registered = *allocations_.at(next_allocation_id_.value);
    if (const cuda::abi::Result code = api_.mem_address_reserve_(
            &registered.reservation, static_cast<std::size_t>(registered.reservation_bytes), 0, 0,
            0);
        code != cuda::abi::success) {
      allocations_.erase(next_allocation_id_.value);
      return fail_cuda("allocation", "cuMemAddressReserve", code);
    }
    const auto logical_base =
        align_up(static_cast<std::uint64_t>(registered.reservation), config_.chunk_bytes);
    const auto logical_end = logical_base.has_value()
                                 ? checked_add(*logical_base, registered.mapped_bytes)
                                 : std::nullopt;
    const auto reservation_end = checked_add(static_cast<std::uint64_t>(registered.reservation),
                                             registered.reservation_bytes);
    if (!logical_base.has_value() || !logical_end.has_value() || !reservation_end.has_value() ||
        *logical_end > *reservation_end) {
      const cuda::abi::Result code = api_.mem_address_free_(
          registered.reservation, static_cast<std::size_t>(registered.reservation_bytes));
      if (code != cuda::abi::success) {
        registered.reservation_quarantined = true;
        return poison_cuda("allocation", "cuMemAddressFree(alignment_rollback)", code);
      }
      registered.reservation = 0;
      allocations_.erase(next_allocation_id_.value);
      return fail(RuntimeStatus::internal_failure, "allocation", "align_logical_base",
                  "padded reservation cannot contain the stable logical range");
    }
    registered.logical_base = static_cast<cuda::abi::DevicePointer>(*logical_base);
    output = RuntimeAllocation{next_allocation_id_, bytes};
    ++telemetry_.allocations_created;
    if (next_allocation_id_.value == std::numeric_limits<std::uint64_t>::max()) {
      next_allocation_id_ = {};
    } else {
      ++next_allocation_id_.value;
    }
    return RuntimeStatus::success;
  }

  [[nodiscard]] RuntimeStatus release(const AllocationId id) {
    auto found = allocations_.find(id.value);
    if (!ready() || found == allocations_.end()) {
      return fail(RuntimeStatus::invalid_argument, "allocation", "release",
                  "allocation handle is unknown");
    }
    Allocation& owner = *found->second;
    for (ChunkRecord& record : owner.chunks) {
      if (record.state == ChunkState::resident_dirty &&
          writeback(record.key) != RuntimeStatus::success) {
        return error_.status;
      }
      if (record.state == ChunkState::resident_clean &&
          unmap(record.key) != RuntimeStatus::success) {
        return error_.status;
      }
    }
    if (!owner.reservation_quarantined && owner.reservation != 0) {
      if (const cuda::abi::Result code = api_.mem_address_free_(
              owner.reservation, static_cast<std::size_t>(owner.reservation_bytes));
          code != cuda::abi::success) {
        return fail_cuda("cleanup", "cuMemAddressFree", code);
      }
      owner.reservation = 0;
    }
    if (!owner.backing->release()) {
      return fail(RuntimeStatus::cleanup_failure, "cleanup", "release_pageable_backing",
                  "pageable backing release failed");
    }
    allocations_.erase(found);
    ++telemetry_.allocations_released;
    return RuntimeStatus::success;
  }

  [[nodiscard]] RuntimeStatus write(const AllocationId id, const std::uint64_t offset,
                                    const void* source, const std::uint64_t bytes) {
    Allocation* owner = allocation(id);
    if (!valid_host_range(owner, offset, source, bytes)) {
      return fail(RuntimeStatus::invalid_argument, "host_access", "write",
                  "host write range is invalid");
    }
    if (const RuntimeStatus status = flush_overlapping(*owner, offset, bytes, true);
        status != RuntimeStatus::success) {
      return status;
    }
    std::memcpy(static_cast<std::byte*>(owner->backing->data()) + offset, source,
                static_cast<std::size_t>(bytes));
    return RuntimeStatus::success;
  }

  [[nodiscard]] RuntimeStatus read(const AllocationId id, const std::uint64_t offset,
                                   void* destination, const std::uint64_t bytes) {
    Allocation* owner = allocation(id);
    if (!valid_host_range(owner, offset, destination, bytes)) {
      return fail(RuntimeStatus::invalid_argument, "host_access", "read",
                  "host read range is invalid");
    }
    if (const RuntimeStatus status = flush_overlapping(*owner, offset, bytes, false);
        status != RuntimeStatus::success) {
      return status;
    }
    std::memcpy(destination, static_cast<const std::byte*>(owner->backing->data()) + offset,
                static_cast<std::size_t>(bytes));
    return RuntimeStatus::success;
  }

  [[nodiscard]] RuntimeStatus prefetch(const std::span<const AccessRange> ranges) {
    std::vector<AllocationLayout> layouts = allocation_layouts();
    const AccessPlanResult plan =
        normalize_and_split_accesses(layouts, ranges, config_.chunk_bytes);
    if (!plan) {
      return fail(RuntimeStatus::invalid_argument, "transaction", "prefetch",
                  std::string("invalid access ranges: ") +
                      std::string(access_plan_error_name(plan.error)));
    }
    if (const RuntimeStatus status = refresh_budget(plan.chunks.size(), 0, false);
        status != RuntimeStatus::success) {
      return status;
    }
    if (plan.chunks.size() > frame_capacity_) {
      return fail(RuntimeStatus::budget_pressure, "transaction", "prefetch",
                  "prefetch working set exceeds the live cache target");
    }
    for (const ChunkAccessPlan& chunk_plan : plan.chunks) {
      bool hit = false;
      if (const RuntimeStatus status = ensure_resident(chunk_plan, true, hit);
          status != RuntimeStatus::success) {
        return status;
      }
    }
    ++telemetry_.prefetches;
    return RuntimeStatus::success;
  }

  [[nodiscard]] RuntimeStatus execute(const std::span<const AccessRange> ranges,
                                      const std::uint64_t workspace_bytes,
                                      const TransactionCallback& callback) {
    if (!ready() || !callback) {
      return fail(RuntimeStatus::invalid_argument, "transaction", "execute",
                  "transaction callback or runtime state is invalid");
    }
    std::vector<AllocationLayout> layouts = allocation_layouts();
    const AccessPlanResult plan =
        normalize_and_split_accesses(layouts, ranges, config_.chunk_bytes);
    if (!plan || plan.chunks.empty()) {
      return fail(RuntimeStatus::invalid_argument, "transaction", "normalize_accesses",
                  std::string("invalid access ranges: ") +
                      std::string(access_plan_error_name(plan.error)));
    }
    if (const RuntimeStatus status = refresh_budget(plan.chunks.size(), workspace_bytes, false);
        status != RuntimeStatus::success) {
      return status;
    }
    if (plan.chunks.size() > frame_capacity_) {
      return fail(RuntimeStatus::budget_pressure, "transaction", "working_set",
                  "declared working set exceeds the live cache target");
    }
    if (const RuntimeStatus status = ensure_workspace(workspace_bytes);
        status != RuntimeStatus::success) {
      return status;
    }

    ++telemetry_.transactions_submitted;
    std::vector<ChunkRecord*> pinned;
    pinned.reserve(plan.chunks.size());
    const auto retire_prelaunch_generations = [&]() noexcept {
      bool retired = true;
      for (ChunkRecord* record : pinned) {
        if (record != nullptr && complete_event_generation(*record, record->event_generation) !=
                                     EventGenerationResult::success) {
          retired = false;
        }
      }
      unpin(pinned);
      return retired;
    };
    for (const ChunkAccessPlan& chunk_plan : plan.chunks) {
      bool hit = false;
      if (const RuntimeStatus status = ensure_resident(chunk_plan, false, hit);
          status != RuntimeStatus::success) {
        if (!retire_prelaunch_generations()) {
          return poison("transaction", "prepare_rollback",
                        "pre-launch working-set generations could not retire");
        }
        if (status == RuntimeStatus::budget_pressure) {
          const RuntimeStatus shrink_status = refresh_budget(0, workspace_bytes_, true);
          if (shrink_status != RuntimeStatus::success &&
              shrink_status != RuntimeStatus::budget_pressure) {
            return shrink_status;
          }
        }
        return status;
      }
      ChunkRecord* record = chunk(chunk_plan.key);
      if (record == nullptr || begin_event_generation(*record) != EventGenerationResult::success) {
        (void)retire_prelaunch_generations();
        return poison("transaction", "pin_working_set", "chunk event generation cannot be started");
      }
      ++record->pin_count;
      record->in_current_working_set = true;
      pinned.push_back(record);
    }

    std::vector<ResolvedRange> resolved;
    try {
      resolved.reserve(plan.normalized_ranges.size());
      for (const AccessRange& range : plan.normalized_ranges) {
        const Allocation* owner = allocation(range.allocation_id);
        if (owner == nullptr) {
          (void)retire_prelaunch_generations();
          return poison("transaction", "resolve_range", "normalized allocation disappeared");
        }
        resolved.push_back(ResolvedRange{range.allocation_id, range.offset_bytes,
                                         range.length_bytes, range.mode,
                                         owner->logical_base + range.offset_bytes});
      }
    } catch (const std::bad_alloc&) {
      if (!retire_prelaunch_generations()) {
        return poison("transaction", "resolve_rollback",
                      "pre-launch generations could not retire after host OOM");
      }
      return fail(RuntimeStatus::host_oom, "transaction", "resolve_ranges",
                  "resolved range allocation failed");
    }

    if (const cuda::abi::Result code = api_.event_record_(compute_started_, compute_stream_);
        code != cuda::abi::success) {
      if (!retire_prelaunch_generations()) {
        return poison("transaction", "compute_start_rollback",
                      "pre-launch generations could not retire after event failure");
      }
      return poison_cuda("transaction", "cuEventRecord(compute_start)", code);
    }
    RuntimeStatus callback_status = RuntimeStatus::internal_failure;
    try {
      callback_status =
          callback(TransactionContext{resolved, workspace_, workspace_bytes, compute_stream_});
    } catch (...) {
      callback_status = RuntimeStatus::callback_failed;
    }
    const cuda::abi::Result record_result = api_.event_record_(compute_done_, compute_stream_);
    if (record_result != cuda::abi::success) {
      async_resources_quarantined_ = true;
      unpin(pinned);
      return poison_cuda("transaction", "cuEventRecord(compute_done)", record_result);
    }
    if (callback_status == RuntimeStatus::callback_skipped) {
      const RuntimeStatus drain_status = wait_event(compute_done_, "callback_skip_drain");
      if (drain_status == RuntimeStatus::success) {
        for (ChunkRecord* record : pinned) {
          if (complete_event_generation(*record, record->event_generation) !=
              EventGenerationResult::success) {
            unpin(pinned);
            return poison("transaction", "callback_skip_retire",
                          "skipped callback generation could not retire");
          }
        }
      }
      unpin(pinned);
      return drain_status == RuntimeStatus::success ? RuntimeStatus::callback_skipped
                                                    : drain_status;
    }
    if (callback_status != RuntimeStatus::success) {
      const RuntimeStatus drain_status = wait_event(compute_done_, "callback_drain");
      if (drain_status != RuntimeStatus::success) {
        unpin(pinned);
        return drain_status;
      }
      for (ChunkRecord* record : pinned) {
        if (complete_event_generation(*record, record->event_generation) !=
            EventGenerationResult::success) {
          unpin(pinned);
          return poison("transaction", "callback_failure_retire",
                        "failed callback generation could not retire");
        }
      }
      unpin(pinned);
      return poison("transaction", "callback", "transaction callback rejected submission",
                    RuntimeStatus::callback_failed);
    }
    for (const ChunkAccessPlan& chunk_plan : plan.chunks) {
      if (!chunk_plan.marks_dirty) {
        continue;
      }
      ChunkRecord* record = chunk(chunk_plan.key);
      if (record != nullptr && record->state == ChunkState::resident_clean) {
        if (transition_chunk_state(record->state, ChunkState::resident_dirty) !=
            StateTransitionResult::success) {
          unpin(pinned);
          return poison_transition("transaction", "mark_dirty", "illegal dirty transition");
        }
      }
    }
    if (const RuntimeStatus status = wait_event(compute_done_, "transaction_complete");
        status != RuntimeStatus::success) {
      unpin(pinned);
      return status;
    }
    float elapsed = 0.0F;
    if (const cuda::abi::Result code =
            api_.event_elapsed_time_(&elapsed, compute_started_, compute_done_);
        code != cuda::abi::success) {
      unpin(pinned);
      return poison_cuda("transaction", "cuEventElapsedTime", code);
    }
    telemetry_.last_transaction_ms = static_cast<double>(elapsed);
    for (ChunkRecord* record : pinned) {
      if (complete_event_generation(*record, record->event_generation) !=
          EventGenerationResult::success) {
        unpin(pinned);
        return poison("transaction", "retire_generation",
                      "working-set event generation could not retire");
      }
    }
    unpin(pinned);
    ++telemetry_.transactions_completed;
    if (config_.maximum_transaction_duration.count() > 0 &&
        static_cast<double>(elapsed) >
            static_cast<double>(config_.maximum_transaction_duration.count())) {
      launches_blocked_ = true;
      ++telemetry_.watchdog_rejections;
      return fail(RuntimeStatus::timeout, "transaction", "duration_guard",
                  "transaction exceeded the configured kernel safety duration");
    }
    return RuntimeStatus::success;
  }

  [[nodiscard]] RuntimeStatus drain(const bool release_mappings) {
    if (!setup_complete_) {
      return RuntimeStatus::success;
    }
    for (auto& [id, owner] : allocations_) {
      (void)id;
      for (ChunkRecord& record : owner->chunks) {
        if (record.state == ChunkState::resident_dirty) {
          if (const RuntimeStatus status = writeback(record.key);
              status != RuntimeStatus::success) {
            return status;
          }
        }
      }
    }
    if (release_mappings) {
      for (auto& [id, owner] : allocations_) {
        (void)id;
        for (ChunkRecord& record : owner->chunks) {
          if (record.state == ChunkState::resident_clean) {
            if (const RuntimeStatus status = unmap(record.key); status != RuntimeStatus::success) {
              return status;
            }
          }
        }
      }
    }
    return RuntimeStatus::success;
  }

  [[nodiscard]] RuntimeStatus close() noexcept {
    if (closed_) {
      return close_status_;
    }
    RuntimeStatus aggregate = RuntimeStatus::success;
    const auto note_cleanup_failure = [&]() noexcept {
      aggregate = RuntimeStatus::cleanup_failure;
    };

    // Close is deliberately best-effort. A failure on one frame must not prevent us from
    // retiring other independently safe mappings. Conversely, transient or structurally
    // inconsistent mappings remain quarantined rather than being unmapped speculatively.
    if (setup_complete_) {
      for (std::size_t index = 0; index < frames_.size(); ++index) {
        Frame& frame = frames_[index];
        if (!frame.mapped) {
          continue;
        }
        if (!frame.key.has_value()) {
          telemetry_.quarantined = true;
          poisoned_ = true;
          note_cleanup_failure();
          continue;
        }

        Allocation* owner = allocation(frame.key->allocation_id);
        ChunkRecord* record = chunk(*frame.key);
        const bool linked =
            owner != nullptr && record != nullptr &&
            record->frame_index == std::optional<std::uint64_t>{index} &&
            frame.address == owner->logical_base + frame.key->chunk_index * config_.chunk_bytes;
        if (!linked) {
          if (owner != nullptr) {
            quarantine_mapping(*owner);
          } else {
            telemetry_.quarantined = true;
            poisoned_ = true;
          }
          note_cleanup_failure();
          continue;
        }
        if (frame.quarantined) {
          note_cleanup_failure();
          continue;
        }

        RuntimeStatus status = RuntimeStatus::cleanup_failure;
        switch (record->state) {
        case ChunkState::resident_dirty:
          status = writeback(record->key);
          if (status == RuntimeStatus::success) {
            status = unmap(record->key);
          }
          break;
        case ChunkState::resident_clean:
          status = unmap(record->key);
          break;
        case ChunkState::evicting:
          status = is_safe_to_unmap(*record) ? retire_evicting_mapping(*record, *owner, frame)
                                             : RuntimeStatus::cleanup_failure;
          break;
        case ChunkState::mapping:
          if (record->pin_count == 0 && !record->in_current_working_set &&
              !record->staging_slot.has_value() && record->event_generation == 0 &&
              record->completed_generation == 0) {
            status =
                rollback_provisional_mapping(*record, *owner, frame, "cuMemUnmap(close_rollback)");
          }
          break;
        default:
          break;
        }
        if (status != RuntimeStatus::success) {
          if (frame.mapped) {
            quarantine_mapping(*owner);
          }
          note_cleanup_failure();
        }
      }
    }

    const bool unidentified_live_mapping =
        std::any_of(frames_.begin(), frames_.end(), [&](const Frame& frame) {
          return frame.mapped &&
                 (!frame.key.has_value() || allocation(frame.key->allocation_id) == nullptr);
        });
    for (auto& [id, owner] : allocations_) {
      (void)id;
      const Allocation* const registered_owner = owner.get();
      const bool live_mapping =
          unidentified_live_mapping ||
          std::any_of(frames_.begin(), frames_.end(), [registered_owner](const Frame& frame) {
            if (!frame.mapped) {
              return false;
            }
            if (frame.key.has_value() && frame.key->allocation_id == registered_owner->id) {
              return true;
            }
            return frame.address >= registered_owner->reservation &&
                   frame.address - registered_owner->reservation <
                       registered_owner->reservation_bytes;
          });
      if (live_mapping) {
        quarantine_mapping(*owner);
        note_cleanup_failure();
      } else if (owner->reservation_quarantined) {
        note_cleanup_failure();
      } else if (owner->reservation != 0) {
        if (api_.mem_address_free_(owner->reservation,
                                   static_cast<std::size_t>(owner->reservation_bytes)) !=
            cuda::abi::success) {
          note_cleanup_failure();
        } else {
          owner->reservation = 0;
        }
      }
      if (owner->backing != nullptr && !owner->backing->release()) {
        note_cleanup_failure();
      }
    }
    allocations_.clear();
    if (workspace_ != 0 && !async_resources_quarantined_) {
      if (api_.mem_free_(workspace_) != cuda::abi::success) {
        note_cleanup_failure();
      } else {
        workspace_ = 0;
        workspace_bytes_ = 0;
      }
    }
    for (Frame& frame : frames_) {
      if (frame.handle == 0) {
        continue;
      }
      if (frame.mapped) {
        telemetry_.quarantined = true;
        poisoned_ = true;
        note_cleanup_failure();
        if (api_.mem_release_(frame.handle) != cuda::abi::success) {
          note_cleanup_failure();
        } else {
          ++telemetry_.handles_released;
          frame.handle = 0;
        }
        continue;
      }
      if (api_.mem_release_(frame.handle) != cuda::abi::success) {
        note_cleanup_failure();
      } else {
        ++telemetry_.handles_released;
        frame.handle = 0;
      }
    }
    if (async_resources_quarantined_) {
      // No trustworthy completion boundary exists. Keep memory, event, stream, workspace, and
      // owned-context resources alive until the quarantine process exits; destroying any of them
      // here could race an unobserved DMA transfer or kernel.
      note_cleanup_failure();
    } else {
      destroy_staging_pool(h2d_staging_, aggregate);
      destroy_staging_pool(d2h_staging_, aggregate);
      destroy_event(compute_started_, aggregate);
      destroy_event(compute_done_, aggregate);
      destroy_stream(h2d_stream_, aggregate);
      destroy_stream(compute_stream_, aggregate);
      destroy_stream(d2h_stream_, aggregate);
    }
    if (context_pushed_) {
      cuda::abi::Context popped = nullptr;
      if (api_.context_pop_current_(&popped) != cuda::abi::success || popped != context_) {
        aggregate = RuntimeStatus::cleanup_failure;
      }
      context_pushed_ = false;
    }
    if (owns_context_ && context_ != nullptr && !async_resources_quarantined_) {
      if (api_.context_destroy_(context_) != cuda::abi::success) {
        aggregate = RuntimeStatus::cleanup_failure;
      }
    }
    owns_context_ = false;
    context_ = nullptr;
    setup_complete_ = false;
    closed_ = true;
    close_status_ = aggregate;
    return aggregate;
  }

  [[nodiscard]] cuda::abi::Context context() const noexcept {
    return context_;
  }
  [[nodiscard]] cuda::abi::Device device() const noexcept {
    return device_;
  }
  [[nodiscard]] std::uint64_t chunk_bytes() const noexcept {
    return config_.chunk_bytes;
  }
  [[nodiscard]] std::uint64_t target_bytes() const noexcept {
    return telemetry_.target_bytes;
  }
  [[nodiscard]] const RuntimeTelemetry& telemetry() const noexcept {
    return telemetry_;
  }
  [[nodiscard]] const RuntimeError& error() const noexcept {
    return error_;
  }
  [[nodiscard]] bool poisoned() const noexcept {
    return poisoned_;
  }

  [[nodiscard]] bool async_completion_unknown() const noexcept {
    return async_resources_quarantined_;
  }

  [[nodiscard]] std::optional<std::uint64_t> allocation_size(const AllocationId id) const noexcept {
    const Allocation* owner = allocation(id);
    return owner == nullptr ? std::nullopt : std::optional<std::uint64_t>{owner->logical_bytes};
  }

private:
  [[nodiscard]] bool have_required_symbols() const noexcept {
    return api_.init_ != nullptr && api_.device_get_ != nullptr &&
           api_.context_get_device_ != nullptr && api_.context_create_ != nullptr &&
           api_.context_destroy_ != nullptr && api_.context_push_current_ != nullptr &&
           api_.context_pop_current_ != nullptr && api_.mem_get_info_ != nullptr &&
           api_.mem_get_allocation_granularity_ != nullptr &&
           api_.mem_address_reserve_ != nullptr && api_.mem_address_free_ != nullptr &&
           api_.mem_create_ != nullptr && api_.mem_release_ != nullptr &&
           api_.mem_map_ != nullptr && api_.mem_unmap_ != nullptr &&
           api_.mem_set_access_ != nullptr && api_.mem_alloc_ != nullptr &&
           api_.mem_free_ != nullptr && api_.mem_host_alloc_ != nullptr &&
           api_.mem_free_host_ != nullptr && api_.memcpy_h2d_async_ != nullptr &&
           api_.memcpy_d2h_async_ != nullptr && api_.stream_create_ != nullptr &&
           api_.stream_destroy_ != nullptr && api_.event_create_ != nullptr &&
           api_.event_destroy_ != nullptr && api_.event_record_ != nullptr &&
           api_.event_query_ != nullptr && api_.event_elapsed_time_ != nullptr;
  }

  void initialize_wddm_identity() noexcept {
#ifdef _WIN32
    if (api_.device_get_luid_ == nullptr) {
      return;
    }
    unsigned int node_mask = 0;
    platform::AdapterLuid luid{};
    if (api_.device_get_luid_(reinterpret_cast<char*>(luid.data()), &node_mask, device_) ==
        cuda::abi::success) {
      wddm_luid_ = luid;
      wddm_node_mask_ = node_mask;
    }
#endif
  }

  [[nodiscard]] std::optional<std::uint64_t> query_wddm_available() const {
#ifdef _WIN32
    if (!wddm_luid_.has_value()) {
      return std::nullopt;
    }
    std::vector<probe::Diagnostic> diagnostics;
    const auto adapter = platform::query_dxgi_memory(*wddm_luid_, wddm_node_mask_, diagnostics);
    if (!adapter.has_value() || !adapter->local.has_value()) {
      return std::nullopt;
    }
    const std::uint64_t budget = adapter->local->budget_bytes;
    const std::uint64_t usage = adapter->local->current_usage_bytes;
    return usage < budget ? budget - usage : 0;
#else
    return std::nullopt;
#endif
  }

  [[nodiscard]] bool ready() const noexcept {
    return setup_complete_ && !closed_ && !poisoned_ && !launches_blocked_;
  }

  [[nodiscard]] RuntimeStatus fail(const RuntimeStatus status, std::string stage,
                                   std::string operation, std::string message,
                                   const std::optional<std::int64_t> native = std::nullopt) {
    error_ =
        RuntimeError{status, std::move(stage), std::move(operation), std::move(message), native};
    return status;
  }

  [[nodiscard]] RuntimeStatus fail_cuda(const char* stage, const char* operation,
                                        const cuda::abi::Result code) {
    return fail(classify_cuda(code), stage, operation, cuda_message(code),
                static_cast<std::int64_t>(code));
  }

  [[nodiscard]] RuntimeStatus poison_cuda(const char* stage, const char* operation,
                                          const cuda::abi::Result code) {
    poisoned_ = true;
    telemetry_.quarantined = true;
    return fail(RuntimeStatus::poisoned, stage, operation, cuda_message(code),
                static_cast<std::int64_t>(code));
  }

  [[nodiscard]] RuntimeStatus poison(const char* stage, const char* operation, const char* message,
                                     const RuntimeStatus status = RuntimeStatus::poisoned) {
    poisoned_ = true;
    telemetry_.quarantined = true;
    return fail(status, stage, operation, message);
  }

  [[nodiscard]] RuntimeStatus
  poison_transition(const char* stage, const char* operation, const char* message,
                    const RuntimeStatus status = RuntimeStatus::poisoned) {
    ++telemetry_.unsafe_transitions;
    return poison(stage, operation, message, status);
  }

  [[nodiscard]] std::string cuda_message(const cuda::abi::Result code) const {
    const char* message = nullptr;
    if (api_.get_error_string_ != nullptr &&
        api_.get_error_string_(code, &message) == cuda::abi::success && message != nullptr) {
      return message;
    }
    return "CUDA error " + std::to_string(static_cast<std::int64_t>(code));
  }

  [[nodiscard]] bool create_stream(cuda::abi::Stream& stream, const char* operation) {
    const cuda::abi::Result code = api_.stream_create_(&stream, CU_STREAM_NON_BLOCKING);
    if (code == cuda::abi::success) {
      return true;
    }
    (void)fail_cuda("setup", operation, code);
    return false;
  }

  [[nodiscard]] bool create_event(cuda::abi::Event& event, const char* operation) {
    const cuda::abi::Result code = api_.event_create_(&event, CU_EVENT_DEFAULT);
    if (code == cuda::abi::success) {
      return true;
    }
    (void)fail_cuda("setup", operation, code);
    return false;
  }

  [[nodiscard]] bool create_staging_pool(std::vector<StagingSlot>& pool,
                                         const char* event_operation,
                                         const char* allocation_operation) {
    for (StagingSlot& slot : pool) {
      if (!create_event(slot.done, event_operation)) {
        return false;
      }
      const cuda::abi::Result code = api_.mem_host_alloc_(
          &slot.memory, static_cast<std::size_t>(config_.chunk_bytes), CU_MEMHOSTALLOC_PORTABLE);
      if (code != cuda::abi::success) {
        (void)fail_cuda("setup", allocation_operation, code);
        return false;
      }
    }
    return true;
  }

  void destroy_stream(cuda::abi::Stream& stream, RuntimeStatus& aggregate) noexcept {
    if (stream != nullptr && api_.stream_destroy_(stream) != cuda::abi::success) {
      aggregate = RuntimeStatus::cleanup_failure;
    }
    stream = nullptr;
  }

  void destroy_event(cuda::abi::Event& event, RuntimeStatus& aggregate) noexcept {
    if (event != nullptr && api_.event_destroy_(event) != cuda::abi::success) {
      aggregate = RuntimeStatus::cleanup_failure;
    }
    event = nullptr;
  }

  void destroy_staging_pool(std::vector<StagingSlot>& pool, RuntimeStatus& aggregate) noexcept {
    for (StagingSlot& slot : pool) {
      if (slot.memory != nullptr && api_.mem_free_host_(slot.memory) != cuda::abi::success) {
        aggregate = RuntimeStatus::cleanup_failure;
      }
      slot.memory = nullptr;
      destroy_event(slot.done, aggregate);
    }
    pool.clear();
  }

  [[nodiscard]] RuntimeStatus wait_event(const cuda::abi::Event event, const char* operation) {
    const Clock::time_point deadline = Clock::now() + config_.stall_timeout;
    for (;;) {
      const cuda::abi::Result code = api_.event_query_(event);
      if (code == cuda::abi::success) {
        return RuntimeStatus::success;
      }
      if (code != CUDA_ERROR_NOT_READY) {
        async_resources_quarantined_ = true;
        return poison_cuda("event", operation, code);
      }
      if (Clock::now() >= deadline) {
        async_resources_quarantined_ = true;
        return poison("event", operation,
                      "CUDA event did not retire before the stall deadline; the in-flight "
                      "generation is quarantined",
                      RuntimeStatus::timeout);
      }
      std::this_thread::yield();
    }
  }

  [[nodiscard]] Allocation* allocation(const AllocationId id) noexcept {
    const auto found = allocations_.find(id.value);
    return found == allocations_.end() ? nullptr : found->second.get();
  }

  [[nodiscard]] const Allocation* allocation(const AllocationId id) const noexcept {
    const auto found = allocations_.find(id.value);
    return found == allocations_.end() ? nullptr : found->second.get();
  }

  [[nodiscard]] ChunkRecord* chunk(const ChunkKey key) noexcept {
    Allocation* owner = allocation(key.allocation_id);
    return owner == nullptr || key.chunk_index >= owner->chunks.size()
               ? nullptr
               : &owner->chunks[static_cast<std::size_t>(key.chunk_index)];
  }

  [[nodiscard]] std::vector<AllocationLayout> allocation_layouts() const {
    std::vector<AllocationLayout> layouts;
    layouts.reserve(allocations_.size());
    for (const auto& [id, owner] : allocations_) {
      (void)id;
      layouts.push_back(AllocationLayout{owner->id, owner->logical_bytes});
    }
    return layouts;
  }

  [[nodiscard]] bool valid_host_range(const Allocation* owner, const std::uint64_t offset,
                                      const void* pointer,
                                      const std::uint64_t bytes) const noexcept {
    if (owner == nullptr || pointer == nullptr || bytes == 0 ||
        bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
        offset > owner->logical_bytes) {
      return false;
    }
    return bytes <= owner->logical_bytes - offset;
  }

  [[nodiscard]] RuntimeStatus flush_overlapping(Allocation& owner, const std::uint64_t offset,
                                                const std::uint64_t bytes, const bool invalidate) {
    const std::uint64_t first = offset / config_.chunk_bytes;
    const std::uint64_t last = (offset + bytes - 1ULL) / config_.chunk_bytes;
    for (std::uint64_t index = first; index <= last; ++index) {
      ChunkRecord& record = owner.chunks[static_cast<std::size_t>(index)];
      if (record.state == ChunkState::resident_dirty) {
        if (const RuntimeStatus status = writeback(record.key); status != RuntimeStatus::success) {
          return status;
        }
      }
      if (invalidate && record.state == ChunkState::resident_clean) {
        if (const RuntimeStatus status = unmap(record.key); status != RuntimeStatus::success) {
          return status;
        }
      }
    }
    return RuntimeStatus::success;
  }

  [[nodiscard]] Frame* frame_for(const ChunkKey key) noexcept {
    ChunkRecord* record = chunk(key);
    if (record == nullptr || !record->frame_index.has_value() ||
        *record->frame_index >= frames_.size()) {
      return nullptr;
    }
    return &frames_[static_cast<std::size_t>(*record->frame_index)];
  }

  [[nodiscard]] RuntimeStatus create_handle(Frame& frame) {
    CUmemAllocationProp property{};
    property.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    property.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    property.location.id = device_;
    property.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
    cuda::abi::Result code = api_.mem_create_(
        &frame.handle, static_cast<std::size_t>(config_.chunk_bytes), &property, 0);
    if (code == CUDA_ERROR_OUT_OF_MEMORY) {
      ++telemetry_.target_oom_retries;
      const std::size_t frame_index = static_cast<std::size_t>(&frame - frames_.data());
      const RuntimeStatus refresh = refresh_budget(0, workspace_bytes_, true);
      if (refresh != RuntimeStatus::success) {
        return refresh;
      }
      if (frame_index >= frame_capacity_) {
        return fail(RuntimeStatus::budget_pressure, "cache", "cuMemCreate_retry",
                    "live budget shrink removed the frame selected for allocation");
      }
      code = api_.mem_create_(&frame.handle, static_cast<std::size_t>(config_.chunk_bytes),
                              &property, 0);
    }
    if (code != cuda::abi::success) {
      return fail_cuda("cache", "cuMemCreate", code);
    }
    ++telemetry_.handles_created;
    return RuntimeStatus::success;
  }

  [[nodiscard]] RuntimeStatus refresh_budget(const std::size_t required_frames,
                                             const std::uint64_t required_workspace,
                                             const bool force) {
    const Clock::time_point now = Clock::now();
    if (!force && config_.budget_poll_interval.count() > 0 &&
        now - last_budget_sample_ < config_.budget_poll_interval) {
      return RuntimeStatus::success;
    }
    last_budget_sample_ = now;
    std::size_t cuda_free = 0;
    std::size_t cuda_total = 0;
    if (const cuda::abi::Result code = api_.mem_get_info_(&cuda_free, &cuda_total);
        code != cuda::abi::success) {
      return fail_cuda("budget", "cuMemGetInfo", code);
    }
    (void)cuda_total;
    const std::uint64_t free_bytes = static_cast<std::uint64_t>(cuda_free);
    ++telemetry_.budget_samples;
    telemetry_.cuda_free_end_bytes = free_bytes;
    telemetry_.cuda_free_minimum_bytes = std::min(telemetry_.cuda_free_minimum_bytes, free_bytes);
    const auto managed_with_workspace = checked_add(telemetry_.resident_bytes, workspace_bytes_);
    const std::uint64_t reclaimable =
        managed_with_workspace.has_value() &&
                free_bytes <= std::numeric_limits<std::uint64_t>::max() - *managed_with_workspace
            ? free_bytes + *managed_with_workspace
            : std::numeric_limits<std::uint64_t>::max();
    std::uint64_t externally_reclaimable = reclaimable;
    const std::optional<std::uint64_t> wddm_available = query_wddm_available();
    if (wddm_budget_active_ && !wddm_available.has_value()) {
      return fail(RuntimeStatus::unavailable, "budget", "query_wddm",
                  "WDDM budget observation became unavailable");
    }
    if (wddm_available.has_value()) {
      const std::uint64_t wddm_reclaimable =
          managed_with_workspace.has_value() &&
                  *wddm_available <=
                      std::numeric_limits<std::uint64_t>::max() - *managed_with_workspace
              ? *wddm_available + *managed_with_workspace
              : std::numeric_limits<std::uint64_t>::max();
      externally_reclaimable = std::min(externally_reclaimable, wddm_reclaimable);
      telemetry_.wddm_available_end_bytes = *wddm_available;
      telemetry_.wddm_available_minimum_bytes =
          telemetry_.wddm_available_minimum_bytes.has_value()
              ? std::min(*telemetry_.wddm_available_minimum_bytes, *wddm_available)
              : wddm_available;
    }
    const std::uint64_t after_headroom =
        externally_reclaimable > config_.device_headroom_bytes
            ? externally_reclaimable - config_.device_headroom_bytes
            : 0;
    std::uint64_t observed_target = std::min(configured_target_cap_, after_headroom);
    observed_target -= observed_target % config_.chunk_bytes;
    const auto required_frame_bytes =
        checked_multiply(static_cast<std::uint64_t>(required_frames), config_.chunk_bytes);
    const auto required_total = required_frame_bytes.has_value()
                                    ? checked_add(*required_frame_bytes, required_workspace)
                                    : std::nullopt;
    std::uint64_t observed_frames =
        observed_target > config_.workspace_reserve_bytes
            ? (observed_target - config_.workspace_reserve_bytes) / config_.chunk_bytes
            : 0;
    observed_frames = std::min(observed_frames, maximum_frame_capacity_);

    // Budget pressure must first retire managed residency down to the newly safe target.  The
    // caller may still be unable to launch its next working set, but returning before this drain
    // would leave handles and mappings above the live CUDA/WDDM budget.
    if (observed_frames < frame_capacity_) {
      // A forced refresh can run after earlier chunks of the next working set have already been
      // pinned. Validate the complete shrink tail before changing anything: a partial shrink that
      // later reaches one of those chunks would either violate the pin invariant or leave frame
      // capacity accounting half-mutated. The caller unwinds its pins and may retry after the
      // budget stabilizes.
      for (std::uint64_t index = observed_frames; index < frame_capacity_; ++index) {
        const Frame& frame = frames_[static_cast<std::size_t>(index)];
        if (!frame.key.has_value()) {
          continue;
        }
        const ChunkRecord* record = chunk(*frame.key);
        if (record == nullptr) {
          return poison("budget", "shrink_preflight", "tail frame references an unknown chunk");
        }
        if (!is_victim_eligible(*record)) {
          return fail(RuntimeStatus::budget_pressure, "budget", "shrink_preflight",
                      "active working-set or in-flight frame defers live target shrink");
        }
      }
      for (std::uint64_t index = observed_frames; index < frame_capacity_; ++index) {
        Frame& frame = frames_[static_cast<std::size_t>(index)];
        if (frame.key.has_value()) {
          ChunkRecord* record = chunk(*frame.key);
          if (record == nullptr) {
            return poison("budget", "shrink", "tail frame references an unknown chunk");
          }
          const bool dirty = record->state == ChunkState::resident_dirty;
          if (dirty && writeback(record->key) != RuntimeStatus::success) {
            return error_.status;
          }
          if (unmap(record->key) != RuntimeStatus::success) {
            return error_.status;
          }
          if (dirty) {
            ++telemetry_.dirty_evictions;
          } else {
            ++telemetry_.clean_evictions;
          }
        }
        if (frame.handle != 0) {
          if (const cuda::abi::Result code = api_.mem_release_(frame.handle);
              code != cuda::abi::success) {
            return fail_cuda("budget", "cuMemRelease", code);
          }
          frame.handle = 0;
          ++telemetry_.handles_released;
        }
      }
      frame_capacity_ = observed_frames;
      telemetry_.target_bytes = observed_target;
      telemetry_.target_minimum_bytes = std::min(telemetry_.target_minimum_bytes, observed_target);
      ++telemetry_.budget_shrinks;
      consecutive_growth_samples_ = 0;
    }

    if (!required_total.has_value() || observed_target < *required_total ||
        observed_target <= config_.workspace_reserve_bytes) {
      return fail(RuntimeStatus::budget_pressure, "budget", "working_set",
                  "live CUDA budget fell below the next transaction working set");
    }
    if (observed_frames < static_cast<std::uint64_t>(required_frames)) {
      return fail(RuntimeStatus::budget_pressure, "budget", "working_set",
                  "live CUDA budget cannot retain the next transaction frames");
    }

    if (observed_frames > frame_capacity_) {
      if (++consecutive_growth_samples_ >= 10U &&
          observed_frames >= frame_capacity_ + std::uint64_t{2}) {
        frame_capacity_ = std::min(observed_frames, frame_capacity_ + std::uint64_t{2});
        telemetry_.target_bytes = std::min(observed_target, frame_capacity_ * config_.chunk_bytes +
                                                                config_.workspace_reserve_bytes);
        telemetry_.target_maximum_bytes =
            std::max(telemetry_.target_maximum_bytes, telemetry_.target_bytes);
        ++telemetry_.budget_grows;
        consecutive_growth_samples_ = 0;
      }
    } else {
      consecutive_growth_samples_ = 0;
    }
    return RuntimeStatus::success;
  }

  [[nodiscard]] std::optional<std::size_t> acquire_frame() {
    const std::size_t active_frames = static_cast<std::size_t>(frame_capacity_);
    for (std::size_t index = 0; index < active_frames; ++index) {
      Frame& frame = frames_[index];
      if (!frame.mapped && frame.handle != 0) {
        ++telemetry_.handles_reused;
        return index;
      }
    }
    for (std::size_t index = 0; index < active_frames; ++index) {
      Frame& frame = frames_[index];
      if (!frame.mapped && frame.handle == 0) {
        if (create_handle(frame) != RuntimeStatus::success) {
          return std::nullopt;
        }
        return index;
      }
    }

    std::vector<VictimCandidate> candidates;
    candidates.reserve(frames_.size());
    for (std::size_t index = 0; index < active_frames; ++index) {
      const Frame& frame = frames_[index];
      if (!frame.key.has_value()) {
        continue;
      }
      const ChunkRecord* record = chunk(*frame.key);
      if (record != nullptr) {
        const Allocation* owner = allocation(record->key.allocation_id);
        candidates.push_back(VictimCandidate{
            record->key, record->state, record->pin_count, chunk_has_in_flight_work(*record),
            record->in_current_working_set, owner != nullptr && owner->hint == ResidencyHint::hot});
      }
    }
    const std::optional<ChunkKey> victim = policy_->select_victim(candidates);
    if (!victim.has_value()) {
      (void)fail(RuntimeStatus::budget_pressure, "cache", "select_victim",
                 "no event-safe cache frame is eligible for eviction");
      return std::nullopt;
    }
    ChunkRecord* victim_record = chunk(*victim);
    Frame* victim_frame = frame_for(*victim);
    if (victim_record == nullptr || victim_frame == nullptr) {
      (void)poison("cache", "select_victim", "policy selected an unknown mapping");
      return std::nullopt;
    }
    const std::size_t index = static_cast<std::size_t>(*victim_record->frame_index);
    const bool dirty_victim = victim_record->state == ChunkState::resident_dirty;
    const RuntimeStatus status = dirty_victim ? writeback(*victim) : RuntimeStatus::success;
    if (status != RuntimeStatus::success || unmap(*victim) != RuntimeStatus::success) {
      return std::nullopt;
    }
    if (dirty_victim) {
      ++telemetry_.dirty_evictions;
    } else {
      ++telemetry_.clean_evictions;
    }
    ++telemetry_.handles_reused;
    return index;
  }

  [[nodiscard]] RuntimeStatus ensure_resident(const ChunkAccessPlan& plan, const bool speculative,
                                              bool& hit) {
    ChunkRecord* record = chunk(plan.key);
    Allocation* owner = allocation(plan.key.allocation_id);
    if (record == nullptr || owner == nullptr) {
      return poison("cache", "ensure_resident", "working-set chunk is unknown");
    }
    if (record->state == ChunkState::resident_clean ||
        record->state == ChunkState::resident_dirty) {
      hit = true;
      ++telemetry_.cache_hits;
      policy_->touch(plan.key, ++policy_sequence_, owner->hint == ResidencyHint::streaming);
      return RuntimeStatus::success;
    }
    hit = false;
    ++telemetry_.cache_misses;
    const std::optional<std::size_t> frame_index = acquire_frame();
    if (!frame_index.has_value()) {
      return error_.status;
    }
    Frame& frame = frames_[*frame_index];
    const cuda::abi::DevicePointer address =
        owner->logical_base + plan.key.chunk_index * config_.chunk_bytes;
    const bool physical_alias =
        frame.mapped || frame.handle == 0 ||
        std::any_of(frames_.begin(), frames_.end(), [&](const Frame& other) {
          return &other != &frame && other.mapped && other.handle == frame.handle;
        });
    if (physical_alias) {
      telemetry_.no_physical_aliases = false;
      return poison("cache", "map_alias_check",
                    "a physical frame was already mapped at another logical address");
    }
    if (transition_chunk_state(record->state, ChunkState::mapping) !=
        StateTransitionResult::success) {
      return poison_transition("cache", "map", "illegal mapping state transition");
    }
    if (const cuda::abi::Result code = api_.mem_map_(
            address, static_cast<std::size_t>(config_.chunk_bytes), 0, frame.handle, 0);
        code != cuda::abi::success) {
      if (transition_chunk_state(record->state, ChunkState::host_clean) !=
          StateTransitionResult::success) {
        return poison("cache", "map_rollback", "failed to restore the host-clean state");
      }
      return fail_cuda("cache", "cuMemMap", code);
    }
    frame.key = plan.key;
    frame.address = address;
    frame.mapped = true;
    ++frame.map_generation;
    record->frame_index = static_cast<std::uint64_t>(*frame_index);
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device_;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (const cuda::abi::Result code = api_.mem_set_access_(
            address, static_cast<std::size_t>(config_.chunk_bytes), &access, 1);
        code != cuda::abi::success) {
      const cuda::abi::Result rollback =
          api_.mem_unmap_(address, static_cast<std::size_t>(config_.chunk_bytes));
      if (rollback != cuda::abi::success) {
        owner->reservation_quarantined = true;
        frame.quarantined = true;
        telemetry_.quarantined = true;
        release_quarantined_handle(frame);
        return poison_cuda("cache", "cuMemUnmap(set_access_rollback)", rollback);
      }
      frame.key.reset();
      frame.address = 0;
      frame.mapped = false;
      record->frame_index.reset();
      if (transition_chunk_state(record->state, ChunkState::host_clean) !=
          StateTransitionResult::success) {
        return poison("cache", "set_access_rollback",
                      "failed to restore the host-clean state after unmapping");
      }
      return fail_cuda("cache", "cuMemSetAccess", code);
    }
    ++telemetry_.maps;
    ++telemetry_.set_access;
    telemetry_.resident_bytes += config_.chunk_bytes;
    telemetry_.resident_peak_bytes =
        std::max(telemetry_.resident_peak_bytes, telemetry_.resident_bytes);
    if (address != owner->logical_base + plan.key.chunk_index * config_.chunk_bytes) {
      telemetry_.stable_addresses = false;
      return poison("cache", "stable_address", "VMM mapping address changed");
    }
    if (begin_event_generation(*record) != EventGenerationResult::success) {
      return poison("cache", "h2d_generation", "event generation overflow");
    }
    if (transition_chunk_state(record->state, ChunkState::h2d_in_flight) !=
        StateTransitionResult::success) {
      return poison_transition("cache", "h2d", "illegal H2D state transition");
    }
    const std::uint64_t valid = std::min(
        config_.chunk_bytes, owner->logical_bytes - plan.key.chunk_index * config_.chunk_bytes);
    const std::size_t staging_index = next_h2d_staging_++ % h2d_staging_.size();
    StagingSlot& staging = h2d_staging_[staging_index];
    record->staging_slot = static_cast<std::uint32_t>(staging_index);
    if (plan.requires_h2d) {
      const auto* source = static_cast<const std::byte*>(owner->backing->data()) +
                           plan.key.chunk_index * config_.chunk_bytes;
      std::memcpy(staging.memory, source, static_cast<std::size_t>(valid));
      if (const cuda::abi::Result code = api_.memcpy_h2d_async_(
              address, staging.memory, static_cast<std::size_t>(valid), h2d_stream_);
          code != cuda::abi::success) {
        async_resources_quarantined_ = true;
        frame.quarantined = true;
        quarantine_mapping(*owner);
        return poison_cuda("cache", "cuMemcpyHtoDAsync", code);
      }
      telemetry_.h2d_bytes += valid;
    }
    if (const cuda::abi::Result code = api_.event_record_(staging.done, h2d_stream_);
        code != cuda::abi::success) {
      async_resources_quarantined_ = true;
      frame.quarantined = true;
      quarantine_mapping(*owner);
      return poison_cuda("cache", "cuEventRecord(h2d)", code);
    }
    if (const RuntimeStatus status = wait_event(staging.done, "h2d_complete");
        status != RuntimeStatus::success) {
      return status;
    }
    record->staging_slot.reset();
    if (complete_event_generation(*record, record->event_generation) !=
            EventGenerationResult::success ||
        transition_chunk_state(record->state, ChunkState::resident_clean) !=
            StateTransitionResult::success) {
      return poison("cache", "h2d_retire", "H2D generation could not retire");
    }
    record->speculative = speculative;
    record->sequential_one_touch = owner->hint == ResidencyHint::streaming;
    policy_->insert(plan.key, ++policy_sequence_, speculative, record->sequential_one_touch);
    return RuntimeStatus::success;
  }

  [[nodiscard]] RuntimeStatus writeback(const ChunkKey key) {
    ChunkRecord* record = chunk(key);
    Allocation* owner = allocation(key.allocation_id);
    Frame* frame = frame_for(key);
    if (record == nullptr || owner == nullptr || frame == nullptr ||
        record->state != ChunkState::resident_dirty || !is_victim_eligible(*record)) {
      return poison("cache", "writeback", "dirty chunk is not event-safe");
    }
    if (begin_event_generation(*record) != EventGenerationResult::success ||
        transition_chunk_state(record->state, ChunkState::writeback_queued) !=
            StateTransitionResult::success ||
        transition_chunk_state(record->state, ChunkState::d2h_in_flight) !=
            StateTransitionResult::success) {
      return poison_transition("cache", "writeback", "illegal write-back generation or transition");
    }
    const std::uint64_t valid =
        std::min(config_.chunk_bytes, owner->logical_bytes - key.chunk_index * config_.chunk_bytes);
    const std::size_t staging_index = next_d2h_staging_++ % d2h_staging_.size();
    StagingSlot& staging = d2h_staging_[staging_index];
    record->staging_slot = static_cast<std::uint32_t>(h2d_staging_.size() + staging_index);
    if (const cuda::abi::Result code = api_.memcpy_d2h_async_(
            staging.memory, frame->address, static_cast<std::size_t>(valid), d2h_stream_);
        code != cuda::abi::success) {
      async_resources_quarantined_ = true;
      frame->quarantined = true;
      quarantine_mapping(*owner);
      return poison_cuda("cache", "cuMemcpyDtoHAsync", code);
    }
    if (const cuda::abi::Result code = api_.event_record_(staging.done, d2h_stream_);
        code != cuda::abi::success) {
      async_resources_quarantined_ = true;
      frame->quarantined = true;
      quarantine_mapping(*owner);
      return poison_cuda("cache", "cuEventRecord(d2h)", code);
    }
    if (const RuntimeStatus status = wait_event(staging.done, "d2h_complete");
        status != RuntimeStatus::success) {
      return status;
    }
    std::memcpy(static_cast<std::byte*>(owner->backing->data()) +
                    key.chunk_index * config_.chunk_bytes,
                staging.memory, static_cast<std::size_t>(valid));
    record->staging_slot.reset();
    if (complete_event_generation(*record, record->event_generation) !=
            EventGenerationResult::success ||
        transition_chunk_state(record->state, ChunkState::resident_clean) !=
            StateTransitionResult::success) {
      return poison("cache", "writeback_retire", "D2H generation could not retire");
    }
    telemetry_.d2h_bytes += valid;
    ++telemetry_.dirty_writebacks;
    return RuntimeStatus::success;
  }

  void quarantine_mapping(Allocation& owner) noexcept {
    owner.reservation_quarantined = true;
    telemetry_.quarantined = true;
    poisoned_ = true;
  }

  void release_quarantined_handle(Frame& frame) noexcept {
    if (frame.handle == 0) {
      return;
    }
    if (api_.mem_release_(frame.handle) == cuda::abi::success) {
      ++telemetry_.handles_released;
      frame.handle = 0;
    }
  }

  [[nodiscard]] RuntimeStatus retire_evicting_mapping(ChunkRecord& record, Allocation& owner,
                                                      Frame& frame) {
    const cuda::abi::Result code =
        api_.mem_unmap_(frame.address, static_cast<std::size_t>(config_.chunk_bytes));
    if (code != cuda::abi::success) {
      frame.quarantined = true;
      quarantine_mapping(owner);
      release_quarantined_handle(frame);
      return poison_cuda("cache", "cuMemUnmap", code);
    }
    ++telemetry_.unmaps;
    ++telemetry_.event_boundaries;
    telemetry_.resident_bytes -= config_.chunk_bytes;
    (void)policy_->erase(record.key);
    frame.key.reset();
    frame.address = 0;
    frame.mapped = false;
    frame.quarantined = false;
    record.frame_index.reset();
    record.speculative = false;
    if (transition_chunk_state(record.state, ChunkState::host_clean) !=
        StateTransitionResult::success) {
      return poison_transition("cache", "unmap_retire", "illegal host-clean transition");
    }
    return RuntimeStatus::success;
  }

  [[nodiscard]] RuntimeStatus rollback_provisional_mapping(ChunkRecord& record, Allocation& owner,
                                                           Frame& frame, const char* operation) {
    const cuda::abi::Result code =
        api_.mem_unmap_(frame.address, static_cast<std::size_t>(config_.chunk_bytes));
    if (code != cuda::abi::success) {
      frame.quarantined = true;
      quarantine_mapping(owner);
      release_quarantined_handle(frame);
      return poison_cuda("cache", operation, code);
    }
    frame.key.reset();
    frame.address = 0;
    frame.mapped = false;
    frame.quarantined = false;
    record.frame_index.reset();
    if (transition_chunk_state(record.state, ChunkState::host_clean) !=
        StateTransitionResult::success) {
      return poison_transition("cache", "mapping_rollback",
                               "illegal host-clean rollback transition");
    }
    return RuntimeStatus::success;
  }

  [[nodiscard]] RuntimeStatus unmap(const ChunkKey key) {
    ChunkRecord* record = chunk(key);
    Allocation* owner = allocation(key.allocation_id);
    Frame* frame = frame_for(key);
    if (record == nullptr || owner == nullptr || frame == nullptr ||
        record->state != ChunkState::resident_clean || !is_victim_eligible(*record) ||
        record->event_generation == 0) {
      ++telemetry_.unsafe_remaps;
      return poison("cache", "unmap", "chunk lacks a completed event boundary");
    }
    if (transition_chunk_state(record->state, ChunkState::evicting) !=
        StateTransitionResult::success) {
      ++telemetry_.unsafe_remaps;
      return poison_transition("cache", "unmap", "illegal eviction transition");
    }
    if (!is_safe_to_unmap(*record)) {
      ++telemetry_.unsafe_remaps;
      return poison("cache", "unmap", "chunk failed the event-safe unmap invariant");
    }
    return retire_evicting_mapping(*record, *owner, *frame);
  }

  [[nodiscard]] RuntimeStatus ensure_workspace(const std::uint64_t bytes) {
    if (bytes <= workspace_bytes_) {
      return RuntimeStatus::success;
    }
    if (bytes > telemetry_.target_bytes ||
        bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
      return fail(RuntimeStatus::budget_pressure, "workspace", "reserve",
                  "workspace exceeds the live device target");
    }
    if (workspace_ != 0) {
      if (const cuda::abi::Result code = api_.mem_free_(workspace_); code != cuda::abi::success) {
        return fail_cuda("workspace", "cuMemFree", code);
      }
      workspace_ = 0;
      workspace_bytes_ = 0;
    }
    if (bytes != 0) {
      if (const cuda::abi::Result code =
              api_.mem_alloc_(&workspace_, static_cast<std::size_t>(bytes));
          code != cuda::abi::success) {
        return fail_cuda("workspace", "cuMemAlloc", code);
      }
      workspace_bytes_ = bytes;
      telemetry_.workspace_peak_bytes = std::max(telemetry_.workspace_peak_bytes, bytes);
    }
    return RuntimeStatus::success;
  }

  void unpin(const std::span<ChunkRecord*> records) noexcept {
    for (ChunkRecord* record : records) {
      if (record == nullptr) {
        continue;
      }
      if (record->pin_count != 0) {
        --record->pin_count;
      }
      record->in_current_working_set = false;
      if (record->state == ChunkState::resident_clean ||
          record->state == ChunkState::resident_dirty) {
        policy_->touch(record->key, ++policy_sequence_, record->sequential_one_touch);
      }
    }
  }

  cuda::CudaApi& api_;
  RuntimeConfig config_;
  cuda::abi::Context context_ = nullptr;
  cuda::abi::Device device_ = 0;
  bool owns_context_ = false;
  bool context_pushed_ = false;
  bool setup_complete_ = false;
  bool closed_ = false;
  bool poisoned_ = false;
  bool launches_blocked_ = false;
  bool async_resources_quarantined_ = false;
  RuntimeStatus close_status_ = RuntimeStatus::success;
  AllocationId next_allocation_id_{1};
  std::uint64_t frame_capacity_ = 0;
  std::uint64_t maximum_frame_capacity_ = 0;
  std::uint64_t configured_target_cap_ = 0;
  std::uint32_t consecutive_growth_samples_ = 0;
  bool wddm_budget_active_ = false;
  Clock::time_point last_budget_sample_{};
#ifdef _WIN32
  std::optional<platform::AdapterLuid> wddm_luid_;
  std::uint32_t wddm_node_mask_ = 0;
#endif
  std::uint64_t policy_sequence_ = 0;
  std::unordered_map<std::uint64_t, std::unique_ptr<Allocation>> allocations_;
  std::vector<Frame> frames_;
  std::unique_ptr<VictimPolicy> policy_;
  cuda::abi::Stream h2d_stream_ = nullptr;
  cuda::abi::Stream compute_stream_ = nullptr;
  cuda::abi::Stream d2h_stream_ = nullptr;
  cuda::abi::Event compute_started_ = nullptr;
  cuda::abi::Event compute_done_ = nullptr;
  std::vector<StagingSlot> h2d_staging_;
  std::vector<StagingSlot> d2h_staging_;
  std::size_t next_h2d_staging_ = 0;
  std::size_t next_d2h_staging_ = 0;
  cuda::abi::DevicePointer workspace_ = 0;
  std::uint64_t workspace_bytes_ = 0;
  RuntimeTelemetry telemetry_;
  RuntimeError error_;
};

Runtime::Runtime(cuda::CudaApi& api, RuntimeConfig config)
    : impl_(std::make_unique<Impl>(api, std::move(config))) {}

Runtime::~Runtime() {
  (void)impl_->close();
}

RuntimeStatus Runtime::setup() {
  return impl_->setup();
}
RuntimeStatus Runtime::allocate(const std::uint64_t bytes, const ResidencyHint hint,
                                RuntimeAllocation& output) {
  return impl_->allocate(bytes, hint, output);
}
RuntimeStatus Runtime::release(const AllocationId id) {
  return impl_->release(id);
}
RuntimeStatus Runtime::write(const AllocationId id, const std::uint64_t offset, const void* source,
                             const std::uint64_t bytes) {
  return impl_->write(id, offset, source, bytes);
}
RuntimeStatus Runtime::read(const AllocationId id, const std::uint64_t offset, void* destination,
                            const std::uint64_t bytes) {
  return impl_->read(id, offset, destination, bytes);
}
RuntimeStatus Runtime::prefetch(const std::span<const AccessRange> ranges) {
  return impl_->prefetch(ranges);
}
RuntimeStatus Runtime::execute(const std::span<const AccessRange> ranges,
                               const std::uint64_t workspace_bytes,
                               const TransactionCallback& callback) {
  return impl_->execute(ranges, workspace_bytes, callback);
}
RuntimeStatus Runtime::drain(const bool release_mappings) {
  return impl_->drain(release_mappings);
}
RuntimeStatus Runtime::close() noexcept {
  return impl_->close();
}
cuda::abi::Context Runtime::context() const noexcept {
  return impl_->context();
}
cuda::abi::Device Runtime::device() const noexcept {
  return impl_->device();
}
std::uint64_t Runtime::chunk_bytes() const noexcept {
  return impl_->chunk_bytes();
}
std::uint64_t Runtime::target_bytes() const noexcept {
  return impl_->target_bytes();
}
std::optional<std::uint64_t> Runtime::allocation_size(const AllocationId id) const noexcept {
  return impl_->allocation_size(id);
}
const RuntimeTelemetry& Runtime::telemetry() const noexcept {
  return impl_->telemetry();
}
const RuntimeError& Runtime::error() const noexcept {
  return impl_->error();
}
bool Runtime::poisoned() const noexcept {
  return impl_->poisoned();
}

bool Runtime::async_completion_unknown() const noexcept {
  return impl_->async_completion_unknown();
}

} // namespace xvram::residency
