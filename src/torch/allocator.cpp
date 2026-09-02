#include "torch/allocator.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>

namespace xvram::torch_allocator {
namespace {

template <std::size_t Size>
void copy_text(std::array<char, Size>& destination, const char* source) noexcept {
  destination.fill('\0');
  if (source == nullptr) {
    return;
  }
  const std::size_t length = std::min(std::strlen(source), Size - 1U);
  std::memcpy(destination.data(), source, length);
}

[[nodiscard]] std::uint64_t saturating_add(const std::uint64_t left,
                                           const std::uint64_t right) noexcept {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    return std::numeric_limits<std::uint64_t>::max();
  }
  return left + right;
}

void add_counter(std::uint64_t& value, const std::uint64_t increment = 1U) noexcept {
  value = saturating_add(value, increment);
}

void subtract_counter(std::uint64_t& value, const std::uint64_t decrement) noexcept {
  value = decrement > value ? 0U : value - decrement;
}

[[nodiscard]] std::uint64_t as_u64(const std::size_t value) noexcept {
  return static_cast<std::uint64_t>(value);
}

} // namespace

std::optional<std::size_t> checked_align_up(const std::size_t value,
                                            const std::size_t alignment) noexcept {
  if (alignment == 0U) {
    return std::nullopt;
  }
  const std::size_t remainder = value % alignment;
  if (remainder == 0U) {
    return value;
  }
  const std::size_t increment = alignment - remainder;
  if (value > std::numeric_limits<std::size_t>::max() - increment) {
    return std::nullopt;
  }
  return value + increment;
}

SegmentAllocator::SegmentAllocator(cuda::CudaApi& api,
                                   const std::size_t registry_limit) noexcept
    : api_(api), registry_limit_(registry_limit) {}

bool SegmentAllocator::dispatch_available() const noexcept {
  return api_.context_get_current_ != nullptr && api_.context_get_device_ != nullptr &&
         api_.context_push_current_ != nullptr && api_.context_pop_current_ != nullptr &&
         api_.stream_is_capturing_ != nullptr &&
         api_.mem_get_allocation_granularity_ != nullptr &&
         api_.mem_address_reserve_ != nullptr && api_.mem_address_free_ != nullptr &&
         api_.mem_create_ != nullptr && api_.mem_release_ != nullptr &&
         api_.mem_map_ != nullptr && api_.mem_unmap_ != nullptr &&
         api_.mem_set_access_ != nullptr && api_.event_create_ != nullptr &&
         api_.event_record_ != nullptr && api_.event_synchronize_ != nullptr &&
         api_.event_destroy_ != nullptr;
}

void SegmentAllocator::record_failure(const xvram_torch_allocator_status status,
                                       const char* stage, const char* operation,
                                       const char* message,
                                       const cuda::abi::Result native_code) noexcept {
  if (poisoned_) {
    restore_poison_diagnostic();
    return;
  }
  stats_.last_status = status;
  stats_.last_native_error = static_cast<std::int64_t>(native_code);
  last_error_.status = status;
  last_error_.native_code = static_cast<std::int64_t>(native_code);
  copy_text(last_error_.stage, stage);
  copy_text(last_error_.operation, operation);
  copy_text(last_error_.message, message);
}

void SegmentAllocator::record_cuda_failure(const char* stage, const char* operation,
                                           const cuda::abi::Result code) noexcept {
  const char* name = nullptr;
  const char* description = nullptr;
  if (api_.get_error_name_ != nullptr) {
    (void)api_.get_error_name_(code, &name);
  }
  if (api_.get_error_string_ != nullptr) {
    (void)api_.get_error_string_(code, &description);
  }

  std::array<char, 256> message{};
  if (name != nullptr && description != nullptr) {
    (void)std::snprintf(message.data(), message.size(), "%s: %s", name, description);
  } else if (name != nullptr) {
    (void)std::snprintf(message.data(), message.size(), "%s", name);
  } else {
    (void)std::snprintf(message.data(), message.size(), "CUDA driver error %lld",
                        static_cast<long long>(code));
  }

  const xvram_torch_allocator_status status =
      code == CUDA_ERROR_OUT_OF_MEMORY ? XVRAM_TORCH_ALLOCATOR_OUT_OF_MEMORY
                                       : XVRAM_TORCH_ALLOCATOR_CUDA_ERROR;
  if (code == CUDA_ERROR_OUT_OF_MEMORY) {
    add_counter(stats_.oom_failures);
    errno = ENOMEM;
  }
  record_failure(status, stage, operation, message.data(), code);
}

bool SegmentAllocator::capture_is_safe(const cuda::abi::Stream stream,
                                       const char* operation) noexcept {
  cuda::abi::StreamCaptureStatus capture_status = cuda::abi::stream_capture_invalidated;
  const cuda::abi::Result code = api_.stream_is_capturing_(stream, &capture_status);
  if (code != cuda::abi::success) {
    record_cuda_failure("stream", "cuStreamIsCapturing", code);
    return false;
  }
  if (capture_status != cuda::abi::stream_capture_none) {
    add_counter(stats_.capture_rejections);
    record_failure(XVRAM_TORCH_ALLOCATOR_CAPTURE_UNSUPPORTED, "stream", operation,
                   "new or released VMM segments are not supported during CUDA stream capture");
    errno = ENOMEM;
    return false;
  }
  return true;
}

bool SegmentAllocator::rollback_segment(Segment& segment) noexcept {
  bool complete = true;
  if (segment.mapped) {
    const cuda::abi::Result unmap_code = api_.mem_unmap_(segment.address, segment.mapped_bytes);
    if (unmap_code == cuda::abi::success) {
      segment.mapped = false;
      segment.access_set = false;
      add_counter(stats_.unmaps);
      subtract_counter(stats_.mapped_bytes_current, as_u64(segment.mapped_bytes));
    } else {
      record_cuda_failure("rollback", "cuMemUnmap", unmap_code);
      complete = false;
    }
  }

  if (segment.handle != 0U) {
    const cuda::abi::Result release_code = api_.mem_release_(segment.handle);
    if (release_code == cuda::abi::success) {
      segment.handle = 0U;
      add_counter(stats_.handle_releases);
    } else {
      record_cuda_failure("rollback", "cuMemRelease", release_code);
      complete = false;
    }
  }

  if (segment.reservation_active && !segment.mapped && segment.handle == 0U) {
    const cuda::abi::Result free_code =
        api_.mem_address_free_(segment.address, segment.mapped_bytes);
    if (free_code == cuda::abi::success) {
      segment.reservation_active = false;
      add_counter(stats_.reservation_frees);
    } else {
      record_cuda_failure("rollback", "cuMemAddressFree", free_code);
      complete = false;
    }
  }
  return complete && !segment.mapped && segment.handle == 0U && !segment.reservation_active;
}

void SegmentAllocator::poison_current_failure() noexcept {
  if (!poisoned_) {
    poison_error_ = last_error_;
    poison_error_.status = XVRAM_TORCH_ALLOCATOR_QUARANTINED;
    if (last_error_.status == XVRAM_TORCH_ALLOCATOR_SUCCESS) {
      copy_text(poison_error_.stage, "allocator");
      copy_text(poison_error_.operation, "quarantine");
      copy_text(poison_error_.message,
                "an unsafe cleanup boundary quarantined the allocator process boundary");
    }
    poisoned_ = true;
  }
  restore_poison_diagnostic();
}

void SegmentAllocator::restore_poison_diagnostic() noexcept {
  if (!poisoned_) {
    return;
  }
  stats_.last_status = XVRAM_TORCH_ALLOCATOR_QUARANTINED;
  stats_.last_native_error = poison_error_.native_code;
  last_error_ = poison_error_;
}

void SegmentAllocator::quarantine(Segment segment) noexcept {
  const auto found = segments_.find(segment.address);
  if (found == segments_.end()) {
    return;
  }
  if (!found->second.quarantined) {
    found->second = segment;
    found->second.quarantined = true;
    add_counter(stats_.quarantined_segments);
    if (segment.mapped) {
      add_counter(stats_.quarantined_mapped_bytes, as_u64(segment.mapped_bytes));
    }
  }
  poison_current_failure();
}

void SegmentAllocator::quarantine_untracked(Segment segment) noexcept {
  segment.quarantined = true;
  emergency_quarantine_ = segment;
  add_counter(stats_.quarantined_segments);
  if (segment.mapped) {
    add_counter(stats_.quarantined_mapped_bytes, as_u64(segment.mapped_bytes));
  }
  poison_current_failure();
}

void SegmentAllocator::retire_active_telemetry(const Segment& segment) noexcept {
  if (!segment.active) {
    return;
  }
  subtract_counter(stats_.requested_bytes_current, as_u64(segment.requested_bytes));
  subtract_counter(stats_.active_segments, 1U);
}

void* SegmentAllocator::allocate(const std::size_t bytes, const int device,
                                 const cuda::abi::Stream stream) noexcept {
  std::lock_guard lock(mutex_);
  add_counter(stats_.allocation_calls);

  try {
    if (poisoned_) {
      add_counter(stats_.allocation_failures);
      record_failure(XVRAM_TORCH_ALLOCATOR_QUARANTINED, "allocator", "allocate",
                     "a prior cleanup failure quarantined the allocator process boundary");
      errno = ENOMEM;
      return nullptr;
    }
    if (bytes == 0U || device < 0) {
      add_counter(stats_.allocation_failures);
      record_failure(XVRAM_TORCH_ALLOCATOR_INVALID_ARGUMENT, "allocation", "validate",
                     "allocation size must be nonzero and device ordinal must be nonnegative");
      errno = EINVAL;
      return nullptr;
    }

    const cuda::CudaApi::LoadResult load = api_.load();
    if (load.status != cuda::CudaApi::LoadStatus::loaded || !dispatch_available()) {
      add_counter(stats_.allocation_failures);
      record_failure(XVRAM_TORCH_ALLOCATOR_UNAVAILABLE, "loader", "CUDA Driver API",
                     "required CUDA VMM, context, event, or stream-capture symbols are unavailable");
      errno = ENOMEM;
      return nullptr;
    }

    cuda::abi::Context context = nullptr;
    cuda::abi::Result code = api_.context_get_current_(&context);
    if (code != cuda::abi::success) {
      add_counter(stats_.allocation_failures);
      record_cuda_failure("context", "cuCtxGetCurrent", code);
      return nullptr;
    }
    if (context == nullptr) {
      add_counter(stats_.allocation_failures);
      record_failure(XVRAM_TORCH_ALLOCATOR_CONTEXT_MISMATCH, "context", "cuCtxGetCurrent",
                     "PyTorch has no current CUDA context on the allocator callback thread");
      add_counter(stats_.context_mismatches);
      errno = ENOMEM;
      return nullptr;
    }

    cuda::abi::Device current_device = -1;
    code = api_.context_get_device_(&current_device);
    if (code != cuda::abi::success) {
      add_counter(stats_.allocation_failures);
      record_cuda_failure("context", "cuCtxGetDevice", code);
      return nullptr;
    }
    if (current_device != device) {
      add_counter(stats_.allocation_failures);
      add_counter(stats_.context_mismatches);
      record_failure(XVRAM_TORCH_ALLOCATOR_CONTEXT_MISMATCH, "context", "cuCtxGetDevice",
                     "allocator device does not match the current PyTorch CUDA context");
      errno = ENOMEM;
      return nullptr;
    }
    if (!capture_is_safe(stream, "allocate")) {
      add_counter(stats_.allocation_failures);
      return nullptr;
    }

    cuda::abi::MemAllocationProp property{};
    property.type = cuda::abi::allocation_type_pinned;
    property.location.type = cuda::abi::location_device;
    property.location.id = current_device;

    std::size_t granularity = 0U;
    code = api_.mem_get_allocation_granularity_(&granularity, &property,
                                                 cuda::abi::granularity_minimum);
    if (code != cuda::abi::success) {
      add_counter(stats_.allocation_failures);
      record_cuda_failure("allocation", "cuMemGetAllocationGranularity", code);
      return nullptr;
    }
    const std::optional<std::size_t> mapped_bytes = checked_align_up(bytes, granularity);
    if (!mapped_bytes.has_value() || *mapped_bytes == 0U) {
      add_counter(stats_.allocation_failures);
      record_failure(XVRAM_TORCH_ALLOCATOR_INVALID_ARGUMENT, "allocation", "align",
                     "allocation size overflows the CUDA VMM granularity");
      errno = ENOMEM;
      return nullptr;
    }

    cuda::abi::DevicePointer address = 0;
    code = api_.mem_address_reserve_(&address, *mapped_bytes, 0U, 0U, 0U);
    if (code != cuda::abi::success) {
      add_counter(stats_.allocation_failures);
      record_cuda_failure("allocation", "cuMemAddressReserve", code);
      return nullptr;
    }
    add_counter(stats_.reservations);

    Segment initial{};
    initial.address = address;
    initial.requested_bytes = bytes;
    initial.mapped_bytes = *mapped_bytes;
    initial.context = context;
    initial.allocation_stream = stream;
    initial.device = device;
    initial.reservation_active = true;

    decltype(segments_)::iterator position;
    bool inserted = false;
    const bool simulate_registry_failure = segments_.size() >= registry_limit_;
    try {
      if (simulate_registry_failure) {
        throw std::bad_alloc{};
      }
      const auto result = segments_.emplace(address, initial);
      position = result.first;
      inserted = result.second;
    } catch (const std::bad_alloc&) {
      add_counter(stats_.allocation_failures);
      add_counter(stats_.oom_failures);
      const cuda::abi::Result free_code = api_.mem_address_free_(address, *mapped_bytes);
      if (free_code == cuda::abi::success) {
        add_counter(stats_.reservation_frees);
        record_failure(XVRAM_TORCH_ALLOCATOR_OUT_OF_MEMORY, "host", "segment registry",
                       "host memory allocation failed while tracking a CUDA VMM segment");
      } else {
        record_cuda_failure("rollback", "cuMemAddressFree", free_code);
        quarantine_untracked(initial);
      }
      errno = ENOMEM;
      return nullptr;
    } catch (...) {
      add_counter(stats_.allocation_failures);
      const cuda::abi::Result free_code = api_.mem_address_free_(address, *mapped_bytes);
      if (free_code == cuda::abi::success) {
        add_counter(stats_.reservation_frees);
        record_failure(XVRAM_TORCH_ALLOCATOR_INTERNAL_ERROR, "registry", "insert",
                       "unexpected failure while tracking a CUDA VMM segment");
      } else {
        record_cuda_failure("rollback", "cuMemAddressFree", free_code);
        quarantine_untracked(initial);
      }
      errno = ENOMEM;
      return nullptr;
    }
    if (!inserted) {
      const cuda::abi::Result free_code = api_.mem_address_free_(address, *mapped_bytes);
      if (free_code == cuda::abi::success) {
        add_counter(stats_.reservation_frees);
        record_failure(XVRAM_TORCH_ALLOCATOR_INTERNAL_ERROR, "registry", "insert",
                       "CUDA returned a virtual address already present in the segment registry");
      } else {
        record_cuda_failure("rollback", "cuMemAddressFree", free_code);
        quarantine_untracked(initial);
      }
      add_counter(stats_.allocation_failures);
      errno = ENOMEM;
      return nullptr;
    }
    Segment& segment = position->second;

    const auto fail_setup = [&](const char* operation, const cuda::abi::Result failure_code) {
      record_cuda_failure("allocation", operation, failure_code);
      add_counter(stats_.allocation_failures);
      Segment failed = segment;
      const bool rolled_back = rollback_segment(failed);
      if (rolled_back) {
        segments_.erase(position);
      } else {
        position->second = failed;
        quarantine(failed);
      }
      return static_cast<void*>(nullptr);
    };

    code = api_.mem_create_(&segment.handle, *mapped_bytes, &property, 0U);
    if (code != cuda::abi::success) {
      return fail_setup("cuMemCreate", code);
    }
    add_counter(stats_.handles_created);

    code = api_.mem_map_(address, *mapped_bytes, 0U, segment.handle, 0U);
    if (code != cuda::abi::success) {
      return fail_setup("cuMemMap", code);
    }
    segment.mapped = true;
    add_counter(stats_.maps);
    add_counter(stats_.mapped_bytes_total, as_u64(*mapped_bytes));
    add_counter(stats_.mapped_bytes_current, as_u64(*mapped_bytes));
    stats_.mapped_bytes_peak = std::max(stats_.mapped_bytes_peak, stats_.mapped_bytes_current);

    CUmemAccessDesc access{};
    access.location = property.location;
    access.flags = cuda::abi::mem_access_read_write;
    code = api_.mem_set_access_(address, *mapped_bytes, &access, 1U);
    if (code != cuda::abi::success) {
      return fail_setup("cuMemSetAccess", code);
    }
    segment.access_set = true;
    add_counter(stats_.set_access_calls);

    segment.active = true;
    add_counter(stats_.requested_bytes_total, as_u64(bytes));
    add_counter(stats_.requested_bytes_current, as_u64(bytes));
    stats_.requested_bytes_peak =
        std::max(stats_.requested_bytes_peak, stats_.requested_bytes_current);
    add_counter(stats_.active_segments);
    stats_.active_segments_peak = std::max(stats_.active_segments_peak, stats_.active_segments);
    stats_.last_status = XVRAM_TORCH_ALLOCATOR_SUCCESS;
    stats_.last_native_error = 0;
    last_error_ = {};
    return reinterpret_cast<void*>(static_cast<std::uintptr_t>(address));
  } catch (const std::bad_alloc&) {
    add_counter(stats_.allocation_failures);
    add_counter(stats_.oom_failures);
    record_failure(XVRAM_TORCH_ALLOCATOR_OUT_OF_MEMORY, "host", "segment registry",
                   "host memory allocation failed while tracking a CUDA VMM segment");
    errno = ENOMEM;
    return nullptr;
  } catch (...) {
    add_counter(stats_.allocation_failures);
    record_failure(XVRAM_TORCH_ALLOCATOR_INTERNAL_ERROR, "allocator", "allocate",
                   "unexpected internal allocator failure");
    errno = ENOMEM;
    return nullptr;
  }
}

void SegmentAllocator::deallocate(void* pointer, const std::size_t bytes, const int device,
                                  const cuda::abi::Stream stream) noexcept {
  std::lock_guard lock(mutex_);
  add_counter(stats_.free_calls);
  if (pointer == nullptr) {
    if (poisoned_) {
      restore_poison_diagnostic();
    } else {
      stats_.last_status = XVRAM_TORCH_ALLOCATOR_SUCCESS;
      stats_.last_native_error = 0;
      last_error_ = {};
    }
    return;
  }

  const auto address = static_cast<cuda::abi::DevicePointer>(
      reinterpret_cast<std::uintptr_t>(pointer));
  const auto position = segments_.find(address);
  if (position == segments_.end()) {
    add_counter(stats_.free_failures);
    record_failure(XVRAM_TORCH_ALLOCATOR_UNKNOWN_POINTER, "registry", "free",
                   "pointer is not an xVRAM PyTorch segment");
    return;
  }
  Segment& stored = position->second;
  if (stored.quarantined) {
    add_counter(stats_.free_failures);
    record_failure(XVRAM_TORCH_ALLOCATOR_QUARANTINED, "registry", "free",
                   "segment was already quarantined after an earlier unsafe cleanup boundary");
    return;
  }
  if (bytes != stored.requested_bytes) {
    add_counter(stats_.size_mismatches);
  }
  if (stream != stored.allocation_stream) {
    add_counter(stats_.stream_mismatches);
  }
  if (device != stored.device) {
    add_counter(stats_.context_mismatches);
  }

  bool pushed = false;
  bool cleanup_failed = false;
  auto restore_context = [&]() noexcept {
    if (!pushed) {
      return;
    }
    cuda::abi::Context popped = nullptr;
    const cuda::abi::Result pop_code = api_.context_pop_current_(&popped);
    if (pop_code != cuda::abi::success || popped != stored.context) {
      cleanup_failed = true;
      if (pop_code != cuda::abi::success) {
        record_cuda_failure("context", "cuCtxPopCurrent", pop_code);
      } else {
        record_failure(XVRAM_TORCH_ALLOCATOR_CONTEXT_MISMATCH, "context",
                       "cuCtxPopCurrent", "CUDA popped a context other than the captured context");
      }
      poison_current_failure();
    }
    pushed = false;
  };

  cuda::abi::Context current = nullptr;
  cuda::abi::Result code = api_.context_get_current_(&current);
  if (code != cuda::abi::success) {
    add_counter(stats_.free_failures);
    record_cuda_failure("context", "cuCtxGetCurrent", code);
    quarantine(stored);
    return;
  }
  if (current != stored.context) {
    add_counter(stats_.context_mismatches);
    code = api_.context_push_current_(stored.context);
    if (code != cuda::abi::success) {
      add_counter(stats_.free_failures);
      record_cuda_failure("context", "cuCtxPushCurrent", code);
      quarantine(stored);
      return;
    }
    pushed = true;
  }

  cuda::abi::Device current_device = -1;
  code = api_.context_get_device_(&current_device);
  if (code != cuda::abi::success || current_device != stored.device) {
    add_counter(stats_.free_failures);
    if (code != cuda::abi::success) {
      record_cuda_failure("context", "cuCtxGetDevice", code);
    } else {
      record_failure(XVRAM_TORCH_ALLOCATOR_CONTEXT_MISMATCH, "context", "cuCtxGetDevice",
                     "captured segment context no longer belongs to its original CUDA device");
    }
    restore_context();
    quarantine(stored);
    return;
  }

  if (!capture_is_safe(stored.allocation_stream, "free")) {
    add_counter(stats_.free_failures);
    restore_context();
    quarantine(stored);
    return;
  }

  cuda::abi::Event completion = nullptr;
  code = api_.event_create_(&completion, CU_EVENT_DISABLE_TIMING);
  if (code != cuda::abi::success) {
    add_counter(stats_.free_failures);
    record_cuda_failure("fence", "cuEventCreate", code);
    restore_context();
    quarantine(stored);
    return;
  }
  code = api_.event_record_(completion, stored.allocation_stream);
  if (code != cuda::abi::success) {
    add_counter(stats_.free_failures);
    record_cuda_failure("fence", "cuEventRecord", code);
    (void)api_.event_destroy_(completion);
    restore_context();
    quarantine(stored);
    return;
  }
  code = api_.event_synchronize_(completion);
  if (code != cuda::abi::success) {
    add_counter(stats_.free_failures);
    record_cuda_failure("fence", "cuEventSynchronize", code);
    (void)api_.event_destroy_(completion);
    restore_context();
    quarantine(stored);
    return;
  }
  add_counter(stats_.event_boundaries);

  code = api_.event_destroy_(completion);
  if (code != cuda::abi::success) {
    cleanup_failed = true;
    record_cuda_failure("cleanup", "cuEventDestroy", code);
  }

  code = api_.mem_unmap_(stored.address, stored.mapped_bytes);
  if (code != cuda::abi::success) {
    add_counter(stats_.free_failures);
    record_cuda_failure("cleanup", "cuMemUnmap", code);
    if (stored.handle != 0U) {
      const cuda::abi::Result release_code = api_.mem_release_(stored.handle);
      if (release_code == cuda::abi::success) {
        stored.handle = 0U;
        add_counter(stats_.handle_releases);
      } else {
        record_cuda_failure("cleanup", "cuMemRelease", release_code);
      }
    }
    restore_context();
    quarantine(stored);
    return;
  }
  stored.mapped = false;
  stored.access_set = false;
  add_counter(stats_.unmaps);
  subtract_counter(stats_.mapped_bytes_current, as_u64(stored.mapped_bytes));

  if (stored.handle != 0U) {
    code = api_.mem_release_(stored.handle);
    if (code == cuda::abi::success) {
      stored.handle = 0U;
      add_counter(stats_.handle_releases);
    } else {
      cleanup_failed = true;
      record_cuda_failure("cleanup", "cuMemRelease", code);
    }
  }
  if (stored.reservation_active && stored.handle == 0U) {
    code = api_.mem_address_free_(stored.address, stored.mapped_bytes);
    if (code == cuda::abi::success) {
      stored.reservation_active = false;
      add_counter(stats_.reservation_frees);
    } else {
      cleanup_failed = true;
      record_cuda_failure("cleanup", "cuMemAddressFree", code);
    }
  }

  restore_context();
  if (stored.handle != 0U || stored.reservation_active) {
    add_counter(stats_.free_failures);
    quarantine(stored);
    return;
  }

  Segment retired = stored;
  retire_active_telemetry(retired);
  segments_.erase(position);
  if (cleanup_failed) {
    add_counter(stats_.free_failures);
    return;
  }
  if (poisoned_) {
    restore_poison_diagnostic();
  } else {
    stats_.last_status = XVRAM_TORCH_ALLOCATOR_SUCCESS;
    stats_.last_native_error = 0;
    last_error_ = {};
  }
}

xvram_torch_allocator_stats_v1 SegmentAllocator::stats() const noexcept {
  std::lock_guard lock(mutex_);
  return stats_;
}

LastError SegmentAllocator::last_error() const noexcept {
  std::lock_guard lock(mutex_);
  return last_error_;
}

std::size_t SegmentAllocator::live_allocation_count() const noexcept {
  std::lock_guard lock(mutex_);
  return segments_.size() + (emergency_quarantine_.has_value() ? 1U : 0U);
}

void SegmentAllocator::reset_stats() noexcept {
  std::lock_guard lock(mutex_);
  xvram_torch_allocator_stats_v1 reset = XVRAM_TORCH_ALLOCATOR_STATS_V1_INIT;
  for (const auto& [address, segment] : segments_) {
    (void)address;
    if (segment.active) {
      add_counter(reset.requested_bytes_current, as_u64(segment.requested_bytes));
      add_counter(reset.requested_bytes_total, as_u64(segment.requested_bytes));
      add_counter(reset.active_segments);
    }
    if (segment.reservation_active) {
      add_counter(reset.reservations);
    }
    if (segment.handle != 0U) {
      add_counter(reset.handles_created);
    }
    if (segment.mapped) {
      add_counter(reset.mapped_bytes_current, as_u64(segment.mapped_bytes));
      add_counter(reset.mapped_bytes_total, as_u64(segment.mapped_bytes));
      add_counter(reset.maps);
    }
    if (segment.access_set) {
      add_counter(reset.set_access_calls);
    }
    if (segment.quarantined) {
      add_counter(reset.quarantined_segments);
      if (segment.mapped) {
        add_counter(reset.quarantined_mapped_bytes, as_u64(segment.mapped_bytes));
      }
    }
  }
  if (emergency_quarantine_.has_value()) {
    const Segment& segment = *emergency_quarantine_;
    if (segment.reservation_active) {
      add_counter(reset.reservations);
    }
    if (segment.handle != 0U) {
      add_counter(reset.handles_created);
    }
    if (segment.mapped) {
      add_counter(reset.mapped_bytes_current, as_u64(segment.mapped_bytes));
      add_counter(reset.mapped_bytes_total, as_u64(segment.mapped_bytes));
      add_counter(reset.maps);
    }
    if (segment.access_set) {
      add_counter(reset.set_access_calls);
    }
    add_counter(reset.quarantined_segments);
    if (segment.mapped) {
      add_counter(reset.quarantined_mapped_bytes, as_u64(segment.mapped_bytes));
    }
  }
  reset.requested_bytes_peak = reset.requested_bytes_current;
  reset.mapped_bytes_peak = reset.mapped_bytes_current;
  reset.active_segments_peak = reset.active_segments;
  if (poisoned_) {
    reset.last_status = XVRAM_TORCH_ALLOCATOR_QUARANTINED;
    reset.last_native_error = poison_error_.native_code;
    last_error_ = poison_error_;
  } else {
    last_error_ = {};
  }
  stats_ = reset;
}

} // namespace xvram::torch_allocator
