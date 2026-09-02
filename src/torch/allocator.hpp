#pragma once

#include "platform/cuda/cuda_api.hpp"
#include "xvram/torch_allocator.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <unordered_map>

namespace xvram::torch_allocator {

[[nodiscard]] std::optional<std::size_t> checked_align_up(std::size_t value,
                                                          std::size_t alignment) noexcept;

struct LastError {
  xvram_torch_allocator_status status = XVRAM_TORCH_ALLOCATOR_SUCCESS;
  std::int64_t native_code = 0;
  std::array<char, 32> stage{};
  std::array<char, 64> operation{};
  std::array<char, 256> message{};
};

class SegmentAllocator {
public:
  explicit SegmentAllocator(
      cuda::CudaApi& api,
      std::size_t registry_limit = std::numeric_limits<std::size_t>::max()) noexcept;

  SegmentAllocator(const SegmentAllocator&) = delete;
  SegmentAllocator& operator=(const SegmentAllocator&) = delete;

  [[nodiscard]] void* allocate(std::size_t bytes, int device,
                               cuda::abi::Stream stream) noexcept;
  void deallocate(void* pointer, std::size_t bytes, int device,
                  cuda::abi::Stream stream) noexcept;

  [[nodiscard]] xvram_torch_allocator_stats_v1 stats() const noexcept;
  [[nodiscard]] LastError last_error() const noexcept;
  [[nodiscard]] std::size_t live_allocation_count() const noexcept;
  void reset_stats() noexcept;

private:
  struct Segment {
    cuda::abi::DevicePointer address = 0;
    std::size_t requested_bytes = 0;
    std::size_t mapped_bytes = 0;
    cuda::abi::GenericAllocationHandle handle = 0;
    cuda::abi::Context context = nullptr;
    cuda::abi::Stream allocation_stream = nullptr;
    int device = -1;
    bool reservation_active = false;
    bool mapped = false;
    bool access_set = false;
    bool active = false;
    bool quarantined = false;
  };

  [[nodiscard]] bool dispatch_available() const noexcept;
  [[nodiscard]] bool capture_is_safe(cuda::abi::Stream stream, const char* operation) noexcept;
  [[nodiscard]] bool rollback_segment(Segment& segment) noexcept;
  void poison_current_failure() noexcept;
  void restore_poison_diagnostic() noexcept;
  void quarantine(Segment segment) noexcept;
  void quarantine_untracked(Segment segment) noexcept;
  void record_failure(xvram_torch_allocator_status status, const char* stage,
                      const char* operation, const char* message,
                      cuda::abi::Result native_code = cuda::abi::success) noexcept;
  void record_cuda_failure(const char* stage, const char* operation,
                           cuda::abi::Result code) noexcept;
  void retire_active_telemetry(const Segment& segment) noexcept;

  cuda::CudaApi& api_;
  mutable std::mutex mutex_;
  std::unordered_map<cuda::abi::DevicePointer, Segment> segments_;
  std::optional<Segment> emergency_quarantine_;
  std::size_t registry_limit_ = std::numeric_limits<std::size_t>::max();
  bool poisoned_ = false;
  xvram_torch_allocator_stats_v1 stats_ = XVRAM_TORCH_ALLOCATOR_STATS_V1_INIT;
  LastError last_error_;
  LastError poison_error_;
};

} // namespace xvram::torch_allocator
