#include "residency/runtime.hpp"

#include "platform/nvcomp/nvcomp_api.hpp"
#include "platform/system_info.hpp"
#ifdef _WIN32
#include "platform/dxgi_memory.hpp"
#endif
#include "residency/cpu_codec_pool.hpp"
#include "residency/lz4_codec.hpp"
#include "residency/nvcomp_pipeline.hpp"
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

class RawOnlyBlockCodec final : public BlockCodec {
public:
  [[nodiscard]] CompressionCodec codec() const noexcept override {
    // HostBackingStore validates the container codec family even when every block is raw. This
    // sentinel deliberately exposes no encoder/decoder and avoids initializing LZ4 for ABI v1.
    return CompressionCodec::lz4;
  }
  [[nodiscard]] std::string_view implementation_name() const noexcept override {
    return "raw-only";
  }
  [[nodiscard]] std::optional<std::size_t>
  maximum_compressed_bytes(std::size_t) const noexcept override {
    return std::nullopt;
  }
  [[nodiscard]] CodecStatus compress(std::span<const std::byte>, std::span<std::byte>,
                                     std::size_t& written_bytes) const noexcept override {
    written_bytes = 0;
    return CodecStatus::invalid_argument;
  }
  [[nodiscard]] CodecStatus decompress(std::span<const std::byte>,
                                       std::span<std::byte>) const noexcept override {
    return CodecStatus::invalid_argument;
  }
};

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

[[nodiscard]] std::uint64_t saturating_add(const std::uint64_t left,
                                           const std::uint64_t right) noexcept {
  return right > std::numeric_limits<std::uint64_t>::max() - left
             ? std::numeric_limits<std::uint64_t>::max()
             : left + right;
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

[[nodiscard]] std::uint64_t content_identity(const ContentToken token) noexcept {
  const std::uint64_t rotated_low = (token.low << 1U) | (token.low >> 63U);
  const std::uint64_t identity = token.high ^ rotated_low;
  return identity == 0U ? 1U : identity;
}

void observe_latency_per_byte(double& ewma_us_per_byte, const std::uint64_t bytes,
                              const std::chrono::nanoseconds elapsed) noexcept {
  if (bytes == 0U || elapsed.count() < 0) {
    return;
  }
  const double sample = static_cast<double>(elapsed.count()) / 1000.0 / static_cast<double>(bytes);
  if (sample < 0.0) {
    return;
  }
  ewma_us_per_byte = ewma_us_per_byte <= 0.0 ? sample : sample * 0.25 + ewma_us_per_byte * 0.75;
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
    std::vector<ChunkRecord> chunks;
    std::vector<CompressionPath> backing_paths;
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

  struct RetiredReservation {
    cuda::abi::DevicePointer address = 0;
    std::uint64_t bytes = 0;
  };

  struct StagingSlot {
    void* memory = nullptr;
    cuda::abi::Event done = nullptr;
    std::uint64_t generation = 0;
  };

  struct LeaseChunk {
    ChunkKey key;
    std::uint64_t generation = 0;
    bool marks_dirty = false;
    bool provisional_full_write = false;
    bool spill_reserved_here = false;
  };

  struct RawSpill {
    HostBudgetReservation reservation;
    PreparedRawBacking backing;
  };

  struct CpuCompressionCandidate {
    Lz4BlocksV1 container;
    HostBudgetReservation reservation;
    std::uint64_t stored_payload_bytes = 0;
    std::uint64_t operation_id = 0;
  };

  struct ActiveExternalLease {
    ExternalLeaseId id;
    ExternalLeaseState state = ExternalLeaseState::armed;
    ExternalSealMode seal_mode = ExternalSealMode::success;
    std::vector<LeaseChunk> chunks;
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
    telemetry_.safe_device_budget_minimum_bytes = target;
    telemetry_.cuda_free_minimum_bytes = free_u64;
    telemetry_.cuda_free_end_bytes = free_u64;
    telemetry_.wddm_available_minimum_bytes = wddm_available;
    telemetry_.wddm_available_end_bytes = wddm_available;
    telemetry_.budget_samples = 1;
    configured_target_cap_ = config_.cache_target_bytes == 0
                                 ? static_cast<std::uint64_t>(cuda_total)
                                 : config_.cache_target_bytes;
    std::uint64_t device_reserve = config_.workspace_reserve_bytes;
    if (config_.compression_mode != CompressionMode::disabled) {
      const auto codec_slot_reserve = checked_multiply(config_.chunk_bytes, config_.codec_slots);
      const auto with_workspace =
          checked_add(device_reserve, config_.compression_workspace_cap_bytes);
      const auto with_slots = with_workspace.has_value() && codec_slot_reserve.has_value()
                                  ? checked_add(*with_workspace, *codec_slot_reserve)
                                  : std::nullopt;
      if (!with_slots.has_value()) {
        return fail(RuntimeStatus::invalid_argument, "setup", "codec_device_reserve",
                    "codec workspace and slot reserve overflowed");
      }
      device_reserve = *with_slots;
    }
    fixed_device_reserve_bytes_ = device_reserve;
    const std::uint64_t frame_budget = device_reserve < target ? target - device_reserve : 0;
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
    if (config_.compression_mode != CompressionMode::disabled &&
        (config_.compression_workspace_cap_bytes == 0U || config_.codec_slots < 2U ||
         config_.codec_slots > 8U || config_.codec_workers == 0U || config_.codec_workers > 8U)) {
      return fail(RuntimeStatus::invalid_argument, "setup", "compression_config",
                  "compression workspace, slots, or workers are out of range");
    }
    if (config_.compression_mode != CompressionMode::disabled &&
        (config_.resolve_host_budget ||
         (config_.host_store_cap_bytes != 0U && config_.host_headroom_bytes != 0U))) {
      HostMemorySample sample;
      try {
        if (config_.host_memory_sample) {
          sample = config_.host_memory_sample();
        } else {
          const probe::SystemInfo system = platform::collect_system_info();
          sample = {system.physical_memory_bytes, system.available_memory_bytes};
        }
      } catch (...) {
        return fail(RuntimeStatus::internal_failure, "setup", "host_budget",
                    "host memory sampling failed");
      }
      if (!sample.available_bytes.has_value() ||
          (config_.resolve_host_budget && config_.host_headroom_bytes == 0U &&
           !sample.physical_bytes.has_value())) {
        return fail(RuntimeStatus::unavailable, "setup", "host_budget",
                    "compressed backing admission requires host memory telemetry");
      }
      if (config_.resolve_host_budget) {
        if (config_.host_headroom_bytes == 0U) {
          constexpr std::uint64_t minimum_headroom = 4ULL * 1024ULL * 1024ULL * 1024ULL;
          config_.host_headroom_bytes = std::max(minimum_headroom, *sample.physical_bytes / 4U);
        }
        if (config_.host_store_cap_bytes == 0U) {
          if (*sample.available_bytes <= config_.host_headroom_bytes) {
            return fail(RuntimeStatus::host_oom, "setup", "host_budget",
                        "available RAM does not leave the requested host headroom");
          }
          config_.host_store_cap_bytes = *sample.available_bytes - config_.host_headroom_bytes;
        }
      }
      const auto requested_host =
          checked_add(config_.host_store_cap_bytes, config_.host_headroom_bytes);
      if (!requested_host.has_value() || *requested_host > *sample.available_bytes) {
        return fail(RuntimeStatus::unavailable, "setup", "host_budget",
                    "host store cap plus headroom exceeds currently available RAM");
      }
    }
    telemetry_.host_headroom_bytes = config_.host_headroom_bytes;
    const auto pinned_staging_bytes = checked_multiply(config_.chunk_bytes, config_.staging_slots);
    if (!pinned_staging_bytes.has_value()) {
      return fail(RuntimeStatus::invalid_argument, "setup", "staging_slots",
                  "pinned staging pool size overflowed");
    }

    if (!create_stream(h2d_stream_, "cuStreamCreate(h2d)") ||
        !create_stream(compute_stream_, "cuStreamCreate(compute)") ||
        !create_stream(d2h_stream_, "cuStreamCreate(d2h)") ||
        !create_event(compute_ready_, "cuEventCreate(compute_ready)") ||
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
    try {
      if (config_.compression_mode == CompressionMode::disabled) {
        backing_codec_ = std::make_unique<RawOnlyBlockCodec>();
      } else {
        backing_codec_ = std::make_unique<Lz4BlockCodec>();
      }
      backing_store_ = std::make_unique<HostBackingStore>(
          HostBackingConfig{config_.chunk_bytes, config_.host_store_cap_bytes}, *backing_codec_);
      if (config_.compression_mode != CompressionMode::disabled) {
        cpu_codec_queue_capacity_ =
            std::min<std::uint32_t>(cpu_codec_max_queue_capacity,
                                    std::max<std::uint32_t>(64U, config_.codec_workers * 32U));
        cpu_codec_pool_ = std::make_unique<CpuCodecWorkerPool>(
            CpuCodecPoolConfig{config_.codec_workers, cpu_codec_queue_capacity_}, *backing_codec_);
        if (!cpu_codec_pool_->valid()) {
          return fail(RuntimeStatus::invalid_argument, "setup", "create_cpu_codec_pool",
                      "CPU codec worker pool configuration is invalid");
        }
      }
    } catch (const std::bad_alloc&) {
      return fail(RuntimeStatus::host_oom, "setup", "create_backing_store",
                  "host backing store allocation failed");
    } catch (...) {
      return fail(RuntimeStatus::internal_failure, "setup", "create_backing_store",
                  "host backing store initialization failed");
    }
    if (const HostBudgetStatus budget_status = backing_store_->budget().reserve(
            HostBudgetCategory::pinned_staging, *pinned_staging_bytes, pinned_budget_reservation_);
        budget_status != HostBudgetStatus::success) {
      return fail(RuntimeStatus::host_oom, "setup", "reserve_pinned_staging",
                  std::string(host_budget_status_name(budget_status)));
    }
    pinned_budget_active_ = true;

    if (config_.compression_mode != CompressionMode::disabled) {
      if (config_.nvcomp_api != nullptr) {
        nvcomp_api_ = config_.nvcomp_api;
      } else {
        owned_nvcomp_api_ = std::make_unique<nvcomp::NvcompApi>();
        nvcomp_api_ = owned_nvcomp_api_.get();
      }
      const nvcomp::NvcompLoadResult codec_load = nvcomp_api_->load();
      nvcomp_available_ = codec_load.status == nvcomp::NvcompLoadStatus::loaded;
      telemetry_.nvcomp_available = nvcomp_available_;
      telemetry_.nvcomp_app_local = nvcomp_available_ && nvcomp_api_->library_source() ==
                                                             nvcomp::NvcompLibrarySource::app_local;
      telemetry_.nvcomp_version = nvcomp_available_ ? nvcomp_api_->version_string() : std::string{};
      telemetry_.nvcomp_library_sha256 = nvcomp_available_ && nvcomp_api_->integrity_verified()
                                             ? nvcomp_api_->library_sha256()
                                             : std::string{};
      // CPU LZ4 remains the authoritative, lossless fallback. nvCOMP unavailability therefore
      // does not reject an adaptive/capacity session.
      if (!nvcomp_available_) {
        ++telemetry_.gpu_codec_fallbacks;
        if (config_.forced_compression_path.has_value() &&
            config_.forced_compression_path != CompressionPath::raw) {
          return fail(RuntimeStatus::unsupported, "setup", "load_nvcomp",
                      "forced nvCOMP path is unavailable; no silent downgrade is permitted");
        }
      } else {
        NvcompPipelineConfig pipeline_config{config_.chunk_bytes,
                                             config_.codec_slots,
                                             config_.compression_workspace_cap_bytes,
                                             config_.stall_timeout,
                                             nullptr,
                                             nullptr};
        pipeline_config.trace_observer = &Impl::compression_trace_trampoline;
        pipeline_config.trace_user_data = this;
        nvcomp_pipeline_ = std::make_unique<NvcompLz4Pipeline>(api_, *nvcomp_api_, *backing_codec_,
                                                               pipeline_config);
        const NvcompPipelineStatus pipeline_status = nvcomp_pipeline_->setup();
        if (pipeline_status != NvcompPipelineStatus::success) {
          ++telemetry_.gpu_codec_fallbacks;
          (void)nvcomp_pipeline_->close();
          nvcomp_pipeline_.reset();
          nvcomp_available_ = false;
          telemetry_.nvcomp_available = false;
          telemetry_.nvcomp_app_local = false;
          telemetry_.nvcomp_version.clear();
          telemetry_.nvcomp_library_sha256.clear();
          codec_managed_device_bytes_ = 0;
          if (config_.forced_compression_path.has_value() &&
              config_.forced_compression_path != CompressionPath::raw) {
            return fail(RuntimeStatus::unsupported, "setup", "initialize_nvcomp",
                        "forced nvCOMP path could not initialize its event-safe pipeline");
          }
        } else {
          sync_codec_telemetry();
          const std::uint64_t codec_pinned_bytes = nvcomp_pipeline_->telemetry().pinned_slot_bytes;
          const HostBudgetStatus codec_budget_status = backing_store_->budget().reserve(
              HostBudgetCategory::pinned_staging, codec_pinned_bytes,
              codec_pinned_budget_reservation_);
          if (codec_budget_status != HostBudgetStatus::success) {
            ++telemetry_.gpu_codec_fallbacks;
            const NvcompPipelineStatus close_status = nvcomp_pipeline_->close();
            // Preserve the historical codec peak while clearing the current device-byte
            // accounting after a successful rollback.  This path may continue with the CPU
            // codec, so stale current bytes or an uncharged setup peak would make an otherwise
            // successful report internally inconsistent.
            sync_codec_telemetry();
            if (close_status != NvcompPipelineStatus::success) {
              async_resources_quarantined_ = true;
              telemetry_.quarantined = true;
              return poison("codec", "reserve_pinned_slots",
                            "nvCOMP slots exceeded the host budget and could not be cleaned up");
            }
            nvcomp_pipeline_.reset();
            nvcomp_available_ = false;
            telemetry_.nvcomp_available = false;
            telemetry_.nvcomp_app_local = false;
            telemetry_.nvcomp_version.clear();
            telemetry_.nvcomp_library_sha256.clear();
            codec_managed_device_bytes_ = 0;
            if (config_.forced_compression_path.has_value() &&
                config_.forced_compression_path != CompressionPath::raw) {
              return fail(RuntimeStatus::host_oom, "setup", "reserve_pinned_slots",
                          "forced nvCOMP path exceeds the configured host budget");
            }
          } else {
            codec_pinned_budget_active_ = true;
            suspended_codec_device_bytes_ = codec_managed_device_bytes_;
            suspended_codec_pinned_bytes_ = codec_pinned_bytes;
          }
        }
      }
    }
    if (config_.compression_mode != CompressionMode::disabled) {
      const auto actual_reserve =
          checked_add(config_.workspace_reserve_bytes, codec_managed_device_bytes_);
      if (!actual_reserve.has_value()) {
        return fail(RuntimeStatus::invalid_argument, "setup", "codec_device_reserve",
                    "actual codec device allocation accounting overflowed");
      }
      const std::uint64_t actual_frame_capacity =
          *actual_reserve < target ? (target - *actual_reserve) / config_.chunk_bytes : 0U;
      if (actual_frame_capacity < 2U) {
        return fail(RuntimeStatus::budget_pressure, "setup", "codec_device_reserve",
                    "actual codec resources leave fewer than two cache frames");
      }
      if (actual_frame_capacity > frame_capacity_) {
        try {
          frames_.resize(static_cast<std::size_t>(actual_frame_capacity));
        } catch (const std::bad_alloc&) {
          return fail(RuntimeStatus::host_oom, "setup", "resize_frame_metadata",
                      "cache frame metadata allocation failed");
        }
      } else if (actual_frame_capacity < frame_capacity_) {
        frames_.resize(static_cast<std::size_t>(actual_frame_capacity));
      }
      frame_capacity_ = actual_frame_capacity;
      maximum_frame_capacity_ = actual_frame_capacity;
      fixed_device_reserve_bytes_ = *actual_reserve;
    }
    telemetry_.device_reserve_bytes_peak =
        std::max(telemetry_.device_reserve_bytes_peak, fixed_device_reserve_bytes_);
    policy_ = config_.policy == RuntimePolicy::lru
                  ? std::unique_ptr<VictimPolicy>(std::make_unique<LruPolicy>())
                  : std::unique_ptr<VictimPolicy>(std::make_unique<ClockPolicy>());
    setup_complete_ = true;
    refresh_host_telemetry();
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
    try {
      owner->chunks.resize(static_cast<std::size_t>(*chunk_count));
      owner->backing_paths.assign(static_cast<std::size_t>(*chunk_count), CompressionPath::raw);
    } catch (const std::bad_alloc&) {
      return fail(RuntimeStatus::host_oom, "allocation", "allocate_chunk_metadata",
                  "chunk metadata allocation failed");
    } catch (...) {
      return fail(RuntimeStatus::internal_failure, "allocation", "allocate_chunk_metadata",
                  "chunk metadata allocation raised an unexpected exception");
    }
    for (std::uint64_t index = 0; index < *chunk_count; ++index) {
      const ChunkKey key{owner->id, index};
      owner->chunks[static_cast<std::size_t>(index)].key = key;
      const std::uint64_t begin = index * config_.chunk_bytes;
      const std::uint64_t valid = std::min(config_.chunk_bytes, bytes - begin);
      const BackingRepresentation initial = config_.compression_mode == CompressionMode::disabled
                                                ? BackingRepresentation::raw
                                                : BackingRepresentation::implicit_zero;
      const BackingResult registered = backing_store_->register_chunk(key, valid, initial);
      if (!registered) {
        for (std::uint64_t rollback = 0; rollback < index; ++rollback) {
          const ChunkKey rollback_key{owner->id, rollback};
          const BackingResult info = backing_store_->inspect(rollback_key);
          if (info) {
            (void)backing_store_->erase_chunk(rollback_key, info.chunk.generation);
            notify_lifecycle_progress();
          }
        }
        return fail(backing_runtime_status(registered.status), "allocation", "register_host_chunk",
                    std::string(backing_status_name(registered.status)));
      }
      ++telemetry_.generations_created;
      ++telemetry_.generations_committed;
      notify_lifecycle_progress();
    }
    const auto rollback_registered_backing = [&]() noexcept {
      for (std::uint64_t index = 0; index < *chunk_count; ++index) {
        const ChunkKey key{next_allocation_id_, index};
        const BackingResult info = backing_store_->inspect(key);
        if (info) {
          (void)backing_store_->erase_chunk(key, info.chunk.generation);
          notify_lifecycle_progress();
        }
      }
      refresh_host_telemetry();
    };
    try {
      auto [position, inserted] = allocations_.try_emplace(owner->id.value, std::move(owner));
      if (!inserted) {
        rollback_registered_backing();
        return fail(RuntimeStatus::internal_failure, "allocation", "register_allocation",
                    "allocation identifier was already registered");
      }
    } catch (const std::bad_alloc&) {
      rollback_registered_backing();
      return fail(RuntimeStatus::host_oom, "allocation", "register_allocation",
                  "allocation registry growth failed");
    } catch (...) {
      rollback_registered_backing();
      return fail(RuntimeStatus::internal_failure, "allocation", "register_allocation",
                  "allocation registry raised an unexpected exception");
    }

    Allocation& registered = *allocations_.at(next_allocation_id_.value);
    if (const cuda::abi::Result code = api_.mem_address_reserve_(
            &registered.reservation, static_cast<std::size_t>(registered.reservation_bytes), 0, 0,
            0);
        code != cuda::abi::success) {
      rollback_registered_backing();
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
      rollback_registered_backing();
      allocations_.erase(next_allocation_id_.value);
      return fail(RuntimeStatus::internal_failure, "allocation", "align_logical_base",
                  "padded reservation cannot contain the stable logical range");
    }
    registered.logical_base = static_cast<cuda::abi::DevicePointer>(*logical_base);
    output = RuntimeAllocation{next_allocation_id_, bytes};
    ++telemetry_.allocations_created;
    telemetry_.logical_bytes += bytes;
    refresh_host_telemetry();
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
    if (active_lease_references(id)) {
      return fail(RuntimeStatus::invalid_argument, "allocation", "release",
                  "allocation is referenced by an active external lease");
    }
    Allocation& owner = *found->second;
    if (config_.retain_released_va && owner.reservation != 0) {
      // Reserve the eventual ownership transfer before unmapping or erasing any backing. The
      // append below must not allocate after the allocation has been destructively retired.
      if (!checked_add(telemetry_.retired_va_bytes, owner.reservation_bytes).has_value() ||
          retired_reservations_.size() == retired_reservations_.max_size()) {
        return fail(RuntimeStatus::host_oom, "allocation", "reserve_retired_va_ledger",
                    "retired virtual reservation ledger capacity is exhausted");
      }
      if (retired_reservations_.size() == retired_reservations_.capacity()) {
        try {
          const std::size_t capacity = retired_reservations_.capacity();
          const std::size_t maximum = retired_reservations_.max_size();
          retired_reservations_.reserve(capacity == 0 ? 1U
                                                     : (capacity > maximum / 2U ? maximum
                                                                                 : capacity * 2U));
        } catch (const std::bad_alloc&) {
          return fail(RuntimeStatus::host_oom, "allocation", "reserve_retired_va_ledger",
                      "retired virtual reservation ledger allocation failed");
        } catch (...) {
          return fail(RuntimeStatus::internal_failure, "allocation", "reserve_retired_va_ledger",
                      "retired virtual reservation ledger growth failed");
        }
      }
    }
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
    if (!config_.retain_released_va && !owner.reservation_quarantined && owner.reservation != 0) {
      if (const cuda::abi::Result code = api_.mem_address_free_(
              owner.reservation, static_cast<std::size_t>(owner.reservation_bytes));
          code != cuda::abi::success) {
        return fail_cuda("cleanup", "cuMemAddressFree", code);
      }
      owner.reservation = 0;
    }
    for (const ChunkRecord& record : owner.chunks) {
      release_spill(record.key);
      const BackingResult info = backing_store_->inspect(record.key);
      if (info && !backing_store_->erase_chunk(record.key, info.chunk.generation)) {
        return fail(RuntimeStatus::cleanup_failure, "cleanup", "erase_host_chunk",
                    "authoritative backing generation could not be released");
      }
      notify_lifecycle_progress();
    }
    if (config_.retain_released_va && owner.reservation != 0) {
      retired_reservations_.push_back({owner.reservation, owner.reservation_bytes});
      ++telemetry_.retired_va_reservations;
      telemetry_.retired_va_bytes += owner.reservation_bytes;
      owner.reservation = 0;
    }
    telemetry_.logical_bytes = telemetry_.logical_bytes >= owner.logical_bytes
                                   ? telemetry_.logical_bytes - owner.logical_bytes
                                   : 0U;
    allocations_.erase(found);
    ++telemetry_.allocations_released;
    refresh_host_telemetry();
    return RuntimeStatus::success;
  }

  [[nodiscard]] RuntimeStatus discard_dead(const AllocationId id) {
    const Allocation* owner = allocation(id);
    if (!ready() || owner == nullptr) {
      return fail(RuntimeStatus::invalid_argument, "allocation", "discard_dead",
                  "allocation handle is unknown");
    }
    return discard_dead(id, 0, owner->logical_bytes);
  }

  [[nodiscard]] RuntimeStatus discard_dead(const AllocationId id, const std::uint64_t offset,
                                           const std::uint64_t bytes) {
    Allocation* owner = allocation(id);
    if (!ready() || owner == nullptr || bytes == 0 || offset > owner->logical_bytes ||
        bytes > owner->logical_bytes - offset) {
      return fail(RuntimeStatus::invalid_argument, "allocation", "discard_dead",
                  "dead allocation range is invalid");
    }
    const std::uint64_t end = offset + bytes;
    if (offset % config_.chunk_bytes != 0 ||
        (end != owner->logical_bytes && end % config_.chunk_bytes != 0)) {
      return fail(RuntimeStatus::invalid_argument, "allocation", "discard_dead",
                  "dead ranges must cover complete logical chunks");
    }
    const std::uint64_t first = offset / config_.chunk_bytes;
    const std::uint64_t last = (end - 1U) / config_.chunk_bytes;

    for (std::uint64_t index = first; index <= last; ++index) {
      const ChunkRecord& record = owner->chunks[static_cast<std::size_t>(index)];
      if (record.state == ChunkState::host_clean && record.pin_count == 0 &&
          !record.in_current_working_set && !record.staging_slot.has_value() &&
          validate_chunk_record(record) == ChunkInvariantError::none) {
        continue;
      }
      if ((record.state != ChunkState::resident_clean &&
           record.state != ChunkState::resident_dirty) ||
          !is_victim_eligible(record) || record.event_generation == 0 ||
          record.completed_generation != record.event_generation) {
        return fail(RuntimeStatus::invalid_argument, "allocation", "discard_dead",
                    "dead range still has pinned or in-flight users");
      }
    }

    for (std::uint64_t index = first; index <= last; ++index) {
      ChunkRecord& record = owner->chunks[static_cast<std::size_t>(index)];
      RuntimeStatus status = RuntimeStatus::success;
      if (record.state == ChunkState::resident_clean) {
        status = unmap(record.key);
      } else if (record.state == ChunkState::resident_dirty) {
        status = discard_dirty_mapping(record.key);
      }
      if (status != RuntimeStatus::success) {
        return status;
      }
      release_spill(record.key);
      const BackingResult info = backing_store_->inspect(record.key);
      if (!info) {
        return poison("cache", "discard_dead", "host backing chunk disappeared");
      }
      const BackingResult invalidated =
          backing_store_->invalidate(record.key, info.chunk.generation);
      if (!invalidated) {
        return fail(backing_runtime_status(invalidated.status), "cache", "discard_dead",
                    std::string(backing_status_name(invalidated.status)));
      }
      set_backing_path(record.key, CompressionPath::raw);
      ++telemetry_.generations_created;
      ++telemetry_.generations_committed;
    }
    refresh_host_telemetry();
    return RuntimeStatus::success;
  }

  [[nodiscard]] RuntimeStatus write(const AllocationId id, const std::uint64_t offset,
                                    const void* source, const std::uint64_t bytes) {
    Allocation* owner = allocation(id);
    if (!valid_host_range(owner, offset, source, bytes) || active_lease_references(id)) {
      return fail(RuntimeStatus::invalid_argument, "host_access", "write",
                  "host write range is invalid");
    }
    if (const RuntimeStatus status = flush_overlapping(*owner, offset, bytes, true);
        status != RuntimeStatus::success) {
      return status;
    }
    if (!host_write_covers_invalid_chunks(*owner, offset, bytes)) {
      return fail(RuntimeStatus::invalid_argument, "host_access", "write",
                  "host writes must fully replace every discarded chunk they overlap");
    }
    const auto* input = static_cast<const std::byte*>(source);
    std::uint64_t cursor = offset;
    std::uint64_t remaining = bytes;
    while (remaining != 0U) {
      const std::uint64_t chunk_index = cursor / config_.chunk_bytes;
      const std::uint64_t chunk_offset = cursor % config_.chunk_bytes;
      const ChunkKey key{id, chunk_index};
      const BackingResult info = backing_store_->inspect(key);
      if (!info) {
        return poison("host_access", "write", "authoritative host chunk disappeared");
      }
      const std::uint64_t segment = std::min(remaining, info.chunk.valid_bytes - chunk_offset);
      const auto segment_input =
          std::span<const std::byte>{input, static_cast<std::size_t>(segment)};
      const bool full_replace = chunk_offset == 0U && segment == info.chunk.valid_bytes;
      const BackingResult written =
          full_replace
              ? backing_store_->replace_raw(key, info.chunk.generation, segment_input)
              : backing_store_->write(key, info.chunk.generation, chunk_offset, segment_input);
      if (!written) {
        return fail(backing_runtime_status(written.status), "host_access", "write_backing",
                    std::string(backing_status_name(written.status)));
      }
      set_backing_path(key, CompressionPath::raw);
      ++telemetry_.generations_created;
      ++telemetry_.generations_committed;
      if (const RuntimeStatus status = consider_compression(key);
          status != RuntimeStatus::success) {
        return status;
      }
      input += segment;
      cursor += segment;
      remaining -= segment;
    }
    refresh_host_telemetry();
    return RuntimeStatus::success;
  }

  [[nodiscard]] RuntimeStatus read(const AllocationId id, const std::uint64_t offset,
                                   void* destination, const std::uint64_t bytes) {
    Allocation* owner = allocation(id);
    if (!valid_host_range(owner, offset, destination, bytes) || active_lease_references(id)) {
      return fail(RuntimeStatus::invalid_argument, "host_access", "read",
                  "host read range is invalid");
    }
    if (const RuntimeStatus status = flush_overlapping(*owner, offset, bytes, false);
        status != RuntimeStatus::success) {
      return status;
    }
    if (!host_range_is_valid(*owner, offset, bytes)) {
      return fail(RuntimeStatus::invalid_argument, "host_access", "read",
                  "host read overlaps liveness-discarded data");
    }
    auto* output = static_cast<std::byte*>(destination);
    std::uint64_t cursor = offset;
    std::uint64_t remaining = bytes;
    while (remaining != 0U) {
      const std::uint64_t chunk_index = cursor / config_.chunk_bytes;
      const std::uint64_t chunk_offset = cursor % config_.chunk_bytes;
      const ChunkKey key{id, chunk_index};
      const BackingResult info = backing_store_->inspect(key);
      if (!info) {
        return poison("host_access", "read", "authoritative host chunk disappeared");
      }
      const std::uint64_t segment = std::min(remaining, info.chunk.valid_bytes - chunk_offset);
      const auto destination_span = std::span<std::byte>{output, static_cast<std::size_t>(segment)};
      const auto started = Clock::now();
      const BackingResult read_result = backing_store_->read(key, chunk_offset, destination_span);
      const auto elapsed =
          std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - started);
      if (!read_result) {
        return fail(backing_runtime_status(read_result.status), "host_access", "read_backing",
                    std::string(backing_status_name(read_result.status)));
      }
      if (info.chunk.representation == BackingRepresentation::lz4_blocks) {
        ++telemetry_.decompression_attempts;
        ++telemetry_.decompression_commits;
        ++telemetry_.cpu_decode_operations;
        telemetry_.cpu_decode_nanoseconds += static_cast<std::uint64_t>(elapsed.count());
      }
      output += segment;
      cursor += segment;
      remaining -= segment;
    }
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
    if (!plan_has_valid_host_sources(plan.chunks)) {
      return fail(RuntimeStatus::invalid_argument, "transaction", "prefetch",
                  "prefetch cannot read a liveness-discarded host chunk");
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

  [[nodiscard]] RuntimeStatus acquire_external(const ExternalLeaseRequest& request,
                                               ExternalLease& output) {
    output = {};
    if (!ready() || request.ranges.empty() || active_external_lease_.has_value() ||
        !next_external_lease_id_) {
      return fail(RuntimeStatus::invalid_argument, "transaction", "acquire_external",
                  "runtime state, access ranges, or lease identifier is invalid");
    }

    cuda::abi::Context current_context = nullptr;
    if (const cuda::abi::Result code = api_.context_get_current_(&current_context);
        code != cuda::abi::success) {
      return fail_cuda("transaction", "cuCtxGetCurrent", code);
    }
    if (current_context != context_) {
      return fail(RuntimeStatus::invalid_argument, "transaction", "context",
                  "external lease must be acquired with the runtime CUDA context current");
    }
    cuda::abi::Device current_device = -1;
    if (const cuda::abi::Result code = api_.context_get_device_(&current_device);
        code != cuda::abi::success) {
      return fail_cuda("transaction", "cuCtxGetDevice", code);
    }
    if (current_device != device_) {
      return fail(RuntimeStatus::invalid_argument, "transaction", "device",
                  "external lease CUDA device does not match the runtime device");
    }
    cuda::abi::Context stream_context = nullptr;
    if (const cuda::abi::Result code = api_.stream_get_context_(compute_stream_, &stream_context);
        code != cuda::abi::success) {
      return fail_cuda("transaction", "cuStreamGetCtx", code);
    }
    if (stream_context != context_) {
      return fail(RuntimeStatus::invalid_argument, "transaction", "stream_context",
                  "runtime compute stream belongs to a different CUDA context");
    }
    cuda::abi::StreamCaptureStatus capture = cuda::abi::stream_capture_invalidated;
    if (const cuda::abi::Result code = api_.stream_is_capturing_(compute_stream_, &capture);
        code != cuda::abi::success) {
      return fail_cuda("transaction", "cuStreamIsCapturing", code);
    }
    if (capture != cuda::abi::stream_capture_none) {
      return fail(RuntimeStatus::unsupported, "transaction", "stream_capture",
                  "external VMM leases are not supported during CUDA stream capture");
    }

    std::vector<AllocationLayout> layouts = allocation_layouts();
    const AccessPlanResult plan =
        normalize_and_split_accesses(layouts, request.ranges, config_.chunk_bytes);
    if (!plan || plan.chunks.empty()) {
      return fail(RuntimeStatus::invalid_argument, "transaction", "normalize_accesses",
                  std::string("invalid access ranges: ") +
                      std::string(access_plan_error_name(plan.error)));
    }
    if (!plan_has_valid_host_sources(plan.chunks)) {
      return fail(RuntimeStatus::invalid_argument, "transaction", "acquire_external",
                  "discarded chunks require a full write-only replacement");
    }
    if (const RuntimeStatus status =
            refresh_budget(plan.chunks.size(), request.workspace_bytes, false);
        status != RuntimeStatus::success) {
      return status;
    }
    if (plan.chunks.size() > frame_capacity_) {
      return fail(RuntimeStatus::budget_pressure, "transaction", "working_set",
                  "declared working set exceeds the live cache target");
    }
    if (const RuntimeStatus status = ensure_workspace(request.workspace_bytes);
        status != RuntimeStatus::success) {
      return status;
    }

    std::vector<ResolvedRange> resolved;
    ActiveExternalLease lease;
    lease.id = next_external_lease_id_;
    try {
      resolved.reserve(plan.normalized_ranges.size());
      for (const AccessRange& range : plan.normalized_ranges) {
        const Allocation* owner = allocation(range.allocation_id);
        if (owner == nullptr) {
          return poison("transaction", "resolve_range", "normalized allocation disappeared");
        }
        resolved.push_back(ResolvedRange{range.allocation_id, range.offset_bytes,
                                         range.length_bytes, range.mode,
                                         owner->logical_base + range.offset_bytes});
      }
      lease.chunks.reserve(plan.chunks.size());
      for (const ChunkAccessPlan& chunk_plan : plan.chunks) {
        lease.chunks.push_back(LeaseChunk{chunk_plan.key, 0, chunk_plan.marks_dirty, false});
      }
    } catch (const std::bad_alloc&) {
      return fail(RuntimeStatus::host_oom, "transaction", "prepare_lease",
                  "external lease metadata allocation failed");
    } catch (...) {
      return fail(RuntimeStatus::internal_failure, "transaction", "prepare_lease",
                  "external lease metadata allocation raised an unexpected exception");
    }
    if (next_external_lease_id_.value == std::numeric_limits<std::uint64_t>::max()) {
      next_external_lease_id_ = {};
    } else {
      ++next_external_lease_id_.value;
    }

    const auto rollback = [&]() noexcept {
      bool retired = true;
      for (LeaseChunk& use : lease.chunks) {
        if (use.generation == 0) {
          continue;
        }
        ChunkRecord* record = chunk(use.key);
        if (record == nullptr ||
            complete_event_generation(*record, use.generation) != EventGenerationResult::success) {
          retired = false;
        }
      }
      for (LeaseChunk& use : lease.chunks) {
        if (use.generation == 0) {
          continue;
        }
        ChunkRecord* record = chunk(use.key);
        if (record == nullptr) {
          retired = false;
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
      for (LeaseChunk& use : lease.chunks) {
        if (!use.provisional_full_write) {
          continue;
        }
        ChunkRecord* record = chunk(use.key);
        if (record != nullptr && record->state == ChunkState::resident_clean &&
            unmap(use.key) != RuntimeStatus::success) {
          retired = false;
        }
      }
      for (const LeaseChunk& use : lease.chunks) {
        if (use.spill_reserved_here) {
          release_spill(use.key);
        }
      }
      return retired;
    };

    ++telemetry_.transactions_submitted;
    for (std::size_t index = 0; index < plan.chunks.size(); ++index) {
      const ChunkAccessPlan& chunk_plan = plan.chunks[index];
      LeaseChunk& use = lease.chunks[index];
      ChunkRecord* record = chunk(chunk_plan.key);
      const bool provisional_full_write = record != nullptr &&
                                          record->state == ChunkState::host_clean &&
                                          chunk_plan.full_write_only;
      if (chunk_plan.marks_dirty &&
          spill_reservations_.find(chunk_plan.key) == spill_reservations_.end()) {
        const RuntimeStatus spill_status = reserve_spill(chunk_plan.key, plan.chunks);
        if (spill_status != RuntimeStatus::success) {
          if (!rollback()) {
            return poison("transaction", "spill_rollback",
                          "spill reservations could not roll back before launch");
          }
          return spill_status;
        }
        use.spill_reserved_here = true;
      }
      bool hit = false;
      const RuntimeStatus resident_status = ensure_resident(chunk_plan, false, hit);
      if (resident_status != RuntimeStatus::success) {
        if (!rollback()) {
          return poison("transaction", "prepare_rollback",
                        "pre-launch working-set generations could not retire");
        }
        if (resident_status == RuntimeStatus::budget_pressure) {
          const RuntimeStatus shrink_status = refresh_budget(0, workspace_bytes_, true);
          if (shrink_status != RuntimeStatus::success &&
              shrink_status != RuntimeStatus::budget_pressure) {
            return shrink_status;
          }
        }
        return resident_status;
      }
      use.provisional_full_write = provisional_full_write;
      record = chunk(chunk_plan.key);
      if (record == nullptr || begin_event_generation(*record) != EventGenerationResult::success) {
        if (!rollback()) {
          return poison("transaction", "pin_rollback",
                        "working-set generations could not retire after pin failure");
        }
        return poison("transaction", "pin_working_set", "chunk event generation cannot be started");
      }
      use.generation = record->event_generation;
      ++record->pin_count;
      record->in_current_working_set = true;
    }

    if (const cuda::abi::Result code = api_.event_record_(compute_ready_, h2d_stream_);
        code != cuda::abi::success) {
      if (!rollback()) {
        return poison("transaction", "ready_rollback",
                      "working-set generations could not retire after ready-event failure");
      }
      return fail_cuda("transaction", "cuEventRecord(compute_ready)", code);
    }
    if (const cuda::abi::Result code = api_.stream_wait_event_(compute_stream_, compute_ready_, 0U);
        code != cuda::abi::success) {
      if (!rollback()) {
        return poison("transaction", "stream_wait_rollback",
                      "working-set generations could not retire after stream-wait failure");
      }
      return fail_cuda("transaction", "cuStreamWaitEvent(compute_ready)", code);
    }
    if (const cuda::abi::Result code = api_.event_record_(compute_started_, compute_stream_);
        code != cuda::abi::success) {
      active_external_lease_ = std::move(lease);
      async_resources_quarantined_ = true;
      return poison_cuda("transaction", "cuEventRecord(compute_start)", code);
    }

    output.id = lease.id;
    output.ranges = std::move(resolved);
    output.workspace_address = workspace_;
    output.workspace_bytes = request.workspace_bytes;
    output.stream = compute_stream_;
    active_external_lease_ = std::move(lease);
    return RuntimeStatus::success;
  }

  [[nodiscard]] RuntimeStatus seal_external(const ExternalLeaseId lease_id,
                                            const ExternalSealMode mode) {
    if (!lease_id || !active_external_lease_.has_value() ||
        active_external_lease_->id != lease_id ||
        active_external_lease_->state != ExternalLeaseState::armed) {
      return fail(RuntimeStatus::invalid_argument, "transaction", "seal_external",
                  "external lease is unknown or is not armed");
    }
    ActiveExternalLease& lease = *active_external_lease_;
    if (const cuda::abi::Result code = api_.event_record_(compute_done_, compute_stream_);
        code != cuda::abi::success) {
      lease.state = ExternalLeaseState::quarantined;
      async_resources_quarantined_ = true;
      return poison_cuda("transaction", "cuEventRecord(compute_done)", code);
    }
    lease.state = ExternalLeaseState::submitted;
    lease.seal_mode = mode;
    if (mode != ExternalSealMode::cancelled_before_submission) {
      for (const LeaseChunk& use : lease.chunks) {
        if (!use.marks_dirty) {
          continue;
        }
        ChunkRecord* record = chunk(use.key);
        if (record == nullptr) {
          return poison("transaction", "mark_dirty", "leased chunk disappeared after sealing");
        }
        if (record->state == ChunkState::resident_clean &&
            transition_chunk_state(record->state, ChunkState::resident_dirty) !=
                StateTransitionResult::success) {
          return poison_transition("transaction", "mark_dirty", "illegal dirty transition");
        }
        if (record->state != ChunkState::resident_dirty) {
          return poison_transition("transaction", "mark_dirty",
                                   "write-capable leased chunk is not resident");
        }
      }
    }
    return RuntimeStatus::success;
  }

  [[nodiscard]] RuntimeStatus poll_external(const ExternalLeaseId lease_id,
                                            ExternalLeasePoll& output) {
    if (!lease_id) {
      return fail(RuntimeStatus::invalid_argument, "transaction", "poll_external",
                  "external lease identifier is invalid");
    }
    if (last_external_lease_.has_value() && last_external_lease_->id == lease_id) {
      output = *last_external_lease_;
      return RuntimeStatus::success;
    }
    if (!active_external_lease_.has_value() || active_external_lease_->id != lease_id) {
      return fail(RuntimeStatus::invalid_argument, "transaction", "poll_external",
                  "external lease is unknown");
    }
    ActiveExternalLease& lease = *active_external_lease_;
    output = ExternalLeasePoll{lease.id, lease.state, RuntimeStatus::success, 0.0};
    if (lease.state == ExternalLeaseState::armed) {
      return RuntimeStatus::success;
    }
    if (lease.state == ExternalLeaseState::quarantined) {
      output.result = RuntimeStatus::poisoned;
      return RuntimeStatus::poisoned;
    }
    if (lease.state != ExternalLeaseState::submitted) {
      return poison("transaction", "poll_external", "external lease state is invalid");
    }

    const cuda::abi::Result query = api_.event_query_(compute_done_);
    if (query == CUDA_ERROR_NOT_READY) {
      return RuntimeStatus::success;
    }
    if (query != cuda::abi::success) {
      lease.state = ExternalLeaseState::quarantined;
      output.state = lease.state;
      output.result = RuntimeStatus::poisoned;
      async_resources_quarantined_ = true;
      return poison_cuda("event", "cuEventQuery(compute_done)", query);
    }

    float elapsed = 0.0F;
    if (const cuda::abi::Result code =
            api_.event_elapsed_time_(&elapsed, compute_started_, compute_done_);
        code != cuda::abi::success) {
      lease.state = ExternalLeaseState::quarantined;
      output.state = lease.state;
      output.result = RuntimeStatus::poisoned;
      async_resources_quarantined_ = true;
      return poison_cuda("transaction", "cuEventElapsedTime", code);
    }
    telemetry_.last_transaction_ms = static_cast<double>(elapsed);

    bool retired = true;
    for (const LeaseChunk& use : lease.chunks) {
      ChunkRecord* record = chunk(use.key);
      if (record == nullptr ||
          complete_event_generation(*record, use.generation) != EventGenerationResult::success) {
        retired = false;
      }
    }
    for (const LeaseChunk& use : lease.chunks) {
      ChunkRecord* record = chunk(use.key);
      if (record == nullptr) {
        retired = false;
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
    if (!retired) {
      lease.state = ExternalLeaseState::quarantined;
      output.state = lease.state;
      output.result = RuntimeStatus::poisoned;
      return poison("transaction", "retire_generation",
                    "external working-set generation could not retire");
    }

    if (lease.seal_mode == ExternalSealMode::cancelled_before_submission) {
      for (const LeaseChunk& use : lease.chunks) {
        if (!use.provisional_full_write) {
          continue;
        }
        ChunkRecord* record = chunk(use.key);
        if (record != nullptr && record->state == ChunkState::resident_clean &&
            unmap(use.key) != RuntimeStatus::success) {
          lease.state = ExternalLeaseState::quarantined;
          output.state = lease.state;
          output.result = RuntimeStatus::poisoned;
          return error_.status;
        }
      }
      for (const LeaseChunk& use : lease.chunks) {
        if (use.spill_reserved_here) {
          release_spill(use.key);
        }
      }
    }

    output.elapsed_ms = static_cast<double>(elapsed);
    if (lease.seal_mode == ExternalSealMode::cancelled_before_submission) {
      output.state = ExternalLeaseState::cancelled;
      output.result = RuntimeStatus::callback_skipped;
    } else if (lease.seal_mode == ExternalSealMode::failed_after_possible_submission) {
      output.state = ExternalLeaseState::failed;
      output.result = RuntimeStatus::callback_failed;
    } else {
      output.state = ExternalLeaseState::completed;
      output.result = RuntimeStatus::success;
      ++telemetry_.transactions_completed;
      if (config_.maximum_transaction_duration.count() > 0 &&
          static_cast<double>(elapsed) >
              static_cast<double>(config_.maximum_transaction_duration.count())) {
        launches_blocked_ = true;
        ++telemetry_.watchdog_rejections;
        output.state = ExternalLeaseState::failed;
        output.result = RuntimeStatus::timeout;
        (void)fail(RuntimeStatus::timeout, "transaction", "duration_guard",
                   "transaction exceeded the configured kernel safety duration");
      }
    }

    last_external_lease_ = output;
    active_external_lease_.reset();
    if (output.result == RuntimeStatus::callback_failed) {
      (void)poison("transaction", "callback", "transaction callback rejected submission",
                   RuntimeStatus::callback_failed);
    }
    return RuntimeStatus::success;
  }

  [[nodiscard]] RuntimeStatus wait_external(const ExternalLeaseId lease_id,
                                            const std::chrono::milliseconds timeout,
                                            ExternalLeasePoll& output) {
    if (timeout.count() <= 0) {
      return fail(RuntimeStatus::invalid_argument, "transaction", "wait_external",
                  "external lease timeout must be positive");
    }
    if (active_external_lease_.has_value() && active_external_lease_->id == lease_id &&
        active_external_lease_->state == ExternalLeaseState::armed) {
      return fail(RuntimeStatus::invalid_argument, "transaction", "wait_external",
                  "external lease must be sealed before it can be waited");
    }
    const Clock::time_point now = Clock::now();
    const std::chrono::milliseconds maximum_timeout =
        std::chrono::duration_cast<std::chrono::milliseconds>(Clock::time_point::max() - now);
    const Clock::time_point deadline =
        timeout >= maximum_timeout ? Clock::time_point::max() : now + timeout;
    for (;;) {
      const RuntimeStatus status = poll_external(lease_id, output);
      if (status != RuntimeStatus::success) {
        return status;
      }
      if (output.state == ExternalLeaseState::completed ||
          output.state == ExternalLeaseState::cancelled ||
          output.state == ExternalLeaseState::failed) {
        return output.result;
      }
      if (Clock::now() >= deadline) {
        if (active_external_lease_.has_value() && active_external_lease_->id == lease_id) {
          active_external_lease_->state = ExternalLeaseState::quarantined;
        }
        output.state = ExternalLeaseState::quarantined;
        output.result = RuntimeStatus::timeout;
        async_resources_quarantined_ = true;
        return poison("event", "wait_external",
                      "external completion event did not retire before the stall deadline",
                      RuntimeStatus::timeout);
      }
      std::this_thread::yield();
    }
  }

  [[nodiscard]] RuntimeStatus execute(const std::span<const AccessRange> ranges,
                                      const std::uint64_t workspace_bytes,
                                      const TransactionCallback& callback) {
    if (!callback) {
      return fail(RuntimeStatus::invalid_argument, "transaction", "execute",
                  "transaction callback is invalid");
    }
    ExternalLease lease;
    const ExternalLeaseRequest request{ranges, workspace_bytes};
    if (const RuntimeStatus status = acquire_external(request, lease);
        status != RuntimeStatus::success) {
      return status;
    }

    RuntimeStatus callback_status = RuntimeStatus::internal_failure;
    try {
      callback_status = callback(TransactionContext{lease.ranges, lease.workspace_address,
                                                    lease.workspace_bytes, lease.stream});
    } catch (...) {
      callback_status = RuntimeStatus::callback_failed;
    }
    const ExternalSealMode seal_mode =
        callback_status == RuntimeStatus::success
            ? ExternalSealMode::success
            : (callback_status == RuntimeStatus::callback_skipped
                   ? ExternalSealMode::cancelled_before_submission
                   : ExternalSealMode::failed_after_possible_submission);
    if (const RuntimeStatus status = seal_external(lease.id, seal_mode);
        status != RuntimeStatus::success) {
      return status;
    }
    ExternalLeasePoll completion;
    return wait_external(lease.id, config_.stall_timeout, completion);
  }

  [[nodiscard]] RuntimeStatus drain(const bool release_mappings) {
    if (!setup_complete_) {
      return RuntimeStatus::success;
    }
    if (active_external_lease_.has_value()) {
      if (active_external_lease_->state != ExternalLeaseState::submitted) {
        return fail(RuntimeStatus::invalid_argument, "transaction", "drain",
                    "an armed external lease must be sealed before draining");
      }
      ExternalLeasePoll completion;
      const RuntimeStatus status =
          wait_external(active_external_lease_->id, config_.stall_timeout, completion);
      if (status != RuntimeStatus::success && status != RuntimeStatus::callback_skipped &&
          status != RuntimeStatus::callback_failed) {
        return status;
      }
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
    bool cleanup_ledger_known = true;
    bool cpu_operations_drained = true;
    const auto note_cleanup_failure = [&]() noexcept {
      aggregate = RuntimeStatus::cleanup_failure;
    };

    if (active_external_lease_.has_value()) {
      if (active_external_lease_->state == ExternalLeaseState::submitted) {
        ExternalLeasePoll completion;
        const RuntimeStatus status =
            wait_external(active_external_lease_->id, config_.stall_timeout, completion);
        if (status != RuntimeStatus::success && status != RuntimeStatus::callback_skipped &&
            status != RuntimeStatus::callback_failed) {
          note_cleanup_failure();
        }
      }
      if (active_external_lease_.has_value()) {
        // An armed or unobservable lease may already have client work in flight. It is never safe
        // to infer cancellation from the absence of a seal call.
        active_external_lease_->state = ExternalLeaseState::quarantined;
        async_resources_quarantined_ = true;
        telemetry_.quarantined = true;
        poisoned_ = true;
        note_cleanup_failure();
      }
    }

    // Close is deliberately best-effort. A failure on one frame must not prevent us from
    // retiring other independently safe mappings. Conversely, transient or structurally
    // inconsistent mappings remain quarantined rather than being unmapped speculatively.
    if (setup_complete_) {
      for (std::size_t index = 0; index < frames_.size(); ++index) {
        Frame& frame = frames_[index];
        if (!frame.mapped) {
          continue;
        }
        if (async_resources_quarantined_) {
          // An earlier unknown submission may still own a shared transfer staging slot/stream.
          // Do not start another writeback while cleaning other dirty frames: cycling the pool
          // could reuse that slot before its unobservable generation has retired. Keep every
          // mapped frame at the worker boundary; known-safe unmap failures still use best-effort
          // independent cleanup below because they do not set this asynchronous flag.
          frame.quarantined = true;
          if (frame.key.has_value()) {
            if (Allocation* owner = allocation(frame.key->allocation_id); owner != nullptr) {
              quarantine_mapping(*owner);
            }
          }
          note_cleanup_failure();
          continue;
        }
        if (!frame.key.has_value()) {
          cleanup_ledger_known = false;
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
          cleanup_ledger_known = false;
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
      for (const ChunkRecord& record : owner->chunks) {
        release_spill(record.key);
        const BackingResult info =
            backing_store_ != nullptr ? backing_store_->inspect(record.key) : BackingResult{};
        if (backing_store_ != nullptr && info &&
            !backing_store_->erase_chunk(record.key, info.chunk.generation)) {
          note_cleanup_failure();
        }
        notify_lifecycle_progress();
      }
    }
    // Every retired reservation was fully unmapped before release committed. It remains
    // independently safe to free even if another, still-live allocation is quarantined. Keep
    // failed entries owned by this ledger for the rest of the runtime lifetime.
    for (RetiredReservation& reservation : retired_reservations_) {
      if (api_.mem_address_free_(reservation.address, static_cast<std::size_t>(reservation.bytes)) !=
          cuda::abi::success) {
        telemetry_.quarantined = true;
        poisoned_ = true;
        note_cleanup_failure();
      } else {
        --telemetry_.retired_va_reservations;
        telemetry_.retired_va_bytes -= reservation.bytes;
        ++telemetry_.retired_va_reservations_freed;
        telemetry_.retired_va_bytes_freed += reservation.bytes;
        reservation.address = 0;
      }
      notify_lifecycle_progress();
    }
    std::erase_if(retired_reservations_,
                  [](const RetiredReservation& reservation) { return reservation.address == 0; });
    refresh_host_telemetry();
    bool chunk_users_retired = cleanup_ledger_known;
    for (const auto& [id, owner] : allocations_) {
      (void)id;
      for (const ChunkRecord& record : owner->chunks) {
        chunk_users_retired = chunk_users_retired && record.pin_count == 0 &&
                              !record.in_current_working_set && !record.staging_slot.has_value() &&
                              record.event_generation == record.completed_generation;
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
        if (async_resources_quarantined_) {
          // A failed event record/query or an unsealed external lease leaves it unknown whether
          // the GPU still references this physical allocation. Keep the VMM handle together with
          // its mapping and reservation until the isolated worker exits. This differs from a
          // completed cuMemUnmap failure: in that event-safe case CUDA permits dropping the
          // explicit handle while the mapping itself retains the allocation.
          continue;
        }
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
    if (nvcomp_pipeline_ != nullptr) {
      const NvcompPipelineStatus codec_close = nvcomp_pipeline_->close();
      sync_codec_telemetry();
      if (codec_close != NvcompPipelineStatus::success) {
        note_cleanup_failure();
        if (codec_close == NvcompPipelineStatus::quarantined || nvcomp_pipeline_->quarantined()) {
          async_resources_quarantined_ = true;
          telemetry_.quarantined = true;
          poisoned_ = true;
        }
      }
      nvcomp_pipeline_.reset();
    }
    if (cpu_codec_pool_ != nullptr) {
      if (cpu_codec_pool_->close() != CpuCodecPoolStatus::success) {
        cpu_operations_drained = false;
        note_cleanup_failure();
      }
      cpu_codec_pool_.reset();
    }
    if (backing_store_ != nullptr && codec_pinned_budget_active_ && !async_resources_quarantined_) {
      if (backing_store_->budget().release(codec_pinned_budget_reservation_) !=
          HostBudgetStatus::success) {
        note_cleanup_failure();
      }
      codec_pinned_budget_active_ = false;
      refresh_host_telemetry();
    }
    if (async_resources_quarantined_) {
      // No trustworthy completion boundary exists. Keep memory, event, stream, workspace, and
      // owned-context resources alive until the quarantine process exits; destroying any of them
      // here could race an unobserved DMA transfer or kernel.
      if (owned_nvcomp_api_ != nullptr) {
        owned_nvcomp_api_->abandon();
      }
      note_cleanup_failure();
    } else {
      destroy_staging_pool(h2d_staging_, aggregate);
      destroy_staging_pool(d2h_staging_, aggregate);
      destroy_event(compute_ready_, aggregate);
      destroy_event(compute_started_, aggregate);
      destroy_event(compute_done_, aggregate);
      destroy_stream(h2d_stream_, aggregate);
      destroy_stream(compute_stream_, aggregate);
      destroy_stream(d2h_stream_, aggregate);
    }
    if (backing_store_ != nullptr && pinned_budget_active_ && !async_resources_quarantined_) {
      if (backing_store_->budget().release(pinned_budget_reservation_) !=
          HostBudgetStatus::success) {
        note_cleanup_failure();
      }
      pinned_budget_active_ = false;
      refresh_host_telemetry();
    }
    backing_store_.reset();
    backing_codec_.reset();
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
    telemetry_.cleanup_events_drained =
        chunk_users_retired && !active_external_lease_.has_value() &&
        !async_resources_quarantined_ &&
        telemetry_.codec_events_recorded == telemetry_.codec_events_retired;
    telemetry_.cleanup_operations_drained =
        telemetry_.cleanup_events_drained && cpu_operations_drained;
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

  [[nodiscard]] std::optional<cuda::abi::DevicePointer>
  allocation_address(const AllocationId id) const noexcept {
    const Allocation* owner = allocation(id);
    if (!ready() || owner == nullptr || owner->reservation == 0 ||
        owner->reservation_quarantined || owner->logical_base == 0) {
      return std::nullopt;
    }
    return owner->logical_base;
  }

  [[nodiscard]] std::optional<HostChunkInfo>
  backing_info(const AllocationId id, const std::uint64_t chunk_index) const noexcept {
    const Allocation* owner = allocation(id);
    if (owner == nullptr || backing_store_ == nullptr || chunk_index >= owner->chunks.size()) {
      return std::nullopt;
    }
    const BackingResult info =
        backing_store_->inspect(owner->chunks[static_cast<std::size_t>(chunk_index)].key);
    return info ? std::optional<HostChunkInfo>{info.chunk} : std::nullopt;
  }

private:
  [[nodiscard]] std::chrono::nanoseconds
  compression_cost_duration(const CompressionPath path,
                            const std::chrono::nanoseconds observed) const noexcept {
    if (config_.compression_cost_sample) {
      try {
        const auto sample = config_.compression_cost_sample(path, observed);
        if (sample.count() >= 0) {
          return sample;
        }
      } catch (...) {
        // Test instrumentation must not invalidate measured costs or alter runtime safety.
      }
    }
    return observed;
  }

  void notify_lifecycle_progress() noexcept {
    if (!config_.lifecycle_progress) {
      return;
    }
    try {
      config_.lifecycle_progress();
    } catch (...) {
      // Runtime correctness never depends on an observer supplied by a controller frontend.
    }
  }

  void emit_compression_trace(const CompressionTraceEvent& event) noexcept {
    if (!config_.compression_trace) {
      return;
    }
    try {
      config_.compression_trace(event);
    } catch (...) {
      // Tracing is advisory. An observer must never perturb an event-safe runtime transition.
    }
  }

  static void compression_trace_trampoline(void* user_data,
                                           const CompressionTraceEvent& event) noexcept {
    if (user_data != nullptr) {
      static_cast<Impl*>(user_data)->emit_compression_trace(event);
    }
  }

  [[nodiscard]] CompressionPath backing_path(const ChunkKey key) const noexcept {
    const Allocation* owner = allocation(key.allocation_id);
    if (owner == nullptr || key.chunk_index >= owner->backing_paths.size()) {
      return CompressionPath::raw;
    }
    return owner->backing_paths[static_cast<std::size_t>(key.chunk_index)];
  }

  void set_backing_path(const ChunkKey key, const CompressionPath path) noexcept {
    Allocation* owner = allocation(key.allocation_id);
    if (owner != nullptr && key.chunk_index < owner->backing_paths.size()) {
      owner->backing_paths[static_cast<std::size_t>(key.chunk_index)] = path;
    }
  }

  void emit_generation_discard(const ChunkKey key, const std::uint64_t operation_id,
                               const HostChunkInfo& source, const std::uint64_t target_generation,
                               const std::optional<std::uint64_t> slot_generation,
                               const CompressionPath path, const std::uint64_t physical_bytes,
                               const std::string_view reason,
                               const bool speculative = false) noexcept {
    CompressionTraceEvent event;
    event.kind = CompressionTraceEventKind::generation_discard;
    event.key = key;
    event.operation_id = operation_id;
    event.source_generation = source.generation;
    event.target_generation = target_generation;
    event.slot_generation = slot_generation;
    event.path = path;
    event.from_representation = source.representation;
    event.to_representation = BackingRepresentation::lz4_blocks;
    event.logical_bytes = source.valid_bytes;
    event.physical_bytes = physical_bytes;
    event.reason = reason;
    event.speculative = speculative;
    emit_compression_trace(event);
  }

  [[nodiscard]] RuntimeStatus suspend_codec_pipeline_for_budget() {
    if (nvcomp_pipeline_ == nullptr) {
      return RuntimeStatus::success;
    }
    const NvcompPipelineTelemetry resources = nvcomp_pipeline_->telemetry();
    const NvcompPipelineStatus close_status = nvcomp_pipeline_->close();
    sync_codec_telemetry();
    if (close_status != NvcompPipelineStatus::success) {
      if (close_status == NvcompPipelineStatus::quarantined || nvcomp_pipeline_->quarantined()) {
        async_resources_quarantined_ = true;
        telemetry_.quarantined = true;
        return poison("budget", "close_codec_pipeline",
                      "codec completion became unknown while shrinking the live device target");
      }
      return fail(RuntimeStatus::cleanup_failure, "budget", "close_codec_pipeline",
                  "codec resources could not be released while shrinking the live device target");
    }
    if (codec_pinned_budget_active_) {
      const HostBudgetStatus released =
          backing_store_->budget().release(codec_pinned_budget_reservation_);
      if (released != HostBudgetStatus::success) {
        return fail(RuntimeStatus::cleanup_failure, "budget", "release_codec_host_budget",
                    std::string(host_budget_status_name(released)));
      }
      codec_pinned_budget_active_ = false;
    }
    suspended_codec_device_bytes_ = resources.device_slot_bytes;
    suspended_codec_pinned_bytes_ = resources.pinned_slot_bytes;
    fold_codec_epoch(resources);
    nvcomp_pipeline_.reset();
    codec_managed_device_bytes_ = 0;
    codec_pipeline_suspended_for_budget_ = true;
    consecutive_codec_restore_samples_ = 0;
    fixed_device_reserve_bytes_ = config_.workspace_reserve_bytes;
    sync_codec_telemetry();
    refresh_host_telemetry();
    return RuntimeStatus::success;
  }

  [[nodiscard]] RuntimeStatus restore_codec_pipeline_after_budget_growth() {
    if (!codec_pipeline_suspended_for_budget_ || nvcomp_pipeline_ != nullptr ||
        !nvcomp_available_ || nvcomp_api_ == nullptr || backing_codec_ == nullptr ||
        backing_store_ == nullptr || suspended_codec_device_bytes_ == 0U ||
        suspended_codec_pinned_bytes_ == 0U) {
      return RuntimeStatus::success;
    }

    HostBudgetReservation pinned_reservation;
    const HostBudgetStatus admitted = backing_store_->budget().reserve(
        HostBudgetCategory::pinned_staging, suspended_codec_pinned_bytes_, pinned_reservation);
    if (admitted != HostBudgetStatus::success) {
      ++telemetry_.gpu_codec_fallbacks;
      refresh_host_telemetry();
      return RuntimeStatus::success;
    }

    std::unique_ptr<NvcompLz4Pipeline> candidate;
    try {
      NvcompPipelineConfig pipeline_config{config_.chunk_bytes,
                                           config_.codec_slots,
                                           config_.compression_workspace_cap_bytes,
                                           config_.stall_timeout,
                                           nullptr,
                                           nullptr};
      pipeline_config.trace_observer = &Impl::compression_trace_trampoline;
      pipeline_config.trace_user_data = this;
      candidate =
          std::make_unique<NvcompLz4Pipeline>(api_, *nvcomp_api_, *backing_codec_, pipeline_config);
    } catch (const std::bad_alloc&) {
      (void)backing_store_->budget().release(pinned_reservation);
      ++telemetry_.gpu_codec_fallbacks;
      refresh_host_telemetry();
      return RuntimeStatus::success;
    } catch (...) {
      (void)backing_store_->budget().release(pinned_reservation);
      ++telemetry_.gpu_codec_fallbacks;
      refresh_host_telemetry();
      return RuntimeStatus::success;
    }

    const NvcompPipelineStatus setup_status = candidate->setup();
    const NvcompPipelineTelemetry setup_resources = candidate->telemetry();
    const bool resources_match =
        setup_status == NvcompPipelineStatus::success &&
        setup_resources.device_slot_bytes == suspended_codec_device_bytes_ &&
        setup_resources.pinned_slot_bytes == suspended_codec_pinned_bytes_;
    if (!resources_match) {
      const NvcompPipelineStatus close_status = candidate->close();
      const NvcompPipelineTelemetry retired = candidate->telemetry();
      if (close_status != NvcompPipelineStatus::success) {
        // Preserve every resource and its host-ledger charge for cleanup retry. Setup has not
        // submitted user work, so an ordinary cleanup failure is not mislabeled as an unknown
        // completion; a true pipeline quarantine still remains a worker quarantine boundary.
        nvcomp_pipeline_ = std::move(candidate);
        codec_pinned_budget_reservation_ = pinned_reservation;
        codec_pinned_budget_active_ = true;
        sync_codec_telemetry();
        if (const auto reserve =
                checked_add(config_.workspace_reserve_bytes, codec_managed_device_bytes_);
            reserve.has_value()) {
          fixed_device_reserve_bytes_ = *reserve;
        }
        launches_blocked_ = true;
        if (close_status == NvcompPipelineStatus::quarantined || nvcomp_pipeline_->quarantined()) {
          async_resources_quarantined_ = true;
          telemetry_.quarantined = true;
          return poison("budget", "restore_codec_pipeline",
                        "codec restoration entered an unknown post-submission state");
        }
        return fail(RuntimeStatus::cleanup_failure, "budget", "restore_codec_pipeline",
                    "failed codec restoration retained resources for cleanup retry");
      }
      fold_codec_epoch(retired);
      if (backing_store_->budget().release(pinned_reservation) != HostBudgetStatus::success) {
        return poison("budget", "restore_codec_host_budget",
                      "failed codec restoration retained an unknown host-budget reservation");
      }
      sync_codec_telemetry();
      ++telemetry_.gpu_codec_fallbacks;
      refresh_host_telemetry();
      return RuntimeStatus::success;
    }

    nvcomp_pipeline_ = std::move(candidate);
    codec_pinned_budget_reservation_ = pinned_reservation;
    codec_pinned_budget_active_ = true;
    codec_pipeline_suspended_for_budget_ = false;
    consecutive_codec_restore_samples_ = 0;
    sync_codec_telemetry();
    const auto restored_reserve =
        checked_add(config_.workspace_reserve_bytes, codec_managed_device_bytes_);
    if (!restored_reserve.has_value()) {
      return poison("budget", "restore_codec_accounting",
                    "restored codec device-byte accounting overflowed");
    }
    fixed_device_reserve_bytes_ = *restored_reserve;
    telemetry_.device_reserve_bytes_peak =
        std::max(telemetry_.device_reserve_bytes_peak, fixed_device_reserve_bytes_);
    refresh_host_telemetry();
    return RuntimeStatus::success;
  }

  void fold_codec_epoch(const NvcompPipelineTelemetry& source) noexcept {
    telemetry_.codec_slots_peak = std::max(telemetry_.codec_slots_peak, source.slots_created);
    retired_codec_telemetry_.workspace_peak_bytes =
        std::max(retired_codec_telemetry_.workspace_peak_bytes, source.workspace_peak_bytes);
    retired_codec_telemetry_.device_slot_peak_bytes =
        std::max(retired_codec_telemetry_.device_slot_peak_bytes, source.device_slot_peak_bytes);
    retired_codec_telemetry_.pinned_slot_peak_bytes =
        std::max(retired_codec_telemetry_.pinned_slot_peak_bytes, source.pinned_slot_peak_bytes);
    retired_codec_telemetry_.device_slot_capacity_bytes = std::max(
        retired_codec_telemetry_.device_slot_capacity_bytes, source.device_slot_capacity_bytes);
    retired_codec_telemetry_.events_recorded =
        saturating_add(retired_codec_telemetry_.events_recorded, source.events_recorded);
    retired_codec_telemetry_.events_retired =
        saturating_add(retired_codec_telemetry_.events_retired, source.events_retired);
    retired_codec_telemetry_.encode_batches =
        saturating_add(retired_codec_telemetry_.encode_batches, source.encode_batches);
    retired_codec_telemetry_.decode_batches =
        saturating_add(retired_codec_telemetry_.decode_batches, source.decode_batches);
    retired_codec_telemetry_.verification_failures = saturating_add(
        retired_codec_telemetry_.verification_failures, source.verification_failures);
    retired_codec_telemetry_.slots_created =
        saturating_add(retired_codec_telemetry_.slots_created, source.slots_created);
    retired_codec_telemetry_.slots_reused =
        saturating_add(retired_codec_telemetry_.slots_reused, source.slots_reused);
    retired_codec_telemetry_.verification_nanoseconds = saturating_add(
        retired_codec_telemetry_.verification_nanoseconds, source.verification_nanoseconds);
    retired_codec_telemetry_.codec_failures =
        saturating_add(retired_codec_telemetry_.codec_failures, source.codec_failures);
  }

  void sync_codec_telemetry() noexcept {
    const NvcompPipelineTelemetry empty{};
    const NvcompPipelineTelemetry& source =
        nvcomp_pipeline_ != nullptr ? nvcomp_pipeline_->telemetry() : empty;
    telemetry_.codec_workspace_bytes = nvcomp_pipeline_ != nullptr ? source.workspace_bytes : 0U;
    telemetry_.codec_workspace_peak_bytes =
        std::max({telemetry_.codec_workspace_peak_bytes,
                  retired_codec_telemetry_.workspace_peak_bytes, source.workspace_peak_bytes});
    telemetry_.codec_slot_bytes =
        nvcomp_pipeline_ != nullptr && source.device_slot_bytes >= source.workspace_bytes
            ? source.device_slot_bytes - source.workspace_bytes
            : 0U;
    telemetry_.codec_slot_peak_bytes =
        std::max(telemetry_.codec_slot_peak_bytes,
                 std::max(retired_codec_telemetry_.device_slot_peak_bytes >=
                                  retired_codec_telemetry_.workspace_peak_bytes
                              ? retired_codec_telemetry_.device_slot_peak_bytes -
                                    retired_codec_telemetry_.workspace_peak_bytes
                              : 0U,
                          source.device_slot_peak_bytes >= source.workspace_peak_bytes
                              ? source.device_slot_peak_bytes - source.workspace_peak_bytes
                              : 0U));
    telemetry_.codec_slot_capacity_bytes = std::max(
        {telemetry_.codec_slot_capacity_bytes, retired_codec_telemetry_.device_slot_capacity_bytes,
         source.device_slot_capacity_bytes});
    telemetry_.codec_events_recorded =
        saturating_add(retired_codec_telemetry_.events_recorded, source.events_recorded);
    telemetry_.codec_events_retired =
        saturating_add(retired_codec_telemetry_.events_retired, source.events_retired);
    telemetry_.gpu_encode_operations =
        saturating_add(retired_codec_telemetry_.encode_batches, source.encode_batches);
    telemetry_.gpu_decode_operations =
        saturating_add(retired_codec_telemetry_.decode_batches, source.decode_batches);
    telemetry_.codec_verification_failures = saturating_add(
        retired_codec_telemetry_.verification_failures, source.verification_failures);
    telemetry_.codec_slots_created =
        saturating_add(retired_codec_telemetry_.slots_created, source.slots_created);
    telemetry_.codec_slots_reused =
        saturating_add(retired_codec_telemetry_.slots_reused, source.slots_reused);
    telemetry_.codec_slots_peak = std::max(telemetry_.codec_slots_peak, source.slots_created);
    telemetry_.verification_nanoseconds = saturating_add(
        retired_codec_telemetry_.verification_nanoseconds, source.verification_nanoseconds);
    codec_managed_device_bytes_ = nvcomp_pipeline_ != nullptr ? source.device_slot_bytes : 0U;
    // NvcompPipelineTelemetry::device_slot_peak_bytes includes both non-workspace slot
    // allocations and the per-slot workspace.  Charge that exact historical peak together with
    // the runtime compute reserve even if setup later rolls back to the CPU codec.
    if (const auto reserve_peak =
            checked_add(config_.workspace_reserve_bytes, source.device_slot_peak_bytes);
        reserve_peak.has_value()) {
      telemetry_.device_reserve_bytes_peak =
          std::max(telemetry_.device_reserve_bytes_peak, *reserve_peak);
    } else {
      telemetry_.device_reserve_bytes_peak = std::numeric_limits<std::uint64_t>::max();
      ++telemetry_.device_budget_violation_count;
    }
    telemetry_.gpu_codec_fallbacks =
        std::max(telemetry_.gpu_codec_fallbacks,
                 saturating_add(retired_codec_telemetry_.codec_failures, source.codec_failures));
  }

  void sync_cpu_codec_telemetry() noexcept {
    if (cpu_codec_pool_ == nullptr) {
      return;
    }
    const std::uint64_t submitted = cpu_codec_pool_->telemetry().encode_submitted;
    const std::uint64_t delta = submitted >= observed_cpu_codec_blocks_submitted_
                                    ? submitted - observed_cpu_codec_blocks_submitted_
                                    : submitted;
    telemetry_.cpu_codec_blocks_submitted =
        saturating_add(telemetry_.cpu_codec_blocks_submitted, delta);
    observed_cpu_codec_blocks_submitted_ = submitted;
  }

  [[nodiscard]] static RuntimeStatus backing_runtime_status(const BackingStatus status) noexcept {
    switch (status) {
    case BackingStatus::success:
      return RuntimeStatus::success;
    case BackingStatus::invalid_configuration:
    case BackingStatus::invalid_argument:
    case BackingStatus::duplicate_chunk:
    case BackingStatus::chunk_not_found:
    case BackingStatus::invalid_state:
    case BackingStatus::stale_generation:
    case BackingStatus::generation_overflow:
    case BackingStatus::range_overflow:
    case BackingStatus::range_out_of_bounds:
      return RuntimeStatus::invalid_argument;
    case BackingStatus::host_budget_exceeded:
    case BackingStatus::allocation_failure:
      return RuntimeStatus::host_oom;
    case BackingStatus::codec_failure:
    case BackingStatus::corrupt_data:
      return RuntimeStatus::poisoned;
    case BackingStatus::not_beneficial:
      return RuntimeStatus::success;
    case BackingStatus::internal_failure:
      return RuntimeStatus::internal_failure;
    }
    return RuntimeStatus::internal_failure;
  }

  void refresh_host_telemetry() noexcept {
    if (backing_store_ == nullptr) {
      telemetry_.host_stored_bytes = 0;
      telemetry_.host_raw_bytes = 0;
      telemetry_.host_compressed_bytes = 0;
      telemetry_.host_implicit_zero_bytes = 0;
      telemetry_.host_invalid_bytes = 0;
      telemetry_.host_invalid_chunks = 0;
      telemetry_.host_implicit_zero_chunks = 0;
      telemetry_.host_raw_chunks = 0;
      telemetry_.host_lz4_chunks = 0;
      telemetry_.host_authoritative_bytes = 0;
      telemetry_.host_budget_bytes = 0;
      return;
    }
    std::uint64_t stored = 0;
    std::uint64_t raw = 0;
    std::uint64_t compressed = 0;
    std::uint64_t implicit_zero = 0;
    std::uint64_t invalid = 0;
    std::uint64_t invalid_chunks = 0;
    std::uint64_t implicit_zero_chunks = 0;
    std::uint64_t raw_chunks = 0;
    std::uint64_t lz4_chunks = 0;
    for (const auto& [id, owner] : allocations_) {
      (void)id;
      for (const ChunkRecord& record : owner->chunks) {
        const BackingResult info = backing_store_->inspect(record.key);
        if (!info) {
          continue;
        }
        stored += info.chunk.stored_payload_bytes;
        switch (info.chunk.representation) {
        case BackingRepresentation::raw:
          raw += info.chunk.stored_payload_bytes;
          ++raw_chunks;
          break;
        case BackingRepresentation::lz4_blocks:
          compressed += info.chunk.stored_payload_bytes;
          ++lz4_chunks;
          break;
        case BackingRepresentation::implicit_zero:
          implicit_zero += info.chunk.valid_bytes;
          ++implicit_zero_chunks;
          break;
        case BackingRepresentation::invalid:
          invalid += info.chunk.valid_bytes;
          ++invalid_chunks;
          break;
        }
      }
    }
    telemetry_.host_stored_bytes = stored;
    telemetry_.host_stored_peak_bytes = std::max(telemetry_.host_stored_peak_bytes, stored);
    telemetry_.host_raw_bytes = raw;
    telemetry_.host_raw_peak_bytes = std::max(telemetry_.host_raw_peak_bytes, raw);
    telemetry_.host_compressed_bytes = compressed;
    telemetry_.host_compressed_peak_bytes =
        std::max(telemetry_.host_compressed_peak_bytes, compressed);
    telemetry_.host_implicit_zero_bytes = implicit_zero;
    telemetry_.host_invalid_bytes = invalid;
    telemetry_.host_invalid_chunks = invalid_chunks;
    telemetry_.host_implicit_zero_chunks = implicit_zero_chunks;
    telemetry_.host_raw_chunks = raw_chunks;
    telemetry_.host_lz4_chunks = lz4_chunks;
    const HostBudgetSnapshot budget = backing_store_->budget().snapshot();
    telemetry_.host_store_cap_bytes = budget.limit_bytes;
    telemetry_.host_authoritative_bytes = budget.authoritative_bytes;
    telemetry_.host_authoritative_peak_bytes =
        std::max(telemetry_.host_authoritative_peak_bytes, budget.authoritative_peak_bytes);
    telemetry_.host_budget_bytes = budget.total_bytes;
    telemetry_.host_budget_peak_bytes =
        std::max(telemetry_.host_budget_peak_bytes, budget.peak_bytes);
    telemetry_.conversion_scratch_peak_bytes =
        std::max(telemetry_.conversion_scratch_peak_bytes, budget.conversion_scratch_peak_bytes);
    telemetry_.spill_reserved_peak_bytes =
        std::max(telemetry_.spill_reserved_peak_bytes, budget.spill_peak_bytes);
  }

  [[nodiscard]] RuntimeStatus reserve_spill(const ChunkKey key,
                                            const std::span<const ChunkAccessPlan> working_set) {
    if (spill_reservations_.contains(key)) {
      return RuntimeStatus::success;
    }
    const BackingResult info = backing_store_->inspect(key);
    if (!info) {
      return fail(RuntimeStatus::invalid_argument, "compression", "reserve_spill",
                  "write-capable chunk has no backing generation");
    }
    const std::optional<std::uint64_t> maximum_charge =
        maximum_raw_backing_charge(info.chunk.valid_bytes);
    if (!maximum_charge.has_value()) {
      return fail(RuntimeStatus::invalid_argument, "compression", "reserve_spill",
                  "raw spill size overflowed");
    }
    RawSpill spill;
    HostBudgetStatus budget_status = backing_store_->budget().reserve(
        HostBudgetCategory::spill, *maximum_charge, spill.reservation);
    while (budget_status == HostBudgetStatus::limit_exceeded) {
      // Retired dirty frames can retain spill credits until VRAM eviction even when the host
      // budget is the tighter bound. Commit one existing spill at a time before admitting a new
      // write. Protect the entire upcoming working set, including chunks not yet pinned by the
      // acquire loop. A successful writeback removes its spill entry, making this loop bounded.
      std::optional<ChunkKey> victim;
      for (const auto& [candidate, reserved] : spill_reservations_) {
        (void)reserved;
        if (std::any_of(working_set.begin(), working_set.end(),
                        [&](const ChunkAccessPlan& access) { return access.key == candidate; })) {
          continue;
        }
        const ChunkRecord* record = chunk(candidate);
        const Frame* frame = frame_for(candidate);
        if (record == nullptr || record->state != ChunkState::resident_dirty ||
            !is_victim_eligible(*record) || record->event_generation == 0 ||
            record->completed_generation != record->event_generation || frame == nullptr ||
            !frame->mapped || frame->quarantined) {
          continue;
        }
        if (!victim.has_value() || candidate < *victim) {
          victim = candidate;
        }
      }
      if (!victim.has_value()) {
        break;
      }
      if (const RuntimeStatus status = writeback(*victim); status != RuntimeStatus::success) {
        return status;
      }
      notify_lifecycle_progress();
      // A compressed/zero authority replaced by raw may release much less than one chunk;
      // always consult the exact ledger again rather than assuming a full spill was reclaimed.
      budget_status = backing_store_->budget().reserve(HostBudgetCategory::spill, *maximum_charge,
                                                       spill.reservation);
    }
    if (budget_status != HostBudgetStatus::success) {
      return fail(RuntimeStatus::host_oom, "compression", "reserve_spill",
                  "raw spill admission failed before the write-capable lease");
    }
    const BackingResult prepared =
        backing_store_->prepare_raw_replacement(key, info.chunk.generation, spill.backing);
    if (!prepared) {
      (void)backing_store_->budget().release(spill.reservation);
      return fail(backing_runtime_status(prepared.status), "compression", "prepare_raw_spill",
                  std::string(backing_status_name(prepared.status)));
    }
    try {
      spill_reservations_.emplace(key, std::move(spill));
    } catch (const std::bad_alloc&) {
      (void)backing_store_->budget().release(spill.reservation);
      return fail(RuntimeStatus::host_oom, "compression", "reserve_spill",
                  "raw spill allocation failed before launch");
    } catch (...) {
      (void)backing_store_->budget().release(spill.reservation);
      return fail(RuntimeStatus::internal_failure, "compression", "reserve_spill",
                  "spill reservation metadata failed unexpectedly");
    }
    const HostBudgetSnapshot snapshot = backing_store_->budget().snapshot();
    telemetry_.spill_reserved_bytes = snapshot.spill_bytes;
    telemetry_.spill_reserved_peak_bytes =
        std::max(telemetry_.spill_reserved_peak_bytes, snapshot.spill_bytes);
    return RuntimeStatus::success;
  }

  void release_spill(const ChunkKey key) noexcept {
    const auto found = spill_reservations_.find(key);
    if (found == spill_reservations_.end()) {
      return;
    }
    (void)backing_store_->budget().release(found->second.reservation);
    spill_reservations_.erase(found);
    telemetry_.spill_reserved_bytes = backing_store_->budget().snapshot().spill_bytes;
  }

  [[nodiscard]] BackingStatus prepare_cpu_compression(const ChunkKey key,
                                                      const HostChunkInfo& source,
                                                      CpuCompressionCandidate& output) {
    output = {};
    if (cpu_codec_pool_ == nullptr || !cpu_codec_pool_->valid() || source.valid_bytes == 0 ||
        source.generation == std::numeric_limits<std::uint64_t>::max()) {
      return BackingStatus::codec_failure;
    }
    const std::optional<std::uint64_t> candidate_charge =
        maximum_raw_backing_charge(source.valid_bytes);
    const std::uint64_t block_count =
        (source.valid_bytes + compression_block_bytes - 1U) / compression_block_bytes;
    const std::uint64_t batch_count =
        std::min<std::uint64_t>(block_count, cpu_codec_queue_capacity_);
    const auto queued_input_bytes = checked_multiply(batch_count, compression_block_bytes);
    const auto maximum_encoded_block =
        backing_codec_->maximum_compressed_bytes(static_cast<std::size_t>(compression_block_bytes));
    const auto queued_output_bytes =
        maximum_encoded_block.has_value()
            ? checked_multiply(batch_count, static_cast<std::uint64_t>(*maximum_encoded_block))
            : std::nullopt;
    // Each outstanding job owns its raw input and may simultaneously hold a compressBound-sized
    // output while it is repacked into the exact-sized candidate allocation. A single contiguous
    // snapshot avoids 1024 separately verified backing reads for a 64-MiB chunk. The candidate
    // itself is covered by candidate_charge; every transient side is covered by scratch credit.
    const auto queued_input_and_output =
        queued_input_bytes.has_value() && queued_output_bytes.has_value()
            ? checked_add(*queued_input_bytes, *queued_output_bytes)
            : std::nullopt;
    const auto block_scratch_bytes =
        queued_input_and_output.has_value()
            ? checked_add(*queued_input_and_output, compression_block_bytes)
            : std::nullopt;
    const auto scratch_bytes = block_scratch_bytes.has_value()
                                   ? checked_add(*block_scratch_bytes, source.valid_bytes)
                                   : std::nullopt;
    if (!candidate_charge.has_value() || !scratch_bytes.has_value() ||
        block_count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
      return BackingStatus::range_overflow;
    }

    HostBudgetReservation candidate_reservation;
    HostBudgetReservation scratch_reservation;
    const auto release_reservations = [&]() noexcept {
      (void)backing_store_->budget().release(candidate_reservation);
      (void)backing_store_->budget().release(scratch_reservation);
    };
    HostBudgetStatus budget = backing_store_->budget().reserve(
        HostBudgetCategory::conversion_scratch, *candidate_charge, candidate_reservation);
    if (budget == HostBudgetStatus::success) {
      budget = backing_store_->budget().reserve(HostBudgetCategory::conversion_scratch,
                                                *scratch_bytes, scratch_reservation);
    }
    if (budget != HostBudgetStatus::success) {
      release_reservations();
      return budget == HostBudgetStatus::limit_exceeded ? BackingStatus::host_budget_exceeded
                                                        : BackingStatus::allocation_failure;
    }

    struct PendingBlock {
      CpuCodecTicket ticket;
      std::uint64_t index = 0;
      std::uint32_t bytes = 0;
    };
    try {
      std::vector<std::byte> source_snapshot(static_cast<std::size_t>(source.valid_bytes));
      const BackingResult snapshot = backing_store_->read(key, 0, source_snapshot);
      if (!snapshot) {
        release_reservations();
        return snapshot.status;
      }
      Lz4BlocksV1 candidate;
      candidate.valid_bytes = source.valid_bytes;
      candidate.generation = source.generation + 1U;
      candidate.content_token = source.content_token;
      candidate.blocks.resize(static_cast<std::size_t>(block_count));
      std::vector<PendingBlock> pending;
      pending.reserve(static_cast<std::size_t>(batch_count));

      for (std::uint64_t first = 0; first < block_count; first += batch_count) {
        pending.clear();
        const std::uint64_t end = std::min(block_count, first + batch_count);
        for (std::uint64_t index = first; index < end; ++index) {
          const std::uint64_t offset = index * compression_block_bytes;
          const std::uint32_t bytes = static_cast<std::uint32_t>(
              std::min(compression_block_bytes, source.valid_bytes - offset));
          Lz4BlockV1& block = candidate.blocks[static_cast<std::size_t>(index)];
          block.uncompressed_bytes = bytes;

          // The candidate's immediately preceding block is a bounded one-entry encoded-result
          // cache. Compare against the charged source snapshot before submitting work, then copy
          // the exact-sized payload after its representative retires. This works across batch
          // boundaries and avoids retaining an uncharged dictionary of arbitrary unique blocks.
          const bool repeats_previous =
              index != 0U &&
              candidate.blocks[static_cast<std::size_t>(index - 1U)].uncompressed_bytes == bytes &&
              std::memcmp(source_snapshot.data() + static_cast<std::size_t>(offset),
                          source_snapshot.data() +
                              static_cast<std::size_t>(offset - compression_block_bytes),
                          bytes) == 0;
          if (repeats_previous) {
            continue;
          }
          std::vector<std::byte> input(static_cast<std::size_t>(bytes));
          std::memcpy(input.data(), source_snapshot.data() + static_cast<std::size_t>(offset),
                      bytes);
          CpuCodecTicket ticket;
          const CpuCodecPoolStatus submitted =
              cpu_codec_pool_->submit_encode(key, source.generation, std::move(input), ticket);
          if (submitted != CpuCodecPoolStatus::success) {
            for (const PendingBlock& accepted : pending) {
              CpuCodecResult ignored;
              const CpuCodecPoolStatus retired =
                  cpu_codec_pool_->wait(accepted.ticket, config_.stall_timeout, &ignored);
              if (retired == CpuCodecPoolStatus::timeout ||
                  retired == CpuCodecPoolStatus::not_ready) {
                cpu_codec_timeout_ = true;
              }
            }
            if (!cpu_codec_timeout_) {
              release_reservations();
            }
            return submitted == CpuCodecPoolStatus::allocation_failure
                       ? BackingStatus::allocation_failure
                       : BackingStatus::codec_failure;
          }
          if (output.operation_id == 0U) {
            output.operation_id = ticket.operation_id;
          }
          pending.push_back(PendingBlock{ticket, index, bytes});
        }

        BackingStatus batch_status = BackingStatus::success;
        for (const PendingBlock& accepted : pending) {
          CpuCodecResult result;
          const CpuCodecPoolStatus completed =
              cpu_codec_pool_->wait(accepted.ticket, config_.stall_timeout, &result);
          if (completed == CpuCodecPoolStatus::timeout ||
              completed == CpuCodecPoolStatus::not_ready) {
            cpu_codec_timeout_ = true;
          }
          if (completed != CpuCodecPoolStatus::success ||
              result.ticket.source_generation != source.generation) {
            batch_status = completed == CpuCodecPoolStatus::allocation_failure
                               ? BackingStatus::allocation_failure
                               : BackingStatus::codec_failure;
            continue;
          }
          Lz4BlockV1& block = candidate.blocks[static_cast<std::size_t>(accepted.index)];
          if (!result.encoded_as_raw && !result.output.empty() &&
              result.output.size() < accepted.bytes) {
            block.storage = BlockStorage::lz4;
            block.payload = std::move(result.output);
          } else if (result.encoded_as_raw && result.output.size() == accepted.bytes) {
            block.storage = BlockStorage::raw;
            block.payload = std::move(result.output);
          } else {
            batch_status = BackingStatus::codec_failure;
            continue;
          }
          if (block.payload.size() >
              std::numeric_limits<std::uint64_t>::max() - output.stored_payload_bytes) {
            batch_status = BackingStatus::range_overflow;
            continue;
          }
          output.stored_payload_bytes += static_cast<std::uint64_t>(block.payload.size());
        }
        if (batch_status == BackingStatus::success) {
          for (std::uint64_t index = first; index < end; ++index) {
            Lz4BlockV1& block = candidate.blocks[static_cast<std::size_t>(index)];
            if (!block.payload.empty()) {
              continue;
            }
            if (index == 0U) {
              batch_status = BackingStatus::codec_failure;
              break;
            }
            const Lz4BlockV1& previous = candidate.blocks[static_cast<std::size_t>(index - 1U)];
            if (previous.payload.empty() ||
                previous.uncompressed_bytes != block.uncompressed_bytes) {
              batch_status = BackingStatus::codec_failure;
              break;
            }
            block.storage = previous.storage;
            block.payload = previous.payload;
            if (block.payload.size() >
                std::numeric_limits<std::uint64_t>::max() - output.stored_payload_bytes) {
              batch_status = BackingStatus::range_overflow;
              break;
            }
            output.stored_payload_bytes += static_cast<std::uint64_t>(block.payload.size());
          }
        }
        if (batch_status != BackingStatus::success) {
          if (!cpu_codec_timeout_) {
            release_reservations();
          }
          return batch_status;
        }
      }
      // Every block is now present in a complete immutable candidate generation. From this point
      // onward the caller must either commit it, or this helper/caller must record a safe discard.
      ++telemetry_.generations_created;
      if (backing_store_->budget().release(scratch_reservation) != HostBudgetStatus::success) {
        if (backing_store_->budget().release(candidate_reservation) == HostBudgetStatus::success) {
          ++telemetry_.generations_discarded;
          emit_generation_discard(key, output.operation_id, source, candidate.generation,
                                  std::nullopt, CompressionPath::cpu_lz4_gpu_decode,
                                  output.stored_payload_bytes,
                                  "cpu_candidate_scratch_release_failed");
        }
        return BackingStatus::internal_failure;
      }
      if (output.stored_payload_bytes >= source.valid_bytes) {
        const std::uint64_t operation_id = output.operation_id;
        const std::uint64_t physical_bytes = output.stored_payload_bytes;
        const HostBudgetStatus released = backing_store_->budget().release(candidate_reservation);
        output = {};
        if (released == HostBudgetStatus::success) {
          ++telemetry_.generations_discarded;
          emit_generation_discard(key, operation_id, source, candidate.generation, std::nullopt,
                                  CompressionPath::cpu_lz4_gpu_decode, physical_bytes,
                                  "cpu_candidate_expansion_rejected");
        }
        return released == HostBudgetStatus::success ? BackingStatus::not_beneficial
                                                     : BackingStatus::internal_failure;
      }
      output.container = std::move(candidate);
      output.reservation = candidate_reservation;
      candidate_reservation = {};
      return BackingStatus::success;
    } catch (const std::bad_alloc&) {
      release_reservations();
      output = {};
      return BackingStatus::allocation_failure;
    } catch (...) {
      release_reservations();
      output = {};
      return BackingStatus::internal_failure;
    }
  }

  [[nodiscard]] RuntimeStatus consider_compression(const ChunkKey key) {
    if (config_.compression_mode == CompressionMode::disabled ||
        config_.forced_compression_path == CompressionPath::raw ||
        config_.forced_compression_path == CompressionPath::nvcomp_gpu_codec) {
      ++telemetry_.raw_path_decisions;
      return RuntimeStatus::success;
    }
    BackingResult info = backing_store_->inspect(key);
    if (!info || info.chunk.representation == BackingRepresentation::invalid) {
      return fail(RuntimeStatus::invalid_argument, "compression", "consider_compression",
                  "compression candidate has no host authority");
    }
    if (info.chunk.representation == BackingRepresentation::implicit_zero ||
        info.chunk.representation == BackingRepresentation::lz4_blocks) {
      return RuntimeStatus::success;
    }

    CompressionCostModel& model = cost_models_.try_emplace(key).first->second;
    const std::uint64_t content_generation = content_identity(info.chunk.content_token);
    if (model.generation() != content_generation) {
      model.reset(content_generation);
    }
    const double raw_transfer_us =
        raw_h2d_us_per_byte_ * static_cast<double>(info.chunk.valid_bytes);
    const auto reuse_found = chunk_reuse_counts_.find(key);
    const std::uint64_t reuse_count =
        reuse_found == chunk_reuse_counts_.end() ? 0U : reuse_found->second;
    const CpuCodecPoolTelemetry pool_telemetry = cpu_codec_pool_->telemetry();
    // Historical peak_outstanding reflects our deliberately bounded batch size, not current CPU
    // scarcity. Using it here permanently multiplied an already end-to-end encode sample by the
    // queue depth (64/2 by default), making adaptive compression impossible after one normal
    // batch. Charge only live competing work; the measured encode duration already includes this
    // candidate's own queueing.
    const std::uint64_t live_cpu_work =
        saturating_add(pool_telemetry.queued, pool_telemetry.running);
    const double cpu_availability =
        live_cpu_work <= config_.codec_workers
            ? 1.0
            : static_cast<double>(config_.codec_workers) / static_cast<double>(live_cpu_work);
    (void)model.observe(content_generation,
                        CostObservation{CompressionPath::raw, info.chunk.valid_bytes,
                                        info.chunk.valid_bytes, 0.0, 0.0, raw_transfer_us, 0.0, 0.0,
                                        0.0, 0.0, 1.0, reuse_count, false});

    const bool force_cpu = config_.forced_compression_path == CompressionPath::cpu_lz4_gpu_decode;
    const std::uint32_t attempts =
        config_.compression_mode == CompressionMode::capacity || force_cpu ? 1U : 3U;
    for (std::uint32_t attempt = 0; attempt < attempts; ++attempt) {
      info = backing_store_->inspect(key);
      if (!info) {
        return poison("compression", "inspect", "compression candidate disappeared");
      }
      // Two calibration samples are required for every selectable path. Raw transfer is cheap to
      // observe and remains history-bearing across immutable content generations.
      (void)model.observe(content_generation,
                          CostObservation{CompressionPath::raw, info.chunk.valid_bytes,
                                          info.chunk.valid_bytes, 0.0, 0.0, raw_transfer_us, 0.0,
                                          0.0, 0.0, 0.0, 1.0, reuse_count, false});
      ++telemetry_.compression_attempts;
      ++telemetry_.cpu_encode_operations;
      ++telemetry_.codec_calibration_samples;
      const auto started = Clock::now();
      CpuCompressionCandidate candidate;
      const BackingStatus encoded = prepare_cpu_compression(key, info.chunk, candidate);
      sync_cpu_codec_telemetry();
      const auto elapsed =
          std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - started);
      telemetry_.cpu_encode_nanoseconds += static_cast<std::uint64_t>(elapsed.count());
      const double encode_us =
          static_cast<double>(
              compression_cost_duration(CompressionPath::cpu_lz4_gpu_decode, elapsed).count()) /
          1000.0;
      if (encoded != BackingStatus::success) {
        if (cpu_codec_timeout_) {
          launches_blocked_ = true;
          return fail(RuntimeStatus::timeout, "compression", "cpu_codec_wait",
                      "CPU codec operation exceeded the bounded retirement timeout");
        }
        if (encoded == BackingStatus::not_beneficial) {
          // prepare_cpu_compression completed an immutable candidate generation before proving
          // that its payload did not reduce storage. The helper released its reservation and
          // destroyed the candidate locally, so this is a complete, safely discarded lifecycle.
          ++telemetry_.expansion_rejections;
          ++telemetry_.raw_fallbacks;
          const double decode_us =
              (gpu_decode_us_per_byte_ > 0.0 ? gpu_decode_us_per_byte_ : cpu_decode_us_per_byte_) *
              static_cast<double>(info.chunk.valid_bytes);
          (void)model.observe(content_generation,
                              CostObservation{CompressionPath::cpu_lz4_gpu_decode,
                                              info.chunk.valid_bytes, info.chunk.valid_bytes,
                                              encode_us, decode_us, raw_transfer_us, 0.0, 0.0, 0.0,
                                              0.0, cpu_availability, reuse_count, false});
          const auto probe = forecast_clean_reuse_probe(raw_transfer_us, encode_us, decode_us,
                                                        raw_transfer_us, reuse_count);
          if (probe.has_value()) {
            (void)model.record_probe(content_generation, CompressionPath::cpu_lz4_gpu_decode,
                                     probe->raw_total_us, probe->candidate_total_us);
          }
          continue;
        }
        ++telemetry_.cpu_codec_fallbacks;
        if (!force_cpu && (encoded == BackingStatus::host_budget_exceeded ||
                           encoded == BackingStatus::allocation_failure ||
                           encoded == BackingStatus::codec_failure)) {
          ++telemetry_.raw_fallbacks;
          ++telemetry_.raw_path_decisions;
          refresh_host_telemetry();
          return RuntimeStatus::success;
        }
        return fail(backing_runtime_status(encoded), "compression", "encode_lz4",
                    std::string(backing_status_name(encoded)));
      }
      // The successful prepare already counted this complete immutable generation. Resolve it
      // exactly once below as either the new authority or a safely discarded policy candidate.
      const double compressed_transfer_us =
          raw_h2d_us_per_byte_ * static_cast<double>(candidate.stored_payload_bytes);
      const double decode_us =
          (gpu_decode_us_per_byte_ > 0.0 ? gpu_decode_us_per_byte_ : cpu_decode_us_per_byte_) *
          static_cast<double>(candidate.container.valid_bytes);
      (void)model.observe(content_generation,
                          CostObservation{CompressionPath::cpu_lz4_gpu_decode,
                                          candidate.container.valid_bytes,
                                          candidate.stored_payload_bytes, encode_us, decode_us,
                                          compressed_transfer_us, 0.0, 0.0, 0.0, 0.0,
                                          cpu_availability, reuse_count, false});
      const auto probe = forecast_clean_reuse_probe(raw_transfer_us, encode_us, decode_us,
                                                    compressed_transfer_us, reuse_count);
      const bool favorable =
          probe.has_value() &&
          model.record_probe(content_generation, CompressionPath::cpu_lz4_gpu_decode,
                             probe->raw_total_us, probe->candidate_total_us);
      const CostDecision decision = model.decide(
          config_.compression_mode, candidate.container.valid_bytes, true, nvcomp_available_);
      const bool commit_candidate =
          config_.compression_mode == CompressionMode::capacity || force_cpu ||
          (favorable && !decision.calibration_required && decision.path != CompressionPath::raw);
      if (commit_candidate) {
        const std::uint64_t candidate_operation_id = candidate.operation_id;
        const std::uint64_t candidate_generation = candidate.container.generation;
        const std::uint64_t candidate_physical_bytes = candidate.stored_payload_bytes;
        const BackingResult committed = backing_store_->commit_lz4_blocks(
            key, info.chunk.generation, std::move(candidate.container), &candidate.reservation);
        if (!committed) {
          const HostBudgetStatus released = backing_store_->budget().release(candidate.reservation);
          if (released != HostBudgetStatus::success) {
            return poison("compression", "discard_candidate",
                          "failed CPU codec commit retained an unknown host-budget reservation");
          }
          ++telemetry_.generations_discarded;
          emit_generation_discard(key, candidate_operation_id, info.chunk, candidate_generation,
                                  std::nullopt, CompressionPath::cpu_lz4_gpu_decode,
                                  candidate_physical_bytes, "cpu_candidate_commit_rejected");
          return fail(backing_runtime_status(committed.status), "compression", "commit_lz4",
                      std::string(backing_status_name(committed.status)));
        }
        ++telemetry_.compression_commits;
        ++telemetry_.generations_committed;
        ++telemetry_.cpu_lz4_gpu_decode_decisions;
        set_backing_path(key, CompressionPath::cpu_lz4_gpu_decode);
        refresh_host_telemetry();
        return RuntimeStatus::success;
      }
      if (backing_store_->budget().release(candidate.reservation) != HostBudgetStatus::success) {
        return poison("compression", "discard_candidate",
                      "rejected CPU codec candidate retained an unknown host-budget reservation");
      }
      ++telemetry_.generations_discarded;
      emit_generation_discard(key, candidate.operation_id, info.chunk,
                              candidate.container.generation, std::nullopt,
                              CompressionPath::cpu_lz4_gpu_decode, candidate.stored_payload_bytes,
                              "cpu_candidate_policy_rejected");
    }

    if (model.never_compress()) {
      ++telemetry_.never_compress_decisions;
    }
    ++telemetry_.raw_path_decisions;
    refresh_host_telemetry();
    return RuntimeStatus::success;
  }

  [[nodiscard]] bool have_required_symbols() const noexcept {
    return api_.init_ != nullptr && api_.device_get_ != nullptr &&
           api_.context_get_current_ != nullptr && api_.context_get_device_ != nullptr &&
           api_.context_create_ != nullptr && api_.context_destroy_ != nullptr &&
           api_.context_push_current_ != nullptr && api_.context_pop_current_ != nullptr &&
           api_.mem_get_info_ != nullptr && api_.mem_get_allocation_granularity_ != nullptr &&
           api_.mem_address_reserve_ != nullptr && api_.mem_address_free_ != nullptr &&
           api_.mem_create_ != nullptr && api_.mem_release_ != nullptr &&
           api_.mem_map_ != nullptr && api_.mem_unmap_ != nullptr &&
           api_.mem_set_access_ != nullptr && api_.mem_get_access_ != nullptr &&
           api_.mem_alloc_ != nullptr && api_.mem_free_ != nullptr &&
           api_.mem_host_alloc_ != nullptr && api_.mem_free_host_ != nullptr &&
           api_.memcpy_h2d_async_ != nullptr && api_.memcpy_d2h_async_ != nullptr &&
           api_.stream_create_ != nullptr && api_.stream_destroy_ != nullptr &&
           api_.stream_get_context_ != nullptr && api_.stream_wait_event_ != nullptr &&
           api_.stream_is_capturing_ != nullptr && api_.event_create_ != nullptr &&
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
    if (status == RuntimeStatus::poisoned) {
      // Authoritative-data corruption is a terminal logical failure even when no CUDA work is
      // in flight. Keep the distinct async quarantine flag clear so safe cleanup can still run.
      poisoned_ = true;
    }
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

  [[nodiscard]] bool active_lease_references(const AllocationId id) const noexcept {
    if (!active_external_lease_.has_value()) {
      return false;
    }
    return std::any_of(active_external_lease_->chunks.begin(), active_external_lease_->chunks.end(),
                       [&](const LeaseChunk& use) { return use.key.allocation_id == id; });
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

  [[nodiscard]] bool
  plan_has_valid_host_sources(const std::span<const ChunkAccessPlan> chunks) const noexcept {
    for (const ChunkAccessPlan& plan : chunks) {
      const Allocation* owner = allocation(plan.key.allocation_id);
      if (owner == nullptr || plan.key.chunk_index >= owner->chunks.size()) {
        return false;
      }
      const ChunkRecord& record = owner->chunks[static_cast<std::size_t>(plan.key.chunk_index)];
      const bool resident =
          record.state == ChunkState::resident_clean || record.state == ChunkState::resident_dirty;
      const BackingResult info = backing_store_->inspect(plan.key);
      if (!info || (plan.requires_h2d && !resident &&
                    info.chunk.representation == BackingRepresentation::invalid)) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool host_range_is_valid(const Allocation& owner, const std::uint64_t offset,
                                         const std::uint64_t bytes) const noexcept {
    const std::uint64_t first = offset / config_.chunk_bytes;
    const std::uint64_t last = (offset + bytes - 1U) / config_.chunk_bytes;
    for (std::uint64_t index = first; index <= last; ++index) {
      const BackingResult info = backing_store_->inspect(ChunkKey{owner.id, index});
      if (!info || info.chunk.representation == BackingRepresentation::invalid) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool host_write_covers_invalid_chunks(const Allocation& owner,
                                                      const std::uint64_t offset,
                                                      const std::uint64_t bytes) const noexcept {
    const std::uint64_t end = offset + bytes;
    const std::uint64_t first = offset / config_.chunk_bytes;
    const std::uint64_t last = (end - 1U) / config_.chunk_bytes;
    for (std::uint64_t index = first; index <= last; ++index) {
      const BackingResult info = backing_store_->inspect(ChunkKey{owner.id, index});
      if (!info) {
        return false;
      }
      if (info.chunk.representation != BackingRepresentation::invalid) {
        continue;
      }
      const std::uint64_t chunk_begin = index * config_.chunk_bytes;
      const std::uint64_t chunk_end =
          chunk_begin + std::min(config_.chunk_bytes, owner.logical_bytes - chunk_begin);
      if (offset > chunk_begin || end < chunk_end) {
        return false;
      }
    }
    return true;
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
    const auto managed_compute = checked_add(telemetry_.resident_bytes, workspace_bytes_);
    const auto managed_with_workspace =
        managed_compute.has_value() ? checked_add(*managed_compute, codec_managed_device_bytes_)
                                    : std::nullopt;
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
    telemetry_.safe_device_budget_minimum_bytes =
        telemetry_.safe_device_budget_minimum_bytes == 0U
            ? observed_target
            : std::min(telemetry_.safe_device_budget_minimum_bytes, observed_target);
    const auto required_frame_bytes =
        checked_multiply(static_cast<std::uint64_t>(required_frames), config_.chunk_bytes);
    std::uint64_t codec_only_reserve =
        fixed_device_reserve_bytes_ >= config_.workspace_reserve_bytes
            ? fixed_device_reserve_bytes_ - config_.workspace_reserve_bytes
            : 0U;
    const auto required_with_workspace =
        required_frame_bytes.has_value() ? checked_add(*required_frame_bytes, required_workspace)
                                         : std::nullopt;
    auto required_total = required_with_workspace.has_value()
                              ? checked_add(*required_with_workspace, codec_only_reserve)
                              : std::nullopt;
    if (nvcomp_pipeline_ != nullptr &&
        (!required_total.has_value() || observed_target < *required_total ||
         observed_target <= fixed_device_reserve_bytes_)) {
      const RuntimeStatus suspended = suspend_codec_pipeline_for_budget();
      if (suspended != RuntimeStatus::success) {
        return suspended;
      }
      codec_only_reserve = 0;
      required_total = required_with_workspace;
    }
    std::uint64_t observed_frames =
        observed_target > fixed_device_reserve_bytes_
            ? (observed_target - fixed_device_reserve_bytes_) / config_.chunk_bytes
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

    if (codec_pipeline_suspended_for_budget_) {
      const auto restored_fixed_reserve =
          checked_add(config_.workspace_reserve_bytes, suspended_codec_device_bytes_);
      const auto current_frame_bytes = checked_multiply(frame_capacity_, config_.chunk_bytes);
      const auto current_with_workspace =
          current_frame_bytes.has_value() ? checked_add(*current_frame_bytes, required_workspace)
                                          : std::nullopt;
      const auto current_with_codec =
          current_with_workspace.has_value()
              ? checked_add(*current_with_workspace, suspended_codec_device_bytes_)
              : std::nullopt;
      const auto required_with_codec =
          required_with_workspace.has_value()
              ? checked_add(*required_with_workspace, suspended_codec_device_bytes_)
              : std::nullopt;
      const HostBudgetSnapshot host = backing_store_->budget().snapshot();
      const bool host_room = suspended_codec_pinned_bytes_ <= host.limit_bytes - host.total_bytes;
      const bool safe_restore_sample =
          restored_fixed_reserve.has_value() && current_with_codec.has_value() &&
          required_with_codec.has_value() && host_room && observed_target >= *current_with_codec &&
          observed_target >= *required_with_codec && observed_target > *restored_fixed_reserve &&
          !active_external_lease_.has_value() && !async_resources_quarantined_;
      if (safe_restore_sample) {
        if (consecutive_codec_restore_samples_ != std::numeric_limits<std::uint32_t>::max()) {
          ++consecutive_codec_restore_samples_;
        }
      } else {
        consecutive_codec_restore_samples_ = 0;
      }
      if (consecutive_codec_restore_samples_ >= 10U) {
        consecutive_codec_restore_samples_ = 0;
        const RuntimeStatus restored = restore_codec_pipeline_after_budget_growth();
        if (restored != RuntimeStatus::success) {
          return restored;
        }
        if (nvcomp_pipeline_ != nullptr) {
          codec_only_reserve = codec_managed_device_bytes_;
          required_total = required_with_workspace.has_value()
                               ? checked_add(*required_with_workspace, codec_only_reserve)
                               : std::nullopt;
          observed_frames =
              observed_target > fixed_device_reserve_bytes_
                  ? (observed_target - fixed_device_reserve_bytes_) / config_.chunk_bytes
                  : 0U;
          observed_frames = std::min(observed_frames, maximum_frame_capacity_);
          const auto active_frame_bytes = checked_multiply(frame_capacity_, config_.chunk_bytes);
          const auto restored_target =
              active_frame_bytes.has_value()
                  ? checked_add(*active_frame_bytes, fixed_device_reserve_bytes_)
                  : std::nullopt;
          if (!restored_target.has_value()) {
            ++telemetry_.device_budget_violation_count;
            return fail(RuntimeStatus::internal_failure, "budget", "restore_codec_target",
                        "restored cache-target accounting overflowed");
          }
          telemetry_.target_bytes = std::min(observed_target, *restored_target);
          telemetry_.target_maximum_bytes =
              std::max(telemetry_.target_maximum_bytes, telemetry_.target_bytes);
        }
      }
    }

    const auto managed_resident_and_workspace =
        checked_add(telemetry_.resident_bytes, workspace_bytes_);
    const auto managed_device =
        managed_resident_and_workspace.has_value()
            ? checked_add(*managed_resident_and_workspace, codec_managed_device_bytes_)
            : std::nullopt;
    if (!managed_device.has_value()) {
      ++telemetry_.device_budget_violation_count;
      return fail(RuntimeStatus::internal_failure, "budget", "managed_device_accounting",
                  "managed device resource accounting overflowed");
    }
    telemetry_.managed_device_bytes_peak =
        std::max(telemetry_.managed_device_bytes_peak, *managed_device);
    telemetry_.device_reserve_bytes_peak =
        std::max(telemetry_.device_reserve_bytes_peak, fixed_device_reserve_bytes_);
    if (*managed_device > observed_target) {
      ++telemetry_.device_budget_violation_count;
      return fail(RuntimeStatus::budget_pressure, "budget", "managed_device_accounting",
                  "managed residency, compute scratch, and codec resources exceed the live safe "
                  "device target");
    }

    if (!required_total.has_value() || observed_target < *required_total ||
        observed_target <= fixed_device_reserve_bytes_) {
      return fail(RuntimeStatus::budget_pressure, "budget", "working_set",
                  "live CUDA budget fell below the next transaction working set");
    }
    if (observed_frames < static_cast<std::uint64_t>(required_frames)) {
      return fail(RuntimeStatus::budget_pressure, "budget", "working_set",
                  "live CUDA budget cannot retain the next transaction frames");
    }

    if (!codec_pipeline_suspended_for_budget_ && observed_frames > frame_capacity_) {
      if (++consecutive_growth_samples_ >= 10U &&
          observed_frames >= frame_capacity_ + std::uint64_t{2}) {
        frame_capacity_ = std::min(observed_frames, frame_capacity_ + std::uint64_t{2});
        telemetry_.target_bytes = std::min(observed_target, frame_capacity_ * config_.chunk_bytes +
                                                                fixed_device_reserve_bytes_);
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
      try {
        std::uint64_t& reuse = chunk_reuse_counts_[plan.key];
        if (reuse != std::numeric_limits<std::uint64_t>::max()) {
          ++reuse;
        }
      } catch (...) {
        // Cost telemetry is advisory; allocation failure must not fail a valid cache hit.
      }
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
    CUmemLocation access_location{};
    access_location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_location.id = device_;
    unsigned long long observed_access = 0;
    if (const cuda::abi::Result code =
            api_.mem_get_access_(&observed_access, &access_location, address);
        code != cuda::abi::success ||
        observed_access != static_cast<unsigned long long>(CU_MEM_ACCESS_FLAGS_PROT_READWRITE)) {
      const cuda::abi::Result rollback =
          api_.mem_unmap_(address, static_cast<std::size_t>(config_.chunk_bytes));
      if (rollback != cuda::abi::success) {
        owner->reservation_quarantined = true;
        frame.quarantined = true;
        telemetry_.quarantined = true;
        release_quarantined_handle(frame);
        return poison_cuda("cache", "cuMemUnmap(get_access_rollback)", rollback);
      }
      frame.key.reset();
      frame.address = 0;
      frame.mapped = false;
      record->frame_index.reset();
      if (transition_chunk_state(record->state, ChunkState::host_clean) !=
          StateTransitionResult::success) {
        return poison("cache", "get_access_rollback",
                      "failed to restore the host-clean state after access verification");
      }
      return code == cuda::abi::success
                 ? fail(RuntimeStatus::cuda_failure, "cache", "cuMemGetAccess",
                        "CUDA mapping did not retain read-write access")
                 : fail_cuda("cache", "cuMemGetAccess", code);
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
    const auto retire_failed_h2d = [&](const RuntimeStatus original_status) -> RuntimeStatus {
      const RuntimeError original_error = error_;
      // No user work has seen this mapping yet, but the strict remap invariant still requires a
      // queryable event boundary before the physical handle or VA can be reused.
      const cuda::abi::Result retirement_record = api_.event_record_(staging.done, h2d_stream_);
      if (retirement_record != cuda::abi::success) {
        async_resources_quarantined_ = true;
        frame.quarantined = true;
        quarantine_mapping(*owner);
        return poison_cuda("cache", "cuEventRecord(failed_h2d)", retirement_record);
      }
      if (wait_event(staging.done, "failed_h2d_retirement") != RuntimeStatus::success) {
        frame.quarantined = true;
        quarantine_mapping(*owner);
        return error_.status;
      }
      record->staging_slot.reset();
      if (complete_event_generation(*record, record->event_generation) !=
              EventGenerationResult::success ||
          transition_chunk_state(record->state, ChunkState::resident_clean) !=
              StateTransitionResult::success) {
        frame.quarantined = true;
        quarantine_mapping(*owner);
        return poison_transition("cache", "failed_h2d_retirement",
                                 "failed H2D mapping could not retire safely");
      }
      const RuntimeStatus unmap_status = unmap(plan.key);
      if (unmap_status != RuntimeStatus::success) {
        return unmap_status;
      }
      error_ = original_error;
      return original_status;
    };
    if (staging.generation == std::numeric_limits<std::uint64_t>::max()) {
      const RuntimeStatus failed =
          fail(RuntimeStatus::internal_failure, "cache", "h2d_staging_generation",
               "H2D staging generation overflowed");
      return retire_failed_h2d(failed);
    }
    ++staging.generation;
    bool codec_transfer_completed = false;
    std::optional<CompressionTraceEvent> raw_h2d_trace;
    Clock::time_point raw_h2d_started{};
    std::uint64_t transfer_content_identity = 0;
    if (plan.requires_h2d) {
      const BackingResult info = backing_store_->inspect(plan.key);
      if (!info || info.chunk.representation == BackingRepresentation::invalid) {
        const RuntimeStatus failed = fail(RuntimeStatus::poisoned, "cache", "read_host_backing",
                                          "H2D source has no authoritative host generation");
        return retire_failed_h2d(failed);
      }
      transfer_content_identity = content_identity(info.chunk.content_token);
      if (info.chunk.representation == BackingRepresentation::lz4_blocks &&
          nvcomp_pipeline_ != nullptr && config_.forced_compression_path != CompressionPath::raw) {
        HostBudgetReservation export_scratch;
        const std::optional<std::uint64_t> export_charge =
            maximum_raw_backing_charge(info.chunk.valid_bytes);
        const HostBudgetStatus export_budget =
            export_charge.has_value()
                ? backing_store_->budget().reserve(HostBudgetCategory::conversion_scratch,
                                                   *export_charge, export_scratch)
                : HostBudgetStatus::reservation_overflow;
        if (export_budget != HostBudgetStatus::success) {
          const RuntimeStatus failed =
              fail(RuntimeStatus::host_oom, "cache", "reserve_codec_export",
                   "compressed H2D export exceeds the live host-store budget");
          return retire_failed_h2d(failed);
        }
        Lz4BlocksV1 container;
        const BackingStatus exported = backing_store_->export_lz4_blocks(plan.key, container);
        if (exported != BackingStatus::success) {
          (void)backing_store_->budget().release(export_scratch);
          const RuntimeStatus failed =
              fail(backing_runtime_status(exported), "cache", "export_lz4_blocks",
                   std::string(backing_status_name(exported)));
          return retire_failed_h2d(failed);
        }
        const NvcompPipelineTelemetry codec_before = nvcomp_pipeline_->telemetry();
        NvcompPipelineTicket ticket;
        const auto started = Clock::now();
        NvcompDecodeRequest decode_request{&container, address, plan.key, backing_path(plan.key),
                                           speculative};
        NvcompPipelineStatus codec_status = nvcomp_pipeline_->decode(decode_request, ticket);
        if (codec_status == NvcompPipelineStatus::success) {
          codec_status = nvcomp_pipeline_->wait(ticket);
        }
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - started);
        const NvcompPipelineTelemetry codec_after = nvcomp_pipeline_->telemetry();
        (void)backing_store_->budget().release(export_scratch);
        const std::uint64_t payload_bytes =
            codec_after.pcie_h2d_payload_bytes - codec_before.pcie_h2d_payload_bytes;
        const std::uint64_t metadata_bytes =
            codec_after.pcie_h2d_metadata_bytes - codec_before.pcie_h2d_metadata_bytes;
        telemetry_.pcie_h2d_payload_bytes += payload_bytes;
        telemetry_.pcie_h2d_metadata_bytes += metadata_bytes;
        telemetry_.pcie_h2d_bytes += payload_bytes + metadata_bytes;
        sync_codec_telemetry();
        if (codec_status == NvcompPipelineStatus::success) {
          codec_transfer_completed = true;
          ++telemetry_.decompression_attempts;
          ++telemetry_.decompression_commits;
          ++telemetry_.gpu_decode_operations;
          ++telemetry_.cpu_lz4_gpu_decode_decisions;
          telemetry_.gpu_decode_nanoseconds += static_cast<std::uint64_t>(elapsed.count());
          observe_latency_per_byte(gpu_decode_us_per_byte_, valid, elapsed);
        } else if (codec_status == NvcompPipelineStatus::quarantined ||
                   nvcomp_pipeline_->quarantined()) {
          async_resources_quarantined_ = true;
          frame.quarantined = true;
          quarantine_mapping(*owner);
          return poison("codec", "nvcomp_decode", nvcomp_pipeline_->error().c_str());
        } else {
          ++telemetry_.gpu_codec_fallbacks;
        }
      }
      if (!codec_transfer_completed) {
        const auto decode_started = Clock::now();
        HostBudgetReservation decode_scratch;
        if (info.chunk.representation == BackingRepresentation::lz4_blocks &&
            backing_store_->budget().reserve(HostBudgetCategory::conversion_scratch,
                                             compression_block_bytes,
                                             decode_scratch) != HostBudgetStatus::success) {
          const RuntimeStatus failed =
              fail(RuntimeStatus::host_oom, "cache", "reserve_cpu_decode_scratch",
                   "CPU decode scratch exceeds the host-store budget");
          return retire_failed_h2d(failed);
        }
        const BackingResult materialized =
            backing_store_->read(plan.key, 0,
                                 std::span<std::byte>{static_cast<std::byte*>(staging.memory),
                                                      static_cast<std::size_t>(valid)});
        if (decode_scratch.id != 0U) {
          (void)backing_store_->budget().release(decode_scratch);
        }
        const auto decode_elapsed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - decode_started);
        if (!materialized) {
          const RuntimeStatus failed =
              fail(backing_runtime_status(materialized.status), "cache", "decode_host_backing",
                   std::string(backing_status_name(materialized.status)));
          return retire_failed_h2d(failed);
        }
        if (info.chunk.representation == BackingRepresentation::lz4_blocks) {
          ++telemetry_.decompression_attempts;
          ++telemetry_.decompression_commits;
          ++telemetry_.cpu_decode_operations;
          ++telemetry_.cpu_codec_fallbacks;
          telemetry_.cpu_decode_nanoseconds += static_cast<std::uint64_t>(decode_elapsed.count());
          observe_latency_per_byte(cpu_decode_us_per_byte_, valid, decode_elapsed);
        } else {
          ++telemetry_.raw_path_decisions;
        }
        raw_h2d_started = Clock::now();
        if (const cuda::abi::Result code = api_.memcpy_h2d_async_(
                address, staging.memory, static_cast<std::size_t>(valid), h2d_stream_);
            code != cuda::abi::success) {
          async_resources_quarantined_ = true;
          frame.quarantined = true;
          quarantine_mapping(*owner);
          return poison_cuda("cache", "cuMemcpyHtoDAsync", code);
        }
        CompressionTraceEvent transfer_event;
        transfer_event.kind = CompressionTraceEventKind::h2d_submit;
        transfer_event.key = plan.key;
        transfer_event.operation_id = record->event_generation;
        transfer_event.source_generation = info.chunk.generation;
        transfer_event.slot_generation = staging.generation;
        transfer_event.path = backing_path(plan.key);
        transfer_event.from_representation = info.chunk.representation;
        transfer_event.logical_bytes = valid;
        transfer_event.physical_bytes = valid;
        transfer_event.reason = info.chunk.representation == BackingRepresentation::lz4_blocks
                                    ? "cpu_decoded_raw_payload_enqueued"
                                    : "raw_host_payload_enqueued";
        transfer_event.speculative = speculative;
        raw_h2d_trace = transfer_event;
        emit_compression_trace(*raw_h2d_trace);
        telemetry_.pcie_h2d_payload_bytes += valid;
        telemetry_.pcie_h2d_bytes += valid;
      }
      telemetry_.h2d_bytes += valid;
      telemetry_.logical_h2d_bytes += valid;
    }
    if (!codec_transfer_completed) {
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
      if (raw_h2d_started != Clock::time_point{} && valid != 0U) {
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - raw_h2d_started);
        const auto cost_elapsed = compression_cost_duration(CompressionPath::raw, elapsed);
        observe_latency_per_byte(raw_h2d_us_per_byte_, valid, cost_elapsed);
        if (transfer_content_identity != 0U) {
          CompressionCostModel& model = cost_models_.try_emplace(plan.key).first->second;
          if (model.generation() != transfer_content_identity) {
            model.reset(transfer_content_identity);
          }
          (void)model.observe(transfer_content_identity,
                              CostObservation{CompressionPath::raw, valid, valid, 0.0, 0.0,
                                              static_cast<double>(cost_elapsed.count()) / 1000.0});
        }
      }
    }
    record->staging_slot.reset();
    if (complete_event_generation(*record, record->event_generation) !=
            EventGenerationResult::success ||
        transition_chunk_state(record->state, ChunkState::resident_clean) !=
            StateTransitionResult::success) {
      return poison("cache", "h2d_retire", "H2D generation could not retire");
    }
    if (raw_h2d_trace.has_value()) {
      raw_h2d_trace->kind = CompressionTraceEventKind::h2d_retire;
      raw_h2d_trace->reason = "raw_h2d_generation_retired";
      emit_compression_trace(*raw_h2d_trace);
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
    const BackingResult before = backing_store_->inspect(key);
    if (!before) {
      return poison("cache", "writeback_commit", "authoritative host chunk disappeared");
    }
    bool authority_committed = false;
    bool compression_decided = false;
    bool codec_candidate_transferred = false;
    std::uint64_t codec_candidate_physical_bytes = 0;
    std::optional<NvcompPipelineTicket> codec_commit_ticket;
    bool codec_candidate_generation_pending = false;
    HostBudgetReservation codec_candidate_reservation;
    HostBudgetReservation codec_verification_reservation;
    bool codec_candidate_reserved = false;
    const bool gpu_encode_allowed =
        !config_.forced_compression_path.has_value() ||
        config_.forced_compression_path == CompressionPath::nvcomp_gpu_codec;
    bool learned_raw_skip = false;
    if (config_.compression_mode == CompressionMode::adaptive &&
        !config_.forced_compression_path.has_value() && nvcomp_pipeline_ != nullptr) {
      const auto model_found = cost_models_.find(key);
      if (model_found != cost_models_.end()) {
        const std::uint64_t identity = content_identity(before.chunk.content_token);
        if (model_found->second.generation() == identity) {
          const CostDecision learned =
              model_found->second.decide(CompressionMode::adaptive, valid, true, true);
          // A generation-scoped never-compress mark is terminal. Otherwise require both the raw
          // and GPU paths to have enough samples before suppressing a real codec probe: a raw
          // decision caused merely by missing GPU calibration must not disable compression.
          learned_raw_skip = learned.never_compress ||
                             (learned.path == CompressionPath::raw && learned.raw.confident &&
                              learned.nvcomp_gpu_codec.confident);
        }
      }
      if (learned_raw_skip) {
        compression_decided = true;
        if (model_found->second.never_compress()) {
          ++telemetry_.never_compress_decisions;
        } else {
          ++telemetry_.raw_path_decisions;
        }
      }
    }
    if (config_.compression_mode != CompressionMode::disabled && nvcomp_pipeline_ != nullptr &&
        gpu_encode_allowed && !learned_raw_skip) {
      const std::optional<std::uint64_t> maximum_candidate_charge =
          maximum_raw_backing_charge(valid);
      const HostBudgetStatus candidate_budget_status =
          maximum_candidate_charge.has_value()
              ? backing_store_->budget().reserve(HostBudgetCategory::conversion_scratch,
                                                 *maximum_candidate_charge,
                                                 codec_candidate_reservation)
              : HostBudgetStatus::reservation_overflow;
      const HostBudgetStatus verification_budget_status =
          candidate_budget_status == HostBudgetStatus::success
              ? backing_store_->budget().reserve(HostBudgetCategory::conversion_scratch,
                                                 compression_block_bytes,
                                                 codec_verification_reservation)
              : candidate_budget_status;
      if (candidate_budget_status != HostBudgetStatus::success ||
          verification_budget_status != HostBudgetStatus::success) {
        if (candidate_budget_status == HostBudgetStatus::success) {
          (void)backing_store_->budget().release(codec_candidate_reservation);
        }
        ++telemetry_.gpu_codec_fallbacks;
      } else {
        codec_candidate_reserved = true;
        const NvcompPipelineTelemetry codec_before = nvcomp_pipeline_->telemetry();
        NvcompPipelineTicket ticket;
        const auto started = Clock::now();
        NvcompEncodeRequest encode_request{frame->address, valid, before.chunk.generation,
                                           std::nullopt, key, CompressionPath::nvcomp_gpu_codec,
                                           record->speculative};
        NvcompPipelineStatus codec_status = nvcomp_pipeline_->encode(encode_request, ticket);
        Lz4BlocksV1 candidate;
        if (codec_status == NvcompPipelineStatus::success) {
          codec_status = nvcomp_pipeline_->wait(ticket, &candidate);
        }
        (void)backing_store_->budget().release(codec_verification_reservation);
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - started);
        const NvcompPipelineTelemetry codec_after = nvcomp_pipeline_->telemetry();
        const std::uint64_t candidate_stored_bytes =
            codec_after.pcie_d2h_payload_bytes - codec_before.pcie_d2h_payload_bytes;
        const std::uint64_t metadata_bytes =
            codec_after.pcie_d2h_metadata_bytes - codec_before.pcie_d2h_metadata_bytes;
        const std::uint64_t pcie_bytes = candidate_stored_bytes + metadata_bytes;
        telemetry_.pcie_d2h_payload_bytes += candidate_stored_bytes;
        telemetry_.pcie_d2h_metadata_bytes += metadata_bytes;
        telemetry_.pcie_d2h_bytes += pcie_bytes;
        if (codec_status == NvcompPipelineStatus::success) {
          codec_candidate_transferred = true;
          codec_commit_ticket = ticket;
          codec_candidate_generation_pending = true;
          ++telemetry_.generations_created;
          ++telemetry_.compression_attempts;
          ++telemetry_.gpu_encode_operations;
          ++telemetry_.codec_calibration_samples;
          telemetry_.gpu_encode_nanoseconds += static_cast<std::uint64_t>(elapsed.count());
          codec_candidate_physical_bytes = candidate_stored_bytes;
          const double raw_transfer_us = raw_h2d_us_per_byte_ * static_cast<double>(valid);
          const double candidate_us = static_cast<double>(elapsed.count()) / 1000.0;
          const double compressed_transfer_us =
              raw_h2d_us_per_byte_ * static_cast<double>(candidate_stored_bytes);
          const double encode_us = std::max(0.0, candidate_us - compressed_transfer_us);
          const double decode_us = gpu_decode_us_per_byte_ * static_cast<double>(valid);
          const auto reuse_found = chunk_reuse_counts_.find(key);
          const std::uint64_t reuse_count =
              reuse_found == chunk_reuse_counts_.end() ? 0U : reuse_found->second;
          CompressionCostModel& model = cost_models_.try_emplace(key).first->second;
          const std::uint64_t identity = content_identity(candidate.content_token);
          if (model.generation() != identity) {
            model.reset(identity);
          }
          (void)model.observe(identity, CostObservation{CompressionPath::raw, valid, valid, 0.0,
                                                        0.0, raw_transfer_us, 0.0, 0.0, 0.0, 0.0,
                                                        1.0, reuse_count, true});
          (void)model.observe(identity, CostObservation{CompressionPath::nvcomp_gpu_codec, valid,
                                                        candidate_stored_bytes, encode_us,
                                                        decode_us, compressed_transfer_us, 0.0, 0.0,
                                                        0.0, 0.0, 1.0, reuse_count, true});
          const bool favorable = model.record_probe(identity, CompressionPath::nvcomp_gpu_codec,
                                                    raw_transfer_us, candidate_us + decode_us);
          const CostDecision cost = model.decide(config_.compression_mode, valid, true, true);
          const bool select_compressed = config_.compression_mode != CompressionMode::adaptive ||
                                         config_.forced_compression_path.has_value() ||
                                         (favorable && !cost.calibration_required &&
                                          cost.path == CompressionPath::nvcomp_gpu_codec);

          if (select_compressed) {
            const BackingResult compressed = backing_store_->commit_lz4_blocks(
                key, before.chunk.generation, std::move(candidate), &codec_candidate_reservation);
            if (compressed) {
              codec_candidate_reserved = false;
              release_spill(key);
              authority_committed = true;
              compression_decided = true;
              ++telemetry_.compression_commits;
              ++telemetry_.generations_committed;
              codec_candidate_generation_pending = false;
              set_backing_path(key, CompressionPath::nvcomp_gpu_codec);
              const NvcompPipelineStatus acknowledged =
                  nvcomp_pipeline_->acknowledge_host_commit(ticket);
              if (acknowledged != NvcompPipelineStatus::success) {
                async_resources_quarantined_ = nvcomp_pipeline_->quarantined();
                frame->quarantined = true;
                quarantine_mapping(*owner);
                return poison("codec", "acknowledge_host_commit",
                              nvcomp_pipeline_->error().c_str());
              }
              codec_commit_ticket.reset();
              ++telemetry_.gpu_lz4_decisions;
            } else {
              ++telemetry_.atomic_commit_failures;
              ++telemetry_.gpu_codec_fallbacks;
            }
          } else {
            // The encoded candidate was fully verified, but adaptive policy predicts raw to be
            // faster. Preserve the pre-admitted spill and perform a direct raw D2H below; do not
            // transiently make the compressed candidate authoritative only to decode it on CPU.
            compression_decided = true;
            ++telemetry_.raw_fallbacks;
            if (model.never_compress()) {
              ++telemetry_.never_compress_decisions;
            } else {
              ++telemetry_.raw_path_decisions;
            }
          }
          sync_codec_telemetry();
        } else if (codec_status == NvcompPipelineStatus::quarantined ||
                   nvcomp_pipeline_->quarantined()) {
          (void)backing_store_->budget().release(codec_candidate_reservation);
          codec_candidate_reserved = false;
          sync_codec_telemetry();
          async_resources_quarantined_ = true;
          frame->quarantined = true;
          quarantine_mapping(*owner);
          return poison("codec", "nvcomp_encode", nvcomp_pipeline_->error().c_str());
        } else {
          ++telemetry_.gpu_codec_fallbacks;
        }
        if (codec_candidate_reserved) {
          if (backing_store_->budget().release(codec_candidate_reservation) !=
              HostBudgetStatus::success) {
            if (codec_commit_ticket.has_value()) {
              (void)nvcomp_pipeline_->quarantine_uncommitted(*codec_commit_ticket);
              async_resources_quarantined_ = true;
              frame->quarantined = true;
              quarantine_mapping(*owner);
            }
            return poison("compression", "discard_candidate",
                          "GPU codec candidate retained an unknown host-budget reservation");
          }
          codec_candidate_reserved = false;
        }
      }
    }

    std::optional<CompressionTraceEvent> raw_d2h_trace;
    if (!authority_committed) {
      if (codec_candidate_transferred) {
        // The verified compressed candidate crossed PCIe but was rejected or failed its atomic
        // authority commit. Count that full logical attempt separately from the raw spill D2H
        // below so logical/physical transfer reconciliation does not hide double transfers.
        telemetry_.logical_d2h_bytes = saturating_add(telemetry_.logical_d2h_bytes, valid);
        telemetry_.rejected_candidate_logical_d2h_bytes =
            saturating_add(telemetry_.rejected_candidate_logical_d2h_bytes, valid);
      }
      if (staging.generation == std::numeric_limits<std::uint64_t>::max()) {
        if (codec_commit_ticket.has_value() && nvcomp_pipeline_ != nullptr) {
          (void)nvcomp_pipeline_->quarantine_uncommitted(*codec_commit_ticket);
        }
        async_resources_quarantined_ = true;
        frame->quarantined = true;
        quarantine_mapping(*owner);
        return poison("cache", "d2h_staging_generation", "D2H staging generation overflowed");
      }
      ++staging.generation;
      if (const cuda::abi::Result code = api_.memcpy_d2h_async_(
              staging.memory, frame->address, static_cast<std::size_t>(valid), d2h_stream_);
          code != cuda::abi::success) {
        async_resources_quarantined_ = true;
        frame->quarantined = true;
        quarantine_mapping(*owner);
        return poison_cuda("cache", "cuMemcpyDtoHAsync", code);
      }
      CompressionTraceEvent transfer_event;
      transfer_event.kind = CompressionTraceEventKind::d2h_submit;
      transfer_event.key = key;
      transfer_event.operation_id = record->event_generation;
      transfer_event.source_generation = before.chunk.generation;
      if (before.chunk.generation != std::numeric_limits<std::uint64_t>::max()) {
        transfer_event.target_generation = before.chunk.generation + 1U;
      }
      transfer_event.slot_generation = staging.generation;
      transfer_event.path = CompressionPath::raw;
      transfer_event.from_representation = before.chunk.representation;
      transfer_event.to_representation = BackingRepresentation::raw;
      transfer_event.logical_bytes = valid;
      transfer_event.physical_bytes = valid;
      transfer_event.reason = codec_candidate_transferred
                                  ? "raw_spill_after_codec_rejection_enqueued"
                                  : "raw_writeback_enqueued";
      transfer_event.speculative = record->speculative;
      raw_d2h_trace = transfer_event;
      emit_compression_trace(*raw_d2h_trace);
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
      BackingResult committed;
      const auto spill = spill_reservations_.find(key);
      if (spill != spill_reservations_.end()) {
        const BackingStatus filled = backing_store_->fill_prepared_raw(
            spill->second.backing,
            std::span<const std::byte>{static_cast<const std::byte*>(staging.memory),
                                       static_cast<std::size_t>(valid)});
        if (filled != BackingStatus::success) {
          const std::string filled_name{backing_status_name(filled)};
          return poison("cache", "fill_raw_spill", filled_name.c_str());
        }
        committed = backing_store_->commit_prepared_raw(
            key, before.chunk.generation, spill->second.backing, &spill->second.reservation);
        if (committed) {
          spill_reservations_.erase(spill);
          telemetry_.spill_reserved_bytes = backing_store_->budget().snapshot().spill_bytes;
        }
      } else {
        committed = backing_store_->replace_raw(
            key, before.chunk.generation,
            std::span<const std::byte>{static_cast<const std::byte*>(staging.memory),
                                       static_cast<std::size_t>(valid)});
      }
      if (!committed) {
        if (codec_commit_ticket.has_value() && nvcomp_pipeline_ != nullptr) {
          (void)nvcomp_pipeline_->quarantine_uncommitted(*codec_commit_ticket);
          async_resources_quarantined_ = true;
          frame->quarantined = true;
          quarantine_mapping(*owner);
        }
        return fail(backing_runtime_status(committed.status), "cache", "writeback_commit",
                    std::string(backing_status_name(committed.status)));
      }
      set_backing_path(key, CompressionPath::raw);
      if (codec_commit_ticket.has_value() && nvcomp_pipeline_ != nullptr) {
        const NvcompPipelineStatus acknowledged =
            nvcomp_pipeline_->acknowledge_host_commit(*codec_commit_ticket);
        if (acknowledged != NvcompPipelineStatus::success) {
          async_resources_quarantined_ = nvcomp_pipeline_->quarantined();
          frame->quarantined = true;
          quarantine_mapping(*owner);
          return poison("codec", "acknowledge_host_commit", nvcomp_pipeline_->error().c_str());
        }
        if (codec_candidate_generation_pending) {
          ++telemetry_.generations_discarded;
          emit_generation_discard(key, codec_commit_ticket->operation_id, before.chunk,
                                  before.chunk.generation + 1U,
                                  codec_commit_ticket->slot_generation,
                                  CompressionPath::nvcomp_gpu_codec, codec_candidate_physical_bytes,
                                  "gpu_candidate_raw_fallback_committed", record->speculative);
          codec_candidate_generation_pending = false;
        }
        codec_commit_ticket.reset();
      }
      ++telemetry_.generations_created;
      ++telemetry_.generations_committed;
      telemetry_.pcie_d2h_payload_bytes += valid;
      telemetry_.pcie_d2h_bytes += valid;
    }
    record->staging_slot.reset();
    if (complete_event_generation(*record, record->event_generation) !=
            EventGenerationResult::success ||
        transition_chunk_state(record->state, ChunkState::resident_clean) !=
            StateTransitionResult::success) {
      return poison("cache", "writeback_retire", "D2H generation could not retire");
    }
    if (raw_d2h_trace.has_value()) {
      raw_d2h_trace->kind = CompressionTraceEventKind::d2h_retire;
      raw_d2h_trace->reason = "raw_d2h_generation_committed";
      emit_compression_trace(*raw_d2h_trace);
    }
    telemetry_.d2h_bytes += valid;
    telemetry_.logical_d2h_bytes += valid;
    if (owner->hint == ResidencyHint::hot) {
      telemetry_.hot_allocation_d2h_bytes += valid;
    } else {
      telemetry_.non_hot_allocation_d2h_bytes += valid;
    }
    ++telemetry_.dirty_writebacks;
    if (!compression_decided) {
      if (const RuntimeStatus status = consider_compression(key);
          status != RuntimeStatus::success) {
        return status;
      }
    }
    refresh_host_telemetry();
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

  [[nodiscard]] RuntimeStatus discard_dirty_mapping(const ChunkKey key) {
    ChunkRecord* record = chunk(key);
    Allocation* owner = allocation(key.allocation_id);
    Frame* frame = frame_for(key);
    if (record == nullptr || owner == nullptr || frame == nullptr ||
        record->state != ChunkState::resident_dirty || !is_victim_eligible(*record) ||
        record->event_generation == 0 || record->completed_generation != record->event_generation) {
      ++telemetry_.unsafe_remaps;
      return poison("cache", "discard_dead",
                    "dirty dead chunk lacks a completed last-use event boundary");
    }
    const cuda::abi::Result code =
        api_.mem_unmap_(frame->address, static_cast<std::size_t>(config_.chunk_bytes));
    if (code != cuda::abi::success) {
      frame->quarantined = true;
      quarantine_mapping(*owner);
      release_quarantined_handle(*frame);
      return poison_cuda("cache", "cuMemUnmap(discard_dead)", code);
    }
    ++telemetry_.unmaps;
    ++telemetry_.event_boundaries;
    telemetry_.resident_bytes -= config_.chunk_bytes;
    (void)policy_->erase(record->key);
    frame->key.reset();
    frame->address = 0;
    frame->mapped = false;
    frame->quarantined = false;
    record->frame_index.reset();
    record->speculative = false;
    // The range is liveness-proven dead. Its dirty contents deliberately do not become
    // host-visible; discard_dead() records that the next use must fully overwrite the chunk.
    record->state = ChunkState::host_clean;
    return RuntimeStatus::success;
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
  std::uint64_t fixed_device_reserve_bytes_ = 0;
  std::uint64_t codec_managed_device_bytes_ = 0;
  std::uint32_t consecutive_growth_samples_ = 0;
  std::uint32_t consecutive_codec_restore_samples_ = 0;
  std::uint64_t suspended_codec_device_bytes_ = 0;
  std::uint64_t suspended_codec_pinned_bytes_ = 0;
  bool codec_pipeline_suspended_for_budget_ = false;
  NvcompPipelineTelemetry retired_codec_telemetry_;
  bool wddm_budget_active_ = false;
  Clock::time_point last_budget_sample_{};
#ifdef _WIN32
  std::optional<platform::AdapterLuid> wddm_luid_;
  std::uint32_t wddm_node_mask_ = 0;
#endif
  std::uint64_t policy_sequence_ = 0;
  std::unordered_map<std::uint64_t, std::unique_ptr<Allocation>> allocations_;
  std::vector<RetiredReservation> retired_reservations_;
  std::unique_ptr<BlockCodec> backing_codec_;
  std::unique_ptr<HostBackingStore> backing_store_;
  std::unique_ptr<CpuCodecWorkerPool> cpu_codec_pool_;
  std::uint32_t cpu_codec_queue_capacity_ = 0;
  std::uint64_t observed_cpu_codec_blocks_submitted_ = 0;
  bool cpu_codec_timeout_ = false;
  HostBudgetReservation pinned_budget_reservation_;
  bool pinned_budget_active_ = false;
  HostBudgetReservation codec_pinned_budget_reservation_;
  bool codec_pinned_budget_active_ = false;
  std::unordered_map<ChunkKey, RawSpill, ChunkKeyHash> spill_reservations_;
  std::unordered_map<ChunkKey, CompressionCostModel, ChunkKeyHash> cost_models_;
  std::unordered_map<ChunkKey, std::uint64_t, ChunkKeyHash> chunk_reuse_counts_;
  double raw_h2d_us_per_byte_ = 1'000'000.0 / 12'000'000'000.0;
  double cpu_decode_us_per_byte_ = 0.0;
  double gpu_decode_us_per_byte_ = 0.0;
  std::unique_ptr<nvcomp::NvcompApi> owned_nvcomp_api_;
  nvcomp::NvcompApi* nvcomp_api_ = nullptr;
  bool nvcomp_available_ = false;
  std::unique_ptr<NvcompLz4Pipeline> nvcomp_pipeline_;
  std::vector<Frame> frames_;
  std::unique_ptr<VictimPolicy> policy_;
  cuda::abi::Stream h2d_stream_ = nullptr;
  cuda::abi::Stream compute_stream_ = nullptr;
  cuda::abi::Stream d2h_stream_ = nullptr;
  cuda::abi::Event compute_ready_ = nullptr;
  cuda::abi::Event compute_started_ = nullptr;
  cuda::abi::Event compute_done_ = nullptr;
  std::vector<StagingSlot> h2d_staging_;
  std::vector<StagingSlot> d2h_staging_;
  std::size_t next_h2d_staging_ = 0;
  std::size_t next_d2h_staging_ = 0;
  cuda::abi::DevicePointer workspace_ = 0;
  std::uint64_t workspace_bytes_ = 0;
  ExternalLeaseId next_external_lease_id_{1};
  std::optional<ActiveExternalLease> active_external_lease_;
  std::optional<ExternalLeasePoll> last_external_lease_;
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
RuntimeStatus Runtime::discard_dead(const AllocationId id) {
  return impl_->discard_dead(id);
}
RuntimeStatus Runtime::discard_dead(const AllocationId id, const std::uint64_t offset,
                                    const std::uint64_t bytes) {
  return impl_->discard_dead(id, offset, bytes);
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
RuntimeStatus Runtime::acquire_external(const ExternalLeaseRequest& request,
                                        ExternalLease& output) {
  return impl_->acquire_external(request, output);
}
RuntimeStatus Runtime::seal_external(const ExternalLeaseId lease_id, const ExternalSealMode mode) {
  return impl_->seal_external(lease_id, mode);
}
RuntimeStatus Runtime::poll_external(const ExternalLeaseId lease_id, ExternalLeasePoll& output) {
  return impl_->poll_external(lease_id, output);
}
RuntimeStatus Runtime::wait_external(const ExternalLeaseId lease_id,
                                     const std::chrono::milliseconds timeout,
                                     ExternalLeasePoll& output) {
  return impl_->wait_external(lease_id, timeout, output);
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
std::optional<cuda::abi::DevicePointer>
Runtime::allocation_address(const AllocationId id) const noexcept {
  return impl_->allocation_address(id);
}
std::optional<HostChunkInfo> Runtime::backing_info(const AllocationId id,
                                                   const std::uint64_t chunk_index) const noexcept {
  return impl_->backing_info(id, chunk_index);
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
