#include "cuda_compat/backend.hpp"

#include "gemm/executor.hpp"
#include "platform/cublas/cublas_api.hpp"
#include "platform/cuda/cuda_api.hpp"
#include "platform/system_info.hpp"
#include "residency/runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace xvram::cuda_compat {
namespace {

[[nodiscard]] xvram_status public_status(const residency::RuntimeStatus status) noexcept {
  using residency::RuntimeStatus;
  switch (status) {
  case RuntimeStatus::success:
    return XVRAM_STATUS_SUCCESS;
  case RuntimeStatus::invalid_argument:
    return XVRAM_STATUS_INVALID_ARGUMENT;
  case RuntimeStatus::unavailable:
    return XVRAM_STATUS_UNAVAILABLE;
  case RuntimeStatus::unsupported:
    return XVRAM_STATUS_UNSUPPORTED;
  case RuntimeStatus::host_oom:
    return XVRAM_STATUS_HOST_OUT_OF_MEMORY;
  case RuntimeStatus::device_oom:
    return XVRAM_STATUS_DEVICE_OUT_OF_MEMORY;
  case RuntimeStatus::budget_pressure:
    return XVRAM_STATUS_BUDGET_PRESSURE;
  case RuntimeStatus::timeout:
    return XVRAM_STATUS_TIMEOUT;
  case RuntimeStatus::callback_skipped:
    return XVRAM_STATUS_CANCELLED;
  case RuntimeStatus::callback_failed:
    return XVRAM_STATUS_CALLBACK_FAILED;
  case RuntimeStatus::cuda_failure:
    return XVRAM_STATUS_CUDA_ERROR;
  case RuntimeStatus::poisoned:
    return XVRAM_STATUS_POISONED;
  case RuntimeStatus::cleanup_failure:
    return XVRAM_STATUS_CLEANUP_FAILED;
  case RuntimeStatus::internal_failure:
    return XVRAM_STATUS_INTERNAL;
  }
  return XVRAM_STATUS_INTERNAL;
}

[[nodiscard]] sdk::Error runtime_error(const residency::Runtime& runtime,
                                       const residency::RuntimeStatus status) {
  const auto& source = runtime.error();
  return sdk::make_native_error(public_status(status),
                                source.native_code ? XVRAM_NATIVE_ERROR_CUDA
                                                   : XVRAM_NATIVE_ERROR_INTERNAL,
                                source.native_code.value_or(0), source.stage, source.operation,
                                source.native_code ? "CUDA" : "", source.message);
}

[[nodiscard]] sdk::Error cuda_error(const cuda::abi::Result status,
                                    const std::string_view operation) {
  return sdk::make_native_error(XVRAM_STATUS_CUDA_ERROR, XVRAM_NATIVE_ERROR_CUDA, status,
                                "cuda_compat", operation, "CUDA", "CUDA metadata query failed");
}

[[nodiscard]] sdk::Error blas_error(const cublas::abi::Status status,
                                    const std::string_view operation,
                                    const xvram_status result = XVRAM_STATUS_CUBLAS_ERROR) {
  return sdk::make_native_error(result, XVRAM_NATIVE_ERROR_CUBLAS, status, "cuda_compat", operation,
                                "cuBLAS", "cuBLAS resource operation failed");
}

template <std::size_t Size>
[[nodiscard]] std::filesystem::path library_path(const char (&text)[Size]) {
  return std::filesystem::path(std::u8string(text, std::find(text, text + Size, '\0')));
}

class RuntimeBackend final : public Backend {
public:
  ~RuntimeBackend() override {
    // Adapter normally closes explicitly on its worker. Keep the observer and snapshot alive
    // during fallback cleanup too: Runtime's lifecycle callback captures this backend.
    try {
      (void)close();
    } catch (...) {
    }
  }

  sdk::Error initialize(const xvram_cuda_compat_config_v1& config, SnapshotSink sink) override {
    sink_ = std::move(sink);
    config_ = config.session;
    snapshot_.struct_size = sizeof(snapshot_);
    snapshot_.runtime.struct_size = sizeof(snapshot_.runtime);
    snapshot_.device_ordinal = config_.device_ordinal;

    const auto system = platform::collect_system_info();
    if (!system.physical_memory_bytes || !system.available_memory_bytes) {
      return sdk::make_error(XVRAM_STATUS_UNAVAILABLE, "cuda_compat", "host_budget",
                             "host memory telemetry is required for raw backing admission");
    }
    snapshot_.host_physical_bytes = *system.physical_memory_bytes;
    snapshot_.host_available_bytes = *system.available_memory_bytes;
    constexpr std::uint64_t minimum_host_headroom = 4ULL * 1024ULL * 1024ULL * 1024ULL;
    const std::uint64_t headroom =
        std::max(minimum_host_headroom, snapshot_.host_physical_bytes / 4U);
    if (snapshot_.host_available_bytes <= headroom) {
      publish();
      return sdk::make_error(XVRAM_STATUS_HOST_OUT_OF_MEMORY, "cuda_compat", "host_budget",
                             "available RAM does not leave the automatic host headroom");
    }

    residency::RuntimeConfig runtime_config;
    runtime_config.device_ordinal = config_.device_ordinal;
    runtime_config.context_mode = residency::RuntimeContextMode::isolated;
    runtime_config.retain_released_va = true;
    runtime_config.chunk_bytes = config_.chunk_size_bytes;
    runtime_config.cache_target_bytes = config_.cache_target_bytes;
    runtime_config.device_headroom_bytes = config_.device_headroom_bytes;
    runtime_config.workspace_reserve_bytes = config_.workspace_cap_bytes;
    runtime_config.staging_slots = config_.staging_slots;
    runtime_config.policy = config_.cache_policy == XVRAM_CACHE_POLICY_LRU
                                ? residency::RuntimePolicy::lru
                                : residency::RuntimePolicy::clock;
    runtime_config.budget_poll_interval = std::chrono::milliseconds(config_.budget_poll_ms);
    runtime_config.maximum_transaction_duration = std::chrono::milliseconds(
        config_.max_transaction_ms == 0 ? 250
                                        : std::min<std::uint64_t>(250, config_.max_transaction_ms));
    runtime_config.stall_timeout = std::chrono::milliseconds(config.stall_timeout_ms);
    runtime_config.host_headroom_bytes = headroom;
    // This is the total host-ledger limit: Runtime also charges its pinned staging here. Do not
    // subtract staging separately, or the same physical RAM would be reserved twice.
    runtime_config.host_store_cap_bytes = snapshot_.host_available_bytes - headroom;
    runtime_config.lifecycle_progress = [this]() {
      ++progress_sequence_;
      publish();
    };
    runtime_ = std::make_unique<residency::Runtime>(cuda_, std::move(runtime_config));
    if (const auto status = runtime_->setup(); status != residency::RuntimeStatus::success) {
      publish();
      return runtime_error(*runtime_, status);
    }

    if (cuda_.device_get_count_ == nullptr || cuda_.device_get_name_ == nullptr ||
        cuda_.device_total_memory_ == nullptr || cuda_.driver_get_version_ == nullptr) {
      return sdk::make_error(XVRAM_STATUS_UNAVAILABLE, "cuda_compat", "device_metadata",
                             "CUDA driver does not expose the required device metadata");
    }
    int devices = 0;
    int driver_version = 0;
    std::size_t device_bytes = 0;
    if (const auto status = cuda_.device_get_count_(&devices); status != cuda::abi::success) {
      return cuda_error(status, "cuDeviceGetCount");
    }
    if (const auto status = cuda_.device_get_name_(snapshot_.device_name,
                                                   static_cast<int>(sizeof(snapshot_.device_name)),
                                                   runtime_->device());
        status != cuda::abi::success) {
      return cuda_error(status, "cuDeviceGetName");
    }
    if (const auto status = cuda_.device_total_memory_(&device_bytes, runtime_->device());
        status != cuda::abi::success) {
      return cuda_error(status, "cuDeviceTotalMem");
    }
    if (const auto status = cuda_.driver_get_version_(&driver_version);
        status != cuda::abi::success) {
      return cuda_error(status, "cuDriverGetVersion");
    }
    snapshot_.device_count = static_cast<std::uint32_t>(std::max(devices, 0));
    snapshot_.total_vram_bytes = static_cast<std::uint64_t>(device_bytes);
    snapshot_.runtime.cuda_driver_version = static_cast<std::uint32_t>(std::max(driver_version, 0));

    const auto loaded =
        cublas_.load({library_path(config.cublas_library), library_path(config.cublas_lt_library)});
    if (loaded.status != cublas::CublasLoadStatus::loaded) {
      publish();
      return sdk::make_error(loaded.status == cublas::CublasLoadStatus::invalid_paths
                                 ? XVRAM_STATUS_INVALID_ARGUMENT
                                 : XVRAM_STATUS_UNAVAILABLE,
                             "cuda_compat", "load_cublas", cublas_.error());
    }
    const auto& dispatch = cublas_.dispatch();
    if (const auto status = dispatch.create(&handle_); status != cublas::abi::success) {
      handle_ = nullptr;
      return blas_error(status, "cublasCreate_v2");
    }
    int version = 0;
    if (const auto status = dispatch.get_version(handle_, &version);
        status != cublas::abi::success) {
      return blas_error(status, "cublasGetVersion_v2");
    }
    snapshot_.runtime.cublas_version = static_cast<std::uint32_t>(std::max(version, 0));
    switch (cublas_.library_source()) {
    case cublas::CublasLibrarySource::system:
      snapshot_.runtime.cublas_library_source = XVRAM_CUBLAS_SOURCE_SYSTEM;
      break;
    case cublas::CublasLibrarySource::app_local:
      snapshot_.runtime.cublas_library_source = XVRAM_CUBLAS_SOURCE_APP_LOCAL;
      break;
    case cublas::CublasLibrarySource::explicit_path:
      snapshot_.runtime.cublas_library_source = XVRAM_CUBLAS_SOURCE_EXPLICIT;
      break;
    default:
      break;
    }
    if (cublas_.has_lt()) {
      lt_ = std::make_unique<cublas::LtMatmulExecutor>(dispatch);
      if (lt_->initialize() == cublas::abi::success) {
        snapshot_.runtime.cublas_lt_available = 1U;
        snapshot_.runtime.cublas_lt_version = static_cast<std::uint32_t>(std::min<std::size_t>(
            lt_->library_version(), std::numeric_limits<std::uint32_t>::max()));
      } else {
        lt_.reset();
      }
    }
    // Context and library initialization can consume host RAM. Tighten frontend admission using
    // one post-setup sample, while Runtime keeps charging staging and backing to its original
    // ledger. Both bounds apply; the stricter frontend cap never excludes already charged bytes.
    const auto ready_system = platform::collect_system_info();
    if (!ready_system.available_memory_bytes) {
      publish();
      return sdk::make_error(XVRAM_STATUS_UNAVAILABLE, "cuda_compat", "host_budget",
                             "post-context host memory telemetry is unavailable");
    }
    snapshot_.host_available_bytes = *ready_system.available_memory_bytes;
    if (snapshot_.host_available_bytes <= headroom) {
      publish();
      return sdk::make_error(XVRAM_STATUS_HOST_OUT_OF_MEMORY, "cuda_compat", "host_budget",
                             "context setup left insufficient host headroom");
    }
    const std::uint64_t already_charged = runtime_->telemetry().host_budget_bytes;
    const std::uint64_t original_cap = runtime_->telemetry().host_store_cap_bytes;
    if (already_charged > original_cap) {
      publish();
      return sdk::make_error(XVRAM_STATUS_HOST_OUT_OF_MEMORY, "cuda_compat", "host_budget",
                             "context setup and pinned staging exhaust the raw host budget");
    }
    // The live available-RAM sample already excludes pinned memory. Add its ledger charge back
    // to the total cap; subtracting it again would reserve the same physical bytes twice.
    host_admission_cap_ = already_charged + std::min(original_cap - already_charged,
                                                     snapshot_.host_available_bytes - headroom);
    publish();
    return {};
  }

  BackendResult allocate(const std::uint64_t bytes, Allocation& output) override {
    output = {};
    const std::uint64_t charged = runtime_->telemetry().host_budget_bytes;
    if (charged > host_admission_cap_ || bytes > host_admission_cap_ - charged) {
      publish();
      return {sdk::make_error(XVRAM_STATUS_HOST_OUT_OF_MEMORY, "cuda_compat", "allocation_budget",
                              "raw allocation exceeds host backing and staging admission budget"),
              false};
    }
    residency::RuntimeAllocation allocation;
    const auto status = runtime_->allocate(bytes, residency::ResidencyHint::normal, allocation);
    if (status != residency::RuntimeStatus::success) {
      return finish(status, false);
    }
    const auto address = runtime_->allocation_address(allocation.id);
    if (!address) {
      publish();
      return {sdk::make_error(XVRAM_STATUS_INTERNAL, "cuda_compat", "allocation_address",
                              "runtime allocation did not expose a usable logical address"),
              true};
    }
    output = Allocation{allocation.id, static_cast<std::uint64_t>(*address), allocation.bytes};
    publish();
    return {};
  }

  BackendResult release(const residency::AllocationId id) override {
    return finish(runtime_->release(id), true);
  }

  BackendResult write(const residency::AllocationId id, const std::uint64_t offset,
                      const void* source, const std::uint64_t bytes) override {
    return finish(runtime_->write(id, offset, source, bytes), true);
  }

  BackendResult read(const residency::AllocationId id, const std::uint64_t offset,
                     void* destination, const std::uint64_t bytes) override {
    return finish(runtime_->read(id, offset, destination, bytes), true);
  }

  BackendResult gemm(const gemm::GemmProblem& problem) override {
    const auto validation = gemm::validate_problem(problem);
    if (!validation) {
      return {sdk::make_error(XVRAM_STATUS_INVALID_ARGUMENT, "cuda_compat", "validate_gemm",
                              gemm::problem_error_name(validation.error)),
              false};
    }
    const std::uint64_t workspace =
        std::min<std::uint64_t>(config_.workspace_cap_bytes, 4ULL * 1024ULL * 1024ULL);
    const std::uint64_t chunk = runtime_->chunk_bytes();
    const std::uint64_t target = runtime_->target_bytes();
    if (chunk == 0 || target <= config_.workspace_cap_bytes) {
      return {sdk::make_error(XVRAM_STATUS_BUDGET_PRESSURE, "cuda_compat", "make_plan",
                              "live target leaves no complete cache frame after workspace reserve"),
              false};
    }
    // Runtime reserves the session cap when sizing its frame pool, independently of the smaller
    // workspace used by this GEMM. Match SDK's frame arithmetic exactly.
    const std::uint64_t frames = ((target - config_.workspace_cap_bytes) / chunk) * chunk;
    gemm::PlannerConfig planner;
    planner.chunk_bytes = chunk;
    planner.cache_target_bytes = frames + workspace;
    planner.workspace_bytes = workspace;
    // Tall-vector GEMM needs larger K panels to keep an oversized dot product below the planner's
    // tile limit. The planner still reduces this geometry to the observed physical frame budget.
    planner.preferred_geometry = {4096, 4096,
                                  problem.m <= 16U && problem.n <= 16U ? 1'048'576U : 1024U};
    gemm::GemmPlan plan;
    try {
      plan = gemm::make_plan(problem, planner);
    } catch (...) {
      return {sdk::exception_error("make_plan"), false};
    }
    if (!plan) {
      const bool pressure = plan.error == gemm::PlanError::working_set_too_large ||
                            plan.error == gemm::PlanError::workspace_exceeds_cache;
      return {
          sdk::make_error(pressure ? XVRAM_STATUS_BUDGET_PRESSURE : XVRAM_STATUS_INVALID_ARGUMENT,
                          "cuda_compat", "make_plan", gemm::plan_error_name(plan.error)),
          false};
    }

    bool possible_submission = false;
    gemm::ExecutionHooks hooks;
    hooks.boundary = [&](const gemm::ExecutionBoundary boundary, std::size_t) {
      if (boundary == gemm::ExecutionBoundary::after_residency) {
        possible_submission = true;
      }
      return gemm::ExecutionControl::proceed;
    };
    hooks.algorithm = [&](const cublas::PreferredGemmResult& result) {
      ++snapshot_.tiles_submitted;
      if (result.path == cublas::PreferredGemmPath::cublas_lt && result.lt.algorithm_cache_hit) {
        ++snapshot_.runtime.algorithm_cache_hits;
      } else {
        ++snapshot_.runtime.algorithm_selections;
      }
    };
    hooks.progress = [&](std::size_t, std::size_t) {
      ++snapshot_.tiles_retired;
      ++progress_sequence_;
      publish();
    };
    const gemm::TiledGemmResources resources{
        *runtime_,
        {cublas_.dispatch(), handle_,
         gemm::is_cublas_lt_exact_proof_eligible(problem) ? lt_.get() : nullptr}};
    const gemm::TiledGemmRequest request{problem, plan, workspace, config_.prefetch_distance};
    const auto result = gemm::execute_tiled_gemm(resources, request, hooks);
    publish();
    if (result) {
      return {};
    }
    if (result.status == gemm::ExecutionStatus::runtime_failure) {
      return finish(result.runtime_status, possible_submission);
    }
    sdk::Error error;
    if (result.status == gemm::ExecutionStatus::cublas_failure) {
      error = sdk::make_native_error(XVRAM_STATUS_CUBLAS_ERROR, XVRAM_NATIVE_ERROR_CUBLAS,
                                     result.cublas_status, result.stage, result.operation, "cuBLAS",
                                     result.message);
    } else {
      const xvram_status status =
          result.status == gemm::ExecutionStatus::unsupported        ? XVRAM_STATUS_UNSUPPORTED
          : result.status == gemm::ExecutionStatus::unavailable      ? XVRAM_STATUS_UNAVAILABLE
          : result.status == gemm::ExecutionStatus::invalid_argument ? XVRAM_STATUS_INVALID_ARGUMENT
          : result.status == gemm::ExecutionStatus::deadline_expired ? XVRAM_STATUS_TIMEOUT
          : result.status == gemm::ExecutionStatus::cancelled        ? XVRAM_STATUS_CANCELLED
                                                                     : XVRAM_STATUS_INTERNAL;
      error = sdk::make_error(status, result.stage, result.operation, result.message);
    }
    return {std::move(error), possible_submission || runtime_->poisoned()};
  }

  BackendResult synchronize() override {
    // GEMM already retires every tile. Drain also makes all dirty host backing authoritative.
    return finish(runtime_->drain(false), true);
  }

  BackendResult close() override {
    if (closed_) {
      return close_result_;
    }
    const bool quarantine = runtime_ != nullptr && runtime_->async_completion_unknown();
    cublas::abi::Status blas_cleanup = cublas::abi::success;
    if (quarantine) {
      if (lt_) {
        lt_->abandon();
      }
      handle_ = nullptr;
      cublas_.abandon();
    } else {
      if (lt_) {
        blas_cleanup = lt_->close();
      }
      if (handle_ != nullptr) {
        const auto status = cublas_.dispatch().destroy(handle_);
        if (blas_cleanup == cublas::abi::success) {
          blas_cleanup = status;
        }
        handle_ = nullptr;
      }
    }
    lt_.reset();
    const auto status = runtime_ ? runtime_->close() : residency::RuntimeStatus::success;
    if (quarantine || (runtime_ && runtime_->async_completion_unknown())) {
      cublas_.abandon();
      cuda_.abandon();
    }
    if (blas_cleanup != cublas::abi::success) {
      close_result_ = {blas_error(blas_cleanup, "destroy_cublas", XVRAM_STATUS_CLEANUP_FAILED),
                       true};
    } else if (status != residency::RuntimeStatus::success) {
      close_result_ = {sdk::make_error(XVRAM_STATUS_CLEANUP_FAILED, "cuda_compat", "runtime_close",
                                       "runtime resources could not all be safely released"),
                       true};
    }
    closed_ = true;
    snapshot_.cleanup_completed = close_result_.error ? 0U : 1U;
    publish();
    return close_result_;
  }

private:
  BackendResult finish(const residency::RuntimeStatus status, const bool possible_mutation) {
    publish();
    if (status == residency::RuntimeStatus::success) {
      return {};
    }
    const bool failed_runtime = runtime_->poisoned() || runtime_->async_completion_unknown() ||
                                status == residency::RuntimeStatus::cleanup_failure ||
                                status == residency::RuntimeStatus::internal_failure ||
                                status == residency::RuntimeStatus::cuda_failure;
    return {runtime_error(*runtime_, status), possible_mutation || failed_runtime};
  }

  void publish() noexcept {
    if (runtime_) {
      const auto& source = runtime_->telemetry();
      auto& output = snapshot_.runtime;
      output.flags = (source.stable_addresses ? XVRAM_TELEMETRY_STABLE_VIRTUAL_ADDRESSES : 0U) |
                     (source.no_physical_aliases ? XVRAM_TELEMETRY_NO_PHYSICAL_ALIASES : 0U);
      output.cache_target_bytes = source.target_bytes;
      output.cache_target_minimum_bytes = source.target_minimum_bytes;
      output.cache_target_maximum_bytes = source.target_maximum_bytes;
      output.resident_bytes = source.resident_bytes;
      output.resident_bytes_peak = source.resident_peak_bytes;
      output.workspace_bytes = source.workspace_peak_bytes;
      output.bytes_h2d = source.h2d_bytes;
      output.bytes_d2h = source.d2h_bytes;
      output.cache_hits = source.cache_hits;
      output.cache_misses = source.cache_misses;
      output.clean_evictions = source.clean_evictions;
      output.dirty_evictions = source.dirty_evictions;
      output.writebacks_completed = source.dirty_writebacks;
      output.mappings = source.maps;
      output.unmaps = source.unmaps;
      output.set_access_calls = source.set_access;
      output.handle_reuses = source.handles_reused;
      output.unsafe_remaps = source.unsafe_remaps;
      output.unsafe_transitions = source.unsafe_transitions;
      output.budget_shrinks = source.budget_shrinks;
      output.budget_grows = source.budget_grows;
      output.target_oom_retries = source.target_oom_retries;
      output.watchdog_rejections = static_cast<std::uint32_t>(std::min<std::uint64_t>(
          source.watchdog_rejections, std::numeric_limits<std::uint32_t>::max()));
      output.physical_handles_created = source.handles_created;
      output.physical_handles_released = source.handles_released;
      output.event_boundaries = source.event_boundaries;
      output.pinned_staging_bytes = source.pinned_staging_bytes;
      output.transactions_completed = source.transactions_completed;
      output.last_transaction_ms = source.last_transaction_ms;
      output.budget_sample_count = source.budget_samples;
      output.cuda_free_bytes_minimum = source.cuda_free_minimum_bytes;
      output.cuda_free_bytes_end = source.cuda_free_end_bytes;
      output.wddm_budget_observed = source.wddm_available_end_bytes.has_value() ? 1U : 0U;
      output.wddm_available_bytes_minimum = source.wddm_available_minimum_bytes.value_or(0);
      output.wddm_available_bytes_end = source.wddm_available_end_bytes.value_or(0);
      output.poisoned = runtime_->poisoned() ? 1U : 0U;
      snapshot_.host_backing_bytes = source.host_stored_bytes;
      snapshot_.host_backing_peak_bytes = source.host_stored_peak_bytes;
      snapshot_.host_budget_bytes = source.host_budget_bytes;
      snapshot_.retired_va_reservations = source.retired_va_reservations;
      snapshot_.retired_va_reservations_freed = source.retired_va_reservations_freed;
      snapshot_.retired_va_bytes = source.retired_va_bytes;
      snapshot_.retired_va_bytes_freed = source.retired_va_bytes_freed;
      snapshot_.host_store_cap_bytes =
          host_admission_cap_ == 0 ? source.host_store_cap_bytes : host_admission_cap_;
      snapshot_.host_headroom_bytes = source.host_headroom_bytes;
      snapshot_.effective_chunk_bytes = runtime_->chunk_bytes();
      snapshot_.cleanup_operations_drained = source.cleanup_operations_drained ? 1U : 0U;
      snapshot_.cleanup_events_drained = source.cleanup_events_drained ? 1U : 0U;
      snapshot_.quarantined = source.quarantined ? 1U : 0U;
    }
    snapshot_.progress_sequence = progress_sequence_;
    if (sink_) {
      try {
        sink_(snapshot_);
      } catch (...) {
        // Observers must not change the transaction or resource lifecycle outcome.
      }
    }
  }

  cuda::CudaApi cuda_;
  cublas::CublasApi cublas_;
  std::unique_ptr<residency::Runtime> runtime_;
  std::unique_ptr<cublas::LtMatmulExecutor> lt_;
  cublas::abi::Handle handle_ = nullptr;
  xvram_session_config_v1 config_ = XVRAM_SESSION_CONFIG_V1_INIT;
  xvram_cuda_compat_telemetry_v1 snapshot_{};
  SnapshotSink sink_;
  std::uint64_t progress_sequence_ = 0;
  std::uint64_t host_admission_cap_ = 0;
  bool closed_ = false;
  BackendResult close_result_;
};

} // namespace

std::unique_ptr<Backend> make_runtime_backend() {
  return std::make_unique<RuntimeBackend>();
}

} // namespace xvram::cuda_compat
