#include "sdk/backend.hpp"
#include "sdk/deadline.hpp"

#include "gemm/executor.hpp"
#include "gemm/planner.hpp"
#include "platform/cublas/cublas_api.hpp"
#include "platform/cuda/cuda_api.hpp"
#include "residency/runtime.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace xvram::sdk {
namespace {

using Clock = std::chrono::steady_clock;

template <std::size_t Size>
[[nodiscard]] std::string_view bounded_text(const char (&text)[Size]) noexcept {
  const char* const end = std::find(text, text + Size, '\0');
  return {text, static_cast<std::size_t>(end - text)};
}

[[nodiscard]] xvram_status map_runtime_status(const residency::RuntimeStatus status) noexcept {
  switch (status) {
  case residency::RuntimeStatus::success:
    return XVRAM_STATUS_SUCCESS;
  case residency::RuntimeStatus::invalid_argument:
    return XVRAM_STATUS_INVALID_ARGUMENT;
  case residency::RuntimeStatus::unavailable:
    return XVRAM_STATUS_UNAVAILABLE;
  case residency::RuntimeStatus::unsupported:
    return XVRAM_STATUS_UNSUPPORTED;
  case residency::RuntimeStatus::host_oom:
    return XVRAM_STATUS_HOST_OUT_OF_MEMORY;
  case residency::RuntimeStatus::device_oom:
    return XVRAM_STATUS_DEVICE_OUT_OF_MEMORY;
  case residency::RuntimeStatus::budget_pressure:
    return XVRAM_STATUS_BUDGET_PRESSURE;
  case residency::RuntimeStatus::timeout:
    return XVRAM_STATUS_TIMEOUT;
  case residency::RuntimeStatus::callback_skipped:
    return XVRAM_STATUS_TIMEOUT;
  case residency::RuntimeStatus::callback_failed:
    return XVRAM_STATUS_CALLBACK_FAILED;
  case residency::RuntimeStatus::cuda_failure:
    return XVRAM_STATUS_CUDA_ERROR;
  case residency::RuntimeStatus::poisoned:
    return XVRAM_STATUS_POISONED;
  case residency::RuntimeStatus::cleanup_failure:
    return XVRAM_STATUS_CLEANUP_FAILED;
  case residency::RuntimeStatus::internal_failure:
    return XVRAM_STATUS_INTERNAL;
  }
  return XVRAM_STATUS_INTERNAL;
}

[[nodiscard]] Error runtime_error(const residency::Runtime& runtime,
                                  const residency::RuntimeStatus status) {
  const residency::RuntimeError& source = runtime.error();
  return make_native_error(map_runtime_status(status),
                           source.native_code.has_value() ? XVRAM_NATIVE_ERROR_CUDA
                                                          : XVRAM_NATIVE_ERROR_INTERNAL,
                           source.native_code.value_or(0), source.stage, source.operation,
                           source.native_code.has_value() ? "CUDA" : "", source.message);
}

[[nodiscard]] residency::AccessMode to_runtime_access(const xvram_access_mode mode) noexcept {
  if (mode == XVRAM_ACCESS_READ_WRITE) {
    return residency::AccessMode::read_write;
  }
  if (mode == XVRAM_ACCESS_WRITE_ONLY) {
    return residency::AccessMode::write_only;
  }
  return residency::AccessMode::read;
}

[[nodiscard]] gemm::MatrixLayout to_layout(const xvram_matrix_layout value) noexcept {
  return value == XVRAM_MATRIX_COLUMN_MAJOR ? gemm::MatrixLayout::column_major
                                            : gemm::MatrixLayout::row_major;
}

[[nodiscard]] gemm::MatrixOperation to_operation(const xvram_matrix_operation value) noexcept {
  return value == XVRAM_MATRIX_OP_T ? gemm::MatrixOperation::transpose
                                    : gemm::MatrixOperation::none;
}

[[nodiscard]] gemm::ElementType to_element_type(const xvram_data_type value) noexcept {
  switch (value) {
  case XVRAM_DATA_FP16:
    return gemm::ElementType::fp16;
  case XVRAM_DATA_BF16:
    return gemm::ElementType::bf16;
  case XVRAM_DATA_FP64:
    return gemm::ElementType::fp64;
  case XVRAM_DATA_FP32:
  default:
    return gemm::ElementType::fp32;
  }
}

[[nodiscard]] constexpr gemm::ComputeMode to_compute_mode(const xvram_compute_mode value,
                                                          const xvram_data_type type) noexcept {
  if (value == XVRAM_COMPUTE_AUTO) {
    if (type == XVRAM_DATA_FP64) {
      return gemm::ComputeMode::fp64;
    }
    if (type == XVRAM_DATA_FP16 || type == XVRAM_DATA_BF16) {
      return gemm::ComputeMode::strict_fp32;
    }
    return gemm::ComputeMode::fast_tf32;
  }
  if (value == XVRAM_COMPUTE_FP64) {
    return gemm::ComputeMode::fp64;
  }
  return value == XVRAM_COMPUTE_FP32_STRICT ? gemm::ComputeMode::strict_fp32
                                            : gemm::ComputeMode::fast_tf32;
}

static_assert(to_compute_mode(XVRAM_COMPUTE_AUTO, XVRAM_DATA_FP16) ==
              gemm::ComputeMode::strict_fp32);
static_assert(to_compute_mode(XVRAM_COMPUTE_AUTO, XVRAM_DATA_BF16) ==
              gemm::ComputeMode::strict_fp32);
static_assert(to_compute_mode(XVRAM_COMPUTE_AUTO, XVRAM_DATA_FP32) == gemm::ComputeMode::fast_tf32);
static_assert(to_compute_mode(XVRAM_COMPUTE_AUTO, XVRAM_DATA_FP64) == gemm::ComputeMode::fp64);

class RuntimeSession;

class RuntimeOperation final : public BackendOperation {
public:
  explicit RuntimeOperation(const std::uint64_t operation_id, const std::uint64_t units_total = 1) {
    (void)SubmissionDeadline::create(0, Clock::now(), deadline_);
    initialize(operation_id, units_total);
  }

  RuntimeOperation(const std::uint64_t operation_id, const std::uint64_t units_total,
                   SubmissionDeadline deadline)
      : deadline_(std::move(deadline)) {
    initialize(operation_id, units_total);
  }

private:
  void initialize(const std::uint64_t operation_id, const std::uint64_t units_total) noexcept {
    info_ = XVRAM_OPERATION_INFO_V1_INIT;
    info_.operation_id = operation_id;
    info_.units_total = units_total;
  }

public:
  [[nodiscard]] Error poll(xvram_operation_info_v1& output) override {
    const std::scoped_lock lock(mutex_);
    output = info_;
    return {};
  }

  [[nodiscard]] Error wait(const std::uint64_t timeout_ms,
                           xvram_operation_info_v1& output) override {
    std::unique_lock lock(mutex_);
    const auto terminal = [&]() {
      return info_.state == XVRAM_OPERATION_COMPLETED || info_.state == XVRAM_OPERATION_CANCELLED ||
             info_.state == XVRAM_OPERATION_FAILED;
    };
    bool completed = true;
    if (!terminal()) {
      if (timeout_ms == XVRAM_TIMEOUT_INFINITE) {
        condition_.wait(lock, terminal);
      } else {
        completed = condition_.wait_for(lock, std::chrono::milliseconds(timeout_ms), terminal);
      }
    }
    output = info_;
    return completed ? Error{}
                     : make_error(XVRAM_STATUS_TIMEOUT, "operation", "wait",
                                  "operation did not finish before the timeout");
  }

  [[nodiscard]] Error cancel() override {
    const std::scoped_lock lock(mutex_);
    if (info_.state == XVRAM_OPERATION_QUEUED) {
      cancel_requested_ = true;
      return {};
    }
    if (info_.state == XVRAM_OPERATION_RUNNING) {
      return make_error(XVRAM_STATUS_BUSY, "operation", "cancel",
                        "a running CUDA operation cannot be cancelled safely");
    }
    return {};
  }

  [[nodiscard]] Error error() const override {
    const std::scoped_lock lock(mutex_);
    return error_;
  }

  [[nodiscard]] bool begin() {
    const std::scoped_lock lock(mutex_);
    if (cancel_requested_) {
      info_.state = XVRAM_OPERATION_CANCELLED;
      info_.result = XVRAM_STATUS_CANCELLED;
      error_ = make_error(XVRAM_STATUS_CANCELLED, "operation", "cancel", "operation cancelled");
      condition_.notify_all();
      return false;
    }
    const Clock::time_point now = Clock::now();
    if (deadline_.expired(now)) {
      info_.state = XVRAM_OPERATION_FAILED;
      info_.result = XVRAM_STATUS_TIMEOUT;
      error_ = deadline_.timeout_error("operation", "queue_deadline");
      info_.elapsed_ms =
          std::chrono::duration<double, std::milli>(now - deadline_.submitted_at()).count();
      condition_.notify_all();
      return false;
    }
    info_.state = XVRAM_OPERATION_RUNNING;
    started_ = now;
    return true;
  }

  [[nodiscard]] Error deadline_error(const char* stage, const char* operation) const {
    return deadline_.expired() ? deadline_.timeout_error(stage, operation) : Error{};
  }

  void finish(const Error& error, const residency::RuntimeTelemetry& before,
              const residency::RuntimeTelemetry& after,
              const std::uint64_t completed_units = 1) noexcept {
    const std::scoped_lock lock(mutex_);
    error_ = error;
    info_.result = error.status;
    info_.state = error ? XVRAM_OPERATION_FAILED : XVRAM_OPERATION_COMPLETED;
    if (!error) {
      info_.units_completed = std::max(completed_units, info_.units_total);
    }
    info_.units_total = std::max(info_.units_total, completed_units);
    info_.elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now() - started_).count();
    info_.bytes_h2d = after.h2d_bytes - before.h2d_bytes;
    info_.bytes_d2h = after.d2h_bytes - before.d2h_bytes;
    info_.cache_hits = after.cache_hits - before.cache_hits;
    info_.cache_misses = after.cache_misses - before.cache_misses;
    condition_.notify_all();
  }

  void advance_unit() noexcept {
    const std::scoped_lock lock(mutex_);
    if (info_.state == XVRAM_OPERATION_RUNNING && info_.units_completed < info_.units_total) {
      ++info_.units_completed;
    }
  }

  [[nodiscard]] std::uint64_t operation_id() const noexcept {
    const std::scoped_lock lock(mutex_);
    return info_.operation_id;
  }

private:
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  xvram_operation_info_v1 info_{};
  Error error_;
  SubmissionDeadline deadline_;
  Clock::time_point started_{};
  bool cancel_requested_ = false;
};

class RuntimeAllocation final : public BackendAllocation {
public:
  RuntimeAllocation(std::weak_ptr<RuntimeSession> owner, const residency::RuntimeAllocation value,
                    const xvram_allocation_desc_v1 desc)
      : owner_(std::move(owner)), value_(value), desc_(desc) {}

  [[nodiscard]] Error info(xvram_allocation_info_v1& output) const override;
  [[nodiscard]] Error write(std::uint64_t offset_bytes, const void* source,
                            std::uint64_t size_bytes) override;
  [[nodiscard]] Error read(std::uint64_t offset_bytes, void* destination,
                           std::uint64_t size_bytes) override;
  [[nodiscard]] Error release() override;

  [[nodiscard]] residency::AllocationId id() const noexcept {
    return value_.id;
  }
  [[nodiscard]] std::uint64_t size() const noexcept {
    return value_.bytes;
  }
  [[nodiscard]] bool released() const noexcept {
    return released_.load(std::memory_order_acquire);
  }

private:
  std::weak_ptr<RuntimeSession> owner_;
  residency::RuntimeAllocation value_;
  xvram_allocation_desc_v1 desc_{};
  std::atomic<bool> released_{false};
};

struct WorkItem {
  std::shared_ptr<RuntimeOperation> operation;
  std::function<Error(RuntimeOperation&)> action;
  bool shutdown_after = false;
};

class RuntimeGemmPlan;

class RuntimeSession final : public BackendSession,
                             public std::enable_shared_from_this<RuntimeSession> {
public:
  RuntimeSession(xvram_session_config_v1 config, cuda::abi::Context attached_context)
      : config_(config), runtime_config_(make_runtime_config(config, attached_context)),
        runtime_(cuda_, runtime_config_),
        close_operation_(std::make_shared<RuntimeOperation>(0, 1)) {}

  ~RuntimeSession() override {
    (void)close(XVRAM_TIMEOUT_INFINITE);
  }

  [[nodiscard]] Error start() {
    try {
      worker_ = std::jthread([this]() { worker_main(); });
    } catch (...) {
      const Error error = exception_error("worker_start");
      {
        const std::scoped_lock lock(queue_mutex_);
        setup_error_ = error;
        setup_finished_ = true;
        worker_exited_ = true;
      }
      return error;
    }
    std::unique_lock lock(queue_mutex_);
    setup_condition_.wait(lock, [&]() { return setup_finished_; });
    return setup_error_;
  }

  [[nodiscard]] Error allocate(const xvram_allocation_desc_v1& desc,
                               std::shared_ptr<BackendAllocation>& output) override {
    std::shared_ptr<BackendAllocation> created;
    const Error error = invoke_sync([&]() -> Error {
      const std::uint64_t effective_alignment = runtime_.chunk_bytes();
      if (desc.alignment_bytes > effective_alignment) {
        return make_error(XVRAM_STATUS_INVALID_ARGUMENT, "allocation", "alignment",
                          "allocation alignment exceeds the runtime chunk alignment");
      }
      residency::RuntimeAllocation value;
      const residency::RuntimeStatus status = runtime_.allocate(
          desc.size_bytes,
          desc.priority == XVRAM_ALLOCATION_HOT
              ? residency::ResidencyHint::hot
              : (desc.priority == XVRAM_ALLOCATION_STREAMING ? residency::ResidencyHint::streaming
                                                             : residency::ResidencyHint::normal),
          value);
      if (status != residency::RuntimeStatus::success) {
        return runtime_error(runtime_, status);
      }
      xvram_allocation_desc_v1 normalized = desc;
      if (normalized.alignment_bytes == 0) {
        normalized.alignment_bytes = effective_alignment;
      }
      try {
        created = std::make_shared<RuntimeAllocation>(shared_from_this(), value, normalized);
      } catch (...) {
        Error handle_error = exception_error("allocation_handle");
        const residency::RuntimeStatus rollback = runtime_.release(value.id);
        return rollback == residency::RuntimeStatus::success ? handle_error
                                                             : runtime_error(runtime_, rollback);
      }
      return {};
    });
    if (error) {
      return error;
    }
    output = std::move(created);
    return {};
  }

  [[nodiscard]] Error prefetch(const PrefetchRequest& request,
                               std::shared_ptr<BackendOperation>& output) override {
    std::vector<residency::AccessRange> ranges;
    if (Error error = convert_ranges(request.ranges, ranges, "prefetch"); error) {
      return error;
    }
    std::shared_ptr<RuntimeOperation> operation;
    if (Error error = enqueue(
            [this, ranges = std::move(ranges)]() {
              const residency::RuntimeStatus status = runtime_.prefetch(ranges);
              return status == residency::RuntimeStatus::success ? Error{}
                                                                 : runtime_error(runtime_, status);
            },
            operation, request.deadline_ms);
        error) {
      return error;
    }
    output = std::move(operation);
    return {};
  }

  [[nodiscard]] Error submit_transaction(const TransactionRequest& request,
                                         std::shared_ptr<BackendOperation>& output) override {
    if (request.workspace_bytes > config_.workspace_cap_bytes) {
      return make_error(XVRAM_STATUS_INVALID_ARGUMENT, "transaction", "workspace",
                        "transaction workspace exceeds the session workspace cap");
    }
    std::vector<residency::AccessRange> ranges;
    if (Error error = convert_ranges(request.ranges, ranges, "transaction"); error) {
      return error;
    }
    const std::uint64_t transaction_id = next_transaction_id_.fetch_add(1);
    const auto callback = request.callback;
    void* const user_data = request.user_data;
    const std::uint64_t workspace_bytes = request.workspace_bytes;
    std::shared_ptr<RuntimeOperation> operation;
    if (Error error = enqueue_progress(
            [this, ranges = std::move(ranges), callback, user_data, transaction_id,
             workspace_bytes](RuntimeOperation& progress) {
              if (Error deadline_error =
                      progress.deadline_error("operation", "pre_launch_deadline");
                  deadline_error) {
                return deadline_error;
              }
              Error callback_error;
              const residency::RuntimeStatus status = runtime_.execute(
                  ranges, workspace_bytes, [&](const residency::TransactionContext& context) {
                    if (Error deadline_error =
                            progress.deadline_error("operation", "post_residency_deadline");
                        deadline_error) {
                      callback_error = std::move(deadline_error);
                      return residency::RuntimeStatus::callback_skipped;
                    }
                    std::vector<xvram_resolved_range_v1> resolved;
                    resolved.reserve(context.ranges.size());
                    for (const residency::ResolvedRange& range : context.ranges) {
                      resolved.push_back(xvram_resolved_range_v1{
                          range.allocation_id.value, range.offset_bytes, range.length_bytes,
                          static_cast<std::uint64_t>(range.device_address),
                          static_cast<xvram_access_mode>(
                              range.mode == residency::AccessMode::read
                                  ? XVRAM_ACCESS_READ
                                  : (range.mode == residency::AccessMode::read_write
                                         ? XVRAM_ACCESS_READ_WRITE
                                         : XVRAM_ACCESS_WRITE_ONLY)),
                          0U});
                    }
                    xvram_transaction_context_v1 public_context{};
                    public_context.struct_size = sizeof(public_context);
                    public_context.operation_id = progress.operation_id();
                    public_context.transaction_id = transaction_id;
                    public_context.ranges = resolved.data();
                    public_context.range_count = resolved.size();
                    public_context.workspace_device_address =
                        static_cast<std::uint64_t>(context.workspace_address);
                    public_context.workspace_bytes = context.workspace_bytes;
                    public_context.native_stream = static_cast<std::uint64_t>(
                        reinterpret_cast<std::uintptr_t>(context.stream));
                    xvram_error_info_v1 native_error = XVRAM_ERROR_INFO_V1_INIT;
                    const xvram_status callback_status =
                        callback(&public_context, user_data, &native_error);
                    if (callback_status != XVRAM_STATUS_SUCCESS) {
                      const std::string_view message = bounded_text(native_error.message);
                      callback_error = make_native_error(
                          callback_status, XVRAM_NATIVE_ERROR_CALLBACK, native_error.native_code,
                          bounded_text(native_error.stage), bounded_text(native_error.operation),
                          bounded_text(native_error.native_name),
                          message.empty() ? std::string_view{"transaction callback failed"}
                                          : message);
                      return residency::RuntimeStatus::callback_failed;
                    }
                    return residency::RuntimeStatus::success;
                  });
              if (callback_error) {
                return callback_error;
              }
              return status == residency::RuntimeStatus::success ? Error{}
                                                                 : runtime_error(runtime_, status);
            },
            operation, 1, request.deadline_ms);
        error) {
      return error;
    }
    output = std::move(operation);
    return {};
  }

  [[nodiscard]] Error create_gemm_plan(const GemmRequest& request,
                                       std::shared_ptr<BackendGemmPlan>& output) override;

  [[nodiscard]] Error telemetry(xvram_session_telemetry_v1& output) const override {
    const std::scoped_lock lock(telemetry_mutex_);
    const residency::RuntimeTelemetry& source = telemetry_snapshot_;
    output.flags = 0;
    if (source.stable_addresses) {
      output.flags |= XVRAM_TELEMETRY_STABLE_VIRTUAL_ADDRESSES;
    }
    if (source.no_physical_aliases) {
      output.flags |= XVRAM_TELEMETRY_NO_PHYSICAL_ALIASES;
    }
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
    output.algorithm_selections = algorithm_selections_;
    output.algorithm_cache_hits = algorithm_cache_hits_;
    output.last_transaction_ms = source.last_transaction_ms;
    output.budget_sample_count = source.budget_samples;
    output.cuda_free_bytes_minimum = source.cuda_free_minimum_bytes;
    output.cuda_free_bytes_end = source.cuda_free_end_bytes;
    output.wddm_budget_observed = source.wddm_available_end_bytes.has_value() ? 1U : 0U;
    output.wddm_available_bytes_minimum = source.wddm_available_minimum_bytes.value_or(0);
    output.wddm_available_bytes_end = source.wddm_available_end_bytes.value_or(0);
    output.cuda_driver_version = cuda_driver_version_;
    output.cublas_version = cublas_version_;
    // Availability describes the successfully initialized capability for this session. It must
    // remain observable in the final telemetry snapshot after close destroys the live handle.
    output.cublas_lt_available = cublas_lt_available_ ? 1U : 0U;
    output.cublas_lt_version = cublas_lt_version_;
    output.cublas_library_source = cublas_library_source_;
    output.poisoned = source.quarantined ? 1U : 0U;
    return {};
  }

  [[nodiscard]] Error drain(const std::uint64_t timeout_ms) override {
    return invoke_sync(
        [this]() {
          const residency::RuntimeStatus status = runtime_.drain(false);
          return status == residency::RuntimeStatus::success ? Error{}
                                                             : runtime_error(runtime_, status);
        },
        timeout_ms);
  }

  [[nodiscard]] Error close(const std::uint64_t timeout_ms) override {
    const Clock::time_point started = Clock::now();
    std::shared_ptr<RuntimeOperation> close_operation;
    Error setup_error;
    {
      const std::scoped_lock lock(queue_mutex_);
      if (!closing_) {
        if (!setup_error_) {
          close_requested_ = true;
        }
        closing_ = true;
      }
      close_operation = close_operation_;
      setup_error = setup_error_;
    }
    queue_condition_.notify_all();

    if (setup_error) {
      if (!wait_for_worker_exit(remaining_timeout(started, timeout_ms))) {
        return make_error(XVRAM_STATUS_TIMEOUT, "session", "close",
                          "session setup cleanup did not finish before the timeout");
      }
      join_worker_once();
      return setup_error;
    }
    if (!close_operation) {
      return make_error(XVRAM_STATUS_INTERNAL, "session", "close",
                        "session close operation is unavailable");
    }

    xvram_operation_info_v1 info = XVRAM_OPERATION_INFO_V1_INIT;
    if (Error wait_error = close_operation->wait(remaining_timeout(started, timeout_ms), info);
        wait_error) {
      return wait_error;
    }
    const Error result = close_operation->error();
    if (!wait_for_worker_exit(remaining_timeout(started, timeout_ms))) {
      return make_error(XVRAM_STATUS_TIMEOUT, "session", "close",
                        "worker did not retire after cleanup before the timeout");
    }
    join_worker_once();
    return result;
  }

  [[nodiscard]] Error write(const residency::AllocationId id, const std::uint64_t offset,
                            const void* source, const std::uint64_t bytes) {
    return invoke_sync([this, id, offset, source, bytes]() {
      const residency::RuntimeStatus status = runtime_.write(id, offset, source, bytes);
      return status == residency::RuntimeStatus::success ? Error{}
                                                         : runtime_error(runtime_, status);
    });
  }

  [[nodiscard]] Error read(const residency::AllocationId id, const std::uint64_t offset,
                           void* destination, const std::uint64_t bytes) {
    return invoke_sync([this, id, offset, destination, bytes]() {
      const residency::RuntimeStatus status = runtime_.read(id, offset, destination, bytes);
      return status == residency::RuntimeStatus::success ? Error{}
                                                         : runtime_error(runtime_, status);
    });
  }

  [[nodiscard]] Error release_allocation(const residency::AllocationId id) {
    return invoke_sync([this, id]() {
      const residency::RuntimeStatus status = runtime_.release(id);
      return status == residency::RuntimeStatus::success ? Error{}
                                                         : runtime_error(runtime_, status);
    });
  }

  [[nodiscard]] Error enqueue(std::function<Error()> action,
                              std::shared_ptr<RuntimeOperation>& output,
                              const std::uint64_t deadline_ms = 0) {
    return enqueue_progress(
        [action = std::move(action)](RuntimeOperation& operation) mutable {
          if (Error deadline_error = operation.deadline_error("operation", "pre_launch_deadline");
              deadline_error) {
            return deadline_error;
          }
          return action();
        },
        output, 1, deadline_ms);
  }

  [[nodiscard]] Error enqueue_progress(std::function<Error(RuntimeOperation&)> action,
                                       std::shared_ptr<RuntimeOperation>& output,
                                       const std::uint64_t units_total,
                                       const std::uint64_t deadline_ms = 0) {
    SubmissionDeadline deadline;
    if (!SubmissionDeadline::create(deadline_ms, Clock::now(), deadline)) {
      return make_error(XVRAM_STATUS_INVALID_ARGUMENT, "deadline", "submit",
                        "deadline_ms cannot be represented by the monotonic clock");
    }
    auto operation = std::make_shared<RuntimeOperation>(next_operation_id_.fetch_add(1),
                                                        units_total, std::move(deadline));
    {
      const std::scoped_lock lock(queue_mutex_);
      if (closing_) {
        return make_error(XVRAM_STATUS_CLOSED, "session", "submit", "session is closing");
      }
      queue_.push_back(WorkItem{operation, std::move(action), false});
    }
    queue_condition_.notify_one();
    output = std::move(operation);
    return {};
  }

  [[nodiscard]] residency::Runtime& runtime() noexcept {
    return runtime_;
  }
  [[nodiscard]] const xvram_session_config_v1& config() const noexcept {
    return config_;
  }
  [[nodiscard]] std::uint64_t chunk_size_bytes() const noexcept override {
    return runtime_.chunk_bytes();
  }
  [[nodiscard]] const cublas::CublasDispatch& cublas_dispatch() const noexcept {
    return cublas_.dispatch();
  }
  [[nodiscard]] cublas::abi::Handle cublas_handle() const noexcept {
    return cublas_handle_;
  }
  [[nodiscard]] bool cublas_ready() const noexcept {
    return cublas_ready_;
  }
  [[nodiscard]] bool cublas_lt_ready() const noexcept {
    return cublas_lt_ != nullptr && cublas_lt_->ready();
  }
  [[nodiscard]] cublas::LtMatmulExecutor* cublas_lt() noexcept {
    return cublas_lt_.get();
  }
  void record_algorithm(const cublas::PreferredGemmResult& result) noexcept {
    if (!result) {
      return;
    }
    const std::scoped_lock lock(telemetry_mutex_);
    if (result.path == cublas::PreferredGemmPath::cublas_core) {
      ++algorithm_selections_;
    } else if (result.lt.algorithm_cache_hit) {
      ++algorithm_cache_hits_;
    } else {
      ++algorithm_selections_;
    }
  }

private:
  [[nodiscard]] static residency::RuntimeConfig
  make_runtime_config(const xvram_session_config_v1& config,
                      const cuda::abi::Context attached_context) {
    residency::RuntimeConfig output;
    output.device_ordinal = config.device_ordinal;
    output.context_mode = config.context_mode == XVRAM_CONTEXT_ATTACH_CURRENT
                              ? residency::RuntimeContextMode::attach_current
                              : residency::RuntimeContextMode::isolated;
    output.attached_context = attached_context;
    output.chunk_bytes = config.chunk_size_bytes;
    output.cache_target_bytes = config.cache_target_bytes;
    output.device_headroom_bytes = config.device_headroom_bytes;
    output.workspace_reserve_bytes = config.workspace_cap_bytes;
    output.staging_slots = config.staging_slots;
    output.budget_poll_interval = std::chrono::milliseconds(config.budget_poll_ms);
    output.policy = config.cache_policy == XVRAM_CACHE_POLICY_LRU ? residency::RuntimePolicy::lru
                                                                  : residency::RuntimePolicy::clock;
    output.maximum_transaction_duration =
        config.max_transaction_ms == 0 ? std::chrono::milliseconds(250)
                                       : std::chrono::milliseconds(config.max_transaction_ms);
    return output;
  }

  [[nodiscard]] Error convert_ranges(const std::vector<AccessRequest>& input,
                                     std::vector<residency::AccessRange>& output,
                                     const char* operation) const {
    output.reserve(input.size());
    for (const AccessRequest& request : input) {
      const auto allocation = std::dynamic_pointer_cast<RuntimeAllocation>(request.allocation);
      if (!allocation || allocation->released()) {
        return make_error(XVRAM_STATUS_INVALID_ARGUMENT, "transaction", operation,
                          "allocation is released or belongs to another backend");
      }
      output.push_back(residency::AccessRange{allocation->id(), request.offset_bytes,
                                              request.length_bytes,
                                              to_runtime_access(request.mode)});
    }
    return {};
  }

  [[nodiscard]] Error invoke_sync(std::function<Error()> action,
                                  const std::uint64_t timeout_ms = XVRAM_TIMEOUT_INFINITE) {
    std::shared_ptr<RuntimeOperation> operation;
    if (Error enqueue_error = enqueue(std::move(action), operation); enqueue_error) {
      return enqueue_error;
    }
    xvram_operation_info_v1 info = XVRAM_OPERATION_INFO_V1_INIT;
    if (Error wait_error = operation->wait(timeout_ms, info); wait_error) {
      return wait_error;
    }
    return operation->error();
  }

  void worker_main() noexcept {
    try {
      const residency::RuntimeStatus setup_status = runtime_.setup();
      if (setup_status != residency::RuntimeStatus::success) {
        setup_error_ = runtime_error(runtime_, setup_status);
      } else {
        initialize_cublas();
      }
    } catch (...) {
      setup_error_ = exception_error("worker_setup");
    }
    {
      const std::scoped_lock lock(queue_mutex_);
      setup_finished_ = true;
      refresh_telemetry();
    }
    setup_condition_.notify_all();
    if (setup_error_) {
      (void)runtime_.close();
      refresh_telemetry();
      signal_worker_exit();
      return;
    }

    for (;;) {
      WorkItem item;
      {
        std::unique_lock lock(queue_mutex_);
        queue_condition_.wait(
            lock, [&]() { return stop_worker_ || close_requested_ || !queue_.empty(); });
        if (stop_worker_ && queue_.empty()) {
          break;
        }
        if (!queue_.empty()) {
          item = std::move(queue_.front());
          queue_.pop_front();
        } else {
          item = WorkItem{close_operation_, {}, true};
          close_requested_ = false;
        }
      }
      if (!item.operation->begin()) {
        continue;
      }
      const residency::RuntimeTelemetry before = runtime_.telemetry();
      Error result;
      try {
        if (item.shutdown_after) {
          const bool quarantine = runtime_.async_completion_unknown();
          const Error cublas_cleanup_error = quarantine ? abandon_cublas() : destroy_cublas();
          const residency::RuntimeStatus status = runtime_.close();
          if (quarantine || runtime_.async_completion_unknown()) {
            cuda_.abandon();
          }
          result = cublas_cleanup_error ? cublas_cleanup_error
                                        : (status == residency::RuntimeStatus::success
                                               ? Error{}
                                               : runtime_error(runtime_, status));
        } else {
          result = item.action(*item.operation);
        }
      } catch (...) {
        result = exception_error("worker_action");
      }
      const residency::RuntimeTelemetry after = runtime_.telemetry();
      refresh_telemetry(after);
      item.operation->finish(result, before, after);
      if (item.shutdown_after) {
        const std::scoped_lock lock(queue_mutex_);
        stop_worker_ = true;
      }
    }
    signal_worker_exit();
  }

  void initialize_cublas() noexcept {
    try {
      int driver_version = 0;
      if (cuda_.driver_get_version_ != nullptr &&
          cuda_.driver_get_version_(&driver_version) == cuda::abi::success && driver_version > 0) {
        cuda_driver_version_ = static_cast<std::uint32_t>(driver_version);
      }
      const cublas::CublasLoadResult load = cublas_.load();
      if (load.status != cublas::CublasLoadStatus::loaded) {
        return;
      }
      const cublas::CublasDispatch& dispatch = cublas_.dispatch();
      if (dispatch.create(&cublas_handle_) != cublas::abi::success) {
        cublas_handle_ = nullptr;
        return;
      }
      int version = 0;
      if (dispatch.get_version(cublas_handle_, &version) == cublas::abi::success && version > 0) {
        cublas_version_ = static_cast<std::uint32_t>(version);
      }
      cublas_ready_ = true;
      switch (cublas_.library_source()) {
      case cublas::CublasLibrarySource::system:
        cublas_library_source_ = XVRAM_CUBLAS_SOURCE_SYSTEM;
        break;
      case cublas::CublasLibrarySource::app_local:
        cublas_library_source_ = XVRAM_CUBLAS_SOURCE_APP_LOCAL;
        break;
      case cublas::CublasLibrarySource::explicit_path:
        cublas_library_source_ = XVRAM_CUBLAS_SOURCE_EXPLICIT;
        break;
      case cublas::CublasLibrarySource::unknown:
        cublas_library_source_ = XVRAM_CUBLAS_SOURCE_UNKNOWN;
        break;
      }
      if (!cublas_.has_lt()) {
        return;
      }
      cublas_lt_ = std::make_unique<cublas::LtMatmulExecutor>(dispatch);
      if (cublas_lt_->initialize() != cublas::abi::success) {
        cublas_lt_.reset();
      } else {
        cublas_lt_available_ = true;
        if (cublas_lt_->library_version() <=
            static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
          cublas_lt_version_ = static_cast<std::uint32_t>(cublas_lt_->library_version());
        }
      }
    } catch (const std::bad_alloc&) {
      (void)destroy_cublas();
    } catch (...) {
      (void)destroy_cublas();
    }
  }

  [[nodiscard]] Error destroy_cublas() noexcept {
    cublas::abi::Status cleanup_status = cublas::abi::success;
    const char* cleanup_operation = "";
    if (cublas_lt_ != nullptr) {
      const cublas::abi::Status status = cublas_lt_->close();
      if (status != cublas::abi::success) {
        cleanup_status = status;
        cleanup_operation = "cublasLtDestroy";
      }
      cublas_lt_.reset();
    }
    if (cublas_handle_ != nullptr && cublas_.dispatch().destroy != nullptr) {
      const cublas::abi::Status status = cublas_.dispatch().destroy(cublas_handle_);
      if (status != cublas::abi::success && cleanup_status == cublas::abi::success) {
        cleanup_status = status;
        cleanup_operation = "cublasDestroy_v2";
      }
    }
    cublas_handle_ = nullptr;
    cublas_ready_ = false;
    return cleanup_status == cublas::abi::success
               ? Error{}
               : make_native_error(XVRAM_STATUS_CLEANUP_FAILED, XVRAM_NATIVE_ERROR_CUBLAS,
                                   cleanup_status, "cleanup", cleanup_operation, "cuBLAS",
                                   "cuBLAS handle destruction failed");
  }

  [[nodiscard]] Error abandon_cublas() noexcept {
    if (cublas_lt_ != nullptr) {
      cublas_lt_->abandon();
      cublas_lt_.reset();
    }
    cublas_handle_ = nullptr;
    cublas_ready_ = false;
    cublas_.abandon();
    return {};
  }

  void refresh_telemetry() noexcept {
    refresh_telemetry(runtime_.telemetry());
  }

  void refresh_telemetry(const residency::RuntimeTelemetry& source) noexcept {
    const std::scoped_lock lock(telemetry_mutex_);
    telemetry_snapshot_ = source;
  }

  [[nodiscard]] static std::uint64_t remaining_timeout(const Clock::time_point started,
                                                       const std::uint64_t timeout_ms) noexcept {
    if (timeout_ms == XVRAM_TIMEOUT_INFINITE) {
      return XVRAM_TIMEOUT_INFINITE;
    }
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - started);
    if (elapsed.count() < 0) {
      return timeout_ms;
    }
    const auto elapsed_ms = static_cast<std::uint64_t>(elapsed.count());
    return elapsed_ms >= timeout_ms ? 0U : timeout_ms - elapsed_ms;
  }

  [[nodiscard]] bool wait_for_worker_exit(const std::uint64_t timeout_ms) {
    std::unique_lock lock(queue_mutex_);
    const auto exited = [this]() { return worker_exited_; };
    if (worker_exited_) {
      return true;
    }
    if (timeout_ms == XVRAM_TIMEOUT_INFINITE) {
      worker_exit_condition_.wait(lock, exited);
      return true;
    }
    return worker_exit_condition_.wait_for(lock, std::chrono::milliseconds(timeout_ms), exited);
  }

  void signal_worker_exit() noexcept {
    {
      const std::scoped_lock lock(queue_mutex_);
      worker_exited_ = true;
    }
    worker_exit_condition_.notify_all();
  }

  void join_worker_once() noexcept {
    bool claim_join = false;
    {
      const std::scoped_lock lock(queue_mutex_);
      if (worker_exited_ && !worker_join_claimed_) {
        worker_join_claimed_ = true;
        claim_join = true;
      }
    }
    if (claim_join && worker_.joinable()) {
      worker_.join();
    }
  }

  xvram_session_config_v1 config_{};
  residency::RuntimeConfig runtime_config_;
  cuda::CudaApi cuda_;
  residency::Runtime runtime_;
  cublas::CublasApi cublas_;
  cublas::abi::Handle cublas_handle_ = nullptr;
  std::unique_ptr<cublas::LtMatmulExecutor> cublas_lt_;
  bool cublas_ready_ = false;
  bool cublas_lt_available_ = false;
  std::uint32_t cublas_version_ = 0;
  std::uint32_t cuda_driver_version_ = 0;
  std::uint32_t cublas_lt_version_ = 0;
  std::uint32_t cublas_library_source_ = XVRAM_CUBLAS_SOURCE_UNKNOWN;
  std::uint64_t algorithm_selections_ = 0;
  std::uint64_t algorithm_cache_hits_ = 0;
  std::jthread worker_;
  mutable std::mutex queue_mutex_;
  std::condition_variable queue_condition_;
  std::condition_variable setup_condition_;
  std::condition_variable worker_exit_condition_;
  std::deque<WorkItem> queue_;
  bool setup_finished_ = false;
  bool stop_worker_ = false;
  bool worker_exited_ = false;
  bool worker_join_claimed_ = false;
  Error setup_error_;
  bool closing_ = false;
  bool close_requested_ = false;
  std::shared_ptr<RuntimeOperation> close_operation_;
  std::atomic<std::uint64_t> next_operation_id_{1};
  std::atomic<std::uint64_t> next_transaction_id_{1};
  mutable std::mutex telemetry_mutex_;
  residency::RuntimeTelemetry telemetry_snapshot_;
};

Error RuntimeAllocation::info(xvram_allocation_info_v1& output) const {
  if (released()) {
    return make_error(XVRAM_STATUS_CLOSED, "allocation", "info", "allocation is released");
  }
  output.allocation_id = value_.id.value;
  output.size_bytes = value_.bytes;
  output.alignment_bytes = desc_.alignment_bytes;
  output.priority = desc_.priority;
  output.stable_virtual_address = 1U;
  return {};
}

Error RuntimeAllocation::write(const std::uint64_t offset, const void* source,
                               const std::uint64_t bytes) {
  const auto owner = owner_.lock();
  if (!owner || released()) {
    return make_error(XVRAM_STATUS_CLOSED, "allocation", "write", "allocation is released");
  }
  if (bytes == 0) {
    return {};
  }
  return owner->write(value_.id, offset, source, bytes);
}

Error RuntimeAllocation::read(const std::uint64_t offset, void* destination,
                              const std::uint64_t bytes) {
  const auto owner = owner_.lock();
  if (!owner || released()) {
    return make_error(XVRAM_STATUS_CLOSED, "allocation", "read", "allocation is released");
  }
  if (bytes == 0) {
    return {};
  }
  return owner->read(value_.id, offset, destination, bytes);
}

Error RuntimeAllocation::release() {
  const auto owner = owner_.lock();
  bool expected = false;
  if (!released_.compare_exchange_strong(expected, true)) {
    return {};
  }
  // Runtime::close releases every logical allocation. A surviving public allocation handle still
  // needs an idempotent logical release after its session handle has been closed or destroyed.
  if (!owner) {
    return {};
  }
  Error error = owner->release_allocation(value_.id);
  if (error.status == XVRAM_STATUS_CLOSED) {
    return {};
  }
  if (error) {
    released_.store(false, std::memory_order_release);
  }
  return error;
}

class RuntimeGemmPlan final : public BackendGemmPlan {
public:
  RuntimeGemmPlan(std::weak_ptr<RuntimeSession> owner, GemmRequest request,
                  gemm::GemmProblem problem, gemm::GemmPlan plan,
                  const xvram_compute_mode effective_mode, const bool uses_cublas_lt)
      : owner_(std::move(owner)), request_(std::move(request)), problem_(std::move(problem)),
        plan_(std::move(plan)), effective_mode_(effective_mode), uses_cublas_lt_(uses_cublas_lt) {}

  [[nodiscard]] Error info(xvram_gemm_plan_info_v1& output) const override {
    output.m = problem_.m;
    output.n = problem_.n;
    output.k = problem_.k;
    output.tile_m = plan_.geometry.m;
    output.tile_n = plan_.geometry.n;
    output.tile_k = plan_.geometry.k;
    output.maximum_working_set_bytes = plan_.maximum_resident_bytes;
    output.workspace_bytes =
        plan_.cache_bytes_available_to_chunks == 0 ? 0 : request_.workspace_cap_bytes;
    output.tile_count = plan_.tiles.size();
    output.effective_compute_mode = effective_mode_;
    output.uses_cublas_lt = uses_cublas_lt_ ? 1U : 0U;
    return {};
  }

  [[nodiscard]] Error submit(std::shared_ptr<BackendOperation>& output) override {
    const auto owner = owner_.lock();
    if (!owner) {
      return make_error(XVRAM_STATUS_CLOSED, "gemm", "submit", "session is closed");
    }
    std::shared_ptr<RuntimeOperation> operation;
    if (Error error = owner->enqueue_progress(
            [owner, request = request_, problem = problem_,
             plan = plan_](RuntimeOperation& progress) {
              gemm::ExecutionHooks hooks;
              hooks.boundary = [&](const gemm::ExecutionBoundary boundary, std::size_t) {
                return progress.deadline_error("gemm",
                                               gemm::execution_boundary_operation(boundary))
                           ? gemm::ExecutionControl::deadline_expired
                           : gemm::ExecutionControl::proceed;
              };
              hooks.progress = [&](std::size_t, std::size_t) { progress.advance_unit(); };
              hooks.algorithm =
                  [&](const cublas::PreferredGemmResult& result) { owner->record_algorithm(result); };
              const gemm::TiledGemmResources resources{
                  owner->runtime(),
                  gemm::NativeGemmResources{owner->cublas_dispatch(), owner->cublas_handle(),
                                            owner->cublas_lt()}};
              const gemm::TiledGemmRequest execution{problem, plan, request.workspace_cap_bytes,
                                                     owner->config().prefetch_distance};
              return execution_error(*owner, progress,
                                     gemm::execute_tiled_gemm(resources, execution, hooks));
            },
            operation, plan_.tiles.size(), request_.deadline_ms);
        error) {
      return error;
    }
    output = std::move(operation);
    return {};
  }

private:
  [[nodiscard]] static Error execution_error(RuntimeSession& owner, RuntimeOperation& progress,
                                             const gemm::ExecutionResult& result) {
    switch (result.status) {
    case gemm::ExecutionStatus::success:
      return {};
    case gemm::ExecutionStatus::invalid_argument:
      return make_error(XVRAM_STATUS_INVALID_ARGUMENT, result.stage, result.operation,
                        result.message);
    case gemm::ExecutionStatus::unavailable:
      return make_error(XVRAM_STATUS_UNAVAILABLE, result.stage, result.operation, result.message);
    case gemm::ExecutionStatus::unsupported:
      return make_error(XVRAM_STATUS_UNSUPPORTED, result.stage, result.operation, result.message);
    case gemm::ExecutionStatus::runtime_failure:
      return runtime_error(owner.runtime(), result.runtime_status);
    case gemm::ExecutionStatus::cublas_failure:
      return make_native_error(XVRAM_STATUS_CUBLAS_ERROR, XVRAM_NATIVE_ERROR_CUBLAS,
                               result.cublas_status, result.stage, result.operation,
                               "CUBLAS_STATUS", result.message);
    case gemm::ExecutionStatus::deadline_expired: {
      Error deadline = progress.deadline_error(result.stage.c_str(), result.operation.c_str());
      return deadline ? deadline
                      : make_error(XVRAM_STATUS_TIMEOUT, result.stage, result.operation,
                                   result.message);
    }
    case gemm::ExecutionStatus::cancelled:
      return make_error(XVRAM_STATUS_CANCELLED, result.stage, result.operation, result.message);
    case gemm::ExecutionStatus::internal_failure:
      return make_error(XVRAM_STATUS_INTERNAL, result.stage, result.operation, result.message);
    }
    return make_error(XVRAM_STATUS_INTERNAL, "gemm", "execute_plan",
                      "unrecognized internal GEMM execution status");
  }

  std::weak_ptr<RuntimeSession> owner_;
  GemmRequest request_;
  gemm::GemmProblem problem_;
  gemm::GemmPlan plan_;
  xvram_compute_mode effective_mode_ = XVRAM_COMPUTE_AUTO;
  bool uses_cublas_lt_ = false;
};

Error RuntimeSession::create_gemm_plan(const GemmRequest& request,
                                       std::shared_ptr<BackendGemmPlan>& output) {
  const auto a = std::dynamic_pointer_cast<RuntimeAllocation>(request.a.allocation);
  const auto b = std::dynamic_pointer_cast<RuntimeAllocation>(request.b.allocation);
  const auto c = std::dynamic_pointer_cast<RuntimeAllocation>(request.c.allocation);
  if (!a || !b || !c || a->released() || b->released() || c->released()) {
    return make_error(XVRAM_STATUS_INVALID_ARGUMENT, "gemm", "create_plan",
                      "GEMM allocation is invalid or released");
  }
  return invoke_sync([this, request, a, b, c, &output]() {
    if (a->released() || b->released() || c->released()) {
      return make_error(XVRAM_STATUS_INVALID_ARGUMENT, "gemm", "create_plan",
                        "GEMM allocation was released before planning");
    }
    gemm::GemmProblem problem;
    const auto matrix = [](const MatrixRequest& source, const RuntimeAllocation& allocation) {
      return gemm::MatrixView{allocation.id(),          allocation.size(),
                              source.offset_bytes,      source.rows,
                              source.columns,           source.leading_dimension,
                              to_layout(source.layout), to_element_type(source.data_type)};
    };
    problem.a = matrix(request.a, *a);
    problem.b = matrix(request.b, *b);
    problem.c = matrix(request.c, *c);
    problem.a_operation = to_operation(request.operation_a);
    problem.b_operation = to_operation(request.operation_b);
    problem.m =
        problem.a_operation == gemm::MatrixOperation::none ? problem.a.rows : problem.a.columns;
    problem.k =
        problem.a_operation == gemm::MatrixOperation::none ? problem.a.columns : problem.a.rows;
    problem.n =
        problem.b_operation == gemm::MatrixOperation::none ? problem.b.columns : problem.b.rows;
    problem.compute_mode = to_compute_mode(request.compute_mode, request.a.data_type);
    problem.alpha = request.alpha;
    problem.beta = request.beta;

    const std::uint64_t workspace =
        request.workspace_cap_bytes == 0
            ? std::min<std::uint64_t>(config_.workspace_cap_bytes, 4ULL * 1024ULL * 1024ULL)
            : std::min(request.workspace_cap_bytes, config_.workspace_cap_bytes);
    const std::uint64_t chunk_bytes = runtime_.chunk_bytes();
    const std::uint64_t live_target = runtime_.target_bytes();
    const std::uint64_t workspace_reserve = runtime_config_.workspace_reserve_bytes;
    if (chunk_bytes == 0 || live_target <= workspace_reserve) {
      return make_error(XVRAM_STATUS_BUDGET_PRESSURE, "gemm", "make_plan",
                        "live target leaves no physical cache frames after workspace reserve");
    }
    const std::uint64_t frame_bytes =
        ((live_target - workspace_reserve) / chunk_bytes) * chunk_bytes;
    if (frame_bytes < chunk_bytes) {
      return make_error(XVRAM_STATUS_BUDGET_PRESSURE, "gemm", "make_plan",
                        "live target leaves no complete physical cache frame");
    }

    gemm::PlannerConfig planner;
    planner.chunk_bytes = chunk_bytes;
    // Runtime frame capacity is permanently computed using the session workspace reserve. Give
    // the planner that exact frame budget plus this plan's actual workspace so subtracting the
    // workspace cannot accidentally plan more frames than Runtime::execute can own.
    planner.cache_target_bytes = frame_bytes + workspace;
    planner.workspace_bytes = workspace;
    planner.preferred_geometry =
        gemm::TileGeometry{request.tile_m_hint == 0 ? 4096 : request.tile_m_hint,
                           request.tile_n_hint == 0 ? 4096 : request.tile_n_hint,
                           request.tile_k_hint == 0 ? 1024 : request.tile_k_hint};
    gemm::GemmPlan plan = gemm::make_plan(problem, planner);
    if (!plan) {
      const xvram_status status = plan.error == gemm::PlanError::working_set_too_large
                                      ? XVRAM_STATUS_BUDGET_PRESSURE
                                      : XVRAM_STATUS_INVALID_ARGUMENT;
      return make_error(status, "gemm", "make_plan",
                        std::string(gemm::plan_error_name(plan.error)));
    }
    GemmRequest normalized = request;
    normalized.workspace_cap_bytes = workspace;
    const xvram_compute_mode effective =
        problem.compute_mode == gemm::ComputeMode::fp64
            ? XVRAM_COMPUTE_FP64
            : (problem.compute_mode == gemm::ComputeMode::fast_tf32 ? XVRAM_COMPUTE_FP32_TF32
                                                                    : XVRAM_COMPUTE_FP32_STRICT);
    const bool use_cublas_lt =
        cublas_lt_ready() && gemm::is_cublas_lt_exact_proof_eligible(problem);
    output = std::make_shared<RuntimeGemmPlan>(shared_from_this(), std::move(normalized),
                                               std::move(problem), std::move(plan), effective,
                                               use_cublas_lt);
    return Error{};
  });
}

class RuntimeFactory final : public BackendFactory {
public:
  [[nodiscard]] Error create_session(const xvram_session_config_v1& config,
                                     std::shared_ptr<BackendSession>& output) override {
    cuda::abi::Context attached = nullptr;
    if (config.context_mode == XVRAM_CONTEXT_ATTACH_CURRENT) {
      cuda::CudaApi capture;
      const auto load = capture.load();
      if (load.status != cuda::CudaApi::LoadStatus::loaded || capture.init_ == nullptr ||
          capture.context_get_current_ == nullptr) {
        return make_error(XVRAM_STATUS_UNAVAILABLE, "session", "capture_context",
                          "CUDA driver is unavailable for attach-current mode");
      }
      if (capture.init_(0) != cuda::abi::success ||
          capture.context_get_current_(&attached) != cuda::abi::success || attached == nullptr) {
        return make_error(XVRAM_STATUS_INVALID_ARGUMENT, "session", "capture_context",
                          "no CUDA context is current on the calling thread");
      }
    }
    auto session = std::make_shared<RuntimeSession>(config, attached);
    if (Error error = session->start(); error) {
      return error;
    }
    output = std::move(session);
    return {};
  }
};

[[nodiscard]] BackendFactory* backend_factory() noexcept {
  static RuntimeFactory factory;
  return &factory;
}

struct FactoryInstaller {
  FactoryInstaller() noexcept {
    install_backend_factory(&backend_factory);
  }
};

const FactoryInstaller factory_installer;

} // namespace
} // namespace xvram::sdk
