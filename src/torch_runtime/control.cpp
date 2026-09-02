#include "torch_runtime/control.hpp"

#include "gemm/executor.hpp"
#include "gemm/planner.hpp"
#include "platform/cublas/cublas_api.hpp"
#include "platform/cuda/cuda_api.hpp"
#include "residency/runtime.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace xvram::torch_runtime {
namespace {

struct ErrorState {
  xvram_torch_runtime_status status = XVRAM_TORCH_RUNTIME_SUCCESS;
  std::int64_t native_code = 0;
  std::array<char, 32> stage{};
  std::array<char, 64> operation{};
  std::array<char, 256> message{};
};

struct CustomTelemetry {
  std::atomic<std::uint64_t> views_created{0};
  std::atomic<std::uint64_t> views_live{0};
  std::atomic<std::uint64_t> views_peak{0};
  std::atomic<std::uint64_t> leases_acquired{0};
  std::atomic<std::uint64_t> leases_sealed{0};
  std::atomic<std::uint64_t> leases_retired{0};
  std::atomic<std::uint64_t> events_recorded{0};
  std::atomic<std::uint64_t> events_retired{0};
  std::atomic<std::uint64_t> scratch_allocations{0};
  std::atomic<std::uint64_t> scratch_frees{0};
  std::atomic<std::uint64_t> scratch_failures{0};
  std::atomic<std::uint64_t> scratch_current{0};
  std::atomic<std::uint64_t> scratch_peak{0};
};

struct SessionState;

struct AllocationState {
  xvram_torch_runtime_allocation handle = XVRAM_TORCH_RUNTIME_INVALID_HANDLE;
  residency::AllocationId runtime_id{};
  std::uint64_t bytes = 0;
  xvram_torch_runtime_residency_hint hint = XVRAM_TORCH_RUNTIME_HINT_NORMAL;
  std::weak_ptr<SessionState> owner;
  bool released = false;
};

struct ScratchBlock {
  std::uint64_t address = 0;
  std::uint64_t bytes = 0;
  std::int32_t device = -1;
  cuda::abi::Stream stream = nullptr;
};

struct LeaseState {
  xvram_torch_runtime_lease handle = XVRAM_TORCH_RUNTIME_INVALID_HANDLE;
  residency::ExternalLease runtime_lease;
  residency::ExternalLeasePoll completion;
  std::weak_ptr<SessionState> owner;
  xvram_torch_runtime_lease_state state = XVRAM_TORCH_RUNTIME_LEASE_ARMED;
  xvram_torch_runtime_status result = XVRAM_TORCH_RUNTIME_SUCCESS;
  std::uint64_t generation = 0;
  std::atomic<std::uint64_t> live_views{0};
  std::unordered_map<xvram_torch_runtime_allocation, std::uint64_t> logical_bases;
  std::unordered_map<void*, ScratchBlock> scratch;
  std::uint64_t scratch_cursor = 0;
  bool event_recorded = false;
  bool retired_counted = false;
  bool bridge_poisoned = false;
  std::mutex mutex;
};

struct SessionState {
  xvram_torch_runtime_session handle = XVRAM_TORCH_RUNTIME_INVALID_HANDLE;
  std::int32_t device = -1;
  std::uint64_t scratch_arena_bytes = 0;
  std::thread::id owner_thread;
  cuda::CudaApi api;
  std::unique_ptr<residency::Runtime> runtime;
  cublas::CublasApi cublas;
  cublas::abi::Handle cublas_handle = nullptr;
  std::unique_ptr<cublas::LtMatmulExecutor> cublas_lt;
  std::unordered_map<xvram_torch_runtime_allocation, std::shared_ptr<AllocationState>> allocations;
  std::unordered_map<xvram_torch_runtime_lease, std::shared_ptr<LeaseState>> leases;
  xvram_torch_runtime_lease active_lease = XVRAM_TORCH_RUNTIME_INVALID_HANDLE;
  CustomTelemetry custom;
  ErrorState error;
  bool closed = false;
  std::mutex mutex;
};

struct Registry {
  std::mutex mutex;
  std::unordered_map<xvram_torch_runtime_session, std::shared_ptr<SessionState>> sessions;
  std::unordered_map<xvram_torch_runtime_allocation, std::shared_ptr<AllocationState>> allocations;
  std::unordered_map<xvram_torch_runtime_lease, std::shared_ptr<LeaseState>> leases;
  std::unordered_map<void*, std::weak_ptr<LeaseState>> scratch_owners;
  std::atomic<std::uint64_t> next_handle{1};
  std::atomic<std::uint64_t> next_generation{1};
};

[[nodiscard]] Registry& registry() {
  static Registry* value = new Registry();
  return *value;
}

thread_local std::weak_ptr<LeaseState> active_scratch_lease;

void copy_text(char* destination, const std::size_t capacity,
               const std::string_view source) noexcept {
  if (capacity == 0U) {
    return;
  }
  const std::size_t copied = std::min(capacity - 1U, source.size());
  if (copied != 0U) {
    std::memcpy(destination, source.data(), copied);
  }
  destination[copied] = '\0';
}

void set_error(SessionState& session, const xvram_torch_runtime_status status,
               const std::string_view stage, const std::string_view operation,
               const std::string_view message, const std::int64_t native_code = 0) noexcept {
  session.error = {};
  session.error.status = status;
  session.error.native_code = native_code;
  copy_text(session.error.stage.data(), session.error.stage.size(), stage);
  copy_text(session.error.operation.data(), session.error.operation.size(), operation);
  copy_text(session.error.message.data(), session.error.message.size(), message);
}

void clear_error(SessionState& session) noexcept {
  session.error = {};
}

[[nodiscard]] xvram_torch_runtime_status
map_status(const residency::RuntimeStatus status) noexcept {
  using residency::RuntimeStatus;
  switch (status) {
  case RuntimeStatus::success:
  case RuntimeStatus::callback_skipped:
    return XVRAM_TORCH_RUNTIME_SUCCESS;
  case RuntimeStatus::invalid_argument:
    return XVRAM_TORCH_RUNTIME_INVALID_ARGUMENT;
  case RuntimeStatus::unavailable:
    return XVRAM_TORCH_RUNTIME_UNAVAILABLE;
  case RuntimeStatus::unsupported:
    return XVRAM_TORCH_RUNTIME_UNSUPPORTED;
  case RuntimeStatus::host_oom:
    return XVRAM_TORCH_RUNTIME_HOST_OUT_OF_MEMORY;
  case RuntimeStatus::device_oom:
    return XVRAM_TORCH_RUNTIME_DEVICE_OUT_OF_MEMORY;
  case RuntimeStatus::budget_pressure:
    return XVRAM_TORCH_RUNTIME_BUDGET_PRESSURE;
  case RuntimeStatus::timeout:
    return XVRAM_TORCH_RUNTIME_TIMEOUT;
  case RuntimeStatus::cuda_failure:
    return XVRAM_TORCH_RUNTIME_CUDA_ERROR;
  case RuntimeStatus::poisoned:
    return XVRAM_TORCH_RUNTIME_QUARANTINED;
  case RuntimeStatus::cleanup_failure:
    return XVRAM_TORCH_RUNTIME_CLEANUP_FAILED;
  case RuntimeStatus::callback_failed:
  case RuntimeStatus::internal_failure:
    return XVRAM_TORCH_RUNTIME_INTERNAL_ERROR;
  }
  return XVRAM_TORCH_RUNTIME_INTERNAL_ERROR;
}

[[nodiscard]] xvram_torch_runtime_status
record_runtime_error(SessionState& session, const residency::RuntimeStatus status) {
  const xvram_torch_runtime_status mapped = map_status(status);
  const residency::RuntimeError& error = session.runtime->error();
  set_error(session, mapped, error.stage, error.operation,
            error.message.empty() ? residency::runtime_status_name(status) : error.message,
            error.native_code.value_or(0));
  return mapped;
}

template <typename Structure>
[[nodiscard]] bool compatible(const Structure* value, const std::size_t minimum) noexcept {
  return value != nullptr && value->struct_size >= minimum &&
         value->abi_version == XVRAM_TORCH_RUNTIME_ABI_VERSION_1;
}

template <typename Structure>
[[nodiscard]] xvram_torch_runtime_status copy_output(Structure* output,
                                                     const Structure& value) noexcept {
  constexpr std::size_t prefix = sizeof(std::uint32_t) * 2U;
  if (output == nullptr || output->struct_size < prefix ||
      output->abi_version != XVRAM_TORCH_RUNTIME_ABI_VERSION_1) {
    return XVRAM_TORCH_RUNTIME_INCOMPATIBLE_ABI;
  }
  std::memcpy(output, &value,
              std::min(static_cast<std::size_t>(output->struct_size), sizeof(Structure)));
  return XVRAM_TORCH_RUNTIME_SUCCESS;
}

template <typename Function>
[[nodiscard]] xvram_torch_runtime_status boundary(Function&& function) noexcept {
  try {
    return function();
  } catch (const std::bad_alloc&) {
    return XVRAM_TORCH_RUNTIME_HOST_OUT_OF_MEMORY;
  } catch (...) {
    return XVRAM_TORCH_RUNTIME_INTERNAL_ERROR;
  }
}

[[nodiscard]] std::uint64_t next_handle() noexcept {
  Registry& value = registry();
  std::uint64_t handle = value.next_handle.fetch_add(1U, std::memory_order_relaxed);
  if (handle == 0U) {
    handle = value.next_handle.fetch_add(1U, std::memory_order_relaxed);
  }
  return handle;
}

[[nodiscard]] std::shared_ptr<SessionState>
find_session(const xvram_torch_runtime_session handle) noexcept {
  Registry& value = registry();
  std::lock_guard lock(value.mutex);
  const auto found = value.sessions.find(handle);
  return found == value.sessions.end() ? nullptr : found->second;
}

[[nodiscard]] std::shared_ptr<AllocationState>
find_allocation(const xvram_torch_runtime_allocation handle) noexcept {
  Registry& value = registry();
  std::lock_guard lock(value.mutex);
  const auto found = value.allocations.find(handle);
  return found == value.allocations.end() ? nullptr : found->second;
}

[[nodiscard]] std::shared_ptr<LeaseState>
find_lease(const xvram_torch_runtime_lease handle) noexcept {
  Registry& value = registry();
  std::lock_guard lock(value.mutex);
  const auto found = value.leases.find(handle);
  return found == value.leases.end() ? nullptr : found->second;
}

[[nodiscard]] bool on_owner_thread(const SessionState& session) noexcept {
  return session.owner_thread == std::this_thread::get_id();
}

[[nodiscard]] residency::ResidencyHint
convert_hint(const xvram_torch_runtime_residency_hint hint) noexcept {
  if (hint == XVRAM_TORCH_RUNTIME_HINT_HOT) {
    return residency::ResidencyHint::hot;
  }
  if (hint == XVRAM_TORCH_RUNTIME_HINT_STREAMING) {
    return residency::ResidencyHint::streaming;
  }
  return residency::ResidencyHint::normal;
}

[[nodiscard]] std::optional<residency::AccessMode>
convert_access(const xvram_torch_runtime_access_mode mode) noexcept {
  if (mode == XVRAM_TORCH_RUNTIME_ACCESS_READ) {
    return residency::AccessMode::read;
  }
  if (mode == XVRAM_TORCH_RUNTIME_ACCESS_READ_WRITE) {
    return residency::AccessMode::read_write;
  }
  if (mode == XVRAM_TORCH_RUNTIME_ACCESS_WRITE_ONLY) {
    return residency::AccessMode::write_only;
  }
  return std::nullopt;
}

[[nodiscard]] bool checked_range(const std::uint64_t offset, const std::uint64_t length,
                                 const std::uint64_t size) noexcept {
  return length != 0U && offset <= size && length <= size - offset;
}

[[nodiscard]] xvram_torch_runtime_status
build_ranges(SessionState& session, const xvram_torch_runtime_access_range_v1* source,
             const std::size_t count, std::vector<residency::AccessRange>& output,
             std::unordered_map<std::uint64_t, xvram_torch_runtime_allocation>* reverse = nullptr) {
  if (source == nullptr || count == 0U || count > 4096U) {
    return XVRAM_TORCH_RUNTIME_INVALID_ARGUMENT;
  }
  output.clear();
  output.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    constexpr std::size_t minimum = offsetof(xvram_torch_runtime_access_range_v1, reserved);
    if (!compatible(&source[index], minimum)) {
      return XVRAM_TORCH_RUNTIME_INCOMPATIBLE_ABI;
    }
    const std::shared_ptr<AllocationState> allocation = find_allocation(source[index].allocation);
    const std::shared_ptr<SessionState> owner =
        allocation == nullptr ? nullptr : allocation->owner.lock();
    const std::optional<residency::AccessMode> mode = convert_access(source[index].mode);
    if (allocation == nullptr || owner.get() != &session || allocation->released || !mode ||
        !checked_range(source[index].byte_offset, source[index].byte_length, allocation->bytes)) {
      return XVRAM_TORCH_RUNTIME_INVALID_ARGUMENT;
    }
    output.push_back(
        {allocation->runtime_id, source[index].byte_offset, source[index].byte_length, *mode});
    if (reverse != nullptr) {
      (*reverse)[allocation->runtime_id.value] = allocation->handle;
    }
  }
  return XVRAM_TORCH_RUNTIME_SUCCESS;
}

[[nodiscard]] xvram_torch_runtime_lease_state
convert_lease_state(const residency::ExternalLeaseState state) noexcept {
  using residency::ExternalLeaseState;
  switch (state) {
  case ExternalLeaseState::armed:
    return XVRAM_TORCH_RUNTIME_LEASE_ARMED;
  case ExternalLeaseState::submitted:
    return XVRAM_TORCH_RUNTIME_LEASE_SUBMITTED;
  case ExternalLeaseState::completed:
    return XVRAM_TORCH_RUNTIME_LEASE_COMPLETED;
  case ExternalLeaseState::cancelled:
    return XVRAM_TORCH_RUNTIME_LEASE_CANCELLED;
  case ExternalLeaseState::failed:
    return XVRAM_TORCH_RUNTIME_LEASE_FAILED;
  case ExternalLeaseState::quarantined:
    return XVRAM_TORCH_RUNTIME_LEASE_QUARANTINED;
  }
  return XVRAM_TORCH_RUNTIME_LEASE_QUARANTINED;
}

void update_peak(std::atomic<std::uint64_t>& peak, const std::uint64_t value) noexcept {
  std::uint64_t observed = peak.load(std::memory_order_relaxed);
  while (observed < value &&
         !peak.compare_exchange_weak(observed, value, std::memory_order_relaxed)) {
  }
}

void fill_lease_info(const LeaseState& lease, xvram_torch_runtime_lease_info_v1& output) noexcept {
  output.state = lease.state;
  output.result = lease.result;
  output.live_tensor_storages = lease.live_views.load(std::memory_order_acquire);
  output.resolved_ranges = static_cast<std::uint64_t>(lease.runtime_lease.ranges.size());
  output.elapsed_milliseconds = lease.completion.elapsed_ms;
  if (lease.state == XVRAM_TORCH_RUNTIME_LEASE_ARMED ||
      lease.state == XVRAM_TORCH_RUNTIME_LEASE_SUBMITTED) {
    output.stream = reinterpret_cast<std::uintptr_t>(lease.runtime_lease.stream);
  }
}

void note_retired(SessionState& session, LeaseState& lease) noexcept {
  if (!lease.retired_counted && (lease.state == XVRAM_TORCH_RUNTIME_LEASE_COMPLETED ||
                                 lease.state == XVRAM_TORCH_RUNTIME_LEASE_CANCELLED ||
                                 lease.state == XVRAM_TORCH_RUNTIME_LEASE_FAILED ||
                                 lease.state == XVRAM_TORCH_RUNTIME_LEASE_QUARANTINED)) {
    lease.retired_counted = true;
    session.custom.leases_retired.fetch_add(1U, std::memory_order_relaxed);
    if (lease.event_recorded) {
      session.custom.events_retired.fetch_add(1U, std::memory_order_relaxed);
    }
    if (session.active_lease == lease.handle) {
      session.active_lease = XVRAM_TORCH_RUNTIME_INVALID_HANDLE;
    }
    const std::shared_ptr<LeaseState> active = active_scratch_lease.lock();
    if (active.get() == &lease) {
      active_scratch_lease.reset();
    }
  }
}

[[nodiscard]] std::optional<gemm::MatrixLayout>
convert_gemm_layout(const xvram_torch_runtime_gemm_layout value) noexcept {
  if (value == XVRAM_TORCH_RUNTIME_GEMM_ROW_MAJOR) {
    return gemm::MatrixLayout::row_major;
  }
  if (value == XVRAM_TORCH_RUNTIME_GEMM_COLUMN_MAJOR) {
    return gemm::MatrixLayout::column_major;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<gemm::MatrixOperation>
convert_gemm_operation(const xvram_torch_runtime_gemm_operation value) noexcept {
  if (value == XVRAM_TORCH_RUNTIME_GEMM_OPERATION_NONE) {
    return gemm::MatrixOperation::none;
  }
  if (value == XVRAM_TORCH_RUNTIME_GEMM_OPERATION_TRANSPOSE) {
    return gemm::MatrixOperation::transpose;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<gemm::ElementType>
convert_gemm_dtype(const xvram_torch_runtime_gemm_dtype value) noexcept {
  switch (value) {
  case XVRAM_TORCH_RUNTIME_GEMM_FP16:
    return gemm::ElementType::fp16;
  case XVRAM_TORCH_RUNTIME_GEMM_BF16:
    return gemm::ElementType::bf16;
  case XVRAM_TORCH_RUNTIME_GEMM_FP32:
    return gemm::ElementType::fp32;
  case XVRAM_TORCH_RUNTIME_GEMM_FP64:
    return gemm::ElementType::fp64;
  default:
    return std::nullopt;
  }
}

[[nodiscard]] std::optional<gemm::ComputeMode>
convert_gemm_compute(const xvram_torch_runtime_gemm_compute value) noexcept {
  switch (value) {
  case XVRAM_TORCH_RUNTIME_GEMM_COMPUTE_STRICT_FP32:
    return gemm::ComputeMode::strict_fp32;
  case XVRAM_TORCH_RUNTIME_GEMM_COMPUTE_FAST_TF32:
    return gemm::ComputeMode::fast_tf32;
  case XVRAM_TORCH_RUNTIME_GEMM_COMPUTE_FP64:
    return gemm::ComputeMode::fp64;
  default:
    return std::nullopt;
  }
}

[[nodiscard]] xvram_torch_runtime_gemm_boundary
gemm_boundary_from_operation(const std::string_view operation) noexcept {
  if (operation == "pre_launch_deadline") {
    return XVRAM_TORCH_RUNTIME_GEMM_BOUNDARY_PREFLIGHT;
  }
  if (operation == "tile_deadline") {
    return XVRAM_TORCH_RUNTIME_GEMM_BOUNDARY_BEFORE_TILE;
  }
  if (operation == "post_residency_deadline") {
    return XVRAM_TORCH_RUNTIME_GEMM_BOUNDARY_AFTER_RESIDENCY;
  }
  if (operation == "make_plan" || operation == "execute_plan") {
    return XVRAM_TORCH_RUNTIME_GEMM_BOUNDARY_PLAN;
  }
  return XVRAM_TORCH_RUNTIME_GEMM_BOUNDARY_AFTER_TILE;
}

[[nodiscard]] std::uint64_t counter_delta(const std::uint64_t after,
                                          const std::uint64_t before) noexcept {
  return after >= before ? after - before : 0U;
}

[[nodiscard]] xvram_torch_runtime_status
map_cublas_status(const cublas::abi::Status status) noexcept {
  return status == cublas::abi::allocation_failed ? XVRAM_TORCH_RUNTIME_DEVICE_OUT_OF_MEMORY
                                                  : XVRAM_TORCH_RUNTIME_CUBLAS_ERROR;
}

[[nodiscard]] xvram_torch_runtime_status ensure_cublas(SessionState& session) {
  if (session.cublas_handle != nullptr) {
    return XVRAM_TORCH_RUNTIME_SUCCESS;
  }
  const cublas::CublasLoadResult load = session.cublas.load();
  if (load.status != cublas::CublasLoadStatus::loaded) {
    const std::string_view message = session.cublas.error().empty()
                                         ? cublas::cublas_load_status_name(load.status)
                                         : std::string_view(session.cublas.error());
    set_error(session, XVRAM_TORCH_RUNTIME_UNAVAILABLE, "gemm", "load_cublas", message);
    return XVRAM_TORCH_RUNTIME_UNAVAILABLE;
  }
  const cublas::CublasDispatch& dispatch = session.cublas.dispatch();
  cublas::abi::Status status = cublas::abi::not_initialized;
  if (dispatch.create != nullptr) {
    status = dispatch.create(&session.cublas_handle);
  }
  if (status != cublas::abi::success || session.cublas_handle == nullptr) {
    session.cublas_handle = nullptr;
    const xvram_torch_runtime_status mapped = map_cublas_status(status);
    set_error(session, mapped, "gemm", "cublasCreate_v2", "cuBLAS handle creation failed", status);
    return mapped;
  }
  if (session.cublas.has_lt()) {
    try {
      auto executor = std::make_unique<cublas::LtMatmulExecutor>(dispatch);
      if (executor->initialize() == cublas::abi::success) {
        session.cublas_lt = std::move(executor);
      }
    } catch (const std::bad_alloc&) {
      // The baseline cuBLAS path remains valid when the optional Lt cache cannot be allocated.
    }
  }
  clear_error(session);
  return XVRAM_TORCH_RUNTIME_SUCCESS;
}

[[nodiscard]] xvram_torch_runtime_status destroy_cublas(SessionState& session) noexcept {
  cublas::abi::Status first_error = cublas::abi::success;
  const char* operation = "";
  if (session.cublas_lt != nullptr) {
    const cublas::abi::Status status = session.cublas_lt->close();
    if (status != cublas::abi::success) {
      first_error = status;
      operation = "cublasLtDestroy";
    }
    session.cublas_lt.reset();
  }
  if (session.cublas_handle != nullptr && session.cublas.dispatch().destroy != nullptr) {
    const cublas::abi::Status status = session.cublas.dispatch().destroy(session.cublas_handle);
    if (first_error == cublas::abi::success && status != cublas::abi::success) {
      first_error = status;
      operation = "cublasDestroy_v2";
    }
  }
  session.cublas_handle = nullptr;
  if (first_error == cublas::abi::success) {
    return XVRAM_TORCH_RUNTIME_SUCCESS;
  }
  set_error(session, XVRAM_TORCH_RUNTIME_CLEANUP_FAILED, "cleanup", operation,
            "cuBLAS handle destruction failed", first_error);
  return XVRAM_TORCH_RUNTIME_CLEANUP_FAILED;
}

void abandon_cublas(SessionState& session) noexcept {
  if (session.cublas_lt != nullptr) {
    session.cublas_lt->abandon();
    session.cublas_lt.reset();
  }
  session.cublas_handle = nullptr;
  session.cublas.abandon();
}

[[nodiscard]] const char* XVRAM_TORCH_RUNTIME_CALL
status_name_entry(const xvram_torch_runtime_status status) noexcept {
  static constexpr const char* names[] = {"success",
                                          "invalid_argument",
                                          "incompatible_abi",
                                          "not_found",
                                          "invalid_state",
                                          "views_live",
                                          "unsupported",
                                          "unavailable",
                                          "host_out_of_memory",
                                          "device_out_of_memory",
                                          "budget_pressure",
                                          "timeout",
                                          "cuda_error",
                                          "quarantined",
                                          "cleanup_failed",
                                          "internal_error",
                                          "cublas_error"};
  return status <= XVRAM_TORCH_RUNTIME_CUBLAS_ERROR ? names[status] : "unknown";
}

[[nodiscard]] xvram_torch_runtime_status XVRAM_TORCH_RUNTIME_CALL
session_create_entry(const xvram_torch_runtime_session_config_v1* config,
                     xvram_torch_runtime_session* output) noexcept {
  return boundary([&]() -> xvram_torch_runtime_status {
    constexpr std::size_t minimum = offsetof(xvram_torch_runtime_session_config_v1, reserved);
    if (output == nullptr || !compatible(config, minimum)) {
      return XVRAM_TORCH_RUNTIME_INCOMPATIBLE_ABI;
    }
    *output = XVRAM_TORCH_RUNTIME_INVALID_HANDLE;
    if (config->device_ordinal < 0 || config->chunk_bytes == 0U || config->staging_slots < 2U ||
        config->staging_slots > 8U ||
        (config->policy != XVRAM_TORCH_RUNTIME_POLICY_CLOCK &&
         config->policy != XVRAM_TORCH_RUNTIME_POLICY_LRU)) {
      return XVRAM_TORCH_RUNTIME_INVALID_ARGUMENT;
    }
    Registry& global = registry();
    {
      std::lock_guard lock(global.mutex);
      if (!global.sessions.empty()) {
        return XVRAM_TORCH_RUNTIME_INVALID_STATE;
      }
    }
    auto session = std::make_shared<SessionState>();
    session->handle = next_handle();
    session->device = config->device_ordinal;
    session->scratch_arena_bytes = config->scratch_arena_bytes;
    session->owner_thread = std::this_thread::get_id();
    const auto load = session->api.load();
    if (load.status != cuda::CudaApi::LoadStatus::loaded || session->api.init_ == nullptr ||
        session->api.context_get_current_ == nullptr ||
        session->api.context_get_device_ == nullptr) {
      return XVRAM_TORCH_RUNTIME_UNAVAILABLE;
    }
    if (session->api.init_(0) != cuda::abi::success) {
      return XVRAM_TORCH_RUNTIME_CUDA_ERROR;
    }
    cuda::abi::Context current = nullptr;
    cuda::abi::Device current_device = -1;
    if (session->api.context_get_current_(&current) != cuda::abi::success || current == nullptr ||
        session->api.context_get_device_(&current_device) != cuda::abi::success) {
      return XVRAM_TORCH_RUNTIME_INVALID_STATE;
    }
    if (current_device != config->device_ordinal) {
      return XVRAM_TORCH_RUNTIME_INVALID_ARGUMENT;
    }
    residency::RuntimeConfig runtime_config{};
    runtime_config.device_ordinal = config->device_ordinal;
    runtime_config.context_mode = residency::RuntimeContextMode::attach_current;
    runtime_config.attached_context = current;
    runtime_config.chunk_bytes = config->chunk_bytes;
    runtime_config.cache_target_bytes = config->cache_target_bytes;
    runtime_config.device_headroom_bytes = config->device_headroom_bytes;
    runtime_config.workspace_reserve_bytes = config->scratch_arena_bytes;
    runtime_config.staging_slots = config->staging_slots;
    runtime_config.policy = config->policy == XVRAM_TORCH_RUNTIME_POLICY_LRU
                                ? residency::RuntimePolicy::lru
                                : residency::RuntimePolicy::clock;
    runtime_config.stall_timeout = std::chrono::milliseconds(config->stall_timeout_milliseconds);
    runtime_config.budget_poll_interval =
        std::chrono::milliseconds(config->budget_poll_milliseconds);
    runtime_config.maximum_transaction_duration =
        std::chrono::milliseconds(config->maximum_transaction_milliseconds);
    session->runtime = std::make_unique<residency::Runtime>(session->api, runtime_config);
    const residency::RuntimeStatus setup = session->runtime->setup();
    if (setup != residency::RuntimeStatus::success) {
      return record_runtime_error(*session, setup);
    }
    {
      std::lock_guard lock(global.mutex);
      global.sessions.emplace(session->handle, session);
    }
    *output = session->handle;
    return XVRAM_TORCH_RUNTIME_SUCCESS;
  });
}

[[nodiscard]] xvram_torch_runtime_status XVRAM_TORCH_RUNTIME_CALL session_get_error_entry(
    const xvram_torch_runtime_session handle, xvram_torch_runtime_error_v1* output) noexcept {
  return boundary([&]() -> xvram_torch_runtime_status {
    const std::shared_ptr<SessionState> session = find_session(handle);
    if (session == nullptr) {
      return XVRAM_TORCH_RUNTIME_NOT_FOUND;
    }
    std::lock_guard lock(session->mutex);
    xvram_torch_runtime_error_v1 value = XVRAM_TORCH_RUNTIME_ERROR_V1_INIT;
    value.status = session->error.status;
    value.native_code = session->error.native_code;
    std::memcpy(value.stage, session->error.stage.data(), session->error.stage.size());
    std::memcpy(value.operation, session->error.operation.data(), session->error.operation.size());
    std::memcpy(value.message, session->error.message.data(), session->error.message.size());
    return copy_output(output, value);
  });
}

[[nodiscard]] xvram_torch_runtime_status XVRAM_TORCH_RUNTIME_CALL session_get_telemetry_entry(
    const xvram_torch_runtime_session handle, xvram_torch_runtime_telemetry_v1* output) noexcept {
  return boundary([&]() -> xvram_torch_runtime_status {
    const std::shared_ptr<SessionState> session = find_session(handle);
    if (session == nullptr) {
      return XVRAM_TORCH_RUNTIME_NOT_FOUND;
    }
    std::lock_guard lock(session->mutex);
    const residency::RuntimeTelemetry& source = session->runtime->telemetry();
    xvram_torch_runtime_telemetry_v1 value = XVRAM_TORCH_RUNTIME_TELEMETRY_V1_INIT;
    value.allocations_created = source.allocations_created;
    value.allocations_released = source.allocations_released;
    value.handles_created = source.handles_created;
    value.handles_reused = source.handles_reused;
    value.maps = source.maps;
    value.set_access_calls = source.set_access;
    value.unmaps = source.unmaps;
    value.event_boundaries = source.event_boundaries;
    value.unsafe_remaps = source.unsafe_remaps;
    value.unsafe_transitions = source.unsafe_transitions;
    value.h2d_bytes = source.h2d_bytes;
    value.d2h_bytes = source.d2h_bytes;
    value.cache_hits = source.cache_hits;
    value.cache_misses = source.cache_misses;
    value.clean_evictions = source.clean_evictions;
    value.dirty_evictions = source.dirty_evictions;
    value.dirty_writebacks = source.dirty_writebacks;
    value.prefetches = source.prefetches;
    value.watchdog_rejections = source.watchdog_rejections;
    value.budget_shrinks = source.budget_shrinks;
    value.budget_grows = source.budget_grows;
    value.resident_bytes = source.resident_bytes;
    value.resident_peak_bytes = source.resident_peak_bytes;
    value.cache_target_bytes = source.target_bytes;
    value.tensor_views_created = session->custom.views_created.load();
    value.tensor_views_live = session->custom.views_live.load();
    value.tensor_views_peak = session->custom.views_peak.load();
    value.leases_acquired = session->custom.leases_acquired.load();
    value.leases_sealed = session->custom.leases_sealed.load();
    value.leases_retired = session->custom.leases_retired.load();
    value.events_recorded = session->custom.events_recorded.load();
    value.events_retired = session->custom.events_retired.load();
    value.scratch_arena_bytes = session->scratch_arena_bytes;
    value.scratch_allocation_calls = session->custom.scratch_allocations.load();
    value.scratch_free_calls = session->custom.scratch_frees.load();
    value.scratch_failures = session->custom.scratch_failures.load();
    value.scratch_bytes_current = session->custom.scratch_current.load();
    value.scratch_bytes_peak = session->custom.scratch_peak.load();
    value.stable_addresses = source.stable_addresses ? 1U : 0U;
    value.no_physical_aliases = source.no_physical_aliases ? 1U : 0U;
    value.quarantined = source.quarantined ? 1U : 0U;
    return copy_output(output, value);
  });
}

[[nodiscard]] xvram_torch_runtime_status XVRAM_TORCH_RUNTIME_CALL
allocation_create_entry(const xvram_torch_runtime_session session_handle,
                        const xvram_torch_runtime_allocation_desc_v1* description,
                        xvram_torch_runtime_allocation* output) noexcept {
  return boundary([&]() -> xvram_torch_runtime_status {
    constexpr std::size_t minimum = offsetof(xvram_torch_runtime_allocation_desc_v1, reserved);
    if (output == nullptr || !compatible(description, minimum)) {
      return XVRAM_TORCH_RUNTIME_INCOMPATIBLE_ABI;
    }
    *output = XVRAM_TORCH_RUNTIME_INVALID_HANDLE;
    if (description->bytes == 0U || (description->hint != XVRAM_TORCH_RUNTIME_HINT_NORMAL &&
                                     description->hint != XVRAM_TORCH_RUNTIME_HINT_HOT &&
                                     description->hint != XVRAM_TORCH_RUNTIME_HINT_STREAMING)) {
      return XVRAM_TORCH_RUNTIME_INVALID_ARGUMENT;
    }
    const std::shared_ptr<SessionState> session = find_session(session_handle);
    if (session == nullptr || !on_owner_thread(*session)) {
      return session == nullptr ? XVRAM_TORCH_RUNTIME_NOT_FOUND : XVRAM_TORCH_RUNTIME_INVALID_STATE;
    }
    std::lock_guard lock(session->mutex);
    auto allocation = std::make_shared<AllocationState>();
    allocation->handle = next_handle();
    allocation->hint = description->hint;
    allocation->owner = session;
    residency::RuntimeAllocation created{};
    const residency::RuntimeStatus status =
        session->runtime->allocate(description->bytes, convert_hint(description->hint), created);
    if (status != residency::RuntimeStatus::success) {
      return record_runtime_error(*session, status);
    }
    allocation->runtime_id = created.id;
    allocation->bytes = created.bytes;
    Registry& global = registry();
    try {
      session->allocations.emplace(allocation->handle, allocation);
      std::lock_guard registry_lock(global.mutex);
      if (!global.allocations.emplace(allocation->handle, allocation).second) {
        throw std::bad_alloc();
      }
    } catch (...) {
      session->allocations.erase(allocation->handle);
      (void)session->runtime->release(created.id);
      throw;
    }
    *output = allocation->handle;
    clear_error(*session);
    return XVRAM_TORCH_RUNTIME_SUCCESS;
  });
}

[[nodiscard]] xvram_torch_runtime_status XVRAM_TORCH_RUNTIME_CALL
allocation_get_info_entry(const xvram_torch_runtime_allocation handle,
                          xvram_torch_runtime_allocation_info_v1* output) noexcept {
  const std::shared_ptr<AllocationState> allocation = find_allocation(handle);
  if (allocation == nullptr || allocation->released) {
    return XVRAM_TORCH_RUNTIME_NOT_FOUND;
  }
  xvram_torch_runtime_allocation_info_v1 value = XVRAM_TORCH_RUNTIME_ALLOCATION_INFO_V1_INIT;
  value.bytes = allocation->bytes;
  value.hint = allocation->hint;
  return copy_output(output, value);
}

template <bool Write>
[[nodiscard]] xvram_torch_runtime_status
allocation_transfer(const xvram_torch_runtime_allocation handle, const std::uint64_t offset,
                    void* memory, const std::uint64_t bytes) noexcept {
  return boundary([&]() -> xvram_torch_runtime_status {
    const std::shared_ptr<AllocationState> allocation = find_allocation(handle);
    const std::shared_ptr<SessionState> session =
        allocation == nullptr ? nullptr : allocation->owner.lock();
    if (allocation == nullptr || session == nullptr || allocation->released) {
      return XVRAM_TORCH_RUNTIME_NOT_FOUND;
    }
    if (!on_owner_thread(*session) || memory == nullptr ||
        !checked_range(offset, bytes, allocation->bytes)) {
      return XVRAM_TORCH_RUNTIME_INVALID_ARGUMENT;
    }
    std::lock_guard lock(session->mutex);
    const residency::RuntimeStatus status =
        Write ? session->runtime->write(allocation->runtime_id, offset, memory, bytes)
              : session->runtime->read(allocation->runtime_id, offset, memory, bytes);
    if (status == residency::RuntimeStatus::success) {
      return XVRAM_TORCH_RUNTIME_SUCCESS;
    }
    return record_runtime_error(*session, status);
  });
}

[[nodiscard]] xvram_torch_runtime_status XVRAM_TORCH_RUNTIME_CALL
allocation_write_entry(const xvram_torch_runtime_allocation handle, const std::uint64_t offset,
                       const void* source, const std::uint64_t bytes) noexcept {
  return allocation_transfer<true>(handle, offset, const_cast<void*>(source), bytes);
}

[[nodiscard]] xvram_torch_runtime_status XVRAM_TORCH_RUNTIME_CALL
allocation_read_entry(const xvram_torch_runtime_allocation handle, const std::uint64_t offset,
                      void* destination, const std::uint64_t bytes) noexcept {
  return allocation_transfer<false>(handle, offset, destination, bytes);
}

[[nodiscard]] xvram_torch_runtime_status
allocation_remove(const xvram_torch_runtime_allocation handle, const bool discard,
                  const std::uint64_t offset, const std::uint64_t bytes) noexcept {
  return boundary([&]() -> xvram_torch_runtime_status {
    const std::shared_ptr<AllocationState> allocation = find_allocation(handle);
    const std::shared_ptr<SessionState> session =
        allocation == nullptr ? nullptr : allocation->owner.lock();
    if (allocation == nullptr || session == nullptr || allocation->released) {
      return XVRAM_TORCH_RUNTIME_NOT_FOUND;
    }
    if (!on_owner_thread(*session)) {
      return XVRAM_TORCH_RUNTIME_INVALID_STATE;
    }
    std::lock_guard lock(session->mutex);
    residency::RuntimeStatus status = residency::RuntimeStatus::invalid_argument;
    const bool full_discard = discard && offset == 0U && bytes == 0U;
    if (!discard) {
      status = session->runtime->release(allocation->runtime_id);
    } else if (full_discard) {
      status = session->runtime->discard_dead(allocation->runtime_id);
    } else if (checked_range(offset, bytes, allocation->bytes)) {
      status = session->runtime->discard_dead(allocation->runtime_id, offset, bytes);
    }
    if (status != residency::RuntimeStatus::success) {
      return record_runtime_error(*session, status);
    }
    if (!discard || full_discard) {
      allocation->released = true;
      session->allocations.erase(handle);
      Registry& global = registry();
      std::lock_guard registry_lock(global.mutex);
      global.allocations.erase(handle);
    }
    return XVRAM_TORCH_RUNTIME_SUCCESS;
  });
}

[[nodiscard]] xvram_torch_runtime_status XVRAM_TORCH_RUNTIME_CALL
allocation_release_entry(const xvram_torch_runtime_allocation handle) noexcept {
  return allocation_remove(handle, false, 0U, 0U);
}

[[nodiscard]] xvram_torch_runtime_status XVRAM_TORCH_RUNTIME_CALL
allocation_discard_entry(const xvram_torch_runtime_allocation handle, const std::uint64_t offset,
                         const std::uint64_t bytes) noexcept {
  return allocation_remove(handle, true, offset, bytes);
}

[[nodiscard]] xvram_torch_runtime_status XVRAM_TORCH_RUNTIME_CALL session_prefetch_entry(
    const xvram_torch_runtime_session handle, const xvram_torch_runtime_access_range_v1* source,
    const std::size_t count) noexcept {
  return boundary([&]() -> xvram_torch_runtime_status {
    const std::shared_ptr<SessionState> session = find_session(handle);
    if (session == nullptr || !on_owner_thread(*session)) {
      return session == nullptr ? XVRAM_TORCH_RUNTIME_NOT_FOUND : XVRAM_TORCH_RUNTIME_INVALID_STATE;
    }
    std::lock_guard lock(session->mutex);
    std::vector<residency::AccessRange> ranges;
    const xvram_torch_runtime_status built = build_ranges(*session, source, count, ranges);
    if (built != XVRAM_TORCH_RUNTIME_SUCCESS) {
      return built;
    }
    const residency::RuntimeStatus status = session->runtime->prefetch(ranges);
    if (status == residency::RuntimeStatus::success) {
      return XVRAM_TORCH_RUNTIME_SUCCESS;
    }
    return record_runtime_error(*session, status);
  });
}

[[nodiscard]] xvram_torch_runtime_status XVRAM_TORCH_RUNTIME_CALL lease_acquire_entry(
    const xvram_torch_runtime_session handle, const xvram_torch_runtime_lease_desc_v1* description,
    xvram_torch_runtime_lease* output) noexcept {
  return boundary([&]() -> xvram_torch_runtime_status {
    constexpr std::size_t minimum = offsetof(xvram_torch_runtime_lease_desc_v1, reserved);
    if (output == nullptr || !compatible(description, minimum)) {
      return XVRAM_TORCH_RUNTIME_INCOMPATIBLE_ABI;
    }
    *output = XVRAM_TORCH_RUNTIME_INVALID_HANDLE;
    const std::shared_ptr<SessionState> session = find_session(handle);
    if (session == nullptr || !on_owner_thread(*session)) {
      return session == nullptr ? XVRAM_TORCH_RUNTIME_NOT_FOUND : XVRAM_TORCH_RUNTIME_INVALID_STATE;
    }
    std::lock_guard lock(session->mutex);
    if (session->active_lease != XVRAM_TORCH_RUNTIME_INVALID_HANDLE ||
        !active_scratch_lease.expired()) {
      return XVRAM_TORCH_RUNTIME_INVALID_STATE;
    }
    std::vector<residency::AccessRange> ranges;
    std::unordered_map<std::uint64_t, xvram_torch_runtime_allocation> reverse;
    const xvram_torch_runtime_status built =
        build_ranges(*session, description->ranges, description->range_count, ranges, &reverse);
    if (built != XVRAM_TORCH_RUNTIME_SUCCESS) {
      return built;
    }
    auto lease = std::make_shared<LeaseState>();
    lease->handle = next_handle();
    lease->generation = registry().next_generation.fetch_add(1U, std::memory_order_relaxed);
    lease->owner = session;
    residency::ExternalLease acquired{};
    const residency::RuntimeStatus status = session->runtime->acquire_external(
        residency::ExternalLeaseRequest{ranges, session->scratch_arena_bytes}, acquired);
    if (status != residency::RuntimeStatus::success) {
      return record_runtime_error(*session, status);
    }
    lease->runtime_lease = std::move(acquired);
    const auto rollback = [&]() noexcept {
      const residency::RuntimeStatus sealed = session->runtime->seal_external(
          lease->runtime_lease.id, residency::ExternalSealMode::cancelled_before_submission);
      residency::ExternalLeasePoll ignored{};
      const residency::RuntimeStatus waited =
          sealed == residency::RuntimeStatus::success
              ? session->runtime->wait_external(lease->runtime_lease.id,
                                                std::chrono::milliseconds(5000), ignored)
              : sealed;
      return sealed == residency::RuntimeStatus::success &&
             (waited == residency::RuntimeStatus::success ||
              waited == residency::RuntimeStatus::callback_skipped);
    };
    bool global_inserted = false;
    bool session_inserted = false;
    try {
      for (const residency::ResolvedRange& range : lease->runtime_lease.ranges) {
        const auto found = reverse.find(range.allocation_id.value);
        if (found == reverse.end() || range.device_address < range.offset_bytes) {
          const bool safe = rollback();
          return safe ? XVRAM_TORCH_RUNTIME_INTERNAL_ERROR : XVRAM_TORCH_RUNTIME_QUARANTINED;
        }
        const std::uint64_t base = range.device_address - range.offset_bytes;
        const auto [base_entry, inserted] = lease->logical_bases.emplace(found->second, base);
        if (!inserted && base_entry->second != base) {
          const bool safe = rollback();
          return safe ? XVRAM_TORCH_RUNTIME_INTERNAL_ERROR : XVRAM_TORCH_RUNTIME_QUARANTINED;
        }
      }
      Registry& global = registry();
      {
        std::lock_guard registry_lock(global.mutex);
        global_inserted = global.leases.emplace(lease->handle, lease).second;
      }
      if (!global_inserted) {
        throw std::bad_alloc();
      }
      session_inserted = session->leases.emplace(lease->handle, lease).second;
      if (!session_inserted) {
        throw std::bad_alloc();
      }
    } catch (...) {
      if (session_inserted) {
        session->leases.erase(lease->handle);
      }
      if (global_inserted) {
        Registry& global = registry();
        std::lock_guard registry_lock(global.mutex);
        global.leases.erase(lease->handle);
      }
      (void)rollback();
      throw;
    }
    session->active_lease = lease->handle;
    active_scratch_lease = lease;
    session->custom.leases_acquired.fetch_add(1U, std::memory_order_relaxed);
    *output = lease->handle;
    return XVRAM_TORCH_RUNTIME_SUCCESS;
  });
}

[[nodiscard]] xvram_torch_runtime_status XVRAM_TORCH_RUNTIME_CALL lease_get_info_entry(
    const xvram_torch_runtime_lease handle, xvram_torch_runtime_lease_info_v1* output) noexcept {
  const std::shared_ptr<LeaseState> lease = find_lease(handle);
  if (lease == nullptr) {
    return XVRAM_TORCH_RUNTIME_NOT_FOUND;
  }
  std::lock_guard lock(lease->mutex);
  xvram_torch_runtime_lease_info_v1 value = XVRAM_TORCH_RUNTIME_LEASE_INFO_V1_INIT;
  fill_lease_info(*lease, value);
  return copy_output(output, value);
}

[[nodiscard]] xvram_torch_runtime_status XVRAM_TORCH_RUNTIME_CALL lease_seal_entry(
    const xvram_torch_runtime_lease handle, const xvram_torch_runtime_seal_mode mode) noexcept {
  return boundary([&]() -> xvram_torch_runtime_status {
    const std::shared_ptr<LeaseState> lease = find_lease(handle);
    const std::shared_ptr<SessionState> session = lease == nullptr ? nullptr : lease->owner.lock();
    if (lease == nullptr || session == nullptr) {
      return XVRAM_TORCH_RUNTIME_NOT_FOUND;
    }
    if (!on_owner_thread(*session)) {
      return XVRAM_TORCH_RUNTIME_INVALID_STATE;
    }
    std::lock_guard session_lock(session->mutex);
    std::lock_guard lease_lock(lease->mutex);
    if (lease->state != XVRAM_TORCH_RUNTIME_LEASE_ARMED || lease->bridge_poisoned) {
      return lease->bridge_poisoned ? XVRAM_TORCH_RUNTIME_QUARANTINED
                                    : XVRAM_TORCH_RUNTIME_INVALID_STATE;
    }
    if (lease->live_views.load(std::memory_order_acquire) != 0U || !lease->scratch.empty()) {
      return XVRAM_TORCH_RUNTIME_VIEWS_LIVE;
    }
    residency::ExternalSealMode runtime_mode{};
    if (mode == XVRAM_TORCH_RUNTIME_SEAL_SUCCESS) {
      runtime_mode = residency::ExternalSealMode::success;
    } else if (mode == XVRAM_TORCH_RUNTIME_SEAL_CANCELLED_BEFORE_SUBMISSION) {
      runtime_mode = residency::ExternalSealMode::cancelled_before_submission;
    } else if (mode == XVRAM_TORCH_RUNTIME_SEAL_FAILED_AFTER_POSSIBLE_SUBMISSION) {
      runtime_mode = residency::ExternalSealMode::failed_after_possible_submission;
    } else {
      return XVRAM_TORCH_RUNTIME_INVALID_ARGUMENT;
    }
    const residency::RuntimeStatus status =
        session->runtime->seal_external(lease->runtime_lease.id, runtime_mode);
    if (status != residency::RuntimeStatus::success) {
      return record_runtime_error(*session, status);
    }
    lease->state = XVRAM_TORCH_RUNTIME_LEASE_SUBMITTED;
    lease->event_recorded = mode != XVRAM_TORCH_RUNTIME_SEAL_CANCELLED_BEFORE_SUBMISSION;
    session->custom.leases_sealed.fetch_add(1U, std::memory_order_relaxed);
    if (lease->event_recorded) {
      session->custom.events_recorded.fetch_add(1U, std::memory_order_relaxed);
    }
    active_scratch_lease.reset();
    return XVRAM_TORCH_RUNTIME_SUCCESS;
  });
}

[[nodiscard]] xvram_torch_runtime_status
observe_lease(const xvram_torch_runtime_lease handle, const std::optional<std::uint64_t> timeout,
              xvram_torch_runtime_lease_info_v1* output) noexcept {
  return boundary([&]() -> xvram_torch_runtime_status {
    const std::shared_ptr<LeaseState> lease = find_lease(handle);
    const std::shared_ptr<SessionState> session = lease == nullptr ? nullptr : lease->owner.lock();
    if (lease == nullptr || session == nullptr) {
      return XVRAM_TORCH_RUNTIME_NOT_FOUND;
    }
    if (!on_owner_thread(*session)) {
      return XVRAM_TORCH_RUNTIME_INVALID_STATE;
    }
    std::lock_guard session_lock(session->mutex);
    std::lock_guard lease_lock(lease->mutex);
    if (lease->state != XVRAM_TORCH_RUNTIME_LEASE_SUBMITTED) {
      return XVRAM_TORCH_RUNTIME_INVALID_STATE;
    }
    residency::ExternalLeasePoll completion{};
    const residency::RuntimeStatus status =
        timeout.has_value()
            ? session->runtime->wait_external(lease->runtime_lease.id,
                                              std::chrono::milliseconds(*timeout), completion)
            : session->runtime->poll_external(lease->runtime_lease.id, completion);
    lease->completion = completion;
    lease->state = convert_lease_state(completion.state);
    lease->result = map_status(completion.result);
    note_retired(*session, *lease);
    if (status != residency::RuntimeStatus::success &&
        status != residency::RuntimeStatus::callback_skipped &&
        status != residency::RuntimeStatus::callback_failed) {
      (void)record_runtime_error(*session, status);
    }
    xvram_torch_runtime_lease_info_v1 value = XVRAM_TORCH_RUNTIME_LEASE_INFO_V1_INIT;
    fill_lease_info(*lease, value);
    const xvram_torch_runtime_status copied = copy_output(output, value);
    return copied != XVRAM_TORCH_RUNTIME_SUCCESS ? copied : map_status(status);
  });
}

[[nodiscard]] xvram_torch_runtime_status XVRAM_TORCH_RUNTIME_CALL lease_poll_entry(
    const xvram_torch_runtime_lease handle, xvram_torch_runtime_lease_info_v1* output) noexcept {
  return observe_lease(handle, std::nullopt, output);
}

[[nodiscard]] xvram_torch_runtime_status XVRAM_TORCH_RUNTIME_CALL
lease_wait_entry(const xvram_torch_runtime_lease handle, const std::uint64_t timeout,
                 xvram_torch_runtime_lease_info_v1* output) noexcept {
  return observe_lease(handle, timeout, output);
}

[[nodiscard]] xvram_torch_runtime_status XVRAM_TORCH_RUNTIME_CALL
gemm_execute_entry(const xvram_torch_runtime_session session_handle,
                   const xvram_torch_runtime_gemm_problem_v1* description,
                   xvram_torch_runtime_gemm_result_v1* output) noexcept {
  xvram_torch_runtime_gemm_result_v1 result = XVRAM_TORCH_RUNTIME_GEMM_RESULT_V1_INIT;
  result.status = XVRAM_TORCH_RUNTIME_GEMM_INTERNAL_FAILED;
  result.boundary = XVRAM_TORCH_RUNTIME_GEMM_BOUNDARY_PREFLIGHT;
  constexpr std::size_t output_prefix = sizeof(std::uint32_t) * 2U;
  if (output == nullptr || output->struct_size < output_prefix ||
      output->abi_version != XVRAM_TORCH_RUNTIME_ABI_VERSION_1) {
    return XVRAM_TORCH_RUNTIME_INCOMPATIBLE_ABI;
  }

  const xvram_torch_runtime_status status = boundary([&]() -> xvram_torch_runtime_status {
    constexpr std::size_t problem_minimum = offsetof(xvram_torch_runtime_gemm_problem_v1, reserved);
    constexpr std::size_t matrix_minimum = offsetof(xvram_torch_runtime_gemm_matrix_v1, reserved);
    if (!compatible(description, problem_minimum) ||
        !compatible(description == nullptr ? nullptr : &description->a, matrix_minimum) ||
        !compatible(description == nullptr ? nullptr : &description->b, matrix_minimum) ||
        !compatible(description == nullptr ? nullptr : &description->c, matrix_minimum)) {
      result.status = XVRAM_TORCH_RUNTIME_GEMM_INVALID_PROBLEM;
      return XVRAM_TORCH_RUNTIME_INCOMPATIBLE_ABI;
    }
    if (description->m == 0U || description->n == 0U || description->k == 0U ||
        !std::isfinite(description->alpha) || !std::isfinite(description->beta) ||
        description->beta != 0.0 || description->prefetch_distance > 8U) {
      result.status = XVRAM_TORCH_RUNTIME_GEMM_INVALID_PROBLEM;
      return XVRAM_TORCH_RUNTIME_INVALID_ARGUMENT;
    }

    const std::shared_ptr<SessionState> session = find_session(session_handle);
    if (session == nullptr) {
      result.status = XVRAM_TORCH_RUNTIME_GEMM_INVALID_PROBLEM;
      return XVRAM_TORCH_RUNTIME_NOT_FOUND;
    }
    if (!on_owner_thread(*session)) {
      result.status = XVRAM_TORCH_RUNTIME_GEMM_INVALID_PROBLEM;
      return XVRAM_TORCH_RUNTIME_INVALID_STATE;
    }
    std::lock_guard lock(session->mutex);
    if (session->closed || session->active_lease != XVRAM_TORCH_RUNTIME_INVALID_HANDLE ||
        session->custom.views_live.load(std::memory_order_acquire) != 0U ||
        session->custom.scratch_current.load(std::memory_order_acquire) != 0U) {
      result.status = XVRAM_TORCH_RUNTIME_GEMM_INVALID_PROBLEM;
      set_error(*session, XVRAM_TORCH_RUNTIME_INVALID_STATE, "gemm", "preflight",
                "tiled GEMM requires an idle session with no external lease, tensor view, or "
                "scratch allocation");
      return XVRAM_TORCH_RUNTIME_INVALID_STATE;
    }
    if (description->workspace_bytes > session->scratch_arena_bytes) {
      result.status = XVRAM_TORCH_RUNTIME_GEMM_PLANNING_FAILED;
      result.boundary = XVRAM_TORCH_RUNTIME_GEMM_BOUNDARY_PLAN;
      set_error(*session, XVRAM_TORCH_RUNTIME_BUDGET_PRESSURE, "gemm", "workspace_cap",
                "requested GEMM workspace exceeds the session scratch reserve");
      return XVRAM_TORCH_RUNTIME_BUDGET_PRESSURE;
    }

    const std::optional<gemm::ComputeMode> compute = convert_gemm_compute(description->compute);
    if (!compute.has_value()) {
      result.status = XVRAM_TORCH_RUNTIME_GEMM_INVALID_PROBLEM;
      set_error(*session, XVRAM_TORCH_RUNTIME_INVALID_ARGUMENT, "gemm", "compute_mode",
                "unsupported GEMM compute mode");
      return XVRAM_TORCH_RUNTIME_INVALID_ARGUMENT;
    }

    gemm::GemmProblem problem;
    const auto convert_matrix = [&](const xvram_torch_runtime_gemm_matrix_v1& source,
                                    gemm::MatrixView& destination,
                                    gemm::MatrixOperation& operation) {
      const std::optional<gemm::MatrixLayout> layout = convert_gemm_layout(source.layout);
      const std::optional<gemm::MatrixOperation> converted_operation =
          convert_gemm_operation(source.operation);
      const std::optional<gemm::ElementType> dtype = convert_gemm_dtype(source.dtype);
      const std::shared_ptr<AllocationState> allocation = find_allocation(source.allocation);
      const std::shared_ptr<SessionState> owner =
          allocation == nullptr ? nullptr : allocation->owner.lock();
      if (!layout.has_value() || !converted_operation.has_value() || !dtype.has_value() ||
          allocation == nullptr || owner.get() != session.get() || allocation->released) {
        return false;
      }
      destination = gemm::MatrixView{allocation->runtime_id,
                                     allocation->bytes,
                                     source.byte_offset,
                                     source.rows,
                                     source.columns,
                                     source.leading_dimension,
                                     *layout,
                                     *dtype};
      operation = *converted_operation;
      return true;
    };
    gemm::MatrixOperation c_operation = gemm::MatrixOperation::none;
    if (!convert_matrix(description->a, problem.a, problem.a_operation) ||
        !convert_matrix(description->b, problem.b, problem.b_operation) ||
        !convert_matrix(description->c, problem.c, c_operation)) {
      result.status = XVRAM_TORCH_RUNTIME_GEMM_INVALID_PROBLEM;
      set_error(*session, XVRAM_TORCH_RUNTIME_INVALID_ARGUMENT, "gemm", "matrix",
                "GEMM matrix metadata or allocation handle is invalid");
      return XVRAM_TORCH_RUNTIME_INVALID_ARGUMENT;
    }
    if (c_operation != gemm::MatrixOperation::none) {
      result.status = XVRAM_TORCH_RUNTIME_GEMM_INVALID_PROBLEM;
      set_error(*session, XVRAM_TORCH_RUNTIME_INVALID_ARGUMENT, "gemm", "output_operation",
                "GEMM output transpose is not supported");
      return XVRAM_TORCH_RUNTIME_INVALID_ARGUMENT;
    }
    problem.m = description->m;
    problem.n = description->n;
    problem.k = description->k;
    problem.alpha = description->alpha;
    problem.beta = description->beta;
    problem.compute_mode = *compute;
    const gemm::ProblemValidation validation = gemm::validate_problem(problem);
    if (!validation) {
      result.status = XVRAM_TORCH_RUNTIME_GEMM_INVALID_PROBLEM;
      set_error(*session, XVRAM_TORCH_RUNTIME_INVALID_ARGUMENT, "gemm", "validate_problem",
                gemm::problem_error_name(validation.error));
      return XVRAM_TORCH_RUNTIME_INVALID_ARGUMENT;
    }

    const std::uint64_t chunk_bytes = session->runtime->chunk_bytes();
    const std::uint64_t live_target = session->runtime->target_bytes();
    const std::uint64_t frame_budget = live_target > session->scratch_arena_bytes
                                           ? live_target - session->scratch_arena_bytes
                                           : 0U;
    const std::uint64_t frame_bytes =
        chunk_bytes == 0U ? 0U : (frame_budget / chunk_bytes) * chunk_bytes;
    if (chunk_bytes == 0U || frame_bytes / chunk_bytes < 2U) {
      result.status = XVRAM_TORCH_RUNTIME_GEMM_PLANNING_FAILED;
      result.boundary = XVRAM_TORCH_RUNTIME_GEMM_BOUNDARY_PLAN;
      set_error(*session, XVRAM_TORCH_RUNTIME_BUDGET_PRESSURE, "gemm", "make_plan",
                "live target leaves fewer than two complete cache frames");
      return XVRAM_TORCH_RUNTIME_BUDGET_PRESSURE;
    }
    gemm::PlannerConfig planner;
    planner.chunk_bytes = chunk_bytes;
    planner.workspace_bytes = description->workspace_bytes;
    planner.cache_target_bytes = frame_bytes + description->workspace_bytes;
    planner.preferred_geometry = gemm::TileGeometry{
        description->preferred_tile_m == 0U ? 4096U : description->preferred_tile_m,
        description->preferred_tile_n == 0U ? 4096U : description->preferred_tile_n,
        description->preferred_tile_k == 0U ? 1024U : description->preferred_tile_k};
    const gemm::GemmPlan plan = gemm::make_plan(problem, planner);
    if (!plan) {
      result.status = XVRAM_TORCH_RUNTIME_GEMM_PLANNING_FAILED;
      result.boundary = XVRAM_TORCH_RUNTIME_GEMM_BOUNDARY_PLAN;
      const bool pressure = plan.error == gemm::PlanError::working_set_too_large ||
                            plan.error == gemm::PlanError::invalid_cache_target ||
                            plan.error == gemm::PlanError::workspace_exceeds_cache;
      const xvram_torch_runtime_status mapped =
          pressure ? XVRAM_TORCH_RUNTIME_BUDGET_PRESSURE : XVRAM_TORCH_RUNTIME_INVALID_ARGUMENT;
      set_error(*session, mapped, "gemm", "make_plan", gemm::plan_error_name(plan.error));
      return mapped;
    }
    result.tile_count = static_cast<std::uint64_t>(plan.tiles.size());
    result.workspace_bytes = description->workspace_bytes;
    result.maximum_working_set_bytes = plan.maximum_resident_bytes;
    result.tile_m = plan.geometry.m;
    result.tile_n = plan.geometry.n;
    result.tile_k = plan.geometry.k;

    const xvram_torch_runtime_status loaded = ensure_cublas(*session);
    if (loaded != XVRAM_TORCH_RUNTIME_SUCCESS) {
      result.status = XVRAM_TORCH_RUNTIME_GEMM_CUBLAS_UNAVAILABLE;
      return loaded;
    }

    const residency::RuntimeTelemetry before = session->runtime->telemetry();
    const auto started = std::chrono::steady_clock::now();
    gemm::ExecutionHooks hooks;
    hooks.boundary = [&](const gemm::ExecutionBoundary execution_boundary, std::size_t) {
      switch (execution_boundary) {
      case gemm::ExecutionBoundary::before_execution:
        result.boundary = XVRAM_TORCH_RUNTIME_GEMM_BOUNDARY_PREFLIGHT;
        break;
      case gemm::ExecutionBoundary::before_tile:
        result.boundary = XVRAM_TORCH_RUNTIME_GEMM_BOUNDARY_BEFORE_TILE;
        break;
      case gemm::ExecutionBoundary::after_residency:
        result.boundary = XVRAM_TORCH_RUNTIME_GEMM_BOUNDARY_AFTER_RESIDENCY;
        break;
      case gemm::ExecutionBoundary::after_tile:
        result.boundary = XVRAM_TORCH_RUNTIME_GEMM_BOUNDARY_AFTER_TILE;
        break;
      }
      return gemm::ExecutionControl::proceed;
    };
    hooks.progress = [&](std::size_t completed, std::size_t) {
      result.tiles_completed = static_cast<std::uint64_t>(completed);
      result.maximum_kernel_milliseconds = std::max(
          result.maximum_kernel_milliseconds, session->runtime->telemetry().last_transaction_ms);
    };
    hooks.algorithm = [&](const cublas::PreferredGemmResult& selection) {
      ++result.tiles_submitted;
      if (selection.path == cublas::PreferredGemmPath::cublas_core) {
        ++result.cublas_core_tiles;
      } else {
        ++result.cublas_lt_tiles;
        if (selection.lt.algorithm_cache_hit) {
          ++result.algorithm_cache_hits;
        }
      }
    };
    cublas::LtMatmulExecutor* lt =
        gemm::is_cublas_lt_exact_proof_eligible(problem) ? session->cublas_lt.get() : nullptr;
    const gemm::TiledGemmResources resources{
        *session->runtime,
        gemm::NativeGemmResources{session->cublas.dispatch(), session->cublas_handle, lt}};
    const gemm::TiledGemmRequest request{problem, plan, description->workspace_bytes,
                                         description->prefetch_distance};
    const gemm::ExecutionResult execution = gemm::execute_tiled_gemm(resources, request, hooks);
    const residency::RuntimeTelemetry after = session->runtime->telemetry();
    result.elapsed_milliseconds =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
            .count();
    result.events_recorded =
        counter_delta(after.transactions_submitted, before.transactions_submitted);
    result.events_retired =
        counter_delta(after.transactions_completed, before.transactions_completed);
    result.maps = counter_delta(after.maps, before.maps);
    result.set_access_calls = counter_delta(after.set_access, before.set_access);
    result.h2d_bytes = counter_delta(after.h2d_bytes, before.h2d_bytes);
    result.d2h_bytes = counter_delta(after.d2h_bytes, before.d2h_bytes);
    result.clean_evictions = counter_delta(after.clean_evictions, before.clean_evictions);
    result.dirty_evictions = counter_delta(after.dirty_evictions, before.dirty_evictions);
    result.handles_reused = counter_delta(after.handles_reused, before.handles_reused);
    result.prefetches = counter_delta(after.prefetches, before.prefetches);

    switch (execution.status) {
    case gemm::ExecutionStatus::success:
      result.status = XVRAM_TORCH_RUNTIME_GEMM_COMPLETED;
      result.boundary = XVRAM_TORCH_RUNTIME_GEMM_BOUNDARY_AFTER_TILE;
      clear_error(*session);
      return XVRAM_TORCH_RUNTIME_SUCCESS;
    case gemm::ExecutionStatus::invalid_argument:
      result.status = XVRAM_TORCH_RUNTIME_GEMM_INVALID_PROBLEM;
      set_error(*session, XVRAM_TORCH_RUNTIME_INVALID_ARGUMENT, execution.stage,
                execution.operation, execution.message);
      return XVRAM_TORCH_RUNTIME_INVALID_ARGUMENT;
    case gemm::ExecutionStatus::unavailable:
      result.status = XVRAM_TORCH_RUNTIME_GEMM_CUBLAS_UNAVAILABLE;
      set_error(*session, XVRAM_TORCH_RUNTIME_UNAVAILABLE, execution.stage, execution.operation,
                execution.message);
      return XVRAM_TORCH_RUNTIME_UNAVAILABLE;
    case gemm::ExecutionStatus::unsupported:
      result.status = XVRAM_TORCH_RUNTIME_GEMM_INVALID_PROBLEM;
      set_error(*session, XVRAM_TORCH_RUNTIME_UNSUPPORTED, execution.stage, execution.operation,
                execution.message);
      return XVRAM_TORCH_RUNTIME_UNSUPPORTED;
    case gemm::ExecutionStatus::runtime_failure: {
      result.status = XVRAM_TORCH_RUNTIME_GEMM_RUNTIME_FAILED;
      const xvram_torch_runtime_status mapped =
          record_runtime_error(*session, execution.runtime_status);
      result.boundary = gemm_boundary_from_operation(execution.operation);
      return mapped;
    }
    case gemm::ExecutionStatus::cublas_failure: {
      result.status = XVRAM_TORCH_RUNTIME_GEMM_CUBLAS_FAILED;
      result.native_code = execution.cublas_status;
      const xvram_torch_runtime_status mapped = map_cublas_status(execution.cublas_status);
      set_error(*session, mapped, execution.stage, execution.operation, execution.message,
                execution.cublas_status);
      result.boundary = gemm_boundary_from_operation(execution.operation);
      return mapped;
    }
    case gemm::ExecutionStatus::deadline_expired:
      result.status = XVRAM_TORCH_RUNTIME_GEMM_CANCELLED;
      set_error(*session, XVRAM_TORCH_RUNTIME_TIMEOUT, execution.stage, execution.operation,
                execution.message);
      result.boundary = gemm_boundary_from_operation(execution.operation);
      return XVRAM_TORCH_RUNTIME_TIMEOUT;
    case gemm::ExecutionStatus::cancelled:
      result.status = XVRAM_TORCH_RUNTIME_GEMM_CANCELLED;
      set_error(*session, XVRAM_TORCH_RUNTIME_INVALID_STATE, execution.stage, execution.operation,
                execution.message);
      result.boundary = gemm_boundary_from_operation(execution.operation);
      return XVRAM_TORCH_RUNTIME_INVALID_STATE;
    case gemm::ExecutionStatus::internal_failure:
      result.status = XVRAM_TORCH_RUNTIME_GEMM_INTERNAL_FAILED;
      set_error(*session, XVRAM_TORCH_RUNTIME_INTERNAL_ERROR, execution.stage, execution.operation,
                execution.message);
      result.boundary = gemm_boundary_from_operation(execution.operation);
      return XVRAM_TORCH_RUNTIME_INTERNAL_ERROR;
    }
    result.status = XVRAM_TORCH_RUNTIME_GEMM_INTERNAL_FAILED;
    return XVRAM_TORCH_RUNTIME_INTERNAL_ERROR;
  });

  if (status == XVRAM_TORCH_RUNTIME_HOST_OUT_OF_MEMORY &&
      result.status == XVRAM_TORCH_RUNTIME_GEMM_INTERNAL_FAILED) {
    result.boundary = XVRAM_TORCH_RUNTIME_GEMM_BOUNDARY_PREFLIGHT;
  }
  const xvram_torch_runtime_status copied = copy_output(output, result);
  return copied == XVRAM_TORCH_RUNTIME_SUCCESS ? status : copied;
}

[[nodiscard]] xvram_torch_runtime_status XVRAM_TORCH_RUNTIME_CALL session_close_entry(
    const xvram_torch_runtime_session handle, const std::uint64_t timeout) noexcept {
  return boundary([&]() -> xvram_torch_runtime_status {
    const std::shared_ptr<SessionState> session = find_session(handle);
    if (session == nullptr) {
      return XVRAM_TORCH_RUNTIME_NOT_FOUND;
    }
    if (!on_owner_thread(*session)) {
      return XVRAM_TORCH_RUNTIME_INVALID_STATE;
    }
    std::lock_guard lock(session->mutex);
    if (session->active_lease != XVRAM_TORCH_RUNTIME_INVALID_HANDLE) {
      const auto found = session->leases.find(session->active_lease);
      if (found != session->leases.end()) {
        std::lock_guard lease_lock(found->second->mutex);
        if (found->second->state == XVRAM_TORCH_RUNTIME_LEASE_ARMED ||
            found->second->live_views.load() != 0U || !found->second->scratch.empty()) {
          return XVRAM_TORCH_RUNTIME_INVALID_STATE;
        }
        residency::ExternalLeasePoll completion{};
        const residency::RuntimeStatus waited = session->runtime->wait_external(
            found->second->runtime_lease.id, std::chrono::milliseconds(timeout), completion);
        found->second->completion = completion;
        found->second->state = convert_lease_state(completion.state);
        found->second->result = map_status(completion.result);
        note_retired(*session, *found->second);
        if (waited != residency::RuntimeStatus::success &&
            waited != residency::RuntimeStatus::callback_skipped &&
            waited != residency::RuntimeStatus::callback_failed) {
          return record_runtime_error(*session, waited);
        }
      }
    }
    for (const auto& entry : session->leases) {
      std::lock_guard lease_lock(entry.second->mutex);
      if (entry.second->live_views.load() != 0U || !entry.second->scratch.empty()) {
        return XVRAM_TORCH_RUNTIME_VIEWS_LIVE;
      }
    }
    const residency::RuntimeStatus drained = session->runtime->drain(true);
    const bool quarantine = drained == residency::RuntimeStatus::poisoned ||
                            session->runtime->async_completion_unknown();
    xvram_torch_runtime_status cublas_cleanup = XVRAM_TORCH_RUNTIME_SUCCESS;
    if (quarantine) {
      abandon_cublas(*session);
    } else {
      cublas_cleanup = destroy_cublas(*session);
    }
    const residency::RuntimeStatus status = session->runtime->close();
    if (drained != residency::RuntimeStatus::success) {
      return record_runtime_error(*session, drained);
    }
    if (cublas_cleanup != XVRAM_TORCH_RUNTIME_SUCCESS) {
      return cublas_cleanup;
    }
    if (status != residency::RuntimeStatus::success) {
      return record_runtime_error(*session, status);
    }
    session->closed = true;
    Registry& global = registry();
    std::lock_guard registry_lock(global.mutex);
    for (const auto& entry : session->allocations) {
      global.allocations.erase(entry.first);
    }
    for (const auto& entry : session->leases) {
      global.leases.erase(entry.first);
    }
    global.sessions.erase(handle);
    active_scratch_lease.reset();
    return XVRAM_TORCH_RUNTIME_SUCCESS;
  });
}

} // namespace

struct ViewTicket::State {
  std::weak_ptr<LeaseState> lease;
  std::weak_ptr<SessionState> session;
  std::uint64_t generation = 0;
  std::atomic<bool> released{false};
};

ViewTicket::ViewTicket(std::shared_ptr<State> state) noexcept : state_(std::move(state)) {}

ViewTicket::~ViewTicket() {
  if (state_ == nullptr || state_->released.exchange(true, std::memory_order_acq_rel)) {
    return;
  }
  const std::shared_ptr<LeaseState> lease = state_->lease.lock();
  const std::shared_ptr<SessionState> session = state_->session.lock();
  if (lease == nullptr || session == nullptr) {
    return;
  }
  std::lock_guard lock(lease->mutex);
  if (lease->generation != state_->generation) {
    lease->bridge_poisoned = true;
    return;
  }
  const std::uint64_t prior = lease->live_views.fetch_sub(1U, std::memory_order_acq_rel);
  if (prior == 0U) {
    lease->live_views.store(0U, std::memory_order_release);
    lease->bridge_poisoned = true;
    return;
  }
  session->custom.views_live.fetch_sub(1U, std::memory_order_relaxed);
}

xvram_torch_runtime_status prepare_view(const xvram_torch_runtime_session session_handle,
                                        const xvram_torch_runtime_lease lease_handle,
                                        const xvram_torch_runtime_allocation allocation_handle,
                                        const std::uint64_t byte_offset,
                                        const std::uint64_t required_bytes,
                                        const std::uint64_t alignment,
                                        PreparedView& output) noexcept {
  output = {};
  return boundary([&]() -> xvram_torch_runtime_status {
    if (required_bytes == 0U || alignment == 0U || (alignment & (alignment - 1U)) != 0U) {
      return XVRAM_TORCH_RUNTIME_INVALID_ARGUMENT;
    }
    const std::shared_ptr<SessionState> session = find_session(session_handle);
    const std::shared_ptr<LeaseState> lease = find_lease(lease_handle);
    const std::shared_ptr<AllocationState> allocation = find_allocation(allocation_handle);
    if (session == nullptr || lease == nullptr || allocation == nullptr) {
      return XVRAM_TORCH_RUNTIME_NOT_FOUND;
    }
    if (lease->owner.lock().get() != session.get() ||
        allocation->owner.lock().get() != session.get() || allocation->released ||
        !checked_range(byte_offset, required_bytes, allocation->bytes)) {
      return XVRAM_TORCH_RUNTIME_INVALID_ARGUMENT;
    }
    std::lock_guard lease_lock(lease->mutex);
    if (lease->state != XVRAM_TORCH_RUNTIME_LEASE_ARMED || lease->bridge_poisoned) {
      return XVRAM_TORCH_RUNTIME_INVALID_STATE;
    }
    const auto base = lease->logical_bases.find(allocation_handle);
    if (base == lease->logical_bases.end() ||
        base->second > std::numeric_limits<std::uint64_t>::max() - byte_offset) {
      return XVRAM_TORCH_RUNTIME_INVALID_ARGUMENT;
    }
    const std::uint64_t address = base->second + byte_offset;
    if ((address & (alignment - 1U)) != 0U) {
      return XVRAM_TORCH_RUNTIME_INVALID_ARGUMENT;
    }
    auto ticket_state = std::make_shared<ViewTicket::State>();
    ticket_state->lease = lease;
    ticket_state->session = session;
    ticket_state->generation = lease->generation;
    std::shared_ptr<ViewTicket> ticket(new ViewTicket(std::move(ticket_state)));
    const std::uint64_t live = lease->live_views.fetch_add(1U) + 1U;
    session->custom.views_created.fetch_add(1U);
    const std::uint64_t global_live = session->custom.views_live.fetch_add(1U) + 1U;
    update_peak(session->custom.views_peak, global_live);
    (void)live;
    output.address = reinterpret_cast<void*>(static_cast<std::uintptr_t>(address));
    output.byte_length = allocation->bytes - byte_offset;
    output.device_ordinal = session->device;
    output.ticket = std::move(ticket);
    clear_error(*session);
    return XVRAM_TORCH_RUNTIME_SUCCESS;
  });
}

std::string last_error_message() {
  return "invalid, stale, or unsafe xVRAM resolved view";
}

} // namespace xvram::torch_runtime

extern "C" void* XVRAM_TORCH_RUNTIME_CALL xvram_torch_runtime_scratch_alloc(const std::size_t bytes,
                                                                            const int device,
                                                                            void* stream) {
  using namespace xvram::torch_runtime;
  const std::shared_ptr<LeaseState> lease = active_scratch_lease.lock();
  const std::shared_ptr<SessionState> session = lease == nullptr ? nullptr : lease->owner.lock();
  if (lease == nullptr || session == nullptr || bytes == 0U || device != session->device ||
      stream != lease->runtime_lease.stream) {
    if (session != nullptr) {
      session->custom.scratch_failures.fetch_add(1U);
    }
    return nullptr;
  }
  std::lock_guard lock(lease->mutex);
  constexpr std::uint64_t alignment = 256U;
  const std::uint64_t aligned = (lease->scratch_cursor + alignment - 1U) & ~(alignment - 1U);
  if (lease->state != XVRAM_TORCH_RUNTIME_LEASE_ARMED ||
      lease->runtime_lease.workspace_address == 0U || aligned < lease->scratch_cursor ||
      bytes > session->scratch_arena_bytes - std::min(aligned, session->scratch_arena_bytes) ||
      lease->runtime_lease.workspace_address >
          std::numeric_limits<std::uint64_t>::max() - aligned) {
    session->custom.scratch_failures.fetch_add(1U);
    return nullptr;
  }
  const std::uint64_t address = lease->runtime_lease.workspace_address + aligned;
  void* pointer = reinterpret_cast<void*>(static_cast<std::uintptr_t>(address));
  const std::uint64_t old_cursor = lease->scratch_cursor;
  try {
    if (!lease->scratch
             .emplace(pointer, ScratchBlock{address, static_cast<std::uint64_t>(bytes), device,
                                            reinterpret_cast<xvram::cuda::abi::Stream>(stream)})
             .second) {
      session->custom.scratch_failures.fetch_add(1U);
      return nullptr;
    }
    Registry& global = registry();
    std::lock_guard registry_lock(global.mutex);
    if (!global.scratch_owners.emplace(pointer, lease).second) {
      lease->scratch.erase(pointer);
      session->custom.scratch_failures.fetch_add(1U);
      return nullptr;
    }
    lease->scratch_cursor = aligned + bytes;
  } catch (...) {
    lease->scratch.erase(pointer);
    lease->scratch_cursor = old_cursor;
    session->custom.scratch_failures.fetch_add(1U);
    return nullptr;
  }
  session->custom.scratch_allocations.fetch_add(1U);
  const std::uint64_t current = session->custom.scratch_current.fetch_add(bytes) + bytes;
  update_peak(session->custom.scratch_peak, current);
  return pointer;
}

extern "C" void XVRAM_TORCH_RUNTIME_CALL xvram_torch_runtime_scratch_free(void* pointer,
                                                                          const std::size_t bytes,
                                                                          const int device,
                                                                          void* stream) {
  using namespace xvram::torch_runtime;
  std::shared_ptr<LeaseState> lease;
  {
    Registry& global = registry();
    std::lock_guard lock(global.mutex);
    const auto found = global.scratch_owners.find(pointer);
    if (found != global.scratch_owners.end()) {
      lease = found->second.lock();
    }
  }
  const std::shared_ptr<SessionState> session = lease == nullptr ? nullptr : lease->owner.lock();
  if (lease == nullptr || session == nullptr) {
    return;
  }
  std::lock_guard lock(lease->mutex);
  const auto found = lease->scratch.find(pointer);
  if (found == lease->scratch.end() || found->second.bytes != bytes ||
      found->second.device != device || found->second.stream != stream) {
    lease->bridge_poisoned = true;
    session->custom.scratch_failures.fetch_add(1U);
    return;
  }
  lease->scratch.erase(found);
  {
    Registry& global = registry();
    std::lock_guard registry_lock(global.mutex);
    global.scratch_owners.erase(pointer);
  }
  session->custom.scratch_frees.fetch_add(1U);
  session->custom.scratch_current.fetch_sub(bytes);
}

extern "C" xvram_torch_runtime_status XVRAM_TORCH_RUNTIME_CALL
xvram_torch_runtime_get_api(const std::uint32_t requested_abi_version,
                            const std::uint32_t caller_api_size, void* output_api) {
  using namespace xvram::torch_runtime;
  if (requested_abi_version != XVRAM_TORCH_RUNTIME_ABI_VERSION_1 || output_api == nullptr ||
      caller_api_size < sizeof(xvram_torch_runtime_api_v1)) {
    return XVRAM_TORCH_RUNTIME_INCOMPATIBLE_ABI;
  }
  xvram_torch_runtime_api_v1 value{};
  value.struct_size = sizeof(value);
  value.abi_version = XVRAM_TORCH_RUNTIME_ABI_VERSION_1;
  value.status_name = &status_name_entry;
  value.session_create = &session_create_entry;
  value.session_get_error = &session_get_error_entry;
  value.session_get_telemetry = &session_get_telemetry_entry;
  value.session_close = &session_close_entry;
  value.allocation_create = &allocation_create_entry;
  value.allocation_get_info = &allocation_get_info_entry;
  value.allocation_write = &allocation_write_entry;
  value.allocation_read = &allocation_read_entry;
  value.allocation_release = &allocation_release_entry;
  value.allocation_discard = &allocation_discard_entry;
  value.session_prefetch = &session_prefetch_entry;
  value.lease_acquire = &lease_acquire_entry;
  value.lease_get_info = &lease_get_info_entry;
  value.lease_seal = &lease_seal_entry;
  value.lease_poll = &lease_poll_entry;
  value.lease_wait = &lease_wait_entry;
  value.gemm_execute = &gemm_execute_entry;
  std::memcpy(output_api, &value, sizeof(value));
  return XVRAM_TORCH_RUNTIME_SUCCESS;
}
