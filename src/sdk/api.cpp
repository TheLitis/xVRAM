#include "sdk/backend.hpp"
#include "sdk/deadline.hpp"
#include "sdk/status.hpp"
#include "xvram/xvram.h"

#include <atomic>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string_view>
#include <utility>
#include <vector>

namespace xvram::sdk::detail {

struct SessionState {
  SessionState(std::shared_ptr<BackendSession> value, const std::uint64_t configured_chunk_bytes,
               const std::uint64_t configured_workspace_cap_bytes)
      : backend(std::move(value)), chunk_size_bytes(configured_chunk_bytes),
        workspace_cap_bytes(configured_workspace_cap_bytes) {}

  void record(const Error& error) noexcept {
    if (!error) {
      return;
    }
    try {
      const std::scoped_lock lock(error_mutex);
      last_error = error;
    } catch (...) {
      // Error reporting must never turn a recoverable backend error into an ABI exception.
    }
  }

  [[nodiscard]] Error error() const {
    const std::scoped_lock lock(error_mutex);
    return last_error;
  }

  std::shared_ptr<BackendSession> backend;
  std::uint64_t chunk_size_bytes = 0;
  std::uint64_t workspace_cap_bytes = 0;
  mutable std::mutex error_mutex;
  Error last_error;
  std::atomic<bool> closed{false};
};

} // namespace xvram::sdk::detail

struct xvram_session_t {
  std::shared_ptr<xvram::sdk::detail::SessionState> state;
};

struct xvram_allocation_t {
  std::shared_ptr<xvram::sdk::detail::SessionState> owner;
  std::shared_ptr<xvram::sdk::BackendAllocation> backend;
};

struct xvram_operation_t {
  std::shared_ptr<xvram::sdk::detail::SessionState> owner;
  std::shared_ptr<xvram::sdk::BackendOperation> backend;
};

struct xvram_gemm_plan_t {
  std::shared_ptr<xvram::sdk::detail::SessionState> owner;
  std::shared_ptr<xvram::sdk::BackendGemmPlan> backend;
};

namespace xvram::sdk {
namespace {

std::atomic<BackendFactoryProvider> factory_provider{nullptr};

} // namespace

void install_backend_factory(const BackendFactoryProvider provider) noexcept {
  factory_provider.store(provider, std::memory_order_release);
}

} // namespace xvram::sdk

namespace {

using xvram::sdk::AccessRequest;
using xvram::sdk::Error;
using xvram::sdk::GemmRequest;
using xvram::sdk::MatrixRequest;
using xvram::sdk::PrefetchRequest;
using xvram::sdk::TransactionRequest;
using SessionState = xvram::sdk::detail::SessionState;

[[nodiscard]] Error invalid(const std::string_view operation, const std::string_view message) {
  return xvram::sdk::make_error(XVRAM_STATUS_INVALID_ARGUMENT, "sdk", operation, message);
}

[[nodiscard]] Error incompatible(const std::string_view operation, const std::string_view message) {
  return xvram::sdk::make_error(XVRAM_STATUS_INCOMPATIBLE_ABI, "sdk", operation, message);
}

[[nodiscard]] xvram_status publish(const std::shared_ptr<SessionState>& state,
                                   const Error& error) noexcept {
  if (error && state) {
    state->record(error);
  }
  return error.status;
}

template <typename Function>
[[nodiscard]] xvram_status boundary(const std::shared_ptr<SessionState>& state,
                                    const std::string_view operation,
                                    Function&& function) noexcept {
  try {
    return publish(state, function());
  } catch (const std::bad_alloc&) {
    return publish(state, xvram::sdk::make_error(XVRAM_STATUS_HOST_OUT_OF_MEMORY, "sdk", operation,
                                                 "host allocation failed"));
  } catch (...) {
    const Error error = xvram::sdk::exception_error(operation);
    return publish(state, error);
  }
}

template <typename Structure> [[nodiscard]] bool has_v1_size(const Structure* structure) noexcept {
  return structure != nullptr && structure->struct_size >= sizeof(Structure);
}

[[nodiscard]] bool valid_context_mode(const xvram_context_mode mode) noexcept {
  return mode == XVRAM_CONTEXT_ISOLATED || mode == XVRAM_CONTEXT_ATTACH_CURRENT;
}

[[nodiscard]] bool valid_cache_policy(const xvram_cache_policy policy) noexcept {
  return policy == XVRAM_CACHE_POLICY_CLOCK || policy == XVRAM_CACHE_POLICY_LRU;
}

[[nodiscard]] bool valid_priority(const xvram_allocation_priority priority) noexcept {
  return priority == XVRAM_ALLOCATION_NORMAL || priority == XVRAM_ALLOCATION_HOT ||
         priority == XVRAM_ALLOCATION_STREAMING;
}

[[nodiscard]] bool valid_access_mode(const xvram_access_mode mode) noexcept {
  return mode == XVRAM_ACCESS_READ || mode == XVRAM_ACCESS_READ_WRITE ||
         mode == XVRAM_ACCESS_WRITE_ONLY;
}

[[nodiscard]] bool valid_data_type(const xvram_data_type type) noexcept {
  return type == XVRAM_DATA_FP16 || type == XVRAM_DATA_BF16 || type == XVRAM_DATA_FP32 ||
         type == XVRAM_DATA_FP64;
}

[[nodiscard]] bool valid_layout(const xvram_matrix_layout layout) noexcept {
  return layout == XVRAM_MATRIX_ROW_MAJOR || layout == XVRAM_MATRIX_COLUMN_MAJOR;
}

[[nodiscard]] bool valid_matrix_operation(const xvram_matrix_operation operation) noexcept {
  return operation == XVRAM_MATRIX_OP_N || operation == XVRAM_MATRIX_OP_T;
}

[[nodiscard]] bool valid_compute_mode(const xvram_compute_mode mode) noexcept {
  return mode == XVRAM_COMPUTE_AUTO || mode == XVRAM_COMPUTE_FP32_STRICT ||
         mode == XVRAM_COMPUTE_FP32_TF32 || mode == XVRAM_COMPUTE_FP64;
}

[[nodiscard]] bool all_zero(const uint64_t* values, const std::size_t count) noexcept {
  for (std::size_t index = 0; index < count; ++index) {
    if (values[index] != 0U) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] Error validate_session_config(const xvram_session_config_v1* config) {
  if (!has_v1_size(config)) {
    return incompatible("session_create", "session config is null or smaller than v1");
  }
  if (config->flags != 0U || config->reserved0 != 0U ||
      !all_zero(config->reserved, sizeof(config->reserved) / sizeof(config->reserved[0]))) {
    return invalid("session_create", "unknown flags or non-zero reserved field");
  }
  if (!valid_context_mode(config->context_mode)) {
    return invalid("session_create", "invalid context mode");
  }
  if (!valid_cache_policy(config->cache_policy)) {
    return invalid("session_create", "invalid cache policy");
  }
  if (config->device_ordinal < 0) {
    return invalid("session_create", "device ordinal must be non-negative");
  }
  if (config->chunk_size_bytes == 0U) {
    return invalid("session_create", "chunk size must be non-zero");
  }
  if (config->staging_slots < 2U || config->staging_slots > 8U) {
    return invalid("session_create", "staging slot count must be in [2, 8]");
  }
  if (config->prefetch_distance > 8U) {
    return invalid("session_create", "prefetch distance must be in [0, 8]");
  }
  if (config->budget_poll_ms == 0U ||
      config->budget_poll_ms >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
      config->max_transaction_ms >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    return invalid("session_create", "budget and transaction durations are out of range");
  }
  return {};
}

[[nodiscard]] Error collect_accesses(const std::shared_ptr<SessionState>& owner,
                                     const xvram_access_range_v1* ranges,
                                     const std::uint64_t range_count, const bool allow_empty,
                                     std::vector<AccessRequest>& output,
                                     const std::string_view operation) {
  if (range_count == 0U) {
    if (!allow_empty) {
      return invalid(operation, "at least one access range is required");
    }
    return {};
  }
  if (ranges == nullptr) {
    return invalid(operation, "range array is null");
  }
  if (range_count > static_cast<std::uint64_t>(output.max_size())) {
    return invalid(operation, "range count is too large");
  }

  output.reserve(static_cast<std::size_t>(range_count));
  for (std::uint64_t index = 0; index < range_count; ++index) {
    const xvram_access_range_v1& range = ranges[index];
    if (range.allocation == nullptr || !range.allocation->backend) {
      return invalid(operation, "access range contains a null allocation");
    }
    if (range.allocation->owner.get() != owner.get()) {
      return invalid(operation, "all access ranges must belong to the session");
    }
    if (!valid_access_mode(range.mode) || range.reserved0 != 0U) {
      return invalid(operation, "access range has an invalid mode or reserved field");
    }
    if (range.length_bytes == 0U) {
      return invalid(operation, "access range length must be non-zero");
    }
    if (range.offset_bytes > std::numeric_limits<std::uint64_t>::max() - range.length_bytes) {
      return invalid(operation, "access range overflows uint64");
    }
    output.push_back(AccessRequest{range.allocation->backend, range.offset_bytes,
                                   range.length_bytes, range.mode});
  }
  return {};
}

[[nodiscard]] Error convert_matrix(const std::shared_ptr<SessionState>& owner,
                                   const xvram_matrix_v1& input, MatrixRequest& output,
                                   const std::string_view name) {
  if (input.struct_size < sizeof(xvram_matrix_v1)) {
    return incompatible("gemm_plan_create", "matrix descriptor is smaller than v1");
  }
  if (input.allocation == nullptr || !input.allocation->backend) {
    return invalid("gemm_plan_create", "matrix allocation is null");
  }
  if (input.allocation->owner.get() != owner.get()) {
    return invalid("gemm_plan_create", "matrix allocation belongs to another session");
  }
  if (!valid_data_type(input.data_type) || !valid_layout(input.layout) || input.reserved0 != 0U ||
      !all_zero(input.reserved, sizeof(input.reserved) / sizeof(input.reserved[0]))) {
    return invalid("gemm_plan_create", "matrix descriptor has invalid enum or reserved fields");
  }
  if (input.rows == 0U || input.columns == 0U || input.leading_dimension == 0U) {
    return invalid("gemm_plan_create", "matrix dimensions and leading dimension must be non-zero");
  }

  output = MatrixRequest{input.allocation->backend, input.offset_bytes, input.rows,  input.columns,
                         input.leading_dimension,   input.data_type,    input.layout};
  (void)name;
  return {};
}

[[nodiscard]] Error convert_gemm(const std::shared_ptr<SessionState>& owner,
                                 const xvram_gemm_desc_v1* desc, GemmRequest& output) {
  if (!has_v1_size(desc)) {
    return incompatible("gemm_plan_create", "GEMM descriptor is null or smaller than v1");
  }
  if (desc->flags != 0U || desc->reserved0 != 0U ||
      !all_zero(desc->reserved, sizeof(desc->reserved) / sizeof(desc->reserved[0]))) {
    return invalid("gemm_plan_create", "unknown GEMM flags or non-zero reserved field");
  }
  if (!valid_matrix_operation(desc->operation_a) || !valid_matrix_operation(desc->operation_b) ||
      !valid_compute_mode(desc->compute_mode)) {
    return invalid("gemm_plan_create", "invalid GEMM operation or compute mode");
  }
  if (Error error = xvram::sdk::validate_submission_deadline(desc->deadline_ms, "gemm_plan_create");
      error) {
    return error;
  }

  if (Error error = convert_matrix(owner, desc->a, output.a, "A"); error) {
    return error;
  }
  if (Error error = convert_matrix(owner, desc->b, output.b, "B"); error) {
    return error;
  }
  if (Error error = convert_matrix(owner, desc->c, output.c, "C"); error) {
    return error;
  }

  output.flags = desc->flags;
  output.operation_a = desc->operation_a;
  output.operation_b = desc->operation_b;
  output.compute_mode = desc->compute_mode;
  output.alpha = desc->alpha;
  output.beta = desc->beta;
  output.workspace_cap_bytes = desc->workspace_cap_bytes;
  output.deadline_ms = desc->deadline_ms;
  output.tile_m_hint = desc->tile_m_hint;
  output.tile_n_hint = desc->tile_n_hint;
  output.tile_k_hint = desc->tile_k_hint;
  return {};
}

[[nodiscard]] const char* XVRAM_CALL status_name_entry(const xvram_status status) noexcept {
  return xvram::sdk::status_name(status);
}

[[nodiscard]] xvram_status XVRAM_CALL session_create_entry(const xvram_session_config_v1* config,
                                                           xvram_session* out_session) noexcept {
  if (out_session == nullptr) {
    return XVRAM_STATUS_INVALID_ARGUMENT;
  }
  *out_session = nullptr;

  return boundary({}, "session_create", [&]() -> Error {
    if (Error error = validate_session_config(config); error) {
      return error;
    }
    const xvram::sdk::BackendFactoryProvider provider =
        xvram::sdk::factory_provider.load(std::memory_order_acquire);
    if (provider == nullptr) {
      return xvram::sdk::make_error(XVRAM_STATUS_UNAVAILABLE, "sdk", "session_create",
                                    "runtime backend is not installed");
    }
    xvram::sdk::BackendFactory* factory = provider();
    if (factory == nullptr) {
      return xvram::sdk::make_error(XVRAM_STATUS_UNAVAILABLE, "sdk", "session_create",
                                    "runtime backend provider returned null");
    }

    std::shared_ptr<xvram::sdk::BackendSession> backend;
    if (Error error = factory->create_session(*config, backend); error) {
      return error;
    }
    if (!backend) {
      return xvram::sdk::make_error(XVRAM_STATUS_INTERNAL, "sdk", "session_create",
                                    "backend returned success without a session");
    }

    const std::uint64_t effective_chunk_bytes = backend->chunk_size_bytes();
    if (effective_chunk_bytes == 0U) {
      return xvram::sdk::make_error(XVRAM_STATUS_INTERNAL, "sdk", "session_create",
                                    "backend reported a zero effective chunk size");
    }
    auto state = std::make_shared<SessionState>(std::move(backend), effective_chunk_bytes,
                                                config->workspace_cap_bytes);
    auto* handle = new (std::nothrow) xvram_session_t{std::move(state)};
    if (handle == nullptr) {
      return xvram::sdk::make_error(XVRAM_STATUS_HOST_OUT_OF_MEMORY, "sdk", "session_create",
                                    "failed to allocate session handle");
    }
    *out_session = handle;
    return {};
  });
}

[[nodiscard]] xvram_status XVRAM_CALL
session_get_error_entry(const xvram_session session, xvram_error_info_v1* out_error) noexcept {
  if (session == nullptr || !session->state || !has_v1_size(out_error)) {
    return XVRAM_STATUS_INVALID_ARGUMENT;
  }
  return boundary(session->state, "session_get_error", [&]() -> Error {
    xvram::sdk::write_error_info(session->state->error(), *out_error);
    return {};
  });
}

[[nodiscard]] xvram_status XVRAM_CALL session_get_telemetry_entry(
    const xvram_session session, xvram_session_telemetry_v1* out_telemetry) noexcept {
  if (session == nullptr || !session->state || !has_v1_size(out_telemetry)) {
    return XVRAM_STATUS_INVALID_ARGUMENT;
  }
  return boundary(session->state, "session_get_telemetry", [&]() -> Error {
    const std::uint32_t caller_size = out_telemetry->struct_size;
    xvram_session_telemetry_v1 local{};
    local.struct_size = sizeof(local);
    if (Error error = session->state->backend->telemetry(local); error) {
      return error;
    }
    local.struct_size = caller_size;
    *out_telemetry = local;
    return {};
  });
}

[[nodiscard]] xvram_status XVRAM_CALL session_drain_entry(const xvram_session session,
                                                          const uint64_t timeout_ms) noexcept {
  if (session == nullptr || !session->state) {
    return XVRAM_STATUS_INVALID_ARGUMENT;
  }
  return boundary(session->state, "session_drain",
                  [&]() { return session->state->backend->drain(timeout_ms); });
}

[[nodiscard]] xvram_status XVRAM_CALL session_close_entry(const xvram_session session,
                                                          const uint64_t timeout_ms) noexcept {
  if (session == nullptr || !session->state) {
    return XVRAM_STATUS_INVALID_ARGUMENT;
  }
  return boundary(session->state, "session_close", [&]() -> Error {
    if (session->state->closed.load(std::memory_order_acquire)) {
      return {};
    }
    Error error = session->state->backend->close(timeout_ms);
    if (!error) {
      session->state->closed.store(true, std::memory_order_release);
    }
    return error;
  });
}

void XVRAM_CALL session_release_entry(const xvram_session session) noexcept {
  if (session == nullptr) {
    return;
  }
  try {
    if (session->state && !session->state->closed.load(std::memory_order_acquire)) {
      Error error = session->state->backend->close(XVRAM_TIMEOUT_INFINITE);
      if (!error) {
        session->state->closed.store(true, std::memory_order_release);
      } else {
        session->state->record(error);
      }
    }
  } catch (...) {
    // A release function cannot report an exception; backend destructors remain the final guard.
  }
  delete session;
}

[[nodiscard]] xvram_status XVRAM_CALL
allocation_create_entry(const xvram_session session, const xvram_allocation_desc_v1* desc,
                        xvram_allocation* out_allocation) noexcept {
  if (out_allocation == nullptr) {
    return XVRAM_STATUS_INVALID_ARGUMENT;
  }
  *out_allocation = nullptr;
  if (session == nullptr || !session->state) {
    return XVRAM_STATUS_INVALID_ARGUMENT;
  }
  return boundary(session->state, "allocation_create", [&]() -> Error {
    if (!has_v1_size(desc)) {
      return incompatible("allocation_create", "allocation descriptor is null or smaller than v1");
    }
    if (desc->size_bytes == 0U) {
      return invalid("allocation_create", "allocation size must be non-zero");
    }
    if ((desc->flags & ~XVRAM_ALLOCATION_FLAG_ZERO_INITIALIZE) != 0U || desc->reserved0 != 0U ||
        !all_zero(desc->reserved, sizeof(desc->reserved) / sizeof(desc->reserved[0]))) {
      return invalid("allocation_create", "unknown flags or non-zero reserved field");
    }
    if (!valid_priority(desc->priority)) {
      return invalid("allocation_create", "invalid allocation priority");
    }
    if (desc->alignment_bytes != 0U &&
        (desc->alignment_bytes & (desc->alignment_bytes - 1U)) != 0U) {
      return invalid("allocation_create", "alignment must be zero or a power of two");
    }
    if (desc->alignment_bytes > session->state->chunk_size_bytes ||
        (desc->alignment_bytes != 0U &&
         session->state->chunk_size_bytes % desc->alignment_bytes != 0U)) {
      return invalid("allocation_create", "alignment must divide the effective runtime chunk size");
    }

    std::shared_ptr<xvram::sdk::BackendAllocation> backend;
    if (Error error = session->state->backend->allocate(*desc, backend); error) {
      return error;
    }
    if (!backend) {
      return xvram::sdk::make_error(XVRAM_STATUS_INTERNAL, "sdk", "allocation_create",
                                    "backend returned success without an allocation");
    }
    auto* handle = new (std::nothrow) xvram_allocation_t{session->state, std::move(backend)};
    if (handle == nullptr) {
      // Backend allocation is already live. Roll it back before reporting that the public handle
      // could not be materialized, otherwise a recoverable host OOM leaks logical backing.
      try {
        (void)backend->release();
      } catch (...) {
      }
      return xvram::sdk::make_error(XVRAM_STATUS_HOST_OUT_OF_MEMORY, "sdk", "allocation_create",
                                    "failed to allocate handle");
    }
    *out_allocation = handle;
    return {};
  });
}

[[nodiscard]] xvram_status XVRAM_CALL allocation_get_info_entry(
    const xvram_allocation allocation, xvram_allocation_info_v1* out_info) noexcept {
  if (allocation == nullptr || !allocation->owner || !allocation->backend ||
      !has_v1_size(out_info)) {
    return XVRAM_STATUS_INVALID_ARGUMENT;
  }
  return boundary(allocation->owner, "allocation_get_info", [&]() -> Error {
    const std::uint32_t caller_size = out_info->struct_size;
    xvram_allocation_info_v1 local{};
    local.struct_size = sizeof(local);
    if (Error error = allocation->backend->info(local); error) {
      return error;
    }
    local.struct_size = caller_size;
    *out_info = local;
    return {};
  });
}

[[nodiscard]] xvram_status XVRAM_CALL allocation_write_entry(const xvram_allocation allocation,
                                                             const uint64_t offset_bytes,
                                                             const void* source,
                                                             const uint64_t size_bytes) noexcept {
  if (allocation == nullptr || !allocation->owner || !allocation->backend ||
      (size_bytes != 0U && source == nullptr)) {
    return XVRAM_STATUS_INVALID_ARGUMENT;
  }
  return boundary(allocation->owner, "allocation_write",
                  [&]() { return allocation->backend->write(offset_bytes, source, size_bytes); });
}

[[nodiscard]] xvram_status XVRAM_CALL allocation_read_entry(const xvram_allocation allocation,
                                                            const uint64_t offset_bytes,
                                                            void* destination,
                                                            const uint64_t size_bytes) noexcept {
  if (allocation == nullptr || !allocation->owner || !allocation->backend ||
      (size_bytes != 0U && destination == nullptr)) {
    return XVRAM_STATUS_INVALID_ARGUMENT;
  }
  return boundary(allocation->owner, "allocation_read", [&]() {
    return allocation->backend->read(offset_bytes, destination, size_bytes);
  });
}

[[nodiscard]] xvram_status XVRAM_CALL
allocation_release_entry(const xvram_allocation allocation) noexcept {
  if (allocation == nullptr || !allocation->owner || !allocation->backend) {
    return XVRAM_STATUS_INVALID_ARGUMENT;
  }
  // A successful session close has already released all backing allocations. Handles are allowed
  // to outlive the session handle, so releasing one afterwards is a local, idempotent operation.
  if (allocation->owner->closed.load(std::memory_order_acquire)) {
    delete allocation;
    return XVRAM_STATUS_SUCCESS;
  }
  const xvram_status result = boundary(allocation->owner, "allocation_release",
                                       [&]() { return allocation->backend->release(); });
  if (result == XVRAM_STATUS_SUCCESS) {
    delete allocation;
  }
  return result;
}

[[nodiscard]] xvram_status XVRAM_CALL
prefetch_submit_entry(const xvram_session session, const xvram_prefetch_desc_v1* desc,
                      xvram_operation* out_operation) noexcept;
[[nodiscard]] xvram_status XVRAM_CALL operation_wait_entry(
    xvram_operation operation, uint64_t timeout_ms, xvram_operation_info_v1* out_info) noexcept;
void XVRAM_CALL operation_release_entry(xvram_operation operation) noexcept;

[[nodiscard]] Error wrap_operation(const std::shared_ptr<SessionState>& owner,
                                   std::shared_ptr<xvram::sdk::BackendOperation> backend,
                                   xvram_operation* output) {
  if (!backend) {
    return xvram::sdk::make_error(XVRAM_STATUS_INTERNAL, "sdk", "operation_create",
                                  "backend returned success without an operation");
  }
  auto* handle = new (std::nothrow) xvram_operation_t{owner, std::move(backend)};
  if (handle == nullptr) {
    // The backend may already have queued the operation. Cancellation is best-effort: queued
    // RuntimeOperations retire as CANCELLED, while a racing in-flight CUDA operation remains
    // event-safe and is drained by normal session lifecycle handling.
    try {
      (void)backend->cancel();
    } catch (...) {
    }
    return xvram::sdk::make_error(XVRAM_STATUS_HOST_OUT_OF_MEMORY, "sdk", "operation_create",
                                  "failed to allocate operation handle");
  }
  *output = handle;
  return {};
}

[[nodiscard]] xvram_status XVRAM_CALL
prefetch_submit_entry(const xvram_session session, const xvram_prefetch_desc_v1* desc,
                      xvram_operation* out_operation) noexcept {
  if (out_operation == nullptr) {
    return XVRAM_STATUS_INVALID_ARGUMENT;
  }
  *out_operation = nullptr;
  if (session == nullptr || !session->state) {
    return XVRAM_STATUS_INVALID_ARGUMENT;
  }
  return boundary(session->state, "prefetch_submit", [&]() -> Error {
    if (!has_v1_size(desc)) {
      return incompatible("prefetch_submit", "prefetch descriptor is null or smaller than v1");
    }
    if (desc->flags != 0U ||
        !all_zero(desc->reserved, sizeof(desc->reserved) / sizeof(desc->reserved[0]))) {
      return invalid("prefetch_submit", "unknown flags or non-zero reserved field");
    }
    if (Error error =
            xvram::sdk::validate_submission_deadline(desc->deadline_ms, "prefetch_submit");
        error) {
      return error;
    }
    PrefetchRequest request;
    request.flags = desc->flags;
    request.deadline_ms = desc->deadline_ms;
    if (Error error = collect_accesses(session->state, desc->ranges, desc->range_count, false,
                                       request.ranges, "prefetch_submit");
        error) {
      return error;
    }
    std::shared_ptr<xvram::sdk::BackendOperation> backend;
    if (Error error = session->state->backend->prefetch(request, backend); error) {
      return error;
    }
    return wrap_operation(session->state, std::move(backend), out_operation);
  });
}

[[nodiscard]] xvram_status XVRAM_CALL
transaction_submit_entry(const xvram_session session, const xvram_transaction_desc_v1* desc,
                         xvram_operation* out_operation) noexcept {
  if (out_operation == nullptr) {
    return XVRAM_STATUS_INVALID_ARGUMENT;
  }
  *out_operation = nullptr;
  if (session == nullptr || !session->state) {
    return XVRAM_STATUS_INVALID_ARGUMENT;
  }
  return boundary(session->state, "transaction_submit", [&]() -> Error {
    if (!has_v1_size(desc)) {
      return incompatible("transaction_submit",
                          "transaction descriptor is null or smaller than v1");
    }
    if (desc->flags != 0U ||
        !all_zero(desc->reserved, sizeof(desc->reserved) / sizeof(desc->reserved[0]))) {
      return invalid("transaction_submit", "unknown flags or non-zero reserved field");
    }
    if (desc->callback == nullptr) {
      return invalid("transaction_submit", "transaction callback is null");
    }
    if (desc->workspace_bytes > session->state->workspace_cap_bytes) {
      return invalid("transaction_submit", "transaction workspace exceeds the session cap");
    }
    if (Error error =
            xvram::sdk::validate_submission_deadline(desc->deadline_ms, "transaction_submit");
        error) {
      return error;
    }
    TransactionRequest request;
    request.flags = desc->flags;
    request.workspace_bytes = desc->workspace_bytes;
    request.deadline_ms = desc->deadline_ms;
    request.callback = desc->callback;
    request.user_data = desc->user_data;
    if (Error error = collect_accesses(session->state, desc->ranges, desc->range_count, false,
                                       request.ranges, "transaction_submit");
        error) {
      return error;
    }
    std::shared_ptr<xvram::sdk::BackendOperation> backend;
    if (Error error = session->state->backend->submit_transaction(request, backend); error) {
      return error;
    }
    return wrap_operation(session->state, std::move(backend), out_operation);
  });
}

[[nodiscard]] xvram_status XVRAM_CALL
transaction_execute_entry(const xvram_session session, const xvram_transaction_desc_v1* desc,
                          const uint64_t timeout_ms) noexcept {
  xvram_operation operation = nullptr;
  const xvram_status submit_status = transaction_submit_entry(session, desc, &operation);
  if (submit_status != XVRAM_STATUS_SUCCESS) {
    return submit_status;
  }
  xvram_operation_info_v1 info = XVRAM_OPERATION_INFO_V1_INIT;
  const xvram_status wait_status = operation_wait_entry(operation, timeout_ms, &info);
  if (wait_status != XVRAM_STATUS_SUCCESS) {
    operation_release_entry(operation);
    return wait_status;
  }
  xvram_status result = info.result;
  if (result != XVRAM_STATUS_SUCCESS) {
    result = boundary(operation->owner, "transaction_execute", [&]() -> Error {
      Error error = operation->backend->error();
      return error ? error
                   : xvram::sdk::make_error(info.result, "transaction", "execute",
                                            "transaction operation failed");
    });
  }
  operation_release_entry(operation);
  return result;
}

[[nodiscard]] xvram_status XVRAM_CALL gemm_plan_create_entry(const xvram_session session,
                                                             const xvram_gemm_desc_v1* desc,
                                                             xvram_gemm_plan* out_plan) noexcept {
  if (out_plan == nullptr) {
    return XVRAM_STATUS_INVALID_ARGUMENT;
  }
  *out_plan = nullptr;
  if (session == nullptr || !session->state) {
    return XVRAM_STATUS_INVALID_ARGUMENT;
  }
  return boundary(session->state, "gemm_plan_create", [&]() -> Error {
    GemmRequest request;
    if (Error error = convert_gemm(session->state, desc, request); error) {
      return error;
    }
    std::shared_ptr<xvram::sdk::BackendGemmPlan> backend;
    if (Error error = session->state->backend->create_gemm_plan(request, backend); error) {
      return error;
    }
    if (!backend) {
      return xvram::sdk::make_error(XVRAM_STATUS_INTERNAL, "sdk", "gemm_plan_create",
                                    "backend returned success without a plan");
    }
    auto* handle = new (std::nothrow) xvram_gemm_plan_t{session->state, std::move(backend)};
    if (handle == nullptr) {
      return xvram::sdk::make_error(XVRAM_STATUS_HOST_OUT_OF_MEMORY, "sdk", "gemm_plan_create",
                                    "failed to allocate plan handle");
    }
    *out_plan = handle;
    return {};
  });
}

[[nodiscard]] xvram_status XVRAM_CALL
gemm_plan_get_info_entry(const xvram_gemm_plan plan, xvram_gemm_plan_info_v1* out_info) noexcept {
  if (plan == nullptr || !plan->owner || !plan->backend || !has_v1_size(out_info)) {
    return XVRAM_STATUS_INVALID_ARGUMENT;
  }
  return boundary(plan->owner, "gemm_plan_get_info", [&]() -> Error {
    const std::uint32_t caller_size = out_info->struct_size;
    xvram_gemm_plan_info_v1 local = XVRAM_GEMM_PLAN_INFO_V1_INIT;
    if (Error error = plan->backend->info(local); error) {
      return error;
    }
    local.struct_size = caller_size;
    *out_info = local;
    return {};
  });
}

[[nodiscard]] xvram_status XVRAM_CALL gemm_submit_entry(const xvram_gemm_plan plan,
                                                        xvram_operation* out_operation) noexcept {
  if (out_operation == nullptr) {
    return XVRAM_STATUS_INVALID_ARGUMENT;
  }
  *out_operation = nullptr;
  if (plan == nullptr || !plan->owner || !plan->backend) {
    return XVRAM_STATUS_INVALID_ARGUMENT;
  }
  return boundary(plan->owner, "gemm_submit", [&]() -> Error {
    std::shared_ptr<xvram::sdk::BackendOperation> backend;
    if (Error error = plan->backend->submit(backend); error) {
      return error;
    }
    return wrap_operation(plan->owner, std::move(backend), out_operation);
  });
}

[[nodiscard]] xvram_status XVRAM_CALL gemm_execute_entry(const xvram_gemm_plan plan,
                                                         const uint64_t timeout_ms) noexcept {
  xvram_operation operation = nullptr;
  const xvram_status submit_status = gemm_submit_entry(plan, &operation);
  if (submit_status != XVRAM_STATUS_SUCCESS) {
    return submit_status;
  }
  xvram_operation_info_v1 info = XVRAM_OPERATION_INFO_V1_INIT;
  const xvram_status wait_status = operation_wait_entry(operation, timeout_ms, &info);
  if (wait_status != XVRAM_STATUS_SUCCESS) {
    operation_release_entry(operation);
    return wait_status;
  }
  xvram_status result = info.result;
  if (result != XVRAM_STATUS_SUCCESS) {
    result = boundary(operation->owner, "gemm_execute", [&]() -> Error {
      Error error = operation->backend->error();
      return error
                 ? error
                 : xvram::sdk::make_error(info.result, "gemm", "execute", "GEMM operation failed");
    });
  }
  operation_release_entry(operation);
  return result;
}

void XVRAM_CALL gemm_plan_release_entry(const xvram_gemm_plan plan) noexcept {
  try {
    delete plan;
  } catch (...) {
  }
}

[[nodiscard]] xvram_status XVRAM_CALL
operation_poll_entry(const xvram_operation operation, xvram_operation_info_v1* out_info) noexcept {
  if (operation == nullptr || !operation->owner || !operation->backend || !has_v1_size(out_info)) {
    return XVRAM_STATUS_INVALID_ARGUMENT;
  }
  return boundary(operation->owner, "operation_poll", [&]() -> Error {
    const std::uint32_t caller_size = out_info->struct_size;
    xvram_operation_info_v1 local = XVRAM_OPERATION_INFO_V1_INIT;
    Error error = operation->backend->poll(local);
    local.struct_size = caller_size;
    *out_info = local;
    return error;
  });
}

[[nodiscard]] xvram_status XVRAM_CALL
operation_wait_entry(const xvram_operation operation, const uint64_t timeout_ms,
                     xvram_operation_info_v1* out_info) noexcept {
  if (operation == nullptr || !operation->owner || !operation->backend || !has_v1_size(out_info)) {
    return XVRAM_STATUS_INVALID_ARGUMENT;
  }
  return boundary(operation->owner, "operation_wait", [&]() -> Error {
    const std::uint32_t caller_size = out_info->struct_size;
    xvram_operation_info_v1 local = XVRAM_OPERATION_INFO_V1_INIT;
    Error error = operation->backend->wait(timeout_ms, local);
    local.struct_size = caller_size;
    *out_info = local;
    return error;
  });
}

[[nodiscard]] xvram_status XVRAM_CALL
operation_cancel_entry(const xvram_operation operation) noexcept {
  if (operation == nullptr || !operation->owner || !operation->backend) {
    return XVRAM_STATUS_INVALID_ARGUMENT;
  }
  return boundary(operation->owner, "operation_cancel",
                  [&]() { return operation->backend->cancel(); });
}

[[nodiscard]] xvram_status XVRAM_CALL operation_get_error_entry(
    const xvram_operation operation, xvram_error_info_v1* out_error) noexcept {
  if (operation == nullptr || !operation->owner || !operation->backend || !has_v1_size(out_error)) {
    return XVRAM_STATUS_INVALID_ARGUMENT;
  }
  return boundary(operation->owner, "operation_get_error", [&]() -> Error {
    xvram::sdk::write_error_info(operation->backend->error(), *out_error);
    return {};
  });
}

void XVRAM_CALL operation_release_entry(const xvram_operation operation) noexcept {
  try {
    delete operation;
  } catch (...) {
  }
}

[[nodiscard]] const xvram_api_v1& api_v1() noexcept {
  static const xvram_api_v1 api = [] {
    xvram_api_v1 value{};
    value.struct_size = sizeof(value);
    value.abi_version = XVRAM_ABI_VERSION_1;
    value.status_name = &status_name_entry;
    value.session_create = &session_create_entry;
    value.session_get_error = &session_get_error_entry;
    value.session_get_telemetry = &session_get_telemetry_entry;
    value.session_drain = &session_drain_entry;
    value.session_close = &session_close_entry;
    value.session_release = &session_release_entry;
    value.allocation_create = &allocation_create_entry;
    value.allocation_get_info = &allocation_get_info_entry;
    value.allocation_write = &allocation_write_entry;
    value.allocation_read = &allocation_read_entry;
    value.allocation_release = &allocation_release_entry;
    value.prefetch_submit = &prefetch_submit_entry;
    value.transaction_submit = &transaction_submit_entry;
    value.transaction_execute = &transaction_execute_entry;
    value.gemm_plan_create = &gemm_plan_create_entry;
    value.gemm_plan_get_info = &gemm_plan_get_info_entry;
    value.gemm_submit = &gemm_submit_entry;
    value.gemm_execute = &gemm_execute_entry;
    value.gemm_plan_release = &gemm_plan_release_entry;
    value.operation_poll = &operation_poll_entry;
    value.operation_wait = &operation_wait_entry;
    value.operation_cancel = &operation_cancel_entry;
    value.operation_get_error = &operation_get_error_entry;
    value.operation_release = &operation_release_entry;
    return value;
  }();
  return api;
}

} // namespace

extern "C" XVRAM_API xvram_status XVRAM_CALL xvram_get_api(const uint32_t requested_abi_version,
                                                           const uint32_t caller_api_size,
                                                           void* out_api) {
  if (requested_abi_version != XVRAM_ABI_VERSION_1) {
    return XVRAM_STATUS_INCOMPATIBLE_ABI;
  }
  if (out_api == nullptr || caller_api_size < sizeof(xvram_api_v1)) {
    return XVRAM_STATUS_INVALID_ARGUMENT;
  }
  std::memcpy(out_api, &api_v1(), sizeof(xvram_api_v1));
  return XVRAM_STATUS_SUCCESS;
}
