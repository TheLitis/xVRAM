#include "sdk/backend.hpp"
#include "xvram/xvram.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

static_assert(sizeof(void*) == 8);
static_assert(std::is_standard_layout_v<xvram_error_info_v1>);
static_assert(std::is_standard_layout_v<xvram_session_config_v1>);
static_assert(std::is_standard_layout_v<xvram_api_v1>);
static_assert(sizeof(xvram_error_info_v1) == 440);
static_assert(sizeof(xvram_session_config_v1) == 112);
static_assert(sizeof(xvram_allocation_desc_v1) == 48);
static_assert(sizeof(xvram_allocation_info_v1) == 56);
static_assert(sizeof(xvram_access_range_v1) == 32);
static_assert(sizeof(xvram_prefetch_desc_v1) == 48);
static_assert(sizeof(xvram_resolved_range_v1) == 40);
static_assert(sizeof(xvram_transaction_context_v1) == 80);
static_assert(sizeof(xvram_transaction_desc_v1) == 72);
static_assert(sizeof(xvram_operation_info_v1) == 96);
static_assert(sizeof(xvram_matrix_v1) == 72);
static_assert(sizeof(xvram_gemm_desc_v1) == 320);
static_assert(sizeof(xvram_gemm_plan_info_v1) == 104);
static_assert(sizeof(xvram_session_telemetry_v1) == 320);
static_assert(sizeof(xvram_api_v1) == 336);
static_assert(offsetof(xvram_api_v1, status_name) == 8);
static_assert(offsetof(xvram_api_v1, reserved) == 208);

namespace {

int failures = 0;
int fake_close_calls = 0;
int fake_close_timeouts_remaining = 0;
int fake_allocation_release_calls = 0;
bool fake_backend_closed = false;
bool fake_gemm_failure = false;

void check(const bool condition, const char* expression, const int line) {
  if (!condition) {
    std::cerr << "line " << line << ": check failed: " << expression << '\n';
    ++failures;
  }
}

#define CHECK(expression) check(static_cast<bool>(expression), #expression, __LINE__)

class FakeOperation final : public xvram::sdk::BackendOperation {
public:
  explicit FakeOperation(xvram::sdk::Error error = {}) : error_(std::move(error)) {}

  [[nodiscard]] xvram::sdk::Error poll(xvram_operation_info_v1& output) override {
    fill(output);
    return {};
  }

  [[nodiscard]] xvram::sdk::Error wait(const std::uint64_t,
                                       xvram_operation_info_v1& output) override {
    fill(output);
    return {};
  }

  [[nodiscard]] xvram::sdk::Error cancel() override {
    return xvram::sdk::make_error(XVRAM_STATUS_BUSY, "fake", "cancel",
                                  "operation already completed");
  }

  [[nodiscard]] xvram::sdk::Error error() const override {
    return error_;
  }

private:
  void fill(xvram_operation_info_v1& output) const {
    output.state = error_ ? XVRAM_OPERATION_FAILED : XVRAM_OPERATION_COMPLETED;
    output.result = error_.status;
    output.operation_id = 41;
    output.units_completed = error_ ? 0 : 1;
    output.units_total = 1;
  }

  xvram::sdk::Error error_;
};

class FakeAllocation final : public xvram::sdk::BackendAllocation {
public:
  [[nodiscard]] xvram::sdk::Error info(xvram_allocation_info_v1& output) const override {
    output.allocation_id = 7;
    output.size_bytes = bytes_.size();
    output.priority = XVRAM_ALLOCATION_HOT;
    output.stable_virtual_address = 1;
    return {};
  }

  [[nodiscard]] xvram::sdk::Error write(const std::uint64_t offset, const void* source,
                                        const std::uint64_t size) override {
    if (offset > bytes_.size() || size > bytes_.size() - offset) {
      return xvram::sdk::make_error(XVRAM_STATUS_INVALID_ARGUMENT, "fake", "write",
                                    "out of bounds");
    }
    if (size != 0U) {
      std::memcpy(bytes_.data() + offset, source, static_cast<std::size_t>(size));
    }
    return {};
  }

  [[nodiscard]] xvram::sdk::Error read(const std::uint64_t offset, void* destination,
                                       const std::uint64_t size) override {
    if (offset > bytes_.size() || size > bytes_.size() - offset) {
      return xvram::sdk::make_error(XVRAM_STATUS_INVALID_ARGUMENT, "fake", "read", "out of bounds");
    }
    if (size != 0U) {
      std::memcpy(destination, bytes_.data() + offset, static_cast<std::size_t>(size));
    }
    return {};
  }

  [[nodiscard]] xvram::sdk::Error release() override {
    ++fake_allocation_release_calls;
    if (fake_backend_closed) {
      return xvram::sdk::make_error(XVRAM_STATUS_CLOSED, "fake", "allocation_release",
                                    "session backing is already closed");
    }
    released_ = true;
    return {};
  }

private:
  std::vector<std::uint8_t> bytes_ = std::vector<std::uint8_t>(1024);
  bool released_ = false;
};

class FakePlan final : public xvram::sdk::BackendGemmPlan {
public:
  [[nodiscard]] xvram::sdk::Error info(xvram_gemm_plan_info_v1& output) const override {
    output.m = 4;
    output.n = 4;
    output.k = 4;
    output.tile_m = 4;
    output.tile_n = 4;
    output.tile_k = 4;
    output.tile_count = 1;
    output.effective_compute_mode = XVRAM_COMPUTE_FP32_TF32;
    output.uses_cublas_lt = 1;
    return {};
  }

  [[nodiscard]] xvram::sdk::Error
  submit(std::shared_ptr<xvram::sdk::BackendOperation>& output) override {
    output =
        fake_gemm_failure
            ? std::make_shared<FakeOperation>(xvram::sdk::make_native_error(
                  XVRAM_STATUS_CUBLAS_ERROR, XVRAM_NATIVE_ERROR_CUBLAS, 13, "gemm", "cublasGemmEx",
                  "CUBLAS_STATUS_EXECUTION_FAILED", "injected GEMM execution failure"))
            : std::make_shared<FakeOperation>();
    return {};
  }
};

class FakeSession final : public xvram::sdk::BackendSession {
public:
  [[nodiscard]] xvram::sdk::Error
  allocate(const xvram_allocation_desc_v1&,
           std::shared_ptr<xvram::sdk::BackendAllocation>& output) override {
    output = std::make_shared<FakeAllocation>();
    return {};
  }

  [[nodiscard]] xvram::sdk::Error
  prefetch(const xvram::sdk::PrefetchRequest&,
           std::shared_ptr<xvram::sdk::BackendOperation>& output) override {
    output = std::make_shared<FakeOperation>();
    return {};
  }

  [[nodiscard]] xvram::sdk::Error
  submit_transaction(const xvram::sdk::TransactionRequest& request,
                     std::shared_ptr<xvram::sdk::BackendOperation>& output) override {
    xvram_error_info_v1 callback_error = XVRAM_ERROR_INFO_V1_INIT;
    xvram_transaction_context_v1 context{};
    context.struct_size = sizeof(context);
    context.operation_id = 41;
    context.native_stream = 99;
    const xvram_status status = request.callback(&context, request.user_data, &callback_error);
    if (status != XVRAM_STATUS_SUCCESS) {
      return xvram::sdk::make_error(XVRAM_STATUS_CALLBACK_FAILED, "fake", "callback",
                                    callback_error.message);
    }
    output = std::make_shared<FakeOperation>();
    return {};
  }

  [[nodiscard]] xvram::sdk::Error
  create_gemm_plan(const xvram::sdk::GemmRequest&,
                   std::shared_ptr<xvram::sdk::BackendGemmPlan>& output) override {
    output = std::make_shared<FakePlan>();
    return {};
  }

  [[nodiscard]] xvram::sdk::Error telemetry(xvram_session_telemetry_v1& output) const override {
    output.cache_target_bytes = 512;
    output.cublas_version = 130501;
    output.cublas_lt_available = 1;
    return {};
  }

  [[nodiscard]] xvram::sdk::Error drain(const std::uint64_t) override {
    return {};
  }

  [[nodiscard]] xvram::sdk::Error close(const std::uint64_t) override {
    ++fake_close_calls;
    if (fake_close_timeouts_remaining > 0) {
      --fake_close_timeouts_remaining;
      return xvram::sdk::make_error(XVRAM_STATUS_TIMEOUT, "fake", "close",
                                    "injected close timeout");
    }
    fake_backend_closed = true;
    return {};
  }
};

class FakeFactory final : public xvram::sdk::BackendFactory {
public:
  [[nodiscard]] xvram::sdk::Error
  create_session(const xvram_session_config_v1&,
                 std::shared_ptr<xvram::sdk::BackendSession>& output) override {
    fake_backend_closed = false;
    output = std::make_shared<FakeSession>();
    return {};
  }
};

FakeFactory factory;

[[nodiscard]] xvram::sdk::BackendFactory* provide_factory() noexcept {
  return &factory;
}

xvram_status XVRAM_CALL callback(const xvram_transaction_context_v1* context, void* user_data,
                                 xvram_error_info_v1*) {
  CHECK(context != nullptr);
  CHECK(context->native_stream == 99);
  CHECK(user_data != nullptr);
  *static_cast<bool*>(user_data) = true;
  return XVRAM_STATUS_SUCCESS;
}

void retrieval_tests() {
  xvram_api_v1 api{};
  CHECK(xvram_get_api(99, sizeof(api), &api) == XVRAM_STATUS_INCOMPATIBLE_ABI);
  CHECK(xvram_get_api(XVRAM_ABI_VERSION_1, sizeof(api) - 1U, &api) ==
        XVRAM_STATUS_INVALID_ARGUMENT);
  CHECK(xvram_get_api(XVRAM_ABI_VERSION_1, sizeof(api), &api) == XVRAM_STATUS_SUCCESS);
  CHECK(api.struct_size == sizeof(api));
  CHECK(api.abi_version == XVRAM_ABI_VERSION_1);
  CHECK(std::string_view{api.status_name(XVRAM_STATUS_BUDGET_PRESSURE)} == "budget_pressure");
  CHECK(api.session_create != nullptr);
  CHECK(api.operation_release != nullptr);
}

void adapter_tests() {
  xvram::sdk::install_backend_factory(&provide_factory);
  fake_close_calls = 0;
  fake_close_timeouts_remaining = 0;
  fake_allocation_release_calls = 0;
  fake_backend_closed = false;
  fake_gemm_failure = false;

  xvram_api_v1 api{};
  CHECK(xvram_get_api(XVRAM_ABI_VERSION_1, sizeof(api), &api) == XVRAM_STATUS_SUCCESS);
  xvram_session_config_v1 config = XVRAM_SESSION_CONFIG_V1_INIT;
  xvram_session session = nullptr;
  CHECK(api.session_create(&config, &session) == XVRAM_STATUS_SUCCESS);
  CHECK(session != nullptr);

  xvram_allocation_desc_v1 allocation_desc = XVRAM_ALLOCATION_DESC_V1_INIT;
  allocation_desc.size_bytes = 1024;
  allocation_desc.priority = XVRAM_ALLOCATION_HOT;
  xvram_allocation allocation = nullptr;
  CHECK(api.allocation_create(session, &allocation_desc, &allocation) == XVRAM_STATUS_SUCCESS);

  const std::uint32_t source = 0x58565241U;
  std::uint32_t destination = 0;
  CHECK(api.allocation_write(allocation, 4, &source, sizeof(source)) == XVRAM_STATUS_SUCCESS);
  CHECK(api.allocation_read(allocation, 4, &destination, sizeof(destination)) ==
        XVRAM_STATUS_SUCCESS);
  CHECK(destination == source);

  xvram_allocation_info_v1 allocation_info{};
  allocation_info.struct_size = sizeof(allocation_info);
  CHECK(api.allocation_get_info(allocation, &allocation_info) == XVRAM_STATUS_SUCCESS);
  CHECK(allocation_info.allocation_id == 7);
  CHECK(allocation_info.stable_virtual_address == 1);

  xvram_access_range_v1 range{allocation, 0, 64, XVRAM_ACCESS_READ_WRITE, 0};

  xvram_prefetch_desc_v1 overflow_prefetch{};
  overflow_prefetch.struct_size = sizeof(overflow_prefetch);
  overflow_prefetch.ranges = &range;
  overflow_prefetch.range_count = 1;
  overflow_prefetch.deadline_ms = std::numeric_limits<std::uint64_t>::max();
  xvram_operation rejected_operation = nullptr;
  CHECK(api.prefetch_submit(session, &overflow_prefetch, &rejected_operation) ==
        XVRAM_STATUS_INVALID_ARGUMENT);
  CHECK(rejected_operation == nullptr);

  xvram_prefetch_desc_v1 prefetch{};
  prefetch.struct_size = sizeof(prefetch);
  prefetch.ranges = &range;
  prefetch.range_count = 1;
  xvram_operation prefetch_operation = nullptr;
  CHECK(api.prefetch_submit(session, &prefetch, &prefetch_operation) == XVRAM_STATUS_SUCCESS);
  CHECK(prefetch_operation != nullptr);
  xvram_operation_info_v1 operation_info = XVRAM_OPERATION_INFO_V1_INIT;
  CHECK(api.operation_poll(prefetch_operation, &operation_info) == XVRAM_STATUS_SUCCESS);
  CHECK(operation_info.state == XVRAM_OPERATION_COMPLETED);
  CHECK(operation_info.result == XVRAM_STATUS_SUCCESS);
  operation_info = XVRAM_OPERATION_INFO_V1_INIT;
  CHECK(api.operation_wait(prefetch_operation, 1000, &operation_info) == XVRAM_STATUS_SUCCESS);
  CHECK(operation_info.units_completed == operation_info.units_total);
  xvram_error_info_v1 operation_error = XVRAM_ERROR_INFO_V1_INIT;
  CHECK(api.operation_get_error(prefetch_operation, &operation_error) == XVRAM_STATUS_SUCCESS);
  CHECK(operation_error.status == XVRAM_STATUS_SUCCESS);
  CHECK(api.operation_cancel(prefetch_operation) == XVRAM_STATUS_BUSY);
  api.operation_release(prefetch_operation);

  bool callback_called = false;
  xvram_transaction_desc_v1 transaction{};
  transaction.struct_size = sizeof(transaction);
  transaction.ranges = &range;
  transaction.range_count = 1;
  transaction.callback = &callback;
  transaction.user_data = &callback_called;
  transaction.deadline_ms = std::numeric_limits<std::uint64_t>::max();
  CHECK(api.transaction_submit(session, &transaction, &rejected_operation) ==
        XVRAM_STATUS_INVALID_ARGUMENT);
  CHECK(rejected_operation == nullptr);
  transaction.deadline_ms = 0;
  transaction.workspace_bytes = config.workspace_cap_bytes + 1U;
  CHECK(api.transaction_submit(session, &transaction, &rejected_operation) ==
        XVRAM_STATUS_INVALID_ARGUMENT);
  CHECK(rejected_operation == nullptr);
  CHECK(!callback_called);
  transaction.workspace_bytes = 0;

  xvram_operation transaction_operation = nullptr;
  CHECK(api.transaction_submit(session, &transaction, &transaction_operation) ==
        XVRAM_STATUS_SUCCESS);
  CHECK(callback_called);
  operation_info = XVRAM_OPERATION_INFO_V1_INIT;
  CHECK(api.operation_wait(transaction_operation, 1000, &operation_info) == XVRAM_STATUS_SUCCESS);
  api.operation_release(transaction_operation);

  callback_called = false;
  CHECK(api.transaction_execute(session, &transaction, 1000) == XVRAM_STATUS_SUCCESS);
  CHECK(callback_called);

  xvram_gemm_desc_v1 gemm = XVRAM_GEMM_DESC_V1_INIT;
  gemm.a.allocation = allocation;
  gemm.b.allocation = allocation;
  gemm.c.allocation = allocation;
  gemm.a.rows = gemm.a.columns = gemm.a.leading_dimension = 4;
  gemm.b.rows = gemm.b.columns = gemm.b.leading_dimension = 4;
  gemm.c.rows = gemm.c.columns = gemm.c.leading_dimension = 4;
  xvram_gemm_plan plan = nullptr;
  gemm.deadline_ms = std::numeric_limits<std::uint64_t>::max();
  CHECK(api.gemm_plan_create(session, &gemm, &plan) == XVRAM_STATUS_INVALID_ARGUMENT);
  CHECK(plan == nullptr);
  gemm.deadline_ms = 0;
  CHECK(api.gemm_plan_create(session, &gemm, &plan) == XVRAM_STATUS_SUCCESS);
  xvram_gemm_plan_info_v1 plan_info = XVRAM_GEMM_PLAN_INFO_V1_INIT;
  CHECK(api.gemm_plan_get_info(plan, &plan_info) == XVRAM_STATUS_SUCCESS);
  CHECK(plan_info.tile_count == 1);

  xvram_operation gemm_operation = nullptr;
  CHECK(api.gemm_submit(plan, &gemm_operation) == XVRAM_STATUS_SUCCESS);
  operation_info = XVRAM_OPERATION_INFO_V1_INIT;
  CHECK(api.operation_wait(gemm_operation, 1000, &operation_info) == XVRAM_STATUS_SUCCESS);
  CHECK(operation_info.operation_id == 41);
  api.operation_release(gemm_operation);

  CHECK(api.gemm_execute(plan, 1000) == XVRAM_STATUS_SUCCESS);

  fake_gemm_failure = true;
  CHECK(api.gemm_execute(plan, 1000) == XVRAM_STATUS_CUBLAS_ERROR);
  xvram_error_info_v1 failed_gemm = XVRAM_ERROR_INFO_V1_INIT;
  CHECK(api.session_get_error(session, &failed_gemm) == XVRAM_STATUS_SUCCESS);
  CHECK(failed_gemm.status == XVRAM_STATUS_CUBLAS_ERROR);
  CHECK(failed_gemm.native_domain == XVRAM_NATIVE_ERROR_CUBLAS);
  CHECK(failed_gemm.native_code == 13);
  CHECK(std::string_view{failed_gemm.operation} == "cublasGemmEx");
  fake_gemm_failure = false;

  xvram_session_telemetry_v1 telemetry{};
  telemetry.struct_size = sizeof(telemetry);
  CHECK(api.session_get_telemetry(session, &telemetry) == XVRAM_STATUS_SUCCESS);
  CHECK(telemetry.cublas_lt_available == 1);
  xvram_error_info_v1 session_error = XVRAM_ERROR_INFO_V1_INIT;
  CHECK(api.session_get_error(session, &session_error) == XVRAM_STATUS_SUCCESS);
  CHECK(session_error.status == XVRAM_STATUS_CUBLAS_ERROR);
  CHECK(api.session_drain(session, 1000) == XVRAM_STATUS_SUCCESS);

  api.gemm_plan_release(plan);
  CHECK(api.allocation_release(allocation) == XVRAM_STATUS_SUCCESS);
  CHECK(api.session_close(session, 1000) == XVRAM_STATUS_SUCCESS);
  api.session_release(session);
}

void lifecycle_adapter_tests() {
  xvram_api_v1 api{};
  CHECK(xvram_get_api(XVRAM_ABI_VERSION_1, sizeof(api), &api) == XVRAM_STATUS_SUCCESS);

  fake_close_calls = 0;
  fake_close_timeouts_remaining = 0;
  fake_allocation_release_calls = 0;
  fake_backend_closed = false;

  xvram_session_config_v1 config = XVRAM_SESSION_CONFIG_V1_INIT;
  xvram_session session = nullptr;
  CHECK(api.session_create(&config, &session) == XVRAM_STATUS_SUCCESS);

  xvram_allocation_desc_v1 oversized_alignment = XVRAM_ALLOCATION_DESC_V1_INIT;
  oversized_alignment.size_bytes = 1024;
  oversized_alignment.alignment_bytes = config.chunk_size_bytes * 2U;
  xvram_allocation rejected = nullptr;
  CHECK(api.allocation_create(session, &oversized_alignment, &rejected) ==
        XVRAM_STATUS_INVALID_ARGUMENT);
  CHECK(rejected == nullptr);

  xvram_allocation_desc_v1 valid = XVRAM_ALLOCATION_DESC_V1_INIT;
  valid.size_bytes = 1024;
  valid.alignment_bytes = config.chunk_size_bytes;
  xvram_allocation surviving = nullptr;
  CHECK(api.allocation_create(session, &valid, &surviving) == XVRAM_STATUS_SUCCESS);

  fake_close_timeouts_remaining = 1;
  CHECK(api.session_close(session, 0) == XVRAM_STATUS_TIMEOUT);
  CHECK(fake_close_calls == 1);
  CHECK(api.session_close(session, 1000) == XVRAM_STATUS_SUCCESS);
  CHECK(fake_close_calls == 2);

  // The backend has already released its allocations. Public allocation release must therefore
  // be purely logical and must not invoke the now-closed backend allocation.
  CHECK(api.allocation_release(surviving) == XVRAM_STATUS_SUCCESS);
  CHECK(fake_allocation_release_calls == 0);
  api.session_release(session);

  // session_release is the unconditional safety net: after a bounded close times out it retries
  // the same backend close with an infinite timeout before destroying the handle.
  fake_close_calls = 0;
  fake_close_timeouts_remaining = 1;
  fake_backend_closed = false;
  session = nullptr;
  CHECK(api.session_create(&config, &session) == XVRAM_STATUS_SUCCESS);
  CHECK(api.session_close(session, 0) == XVRAM_STATUS_TIMEOUT);
  api.session_release(session);
  CHECK(fake_close_calls == 2);
  CHECK(fake_backend_closed);
}

} // namespace

int main() {
  retrieval_tests();
  adapter_tests();
  lifecycle_adapter_tests();
  if (failures != 0) {
    std::cerr << failures << " SDK ABI test(s) failed\n";
    return 1;
  }
  return 0;
}
