#include "xvram/internal/torch_runtime_v2.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>

static_assert(sizeof(xvram_torch_runtime_session_config_v2) == 240);
static_assert(sizeof(xvram_torch_runtime_telemetry_v2) == 864);
static_assert(sizeof(xvram_torch_runtime_api_v2) == 400);
static_assert(offsetof(xvram_torch_runtime_api_v2, v1) == 0);
static_assert(offsetof(xvram_torch_runtime_api_v2, reserved) == 288);

namespace {

int failures = 0;

void check(const bool condition, const char* expression, const int line) {
  if (!condition) {
    std::cerr << "line " << line << ": " << expression << '\n';
    ++failures;
  }
}

#define CHECK(expression) check((expression), #expression, __LINE__)

void telemetry_v2_initializer_test() {
  constexpr xvram_torch_runtime_telemetry_v2 initialized = XVRAM_TORCH_RUNTIME_TELEMETRY_V2_INIT;
  static_assert(initialized.struct_size == sizeof(xvram_torch_runtime_telemetry_v2));
  static_assert(initialized.abi_version == XVRAM_TORCH_RUNTIME_ABI_VERSION_2);
  static_assert(initialized.v1.struct_size == sizeof(xvram_torch_runtime_telemetry_v1));
  static_assert(initialized.v1.abi_version == XVRAM_TORCH_RUNTIME_ABI_VERSION_1);
  const auto* bytes = reinterpret_cast<const unsigned char*>(&initialized);
  const auto extension = offsetof(xvram_torch_runtime_telemetry_v2, logical_bytes);
  CHECK(std::all_of(bytes + extension, bytes + sizeof(initialized),
                    [](const unsigned char value) { return value == 0U; }));
}

} // namespace

int main() {
  telemetry_v2_initializer_test();
  std::array<std::byte, sizeof(xvram_torch_runtime_api_v1)> storage{};
  CHECK(xvram_torch_runtime_get_api(XVRAM_TORCH_RUNTIME_ABI_VERSION_1,
                                    static_cast<std::uint32_t>(storage.size() - 1U),
                                    storage.data()) == XVRAM_TORCH_RUNTIME_INCOMPATIBLE_ABI);
  CHECK(xvram_torch_runtime_get_api(UINT32_C(999), static_cast<std::uint32_t>(storage.size()),
                                    storage.data()) == XVRAM_TORCH_RUNTIME_INCOMPATIBLE_ABI);

  auto* api = reinterpret_cast<xvram_torch_runtime_api_v1*>(storage.data());
  CHECK(xvram_torch_runtime_get_api(XVRAM_TORCH_RUNTIME_ABI_VERSION_1,
                                    static_cast<std::uint32_t>(storage.size()),
                                    api) == XVRAM_TORCH_RUNTIME_SUCCESS);
  CHECK(api->struct_size == sizeof(xvram_torch_runtime_api_v1));
  CHECK(api->abi_version == XVRAM_TORCH_RUNTIME_ABI_VERSION_1);
  CHECK(api->status_name != nullptr);
  CHECK(api->session_create != nullptr);
  CHECK(api->session_get_error != nullptr);
  CHECK(api->session_get_telemetry != nullptr);
  CHECK(api->session_close != nullptr);
  CHECK(api->allocation_create != nullptr);
  CHECK(api->allocation_get_info != nullptr);
  CHECK(api->allocation_write != nullptr);
  CHECK(api->allocation_read != nullptr);
  CHECK(api->allocation_release != nullptr);
  CHECK(api->allocation_discard != nullptr);
  CHECK(api->session_prefetch != nullptr);
  CHECK(api->lease_acquire != nullptr);
  CHECK(api->lease_get_info != nullptr);
  CHECK(api->lease_seal != nullptr);
  CHECK(api->lease_poll != nullptr);
  CHECK(api->lease_wait != nullptr);
  CHECK(api->gemm_execute != nullptr);
  CHECK(api->status_name(XVRAM_TORCH_RUNTIME_VIEWS_LIVE) != nullptr);
  CHECK(api->status_name(XVRAM_TORCH_RUNTIME_CUBLAS_ERROR) != nullptr);

  std::array<std::byte, sizeof(xvram_torch_runtime_api_v2)> storage_v2{};
  CHECK(xvram_torch_runtime_get_api(XVRAM_TORCH_RUNTIME_ABI_VERSION_2,
                                    static_cast<std::uint32_t>(storage_v2.size() - 1U),
                                    storage_v2.data()) == XVRAM_TORCH_RUNTIME_INCOMPATIBLE_ABI);
  auto* api_v2 = reinterpret_cast<xvram_torch_runtime_api_v2*>(storage_v2.data());
  CHECK(xvram_torch_runtime_get_api(XVRAM_TORCH_RUNTIME_ABI_VERSION_2,
                                    static_cast<std::uint32_t>(storage_v2.size()),
                                    api_v2) == XVRAM_TORCH_RUNTIME_SUCCESS);
  CHECK(api_v2->v1.struct_size == sizeof(xvram_torch_runtime_api_v2));
  CHECK(api_v2->v1.abi_version == XVRAM_TORCH_RUNTIME_ABI_VERSION_2);
  CHECK(api_v2->v1.session_create == api->session_create);
  CHECK(api_v2->session_create_v2 != nullptr);
  CHECK(api_v2->session_get_telemetry_v2 != nullptr);

  xvram_torch_runtime_session_config_v2 config_v2 = XVRAM_TORCH_RUNTIME_SESSION_CONFIG_V2_INIT;
  xvram_torch_runtime_session session_v2 = XVRAM_TORCH_RUNTIME_INVALID_HANDLE;
  xvram_torch_runtime_session_config_v2 short_config_v2 = config_v2;
  short_config_v2.struct_size =
      static_cast<std::uint32_t>(offsetof(xvram_torch_runtime_session_config_v2, reserved));
  CHECK(api_v2->session_create_v2(&short_config_v2, &session_v2) ==
        XVRAM_TORCH_RUNTIME_INCOMPATIBLE_ABI);
  CHECK(session_v2 == XVRAM_TORCH_RUNTIME_INVALID_HANDLE);
  const xvram_torch_runtime_status create_v2_status =
      api_v2->session_create_v2(&config_v2, &session_v2);
  CHECK(create_v2_status != XVRAM_TORCH_RUNTIME_UNSUPPORTED);
  if (create_v2_status == XVRAM_TORCH_RUNTIME_SUCCESS) {
    CHECK(session_v2 != XVRAM_TORCH_RUNTIME_INVALID_HANDLE);
    xvram_torch_runtime_telemetry_v2 live_telemetry = XVRAM_TORCH_RUNTIME_TELEMETRY_V2_INIT;
    CHECK(api_v2->session_get_telemetry_v2(session_v2, &live_telemetry) ==
          XVRAM_TORCH_RUNTIME_SUCCESS);
    CHECK(api_v2->v1.session_close(session_v2, UINT64_C(5000)) == XVRAM_TORCH_RUNTIME_SUCCESS);
  } else {
    CHECK(create_v2_status == XVRAM_TORCH_RUNTIME_INVALID_ARGUMENT ||
          create_v2_status == XVRAM_TORCH_RUNTIME_INVALID_STATE ||
          create_v2_status == XVRAM_TORCH_RUNTIME_UNAVAILABLE ||
          create_v2_status == XVRAM_TORCH_RUNTIME_HOST_OUT_OF_MEMORY ||
          create_v2_status == XVRAM_TORCH_RUNTIME_DEVICE_OUT_OF_MEMORY ||
          create_v2_status == XVRAM_TORCH_RUNTIME_BUDGET_PRESSURE ||
          create_v2_status == XVRAM_TORCH_RUNTIME_CUDA_ERROR);
    CHECK(session_v2 == XVRAM_TORCH_RUNTIME_INVALID_HANDLE);
  }
  xvram_torch_runtime_telemetry_v2 telemetry_v2 = XVRAM_TORCH_RUNTIME_TELEMETRY_V2_INIT;
  CHECK(api_v2->session_get_telemetry_v2(UINT64_C(999), &telemetry_v2) ==
        XVRAM_TORCH_RUNTIME_NOT_FOUND);
  telemetry_v2.struct_size =
      static_cast<std::uint32_t>(sizeof(xvram_torch_runtime_telemetry_v2) - sizeof(std::uint64_t));
  CHECK(api_v2->session_get_telemetry_v2(UINT64_C(999), &telemetry_v2) ==
        XVRAM_TORCH_RUNTIME_INCOMPATIBLE_ABI);

  xvram_torch_runtime_gemm_problem_v1 problem = XVRAM_TORCH_RUNTIME_GEMM_PROBLEM_V1_INIT;
  xvram_torch_runtime_gemm_result_v1 result = XVRAM_TORCH_RUNTIME_GEMM_RESULT_V1_INIT;
  problem.m = 1U;
  problem.n = 1U;
  problem.k = 1U;
  problem.beta = 1.0;
  CHECK(api->gemm_execute(UINT64_C(999), &problem, &result) ==
        XVRAM_TORCH_RUNTIME_INVALID_ARGUMENT);
  CHECK(result.status == XVRAM_TORCH_RUNTIME_GEMM_INVALID_PROBLEM);
  CHECK(result.boundary == XVRAM_TORCH_RUNTIME_GEMM_BOUNDARY_PREFLIGHT);

  problem.beta = 0.0;
  result = XVRAM_TORCH_RUNTIME_GEMM_RESULT_V1_INIT;
  CHECK(api->gemm_execute(UINT64_C(999), &problem, &result) == XVRAM_TORCH_RUNTIME_NOT_FOUND);
  CHECK(result.status == XVRAM_TORCH_RUNTIME_GEMM_INVALID_PROBLEM);

  result = XVRAM_TORCH_RUNTIME_GEMM_RESULT_V1_INIT;
  problem.abi_version = UINT32_C(999);
  CHECK(api->gemm_execute(UINT64_C(999), &problem, &result) ==
        XVRAM_TORCH_RUNTIME_INCOMPATIBLE_ABI);
  CHECK(result.status == XVRAM_TORCH_RUNTIME_GEMM_INVALID_PROBLEM);

  // Scratch allocation is deliberately impossible without a thread-local,
  // armed lease and therefore cannot silently fall back to cudaMalloc.
  CHECK(xvram_torch_runtime_scratch_alloc(256U, 0, nullptr) == nullptr);
  xvram_torch_runtime_scratch_free(nullptr, 0U, 0, nullptr);

  return failures == 0 ? 0 : 1;
}
