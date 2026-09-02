#include "xvram/internal/torch_runtime.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>

namespace {

int failures = 0;

void check(const bool condition, const char* expression, const int line) {
  if (!condition) {
    std::cerr << "line " << line << ": " << expression << '\n';
    ++failures;
  }
}

#define CHECK(expression) check((expression), #expression, __LINE__)

} // namespace

int main() {
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
