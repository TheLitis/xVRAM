#if defined(CONTRACT_device_compilation)
#define __CUDACC__ 1
#endif
#define XVRAM_CUDA_COMPAT_REMAP_NAMES 1
#include <xvram/cuda_compat.hpp>

#if defined(CONTRACT_unsupported_driver)
#include <cuda.h>
#endif

int main() {
#if defined(CONTRACT_unsupported_runtime)
  return static_cast<int>(cudaMemset(nullptr, 0, 16));
#elif defined(CONTRACT_unsupported_driver)
  return static_cast<int>(cuInit(0));
#elif defined(CONTRACT_unsupported_cublas)
  const double one = 1.0;
  return static_cast<int>(cublasDgemm(nullptr, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 1,
                                    &one, nullptr, 1, nullptr, 1, &one, nullptr, 1));
#elif defined(CONTRACT_unsupported_launch)
  return static_cast<int>(cudaLaunchKernel(nullptr, dim3(1), dim3(1), nullptr, 0, nullptr));
#else
  xvram_cuda_compat_api_v1 api{};
  if (xvram_cuda_compat_get_api(XVRAM_CUDA_COMPAT_ABI_VERSION_1, sizeof(api), &api) !=
      XVRAM_STATUS_SUCCESS) {
    return 1;
  }
  // A successful link must not accidentally initialize or forward into native CUDA.
  void* pointer = nullptr;
  if (cudaMalloc(&pointer, 16) == cudaSuccess || pointer != nullptr) {
    return 2;
  }
  cublasHandle_t handle = nullptr;
  if (cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS || handle != nullptr) {
    return 3;
  }
  return 0;
#endif
}
