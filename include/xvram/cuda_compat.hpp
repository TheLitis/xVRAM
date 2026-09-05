#pragma once

#if defined(__CUDACC__) || defined(__CUDA_ARCH__)
#error "XVRAM_CUDA_COMPAT_HOST_ONLY: compile the opt-in facade with a host C++ compiler"
#endif

#include "cuda_compat.h"
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

// NVIDIA maps these source names to ABI-versioned symbols. The facade owns only the
// names explicitly listed here. Every other CUDA/cuBLAS symbol keeps its native declaration.
#undef cublasCreate
#undef cublasDestroy
#undef cublasGetVersion
#undef cublasSetStream
#undef cublasGetStream
#undef cublasSetPointerMode
#undef cublasGetPointerMode
#undef cublasSgemm

namespace xvram::cuda_compat {
namespace detail {
template <class T> struct Unavailable;
template <class... Args> struct Unavailable<xvram_status(XVRAM_CALL*)(Args...)> {
  static xvram_status XVRAM_CALL call(Args...) noexcept {
    return XVRAM_STATUS_INCOMPATIBLE_ABI;
  }
};
inline const xvram_cuda_compat_api_v1& api() noexcept {
  static const auto value = [] {
    xvram_cuda_compat_api_v1 a{};
    const auto status = xvram_cuda_compat_get_api(XVRAM_CUDA_COMPAT_ABI_VERSION_1, sizeof(a), &a);
    const bool valid = status == XVRAM_STATUS_SUCCESS && a.struct_size >= sizeof(a) &&
                       a.abi_version == XVRAM_CUDA_COMPAT_ABI_VERSION_1;
#define XVRAM_COMPAT_REQUIRE_MEMBER(name)                                                          \
  if (!valid || !a.name)                                                                           \
  a.name = &Unavailable<decltype(a.name)>::call
    XVRAM_COMPAT_REQUIRE_MEMBER(initialize);
    XVRAM_COMPAT_REQUIRE_MEMBER(shutdown);
    XVRAM_COMPAT_REQUIRE_MEMBER(get_error);
    XVRAM_COMPAT_REQUIRE_MEMBER(get_telemetry);
    XVRAM_COMPAT_REQUIRE_MEMBER(malloc_device);
    XVRAM_COMPAT_REQUIRE_MEMBER(free_device);
    XVRAM_COMPAT_REQUIRE_MEMBER(memcpy);
    XVRAM_COMPAT_REQUIRE_MEMBER(get_device);
    XVRAM_COMPAT_REQUIRE_MEMBER(get_device_count);
    XVRAM_COMPAT_REQUIRE_MEMBER(set_device);
    XVRAM_COMPAT_REQUIRE_MEMBER(synchronize);
    XVRAM_COMPAT_REQUIRE_MEMBER(mem_get_info);
    XVRAM_COMPAT_REQUIRE_MEMBER(blas_create);
    XVRAM_COMPAT_REQUIRE_MEMBER(blas_destroy);
    XVRAM_COMPAT_REQUIRE_MEMBER(blas_get_version);
    XVRAM_COMPAT_REQUIRE_MEMBER(blas_set_stream);
    XVRAM_COMPAT_REQUIRE_MEMBER(blas_get_stream);
    XVRAM_COMPAT_REQUIRE_MEMBER(blas_set_pointer_mode);
    XVRAM_COMPAT_REQUIRE_MEMBER(blas_get_pointer_mode);
    XVRAM_COMPAT_REQUIRE_MEMBER(blas_set_math_mode);
    XVRAM_COMPAT_REQUIRE_MEMBER(blas_get_math_mode);
    XVRAM_COMPAT_REQUIRE_MEMBER(blas_sgemm);
#undef XVRAM_COMPAT_REQUIRE_MEMBER
    return a;
  }();
  return value;
}
inline thread_local cudaError_t last_cuda_error = cudaSuccess;
inline cudaError_t cuda_status(xvram_status status) noexcept {
  cudaError_t result = cudaErrorUnknown;
  switch (status) {
  case XVRAM_STATUS_SUCCESS:
    result = cudaSuccess;
    break;
  case XVRAM_STATUS_INVALID_ARGUMENT:
    result = cudaErrorInvalidValue;
    break;
  case XVRAM_STATUS_INCOMPATIBLE_ABI:
  case XVRAM_STATUS_UNSUPPORTED:
    result = cudaErrorNotSupported;
    break;
  case XVRAM_STATUS_UNAVAILABLE:
    result = cudaErrorNoDevice;
    break;
  case XVRAM_STATUS_NOT_READY:
  case XVRAM_STATUS_CLOSED:
    result = cudaErrorInitializationError;
    break;
  case XVRAM_STATUS_HOST_OUT_OF_MEMORY:
  case XVRAM_STATUS_DEVICE_OUT_OF_MEMORY:
  case XVRAM_STATUS_BUDGET_PRESSURE:
    result = cudaErrorMemoryAllocation;
    break;
  case XVRAM_STATUS_TIMEOUT:
    result = cudaErrorLaunchTimeout;
    break;
  case XVRAM_STATUS_POISONED:
  case XVRAM_STATUS_CUDA_ERROR:
  case XVRAM_STATUS_CORRUPTION:
  case XVRAM_STATUS_CLEANUP_FAILED:
    result = cudaErrorLaunchFailure;
    break;
  default:
    break;
  }
  if (result != cudaSuccess)
    last_cuda_error = result;
  return result;
}
inline cublasStatus_t blas_status(xvram_status status) noexcept {
  switch (status) {
  case XVRAM_STATUS_SUCCESS:
    return CUBLAS_STATUS_SUCCESS;
  case XVRAM_STATUS_INVALID_ARGUMENT:
    return CUBLAS_STATUS_INVALID_VALUE;
  case XVRAM_STATUS_INCOMPATIBLE_ABI:
  case XVRAM_STATUS_UNSUPPORTED:
    return CUBLAS_STATUS_NOT_SUPPORTED;
  case XVRAM_STATUS_UNAVAILABLE:
  case XVRAM_STATUS_NOT_READY:
  case XVRAM_STATUS_CLOSED:
    return CUBLAS_STATUS_NOT_INITIALIZED;
  case XVRAM_STATUS_HOST_OUT_OF_MEMORY:
  case XVRAM_STATUS_DEVICE_OUT_OF_MEMORY:
  case XVRAM_STATUS_BUDGET_PRESSURE:
    return CUBLAS_STATUS_ALLOC_FAILED;
  case XVRAM_STATUS_TIMEOUT:
  case XVRAM_STATUS_POISONED:
  case XVRAM_STATUS_CUDA_ERROR:
  case XVRAM_STATUS_CUBLAS_ERROR:
  case XVRAM_STATUS_CORRUPTION:
    return CUBLAS_STATUS_EXECUTION_FAILED;
  default:
    return CUBLAS_STATUS_INTERNAL_ERROR;
  }
}
} // namespace detail
inline cudaError_t cudaMalloc(void** p, size_t n) noexcept {
  return detail::cuda_status(detail::api().malloc_device(p, n));
}
template <class T> inline cudaError_t cudaMalloc(T** p, size_t n) noexcept {
  return detail::cuda_status(detail::api().malloc_device(reinterpret_cast<void**>(p), n));
}
inline cudaError_t cudaFree(void* p) noexcept {
  return detail::cuda_status(detail::api().free_device(p));
}
inline cudaError_t cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind k) noexcept {
  return detail::cuda_status(detail::api().memcpy(d, s, n, static_cast<uint32_t>(k)));
}
inline cudaError_t cudaGetDevice(int* output) noexcept {
  return detail::cuda_status(detail::api().get_device(output));
}
inline cudaError_t cudaGetDeviceCount(int* output) noexcept {
  return detail::cuda_status(detail::api().get_device_count(output));
}
inline cudaError_t cudaSetDevice(int d) noexcept {
  return detail::cuda_status(detail::api().set_device(d));
}
inline cudaError_t cudaDeviceSynchronize() noexcept {
  return detail::cuda_status(detail::api().synchronize());
}
inline cudaError_t cudaPeekAtLastError() noexcept {
  return detail::last_cuda_error;
}
inline cudaError_t cudaGetLastError() noexcept {
  const auto value = detail::last_cuda_error;
  detail::last_cuda_error = cudaSuccess;
  return value;
}
inline const char* cudaGetErrorName(cudaError_t error) noexcept {
  switch (error) {
  case cudaSuccess:
    return "cudaSuccess";
  case cudaErrorInvalidValue:
    return "cudaErrorInvalidValue";
  case cudaErrorMemoryAllocation:
    return "cudaErrorMemoryAllocation";
  case cudaErrorInitializationError:
    return "cudaErrorInitializationError";
  case cudaErrorNoDevice:
    return "cudaErrorNoDevice";
  case cudaErrorNotSupported:
    return "cudaErrorNotSupported";
  case cudaErrorLaunchTimeout:
    return "cudaErrorLaunchTimeout";
  case cudaErrorLaunchFailure:
    return "cudaErrorLaunchFailure";
  default:
    return "cudaErrorUnknown";
  }
}
inline const char* cudaGetErrorString(cudaError_t error) noexcept {
  return xvram::cuda_compat::cudaGetErrorName(error);
}
inline cublasStatus_t cublasCreate(cublasHandle_t* output) noexcept {
  return detail::blas_status(detail::api().blas_create(reinterpret_cast<void**>(output)));
}
inline cublasStatus_t cublasDestroy(cublasHandle_t h) noexcept {
  return detail::blas_status(detail::api().blas_destroy(h));
}
inline cublasStatus_t cublasGetVersion(cublasHandle_t h, int* out) noexcept {
  return detail::blas_status(detail::api().blas_get_version(h, out));
}
inline cublasStatus_t cublasSetStream(cublasHandle_t h, cudaStream_t s) noexcept {
  return detail::blas_status(detail::api().blas_set_stream(h, reinterpret_cast<uintptr_t>(s)));
}
inline cublasStatus_t cublasGetStream(cublasHandle_t h, cudaStream_t* out) noexcept {
  return detail::blas_status(detail::api().blas_get_stream(h, reinterpret_cast<uint64_t*>(out)));
}
inline cublasStatus_t cublasSetPointerMode(cublasHandle_t h, cublasPointerMode_t m) noexcept {
  return detail::blas_status(detail::api().blas_set_pointer_mode(h, static_cast<uint32_t>(m)));
}
inline cublasStatus_t cublasGetPointerMode(cublasHandle_t h, cublasPointerMode_t* out) noexcept {
  static_assert(sizeof(*out) == sizeof(uint32_t));
  return detail::blas_status(
      detail::api().blas_get_pointer_mode(h, reinterpret_cast<uint32_t*>(out)));
}
inline cublasStatus_t cublasSetMathMode(cublasHandle_t h, cublasMath_t m) noexcept {
  return detail::blas_status(detail::api().blas_set_math_mode(h, static_cast<uint32_t>(m)));
}
inline cublasStatus_t cublasGetMathMode(cublasHandle_t h, cublasMath_t* out) noexcept {
  static_assert(sizeof(*out) == sizeof(uint32_t));
  return detail::blas_status(detail::api().blas_get_math_mode(h, reinterpret_cast<uint32_t*>(out)));
}
inline cublasStatus_t cublasSgemm(cublasHandle_t h, cublasOperation_t oa, cublasOperation_t ob,
                                  int m, int n, int k, const float* alpha, const float* a, int lda,
                                  const float* b, int ldb, const float* beta, float* c,
                                  int ldc) noexcept {
  return detail::blas_status(detail::api().blas_sgemm(h, static_cast<uint32_t>(oa),
                                                      static_cast<uint32_t>(ob), m, n, k, alpha, a,
                                                      lda, b, ldb, beta, c, ldc));
}
} // namespace xvram::cuda_compat

/* Define before including this header to redirect the approved source-call spellings.
 * Without this opt-in, use xvram::cuda_compat::cudaMalloc etc. No native library is linked. */
#if defined(XVRAM_CUDA_COMPAT_REMAP_NAMES) && XVRAM_CUDA_COMPAT_REMAP_NAMES
#define cudaMalloc ::xvram::cuda_compat::cudaMalloc
#define cudaFree ::xvram::cuda_compat::cudaFree
#define cudaMemcpy ::xvram::cuda_compat::cudaMemcpy
#define cudaGetDevice ::xvram::cuda_compat::cudaGetDevice
#define cudaGetDeviceCount ::xvram::cuda_compat::cudaGetDeviceCount
#define cudaSetDevice ::xvram::cuda_compat::cudaSetDevice
#define cudaDeviceSynchronize ::xvram::cuda_compat::cudaDeviceSynchronize
#define cudaPeekAtLastError ::xvram::cuda_compat::cudaPeekAtLastError
#define cudaGetLastError ::xvram::cuda_compat::cudaGetLastError
#define cudaGetErrorName ::xvram::cuda_compat::cudaGetErrorName
#define cudaGetErrorString ::xvram::cuda_compat::cudaGetErrorString
#define cublasCreate ::xvram::cuda_compat::cublasCreate
#define cublasDestroy ::xvram::cuda_compat::cublasDestroy
#define cublasGetVersion ::xvram::cuda_compat::cublasGetVersion
#define cublasSetStream ::xvram::cuda_compat::cublasSetStream
#define cublasGetStream ::xvram::cuda_compat::cublasGetStream
#define cublasSetPointerMode ::xvram::cuda_compat::cublasSetPointerMode
#define cublasGetPointerMode ::xvram::cuda_compat::cublasGetPointerMode
#define cublasSetMathMode ::xvram::cuda_compat::cublasSetMathMode
#define cublasGetMathMode ::xvram::cuda_compat::cublasGetMathMode
#define cublasSgemm ::xvram::cuda_compat::cublasSgemm
#endif
