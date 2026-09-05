#include "xvram/cuda_compat.hpp"

#include <atomic>
#include <cstring>
#include <iostream>
#include <string_view>
#include <thread>

namespace {
enum class Mode { rejected, missing_members, wrong_version };
Mode mode = Mode::rejected;
std::atomic<unsigned int> negotiations{0};
int failures = 0;
#define CHECK(value)                                                                               \
  do {                                                                                             \
    if (!(value)) {                                                                                \
      std::cerr << "line " << __LINE__ << ": " #value "\n";                                        \
      ++failures;                                                                                  \
    }                                                                                              \
  } while (false)
} // namespace

// Deliberately supplies no adapter or CUDA implementation. A facade must handle failed
// negotiation and incomplete tables without calling a null pointer or a native library.
extern "C" xvram_status XVRAM_CALL xvram_cuda_compat_get_api(uint32_t version, uint32_t bytes,
                                                             void* output) {
  ++negotiations;
  if (mode == Mode::rejected || version != 1 || bytes < sizeof(xvram_cuda_compat_api_v1))
    return XVRAM_STATUS_INCOMPATIBLE_ABI;
  xvram_cuda_compat_api_v1 value{};
  value.struct_size = sizeof(value);
  value.abi_version = mode == Mode::wrong_version ? 2U : 1U;
  std::memcpy(output, &value, sizeof(value));
  return XVRAM_STATUS_SUCCESS;
}

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--missing-members")
    mode = Mode::missing_members;
  else if (argc == 2 && std::string_view(argv[1]) == "--wrong-version")
    mode = Mode::wrong_version;
  else if (argc != 1)
    return 64;
  namespace cc = xvram::cuda_compat;
  float* pointer = reinterpret_cast<float*>(uintptr_t{17});
  CHECK(cc::cudaMalloc(&pointer, 16) == cudaErrorNotSupported);
  CHECK(pointer == reinterpret_cast<float*>(uintptr_t{17}));
  CHECK(cc::cudaFree(nullptr) == cudaErrorNotSupported);
  CHECK(cc::cudaMemcpy(nullptr, nullptr, 0, cudaMemcpyHostToDevice) == cudaErrorNotSupported);
  int device = 17;
  CHECK(cc::cudaGetDevice(&device) == cudaErrorNotSupported);
  CHECK(device == 17);
  CHECK(cc::cudaGetDeviceCount(&device) == cudaErrorNotSupported);
  CHECK(device == 17);
  CHECK(cc::cudaSetDevice(0) == cudaErrorNotSupported);
  CHECK(cc::cudaDeviceSynchronize() == cudaErrorNotSupported);
  CHECK(cc::cudaPeekAtLastError() == cudaErrorNotSupported);
  CHECK(cc::cudaGetLastError() == cudaErrorNotSupported);
  CHECK(cc::cudaPeekAtLastError() == cudaSuccess);
  CHECK(std::strcmp(cc::cudaGetErrorName(cudaErrorNotSupported), "cudaErrorNotSupported") == 0);
  CHECK(std::strcmp(cc::cudaGetErrorString(cudaErrorNotSupported), "cudaErrorNotSupported") == 0);
  cublasHandle_t handle = reinterpret_cast<cublasHandle_t>(uintptr_t{17});
  CHECK(cc::cublasCreate(&handle) == CUBLAS_STATUS_NOT_SUPPORTED);
  CHECK(handle == reinterpret_cast<cublasHandle_t>(uintptr_t{17}));
  CHECK(cc::cublasDestroy(handle) == CUBLAS_STATUS_NOT_SUPPORTED);
  CHECK(cc::cublasGetVersion(handle, &device) == CUBLAS_STATUS_NOT_SUPPORTED);
  CHECK(device == 17);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(uintptr_t{17});
  CHECK(cc::cublasSetStream(handle, nullptr) == CUBLAS_STATUS_NOT_SUPPORTED);
  CHECK(cc::cublasGetStream(handle, &stream) == CUBLAS_STATUS_NOT_SUPPORTED);
  CHECK(stream == reinterpret_cast<cudaStream_t>(uintptr_t{17}));
  cublasPointerMode_t pointer_mode = CUBLAS_POINTER_MODE_DEVICE;
  CHECK(cc::cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST) == CUBLAS_STATUS_NOT_SUPPORTED);
  CHECK(cc::cublasGetPointerMode(handle, &pointer_mode) == CUBLAS_STATUS_NOT_SUPPORTED);
  CHECK(pointer_mode == CUBLAS_POINTER_MODE_DEVICE);
  cublasMath_t math = CUBLAS_TF32_TENSOR_OP_MATH;
  CHECK(cc::cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH) == CUBLAS_STATUS_NOT_SUPPORTED);
  CHECK(cc::cublasGetMathMode(handle, &math) == CUBLAS_STATUS_NOT_SUPPORTED);
  CHECK(math == CUBLAS_TF32_TENSOR_OP_MATH);
  const float scalar = 1;
  CHECK(cc::cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 1, &scalar, nullptr, 1, nullptr, 1,
                        &scalar, nullptr, 1) == CUBLAS_STATUS_NOT_SUPPORTED);
  bool isolated = false;
  std::thread other([&] {
    isolated = cc::cudaPeekAtLastError() == cudaSuccess &&
               cc::cudaSetDevice(0) == cudaErrorNotSupported &&
               cc::cudaGetLastError() == cudaErrorNotSupported;
  });
  other.join();
  CHECK(isolated);
  CHECK(cc::cudaPeekAtLastError() == cudaSuccess);
  CHECK(negotiations.load() == 1);
  return failures ? 1 : 0;
}
