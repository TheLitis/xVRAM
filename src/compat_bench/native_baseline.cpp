#include "compat_bench/native_baseline.hpp"
#include "platform/cublas/cublas_api.hpp"
#include "platform/cuda/cuda_api.hpp"
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>

namespace xvram::compat_bench {
NativeBaseline native_baseline(const Options& options, const Shape& shape) {
  NativeBaseline result;
  cuda::CudaApi cuda;
  cublas::CublasApi blas;
  if (cuda.load().status != cuda::CudaApi::LoadStatus::loaded || cuda.init_(0) != CUDA_SUCCESS) {
    result.error = "baseline CUDA unavailable";
    return result;
  }
  cublas::CublasLoadOptions load;
  load.core_library = utf8_path(options.cublas_library);
  load.lt_library = utf8_path(options.cublas_lt_library);
  if (blas.load(load).status != cublas::CublasLoadStatus::loaded) {
    result.error = "baseline cuBLAS unavailable";
    return result;
  }
  const auto& dispatch = blas.dispatch();
  CUdevice device = 0;
  CUcontext context = nullptr;
  CUstream stream = nullptr;
  cublas::abi::Handle handle = nullptr;
  std::array<CUdeviceptr, 3> allocations{};
  std::array<void*, 3> staging{};
  const std::array<Operand, 3> operands{Operand::a, Operand::b, Operand::c};
  std::array<Matrix, 3> matrices{};
  const auto check = [&](const bool okay, const char* operation) {
    if (!okay && result.error.empty())
      result.error = std::string("native baseline failed: ") + operation;
    return okay;
  };
  bool ready =
      check(cuda.device_get_(&device, options.device) == CUDA_SUCCESS, "device") &&
      check(cuda.context_create_(&context, 0, device) == CUDA_SUCCESS, "context") &&
      check(cuda.stream_create_(&stream, CU_STREAM_NON_BLOCKING) == CUDA_SUCCESS, "stream") &&
      check(dispatch.create(&handle) == cublas::abi::success, "cuBLAS handle");
  for (std::size_t i = 0; ready && i < 3; ++i) {
    const auto matrix = matrix_storage(shape, operands[i]);
    if (!matrix || matrix->elements > (16ULL << 20U) / sizeof(float)) {
      result.error = "baseline is limited to small resident cases";
      ready = false;
      break;
    }
    matrices[i] = *matrix;
    const auto bytes = static_cast<std::size_t>(matrix->elements * sizeof(float));
    ready = check(cuda.mem_alloc_(&allocations[i], bytes) == CUDA_SUCCESS, "allocation") &&
            check(cuda.mem_host_alloc_(&staging[i], bytes, 0) == CUDA_SUCCESS, "pinned staging");
    if (!ready)
      break;
    fill_pattern(std::span<float>(static_cast<float*>(staging[i]),
                                  static_cast<std::size_t>(matrix->elements)),
                 0, shape, operands[i], options.seed);
    ready = check(cuda.memcpy_h2d_async_(allocations[i], staging[i], bytes, stream) == CUDA_SUCCESS,
                  "H2D");
  }
  if (ready)
    ready = check(cuda.stream_synchronize_(stream) == CUDA_SUCCESS, "initial copy completion");
  const auto start = std::chrono::steady_clock::now();
  for (std::uint32_t pass = 0; ready && pass < options.passes; ++pass) {
    cublas::CoreGemmRequest request;
    request.operation_a =
        shape.transpose_a ? cublas::abi::operation_transpose : cublas::abi::operation_none;
    request.operation_b =
        shape.transpose_b ? cublas::abi::operation_transpose : cublas::abi::operation_none;
    request.m = static_cast<int>(shape.m);
    request.n = static_cast<int>(shape.n);
    request.k = static_cast<int>(shape.k);
    request.a = reinterpret_cast<const void*>(allocations[0] + shape.offset * sizeof(float));
    request.b = reinterpret_cast<const void*>(allocations[1] + shape.offset * sizeof(float));
    request.c = reinterpret_cast<void*>(allocations[2] + shape.offset * sizeof(float));
    request.lda = static_cast<int>(matrices[0].ld);
    request.ldb = static_cast<int>(matrices[1].ld);
    request.ldc = static_cast<int>(matrices[2].ld);
    request.compute_type = cublas::abi::compute_fp32_pedantic;
    request.math_mode = cublas::abi::pedantic_math;
    request.alpha = options.alpha;
    request.beta = options.beta;
    request.stream = stream;
    ready =
        check(static_cast<bool>(cublas::execute_core_gemm(dispatch, handle, request)), "SGEMM") &&
        check(cuda.stream_synchronize_(stream) == CUDA_SUCCESS, "SGEMM completion");
  }
  result.milliseconds =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  if (ready) {
    const auto bytes = static_cast<std::size_t>(matrices[2].elements * sizeof(float));
    ready = check(cuda.memcpy_d2h_async_(staging[2], allocations[2], bytes, stream) == CUDA_SUCCESS,
                  "D2H") &&
            check(cuda.stream_synchronize_(stream) == CUDA_SUCCESS, "D2H completion");
    if (ready) {
      result.c.resize(static_cast<std::size_t>(matrices[2].elements));
      std::memcpy(result.c.data(), staging[2], bytes);
    }
  }
  bool cleanup = true;
  if (stream)
    cleanup = (cuda.stream_synchronize_(stream) == CUDA_SUCCESS) && cleanup;
  if (handle)
    cleanup = (dispatch.destroy(handle) == cublas::abi::success) && cleanup;
  for (auto pointer : allocations)
    if (pointer)
      cleanup = (cuda.mem_free_(pointer) == CUDA_SUCCESS) && cleanup;
  for (auto pointer : staging)
    if (pointer)
      cleanup = (cuda.mem_free_host_(pointer) == CUDA_SUCCESS) && cleanup;
  if (stream)
    cleanup = (cuda.stream_destroy_(stream) == CUDA_SUCCESS) && cleanup;
  if (context)
    cleanup = (cuda.context_destroy_(context) == CUDA_SUCCESS) && cleanup;
  result.cleanup_complete = cleanup;
  result.completed = ready && cleanup;
  if (!cleanup && result.error.empty())
    result.error = "native baseline cleanup failed";
  return result;
}
} // namespace xvram::compat_bench
