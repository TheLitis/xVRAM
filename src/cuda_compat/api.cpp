#include "cuda_compat/backend.hpp"
#include <cstring>

namespace {
xvram::cuda_compat::Adapter& adapter() {
  static xvram::cuda_compat::Adapter value;
  return value;
}
template <class F> xvram_status boundary(F&& fn) noexcept {
  try {
    return fn();
  } catch (...) {
    return XVRAM_STATUS_HOST_OUT_OF_MEMORY;
  }
}
const xvram_cuda_compat_api_v1& table() {
  static const auto value = [] {
    xvram_cuda_compat_api_v1 a{};
    a.struct_size = sizeof(a);
    a.abi_version = XVRAM_CUDA_COMPAT_ABI_VERSION_1;
#define ENTRY(name, args, values)                                                                  \
  a.name = +[] args -> xvram_status { return boundary([&] { return adapter().name values; }); }
    ENTRY(initialize, (const xvram_cuda_compat_config_v1* c), (c));
    ENTRY(shutdown, (), ());
    ENTRY(get_error, (xvram_error_info_v1 * out), (out));
    ENTRY(get_telemetry, (xvram_cuda_compat_telemetry_v1 * out), (out));
    ENTRY(malloc_device, (void** out, uint64_t size), (out, size));
    ENTRY(free_device, (void* p), (p));
    ENTRY(memcpy, (void* d, const void* s, uint64_t n, uint32_t k), (d, s, n, k));
    ENTRY(get_device, (int32_t* out), (out));
    ENTRY(get_device_count, (int32_t* out), (out));
    ENTRY(set_device, (int32_t d), (d));
    ENTRY(synchronize, (), ());
    ENTRY(mem_get_info, (uint64_t* f, uint64_t* t), (f, t));
    ENTRY(blas_create, (void** out), (out));
    ENTRY(blas_destroy, (void* h), (h));
    ENTRY(blas_get_version, (void* h, int32_t* out), (h, out));
    ENTRY(blas_set_stream, (void* h, uint64_t s), (h, s));
    ENTRY(blas_get_stream, (void* h, uint64_t* out), (h, out));
    ENTRY(blas_set_pointer_mode, (void* h, uint32_t m), (h, m));
    ENTRY(blas_get_pointer_mode, (void* h, uint32_t* out), (h, out));
    ENTRY(blas_set_math_mode, (void* h, uint32_t m), (h, m));
    ENTRY(blas_get_math_mode, (void* h, uint32_t* out), (h, out));
    ENTRY(blas_sgemm,
          (void* h, uint32_t oa, uint32_t ob, int32_t m, int32_t n, int32_t k, const float* alpha,
           const float* ap, int32_t lda, const float* bp, int32_t ldb, const float* beta, float* cp,
           int32_t ldc),
          (h, oa, ob, m, n, k, alpha, ap, lda, bp, ldb, beta, cp, ldc));
#undef ENTRY
    return a;
  }();
  return value;
}
} // namespace

extern "C" XVRAM_CUDA_COMPAT_API xvram_status XVRAM_CALL
xvram_cuda_compat_get_api(uint32_t requested, uint32_t size, void* output) {
  if (!output)
    return XVRAM_STATUS_INVALID_ARGUMENT;
  if (requested != XVRAM_CUDA_COMPAT_ABI_VERSION_1 || size < sizeof(xvram_cuda_compat_api_v1))
    return XVRAM_STATUS_INCOMPATIBLE_ABI;
  std::memcpy(output, &table(), sizeof(xvram_cuda_compat_api_v1));
  return XVRAM_STATUS_SUCCESS;
}
