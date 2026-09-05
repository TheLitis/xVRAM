#include <xvram/cuda_compat.h>

int main(void) {
  xvram_cuda_compat_api_v1 api = {0};
  xvram_cuda_compat_config_v1 config = XVRAM_CUDA_COMPAT_CONFIG_V1_INIT;
  if (config.session.context_mode != XVRAM_CONTEXT_ISOLATED ||
      config.session.chunk_size_bytes != UINT64_C(67108864)) {
    return 1;
  }
  if (xvram_cuda_compat_get_api(XVRAM_CUDA_COMPAT_ABI_VERSION_1,
                              (uint32_t)sizeof(api), &api) != XVRAM_STATUS_SUCCESS) {
    return 2;
  }
  if (api.abi_version != XVRAM_CUDA_COMPAT_ABI_VERSION_1 || api.initialize == NULL ||
      api.blas_sgemm == NULL || api.shutdown == NULL) {
    return 3;
  }
  return xvram_cuda_compat_get_api(UINT32_C(99), (uint32_t)sizeof(api), &api) ==
                 XVRAM_STATUS_INCOMPATIBLE_ABI ? 0 : 4;
}
