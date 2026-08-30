#include <stdint.h>
#include <string.h>

#include <xvram/xvram.h>

int main(void) {
  xvram_api_v1 api;
  memset(&api, 0, sizeof(api));
  api.struct_size = (uint32_t)sizeof(api);

  if (xvram_get_api(XVRAM_ABI_VERSION_1, (uint32_t)sizeof(api), &api) != XVRAM_STATUS_SUCCESS) {
    return 1;
  }
  if (api.abi_version != XVRAM_ABI_VERSION_1 || api.struct_size != sizeof(api)) {
    return 2;
  }
  if (api.status_name == NULL || api.session_create == NULL || api.transaction_submit == NULL ||
      api.gemm_plan_create == NULL || api.gemm_submit == NULL) {
    return 3;
  }
  if (strcmp(api.status_name(XVRAM_STATUS_SUCCESS), "success") != 0) {
    return 4;
  }
  return 0;
}
