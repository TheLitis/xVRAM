#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <xvram/xvram_v2.h>

/* These assertions intentionally pin the public 64-bit C ABI layout. */
_Static_assert(sizeof(void*) == 8U, "the installed xVRAM SDK requires a 64-bit consumer");
_Static_assert(XVRAM_ABI_VERSION_CURRENT == XVRAM_ABI_VERSION_2, "xvram_v2.h must select ABI v2");
_Static_assert(sizeof(xvram_api_v1) == 336U, "xvram_api_v1 layout changed");
_Static_assert(sizeof(xvram_api_v2) == 464U, "xvram_api_v2 layout changed");
_Static_assert(offsetof(xvram_api_v2, v1) == 0U, "v2 must embed v1 at offset zero");
_Static_assert(offsetof(xvram_api_v2, session_create_v2) == 336U,
               "session_create_v2 offset changed");
_Static_assert(offsetof(xvram_api_v2, session_get_telemetry_v2) == 344U,
               "session_get_telemetry_v2 offset changed");
_Static_assert(offsetof(xvram_api_v2, reserved) == 352U, "v2 reserved offset changed");

static int v1_table_is_populated(const xvram_api_v1* api) {
  return api->status_name != NULL && api->session_create != NULL &&
         api->session_get_telemetry != NULL && api->transaction_submit != NULL &&
         api->gemm_plan_create != NULL && api->gemm_submit != NULL;
}

int main(void) {
  xvram_api_v1 api_v1;
  xvram_api_v2 api_v2;
  xvram_api_v2 rejected;
  size_t index;

  memset(&api_v1, 0, sizeof(api_v1));
  if (xvram_get_api(XVRAM_ABI_VERSION_1, (uint32_t)sizeof(api_v1), &api_v1) !=
      XVRAM_STATUS_SUCCESS) {
    return 1;
  }
  if (api_v1.struct_size != (uint32_t)sizeof(api_v1) || api_v1.abi_version != XVRAM_ABI_VERSION_1 ||
      !v1_table_is_populated(&api_v1)) {
    return 2;
  }
  if (strcmp(api_v1.status_name(XVRAM_STATUS_SUCCESS), "success") != 0) {
    return 3;
  }

  memset(&api_v2, 0, sizeof(api_v2));
  if (xvram_get_api(XVRAM_ABI_VERSION_2, (uint32_t)sizeof(api_v2), &api_v2) !=
      XVRAM_STATUS_SUCCESS) {
    return 4;
  }
  if (api_v2.v1.struct_size != (uint32_t)sizeof(api_v2) ||
      api_v2.v1.abi_version != XVRAM_ABI_VERSION_2 || !v1_table_is_populated(&api_v2.v1) ||
      api_v2.session_create_v2 == NULL || api_v2.session_get_telemetry_v2 == NULL) {
    return 5;
  }
  if (api_v2.v1.status_name != api_v1.status_name ||
      api_v2.v1.session_create != api_v1.session_create ||
      api_v2.v1.transaction_submit != api_v1.transaction_submit ||
      api_v2.v1.gemm_submit != api_v1.gemm_submit) {
    return 6;
  }
  for (index = 0U; index < sizeof(api_v2.reserved) / sizeof(api_v2.reserved[0]); ++index) {
    if (api_v2.reserved[index] != UINT64_C(0)) {
      return 7;
    }
  }

  memset(&rejected, 0xA5, sizeof(rejected));
  if (xvram_get_api(XVRAM_ABI_VERSION_2, (uint32_t)(sizeof(rejected) - 1U), &rejected) !=
      XVRAM_STATUS_INVALID_ARGUMENT) {
    return 8;
  }
  if (rejected.v1.struct_size != UINT32_C(0xA5A5A5A5)) {
    return 9;
  }
  if (xvram_get_api(UINT32_MAX, (uint32_t)sizeof(rejected), &rejected) !=
      XVRAM_STATUS_INCOMPATIBLE_ABI) {
    return 10;
  }
  if (rejected.v1.struct_size != UINT32_C(0xA5A5A5A5)) {
    return 11;
  }

  return 0;
}
