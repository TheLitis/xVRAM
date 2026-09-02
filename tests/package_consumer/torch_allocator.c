#include "xvram/torch_allocator.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

static int require(int condition, const char* message) {
  if (!condition) {
    fprintf(stderr, "%s\n", message);
    return 0;
  }
  return 1;
}

int main(void) {
  xvram_torch_allocator_stats_v1 stats = XVRAM_TORCH_ALLOCATOR_STATS_V1_INIT;
  xvram_torch_allocator_error_v1 error = XVRAM_TORCH_ALLOCATOR_ERROR_V1_INIT;

  if (!require(XVRAM_TORCH_ALLOCATOR_ABI_VERSION_CURRENT == UINT32_C(1),
               "unexpected allocator ABI version") ||
      !require(stats.struct_size == sizeof(stats), "stats initializer has the wrong size") ||
      !require(stats.abi_version == XVRAM_TORCH_ALLOCATOR_ABI_VERSION_CURRENT,
               "stats initializer has the wrong ABI version")) {
    return 1;
  }

  if (!require(xvram_torch_reset_stats() == XVRAM_TORCH_ALLOCATOR_SUCCESS,
               "xvram_torch_reset_stats failed") ||
      !require(xvram_torch_get_stats(&stats, sizeof(stats)) == XVRAM_TORCH_ALLOCATOR_SUCCESS,
               "xvram_torch_get_stats failed") ||
      !require(stats.struct_size == sizeof(stats), "stats response has the wrong size") ||
      !require(stats.abi_version == XVRAM_TORCH_ALLOCATOR_ABI_VERSION_CURRENT,
               "stats response has the wrong ABI version") ||
      !require(stats.allocation_calls == UINT64_C(0),
               "stats reset left allocation calls behind") ||
      !require(stats.free_calls == UINT64_C(0), "stats reset left free calls behind") ||
      !require(stats.last_status == XVRAM_TORCH_ALLOCATOR_SUCCESS,
               "stats reset left a failure status behind") ||
      !require(xvram_torch_get_last_error(&error, sizeof(error)) ==
                   XVRAM_TORCH_ALLOCATOR_SUCCESS,
               "xvram_torch_get_last_error failed") ||
      !require(error.struct_size == sizeof(error), "error response has the wrong size") ||
      !require(error.status == XVRAM_TORCH_ALLOCATOR_SUCCESS,
               "error response reported a failure after reset")) {
    return 1;
  }

  /* A v1 producer must support a caller that only exposes the versioned prefix. */
  stats = (xvram_torch_allocator_stats_v1)XVRAM_TORCH_ALLOCATOR_STATS_V1_INIT;
  stats.struct_size = (uint32_t)(sizeof(uint32_t) * 2U);
  if (!require(xvram_torch_get_stats(&stats, stats.struct_size) ==
                   XVRAM_TORCH_ALLOCATOR_SUCCESS,
               "prefix-only stats query failed") ||
      !require(stats.struct_size == sizeof(xvram_torch_allocator_stats_v1),
               "prefix-only query did not report the producer size") ||
      !require(stats.abi_version == XVRAM_TORCH_ALLOCATOR_ABI_VERSION_CURRENT,
               "prefix-only query reported the wrong ABI version")) {
    return 1;
  }

  stats = (xvram_torch_allocator_stats_v1)XVRAM_TORCH_ALLOCATOR_STATS_V1_INIT;
  stats.struct_size = (uint32_t)sizeof(uint32_t);
  if (!require(xvram_torch_get_stats(&stats, sizeof(uint32_t)) ==
                   XVRAM_TORCH_ALLOCATOR_INCOMPATIBLE_ABI,
               "undersized stats query was not rejected")) {
    return 1;
  }

  return 0;
}
