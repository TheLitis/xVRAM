#ifndef XVRAM_TORCH_ALLOCATOR_H
#define XVRAM_TORCH_ALLOCATOR_H

#include <stddef.h>
#include <stdint.h>

#if UINTPTR_MAX != UINT64_MAX
#error "The xVRAM PyTorch allocator supports 64-bit targets only"
#endif

#if defined(_WIN32)
#define XVRAM_TORCH_CALL __cdecl
#if defined(XVRAM_TORCH_BUILDING_LIBRARY)
#define XVRAM_TORCH_API __declspec(dllexport)
#elif defined(XVRAM_TORCH_USING_LIBRARY)
#define XVRAM_TORCH_API __declspec(dllimport)
#else
#define XVRAM_TORCH_API
#endif
#elif defined(__GNUC__) && defined(XVRAM_TORCH_BUILDING_LIBRARY)
#define XVRAM_TORCH_CALL
#define XVRAM_TORCH_API __attribute__((visibility("default")))
#else
#define XVRAM_TORCH_CALL
#define XVRAM_TORCH_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define XVRAM_TORCH_ALLOCATOR_ABI_VERSION_1 UINT32_C(1)
#define XVRAM_TORCH_ALLOCATOR_ABI_VERSION_CURRENT XVRAM_TORCH_ALLOCATOR_ABI_VERSION_1

typedef uint32_t xvram_torch_allocator_status;
enum {
  XVRAM_TORCH_ALLOCATOR_SUCCESS = 0,
  XVRAM_TORCH_ALLOCATOR_INVALID_ARGUMENT = 1,
  XVRAM_TORCH_ALLOCATOR_INCOMPATIBLE_ABI = 2,
  XVRAM_TORCH_ALLOCATOR_UNAVAILABLE = 3,
  XVRAM_TORCH_ALLOCATOR_OUT_OF_MEMORY = 4,
  XVRAM_TORCH_ALLOCATOR_CUDA_ERROR = 5,
  XVRAM_TORCH_ALLOCATOR_CONTEXT_MISMATCH = 6,
  XVRAM_TORCH_ALLOCATOR_CAPTURE_UNSUPPORTED = 7,
  XVRAM_TORCH_ALLOCATOR_UNKNOWN_POINTER = 8,
  XVRAM_TORCH_ALLOCATOR_QUARANTINED = 9,
  XVRAM_TORCH_ALLOCATOR_INTERNAL_ERROR = 10
};

/*
 * This is a telemetry ABI, not PyTorch's allocator ABI. Callers initialize
 * struct_size and pass that same byte count to xvram_torch_get_stats().
 */
typedef struct xvram_torch_allocator_stats_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t allocation_calls;
  uint64_t free_calls;
  uint64_t allocation_failures;
  uint64_t free_failures;
  uint64_t requested_bytes_total;
  uint64_t requested_bytes_current;
  uint64_t requested_bytes_peak;
  uint64_t mapped_bytes_total;
  uint64_t mapped_bytes_current;
  uint64_t mapped_bytes_peak;
  uint64_t active_segments;
  uint64_t active_segments_peak;
  uint64_t reservations;
  uint64_t handles_created;
  uint64_t maps;
  uint64_t set_access_calls;
  uint64_t event_boundaries;
  uint64_t unmaps;
  uint64_t handle_releases;
  uint64_t reservation_frees;
  uint64_t capture_rejections;
  uint64_t context_mismatches;
  uint64_t size_mismatches;
  uint64_t stream_mismatches;
  uint64_t oom_failures;
  uint64_t quarantined_segments;
  uint64_t quarantined_mapped_bytes;
  uint64_t unsafe_unmaps;
  int64_t last_native_error;
  xvram_torch_allocator_status last_status;
  uint32_t reserved0;
  uint64_t reserved[8];
} xvram_torch_allocator_stats_v1;

#define XVRAM_TORCH_ALLOCATOR_STATS_V1_INIT                                                       \
  {(uint32_t)sizeof(xvram_torch_allocator_stats_v1),                                               \
   XVRAM_TORCH_ALLOCATOR_ABI_VERSION_1,                                                            \
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,              \
   INT64_C(0), XVRAM_TORCH_ALLOCATOR_SUCCESS, 0, {0, 0, 0, 0, 0, 0, 0, 0}}

typedef struct xvram_torch_allocator_error_v1 {
  uint32_t struct_size;
  xvram_torch_allocator_status status;
  int64_t native_code;
  char stage[32];
  char operation[64];
  char message[256];
} xvram_torch_allocator_error_v1;

#define XVRAM_TORCH_ALLOCATOR_ERROR_V1_INIT                                                       \
  {(uint32_t)sizeof(xvram_torch_allocator_error_v1), XVRAM_TORCH_ALLOCATOR_SUCCESS, INT64_C(0),     \
   {0}, {0}, {0}}

/* Exact callback signatures consumed by torch.cuda.CUDAPluggableAllocator. */
XVRAM_TORCH_API void* XVRAM_TORCH_CALL xvram_torch_alloc(size_t bytes, int device,
                                                          void* stream);
XVRAM_TORCH_API void XVRAM_TORCH_CALL xvram_torch_free(void* pointer, size_t bytes, int device,
                                                        void* stream);

XVRAM_TORCH_API xvram_torch_allocator_status XVRAM_TORCH_CALL
xvram_torch_get_stats(xvram_torch_allocator_stats_v1* output, size_t output_size);
XVRAM_TORCH_API xvram_torch_allocator_status XVRAM_TORCH_CALL xvram_torch_reset_stats(void);
XVRAM_TORCH_API xvram_torch_allocator_status XVRAM_TORCH_CALL
xvram_torch_get_last_error(xvram_torch_allocator_error_v1* output, size_t output_size);

#ifdef __cplusplus
}
#endif

#endif
