#ifndef XVRAM_CUDA_COMPAT_H
#define XVRAM_CUDA_COMPAT_H

#include "xvram.h"

#if defined(_WIN32) && defined(XVRAM_BUILDING_CUDA_COMPAT)
#define XVRAM_CUDA_COMPAT_API __declspec(dllexport)
#elif defined(_WIN32) && defined(XVRAM_USING_CUDA_COMPAT)
#define XVRAM_CUDA_COMPAT_API __declspec(dllimport)
#elif defined(__GNUC__) && defined(XVRAM_BUILDING_CUDA_COMPAT)
#define XVRAM_CUDA_COMPAT_API __attribute__((visibility("default")))
#else
#define XVRAM_CUDA_COMPAT_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define XVRAM_CUDA_COMPAT_ABI_VERSION_1 UINT32_C(1)
enum {
  XVRAM_CUDA_COMPAT_UNINITIALIZED = 0,
  XVRAM_CUDA_COMPAT_INITIALIZING = 1,
  XVRAM_CUDA_COMPAT_READY = 2,
  XVRAM_CUDA_COMPAT_POISONED = 3,
  XVRAM_CUDA_COMPAT_CLOSED = 4
};
enum { XVRAM_CUDA_COMPAT_H2D = 1, XVRAM_CUDA_COMPAT_D2H = 2 };

/* One valid initialization attempt is permitted per process, including after shutdown.
 * Session context must be isolated. Compression is always disabled. Paths are optional,
 * NUL-terminated UTF-8 absolute paths and must be provided together. Reserved fields are zero. */
typedef struct xvram_cuda_compat_config_v1 {
  uint32_t struct_size;
  uint32_t flags;
  xvram_session_config_v1 session;
  uint64_t stall_timeout_ms;
  char cublas_library[1024];
  char cublas_lt_library[1024];
  uint64_t reserved[4];
} xvram_cuda_compat_config_v1;

#define XVRAM_CUDA_COMPAT_CONFIG_V1_INIT                                                           \
  {(uint32_t)sizeof(xvram_cuda_compat_config_v1),                                                  \
   0U,                                                                                             \
   XVRAM_SESSION_CONFIG_V1_INIT,                                                                   \
   UINT64_C(5000),                                                                                 \
   {0},                                                                                            \
   {0},                                                                                            \
   {0, 0, 0, 0}}

/* Coherent snapshots are available while a blocking operation executes and after shutdown.
 * Native addresses and handles are intentionally absent. Operation/lifecycle counters and
 * peaks are cumulative. live_*, logical_bytes, host_backing_bytes, host_budget_bytes, and
 * retired_va_reservations/bytes describe current state; retired_va_*_freed are cumulative. */
typedef struct xvram_cuda_compat_telemetry_v1 {
  uint32_t struct_size;
  uint32_t state;
  xvram_session_telemetry_v1 runtime;
  int32_t device_ordinal;
  uint32_t device_count;
  char device_name[128];
  uint64_t total_vram_bytes;
  uint64_t host_physical_bytes;
  uint64_t host_available_bytes;
  uint64_t host_store_cap_bytes;
  uint64_t host_headroom_bytes;
  uint64_t effective_chunk_bytes;
  /* At quiescence: attempted == completed + rejected. Submitted counts requests
   * dispatched while ready (plus lifecycle work); metadata snapshots are excluded. */
  uint64_t calls_attempted;
  uint64_t calls_submitted;
  uint64_t calls_completed;
  uint64_t calls_rejected;
  uint64_t allocations_created;
  uint64_t allocations_released;
  uint64_t live_allocations;
  uint64_t logical_bytes;
  uint64_t logical_bytes_peak;
  uint64_t handles_created;
  uint64_t handles_destroyed;
  uint64_t live_handles;
  uint64_t h2d_bytes;
  uint64_t d2h_bytes;
  uint64_t gemm_calls;
  uint64_t tiles_submitted;
  uint64_t tiles_retired;
  uint64_t progress_sequence;
  uint64_t retired_va_reservations;
  uint64_t retired_va_reservations_freed;
  uint64_t retired_va_bytes;
  uint64_t retired_va_bytes_freed;
  uint64_t host_backing_bytes;
  uint64_t host_backing_peak_bytes;
  uint64_t host_budget_bytes;
  uint32_t cleanup_operations_drained;
  uint32_t cleanup_events_drained;
  uint32_t cleanup_completed;
  uint32_t quarantined;
  uint64_t reserved[4];
} xvram_cuda_compat_telemetry_v1;

typedef struct xvram_cuda_compat_api_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  xvram_status(XVRAM_CALL* initialize)(const xvram_cuda_compat_config_v1* config);
  xvram_status(XVRAM_CALL* shutdown)(void);
  xvram_status(XVRAM_CALL* get_error)(xvram_error_info_v1* output);
  xvram_status(XVRAM_CALL* get_telemetry)(xvram_cuda_compat_telemetry_v1* output);
  xvram_status(XVRAM_CALL* malloc_device)(void** output, uint64_t bytes);
  xvram_status(XVRAM_CALL* free_device)(void* pointer);
  xvram_status(XVRAM_CALL* memcpy)(void* destination, const void* source, uint64_t bytes,
                                   uint32_t kind);
  xvram_status(XVRAM_CALL* get_device)(int32_t* output);
  xvram_status(XVRAM_CALL* get_device_count)(int32_t* output);
  xvram_status(XVRAM_CALL* set_device)(int32_t device);
  xvram_status(XVRAM_CALL* synchronize)(void);
  /* Control-plane observation only; no cudaMemGetInfo facade is provided. */
  xvram_status(XVRAM_CALL* mem_get_info)(uint64_t* free_bytes, uint64_t* total_bytes);
  xvram_status(XVRAM_CALL* blas_create)(void** output);
  xvram_status(XVRAM_CALL* blas_destroy)(void* handle);
  xvram_status(XVRAM_CALL* blas_get_version)(void* handle, int32_t* output);
  xvram_status(XVRAM_CALL* blas_set_stream)(void* handle, uint64_t stream);
  xvram_status(XVRAM_CALL* blas_get_stream)(void* handle, uint64_t* output);
  xvram_status(XVRAM_CALL* blas_set_pointer_mode)(void* handle, uint32_t mode);
  xvram_status(XVRAM_CALL* blas_get_pointer_mode)(void* handle, uint32_t* output);
  xvram_status(XVRAM_CALL* blas_set_math_mode)(void* handle, uint32_t mode);
  xvram_status(XVRAM_CALL* blas_get_math_mode)(void* handle, uint32_t* output);
  /* Positive FP32 column-major N/T only; every matrix must belong to this adapter.
   * alpha/beta must be host pointers; no native-pointer passthrough is performed. */
  xvram_status(XVRAM_CALL* blas_sgemm)(void* handle, uint32_t op_a, uint32_t op_b, int32_t m,
                                       int32_t n, int32_t k, const float* alpha, const float* a,
                                       int32_t lda, const float* b, int32_t ldb, const float* beta,
                                       float* c, int32_t ldc);
  uint64_t reserved[8];
} xvram_cuda_compat_api_v1;

XVRAM_CUDA_COMPAT_API xvram_status XVRAM_CALL
xvram_cuda_compat_get_api(uint32_t requested_abi_version, uint32_t caller_api_size, void* output);

#ifdef __cplusplus
}
#endif
#endif
