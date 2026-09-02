#ifndef XVRAM_INTERNAL_TORCH_RUNTIME_H
#define XVRAM_INTERNAL_TORCH_RUNTIME_H

/* Private Phase 4b ABI. This header is not part of the installed SDK. */

#include <stddef.h>
#include <stdint.h>

#if UINTPTR_MAX != UINT64_MAX
#error "The xVRAM PyTorch runtime supports 64-bit targets only"
#endif

#if defined(_WIN32)
#define XVRAM_TORCH_RUNTIME_CALL __cdecl
#if defined(XVRAM_TORCH_RUNTIME_BUILDING_LIBRARY)
#define XVRAM_TORCH_RUNTIME_API __declspec(dllexport)
#elif defined(XVRAM_TORCH_RUNTIME_USING_LIBRARY)
#define XVRAM_TORCH_RUNTIME_API __declspec(dllimport)
#else
#define XVRAM_TORCH_RUNTIME_API
#endif
#elif defined(__GNUC__) && defined(XVRAM_TORCH_RUNTIME_BUILDING_LIBRARY)
#define XVRAM_TORCH_RUNTIME_CALL
#define XVRAM_TORCH_RUNTIME_API __attribute__((visibility("default")))
#else
#define XVRAM_TORCH_RUNTIME_CALL
#define XVRAM_TORCH_RUNTIME_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define XVRAM_TORCH_RUNTIME_ABI_VERSION_1 UINT32_C(1)
#define XVRAM_TORCH_RUNTIME_ABI_VERSION_CURRENT XVRAM_TORCH_RUNTIME_ABI_VERSION_1
#define XVRAM_TORCH_RUNTIME_INVALID_HANDLE UINT64_C(0)

typedef uint64_t xvram_torch_runtime_session;
typedef uint64_t xvram_torch_runtime_allocation;
typedef uint64_t xvram_torch_runtime_lease;

typedef uint32_t xvram_torch_runtime_status;
enum {
  XVRAM_TORCH_RUNTIME_SUCCESS = 0,
  XVRAM_TORCH_RUNTIME_INVALID_ARGUMENT = 1,
  XVRAM_TORCH_RUNTIME_INCOMPATIBLE_ABI = 2,
  XVRAM_TORCH_RUNTIME_NOT_FOUND = 3,
  XVRAM_TORCH_RUNTIME_INVALID_STATE = 4,
  XVRAM_TORCH_RUNTIME_VIEWS_LIVE = 5,
  XVRAM_TORCH_RUNTIME_UNSUPPORTED = 6,
  XVRAM_TORCH_RUNTIME_UNAVAILABLE = 7,
  XVRAM_TORCH_RUNTIME_HOST_OUT_OF_MEMORY = 8,
  XVRAM_TORCH_RUNTIME_DEVICE_OUT_OF_MEMORY = 9,
  XVRAM_TORCH_RUNTIME_BUDGET_PRESSURE = 10,
  XVRAM_TORCH_RUNTIME_TIMEOUT = 11,
  XVRAM_TORCH_RUNTIME_CUDA_ERROR = 12,
  XVRAM_TORCH_RUNTIME_QUARANTINED = 13,
  XVRAM_TORCH_RUNTIME_CLEANUP_FAILED = 14,
  XVRAM_TORCH_RUNTIME_INTERNAL_ERROR = 15
};

typedef uint32_t xvram_torch_runtime_policy;
enum { XVRAM_TORCH_RUNTIME_POLICY_CLOCK = 1, XVRAM_TORCH_RUNTIME_POLICY_LRU = 2 };

typedef uint32_t xvram_torch_runtime_residency_hint;
enum {
  XVRAM_TORCH_RUNTIME_HINT_NORMAL = 1,
  XVRAM_TORCH_RUNTIME_HINT_HOT = 2,
  XVRAM_TORCH_RUNTIME_HINT_STREAMING = 3
};

typedef uint32_t xvram_torch_runtime_access_mode;
enum {
  XVRAM_TORCH_RUNTIME_ACCESS_READ = 1,
  XVRAM_TORCH_RUNTIME_ACCESS_READ_WRITE = 2,
  XVRAM_TORCH_RUNTIME_ACCESS_WRITE_ONLY = 3
};

typedef uint32_t xvram_torch_runtime_seal_mode;
enum {
  XVRAM_TORCH_RUNTIME_SEAL_SUCCESS = 1,
  XVRAM_TORCH_RUNTIME_SEAL_CANCELLED_BEFORE_SUBMISSION = 2,
  XVRAM_TORCH_RUNTIME_SEAL_FAILED_AFTER_POSSIBLE_SUBMISSION = 3
};

typedef uint32_t xvram_torch_runtime_lease_state;
enum {
  XVRAM_TORCH_RUNTIME_LEASE_ARMED = 1,
  XVRAM_TORCH_RUNTIME_LEASE_SUBMITTED = 2,
  XVRAM_TORCH_RUNTIME_LEASE_COMPLETED = 3,
  XVRAM_TORCH_RUNTIME_LEASE_CANCELLED = 4,
  XVRAM_TORCH_RUNTIME_LEASE_FAILED = 5,
  XVRAM_TORCH_RUNTIME_LEASE_QUARANTINED = 6
};

typedef struct xvram_torch_runtime_session_config_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  int32_t device_ordinal;
  xvram_torch_runtime_policy policy;
  uint64_t chunk_bytes;
  uint64_t cache_target_bytes;
  uint64_t device_headroom_bytes;
  uint64_t scratch_arena_bytes;
  uint32_t staging_slots;
  uint32_t stall_timeout_milliseconds;
  uint32_t budget_poll_milliseconds;
  uint32_t maximum_transaction_milliseconds;
  uint64_t reserved[8];
} xvram_torch_runtime_session_config_v1;

#define XVRAM_TORCH_RUNTIME_SESSION_CONFIG_V1_INIT                                                 \
  {                                                                                                \
    (uint32_t)sizeof(xvram_torch_runtime_session_config_v1), XVRAM_TORCH_RUNTIME_ABI_VERSION_1,    \
        INT32_C(0), XVRAM_TORCH_RUNTIME_POLICY_CLOCK, UINT64_C(67108864), UINT64_C(0),             \
        UINT64_C(536870912), UINT64_C(536870912), UINT32_C(4), UINT32_C(5000), UINT32_C(100),      \
        UINT32_C(250), {                                                                           \
      UINT64_C(0), UINT64_C(0), UINT64_C(0), UINT64_C(0), UINT64_C(0), UINT64_C(0), UINT64_C(0),   \
          UINT64_C(0)                                                                              \
    }                                                                                              \
  }

typedef struct xvram_torch_runtime_allocation_desc_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t bytes;
  xvram_torch_runtime_residency_hint hint;
  uint32_t reserved0;
  uint64_t reserved[4];
} xvram_torch_runtime_allocation_desc_v1;

#define XVRAM_TORCH_RUNTIME_ALLOCATION_DESC_V1_INIT                                                \
  {                                                                                                \
    (uint32_t)sizeof(xvram_torch_runtime_allocation_desc_v1), XVRAM_TORCH_RUNTIME_ABI_VERSION_1,   \
        UINT64_C(0), XVRAM_TORCH_RUNTIME_HINT_NORMAL, UINT32_C(0), {                               \
      UINT64_C(0), UINT64_C(0), UINT64_C(0), UINT64_C(0)                                           \
    }                                                                                              \
  }

typedef struct xvram_torch_runtime_allocation_info_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t bytes;
  xvram_torch_runtime_residency_hint hint;
  uint32_t reserved0;
  uint64_t reserved[4];
} xvram_torch_runtime_allocation_info_v1;

#define XVRAM_TORCH_RUNTIME_ALLOCATION_INFO_V1_INIT                                                \
  {                                                                                                \
    (uint32_t)sizeof(xvram_torch_runtime_allocation_info_v1), XVRAM_TORCH_RUNTIME_ABI_VERSION_1,   \
        UINT64_C(0), XVRAM_TORCH_RUNTIME_HINT_NORMAL, UINT32_C(0), {                               \
      UINT64_C(0), UINT64_C(0), UINT64_C(0), UINT64_C(0)                                           \
    }                                                                                              \
  }

typedef struct xvram_torch_runtime_access_range_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  xvram_torch_runtime_allocation allocation;
  uint64_t byte_offset;
  uint64_t byte_length;
  xvram_torch_runtime_access_mode mode;
  uint32_t reserved0;
  uint64_t reserved[2];
} xvram_torch_runtime_access_range_v1;

#define XVRAM_TORCH_RUNTIME_ACCESS_RANGE_V1_INIT                                                   \
  {                                                                                                \
    (uint32_t)sizeof(xvram_torch_runtime_access_range_v1), XVRAM_TORCH_RUNTIME_ABI_VERSION_1,      \
        XVRAM_TORCH_RUNTIME_INVALID_HANDLE, UINT64_C(0), UINT64_C(0),                              \
        XVRAM_TORCH_RUNTIME_ACCESS_READ, UINT32_C(0), {                                            \
      UINT64_C(0), UINT64_C(0)                                                                     \
    }                                                                                              \
  }

typedef struct xvram_torch_runtime_lease_desc_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  const xvram_torch_runtime_access_range_v1* ranges;
  size_t range_count;
  uint64_t reserved[6];
} xvram_torch_runtime_lease_desc_v1;

#define XVRAM_TORCH_RUNTIME_LEASE_DESC_V1_INIT                                                     \
  {                                                                                                \
    (uint32_t)sizeof(xvram_torch_runtime_lease_desc_v1), XVRAM_TORCH_RUNTIME_ABI_VERSION_1, NULL,  \
        (size_t)0, {                                                                               \
      UINT64_C(0), UINT64_C(0), UINT64_C(0), UINT64_C(0), UINT64_C(0), UINT64_C(0)                 \
    }                                                                                              \
  }

typedef struct xvram_torch_runtime_lease_info_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  xvram_torch_runtime_lease_state state;
  xvram_torch_runtime_status result;
  uint64_t live_tensor_storages;
  uint64_t resolved_ranges;
  double elapsed_milliseconds;
  /* Ephemeral runtime-owned CUstream identity; never serialize or log it. */
  uintptr_t stream;
  uint64_t reserved[4];
} xvram_torch_runtime_lease_info_v1;

#define XVRAM_TORCH_RUNTIME_LEASE_INFO_V1_INIT                                                     \
  {                                                                                                \
    (uint32_t)sizeof(xvram_torch_runtime_lease_info_v1), XVRAM_TORCH_RUNTIME_ABI_VERSION_1,        \
        XVRAM_TORCH_RUNTIME_LEASE_ARMED, XVRAM_TORCH_RUNTIME_SUCCESS, UINT64_C(0), UINT64_C(0),    \
        0.0, (uintptr_t)0, {                                                                       \
      UINT64_C(0), UINT64_C(0), UINT64_C(0), UINT64_C(0)                                           \
    }                                                                                              \
  }

typedef struct xvram_torch_runtime_telemetry_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t allocations_created;
  uint64_t allocations_released;
  uint64_t handles_created;
  uint64_t handles_reused;
  uint64_t maps;
  uint64_t set_access_calls;
  uint64_t unmaps;
  uint64_t event_boundaries;
  uint64_t unsafe_remaps;
  uint64_t unsafe_transitions;
  uint64_t h2d_bytes;
  uint64_t d2h_bytes;
  uint64_t cache_hits;
  uint64_t cache_misses;
  uint64_t clean_evictions;
  uint64_t dirty_evictions;
  uint64_t dirty_writebacks;
  uint64_t prefetches;
  uint64_t leases_acquired;
  uint64_t leases_sealed;
  uint64_t leases_retired;
  uint64_t events_recorded;
  uint64_t events_retired;
  uint64_t watchdog_rejections;
  uint64_t budget_shrinks;
  uint64_t budget_grows;
  uint64_t resident_bytes;
  uint64_t resident_peak_bytes;
  uint64_t cache_target_bytes;
  uint64_t tensor_views_created;
  uint64_t tensor_views_live;
  uint64_t tensor_views_peak;
  uint64_t scratch_arena_bytes;
  uint64_t scratch_allocation_calls;
  uint64_t scratch_free_calls;
  uint64_t scratch_failures;
  uint64_t scratch_bytes_current;
  uint64_t scratch_bytes_peak;
  uint32_t stable_addresses;
  uint32_t no_physical_aliases;
  uint32_t quarantined;
  uint32_t reserved0;
  uint64_t reserved[8];
} xvram_torch_runtime_telemetry_v1;

#define XVRAM_TORCH_RUNTIME_TELEMETRY_V1_INIT                                                      \
  {(uint32_t)sizeof(xvram_torch_runtime_telemetry_v1), XVRAM_TORCH_RUNTIME_ABI_VERSION_1}

typedef struct xvram_torch_runtime_error_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  xvram_torch_runtime_status status;
  uint32_t reserved0;
  int64_t native_code;
  char stage[32];
  char operation[64];
  char message[256];
} xvram_torch_runtime_error_v1;

#define XVRAM_TORCH_RUNTIME_ERROR_V1_INIT                                                          \
  {(uint32_t)sizeof(xvram_torch_runtime_error_v1),                                                 \
   XVRAM_TORCH_RUNTIME_ABI_VERSION_1,                                                              \
   XVRAM_TORCH_RUNTIME_SUCCESS,                                                                    \
   UINT32_C(0),                                                                                    \
   INT64_C(0),                                                                                     \
   {0},                                                                                            \
   {0},                                                                                            \
   {0}}

typedef struct xvram_torch_runtime_api_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  const char*(XVRAM_TORCH_RUNTIME_CALL* status_name)(xvram_torch_runtime_status status);
  xvram_torch_runtime_status(XVRAM_TORCH_RUNTIME_CALL* session_create)(
      const xvram_torch_runtime_session_config_v1*, xvram_torch_runtime_session*);
  xvram_torch_runtime_status(XVRAM_TORCH_RUNTIME_CALL* session_get_error)(
      xvram_torch_runtime_session, xvram_torch_runtime_error_v1*);
  xvram_torch_runtime_status(XVRAM_TORCH_RUNTIME_CALL* session_get_telemetry)(
      xvram_torch_runtime_session, xvram_torch_runtime_telemetry_v1*);
  xvram_torch_runtime_status(XVRAM_TORCH_RUNTIME_CALL* session_close)(
      xvram_torch_runtime_session, uint64_t timeout_milliseconds);
  xvram_torch_runtime_status(XVRAM_TORCH_RUNTIME_CALL* allocation_create)(
      xvram_torch_runtime_session, const xvram_torch_runtime_allocation_desc_v1*,
      xvram_torch_runtime_allocation*);
  xvram_torch_runtime_status(XVRAM_TORCH_RUNTIME_CALL* allocation_get_info)(
      xvram_torch_runtime_allocation, xvram_torch_runtime_allocation_info_v1*);
  xvram_torch_runtime_status(XVRAM_TORCH_RUNTIME_CALL* allocation_write)(
      xvram_torch_runtime_allocation, uint64_t byte_offset, const void*, uint64_t byte_length);
  xvram_torch_runtime_status(XVRAM_TORCH_RUNTIME_CALL* allocation_read)(
      xvram_torch_runtime_allocation, uint64_t byte_offset, void*, uint64_t byte_length);
  xvram_torch_runtime_status(XVRAM_TORCH_RUNTIME_CALL* allocation_release)(
      xvram_torch_runtime_allocation);
  xvram_torch_runtime_status(XVRAM_TORCH_RUNTIME_CALL* allocation_discard)(
      xvram_torch_runtime_allocation, uint64_t byte_offset, uint64_t byte_length);
  xvram_torch_runtime_status(XVRAM_TORCH_RUNTIME_CALL* session_prefetch)(
      xvram_torch_runtime_session, const xvram_torch_runtime_access_range_v1*, size_t range_count);
  xvram_torch_runtime_status(XVRAM_TORCH_RUNTIME_CALL* lease_acquire)(
      xvram_torch_runtime_session, const xvram_torch_runtime_lease_desc_v1*,
      xvram_torch_runtime_lease*);
  xvram_torch_runtime_status(XVRAM_TORCH_RUNTIME_CALL* lease_get_info)(
      xvram_torch_runtime_lease, xvram_torch_runtime_lease_info_v1*);
  xvram_torch_runtime_status(XVRAM_TORCH_RUNTIME_CALL* lease_seal)(xvram_torch_runtime_lease,
                                                                   xvram_torch_runtime_seal_mode);
  xvram_torch_runtime_status(XVRAM_TORCH_RUNTIME_CALL* lease_poll)(
      xvram_torch_runtime_lease, xvram_torch_runtime_lease_info_v1*);
  xvram_torch_runtime_status(XVRAM_TORCH_RUNTIME_CALL* lease_wait)(
      xvram_torch_runtime_lease, uint64_t timeout_milliseconds, xvram_torch_runtime_lease_info_v1*);
  uint64_t reserved[16];
} xvram_torch_runtime_api_v1;

/* CUDAPluggableAllocator-compatible, lease-bounded scratch callbacks. */
XVRAM_TORCH_RUNTIME_API void* XVRAM_TORCH_RUNTIME_CALL
xvram_torch_runtime_scratch_alloc(size_t bytes, int device, void* stream);
XVRAM_TORCH_RUNTIME_API void XVRAM_TORCH_RUNTIME_CALL
xvram_torch_runtime_scratch_free(void* pointer, size_t bytes, int device, void* stream);

/* The sole control symbol. */
XVRAM_TORCH_RUNTIME_API xvram_torch_runtime_status XVRAM_TORCH_RUNTIME_CALL
xvram_torch_runtime_get_api(uint32_t requested_abi_version, uint32_t caller_api_size,
                            void* output_api);

#ifdef __cplusplus
}
#endif

#endif
