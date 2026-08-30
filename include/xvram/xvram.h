#ifndef XVRAM_XVRAM_H
#define XVRAM_XVRAM_H

#include <stddef.h>
#include <stdint.h>

#if UINTPTR_MAX != UINT64_MAX
#error "xVRAM SDK v1 supports 64-bit targets only"
#endif

#if defined(_WIN32)
#define XVRAM_CALL __cdecl
#if defined(XVRAM_BUILDING_LIBRARY)
#define XVRAM_API __declspec(dllexport)
#elif defined(XVRAM_USING_LIBRARY)
#define XVRAM_API __declspec(dllimport)
#else
#define XVRAM_API
#endif
#elif defined(__GNUC__) && defined(XVRAM_BUILDING_LIBRARY)
#define XVRAM_CALL
#define XVRAM_API __attribute__((visibility("default")))
#else
#define XVRAM_CALL
#define XVRAM_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ABI policy
 * ----------
 * Every public structure starts with struct_size. Callers set it to sizeof(the
 * structure they compiled against). xVRAM reads only the v1 prefix and rejects
 * structures smaller than that prefix. Reserved fields must be zero.
 *
 * All byte counts, element counts, logical identifiers, CUDA device addresses,
 * streams, and context values use fixed-width integers. Native CUDA header types
 * are intentionally absent from this header.
 */

#define XVRAM_ABI_VERSION_1 UINT32_C(1)
#define XVRAM_ABI_VERSION_CURRENT XVRAM_ABI_VERSION_1
#define XVRAM_TIMEOUT_INFINITE UINT64_MAX

typedef struct xvram_session_t* xvram_session;
typedef struct xvram_allocation_t* xvram_allocation;
typedef struct xvram_operation_t* xvram_operation;
typedef struct xvram_gemm_plan_t* xvram_gemm_plan;

typedef uint32_t xvram_status;
enum {
  XVRAM_STATUS_SUCCESS = 0,
  XVRAM_STATUS_INVALID_ARGUMENT = 1,
  XVRAM_STATUS_INCOMPATIBLE_ABI = 2,
  XVRAM_STATUS_UNSUPPORTED = 3,
  XVRAM_STATUS_UNAVAILABLE = 4,
  XVRAM_STATUS_HOST_OUT_OF_MEMORY = 5,
  XVRAM_STATUS_DEVICE_OUT_OF_MEMORY = 6,
  XVRAM_STATUS_BUDGET_PRESSURE = 7,
  XVRAM_STATUS_BUSY = 8,
  XVRAM_STATUS_NOT_READY = 9,
  XVRAM_STATUS_TIMEOUT = 10,
  XVRAM_STATUS_CORRUPTION = 11,
  XVRAM_STATUS_CALLBACK_FAILED = 12,
  XVRAM_STATUS_CUDA_ERROR = 13,
  XVRAM_STATUS_CUBLAS_ERROR = 14,
  XVRAM_STATUS_POISONED = 15,
  XVRAM_STATUS_CLEANUP_FAILED = 16,
  XVRAM_STATUS_CLOSED = 17,
  XVRAM_STATUS_CANCELLED = 18,
  XVRAM_STATUS_INTERNAL = 19
};

typedef uint32_t xvram_native_error_domain;
enum {
  XVRAM_NATIVE_ERROR_NONE = 0,
  XVRAM_NATIVE_ERROR_CUDA = 1,
  XVRAM_NATIVE_ERROR_CUBLAS = 2,
  XVRAM_NATIVE_ERROR_PLATFORM = 3,
  XVRAM_NATIVE_ERROR_CALLBACK = 4,
  XVRAM_NATIVE_ERROR_INTERNAL = 5
};

typedef struct xvram_error_info_v1 {
  uint32_t struct_size;
  xvram_status status;
  xvram_native_error_domain native_domain;
  uint32_t reserved0;
  int64_t native_code;
  char stage[32];
  char operation[64];
  char native_name[64];
  char message[256];
} xvram_error_info_v1;

#define XVRAM_ERROR_INFO_V1_INIT                                                                   \
  {(uint32_t)sizeof(xvram_error_info_v1),                                                          \
   XVRAM_STATUS_SUCCESS,                                                                           \
   XVRAM_NATIVE_ERROR_NONE,                                                                        \
   0U,                                                                                             \
   INT64_C(0),                                                                                     \
   {0},                                                                                            \
   {0},                                                                                            \
   {0},                                                                                            \
   {0}}

typedef uint32_t xvram_context_mode;
enum { XVRAM_CONTEXT_ISOLATED = 0, XVRAM_CONTEXT_ATTACH_CURRENT = 1 };

typedef uint32_t xvram_cache_policy;
enum { XVRAM_CACHE_POLICY_CLOCK = 0, XVRAM_CACHE_POLICY_LRU = 1 };

typedef struct xvram_session_config_v1 {
  uint32_t struct_size;
  uint32_t flags;
  int32_t device_ordinal;
  xvram_context_mode context_mode;
  uint64_t chunk_size_bytes;
  uint64_t cache_target_bytes;
  uint64_t device_headroom_bytes;
  uint64_t workspace_cap_bytes;
  uint64_t budget_poll_ms;
  uint64_t max_transaction_ms;
  uint32_t staging_slots;
  xvram_cache_policy cache_policy;
  uint32_t prefetch_distance;
  uint32_t reserved0;
  uint64_t reserved[4];
} xvram_session_config_v1;

/* cache_target_bytes and max_transaction_ms use zero for automatic selection. */
#define XVRAM_SESSION_CONFIG_V1_INIT                                                               \
  {(uint32_t)sizeof(xvram_session_config_v1),                                                      \
   0U,                                                                                             \
   0,                                                                                              \
   XVRAM_CONTEXT_ISOLATED,                                                                         \
   UINT64_C(67108864),                                                                             \
   UINT64_C(0),                                                                                    \
   UINT64_C(536870912),                                                                            \
   UINT64_C(4194304),                                                                              \
   UINT64_C(100),                                                                                  \
   UINT64_C(0),                                                                                    \
   4U,                                                                                             \
   XVRAM_CACHE_POLICY_CLOCK,                                                                       \
   2U,                                                                                             \
   0U,                                                                                             \
   {0, 0, 0, 0}}

typedef uint32_t xvram_allocation_priority;
enum { XVRAM_ALLOCATION_NORMAL = 0, XVRAM_ALLOCATION_HOT = 1, XVRAM_ALLOCATION_STREAMING = 2 };

enum { XVRAM_ALLOCATION_FLAG_ZERO_INITIALIZE = UINT32_C(1) << 0 };

typedef struct xvram_allocation_desc_v1 {
  uint32_t struct_size;
  uint32_t flags;
  uint64_t size_bytes;
  uint64_t alignment_bytes;
  xvram_allocation_priority priority;
  uint32_t reserved0;
  uint64_t reserved[2];
} xvram_allocation_desc_v1;

#define XVRAM_ALLOCATION_DESC_V1_INIT                                                              \
  {(uint32_t)sizeof(xvram_allocation_desc_v1),                                                     \
   0U,                                                                                             \
   UINT64_C(0),                                                                                    \
   UINT64_C(0),                                                                                    \
   XVRAM_ALLOCATION_NORMAL,                                                                        \
   0U,                                                                                             \
   {0, 0}}

typedef struct xvram_allocation_info_v1 {
  uint32_t struct_size;
  uint32_t flags;
  uint64_t allocation_id;
  uint64_t size_bytes;
  uint64_t alignment_bytes;
  xvram_allocation_priority priority;
  uint32_t stable_virtual_address;
  uint64_t reserved[2];
} xvram_allocation_info_v1;

typedef uint32_t xvram_access_mode;
enum { XVRAM_ACCESS_READ = 0, XVRAM_ACCESS_READ_WRITE = 1, XVRAM_ACCESS_WRITE_ONLY = 2 };

typedef struct xvram_access_range_v1 {
  xvram_allocation allocation;
  uint64_t offset_bytes;
  uint64_t length_bytes;
  xvram_access_mode mode;
  uint32_t reserved0;
} xvram_access_range_v1;

typedef struct xvram_prefetch_desc_v1 {
  uint32_t struct_size;
  uint32_t flags;
  const xvram_access_range_v1* ranges;
  uint64_t range_count;
  /* Submission-relative monotonic deadline. Zero disables the deadline. */
  uint64_t deadline_ms;
  uint64_t reserved[2];
} xvram_prefetch_desc_v1;

typedef struct xvram_resolved_range_v1 {
  uint64_t allocation_id;
  uint64_t offset_bytes;
  uint64_t length_bytes;
  uint64_t device_address;
  xvram_access_mode mode;
  uint32_t reserved0;
} xvram_resolved_range_v1;

typedef struct xvram_transaction_context_v1 {
  uint32_t struct_size;
  uint32_t flags;
  uint64_t operation_id;
  uint64_t transaction_id;
  const xvram_resolved_range_v1* ranges;
  uint64_t range_count;
  uint64_t workspace_device_address;
  uint64_t workspace_bytes;
  uint64_t native_stream;
  uint64_t reserved[2];
} xvram_transaction_context_v1;

/*
 * The callback runs on the session worker thread with the session CUDA context
 * current. It may only enqueue work on native_stream. It must not synchronize or
 * replace the context, and must not retain any pointer/address from context after
 * returning. Returning anything other than SUCCESS fails the operation; if work
 * was already submitted, the session is conservatively poisoned.
 */
typedef xvram_status(XVRAM_CALL* xvram_transaction_callback_v1)(
    const xvram_transaction_context_v1* context, void* user_data,
    xvram_error_info_v1* callback_error);

typedef struct xvram_transaction_desc_v1 {
  uint32_t struct_size;
  uint32_t flags;
  const xvram_access_range_v1* ranges;
  uint64_t range_count;
  uint64_t workspace_bytes;
  /* Submission-relative monotonic deadline. Zero disables the deadline. */
  uint64_t deadline_ms;
  xvram_transaction_callback_v1 callback;
  void* user_data;
  uint64_t reserved[2];
} xvram_transaction_desc_v1;

typedef uint32_t xvram_operation_state;
enum {
  XVRAM_OPERATION_QUEUED = 0,
  XVRAM_OPERATION_RUNNING = 1,
  XVRAM_OPERATION_COMPLETED = 2,
  XVRAM_OPERATION_CANCELLED = 3,
  XVRAM_OPERATION_FAILED = 4
};

typedef struct xvram_operation_info_v1 {
  uint32_t struct_size;
  xvram_operation_state state;
  xvram_status result;
  uint32_t reserved0;
  uint64_t operation_id;
  uint64_t units_completed;
  uint64_t units_total;
  double elapsed_ms;
  uint64_t bytes_h2d;
  uint64_t bytes_d2h;
  uint64_t cache_hits;
  uint64_t cache_misses;
  uint64_t reserved[2];
} xvram_operation_info_v1;

#define XVRAM_OPERATION_INFO_V1_INIT                                                               \
  {(uint32_t)sizeof(xvram_operation_info_v1),                                                      \
   XVRAM_OPERATION_QUEUED,                                                                         \
   XVRAM_STATUS_SUCCESS,                                                                           \
   0U,                                                                                             \
   UINT64_C(0),                                                                                    \
   UINT64_C(0),                                                                                    \
   UINT64_C(0),                                                                                    \
   0.0,                                                                                            \
   UINT64_C(0),                                                                                    \
   UINT64_C(0),                                                                                    \
   UINT64_C(0),                                                                                    \
   UINT64_C(0),                                                                                    \
   {0, 0}}

typedef uint32_t xvram_data_type;
enum { XVRAM_DATA_FP16 = 0, XVRAM_DATA_BF16 = 1, XVRAM_DATA_FP32 = 2, XVRAM_DATA_FP64 = 3 };

typedef uint32_t xvram_compute_mode;
enum {
  XVRAM_COMPUTE_AUTO = 0,
  XVRAM_COMPUTE_FP32_STRICT = 1,
  XVRAM_COMPUTE_FP32_TF32 = 2,
  XVRAM_COMPUTE_FP64 = 3
};

typedef uint32_t xvram_matrix_layout;
enum { XVRAM_MATRIX_ROW_MAJOR = 0, XVRAM_MATRIX_COLUMN_MAJOR = 1 };

typedef uint32_t xvram_matrix_operation;
enum { XVRAM_MATRIX_OP_N = 0, XVRAM_MATRIX_OP_T = 1 };

typedef struct xvram_matrix_v1 {
  uint32_t struct_size;
  xvram_data_type data_type;
  xvram_allocation allocation;
  uint64_t offset_bytes;
  uint64_t rows;
  uint64_t columns;
  uint64_t leading_dimension;
  xvram_matrix_layout layout;
  uint32_t reserved0;
  uint64_t reserved[2];
} xvram_matrix_v1;

#define XVRAM_MATRIX_V1_INIT                                                                       \
  {(uint32_t)sizeof(xvram_matrix_v1),                                                              \
   XVRAM_DATA_FP32,                                                                                \
   (xvram_allocation)0,                                                                            \
   UINT64_C(0),                                                                                    \
   UINT64_C(0),                                                                                    \
   UINT64_C(0),                                                                                    \
   UINT64_C(0),                                                                                    \
   XVRAM_MATRIX_ROW_MAJOR,                                                                         \
   0U,                                                                                             \
   {0, 0}}

typedef struct xvram_gemm_desc_v1 {
  uint32_t struct_size;
  uint32_t flags;
  xvram_matrix_v1 a;
  xvram_matrix_v1 b;
  xvram_matrix_v1 c;
  xvram_matrix_operation operation_a;
  xvram_matrix_operation operation_b;
  xvram_compute_mode compute_mode;
  uint32_t reserved0;
  double alpha;
  double beta;
  uint64_t workspace_cap_bytes;
  /* Submission-relative monotonic deadline. Zero disables the deadline. */
  uint64_t deadline_ms;
  uint64_t tile_m_hint;
  uint64_t tile_n_hint;
  uint64_t tile_k_hint;
  uint64_t reserved[3];
} xvram_gemm_desc_v1;

/* alpha/beta are converted to FP32 for FP16/BF16/FP32 GEMM and retained as FP64
 * for FP64 GEMM. Zero tile hints select the working-set planner. */
#define XVRAM_GEMM_DESC_V1_INIT                                                                    \
  {(uint32_t)sizeof(xvram_gemm_desc_v1),                                                           \
   0U,                                                                                             \
   XVRAM_MATRIX_V1_INIT,                                                                           \
   XVRAM_MATRIX_V1_INIT,                                                                           \
   XVRAM_MATRIX_V1_INIT,                                                                           \
   XVRAM_MATRIX_OP_N,                                                                              \
   XVRAM_MATRIX_OP_N,                                                                              \
   XVRAM_COMPUTE_AUTO,                                                                             \
   0U,                                                                                             \
   1.0,                                                                                            \
   0.0,                                                                                            \
   UINT64_C(0),                                                                                    \
   UINT64_C(0),                                                                                    \
   UINT64_C(0),                                                                                    \
   UINT64_C(0),                                                                                    \
   UINT64_C(0),                                                                                    \
   {0, 0, 0}}

typedef struct xvram_gemm_plan_info_v1 {
  uint32_t struct_size;
  uint32_t flags;
  uint64_t m;
  uint64_t n;
  uint64_t k;
  uint64_t tile_m;
  uint64_t tile_n;
  uint64_t tile_k;
  uint64_t maximum_working_set_bytes;
  uint64_t workspace_bytes;
  uint64_t tile_count;
  xvram_compute_mode effective_compute_mode;
  uint32_t uses_cublas_lt;
  uint64_t reserved[2];
} xvram_gemm_plan_info_v1;

#define XVRAM_GEMM_PLAN_INFO_V1_INIT                                                               \
  {(uint32_t)sizeof(xvram_gemm_plan_info_v1),                                                      \
   0U,                                                                                             \
   UINT64_C(0),                                                                                    \
   UINT64_C(0),                                                                                    \
   UINT64_C(0),                                                                                    \
   UINT64_C(0),                                                                                    \
   UINT64_C(0),                                                                                    \
   UINT64_C(0),                                                                                    \
   UINT64_C(0),                                                                                    \
   UINT64_C(0),                                                                                    \
   UINT64_C(0),                                                                                    \
   XVRAM_COMPUTE_AUTO,                                                                             \
   0U,                                                                                             \
   {0, 0}}

typedef struct xvram_session_telemetry_v1 {
  uint32_t struct_size;
  uint32_t flags;
  uint64_t cache_target_bytes;
  uint64_t cache_target_minimum_bytes;
  uint64_t cache_target_maximum_bytes;
  uint64_t resident_bytes;
  uint64_t resident_bytes_peak;
  uint64_t workspace_bytes;
  uint64_t bytes_h2d;
  uint64_t bytes_d2h;
  uint64_t cache_hits;
  uint64_t cache_misses;
  uint64_t clean_evictions;
  uint64_t dirty_evictions;
  uint64_t writebacks_completed;
  uint64_t mappings;
  uint64_t unmaps;
  uint64_t set_access_calls;
  uint64_t handle_reuses;
  uint64_t unsafe_remaps;
  uint64_t unsafe_transitions;
  uint64_t budget_shrinks;
  uint64_t budget_grows;
  uint32_t cuda_driver_version;
  uint32_t cublas_version;
  uint32_t cublas_lt_available;
  uint32_t poisoned;
  uint64_t physical_handles_created;
  uint64_t physical_handles_released;
  uint64_t event_boundaries;
  uint64_t pinned_staging_bytes;
  uint64_t transactions_completed;
  uint64_t algorithm_selections;
  uint64_t algorithm_cache_hits;
  double last_transaction_ms;
  uint64_t budget_sample_count;
  uint64_t cuda_free_bytes_minimum;
  uint64_t cuda_free_bytes_end;
  uint32_t cublas_lt_version;
  uint32_t cublas_library_source;
  uint64_t wddm_available_bytes_minimum;
  uint64_t wddm_available_bytes_end;
  uint32_t wddm_budget_observed;
  uint32_t watchdog_rejections;
  uint64_t target_oom_retries;
} xvram_session_telemetry_v1;

enum {
  XVRAM_CUBLAS_SOURCE_UNKNOWN = 0,
  XVRAM_CUBLAS_SOURCE_SYSTEM = 1,
  XVRAM_CUBLAS_SOURCE_APP_LOCAL = 2,
  XVRAM_CUBLAS_SOURCE_EXPLICIT = 3
};

typedef struct xvram_api_v1 {
  uint32_t struct_size;
  uint32_t abi_version;

  const char*(XVRAM_CALL* status_name)(xvram_status status);

  xvram_status(XVRAM_CALL* session_create)(const xvram_session_config_v1* config,
                                           xvram_session* out_session);
  xvram_status(XVRAM_CALL* session_get_error)(xvram_session session,
                                              xvram_error_info_v1* out_error);
  xvram_status(XVRAM_CALL* session_get_telemetry)(xvram_session session,
                                                  xvram_session_telemetry_v1* out_telemetry);
  xvram_status(XVRAM_CALL* session_drain)(xvram_session session, uint64_t timeout_ms);
  xvram_status(XVRAM_CALL* session_close)(xvram_session session, uint64_t timeout_ms);
  void(XVRAM_CALL* session_release)(xvram_session session);

  xvram_status(XVRAM_CALL* allocation_create)(xvram_session session,
                                              const xvram_allocation_desc_v1* desc,
                                              xvram_allocation* out_allocation);
  xvram_status(XVRAM_CALL* allocation_get_info)(xvram_allocation allocation,
                                                xvram_allocation_info_v1* out_info);
  xvram_status(XVRAM_CALL* allocation_write)(xvram_allocation allocation, uint64_t offset_bytes,
                                             const void* source, uint64_t size_bytes);
  xvram_status(XVRAM_CALL* allocation_read)(xvram_allocation allocation, uint64_t offset_bytes,
                                            void* destination, uint64_t size_bytes);
  xvram_status(XVRAM_CALL* allocation_release)(xvram_allocation allocation);

  xvram_status(XVRAM_CALL* prefetch_submit)(xvram_session session,
                                            const xvram_prefetch_desc_v1* desc,
                                            xvram_operation* out_operation);
  xvram_status(XVRAM_CALL* transaction_submit)(xvram_session session,
                                               const xvram_transaction_desc_v1* desc,
                                               xvram_operation* out_operation);
  xvram_status(XVRAM_CALL* transaction_execute)(xvram_session session,
                                                const xvram_transaction_desc_v1* desc,
                                                uint64_t timeout_ms);

  xvram_status(XVRAM_CALL* gemm_plan_create)(xvram_session session, const xvram_gemm_desc_v1* desc,
                                             xvram_gemm_plan* out_plan);
  xvram_status(XVRAM_CALL* gemm_plan_get_info)(xvram_gemm_plan plan,
                                               xvram_gemm_plan_info_v1* out_info);
  xvram_status(XVRAM_CALL* gemm_submit)(xvram_gemm_plan plan, xvram_operation* out_operation);
  xvram_status(XVRAM_CALL* gemm_execute)(xvram_gemm_plan plan, uint64_t timeout_ms);
  void(XVRAM_CALL* gemm_plan_release)(xvram_gemm_plan plan);

  xvram_status(XVRAM_CALL* operation_poll)(xvram_operation operation,
                                           xvram_operation_info_v1* out_info);
  xvram_status(XVRAM_CALL* operation_wait)(xvram_operation operation, uint64_t timeout_ms,
                                           xvram_operation_info_v1* out_info);
  xvram_status(XVRAM_CALL* operation_cancel)(xvram_operation operation);
  xvram_status(XVRAM_CALL* operation_get_error)(xvram_operation operation,
                                                xvram_error_info_v1* out_error);
  void(XVRAM_CALL* operation_release)(xvram_operation operation);

  uint64_t reserved[16];
} xvram_api_v1;

/*
 * The only directly exported SDK symbol. out_api must point to caller_api_size
 * writable bytes. v1 requires caller_api_size >= sizeof(xvram_api_v1).
 */
XVRAM_API xvram_status XVRAM_CALL xvram_get_api(uint32_t requested_abi_version,
                                                uint32_t caller_api_size, void* out_api);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* XVRAM_XVRAM_H */
