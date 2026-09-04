#ifndef XVRAM_INTERNAL_TORCH_RUNTIME_V2_H
#define XVRAM_INTERNAL_TORCH_RUNTIME_V2_H

#include "torch_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

#define XVRAM_TORCH_RUNTIME_ABI_VERSION_2 UINT32_C(2)
#undef XVRAM_TORCH_RUNTIME_ABI_VERSION_CURRENT
#define XVRAM_TORCH_RUNTIME_ABI_VERSION_CURRENT XVRAM_TORCH_RUNTIME_ABI_VERSION_2

typedef uint32_t xvram_torch_runtime_compression_mode;
enum {
  XVRAM_TORCH_RUNTIME_COMPRESSION_DISABLED = 0,
  XVRAM_TORCH_RUNTIME_COMPRESSION_ADAPTIVE = 1,
  XVRAM_TORCH_RUNTIME_COMPRESSION_CAPACITY = 2
};

typedef uint32_t xvram_torch_runtime_compression_codec;
enum {
  XVRAM_TORCH_RUNTIME_COMPRESSION_CODEC_AUTO = 0,
  XVRAM_TORCH_RUNTIME_COMPRESSION_CODEC_LZ4 = 1
};

typedef struct xvram_torch_runtime_session_config_v2 {
  uint32_t struct_size;
  uint32_t abi_version;
  xvram_torch_runtime_session_config_v1 v1;
  xvram_torch_runtime_compression_mode compression_mode;
  xvram_torch_runtime_compression_codec compression_codec;
  uint64_t host_store_cap_bytes;
  uint64_t host_headroom_bytes;
  uint64_t compression_workspace_cap_bytes;
  uint32_t codec_slots;
  uint32_t codec_workers;
  uint64_t reserved[8];
} xvram_torch_runtime_session_config_v2;

#define XVRAM_TORCH_RUNTIME_SESSION_CONFIG_V2_INIT                                                 \
  {                                                                                                \
    (uint32_t)sizeof(xvram_torch_runtime_session_config_v2), XVRAM_TORCH_RUNTIME_ABI_VERSION_2,    \
        XVRAM_TORCH_RUNTIME_SESSION_CONFIG_V1_INIT, XVRAM_TORCH_RUNTIME_COMPRESSION_ADAPTIVE,      \
        XVRAM_TORCH_RUNTIME_COMPRESSION_CODEC_AUTO, UINT64_C(0), UINT64_C(0), UINT64_C(268435456), \
        UINT32_C(2), UINT32_C(2), {                                                                \
      UINT64_C(0), UINT64_C(0), UINT64_C(0), UINT64_C(0), UINT64_C(0), UINT64_C(0), UINT64_C(0),   \
          UINT64_C(0)                                                                              \
    }                                                                                              \
  }

typedef struct xvram_torch_runtime_telemetry_v2 {
  uint32_t struct_size;
  uint32_t abi_version;
  xvram_torch_runtime_telemetry_v1 v1;

  uint64_t logical_bytes;
  uint64_t host_stored_bytes;
  uint64_t host_stored_peak_bytes;
  uint64_t host_raw_bytes;
  uint64_t host_compressed_bytes;
  uint64_t host_implicit_zero_bytes;
  uint64_t host_invalid_bytes;
  uint64_t effective_host_store_cap_bytes;
  uint64_t effective_host_headroom_bytes;
  uint64_t host_budget_bytes;
  uint64_t host_budget_peak_bytes;
  uint64_t conversion_scratch_peak_bytes;

  uint64_t logical_h2d_bytes;
  uint64_t pcie_h2d_bytes;
  uint64_t pcie_h2d_payload_bytes;
  uint64_t pcie_h2d_metadata_bytes;
  uint64_t logical_d2h_bytes;
  uint64_t pcie_d2h_bytes;
  uint64_t pcie_d2h_payload_bytes;
  uint64_t pcie_d2h_metadata_bytes;
  uint64_t rejected_candidate_logical_d2h_bytes;
  /* The PyTorch backend assigns the hot hint exclusively to persistent state. */
  uint64_t hot_allocation_d2h_bytes;
  uint64_t non_hot_allocation_d2h_bytes;

  uint64_t raw_path_decisions;
  uint64_t cpu_lz4_gpu_decode_decisions;
  uint64_t gpu_lz4_decisions;
  uint64_t never_compress_decisions;
  uint64_t raw_fallbacks;
  uint64_t cpu_codec_fallbacks;
  uint64_t gpu_codec_fallbacks;

  uint64_t compression_attempts;
  uint64_t compression_commits;
  uint64_t decompression_attempts;
  uint64_t decompression_commits;
  uint64_t generations_created;
  uint64_t generations_committed;
  uint64_t generations_discarded;

  uint64_t codec_workspace_bytes;
  uint64_t codec_workspace_peak_bytes;
  uint64_t codec_slot_bytes;
  uint64_t codec_slot_peak_bytes;
  uint64_t spill_reserved_bytes;
  uint64_t spill_reserved_peak_bytes;

  uint64_t cpu_encode_nanoseconds;
  uint64_t cpu_decode_nanoseconds;
  uint64_t gpu_encode_nanoseconds;
  uint64_t gpu_decode_nanoseconds;
  uint64_t verification_nanoseconds;
  uint64_t codec_events_recorded;
  uint64_t codec_events_retired;
  uint64_t reserved[8];
} xvram_torch_runtime_telemetry_v2;

#define XVRAM_TORCH_RUNTIME_TELEMETRY_V2_INIT                                                      \
  {(uint32_t)sizeof(xvram_torch_runtime_telemetry_v2), XVRAM_TORCH_RUNTIME_ABI_VERSION_2,          \
   XVRAM_TORCH_RUNTIME_TELEMETRY_V1_INIT}

typedef struct xvram_torch_runtime_api_v2 {
  xvram_torch_runtime_api_v1 v1;
  xvram_torch_runtime_status(XVRAM_TORCH_RUNTIME_CALL* session_create_v2)(
      const xvram_torch_runtime_session_config_v2*, xvram_torch_runtime_session*);
  xvram_torch_runtime_status(XVRAM_TORCH_RUNTIME_CALL* session_get_telemetry_v2)(
      xvram_torch_runtime_session, xvram_torch_runtime_telemetry_v2*);
  uint64_t reserved[14];
} xvram_torch_runtime_api_v2;

/* No additional direct exports are introduced by this private ABI version. */

#ifdef __cplusplus
}
#endif

#endif /* XVRAM_INTERNAL_TORCH_RUNTIME_V2_H */
