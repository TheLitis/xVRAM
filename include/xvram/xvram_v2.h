#ifndef XVRAM_XVRAM_V2_H
#define XVRAM_XVRAM_V2_H

#include <xvram/xvram.h>

#ifdef __cplusplus
extern "C" {
#endif

#define XVRAM_ABI_VERSION_2 UINT32_C(2)
#undef XVRAM_ABI_VERSION_CURRENT
#define XVRAM_ABI_VERSION_CURRENT XVRAM_ABI_VERSION_2

typedef uint32_t xvram_compression_mode;
enum {
  XVRAM_COMPRESSION_DISABLED = 0,
  XVRAM_COMPRESSION_ADAPTIVE = 1,
  XVRAM_COMPRESSION_CAPACITY = 2
};

typedef uint32_t xvram_compression_codec;
enum { XVRAM_COMPRESSION_CODEC_AUTO = 0, XVRAM_COMPRESSION_CODEC_LZ4 = 1 };

/*
 * Compression is opt-in through session_create_v2. The embedded v1
 * configuration remains a complete, independently size-tagged value.
 * Zero host limits request the platform-safe automatic value.
 */
typedef struct xvram_session_config_v2 {
  uint32_t struct_size;
  uint32_t flags;
  xvram_session_config_v1 v1;
  xvram_compression_mode compression_mode;
  xvram_compression_codec compression_codec;
  uint64_t host_store_cap_bytes;
  uint64_t host_headroom_bytes;
  uint64_t compression_workspace_cap_bytes;
  uint32_t codec_slots;
  uint32_t codec_workers;
  uint64_t reserved[8];
} xvram_session_config_v2;

#define XVRAM_SESSION_CONFIG_V2_INIT                                                               \
  {(uint32_t)sizeof(xvram_session_config_v2),                                                      \
   0U,                                                                                             \
   XVRAM_SESSION_CONFIG_V1_INIT,                                                                   \
   XVRAM_COMPRESSION_ADAPTIVE,                                                                     \
   XVRAM_COMPRESSION_CODEC_AUTO,                                                                   \
   UINT64_C(0),                                                                                    \
   UINT64_C(0),                                                                                    \
   UINT64_C(268435456),                                                                            \
   2U,                                                                                             \
   2U,                                                                                             \
   {0, 0, 0, 0, 0, 0, 0, 0}}

typedef struct xvram_session_telemetry_v2 {
  uint32_t struct_size;
  uint32_t flags;
  xvram_session_telemetry_v1 v1;

  uint64_t logical_bytes;
  uint64_t host_stored_bytes;
  uint64_t host_stored_peak_bytes;
  uint64_t host_raw_bytes;
  uint64_t host_compressed_bytes;
  uint64_t host_implicit_zero_bytes;
  uint64_t host_invalid_bytes;

  uint64_t logical_h2d_bytes;
  uint64_t pcie_h2d_bytes;
  uint64_t logical_d2h_bytes;
  uint64_t pcie_d2h_bytes;

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
  uint64_t reserved[8];
} xvram_session_telemetry_v2;

/*
 * The v1 table is the first member so every established operation remains at
 * its frozen offset. xvram_get_api remains the SDK's only exported symbol.
 */
typedef struct xvram_api_v2 {
  xvram_api_v1 v1;
  xvram_status(XVRAM_CALL* session_create_v2)(const xvram_session_config_v2* config,
                                              xvram_session* out_session);
  xvram_status(XVRAM_CALL* session_get_telemetry_v2)(xvram_session session,
                                                     xvram_session_telemetry_v2* out_telemetry);
  uint64_t reserved[14];
} xvram_api_v2;

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* XVRAM_XVRAM_V2_H */
