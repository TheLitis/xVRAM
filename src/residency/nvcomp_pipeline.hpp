#pragma once

#include "platform/cuda/cuda_api.hpp"
#include "platform/nvcomp/nvcomp_api.hpp"
#include "residency/compression.hpp"

#include <chrono>
#include <compare>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace xvram::residency {

inline constexpr std::uint64_t nvcomp_lz4_block_bytes = 64ULL * 1024ULL;

enum class NvcompPipelineStatus {
  success,
  not_ready,
  busy,
  invalid_configuration,
  invalid_argument,
  stale_ticket,
  resource_exhausted,
  cuda_failure,
  codec_failure,
  corrupt_data,
  timeout,
  quarantined,
  cleanup_failure,
  closed,
};

[[nodiscard]] const char* nvcomp_pipeline_status_name(NvcompPipelineStatus status) noexcept;

enum class NvcompOperationKind { decode, encode };

// Optional override for the embedded verifier. The callback submits a bounded verification kernel
// to `stream`. The kernel must compare the
// bytes at `stable_address` with `expected_token` and write one to `device_match` on equality or
// zero otherwise. The result buffer is initialized to zero before submission. Returning success
// means that a kernel may have been submitted, so all later failures are quarantine boundaries.
using NvcompVerificationSubmit = cuda::abi::Result (*)(void* user_data,
                                                       cuda::abi::DevicePointer stable_address,
                                                       std::uint64_t valid_bytes,
                                                       ContentToken expected_token,
                                                       cuda::abi::DevicePointer device_match,
                                                       cuda::abi::Stream stream) noexcept;

struct NvcompPipelineConfig {
  std::uint64_t chunk_bytes = 64ULL * 1024ULL * 1024ULL;
  std::uint32_t slot_count = 2;
  std::uint64_t workspace_cap_bytes = 256ULL * 1024ULL * 1024ULL;
  std::chrono::milliseconds wait_timeout{5000};
  // Null selects the embedded xvram_lz4_verify_v1 PTX kernel.
  NvcompVerificationSubmit verification_submit = nullptr;
  void* verification_user_data = nullptr;
  // Optional synchronous observer for exact codec/transfer generation boundaries. It is advisory
  // and cannot affect pipeline state. Events never contain a CUDA address or native handle.
  CompressionTraceObserver trace_observer = nullptr;
  void* trace_user_data = nullptr;
};

struct NvcompPipelineTicket {
  std::uint64_t operation_id = 0;
  std::uint64_t slot_generation = 0;
  std::uint64_t source_generation = 0;
  std::uint32_t slot_index = 0;
  NvcompOperationKind kind = NvcompOperationKind::decode;

  [[nodiscard]] explicit operator bool() const noexcept {
    return operation_id != 0;
  }
  [[nodiscard]] auto operator<=>(const NvcompPipelineTicket&) const noexcept = default;
};

struct NvcompDecodeRequest {
  const Lz4BlocksV1* source = nullptr;
  cuda::abi::DevicePointer stable_output = 0;
  ChunkKey key{};
  CompressionPath path = CompressionPath::cpu_lz4_gpu_decode;
  bool speculative = false;
};

struct NvcompEncodeRequest {
  cuda::abi::DevicePointer stable_input = 0;
  std::uint64_t valid_bytes = 0;
  std::uint64_t source_generation = 0;
  // When supplied, the CPU-decoded candidate must match this token. If omitted, the verified
  // candidate token becomes authoritative for the new immutable generation.
  std::optional<ContentToken> expected_token;
  ChunkKey key{};
  CompressionPath path = CompressionPath::nvcomp_gpu_codec;
  bool speculative = false;
};

struct NvcompPipelineTelemetry {
  std::uint64_t logical_decode_bytes = 0;
  std::uint64_t logical_encode_bytes = 0;
  std::uint64_t pcie_h2d_payload_bytes = 0;
  std::uint64_t pcie_d2h_payload_bytes = 0;
  std::uint64_t pcie_h2d_metadata_bytes = 0;
  std::uint64_t pcie_d2h_metadata_bytes = 0;

  std::uint64_t workspace_bytes = 0;
  std::uint64_t workspace_peak_bytes = 0;
  // Exact non-workspace device capacity derived from nvCOMP's output-size and alignment
  // requirements, including the per-slot metadata/status/token allocation.
  std::uint64_t device_slot_capacity_bytes = 0;
  std::uint64_t device_slot_bytes = 0;
  std::uint64_t device_slot_peak_bytes = 0;
  std::uint64_t pinned_slot_bytes = 0;
  std::uint64_t pinned_slot_peak_bytes = 0;
  std::uint64_t slots_created = 0;
  std::uint64_t slots_reused = 0;

  std::uint64_t operations_submitted = 0;
  std::uint64_t operations_retired = 0;
  std::uint64_t decode_batches = 0;
  std::uint64_t encode_batches = 0;
  std::uint64_t blocks_decoded = 0;
  std::uint64_t blocks_encoded = 0;
  std::uint64_t raw_h2d_blocks = 0;
  std::uint64_t zero_h2d_blocks = 0;
  std::uint64_t compressed_h2d_blocks = 0;
  std::uint64_t compressed_d2h_blocks = 0;
  std::uint64_t raw_fallback_blocks = 0;
  std::uint64_t raw_fallback_bytes = 0;

  std::uint64_t events_recorded = 0;
  std::uint64_t events_retired = 0;
  std::uint64_t verification_submissions = 0;
  std::uint64_t verification_failures = 0;
  std::uint64_t verification_nanoseconds = 0;
  std::uint64_t codec_failures = 0;
  std::uint64_t cuda_failures = 0;
  std::uint64_t quarantines = 0;
};

class NvcompLz4Pipeline {
public:
  NvcompLz4Pipeline(cuda::CudaApi& cuda_api, nvcomp::NvcompApi& nvcomp_api,
                    const BlockCodec& cpu_lz4_codec, NvcompPipelineConfig config);
  ~NvcompLz4Pipeline();

  NvcompLz4Pipeline(const NvcompLz4Pipeline&) = delete;
  NvcompLz4Pipeline& operator=(const NvcompLz4Pipeline&) = delete;
  NvcompLz4Pipeline(NvcompLz4Pipeline&&) = delete;
  NvcompLz4Pipeline& operator=(NvcompLz4Pipeline&&) = delete;

  [[nodiscard]] NvcompPipelineStatus setup() noexcept;
  [[nodiscard]] NvcompPipelineStatus decode(const NvcompDecodeRequest& request,
                                            NvcompPipelineTicket& ticket) noexcept;
  [[nodiscard]] NvcompPipelineStatus encode(const NvcompEncodeRequest& request,
                                            NvcompPipelineTicket& ticket) noexcept;

  // For encode, candidate must be non-null when the final transfer generation completes. Decode
  // ignores candidate. Decode success retires immediately; encode success deliberately retains
  // its generation-safe slot until acknowledge_host_commit() proves that the dirty generation
  // became authoritative host backing. The authority may be this candidate or a verified raw
  // D2H fallback selected by the adaptive cost model.
  [[nodiscard]] NvcompPipelineStatus poll(const NvcompPipelineTicket& ticket,
                                          Lz4BlocksV1* candidate = nullptr) noexcept;
  [[nodiscard]] NvcompPipelineStatus wait(const NvcompPipelineTicket& ticket,
                                          Lz4BlocksV1* candidate = nullptr) noexcept;
  [[nodiscard]] NvcompPipelineStatus
  acknowledge_host_commit(const NvcompPipelineTicket& ticket) noexcept;
  // A completed encode whose candidate cannot be committed must remain a quarantine boundary:
  // reusing its slot could erase the last known copy of a dirty generation.
  [[nodiscard]] NvcompPipelineStatus
  quarantine_uncommitted(const NvcompPipelineTicket& ticket) noexcept;
  [[nodiscard]] NvcompPipelineStatus close() noexcept;

  [[nodiscard]] const NvcompPipelineConfig& config() const noexcept;
  [[nodiscard]] const NvcompPipelineTelemetry& telemetry() const noexcept;
  [[nodiscard]] const std::string& error() const noexcept;
  [[nodiscard]] std::optional<std::int64_t> native_error() const noexcept;
  // True means a post-submission completion boundary became unknowable. The caller must retain
  // the stable mapping, physical handle and reservation and terminate its isolated worker; close
  // deliberately leaks the slot-owned CUDA/nvCOMP resources in this state.
  [[nodiscard]] bool quarantined() const noexcept;
  [[nodiscard]] bool is_setup() const noexcept;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace xvram::residency
