#include "residency/nvcomp_pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <new>
#include <span>
#include <thread>
#include <utility>
#include <vector>

namespace xvram::residency {
namespace {

using cuda::abi::DevicePointer;
using cuda::abi::Event;
using cuda::abi::Result;
using cuda::abi::Stream;

[[nodiscard]] bool checked_add(const std::uint64_t left, const std::uint64_t right,
                               std::uint64_t& output) noexcept {
  if (left > std::numeric_limits<std::uint64_t>::max() - right) {
    return false;
  }
  output = left + right;
  return true;
}

[[nodiscard]] bool checked_mul(const std::uint64_t left, const std::uint64_t right,
                               std::uint64_t& output) noexcept {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
    return false;
  }
  output = left * right;
  return true;
}

[[nodiscard]] bool checked_align_up(const std::uint64_t value, const std::uint64_t alignment,
                                    std::uint64_t& output) noexcept {
  if (alignment == 0) {
    return false;
  }
  const std::uint64_t remainder = value % alignment;
  if (remainder == 0) {
    output = value;
    return true;
  }
  return checked_add(value, alignment - remainder, output);
}

void saturating_add(std::uint64_t& value, const std::uint64_t delta) noexcept {
  value = delta > std::numeric_limits<std::uint64_t>::max() - value
              ? std::numeric_limits<std::uint64_t>::max()
              : value + delta;
}

[[nodiscard]] bool fits_size(const std::uint64_t value) noexcept {
  return value <= static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
}

[[nodiscard]] void* device_void(const DevicePointer pointer) noexcept {
  return reinterpret_cast<void*>(static_cast<std::uintptr_t>(pointer));
}

[[nodiscard]] cudaStream_t runtime_stream(const Stream stream) noexcept {
  return reinterpret_cast<cudaStream_t>(stream);
}

struct MetadataLayout {
  std::uint64_t input_ptrs = 0;
  std::uint64_t input_sizes = 0;
  std::uint64_t output_ptrs = 0;
  std::uint64_t output_sizes = 0;
  std::uint64_t actual_sizes = 0;
  std::uint64_t statuses = 0;
  std::uint64_t verification_token = 0;
  std::uint64_t verification_match = 0;
  std::uint64_t total_bytes = 0;
};

[[nodiscard]] bool append_region(std::uint64_t& cursor, const std::uint64_t count,
                                 const std::uint64_t element_bytes, const std::uint64_t alignment,
                                 std::uint64_t& offset) noexcept {
  std::uint64_t aligned = 0;
  std::uint64_t bytes = 0;
  std::uint64_t end = 0;
  if (!checked_align_up(cursor, alignment, aligned) || !checked_mul(count, element_bytes, bytes) ||
      !checked_add(aligned, bytes, end)) {
    return false;
  }
  offset = aligned;
  cursor = end;
  return true;
}

[[nodiscard]] bool make_metadata_layout(const std::uint64_t blocks,
                                        MetadataLayout& layout) noexcept {
  std::uint64_t cursor = 0;
  if (!append_region(cursor, blocks, sizeof(void*), alignof(void*), layout.input_ptrs) ||
      !append_region(cursor, blocks, sizeof(std::size_t), alignof(std::size_t),
                     layout.input_sizes) ||
      !append_region(cursor, blocks, sizeof(void*), alignof(void*), layout.output_ptrs) ||
      !append_region(cursor, blocks, sizeof(std::size_t), alignof(std::size_t),
                     layout.output_sizes) ||
      !append_region(cursor, blocks, sizeof(std::size_t), alignof(std::size_t),
                     layout.actual_sizes) ||
      !append_region(cursor, blocks, sizeof(nvcompStatus_t), alignof(nvcompStatus_t),
                     layout.statuses) ||
      !append_region(cursor, 1, sizeof(ContentToken), alignof(ContentToken),
                     layout.verification_token) ||
      !append_region(cursor, 1, sizeof(std::uint32_t), alignof(std::uint32_t),
                     layout.verification_match) ||
      !checked_align_up(cursor, 64, layout.total_bytes)) {
    return false;
  }
  return layout.total_bytes != 0;
}

[[nodiscard]] std::uint64_t mix64(std::uint64_t value) noexcept {
  value ^= value >> 30U;
  value *= 0xBF58476D1CE4E5B9ULL;
  value ^= value >> 27U;
  value *= 0x94D049BB133111EBULL;
  value ^= value >> 31U;
  return value;
}

[[nodiscard]] ContentToken initial_content_token(const std::uint64_t valid_bytes) noexcept {
  return {mix64(0x585652414D484947ULL ^ valid_bytes), mix64(0x585652414D4C4F57ULL + valid_bytes)};
}

// One thread hashes one 64 KiB backing block in little-endian 64-bit words. The two global XOR
// reductions reproduce compute_content_token() exactly; the output is initialized with the
// valid-size seed first. Word hashing keeps verification bounded without serializing 65,536 byte
// recurrences in every thread.
inline constexpr char verification_ptx[] = R"ptx(
.version 8.0
.target sm_52
.address_size 64

.visible .entry xvram_lz4_verify_v1(
    .param .u64 data_base,
    .param .u64 valid_bytes,
    .param .u64 token_out)
{
    .reg .pred %p<4>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<40>;

    ld.param.u64 %rd0, [data_base];
    ld.param.u64 %rd1, [valid_bytes];
    ld.param.u64 %rd2, [token_out];
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.s32 %r3, %r0, %r1, %r2;
    cvt.u64.u32 %rd3, %r3;
    add.u64 %rd4, %rd1, 65535;
    shr.u64 %rd4, %rd4, 16;
    setp.ge.u64 %p0, %rd3, %rd4;
    @%p0 bra VERIFY_DONE;

    shl.b64 %rd5, %rd3, 16;
    sub.u64 %rd6, %rd1, %rd5;
    min.u64 %rd6, %rd6, 65536;
    add.u64 %rd7, %rd0, %rd5;
    mov.u64 %rd8, 0x585652414D4C4F57;
    mov.u64 %rd9, 0x585652414D484947;
    mov.u64 %rd10, 0;

VERIFY_WORDS:
    sub.u64 %rd24, %rd6, %rd10;
    setp.lt.u64 %p1, %rd24, 8;
    @%p1 bra VERIFY_TAIL_BEGIN;
    add.u64 %rd11, %rd7, %rd10;
    ld.global.u64 %rd12, [%rd11];
    add.u64 %rd13, %rd12, 1;
    mul.lo.u64 %rd8, %rd8, 0x00000100000001B3;
    add.u64 %rd8, %rd8, %rd13;
    xor.b64 %rd14, %rd12, 0xA5A5A5A5A5A5A5A5;
    add.u64 %rd14, %rd14, 1;
    mul.lo.u64 %rd9, %rd9, 0x9E3779B185EBCA87;
    add.u64 %rd9, %rd9, %rd14;
    add.u64 %rd10, %rd10, 8;
    bra VERIFY_WORDS;

VERIFY_TAIL_BEGIN:
    setp.eq.u64 %p2, %rd24, 0;
    @%p2 bra VERIFY_MIX;
    mov.u64 %rd12, 0;
    mov.u64 %rd25, 1;

VERIFY_TAIL_BYTES:
    setp.ge.u64 %p3, %rd10, %rd6;
    @%p3 bra VERIFY_TAIL_HASH;
    add.u64 %rd11, %rd7, %rd10;
    ld.global.u8 %r4, [%rd11];
    cvt.u64.u32 %rd13, %r4;
    mul.lo.u64 %rd13, %rd13, %rd25;
    add.u64 %rd12, %rd12, %rd13;
    shl.b64 %rd25, %rd25, 8;
    add.u64 %rd10, %rd10, 1;
    bra VERIFY_TAIL_BYTES;

VERIFY_TAIL_HASH:
    add.u64 %rd13, %rd12, 1;
    mul.lo.u64 %rd8, %rd8, 0x00000100000001B3;
    add.u64 %rd8, %rd8, %rd13;
    xor.b64 %rd14, %rd12, 0xA5A5A5A5A5A5A5A5;
    add.u64 %rd14, %rd14, 1;
    mul.lo.u64 %rd9, %rd9, 0x9E3779B185EBCA87;
    add.u64 %rd9, %rd9, %rd14;

VERIFY_MIX:
    shl.b64 %rd15, %rd6, 32;
    xor.b64 %rd15, %rd15, %rd3;
    shr.u64 %rd16, %rd15, 30;
    xor.b64 %rd15, %rd15, %rd16;
    mul.lo.u64 %rd15, %rd15, 0xBF58476D1CE4E5B9;
    shr.u64 %rd16, %rd15, 27;
    xor.b64 %rd15, %rd15, %rd16;
    mul.lo.u64 %rd15, %rd15, 0x94D049BB133111EB;
    shr.u64 %rd16, %rd15, 31;
    xor.b64 %rd15, %rd15, %rd16;

    xor.b64 %rd17, %rd9, %rd15;
    shr.u64 %rd18, %rd17, 30;
    xor.b64 %rd17, %rd17, %rd18;
    mul.lo.u64 %rd17, %rd17, 0xBF58476D1CE4E5B9;
    shr.u64 %rd18, %rd17, 27;
    xor.b64 %rd17, %rd17, %rd18;
    mul.lo.u64 %rd17, %rd17, 0x94D049BB133111EB;
    shr.u64 %rd18, %rd17, 31;
    xor.b64 %rd17, %rd17, %rd18;

    add.u64 %rd19, %rd8, %rd15;
    shr.u64 %rd20, %rd19, 30;
    xor.b64 %rd19, %rd19, %rd20;
    mul.lo.u64 %rd19, %rd19, 0xBF58476D1CE4E5B9;
    shr.u64 %rd20, %rd19, 27;
    xor.b64 %rd19, %rd19, %rd20;
    mul.lo.u64 %rd19, %rd19, 0x94D049BB133111EB;
    shr.u64 %rd20, %rd19, 31;
    xor.b64 %rd19, %rd19, %rd20;

    atom.global.xor.b64 %rd21, [%rd2], %rd17;
    add.u64 %rd22, %rd2, 8;
    atom.global.xor.b64 %rd23, [%rd22], %rd19;

VERIFY_DONE:
    ret;
}
)ptx";

template <typename Value>
[[nodiscard]] Value* host_region(void* const base, const std::uint64_t offset) noexcept {
  return reinterpret_cast<Value*>(static_cast<std::byte*>(base) + offset);
}

template <typename Value>
[[nodiscard]] Value* device_region(const DevicePointer base, const std::uint64_t offset) noexcept {
  return reinterpret_cast<Value*>(
      static_cast<std::uintptr_t>(base + static_cast<DevicePointer>(offset)));
}

[[nodiscard]] bool validate_container(const Lz4BlocksV1& source,
                                      const std::uint64_t chunk_bytes) noexcept {
  if (source.format_version != lz4_blocks_format_version || source.valid_bytes == 0 ||
      source.valid_bytes > chunk_bytes) {
    return false;
  }
  const std::uint64_t expected_blocks =
      (source.valid_bytes + nvcomp_lz4_block_bytes - 1) / nvcomp_lz4_block_bytes;
  if (expected_blocks != source.blocks.size()) {
    return false;
  }
  std::uint64_t remaining = source.valid_bytes;
  for (const auto& block : source.blocks) {
    const std::uint64_t expected = std::min(remaining, nvcomp_lz4_block_bytes);
    if (block.uncompressed_bytes != expected) {
      return false;
    }
    switch (block.storage) {
    case BlockStorage::implicit_zero:
      if (!block.payload.empty()) {
        return false;
      }
      break;
    case BlockStorage::raw:
      if (block.payload.size() != expected) {
        return false;
      }
      break;
    case BlockStorage::lz4:
      if (block.payload.empty() || block.payload.size() >= expected) {
        return false;
      }
      break;
    }
    remaining -= expected;
  }
  return remaining == 0;
}

} // namespace

const char* nvcomp_pipeline_status_name(const NvcompPipelineStatus status) noexcept {
  switch (status) {
  case NvcompPipelineStatus::success:
    return "success";
  case NvcompPipelineStatus::not_ready:
    return "not_ready";
  case NvcompPipelineStatus::busy:
    return "busy";
  case NvcompPipelineStatus::invalid_configuration:
    return "invalid_configuration";
  case NvcompPipelineStatus::invalid_argument:
    return "invalid_argument";
  case NvcompPipelineStatus::stale_ticket:
    return "stale_ticket";
  case NvcompPipelineStatus::resource_exhausted:
    return "resource_exhausted";
  case NvcompPipelineStatus::cuda_failure:
    return "cuda_failure";
  case NvcompPipelineStatus::codec_failure:
    return "codec_failure";
  case NvcompPipelineStatus::corrupt_data:
    return "corrupt_data";
  case NvcompPipelineStatus::timeout:
    return "timeout";
  case NvcompPipelineStatus::quarantined:
    return "quarantined";
  case NvcompPipelineStatus::cleanup_failure:
    return "cleanup_failure";
  case NvcompPipelineStatus::closed:
    return "closed";
  }
  return "unknown";
}

class NvcompLz4Pipeline::Impl {
public:
  enum class Phase { idle, decode_event, encode_codec_event, encode_transfer_event, encode_ready };

  struct Slot {
    Stream stream = nullptr;
    Event event = nullptr;
    DevicePointer device_payload = 0;
    DevicePointer device_metadata = 0;
    DevicePointer workspace = 0;
    void* host_payload = nullptr;
    void* host_metadata = nullptr;

    bool active = false;
    bool submitted = false;
    bool codec_submitted = false;
    bool accounted = false;
    std::uint64_t generation = 0;
    Phase phase = Phase::idle;
    NvcompPipelineTicket ticket;
    ChunkKey trace_key;
    CompressionPath trace_path = CompressionPath::nvcomp_gpu_codec;
    bool speculative = false;
    bool h2d_trace_submitted = false;
    bool d2h_trace_submitted = false;
    std::uint64_t h2d_physical_bytes = 0;
    std::uint64_t d2h_physical_bytes = 0;
    std::uint64_t encoded_physical_bytes = 0;
    std::uint64_t valid_bytes = 0;
    std::size_t batch_count = 0;
    std::optional<ContentToken> expected_token;
    std::vector<std::uint32_t> batch_blocks;
    std::vector<std::uint64_t> transfer_offsets;
    std::vector<BlockStorage> transfer_storage;
    std::optional<Lz4BlocksV1> completed_candidate;
    bool candidate_delivered = false;
    std::chrono::steady_clock::time_point verification_started{};
  };

  Impl(cuda::CudaApi& cuda_api, nvcomp::NvcompApi& nvcomp_api, const BlockCodec& cpu_lz4_codec,
       NvcompPipelineConfig config)
      : cuda_(cuda_api), nvcomp_(nvcomp_api), cpu_codec_(cpu_lz4_codec),
        config_(std::move(config)) {}

  [[nodiscard]] NvcompPipelineStatus setup() noexcept {
    if (setup_ || closed_ || cleanup_pending_) {
      return fail(NvcompPipelineStatus::invalid_configuration,
                  "pipeline is already set up, closed, or awaiting cleanup retry");
    }
    if (config_.slot_count < 2 || config_.slot_count > 8 || config_.chunk_bytes == 0 ||
        config_.chunk_bytes % nvcomp_lz4_block_bytes != 0 || config_.wait_timeout.count() <= 0 ||
        !fits_size(config_.chunk_bytes) || cpu_codec_.codec() != CompressionCodec::lz4 ||
        !nvcomp_.has_lz4() || !cuda_dispatch_complete()) {
      return fail(NvcompPipelineStatus::invalid_configuration,
                  "invalid slot/chunk/codec configuration or incomplete CUDA/nvCOMP dispatch");
    }

    block_count_ = config_.chunk_bytes / nvcomp_lz4_block_bytes;
    if (block_count_ == 0 || block_count_ > std::numeric_limits<std::uint32_t>::max() ||
        !make_metadata_layout(block_count_, metadata_)) {
      return fail(NvcompPipelineStatus::invalid_configuration, "codec metadata layout overflowed");
    }

    const auto& dispatch = nvcomp_.dispatch();
    compress_options_ = nvcompBatchedLZ4CompressDefaultOpts;
    decompress_options_ = nvcompBatchedLZ4DecompressDefaultOpts;
    // RTX 3070-class devices must never select the Blackwell-only hardware DE.
    decompress_options_.backend = NVCOMP_DECOMPRESS_BACKEND_CUDA;
    decompress_options_.sort_before_hw_decompress = 0;

    if (dispatch.lz4_compress_get_required_alignments(compress_options_, &compress_alignment_) !=
            nvcompSuccess ||
        dispatch.lz4_decompress_get_required_alignments(decompress_options_,
                                                        &decompress_alignment_) != nvcompSuccess ||
        !valid_alignment(compress_alignment_) || !valid_alignment(decompress_alignment_) ||
        nvcomp_lz4_block_bytes % compress_alignment_.input != 0 ||
        nvcomp_lz4_block_bytes % decompress_alignment_.output != 0) {
      return fail(NvcompPipelineStatus::codec_failure,
                  "nvCOMP returned unusable LZ4 alignment requirements");
    }

    std::size_t compress_temp = 0;
    std::size_t decompress_temp = 0;
    if (dispatch.lz4_compress_get_temp_size(
            static_cast<std::size_t>(block_count_),
            static_cast<std::size_t>(nvcomp_lz4_block_bytes), compress_options_, &compress_temp,
            static_cast<std::size_t>(config_.chunk_bytes)) != nvcompSuccess ||
        dispatch.lz4_decompress_get_temp_size(
            static_cast<std::size_t>(block_count_),
            static_cast<std::size_t>(nvcomp_lz4_block_bytes), decompress_options_, &decompress_temp,
            static_cast<std::size_t>(config_.chunk_bytes)) != nvcompSuccess ||
        dispatch.lz4_compress_get_max_output_size(static_cast<std::size_t>(nvcomp_lz4_block_bytes),
                                                  compress_options_,
                                                  &max_compressed_block_bytes_) != nvcompSuccess ||
        max_compressed_block_bytes_ == 0) {
      return fail(NvcompPipelineStatus::codec_failure,
                  "nvCOMP failed to size the LZ4 codec resources");
    }
    workspace_bytes_per_slot_ = std::max(compress_temp, decompress_temp);

    std::uint64_t output_stride = 0;
    std::uint64_t encode_payload = 0;
    std::uint64_t decode_stride = 0;
    std::uint64_t decode_payload = 0;
    if (!checked_align_up(max_compressed_block_bytes_, compress_alignment_.output, output_stride) ||
        !checked_mul(output_stride, block_count_, encode_payload) ||
        !checked_align_up(nvcomp_lz4_block_bytes, decompress_alignment_.input, decode_stride) ||
        !checked_mul(decode_stride, block_count_, decode_payload)) {
      return fail(NvcompPipelineStatus::invalid_configuration, "codec payload layout overflowed");
    }
    output_stride_ = output_stride;
    payload_bytes_per_slot_ = std::max({encode_payload, decode_payload, config_.chunk_bytes});
    if (!fits_size(payload_bytes_per_slot_) || !fits_size(metadata_.total_bytes) ||
        !fits_size(workspace_bytes_per_slot_)) {
      return fail(NvcompPipelineStatus::invalid_configuration,
                  "codec resource size exceeds the platform size limit");
    }

    std::uint64_t total_workspace = 0;
    std::uint64_t slot_capacity = 0;
    std::uint64_t total_slot_capacity = 0;
    if (!checked_mul(workspace_bytes_per_slot_, config_.slot_count, total_workspace) ||
        !checked_add(payload_bytes_per_slot_, metadata_.total_bytes, slot_capacity) ||
        !checked_mul(slot_capacity, config_.slot_count, total_slot_capacity) ||
        (config_.workspace_cap_bytes != 0 && total_workspace > config_.workspace_cap_bytes)) {
      return fail(NvcompPipelineStatus::resource_exhausted,
                  "nvCOMP workspace/slot capacity overflowed or exceeds the configured cap");
    }
    telemetry_.device_slot_capacity_bytes = total_slot_capacity;

    try {
      slots_.resize(config_.slot_count);
      for (std::uint32_t index = 0; index < config_.slot_count; ++index) {
        const NvcompPipelineStatus status = create_slot(slots_[index]);
        if (status != NvcompPipelineStatus::success) {
          if (!cleanup_setup_resources()) {
            return fail(NvcompPipelineStatus::cleanup_failure,
                        "codec slot setup failed and owned resources could not be released");
          }
          return status;
        }
        saturating_add(telemetry_.slots_created, 1);
      }
    } catch (const std::bad_alloc&) {
      if (!cleanup_setup_resources()) {
        return fail(NvcompPipelineStatus::cleanup_failure,
                    "codec slot allocation failed and owned resources could not be released");
      }
      return fail(NvcompPipelineStatus::resource_exhausted,
                  "host allocation failed while creating codec slots");
    } catch (...) {
      if (!cleanup_setup_resources()) {
        return fail(NvcompPipelineStatus::cleanup_failure,
                    "codec slot setup raised and owned resources could not be released");
      }
      return fail(NvcompPipelineStatus::invalid_configuration,
                  "unexpected exception while creating codec slots");
    }

    if (config_.verification_submit == nullptr) {
      Result code = cuda_.module_load_data_(&verification_module_, verification_ptx);
      if (code == CUDA_SUCCESS) {
        code = cuda_.module_get_function_(&verification_function_, verification_module_,
                                          "xvram_lz4_verify_v1");
      }
      if (code != CUDA_SUCCESS) {
        verification_function_ = nullptr;
        if (!cleanup_setup_resources()) {
          return fail(NvcompPipelineStatus::cleanup_failure,
                      "verification module setup failed and owned resources could not be released");
        }
        return cuda_failure("embedded decode verification module setup", code);
      }
    }

    setup_ = true;
    return NvcompPipelineStatus::success;
  }

  [[nodiscard]] NvcompPipelineStatus decode(const NvcompDecodeRequest& request,
                                            NvcompPipelineTicket& output) noexcept {
    output = {};
    if (const auto state = operational_status(); state != NvcompPipelineStatus::success) {
      return state;
    }
    if (request.source == nullptr || request.stable_output == 0 ||
        !validate_container(*request.source, config_.chunk_bytes) ||
        request.stable_output >
            std::numeric_limits<DevicePointer>::max() - request.source->valid_bytes ||
        request.stable_output % decompress_alignment_.output != 0) {
      return fail(NvcompPipelineStatus::invalid_argument,
                  "decode request, container, stable address, or verifier is invalid");
    }

    Slot* slot = acquire_slot(NvcompOperationKind::decode, request.source->generation);
    if (slot == nullptr) {
      return slots_exhausted_ ? NvcompPipelineStatus::resource_exhausted
                              : NvcompPipelineStatus::busy;
    }
    slot->valid_bytes = request.source->valid_bytes;
    slot->expected_token = request.source->content_token;
    slot->trace_key = request.key;
    slot->trace_path = request.path;
    slot->speculative = request.speculative;
    slot->phase = Phase::decode_event;

    try {
      slot->batch_blocks.clear();
      slot->batch_blocks.reserve(request.source->blocks.size());
      std::memset(slot->host_metadata, 0, static_cast<std::size_t>(metadata_.total_bytes));

      auto* input_ptrs = host_region<void*>(slot->host_metadata, metadata_.input_ptrs);
      auto* input_sizes = host_region<std::size_t>(slot->host_metadata, metadata_.input_sizes);
      auto* output_ptrs = host_region<void*>(slot->host_metadata, metadata_.output_ptrs);
      auto* output_sizes = host_region<std::size_t>(slot->host_metadata, metadata_.output_sizes);
      auto* verification_match =
          host_region<std::uint32_t>(slot->host_metadata, metadata_.verification_match);
      auto* verification_token =
          host_region<ContentToken>(slot->host_metadata, metadata_.verification_token);
      *verification_match = 0;
      *verification_token = initial_content_token(request.source->valid_bytes);

      // Keep compressed payloads densely packed at the start of the pinned/device slot so the
      // complete nvCOMP batch needs a single H2D submission. Raw/implicit-zero blocks use unique
      // staging ranges growing down from the end of the same pinned slot and are copied directly
      // to their final stable addresses. Besides reducing driver submission overhead, the split
      // prevents raw staging bytes from being counted or transferred as compressed payload.
      std::uint64_t compressed_offset = 0;
      std::uint64_t raw_staging_offset = payload_bytes_per_slot_;
      std::uint64_t logical_offset = 0;
      std::size_t batch = 0;
      for (std::size_t block_index = 0; block_index < request.source->blocks.size();
           ++block_index) {
        const auto& block = request.source->blocks[block_index];
        const std::uint64_t block_bytes = block.uncompressed_bytes;
        const DevicePointer output_address = request.stable_output + logical_offset;
        if (block.storage == BlockStorage::lz4) {
          if (!checked_align_up(compressed_offset, decompress_alignment_.input,
                                compressed_offset) ||
              block.payload.size() > raw_staging_offset ||
              compressed_offset > raw_staging_offset - block.payload.size()) {
            return abort_before_event(*slot, NvcompPipelineStatus::invalid_argument,
                                      "compressed decode payload exceeds its codec slot");
          }
          std::memcpy(static_cast<std::byte*>(slot->host_payload) + compressed_offset,
                      block.payload.data(), block.payload.size());
          input_ptrs[batch] = device_void(slot->device_payload + compressed_offset);
          input_sizes[batch] = block.payload.size();
          output_ptrs[batch] = device_void(output_address);
          output_sizes[batch] = static_cast<std::size_t>(block_bytes);
          slot->batch_blocks.push_back(static_cast<std::uint32_t>(block_index));
          compressed_offset += block.payload.size();
          ++batch;
          saturating_add(telemetry_.compressed_h2d_blocks, 1);
        } else {
          if (block_bytes > raw_staging_offset ||
              compressed_offset > raw_staging_offset - block_bytes) {
            return abort_before_event(*slot, NvcompPipelineStatus::invalid_argument,
                                      "raw decode payload exceeds its codec slot");
          }
          raw_staging_offset -= block_bytes;
          std::byte* pinned = static_cast<std::byte*>(slot->host_payload) + raw_staging_offset;
          if (block.storage == BlockStorage::raw) {
            std::memcpy(pinned, block.payload.data(), static_cast<std::size_t>(block_bytes));
            saturating_add(telemetry_.raw_h2d_blocks, 1);
          } else {
            std::memset(pinned, 0, static_cast<std::size_t>(block_bytes));
            saturating_add(telemetry_.zero_h2d_blocks, 1);
          }
          if (!copy_h2d(*slot, output_address, pinned, block_bytes, true)) {
            return recover_copy_failure(*slot, "raw/zero H2D submission failed");
          }
        }
        logical_offset += block_bytes;
      }
      slot->batch_count = batch;

      if (batch != 0) {
        if (compressed_offset != 0 &&
            !copy_h2d(*slot, slot->device_payload, slot->host_payload, compressed_offset, true)) {
          return recover_copy_failure(*slot, "compressed H2D submission failed");
        }
        if (!submit_decode_metadata(*slot, batch)) {
          return recover_copy_failure(*slot, "decode metadata H2D submission failed");
        }
      }
      const std::uint64_t verification_offset = config_.verification_submit != nullptr
                                                    ? metadata_.verification_match
                                                    : metadata_.verification_token;
      const std::uint64_t verification_bytes = config_.verification_submit != nullptr
                                                   ? sizeof(*verification_match)
                                                   : sizeof(*verification_token);
      const void* verification_host = config_.verification_submit != nullptr
                                          ? static_cast<const void*>(verification_match)
                                          : static_cast<const void*>(verification_token);
      if (!copy_h2d(*slot, slot->device_metadata + verification_offset, verification_host,
                    verification_bytes, false)) {
        return recover_copy_failure(*slot, "verification initialization failed");
      }
      emit_trace(*slot, CompressionTraceEventKind::h2d_submit, "compressed_host_payload_enqueued");
      slot->h2d_trace_submitted = true;
      if (batch != 0) {
        const nvcompStatus_t codec_status = nvcomp_.dispatch().lz4_decompress_async(
            device_region<const void*>(slot->device_metadata, metadata_.input_ptrs),
            device_region<const std::size_t>(slot->device_metadata, metadata_.input_sizes),
            device_region<const std::size_t>(slot->device_metadata, metadata_.output_sizes),
            device_region<std::size_t>(slot->device_metadata, metadata_.actual_sizes), batch,
            device_void(slot->workspace), static_cast<std::size_t>(workspace_bytes_per_slot_),
            device_region<void*>(slot->device_metadata, metadata_.output_ptrs), decompress_options_,
            device_region<nvcompStatus_t>(slot->device_metadata, metadata_.statuses),
            runtime_stream(slot->stream));
        if (codec_status != nvcompSuccess) {
          saturating_add(telemetry_.codec_failures, 1);
          return recover_codec_launch_failure(*slot, codec_status,
                                              "nvCOMP LZ4 decode launch failed");
        }
        slot->codec_submitted = true;
        slot->submitted = true;
        saturating_add(telemetry_.decode_batches, 1);
        emit_trace(*slot, CompressionTraceEventKind::decode_submit, "nvcomp_decode_enqueued");
      }

      const Result verify_status = submit_verification(
          *slot, request.stable_output, request.source->valid_bytes, request.source->content_token);
      if (verify_status != CUDA_SUCCESS) {
        return quarantine_after_submission(*slot, "decode verification submission failed",
                                           verify_status);
      }
      slot->submitted = true;
      saturating_add(telemetry_.verification_submissions, 1);
      slot->verification_started = std::chrono::steady_clock::now();

      if (batch != 0 &&
          (!copy_d2h(*slot, host_region<std::size_t>(slot->host_metadata, metadata_.actual_sizes),
                     slot->device_metadata + metadata_.actual_sizes, batch * sizeof(std::size_t),
                     false) ||
           !copy_d2h(*slot, host_region<nvcompStatus_t>(slot->host_metadata, metadata_.statuses),
                     slot->device_metadata + metadata_.statuses, batch * sizeof(nvcompStatus_t),
                     false))) {
        return quarantine_after_submission(*slot, "decode result metadata D2H failed",
                                           std::nullopt);
      }
      void* verification_output = config_.verification_submit != nullptr
                                      ? static_cast<void*>(verification_match)
                                      : static_cast<void*>(verification_token);
      if (!copy_d2h(*slot, verification_output, slot->device_metadata + verification_offset,
                    verification_bytes, false) ||
          !record_event(*slot)) {
        return quarantine_after_submission(*slot, "decode completion event could not be recorded",
                                           std::nullopt);
      }

      saturating_add(telemetry_.logical_decode_bytes, request.source->valid_bytes);
      saturating_add(telemetry_.blocks_decoded, request.source->blocks.size());
      saturating_add(telemetry_.operations_submitted, 1);
      output = slot->ticket;
      return NvcompPipelineStatus::success;
    } catch (const std::bad_alloc&) {
      return abort_or_quarantine(*slot, NvcompPipelineStatus::resource_exhausted,
                                 "host allocation failed while preparing decode");
    } catch (...) {
      return abort_or_quarantine(*slot, NvcompPipelineStatus::invalid_argument,
                                 "unexpected exception while preparing decode");
    }
  }

  [[nodiscard]] NvcompPipelineStatus encode(const NvcompEncodeRequest& request,
                                            NvcompPipelineTicket& output) noexcept {
    output = {};
    if (const auto state = operational_status(); state != NvcompPipelineStatus::success) {
      return state;
    }
    if (request.stable_input == 0 || request.valid_bytes == 0 ||
        request.valid_bytes > config_.chunk_bytes ||
        request.source_generation == std::numeric_limits<std::uint64_t>::max() ||
        request.stable_input > std::numeric_limits<DevicePointer>::max() - request.valid_bytes ||
        request.stable_input % compress_alignment_.input != 0) {
      return fail(NvcompPipelineStatus::invalid_argument,
                  "encode request or stable input address is invalid");
    }

    Slot* slot = acquire_slot(NvcompOperationKind::encode, request.source_generation);
    if (slot == nullptr) {
      return slots_exhausted_ ? NvcompPipelineStatus::resource_exhausted
                              : NvcompPipelineStatus::busy;
    }
    slot->valid_bytes = request.valid_bytes;
    slot->expected_token = request.expected_token;
    slot->trace_key = request.key;
    slot->trace_path = request.path;
    slot->speculative = request.speculative;
    slot->phase = Phase::encode_codec_event;

    try {
      const std::uint64_t blocks =
          (request.valid_bytes + nvcomp_lz4_block_bytes - 1) / nvcomp_lz4_block_bytes;
      slot->batch_count = static_cast<std::size_t>(blocks);
      std::memset(slot->host_metadata, 0, static_cast<std::size_t>(metadata_.total_bytes));
      auto* input_ptrs = host_region<void*>(slot->host_metadata, metadata_.input_ptrs);
      auto* input_sizes = host_region<std::size_t>(slot->host_metadata, metadata_.input_sizes);
      auto* output_ptrs = host_region<void*>(slot->host_metadata, metadata_.output_ptrs);
      auto* verification_match =
          host_region<std::uint32_t>(slot->host_metadata, metadata_.verification_match);
      auto* verification_token =
          host_region<ContentToken>(slot->host_metadata, metadata_.verification_token);
      *verification_match = 0;
      *verification_token = initial_content_token(request.valid_bytes);
      std::uint64_t remaining = request.valid_bytes;
      std::uint64_t logical_offset = 0;
      for (std::uint64_t index = 0; index < blocks; ++index) {
        const std::uint64_t bytes = std::min(remaining, nvcomp_lz4_block_bytes);
        const DevicePointer output_address =
            slot->device_payload + static_cast<DevicePointer>(index * output_stride_);
        if (output_address % compress_alignment_.output != 0) {
          return abort_before_event(*slot, NvcompPipelineStatus::invalid_argument,
                                    "nvCOMP output slot is misaligned");
        }
        input_ptrs[index] = device_void(request.stable_input + logical_offset);
        input_sizes[index] = static_cast<std::size_t>(bytes);
        output_ptrs[index] = device_void(output_address);
        remaining -= bytes;
        logical_offset += bytes;
      }
      if (!submit_encode_metadata(*slot, slot->batch_count)) {
        return recover_copy_failure(*slot, "encode metadata H2D submission failed");
      }
      if (config_.verification_submit != nullptr && !request.expected_token.has_value()) {
        return abort_or_quarantine(*slot, NvcompPipelineStatus::invalid_argument,
                                   "custom encode verifier requires an expected content token");
      }
      const std::uint64_t verification_offset = config_.verification_submit != nullptr
                                                    ? metadata_.verification_match
                                                    : metadata_.verification_token;
      const std::uint64_t verification_bytes = config_.verification_submit != nullptr
                                                   ? sizeof(*verification_match)
                                                   : sizeof(*verification_token);
      const void* verification_host = config_.verification_submit != nullptr
                                          ? static_cast<const void*>(verification_match)
                                          : static_cast<const void*>(verification_token);
      if (!copy_h2d(*slot, slot->device_metadata + verification_offset, verification_host,
                    verification_bytes, false)) {
        return recover_copy_failure(*slot, "encode source-token initialization failed");
      }
      const Result verify_status =
          submit_verification(*slot, request.stable_input, request.valid_bytes,
                              request.expected_token.value_or(ContentToken{}));
      if (verify_status != CUDA_SUCCESS) {
        return quarantine_after_submission(*slot, "encode source verification submission failed",
                                           verify_status);
      }
      slot->submitted = true;
      saturating_add(telemetry_.verification_submissions, 1);
      slot->verification_started = std::chrono::steady_clock::now();
      const nvcompStatus_t codec_status = nvcomp_.dispatch().lz4_compress_async(
          device_region<const void*>(slot->device_metadata, metadata_.input_ptrs),
          device_region<const std::size_t>(slot->device_metadata, metadata_.input_sizes),
          static_cast<std::size_t>(nvcomp_lz4_block_bytes), slot->batch_count,
          device_void(slot->workspace), static_cast<std::size_t>(workspace_bytes_per_slot_),
          device_region<void*>(slot->device_metadata, metadata_.output_ptrs),
          device_region<std::size_t>(slot->device_metadata, metadata_.actual_sizes),
          compress_options_,
          device_region<nvcompStatus_t>(slot->device_metadata, metadata_.statuses),
          runtime_stream(slot->stream));
      if (codec_status != nvcompSuccess) {
        saturating_add(telemetry_.codec_failures, 1);
        return recover_codec_launch_failure(*slot, codec_status, "nvCOMP LZ4 encode launch failed");
      }
      slot->codec_submitted = true;
      slot->submitted = true;
      saturating_add(telemetry_.encode_batches, 1);
      emit_trace(*slot, CompressionTraceEventKind::encode_submit, "nvcomp_encode_enqueued");
      if (!copy_d2h(*slot, host_region<std::size_t>(slot->host_metadata, metadata_.actual_sizes),
                    slot->device_metadata + metadata_.actual_sizes,
                    slot->batch_count * sizeof(std::size_t), false) ||
          !copy_d2h(*slot, host_region<nvcompStatus_t>(slot->host_metadata, metadata_.statuses),
                    slot->device_metadata + metadata_.statuses,
                    slot->batch_count * sizeof(nvcompStatus_t), false) ||
          !copy_d2h(*slot,
                    config_.verification_submit != nullptr ? static_cast<void*>(verification_match)
                                                           : static_cast<void*>(verification_token),
                    slot->device_metadata + verification_offset, verification_bytes, false) ||
          !record_event(*slot)) {
        return quarantine_after_submission(*slot, "encode completion event could not be recorded",
                                           std::nullopt);
      }
      saturating_add(telemetry_.logical_encode_bytes, request.valid_bytes);
      saturating_add(telemetry_.blocks_encoded, blocks);
      saturating_add(telemetry_.operations_submitted, 1);
      output = slot->ticket;
      return NvcompPipelineStatus::success;
    } catch (const std::bad_alloc&) {
      return abort_or_quarantine(*slot, NvcompPipelineStatus::resource_exhausted,
                                 "host allocation failed while preparing encode");
    } catch (...) {
      return abort_or_quarantine(*slot, NvcompPipelineStatus::invalid_argument,
                                 "unexpected exception while preparing encode");
    }
  }

  [[nodiscard]] NvcompPipelineStatus poll(const NvcompPipelineTicket& ticket,
                                          Lz4BlocksV1* candidate) noexcept {
    if (quarantined_) {
      return NvcompPipelineStatus::quarantined;
    }
    if (!setup_ || closed_) {
      return NvcompPipelineStatus::closed;
    }
    Slot* slot = ticket_slot(ticket);
    if (slot == nullptr) {
      return NvcompPipelineStatus::stale_ticket;
    }
    if (slot->phase == Phase::encode_ready) {
      if (candidate == nullptr || !slot->completed_candidate.has_value() ||
          slot->candidate_delivered) {
        return NvcompPipelineStatus::invalid_argument;
      }
      *candidate = std::move(*slot->completed_candidate);
      slot->completed_candidate.reset();
      slot->candidate_delivered = true;
      return NvcompPipelineStatus::success;
    }

    const Result query = cuda_.event_query_(slot->event);
    if (query == CUDA_ERROR_NOT_READY) {
      return NvcompPipelineStatus::not_ready;
    }
    if (query != CUDA_SUCCESS) {
      return quarantine_after_submission(*slot, "cuEventQuery returned an unsafe result", query);
    }
    saturating_add(telemetry_.events_retired, 1);

    try {
      switch (slot->phase) {
      case Phase::decode_event:
        return finish_decode(*slot);
      case Phase::encode_codec_event:
        return begin_encode_transfer(*slot);
      case Phase::encode_transfer_event:
        return finish_encode_transfer(*slot, candidate);
      case Phase::idle:
      case Phase::encode_ready:
        return NvcompPipelineStatus::stale_ticket;
      }
    } catch (const std::bad_alloc&) {
      terminal_failure(*slot);
      return fail(NvcompPipelineStatus::resource_exhausted,
                  "host allocation failed while retiring codec operation");
    } catch (...) {
      terminal_failure(*slot);
      return fail(NvcompPipelineStatus::codec_failure,
                  "unexpected exception while retiring codec operation");
    }
    return NvcompPipelineStatus::codec_failure;
  }

  [[nodiscard]] NvcompPipelineStatus
  acknowledge_host_commit(const NvcompPipelineTicket& ticket) noexcept {
    if (const auto state = operational_status(); state != NvcompPipelineStatus::success) {
      return state;
    }
    Slot* slot = ticket_slot(ticket);
    if (slot == nullptr) {
      return NvcompPipelineStatus::stale_ticket;
    }
    if (ticket.kind != NvcompOperationKind::encode || slot->phase != Phase::encode_ready ||
        !slot->candidate_delivered) {
      return fail(NvcompPipelineStatus::invalid_argument,
                  "encode slot cannot retire before candidate delivery and authoritative commit");
    }
    if (slot->d2h_trace_submitted) {
      emit_trace(*slot, CompressionTraceEventKind::d2h_retire,
                 "authoritative_host_commit_acknowledged");
    }
    retire(*slot);
    return NvcompPipelineStatus::success;
  }

  [[nodiscard]] NvcompPipelineStatus
  quarantine_uncommitted(const NvcompPipelineTicket& ticket) noexcept {
    if (quarantined_) {
      return NvcompPipelineStatus::quarantined;
    }
    Slot* slot = ticket_slot(ticket);
    if (slot == nullptr) {
      return NvcompPipelineStatus::stale_ticket;
    }
    if (ticket.kind != NvcompOperationKind::encode || slot->phase != Phase::encode_ready) {
      return fail(NvcompPipelineStatus::invalid_argument,
                  "only a completed encode candidate can be quarantined as uncommitted");
    }
    return quarantine_after_submission(
        *slot, "completed GPU encode could not be committed to host backing", std::nullopt);
  }

  [[nodiscard]] NvcompPipelineStatus wait(const NvcompPipelineTicket& ticket,
                                          Lz4BlocksV1* candidate) noexcept {
    const auto deadline = std::chrono::steady_clock::now() + config_.wait_timeout;
    for (;;) {
      const NvcompPipelineStatus status = poll(ticket, candidate);
      if (status != NvcompPipelineStatus::not_ready) {
        return status;
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        Slot* slot = ticket_slot(ticket);
        if (slot == nullptr) {
          return NvcompPipelineStatus::stale_ticket;
        }
        return quarantine_after_submission(*slot, "codec event exceeded the polling deadline",
                                           std::nullopt);
      }
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  }

  [[nodiscard]] NvcompPipelineStatus close() noexcept {
    if (closed_) {
      return quarantined_ ? NvcompPipelineStatus::quarantined : NvcompPipelineStatus::success;
    }
    if (quarantined_) {
      closed_ = true;
      setup_ = false;
      return NvcompPipelineStatus::quarantined;
    }

    for (auto& slot : slots_) {
      while (slot.active) {
        if (slot.phase == Phase::encode_ready) {
          (void)quarantine_after_submission(
              slot, "close encountered an encode candidate without authoritative host commit",
              std::nullopt);
          closed_ = true;
          setup_ = false;
          return NvcompPipelineStatus::quarantined;
        }
        const Result synchronized = cuda_.event_synchronize_(slot.event);
        if (synchronized != CUDA_SUCCESS) {
          (void)quarantine_after_submission(slot, "cuEventSynchronize failed during close",
                                            synchronized);
          closed_ = true;
          setup_ = false;
          return NvcompPipelineStatus::quarantined;
        }
        Lz4BlocksV1 ignored;
        const NvcompPipelineStatus status = poll(slot.ticket, &ignored);
        if (status != NvcompPipelineStatus::success && status != NvcompPipelineStatus::not_ready) {
          if (status == NvcompPipelineStatus::quarantined) {
            closed_ = true;
            setup_ = false;
            return status;
          }
          // A completed corruption/codec failure is event-safe; continue complete cleanup.
          break;
        }
      }
    }

    const bool cleanup_ok = cleanup_setup_resources();
    setup_ = false;
    if (!cleanup_ok) {
      return fail(NvcompPipelineStatus::cleanup_failure,
                  "pipeline cleanup is incomplete and must be retried");
    }
    closed_ = true;
    return NvcompPipelineStatus::success;
  }

  [[nodiscard]] const NvcompPipelineConfig& config() const noexcept {
    return config_;
  }
  [[nodiscard]] const NvcompPipelineTelemetry& telemetry() const noexcept {
    return telemetry_;
  }
  [[nodiscard]] const std::string& error() const noexcept {
    return error_;
  }
  [[nodiscard]] std::optional<std::int64_t> native_error() const noexcept {
    return native_error_;
  }
  [[nodiscard]] bool quarantined() const noexcept {
    return quarantined_;
  }
  [[nodiscard]] bool is_setup() const noexcept {
    return setup_ && !closed_;
  }

private:
  [[nodiscard]] bool cuda_dispatch_complete() const noexcept {
    return cuda_.mem_alloc_ != nullptr && cuda_.mem_free_ != nullptr &&
           cuda_.mem_host_alloc_ != nullptr && cuda_.mem_free_host_ != nullptr &&
           cuda_.memcpy_h2d_async_ != nullptr && cuda_.memcpy_d2h_async_ != nullptr &&
           cuda_.stream_create_ != nullptr && cuda_.stream_destroy_ != nullptr &&
           cuda_.stream_synchronize_ != nullptr && cuda_.event_create_ != nullptr &&
           cuda_.event_destroy_ != nullptr && cuda_.event_record_ != nullptr &&
           cuda_.event_query_ != nullptr && cuda_.event_synchronize_ != nullptr &&
           (config_.verification_submit != nullptr ||
            (cuda_.module_load_data_ != nullptr && cuda_.module_get_function_ != nullptr &&
             cuda_.module_unload_ != nullptr && cuda_.launch_kernel_ != nullptr));
  }

  [[nodiscard]] static bool
  valid_alignment(const nvcompAlignmentRequirements_t& alignment) noexcept {
    return alignment.input != 0 && alignment.output != 0 && alignment.temp != 0;
  }

  [[nodiscard]] NvcompPipelineStatus create_slot(Slot& slot) noexcept {
    Result code = cuda_.stream_create_(&slot.stream, CU_STREAM_NON_BLOCKING);
    if (code != CUDA_SUCCESS) {
      return cuda_failure("cuStreamCreate(codec)", code);
    }
    code = cuda_.event_create_(&slot.event, CU_EVENT_DISABLE_TIMING);
    if (code != CUDA_SUCCESS) {
      return cuda_failure("cuEventCreate(codec)", code);
    }
    code =
        cuda_.mem_alloc_(&slot.device_payload, static_cast<std::size_t>(payload_bytes_per_slot_));
    if (code != CUDA_SUCCESS) {
      return cuda_failure("cuMemAlloc(codec payload)", code);
    }
    code = cuda_.mem_alloc_(&slot.device_metadata, static_cast<std::size_t>(metadata_.total_bytes));
    if (code != CUDA_SUCCESS) {
      return cuda_failure("cuMemAlloc(codec metadata)", code);
    }
    if (workspace_bytes_per_slot_ != 0) {
      code = cuda_.mem_alloc_(&slot.workspace, static_cast<std::size_t>(workspace_bytes_per_slot_));
      if (code != CUDA_SUCCESS) {
        return cuda_failure("cuMemAlloc(codec workspace)", code);
      }
    }
    code =
        cuda_.mem_host_alloc_(&slot.host_payload, static_cast<std::size_t>(payload_bytes_per_slot_),
                              CU_MEMHOSTALLOC_PORTABLE);
    if (code != CUDA_SUCCESS) {
      return cuda_failure("cuMemHostAlloc(codec payload)", code);
    }
    code =
        cuda_.mem_host_alloc_(&slot.host_metadata, static_cast<std::size_t>(metadata_.total_bytes),
                              CU_MEMHOSTALLOC_PORTABLE);
    if (code != CUDA_SUCCESS) {
      return cuda_failure("cuMemHostAlloc(codec metadata)", code);
    }
    if (slot.device_payload % compress_alignment_.output != 0 ||
        slot.device_payload % decompress_alignment_.input != 0 ||
        (slot.workspace != 0 && (slot.workspace % compress_alignment_.temp != 0 ||
                                 slot.workspace % decompress_alignment_.temp != 0))) {
      return fail(NvcompPipelineStatus::invalid_configuration,
                  "CUDA codec allocation does not satisfy nvCOMP alignment");
    }

    std::uint64_t device_bytes = 0;
    if (!checked_add(payload_bytes_per_slot_, metadata_.total_bytes, device_bytes) ||
        !checked_add(device_bytes, workspace_bytes_per_slot_, device_bytes)) {
      return fail(NvcompPipelineStatus::invalid_configuration,
                  "codec slot byte accounting overflowed");
    }
    saturating_add(telemetry_.workspace_bytes, workspace_bytes_per_slot_);
    telemetry_.workspace_peak_bytes =
        std::max(telemetry_.workspace_peak_bytes, telemetry_.workspace_bytes);
    saturating_add(telemetry_.device_slot_bytes, device_bytes);
    telemetry_.device_slot_peak_bytes =
        std::max(telemetry_.device_slot_peak_bytes, telemetry_.device_slot_bytes);
    std::uint64_t pinned = 0;
    if (!checked_add(payload_bytes_per_slot_, metadata_.total_bytes, pinned)) {
      return fail(NvcompPipelineStatus::invalid_configuration,
                  "pinned codec slot byte accounting overflowed");
    }
    saturating_add(telemetry_.pinned_slot_bytes, pinned);
    telemetry_.pinned_slot_peak_bytes =
        std::max(telemetry_.pinned_slot_peak_bytes, telemetry_.pinned_slot_bytes);
    slot.accounted = true;
    return NvcompPipelineStatus::success;
  }

  [[nodiscard]] bool cleanup_created_slots() noexcept {
    bool cleanup_ok = true;
    for (auto& slot : slots_) {
      cleanup_ok = destroy_slot(slot) && cleanup_ok;
    }
    if (cleanup_ok) {
      slots_.clear();
    }
    return cleanup_ok;
  }

  void note_cleanup_cuda_failure(const Result code) noexcept {
    native_error_ = static_cast<std::int64_t>(code);
    saturating_add(telemetry_.cuda_failures, 1);
  }

  [[nodiscard]] bool unload_verification_module() noexcept {
    if (verification_module_ == nullptr) {
      verification_function_ = nullptr;
      return true;
    }
    const Result code = cuda_.module_unload_(verification_module_);
    if (code != CUDA_SUCCESS) {
      note_cleanup_cuda_failure(code);
      return false;
    }
    verification_module_ = nullptr;
    verification_function_ = nullptr;
    return true;
  }

  [[nodiscard]] bool cleanup_setup_resources() noexcept {
    // Every resource remains idle here: setup has not exposed the pipeline, or close has already
    // drained every active generation. A failed CUDA destroy/free does not prove release, so retain
    // the exact handle and permit a later close() call to retry it.
    const bool slots_released = cleanup_created_slots();
    const bool module_released = unload_verification_module();
    cleanup_pending_ = !slots_released || !module_released;
    return !cleanup_pending_;
  }

  [[nodiscard]] bool destroy_slot(Slot& slot) noexcept {
    bool ok = true;
    if (slot.host_metadata != nullptr) {
      const Result code = cuda_.mem_free_host_(slot.host_metadata);
      if (code == CUDA_SUCCESS) {
        slot.host_metadata = nullptr;
      } else {
        note_cleanup_cuda_failure(code);
        ok = false;
      }
    }
    if (slot.host_payload != nullptr) {
      const Result code = cuda_.mem_free_host_(slot.host_payload);
      if (code == CUDA_SUCCESS) {
        slot.host_payload = nullptr;
      } else {
        note_cleanup_cuda_failure(code);
        ok = false;
      }
    }
    if (slot.workspace != 0) {
      const Result code = cuda_.mem_free_(slot.workspace);
      if (code == CUDA_SUCCESS) {
        slot.workspace = 0;
      } else {
        note_cleanup_cuda_failure(code);
        ok = false;
      }
    }
    if (slot.device_metadata != 0) {
      const Result code = cuda_.mem_free_(slot.device_metadata);
      if (code == CUDA_SUCCESS) {
        slot.device_metadata = 0;
      } else {
        note_cleanup_cuda_failure(code);
        ok = false;
      }
    }
    if (slot.device_payload != 0) {
      const Result code = cuda_.mem_free_(slot.device_payload);
      if (code == CUDA_SUCCESS) {
        slot.device_payload = 0;
      } else {
        note_cleanup_cuda_failure(code);
        ok = false;
      }
    }
    if (slot.event != nullptr) {
      const Result code = cuda_.event_destroy_(slot.event);
      if (code == CUDA_SUCCESS) {
        slot.event = nullptr;
      } else {
        note_cleanup_cuda_failure(code);
        ok = false;
      }
    }
    if (slot.stream != nullptr) {
      const Result code = cuda_.stream_destroy_(slot.stream);
      if (code == CUDA_SUCCESS) {
        slot.stream = nullptr;
      } else {
        note_cleanup_cuda_failure(code);
        ok = false;
      }
    }
    const bool released = slot.host_metadata == nullptr && slot.host_payload == nullptr &&
                          slot.workspace == 0 && slot.device_metadata == 0 &&
                          slot.device_payload == 0 && slot.event == nullptr &&
                          slot.stream == nullptr;
    if (released && slot.accounted) {
      const std::uint64_t device_bytes =
          payload_bytes_per_slot_ + metadata_.total_bytes + workspace_bytes_per_slot_;
      const std::uint64_t pinned_bytes = payload_bytes_per_slot_ + metadata_.total_bytes;
      telemetry_.workspace_bytes -= workspace_bytes_per_slot_;
      telemetry_.device_slot_bytes -= device_bytes;
      telemetry_.pinned_slot_bytes -= pinned_bytes;
      slot.accounted = false;
    }
    return ok && released;
  }

  [[nodiscard]] NvcompPipelineStatus operational_status() const noexcept {
    if (quarantined_) {
      return NvcompPipelineStatus::quarantined;
    }
    if (!setup_ || closed_) {
      return NvcompPipelineStatus::closed;
    }
    return NvcompPipelineStatus::success;
  }

  [[nodiscard]] Slot* acquire_slot(const NvcompOperationKind kind,
                                   const std::uint64_t source_generation) noexcept {
    slots_exhausted_ = true;
    for (std::size_t offset = 0; offset < slots_.size(); ++offset) {
      const std::size_t index = (next_slot_ + offset) % slots_.size();
      Slot& slot = slots_[index];
      if (slot.active || slot.generation == std::numeric_limits<std::uint64_t>::max()) {
        if (slot.active) {
          slots_exhausted_ = false;
        }
        continue;
      }
      if (next_operation_id_ == 0) {
        return nullptr;
      }
      if (slot.generation != 0) {
        saturating_add(telemetry_.slots_reused, 1);
      }
      ++slot.generation;
      slot.active = true;
      slot.submitted = false;
      slot.codec_submitted = false;
      slot.trace_key = {};
      slot.trace_path = CompressionPath::nvcomp_gpu_codec;
      slot.speculative = false;
      slot.h2d_trace_submitted = false;
      slot.d2h_trace_submitted = false;
      slot.h2d_physical_bytes = 0;
      slot.d2h_physical_bytes = 0;
      slot.encoded_physical_bytes = 0;
      slot.phase = Phase::idle;
      slot.ticket = {next_operation_id_, slot.generation, source_generation,
                     static_cast<std::uint32_t>(index), kind};
      if (next_operation_id_ == std::numeric_limits<std::uint64_t>::max()) {
        next_operation_id_ = 0;
      } else {
        ++next_operation_id_;
      }
      next_slot_ = (index + 1) % slots_.size();
      slots_exhausted_ = false;
      return &slot;
    }
    return nullptr;
  }

  [[nodiscard]] Slot* ticket_slot(const NvcompPipelineTicket& ticket) noexcept {
    if (!ticket || ticket.slot_index >= slots_.size()) {
      return nullptr;
    }
    Slot& slot = slots_[ticket.slot_index];
    return slot.active && slot.ticket == ticket ? &slot : nullptr;
  }

  void emit_trace(const Slot& slot, const CompressionTraceEventKind kind,
                  const std::string_view reason) const noexcept {
    if (config_.trace_observer == nullptr) {
      return;
    }
    CompressionTraceEvent event;
    event.kind = kind;
    event.key = slot.trace_key;
    event.operation_id = slot.ticket.operation_id;
    event.source_generation = slot.ticket.source_generation;
    event.slot_generation = slot.ticket.slot_generation;
    event.path = slot.trace_path;
    event.logical_bytes = slot.valid_bytes;
    event.reason = reason;
    event.speculative = slot.speculative;
    switch (kind) {
    case CompressionTraceEventKind::decode_submit:
    case CompressionTraceEventKind::decode_retire:
    case CompressionTraceEventKind::h2d_submit:
    case CompressionTraceEventKind::h2d_retire:
      event.from_representation = BackingRepresentation::lz4_blocks;
      event.physical_bytes = slot.h2d_physical_bytes;
      break;
    case CompressionTraceEventKind::encode_submit:
      event.target_generation = slot.ticket.source_generation + 1U;
      event.to_representation = BackingRepresentation::lz4_blocks;
      break;
    case CompressionTraceEventKind::encode_retire:
      event.target_generation = slot.ticket.source_generation + 1U;
      event.to_representation = BackingRepresentation::lz4_blocks;
      event.physical_bytes = slot.encoded_physical_bytes;
      break;
    case CompressionTraceEventKind::d2h_submit:
    case CompressionTraceEventKind::d2h_retire:
      event.target_generation = slot.ticket.source_generation + 1U;
      event.to_representation = BackingRepresentation::lz4_blocks;
      event.physical_bytes = slot.d2h_physical_bytes;
      break;
    case CompressionTraceEventKind::generation_discard:
      return;
    }
    config_.trace_observer(config_.trace_user_data, event);
  }

  [[nodiscard]] bool copy_h2d(Slot& slot, const DevicePointer destination, const void* source,
                              const std::uint64_t bytes, const bool payload) noexcept {
    if (!fits_size(bytes)) {
      native_error_.reset();
      saturating_add(telemetry_.cuda_failures, 1);
      return false;
    }
    const Result code =
        cuda_.memcpy_h2d_async_(destination, source, static_cast<std::size_t>(bytes), slot.stream);
    if (code != CUDA_SUCCESS) {
      native_error_ = static_cast<std::int64_t>(code);
      saturating_add(telemetry_.cuda_failures, 1);
      return false;
    }
    slot.submitted = true;
    saturating_add(payload ? telemetry_.pcie_h2d_payload_bytes : telemetry_.pcie_h2d_metadata_bytes,
                   bytes);
    saturating_add(slot.h2d_physical_bytes, bytes);
    return true;
  }

  [[nodiscard]] bool copy_d2h(Slot& slot, void* destination, const DevicePointer source,
                              const std::uint64_t bytes, const bool payload) noexcept {
    if (!fits_size(bytes)) {
      native_error_.reset();
      saturating_add(telemetry_.cuda_failures, 1);
      return false;
    }
    const Result code =
        cuda_.memcpy_d2h_async_(destination, source, static_cast<std::size_t>(bytes), slot.stream);
    if (code != CUDA_SUCCESS) {
      native_error_ = static_cast<std::int64_t>(code);
      saturating_add(telemetry_.cuda_failures, 1);
      return false;
    }
    slot.submitted = true;
    saturating_add(payload ? telemetry_.pcie_d2h_payload_bytes : telemetry_.pcie_d2h_metadata_bytes,
                   bytes);
    saturating_add(slot.d2h_physical_bytes, bytes);
    return true;
  }

  [[nodiscard]] bool submit_decode_metadata(Slot& slot, const std::size_t count) noexcept {
    return copy_h2d(slot, slot.device_metadata + metadata_.input_ptrs,
                    host_region<void*>(slot.host_metadata, metadata_.input_ptrs),
                    count * sizeof(void*), false) &&
           copy_h2d(slot, slot.device_metadata + metadata_.input_sizes,
                    host_region<std::size_t>(slot.host_metadata, metadata_.input_sizes),
                    count * sizeof(std::size_t), false) &&
           copy_h2d(slot, slot.device_metadata + metadata_.output_ptrs,
                    host_region<void*>(slot.host_metadata, metadata_.output_ptrs),
                    count * sizeof(void*), false) &&
           copy_h2d(slot, slot.device_metadata + metadata_.output_sizes,
                    host_region<std::size_t>(slot.host_metadata, metadata_.output_sizes),
                    count * sizeof(std::size_t), false);
  }

  [[nodiscard]] bool submit_encode_metadata(Slot& slot, const std::size_t count) noexcept {
    return copy_h2d(slot, slot.device_metadata + metadata_.input_ptrs,
                    host_region<void*>(slot.host_metadata, metadata_.input_ptrs),
                    count * sizeof(void*), false) &&
           copy_h2d(slot, slot.device_metadata + metadata_.input_sizes,
                    host_region<std::size_t>(slot.host_metadata, metadata_.input_sizes),
                    count * sizeof(std::size_t), false) &&
           copy_h2d(slot, slot.device_metadata + metadata_.output_ptrs,
                    host_region<void*>(slot.host_metadata, metadata_.output_ptrs),
                    count * sizeof(void*), false);
  }

  [[nodiscard]] bool record_event(Slot& slot) noexcept {
    const Result code = cuda_.event_record_(slot.event, slot.stream);
    if (code != CUDA_SUCCESS) {
      saturating_add(telemetry_.cuda_failures, 1);
      native_error_ = static_cast<std::int64_t>(code);
      return false;
    }
    saturating_add(telemetry_.events_recorded, 1);
    return true;
  }

  [[nodiscard]] Result submit_verification(Slot& slot, const DevicePointer address,
                                           const std::uint64_t valid_bytes,
                                           const ContentToken expected) noexcept {
    if (config_.verification_submit != nullptr) {
      return config_.verification_submit(
          config_.verification_user_data, address, valid_bytes, expected,
          slot.device_metadata + metadata_.verification_match, slot.stream);
    }
    constexpr unsigned int threads = 256;
    const std::uint64_t block_count =
        (valid_bytes + nvcomp_lz4_block_bytes - 1) / nvcomp_lz4_block_bytes;
    const std::uint64_t grid = (block_count + threads - 1) / threads;
    if (grid == 0 || grid > std::numeric_limits<unsigned int>::max()) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    DevicePointer input = address;
    std::uint64_t bytes = valid_bytes;
    DevicePointer token = slot.device_metadata + metadata_.verification_token;
    void* arguments[] = {&input, &bytes, &token};
    return cuda_.launch_kernel_(verification_function_, static_cast<unsigned int>(grid), 1, 1,
                                threads, 1, 1, 0, slot.stream, arguments, nullptr);
  }

  [[nodiscard]] NvcompPipelineStatus finish_decode(Slot& slot) noexcept {
    retire_verification_timing(slot);
    if (slot.batch_count != 0) {
      const auto* actual = host_region<std::size_t>(slot.host_metadata, metadata_.actual_sizes);
      const auto* statuses = host_region<nvcompStatus_t>(slot.host_metadata, metadata_.statuses);
      for (std::size_t index = 0; index < slot.batch_count; ++index) {
        const std::uint32_t block_index = slot.batch_blocks[index];
        const std::uint64_t logical_offset =
            static_cast<std::uint64_t>(block_index) * nvcomp_lz4_block_bytes;
        const std::size_t expected = static_cast<std::size_t>(
            std::min(slot.valid_bytes - logical_offset, nvcomp_lz4_block_bytes));
        if (statuses[index] != nvcompSuccess || actual[index] != expected) {
          saturating_add(telemetry_.codec_failures, 1);
          terminal_failure(slot);
          return fail(NvcompPipelineStatus::corrupt_data,
                      "nvCOMP decode status or output size failed validation");
        }
      }
    }
    bool verified = false;
    if (config_.verification_submit != nullptr) {
      const auto* verification =
          host_region<std::uint32_t>(slot.host_metadata, metadata_.verification_match);
      verified = *verification == 1U;
    } else {
      const auto* token =
          host_region<ContentToken>(slot.host_metadata, metadata_.verification_token);
      verified = slot.expected_token.has_value() && *token == *slot.expected_token;
    }
    if (!verified) {
      saturating_add(telemetry_.verification_failures, 1);
      terminal_failure(slot);
      return fail(NvcompPipelineStatus::corrupt_data,
                  "GPU verification token rejected the decoded chunk");
    }
    if (slot.h2d_trace_submitted) {
      emit_trace(slot, CompressionTraceEventKind::h2d_retire,
                 "decoded_resident_generation_verified");
    }
    if (slot.codec_submitted) {
      emit_trace(slot, CompressionTraceEventKind::decode_retire,
                 "nvcomp_decode_generation_verified");
    }
    retire(slot);
    return NvcompPipelineStatus::success;
  }

  [[nodiscard]] NvcompPipelineStatus begin_encode_transfer(Slot& slot) noexcept {
    const auto* actual = host_region<std::size_t>(slot.host_metadata, metadata_.actual_sizes);
    const auto* statuses = host_region<nvcompStatus_t>(slot.host_metadata, metadata_.statuses);
    slot.transfer_offsets.assign(slot.batch_count, 0);
    slot.transfer_storage.assign(slot.batch_count, BlockStorage::raw);
    std::uint64_t cursor = 0;
    std::uint64_t remaining = slot.valid_bytes;
    for (std::size_t index = 0; index < slot.batch_count; ++index) {
      const std::uint64_t raw_bytes = std::min(remaining, nvcomp_lz4_block_bytes);
      if (statuses[index] != nvcompSuccess || actual[index] == 0 ||
          actual[index] > max_compressed_block_bytes_) {
        saturating_add(telemetry_.codec_failures, 1);
        terminal_failure(slot);
        return fail(NvcompPipelineStatus::codec_failure,
                    "nvCOMP encode status or output size failed validation");
      }
      const bool compressed = actual[index] < raw_bytes;
      const std::uint64_t transfer_bytes = compressed ? actual[index] : raw_bytes;
      if (cursor > payload_bytes_per_slot_ - transfer_bytes) {
        terminal_failure(slot);
        return fail(NvcompPipelineStatus::codec_failure,
                    "encoded transfer layout exceeds its pinned codec slot");
      }
      slot.transfer_offsets[index] = cursor;
      slot.transfer_storage[index] = compressed ? BlockStorage::lz4 : BlockStorage::raw;
      cursor += transfer_bytes;
      remaining -= raw_bytes;
    }
    slot.encoded_physical_bytes = cursor;
    emit_trace(slot, CompressionTraceEventKind::encode_retire, "nvcomp_encode_generation_verified");
    remaining = slot.valid_bytes;
    for (std::size_t index = 0; index < slot.batch_count; ++index) {
      const std::uint64_t raw_bytes = std::min(remaining, nvcomp_lz4_block_bytes);
      const bool compressed = slot.transfer_storage[index] == BlockStorage::lz4;
      const std::uint64_t transfer_bytes = compressed ? actual[index] : raw_bytes;
      const DevicePointer source =
          compressed ? slot.device_payload + static_cast<DevicePointer>(index * output_stride_)
                     : static_cast<DevicePointer>(reinterpret_cast<std::uintptr_t>(
                           host_region<void*>(slot.host_metadata, metadata_.input_ptrs)[index]));
      if (!copy_d2h(slot, static_cast<std::byte*>(slot.host_payload) + slot.transfer_offsets[index],
                    source, transfer_bytes, true)) {
        return recover_copy_failure(slot, "encoded payload D2H submission failed");
      }
      if (compressed) {
        saturating_add(telemetry_.compressed_d2h_blocks, 1);
      } else {
        saturating_add(telemetry_.raw_fallback_blocks, 1);
        saturating_add(telemetry_.raw_fallback_bytes, raw_bytes);
      }
      remaining -= raw_bytes;
    }
    emit_trace(slot, CompressionTraceEventKind::d2h_submit, "encoded_payload_enqueued");
    slot.d2h_trace_submitted = true;
    slot.phase = Phase::encode_transfer_event;
    if (!record_event(slot)) {
      return quarantine_after_submission(slot, "encoded payload event record failed", std::nullopt);
    }
    return NvcompPipelineStatus::not_ready;
  }

  [[nodiscard]] NvcompPipelineStatus finish_encode_transfer(Slot& slot,
                                                            Lz4BlocksV1* candidate) noexcept {
    retire_verification_timing(slot);
    ContentToken source_token{};
    if (config_.verification_submit != nullptr) {
      const auto* match =
          host_region<std::uint32_t>(slot.host_metadata, metadata_.verification_match);
      if (*match != 1U || !slot.expected_token.has_value()) {
        saturating_add(telemetry_.verification_failures, 1);
        terminal_failure(slot);
        return fail(NvcompPipelineStatus::corrupt_data,
                    "custom verifier rejected the GPU encode source generation");
      }
      source_token = *slot.expected_token;
    } else {
      source_token = *host_region<ContentToken>(slot.host_metadata, metadata_.verification_token);
      if (slot.expected_token.has_value() && source_token != *slot.expected_token) {
        saturating_add(telemetry_.verification_failures, 1);
        terminal_failure(slot);
        return fail(NvcompPipelineStatus::corrupt_data,
                    "GPU encode source token differs from the expected content token");
      }
    }
    Lz4BlocksV1 built;
    built.format_version = lz4_blocks_format_version;
    built.valid_bytes = slot.valid_bytes;
    built.generation = slot.ticket.source_generation + 1;
    built.blocks.reserve(slot.batch_count);
    const auto* actual = host_region<std::size_t>(slot.host_metadata, metadata_.actual_sizes);
    std::uint64_t remaining = slot.valid_bytes;
    for (std::size_t index = 0; index < slot.batch_count; ++index) {
      const std::uint64_t raw_bytes = std::min(remaining, nvcomp_lz4_block_bytes);
      const BlockStorage storage = slot.transfer_storage[index];
      const std::size_t payload_bytes =
          storage == BlockStorage::lz4 ? actual[index] : static_cast<std::size_t>(raw_bytes);
      const std::byte* payload =
          static_cast<const std::byte*>(slot.host_payload) + slot.transfer_offsets[index];
      Lz4BlockV1 block;
      block.storage = storage;
      block.uncompressed_bytes = static_cast<std::uint32_t>(raw_bytes);
      block.payload.assign(payload, payload + payload_bytes);
      built.blocks.push_back(std::move(block));
      remaining -= raw_bytes;
    }
    ContentToken candidate_token{};
    if (compute_lz4_blocks_content_token(built, cpu_codec_, candidate_token) !=
        CodecStatus::success) {
      saturating_add(telemetry_.codec_failures, 1);
      terminal_failure(slot);
      return fail(NvcompPipelineStatus::corrupt_data,
                  "CPU LZ4 verification rejected the GPU-produced candidate");
    }
    built.content_token = candidate_token;
    if (built.content_token != source_token) {
      saturating_add(telemetry_.verification_failures, 1);
      terminal_failure(slot);
      return fail(NvcompPipelineStatus::corrupt_data,
                  "CPU-decoded GPU encode candidate differs from its dirty source generation");
    }

    slot.completed_candidate = std::move(built);
    slot.phase = Phase::encode_ready;
    if (candidate == nullptr) {
      return NvcompPipelineStatus::invalid_argument;
    }
    *candidate = std::move(*slot.completed_candidate);
    slot.completed_candidate.reset();
    slot.candidate_delivered = true;
    return NvcompPipelineStatus::success;
  }

  void retire(Slot& slot) noexcept {
    saturating_add(telemetry_.operations_retired, 1);
    reset_operation(slot);
  }

  void terminal_failure(Slot& slot) noexcept {
    saturating_add(telemetry_.operations_retired, 1);
    reset_operation(slot);
  }

  void reset_operation(Slot& slot) noexcept {
    slot.active = false;
    slot.submitted = false;
    slot.codec_submitted = false;
    slot.trace_key = {};
    slot.trace_path = CompressionPath::nvcomp_gpu_codec;
    slot.speculative = false;
    slot.h2d_trace_submitted = false;
    slot.d2h_trace_submitted = false;
    slot.h2d_physical_bytes = 0;
    slot.d2h_physical_bytes = 0;
    slot.encoded_physical_bytes = 0;
    slot.phase = Phase::idle;
    slot.ticket = {};
    slot.valid_bytes = 0;
    slot.batch_count = 0;
    slot.expected_token.reset();
    slot.batch_blocks.clear();
    slot.transfer_offsets.clear();
    slot.transfer_storage.clear();
    slot.completed_candidate.reset();
    slot.candidate_delivered = false;
    slot.verification_started = {};
  }

  void retire_verification_timing(Slot& slot) noexcept {
    if (slot.verification_started == std::chrono::steady_clock::time_point{}) {
      return;
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - slot.verification_started);
    if (elapsed.count() > 0) {
      saturating_add(telemetry_.verification_nanoseconds,
                     static_cast<std::uint64_t>(elapsed.count()));
    }
    slot.verification_started = {};
  }

  [[nodiscard]] NvcompPipelineStatus
  abort_before_event(Slot& slot, const NvcompPipelineStatus status, const char* message) noexcept {
    reset_operation(slot);
    return fail(status, message);
  }

  [[nodiscard]] NvcompPipelineStatus
  abort_or_quarantine(Slot& slot, const NvcompPipelineStatus status, const char* message) noexcept {
    if (slot.submitted) {
      return recover_after_partial_submission(slot, status, message);
    }
    return abort_before_event(slot, status, message);
  }

  [[nodiscard]] NvcompPipelineStatus recover_copy_failure(Slot& slot,
                                                          const char* message) noexcept {
    return recover_after_partial_submission(slot, NvcompPipelineStatus::cuda_failure, message);
  }

  [[nodiscard]] NvcompPipelineStatus recover_codec_launch_failure(Slot& slot,
                                                                  const nvcompStatus_t codec_status,
                                                                  const char* message) noexcept {
    native_error_ = static_cast<std::int64_t>(codec_status);
    return recover_after_partial_submission(slot, NvcompPipelineStatus::codec_failure, message);
  }

  [[nodiscard]] NvcompPipelineStatus
  recover_after_partial_submission(Slot& slot, const NvcompPipelineStatus status,
                                   const char* message) noexcept {
    if (!slot.submitted) {
      reset_operation(slot);
      return fail(status, message);
    }
    const Result synchronized = cuda_.stream_synchronize_(slot.stream);
    if (synchronized != CUDA_SUCCESS) {
      return quarantine_after_submission(slot, message, synchronized);
    }
    if (slot.h2d_trace_submitted) {
      emit_trace(slot, CompressionTraceEventKind::h2d_retire,
                 "h2d_generation_synchronized_before_codec_fallback");
    }
    terminal_failure(slot);
    return fail(status, message);
  }

  [[nodiscard]] NvcompPipelineStatus
  quarantine_after_submission(Slot&, const char* message,
                              const std::optional<Result> native) noexcept {
    quarantined_ = true;
    telemetry_.quarantines = 1;
    if (native.has_value()) {
      native_error_ = static_cast<std::int64_t>(*native);
      saturating_add(telemetry_.cuda_failures, 1);
    }
    set_error(message);
    // A submitted generation whose completion cannot be proven owns its stream, event, device
    // buffers and nvCOMP module until the isolated worker exits.
    cuda_.abandon();
    nvcomp_.abandon();
    return NvcompPipelineStatus::quarantined;
  }

  [[nodiscard]] NvcompPipelineStatus cuda_failure(const char* operation,
                                                  const Result code) noexcept {
    native_error_ = static_cast<std::int64_t>(code);
    saturating_add(telemetry_.cuda_failures, 1);
    return fail(NvcompPipelineStatus::cuda_failure, operation);
  }

  [[nodiscard]] NvcompPipelineStatus fail(const NvcompPipelineStatus status,
                                          const char* message) noexcept {
    set_error(message);
    return status;
  }

  void set_error(const char* message) noexcept {
    try {
      error_ = message == nullptr ? "" : message;
    } catch (...) {
      error_.clear();
    }
  }

  cuda::CudaApi& cuda_;
  nvcomp::NvcompApi& nvcomp_;
  const BlockCodec& cpu_codec_;
  NvcompPipelineConfig config_;
  NvcompPipelineTelemetry telemetry_;
  std::vector<Slot> slots_;
  MetadataLayout metadata_;
  nvcompBatchedLZ4CompressOpts_t compress_options_{};
  nvcompBatchedLZ4DecompressOpts_t decompress_options_{};
  nvcompAlignmentRequirements_t compress_alignment_{};
  nvcompAlignmentRequirements_t decompress_alignment_{};
  cuda::abi::Module verification_module_ = nullptr;
  cuda::abi::Function verification_function_ = nullptr;
  std::uint64_t block_count_ = 0;
  std::uint64_t output_stride_ = 0;
  std::uint64_t payload_bytes_per_slot_ = 0;
  std::uint64_t workspace_bytes_per_slot_ = 0;
  std::size_t max_compressed_block_bytes_ = 0;
  std::size_t next_slot_ = 0;
  std::uint64_t next_operation_id_ = 1;
  bool slots_exhausted_ = false;
  bool setup_ = false;
  bool closed_ = false;
  bool cleanup_pending_ = false;
  bool quarantined_ = false;
  std::string error_;
  std::optional<std::int64_t> native_error_;
};

NvcompLz4Pipeline::NvcompLz4Pipeline(cuda::CudaApi& cuda_api, nvcomp::NvcompApi& nvcomp_api,
                                     const BlockCodec& cpu_lz4_codec, NvcompPipelineConfig config)
    : impl_(std::make_unique<Impl>(cuda_api, nvcomp_api, cpu_lz4_codec, std::move(config))) {}

NvcompLz4Pipeline::~NvcompLz4Pipeline() {
  if (impl_) {
    (void)impl_->close();
  }
}

NvcompPipelineStatus NvcompLz4Pipeline::setup() noexcept {
  return impl_->setup();
}

NvcompPipelineStatus NvcompLz4Pipeline::decode(const NvcompDecodeRequest& request,
                                               NvcompPipelineTicket& ticket) noexcept {
  return impl_->decode(request, ticket);
}

NvcompPipelineStatus NvcompLz4Pipeline::encode(const NvcompEncodeRequest& request,
                                               NvcompPipelineTicket& ticket) noexcept {
  return impl_->encode(request, ticket);
}

NvcompPipelineStatus NvcompLz4Pipeline::poll(const NvcompPipelineTicket& ticket,
                                             Lz4BlocksV1* candidate) noexcept {
  return impl_->poll(ticket, candidate);
}

NvcompPipelineStatus NvcompLz4Pipeline::wait(const NvcompPipelineTicket& ticket,
                                             Lz4BlocksV1* candidate) noexcept {
  return impl_->wait(ticket, candidate);
}

NvcompPipelineStatus
NvcompLz4Pipeline::acknowledge_host_commit(const NvcompPipelineTicket& ticket) noexcept {
  return impl_->acknowledge_host_commit(ticket);
}

NvcompPipelineStatus
NvcompLz4Pipeline::quarantine_uncommitted(const NvcompPipelineTicket& ticket) noexcept {
  return impl_->quarantine_uncommitted(ticket);
}

NvcompPipelineStatus NvcompLz4Pipeline::close() noexcept {
  return impl_->close();
}

const NvcompPipelineConfig& NvcompLz4Pipeline::config() const noexcept {
  return impl_->config();
}

const NvcompPipelineTelemetry& NvcompLz4Pipeline::telemetry() const noexcept {
  return impl_->telemetry();
}

const std::string& NvcompLz4Pipeline::error() const noexcept {
  return impl_->error();
}

std::optional<std::int64_t> NvcompLz4Pipeline::native_error() const noexcept {
  return impl_->native_error();
}

bool NvcompLz4Pipeline::quarantined() const noexcept {
  return impl_->quarantined();
}

bool NvcompLz4Pipeline::is_setup() const noexcept {
  return impl_->is_setup();
}

} // namespace xvram::residency
