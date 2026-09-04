#pragma once

#include "residency/core.hpp"

#include <compare>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace xvram::residency {

inline constexpr std::uint32_t lz4_blocks_format_version = 1;
inline constexpr std::uint64_t compression_block_bytes = 64ULL * 1024ULL;

enum class CompressionMode {
  disabled,
  adaptive,
  capacity,
};

enum class CompressionCodec {
  automatic,
  lz4,
};

enum class CompressionPath {
  raw,
  cpu_lz4_gpu_decode,
  nvcomp_gpu_codec,
};

enum class BackingRepresentation {
  invalid,
  implicit_zero,
  raw,
  lz4_blocks,
};

[[nodiscard]] std::string_view compression_mode_name(CompressionMode mode) noexcept;
[[nodiscard]] std::string_view compression_codec_name(CompressionCodec codec) noexcept;
[[nodiscard]] std::string_view compression_path_name(CompressionPath path) noexcept;
[[nodiscard]] std::string_view backing_representation_name(BackingRepresentation value) noexcept;

struct ContentToken {
  std::uint64_t high = 0;
  std::uint64_t low = 0;

  [[nodiscard]] auto operator<=>(const ContentToken&) const noexcept = default;
};

// The token is deterministic and sensitive to byte order, block position, and valid tail size. It
// is an integrity token, not a cryptographic authentication code.
[[nodiscard]] ContentToken compute_content_token(std::span<const std::byte> bytes) noexcept;
[[nodiscard]] std::optional<std::uint64_t>
maximum_raw_backing_charge(std::uint64_t valid_bytes) noexcept;

enum class CodecStatus {
  success,
  invalid_argument,
  output_too_small,
  corrupt_input,
  internal_failure,
};

[[nodiscard]] std::string_view codec_status_name(CodecStatus status) noexcept;

// Adapter boundary for LZ4 1.10 and fake-codec tests. Implementations must emit and consume raw
// LZ4 blocks (not frames), must be deterministic for a fixed input, and must not throw.
class BlockCodec {
public:
  virtual ~BlockCodec() = default;

  [[nodiscard]] virtual CompressionCodec codec() const noexcept = 0;
  [[nodiscard]] virtual std::string_view implementation_name() const noexcept = 0;
  [[nodiscard]] virtual std::optional<std::size_t>
  maximum_compressed_bytes(std::size_t input_bytes) const noexcept = 0;
  [[nodiscard]] virtual CodecStatus compress(std::span<const std::byte> input,
                                             std::span<std::byte> output,
                                             std::size_t& written_bytes) const noexcept = 0;
  [[nodiscard]] virtual CodecStatus decompress(std::span<const std::byte> input,
                                               std::span<std::byte> output) const noexcept = 0;
};

enum class BlockStorage : std::uint8_t {
  implicit_zero,
  raw,
  lz4,
};

struct Lz4BlockV1 {
  BlockStorage storage = BlockStorage::implicit_zero;
  std::uint32_t uncompressed_bytes = 0;
  std::vector<std::byte> payload;
};

// Versioned, CPU-decodable host container. A raw block is used whenever LZ4 would expand it;
// implicit_zero is a lossless zero-elision extension and has no payload.
struct Lz4BlocksV1 {
  std::uint32_t format_version = lz4_blocks_format_version;
  std::uint64_t valid_bytes = 0;
  std::uint64_t generation = 0;
  ContentToken content_token;
  std::vector<Lz4BlockV1> blocks;
};

// Decodes at most one 64-KiB block at a time and recomputes the full container token. This is
// used at trust boundaries where an encoded candidate must be verified without materializing a
// second chunk-sized raw copy.
[[nodiscard]] CodecStatus compute_lz4_blocks_content_token(const Lz4BlocksV1& container,
                                                           const BlockCodec& codec,
                                                           ContentToken& token) noexcept;

enum class HostBudgetCategory {
  authoritative,
  conversion_scratch,
  pinned_staging,
  spill,
};

struct HostBudgetReservation {
  std::uint64_t id = 0;
  HostBudgetCategory category = HostBudgetCategory::conversion_scratch;
  std::uint64_t bytes = 0;

  [[nodiscard]] explicit operator bool() const noexcept {
    return id != 0 || bytes == 0;
  }
};

enum class HostBudgetStatus {
  success,
  invalid_argument,
  limit_exceeded,
  reservation_overflow,
  allocation_failure,
  unknown_reservation,
  reservation_mismatch,
};

[[nodiscard]] std::string_view host_budget_status_name(HostBudgetStatus status) noexcept;

struct HostBudgetSnapshot {
  std::uint64_t limit_bytes = 0;
  std::uint64_t total_bytes = 0;
  std::uint64_t authoritative_bytes = 0;
  std::uint64_t conversion_scratch_bytes = 0;
  std::uint64_t pinned_staging_bytes = 0;
  std::uint64_t spill_bytes = 0;
  std::uint64_t peak_bytes = 0;
  std::uint64_t authoritative_peak_bytes = 0;
  std::uint64_t conversion_scratch_peak_bytes = 0;
  std::uint64_t pinned_staging_peak_bytes = 0;
  std::uint64_t spill_peak_bytes = 0;
  std::uint64_t rejected_bytes = 0;
};

class HostBudgetLedger {
public:
  // A zero limit means no configured limit (UINT64_MAX), not a zero-byte budget.
  explicit HostBudgetLedger(std::uint64_t limit_bytes = 0);
  ~HostBudgetLedger();

  HostBudgetLedger(const HostBudgetLedger&) = delete;
  HostBudgetLedger& operator=(const HostBudgetLedger&) = delete;
  HostBudgetLedger(HostBudgetLedger&&) noexcept;
  HostBudgetLedger& operator=(HostBudgetLedger&&) noexcept;

  [[nodiscard]] HostBudgetStatus reserve(HostBudgetCategory category, std::uint64_t bytes,
                                         HostBudgetReservation& reservation) noexcept;
  [[nodiscard]] HostBudgetStatus release(HostBudgetReservation& reservation) noexcept;
  [[nodiscard]] HostBudgetSnapshot snapshot() const noexcept;

  // Atomically reclassifies a conversion reservation as authoritative storage while replacing an
  // older authoritative charge. This is public for backing-store adapters; ordinary callers
  // should use reserve/release.
  [[nodiscard]] HostBudgetStatus
  commit_authoritative(HostBudgetReservation& conversion_reservation,
                       std::uint64_t previous_authoritative_bytes,
                       std::uint64_t replacement_authoritative_bytes) noexcept;
  [[nodiscard]] HostBudgetStatus release_authoritative(std::uint64_t authoritative_bytes) noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

enum class BackingStatus {
  success,
  invalid_configuration,
  invalid_argument,
  duplicate_chunk,
  chunk_not_found,
  invalid_state,
  stale_generation,
  generation_overflow,
  range_overflow,
  range_out_of_bounds,
  host_budget_exceeded,
  allocation_failure,
  codec_failure,
  corrupt_data,
  not_beneficial,
  internal_failure,
};

[[nodiscard]] std::string_view backing_status_name(BackingStatus status) noexcept;

struct HostBackingConfig {
  std::uint64_t chunk_bytes = 64ULL * 1024ULL * 1024ULL;
  std::uint64_t host_store_cap_bytes = 0;
};

struct HostChunkInfo {
  ChunkKey key;
  BackingRepresentation representation = BackingRepresentation::invalid;
  std::uint64_t valid_bytes = 0;
  std::uint64_t generation = 0;
  std::uint64_t stored_payload_bytes = 0;
  std::uint64_t budget_charge_bytes = 0;
  ContentToken content_token;
};

struct BackingResult {
  BackingStatus status = BackingStatus::success;
  HostChunkInfo chunk;

  [[nodiscard]] explicit operator bool() const noexcept {
    return status == BackingStatus::success;
  }
};

class PreparedRawBacking {
public:
  PreparedRawBacking();
  ~PreparedRawBacking();
  PreparedRawBacking(PreparedRawBacking&&) noexcept;
  PreparedRawBacking& operator=(PreparedRawBacking&&) noexcept;

  PreparedRawBacking(const PreparedRawBacking&) = delete;
  PreparedRawBacking& operator=(const PreparedRawBacking&) = delete;

  [[nodiscard]] bool valid() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  friend class HostBackingStore;
};

class HostBackingStore {
public:
  // Exposed only as an incomplete implementation type so translation-unit-local builders can
  // construct immutable images without making their representation part of this internal API.
  struct Impl;

  HostBackingStore(HostBackingConfig config, const BlockCodec& lz4_codec);
  ~HostBackingStore();

  HostBackingStore(const HostBackingStore&) = delete;
  HostBackingStore& operator=(const HostBackingStore&) = delete;
  HostBackingStore(HostBackingStore&&) noexcept;
  HostBackingStore& operator=(HostBackingStore&&) noexcept;

  [[nodiscard]] const HostBackingConfig& config() const noexcept;
  [[nodiscard]] HostBudgetLedger& budget() noexcept;
  [[nodiscard]] const HostBudgetLedger& budget() const noexcept;

  [[nodiscard]] BackingResult register_chunk(
      ChunkKey key, std::uint64_t valid_bytes,
      BackingRepresentation initial_state = BackingRepresentation::implicit_zero) noexcept;
  [[nodiscard]] BackingResult erase_chunk(ChunkKey key, std::uint64_t expected_generation) noexcept;
  [[nodiscard]] BackingResult inspect(ChunkKey key) const noexcept;

  [[nodiscard]] BackingResult read(ChunkKey key, std::uint64_t offset,
                                   std::span<std::byte> output) const noexcept;
  [[nodiscard]] BackingResult write(ChunkKey key, std::uint64_t expected_generation,
                                    std::uint64_t offset,
                                    std::span<const std::byte> input) noexcept;
  [[nodiscard]] BackingResult
  replace_raw(ChunkKey key, std::uint64_t expected_generation, std::span<const std::byte> input,
              HostBudgetReservation* candidate_reservation = nullptr) noexcept;
  [[nodiscard]] BackingResult prepare_raw_replacement(ChunkKey key,
                                                      std::uint64_t expected_generation,
                                                      PreparedRawBacking& output) noexcept;
  [[nodiscard]] BackingStatus fill_prepared_raw(PreparedRawBacking& prepared,
                                                std::span<const std::byte> input) noexcept;
  [[nodiscard]] BackingResult
  commit_prepared_raw(ChunkKey key, std::uint64_t expected_generation, PreparedRawBacking& prepared,
                      HostBudgetReservation* candidate_reservation) noexcept;
  [[nodiscard]] BackingResult set_implicit_zero(ChunkKey key,
                                                std::uint64_t expected_generation) noexcept;
  [[nodiscard]] BackingResult invalidate(ChunkKey key, std::uint64_t expected_generation) noexcept;

  // Converts only after complete encode/decode/token verification. Raw fallback is selected per
  // block, so a committed container never expands its payload beyond valid_bytes.
  [[nodiscard]] BackingResult compress(ChunkKey key, std::uint64_t expected_generation) noexcept;
  [[nodiscard]] BackingResult materialize_raw(ChunkKey key,
                                              std::uint64_t expected_generation) noexcept;

  // Entry point for a GPU-produced LZ4 result. The candidate must name expected_generation + 1;
  // it is fully CPU-decoded and token-verified before the atomic commit.
  [[nodiscard]] BackingResult
  commit_lz4_blocks(ChunkKey key, std::uint64_t expected_generation, Lz4BlocksV1 candidate,
                    HostBudgetReservation* candidate_reservation = nullptr) noexcept;
  [[nodiscard]] BackingStatus export_lz4_blocks(ChunkKey key, Lz4BlocksV1& output) const noexcept;

private:
  std::unique_ptr<Impl> impl_;
};

struct CostObservation {
  CompressionPath path = CompressionPath::raw;
  std::uint64_t raw_bytes = 0;
  std::uint64_t stored_bytes = 0;
  double encode_us = 0.0;
  double decode_us = 0.0;
  double transfer_us = 0.0;
  double staging_us = 0.0;
  double cpu_queue_us = 0.0;
  double sm_opportunity_us = 0.0;
  double gpu_occupancy = 0.0;
  double cpu_availability = 1.0;
  std::uint64_t reuse_count = 0;
  bool dirty = false;
};

struct CostPathMetrics {
  std::uint32_t samples = 0;
  double compression_ratio = 1.0;
  double encode_us_per_raw_byte = 0.0;
  double decode_us_per_raw_byte = 0.0;
  double transfer_us_per_stored_byte = 0.0;
  double staging_us_per_raw_byte = 0.0;
  double cpu_queue_us = 0.0;
  double sm_opportunity_us_per_raw_byte = 0.0;
  double gpu_occupancy = 0.0;
  double cpu_availability = 1.0;
  double reuse_count = 0.0;
  double dirty_rate = 0.0;
};

struct CostEstimate {
  CompressionPath path = CompressionPath::raw;
  bool confident = false;
  std::uint64_t predicted_stored_bytes = 0;
  double predicted_total_us = 0.0;
};

struct CostDecision {
  CompressionPath path = CompressionPath::raw;
  bool calibration_required = true;
  bool margin_satisfied = false;
  bool capacity_override = false;
  bool never_compress = false;
  double required_savings_us = 0.0;
  CostEstimate raw;
  CostEstimate cpu_lz4_gpu_decode;
  CostEstimate nvcomp_gpu_codec;
};

class CompressionCostModel {
public:
  explicit CompressionCostModel(double ewma_alpha = 0.25);
  ~CompressionCostModel();

  CompressionCostModel(const CompressionCostModel&) = delete;
  CompressionCostModel& operator=(const CompressionCostModel&) = delete;
  CompressionCostModel(CompressionCostModel&&) noexcept;
  CompressionCostModel& operator=(CompressionCostModel&&) noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] std::uint64_t generation() const noexcept;

  // Starts generation-local probe tracking for content_generation. Historical EWMA path
  // calibration is deliberately retained so a new immutable backing generation does not have to
  // relearn platform transfer and codec costs.
  void reset(std::uint64_t content_generation) noexcept;
  [[nodiscard]] bool observe(std::uint64_t content_generation,
                             const CostObservation& observation) noexcept;

  // A probe is unfavorable unless it beats raw by max(10%, 50 us). Three consecutive
  // unfavorable probes suppress compressed paths in adaptive mode for this content generation.
  [[nodiscard]] bool record_probe(std::uint64_t content_generation, CompressionPath candidate_path,
                                  double raw_total_us, double candidate_total_us) noexcept;

  [[nodiscard]] CostPathMetrics metrics(CompressionPath path) const noexcept;
  [[nodiscard]] CostEstimate estimate(CompressionPath path, std::uint64_t raw_bytes) const noexcept;
  [[nodiscard]] CostDecision decide(CompressionMode mode, std::uint64_t raw_bytes,
                                    bool allow_cpu_path = true,
                                    bool allow_gpu_path = true) const noexcept;
  [[nodiscard]] bool never_compress() const noexcept;
  [[nodiscard]] std::uint32_t unfavorable_probe_count() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace xvram::residency
