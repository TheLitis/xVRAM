#pragma once

#include "residency/compression.hpp"

#include <chrono>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace xvram::residency {

inline constexpr std::uint32_t cpu_codec_min_workers = 1;
inline constexpr std::uint32_t cpu_codec_max_workers = 8;
inline constexpr std::uint32_t cpu_codec_max_queue_capacity = 4096;

enum class CpuCodecOperation : std::uint8_t {
  encode,
  decode,
};

enum class CpuCodecPoolStatus {
  success,
  not_ready,
  timeout,
  invalid_configuration,
  invalid_argument,
  queue_full,
  stale_ticket,
  closed,
  allocation_failure,
  codec_failure,
  internal_failure,
};

[[nodiscard]] std::string_view cpu_codec_operation_name(CpuCodecOperation operation) noexcept;
[[nodiscard]] std::string_view cpu_codec_pool_status_name(CpuCodecPoolStatus status) noexcept;

struct CpuCodecPoolConfig {
  std::uint32_t worker_count = 2;
  // Outstanding jobs include completed results until the caller consumes them. This makes the
  // memory bound independent of worker speed.
  std::uint32_t queue_capacity = 4;
};

struct CpuCodecTicket {
  std::uint64_t pool_generation = 0;
  std::uint64_t operation_id = 0;
  ChunkKey key;
  std::uint64_t source_generation = 0;
  CpuCodecOperation operation = CpuCodecOperation::encode;

  [[nodiscard]] explicit operator bool() const noexcept {
    return pool_generation != 0 && operation_id != 0;
  }

  [[nodiscard]] auto operator<=>(const CpuCodecTicket&) const noexcept = default;
};

// Input is owned by the request and then by the pool. Passing the request by value gives callers a
// safe copy for lvalues and a zero-copy ownership transfer for rvalues.
struct CpuCodecRequest {
  CpuCodecOperation operation = CpuCodecOperation::encode;
  ChunkKey key;
  std::uint64_t source_generation = 0;
  std::vector<std::byte> input;
  // Required for decode and ignored for encode.
  std::size_t decoded_bytes = 0;
};

struct CpuCodecResult {
  CpuCodecPoolStatus status = CpuCodecPoolStatus::success;
  CodecStatus codec_status = CodecStatus::success;
  CpuCodecTicket ticket;
  std::vector<std::byte> output;
  std::size_t input_bytes = 0;
  std::size_t output_bytes = 0;
  // An encode that would not shrink the input returns the original bytes through output. This
  // lets the caller build a raw block without a second authoritative-backing read.
  bool encoded_as_raw = false;
};

struct CpuCodecPoolTelemetry {
  std::uint64_t outstanding = 0;
  std::uint64_t queued = 0;
  std::uint64_t running = 0;
  std::uint64_t submitted = 0;
  std::uint64_t encode_submitted = 0;
  std::uint64_t decode_submitted = 0;
  std::uint64_t completed = 0;
  std::uint64_t retired = 0;
  std::uint64_t failures = 0;
  std::uint64_t exception_failures = 0;
  std::uint64_t queue_full = 0;
  std::uint64_t input_bytes = 0;
  std::uint64_t output_bytes = 0;
  std::uint64_t peak_outstanding = 0;
  std::uint64_t peak_running = 0;
};

// Injectable functions keep fault/exception tests deterministic. from_block_codec() captures a
// non-owning codec reference, so that codec must outlive the pool.
struct CpuCodecFunctions {
  using MaximumCompressedBytes = std::function<std::optional<std::size_t>(std::size_t input_bytes)>;
  using Compress = std::function<CodecStatus(
      std::span<const std::byte> input, std::span<std::byte> output, std::size_t& written_bytes)>;
  using Decompress =
      std::function<CodecStatus(std::span<const std::byte> input, std::span<std::byte> output)>;

  MaximumCompressedBytes maximum_compressed_bytes;
  Compress compress;
  Decompress decompress;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] static CpuCodecFunctions from_block_codec(const BlockCodec& codec);
};

class CpuCodecWorkerPool {
public:
  CpuCodecWorkerPool(CpuCodecPoolConfig config, const BlockCodec& codec);
  CpuCodecWorkerPool(CpuCodecPoolConfig config, CpuCodecFunctions functions);
  ~CpuCodecWorkerPool();

  CpuCodecWorkerPool(const CpuCodecWorkerPool&) = delete;
  CpuCodecWorkerPool& operator=(const CpuCodecWorkerPool&) = delete;
  CpuCodecWorkerPool(CpuCodecWorkerPool&&) = delete;
  CpuCodecWorkerPool& operator=(CpuCodecWorkerPool&&) = delete;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] const CpuCodecPoolConfig& config() const noexcept;

  [[nodiscard]] CpuCodecPoolStatus submit(CpuCodecRequest request, CpuCodecTicket& ticket) noexcept;
  [[nodiscard]] CpuCodecPoolStatus submit_encode(ChunkKey key, std::uint64_t source_generation,
                                                 std::vector<std::byte> input,
                                                 CpuCodecTicket& ticket) noexcept;
  [[nodiscard]] CpuCodecPoolStatus submit_encode_copy(ChunkKey key, std::uint64_t source_generation,
                                                      std::span<const std::byte> input,
                                                      CpuCodecTicket& ticket) noexcept;
  [[nodiscard]] CpuCodecPoolStatus submit_decode(ChunkKey key, std::uint64_t source_generation,
                                                 std::vector<std::byte> input,
                                                 std::size_t decoded_bytes,
                                                 CpuCodecTicket& ticket) noexcept;
  [[nodiscard]] CpuCodecPoolStatus submit_decode_copy(ChunkKey key, std::uint64_t source_generation,
                                                      std::span<const std::byte> input,
                                                      std::size_t decoded_bytes,
                                                      CpuCodecTicket& ticket) noexcept;

  // Supplying a result consumes the completed job and releases one queue-capacity unit. A null
  // result observes completion without consuming it.
  [[nodiscard]] CpuCodecPoolStatus poll(const CpuCodecTicket& ticket,
                                        CpuCodecResult* result = nullptr) noexcept;
  [[nodiscard]] CpuCodecPoolStatus wait(const CpuCodecTicket& ticket,
                                        std::chrono::milliseconds timeout,
                                        CpuCodecResult* result = nullptr) noexcept;

  // Stops accepting submissions, drains every accepted job, and joins all workers. Completed
  // results remain pollable until destruction. close() is idempotent.
  [[nodiscard]] CpuCodecPoolStatus close() noexcept;
  [[nodiscard]] CpuCodecPoolTelemetry telemetry() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace xvram::residency
