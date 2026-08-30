#pragma once

#include "xvram/gemm/report.hpp"

#include <chrono>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>

namespace xvram::gemm_bench {

enum class RequestedDataType { suite, fp16, bf16, fp32, fp64 };
enum class RequestedComputeMode { automatic, strict_fp32, tf32, fp64 };
enum class RequestedLayout { row_major, column_major };
enum class RequestedOperation { none, transpose };

struct ExecutorOptions {
  std::int32_t device_ordinal = 0;
  std::optional<std::uint64_t> m;
  std::optional<std::uint64_t> n;
  std::optional<std::uint64_t> k;
  RequestedDataType data_type = RequestedDataType::suite;
  RequestedComputeMode compute_mode = RequestedComputeMode::automatic;
  RequestedLayout a_layout = RequestedLayout::row_major;
  RequestedLayout b_layout = RequestedLayout::row_major;
  RequestedLayout c_layout = RequestedLayout::row_major;
  RequestedOperation a_operation = RequestedOperation::none;
  RequestedOperation b_operation = RequestedOperation::none;
  std::uint64_t chunk_bytes = 64ULL * 1024ULL * 1024ULL;
  std::optional<std::uint64_t> cache_target_bytes;
  std::uint64_t workspace_bytes = 4ULL * 1024ULL * 1024ULL;
  std::uint64_t device_headroom_bytes = 512ULL * 1024ULL * 1024ULL;
  std::uint32_t staging_slots = 4;
  std::uint32_t passes = 1;
  std::chrono::seconds timeout{300};
  bool include_identifiers = false;
};

using ProgressCallback =
    std::function<void(const Report&, std::uint64_t completed, std::uint64_t total)>;

[[nodiscard]] Report run_executor(const ExecutorOptions& options,
                                  const ProgressCallback& progress = {});

[[nodiscard]] const char* requested_data_type_name(RequestedDataType value) noexcept;
[[nodiscard]] const char* requested_compute_mode_name(RequestedComputeMode value) noexcept;
[[nodiscard]] const char* requested_layout_name(RequestedLayout value) noexcept;
[[nodiscard]] const char* requested_operation_name(RequestedOperation value) noexcept;

} // namespace xvram::gemm_bench
