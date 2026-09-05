#pragma once
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace xvram::compat_bench {
struct Options {
  std::int32_t device = 0;
  std::optional<std::uint64_t> logical_bytes;
  std::optional<std::uint64_t> m, n, k;
  std::uint64_t chunk_bytes = 64ULL << 20U;
  std::uint64_t cache_bytes = 0;
  std::uint64_t headroom_bytes = 512ULL << 20U;
  std::uint64_t workspace_bytes = 4ULL << 20U;
  std::uint64_t padding = 0;
  std::uint64_t offset_elements = 0;
  std::uint64_t seed = 0x585652414d503661ULL;
  std::uint32_t passes = 2, staging_slots = 4, prefetch_distance = 2;
  std::uint64_t budget_poll_ms = 100, stall_timeout_ms = 5000;
  float alpha = 1.25F, beta = 0.5F;
  bool transpose_a = false, transpose_b = false;
  std::string policy = "clock", scenario = "suite";
  std::string cublas_library, cublas_lt_library;
  std::optional<std::string> json_path, trace_path;
  std::chrono::milliseconds timeout{900000};
  bool pretty = true, text = true, identifiers = false, worker = false;
  bool help = false, version = false;
  std::string test_worker;
};
[[nodiscard]] bool parse_options(int argc, char** argv, Options& options, std::string& error);
[[nodiscard]] std::vector<std::string> worker_arguments(const Options& options);
[[nodiscard]] const char* help_text() noexcept;
[[nodiscard]] std::filesystem::path utf8_path(std::string_view text);
} // namespace xvram::compat_bench
