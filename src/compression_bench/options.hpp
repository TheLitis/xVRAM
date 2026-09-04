#pragma once

#include "residency/compression.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace xvram::compression {

enum class CliAction { run, help, version };
enum class RequestedCompressionPolicy { adaptive, capacity, both };
enum class RequestedPath { automatic, raw, cpu_lz4_gpu, gpu_lz4, all };
enum class RequestedReplacementPolicy { clock, lru, both };
enum class ScenarioKind {
  suite,
  compressible_read,
  incompressible_read,
  reuse,
  dirty_writeback,
  mixed,
  budget_pressure,
};

struct ExecutorOptions {
  std::int32_t device_ordinal = 0;
  std::optional<std::uint64_t> logical_bytes;
  std::uint64_t chunk_bytes = 64ULL * 1024ULL * 1024ULL;
  std::optional<std::uint64_t> cache_target_bytes;
  RequestedCompressionPolicy compression_policy = RequestedCompressionPolicy::adaptive;
  RequestedPath path = RequestedPath::automatic;
  residency::CompressionCodec codec = residency::CompressionCodec::automatic;
  RequestedReplacementPolicy replacement_policy = RequestedReplacementPolicy::clock;
  ScenarioKind scenario = ScenarioKind::suite;
  std::uint32_t passes = 2;
  std::uint32_t warmup_passes = 2;
  std::uint32_t measurement_passes = 5;
  std::optional<std::uint64_t> host_store_cap_bytes;
  std::optional<std::uint64_t> host_headroom_bytes;
  std::uint64_t device_headroom_bytes = 512ULL * 1024ULL * 1024ULL;
  std::uint64_t compression_scratch_cap_bytes = 256ULL * 1024ULL * 1024ULL;
  std::uint32_t codec_slots = 2;
  std::uint32_t codec_workers = 2;
  std::uint32_t staging_slots = 4;
  std::uint32_t prefetch_distance = 2;
  std::chrono::milliseconds budget_poll_interval{100};
  std::chrono::milliseconds stall_timeout{5000};
  std::chrono::milliseconds progress_heartbeat{1000};
  std::uint64_t seed = 0x585652414D503035ULL;
  bool trace_enabled = false;
  bool include_identifiers = false;
};

struct CliOptions {
  ExecutorOptions executor;
  std::chrono::seconds timeout{900};
  std::optional<std::string> json_path;
  std::optional<std::string> trace_path;
  bool pretty_json = true;
  bool print_text = true;
  bool worker = false;
  CliAction action = CliAction::run;
};

[[nodiscard]] bool parse_cli(int argc, char** argv, CliOptions& options, std::string& error);
void print_help(std::ostream& output);
[[nodiscard]] std::vector<std::string> worker_arguments(const CliOptions& options);
[[nodiscard]] std::filesystem::path current_executable_path(const char* argv_zero);

[[nodiscard]] std::string_view compression_policy_name(RequestedCompressionPolicy value) noexcept;
[[nodiscard]] std::string_view path_name(RequestedPath value) noexcept;
[[nodiscard]] std::string_view replacement_policy_name(RequestedReplacementPolicy value) noexcept;
[[nodiscard]] std::string_view scenario_name(ScenarioKind value) noexcept;

} // namespace xvram::compression
