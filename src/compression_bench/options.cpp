#include "compression_bench/options.hpp"

#include "xvram/base/size_parser.hpp"

#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <system_error>

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

namespace xvram::compression {
namespace {

[[nodiscard]] bool parse_u32(const std::string_view text, std::uint32_t& output) noexcept {
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), output, 10);
  return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

[[nodiscard]] bool parse_u64(const std::string_view text, std::uint64_t& output) noexcept {
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), output, 10);
  return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

[[nodiscard]] bool parse_i32(const std::string_view text, std::int32_t& output) noexcept {
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), output, 10);
  return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

[[nodiscard]] bool parse_seed(std::string_view text, std::uint64_t& output) noexcept {
  if (text.starts_with("0x") || text.starts_with("0X")) {
    text.remove_prefix(2);
  }
  if (text.empty() || text.size() > 16U) {
    return false;
  }
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), output, 16);
  return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

[[nodiscard]] std::optional<std::uint64_t> positive_size(const std::string_view text) {
  const ParsedSize parsed = parse_size(text);
  if (!parsed || *parsed.bytes == 0U) {
    return std::nullopt;
  }
  return *parsed.bytes;
}

[[nodiscard]] bool paths_collide(const std::string& left, const std::string& right) {
  std::error_code error;
  const std::filesystem::path left_absolute =
      std::filesystem::absolute(std::filesystem::path(left), error).lexically_normal();
  if (error) {
    return left == right;
  }
  const std::filesystem::path right_absolute =
      std::filesystem::absolute(std::filesystem::path(right), error).lexically_normal();
  return error ? left == right : left_absolute == right_absolute;
}

[[nodiscard]] std::string size_argument(const std::optional<std::uint64_t>& bytes) {
  return bytes.has_value() ? std::to_string(*bytes) + "B" : "auto";
}

[[nodiscard]] std::string size_argument(const std::uint64_t bytes) {
  return std::to_string(bytes) + "B";
}

[[nodiscard]] std::string seed_argument(const std::uint64_t value) {
  std::ostringstream output;
  output << "0x" << std::hex << std::nouppercase << std::setfill('0') << std::setw(16) << value;
  return output.str();
}

} // namespace

std::string_view compression_policy_name(const RequestedCompressionPolicy value) noexcept {
  switch (value) {
  case RequestedCompressionPolicy::adaptive:
    return "adaptive";
  case RequestedCompressionPolicy::capacity:
    return "capacity";
  case RequestedCompressionPolicy::both:
    return "both";
  }
  return "adaptive";
}

std::string_view path_name(const RequestedPath value) noexcept {
  switch (value) {
  case RequestedPath::automatic:
    return "auto";
  case RequestedPath::raw:
    return "raw";
  case RequestedPath::cpu_lz4_gpu:
    return "cpu_lz4_gpu_decode";
  case RequestedPath::gpu_lz4:
    return "nvcomp_gpu_codec";
  case RequestedPath::all:
    return "all";
  }
  return "auto";
}

std::string_view replacement_policy_name(const RequestedReplacementPolicy value) noexcept {
  switch (value) {
  case RequestedReplacementPolicy::clock:
    return "clock";
  case RequestedReplacementPolicy::lru:
    return "lru";
  case RequestedReplacementPolicy::both:
    return "both";
  }
  return "clock";
}

std::string_view scenario_name(const ScenarioKind value) noexcept {
  switch (value) {
  case ScenarioKind::suite:
    return "suite";
  case ScenarioKind::compressible_read:
    return "compressible-read";
  case ScenarioKind::incompressible_read:
    return "incompressible-read";
  case ScenarioKind::reuse:
    return "reuse";
  case ScenarioKind::dirty_writeback:
    return "dirty-writeback";
  case ScenarioKind::mixed:
    return "mixed";
  case ScenarioKind::budget_pressure:
    return "budget-pressure";
  }
  return "suite";
}

void print_help(std::ostream& output) {
  output << R"(xvram-compression-bench - adaptive lossless backing benchmark

Usage:
  xvram-compression-bench [options]

Options:
  --device <ordinal>                       CUDA device ordinal (default: 0)
  --logical-size <auto|size>               Logical heap size (default: auto, raw-safe 1.5x)
  --chunk-size <size>                      Residency chunk size (default: 64 MiB)
  --cache-target <auto|size>               Maximum VRAM cache target (default: auto)
  --compression-policy <adaptive|capacity|both>
                                             Host compression policy (default: adaptive)
  --path <auto|raw|cpu-lz4-gpu|gpu-lz4|all>
                                             Codec/transfer path (default: auto)
  --codec <auto|lz4>                       Lossless codec (default: auto)
  --policy <clock|lru|both>                Replacement policy (default: clock)
  --scenario <suite|compressible-read|incompressible-read|reuse|dirty-writeback|mixed|budget-pressure>
                                             Workload scenario (default: suite)
  --passes <2..8>                          Ordinary scenario passes (default: 2)
  --warmup-passes <0..8>                   Compression warmup passes (default: 2)
  --measurement-passes <1..16>             Measured passes (default: 5)
  --host-store-cap <auto|size>             Authoritative host-store cap (default: auto)
  --host-headroom <auto|size>              Reserved host-memory headroom (default: auto)
  --device-headroom <size>                 Reserved CUDA/WDDM headroom (default: 512 MiB)
  --compression-scratch-cap <size>         Codec workspace cap (default: 256 MiB)
  --codec-slots <2..8>                     Generation-safe codec slots (default: 2)
  --codec-workers <1..8>                   CPU codec workers (default: 2)
  --staging-slots <2..8>                   Pinned transfer slots (default: 4)
  --prefetch-distance <0..8>               Bounded prefetch distance (default: 2)
  --budget-poll-ms <n>                     Live budget sampling interval (default: 100)
  --stall-timeout-ms <n>                   CUDA polling deadline (default: 5000)
  --timeout-seconds <n>                    Controller deadline (default: 900)
  --seed <hex-u64>                         Deterministic workload seed
  --trace <path>                           Write xvram.compression_trace v1 JSONL
  --json <path|->                          Write xvram.adaptive_compression v1 JSON
  --compact-json                           Disable JSON indentation
  --no-text                                Suppress human-readable output
  --include-identifiers                    Include stable UUID/LUID/PCI identifiers
  --version                                Print xVRAM version
  -h, --help                               Show this help

The public process owns all files and isolates CUDA work in a worker process. It never changes
TDR, WDDM, registry, or NVIDIA system settings. Explicit logical sizes have no multiplier clamp.
)";
}

bool parse_cli(const int argc, char** argv, CliOptions& options, std::string& error) {
  std::vector<std::string_view> arguments;
  arguments.reserve(static_cast<std::size_t>(argc > 0 ? argc - 1 : 0));
  for (int index = 1; index < argc; ++index) {
    arguments.emplace_back(argv[index]);
  }
  const auto value_after = [&](std::size_t& index,
                               const std::string_view option) -> std::optional<std::string_view> {
    if (index + 1U >= arguments.size()) {
      error = std::string(option) + " requires a value";
      return std::nullopt;
    }
    ++index;
    return arguments[index];
  };

  for (std::size_t index = 0; index < arguments.size(); ++index) {
    const std::string_view argument = arguments[index];
    if (argument == "-h" || argument == "--help") {
      options.action = CliAction::help;
      continue;
    }
    if (argument == "--version") {
      options.action = CliAction::version;
      continue;
    }
    if (argument == "--worker") {
      options.worker = true;
      continue;
    }
    if (argument == "--trace-enabled") {
      options.executor.trace_enabled = true;
      continue;
    }
    if (argument == "--compact-json") {
      options.pretty_json = false;
      continue;
    }
    if (argument == "--no-text") {
      options.print_text = false;
      continue;
    }
    if (argument == "--include-identifiers") {
      options.executor.include_identifiers = true;
      continue;
    }
    if (argument == "--device") {
      const auto value = value_after(index, argument);
      if (!value.has_value() || !parse_i32(*value, options.executor.device_ordinal) ||
          options.executor.device_ordinal < 0) {
        if (error.empty()) {
          error = "--device must be a non-negative integer";
        }
        return false;
      }
      continue;
    }
    if (argument == "--logical-size" || argument == "--cache-target" ||
        argument == "--host-store-cap" || argument == "--host-headroom") {
      const auto value = value_after(index, argument);
      if (!value.has_value()) {
        return false;
      }
      std::optional<std::uint64_t>* destination = nullptr;
      if (argument == "--logical-size") {
        destination = &options.executor.logical_bytes;
      } else if (argument == "--cache-target") {
        destination = &options.executor.cache_target_bytes;
      } else if (argument == "--host-store-cap") {
        destination = &options.executor.host_store_cap_bytes;
      } else {
        destination = &options.executor.host_headroom_bytes;
      }
      if (*value == "auto") {
        destination->reset();
      } else {
        const auto parsed = positive_size(*value);
        if (!parsed.has_value()) {
          error = std::string(argument) + " must be auto or a positive size";
          return false;
        }
        *destination = *parsed;
      }
      continue;
    }
    if (argument == "--chunk-size" || argument == "--device-headroom" ||
        argument == "--compression-scratch-cap") {
      const auto value = value_after(index, argument);
      const auto parsed = value.has_value() ? positive_size(*value) : std::nullopt;
      if (!parsed.has_value()) {
        if (error.empty()) {
          error = std::string(argument) + " must be a positive size";
        }
        return false;
      }
      if (argument == "--chunk-size") {
        options.executor.chunk_bytes = *parsed;
      } else if (argument == "--device-headroom") {
        options.executor.device_headroom_bytes = *parsed;
      } else {
        options.executor.compression_scratch_cap_bytes = *parsed;
      }
      continue;
    }
    if (argument == "--compression-policy") {
      const auto value = value_after(index, argument);
      if (!value.has_value()) {
        return false;
      }
      if (*value == "adaptive") {
        options.executor.compression_policy = RequestedCompressionPolicy::adaptive;
      } else if (*value == "capacity") {
        options.executor.compression_policy = RequestedCompressionPolicy::capacity;
      } else if (*value == "both") {
        options.executor.compression_policy = RequestedCompressionPolicy::both;
      } else {
        error = "--compression-policy must be adaptive, capacity, or both";
        return false;
      }
      continue;
    }
    if (argument == "--path") {
      const auto value = value_after(index, argument);
      if (!value.has_value()) {
        return false;
      }
      if (*value == "auto") {
        options.executor.path = RequestedPath::automatic;
      } else if (*value == "raw") {
        options.executor.path = RequestedPath::raw;
      } else if (*value == "cpu-lz4-gpu" || *value == "cpu_lz4_gpu_decode") {
        options.executor.path = RequestedPath::cpu_lz4_gpu;
      } else if (*value == "gpu-lz4" || *value == "nvcomp_gpu_codec") {
        options.executor.path = RequestedPath::gpu_lz4;
      } else if (*value == "all") {
        options.executor.path = RequestedPath::all;
      } else {
        error = "--path must be auto, raw, cpu-lz4-gpu, gpu-lz4, or all";
        return false;
      }
      continue;
    }
    if (argument == "--codec") {
      const auto value = value_after(index, argument);
      if (!value.has_value()) {
        return false;
      }
      if (*value == "auto") {
        options.executor.codec = residency::CompressionCodec::automatic;
      } else if (*value == "lz4") {
        options.executor.codec = residency::CompressionCodec::lz4;
      } else {
        error = "--codec must be auto or lz4";
        return false;
      }
      continue;
    }
    if (argument == "--policy") {
      const auto value = value_after(index, argument);
      if (!value.has_value()) {
        return false;
      }
      if (*value == "clock") {
        options.executor.replacement_policy = RequestedReplacementPolicy::clock;
      } else if (*value == "lru") {
        options.executor.replacement_policy = RequestedReplacementPolicy::lru;
      } else if (*value == "both") {
        options.executor.replacement_policy = RequestedReplacementPolicy::both;
      } else {
        error = "--policy must be clock, lru, or both";
        return false;
      }
      continue;
    }
    if (argument == "--scenario") {
      const auto value = value_after(index, argument);
      if (!value.has_value()) {
        return false;
      }
      if (*value == "suite") {
        options.executor.scenario = ScenarioKind::suite;
      } else if (*value == "compressible-read") {
        options.executor.scenario = ScenarioKind::compressible_read;
      } else if (*value == "incompressible-read") {
        options.executor.scenario = ScenarioKind::incompressible_read;
      } else if (*value == "reuse") {
        options.executor.scenario = ScenarioKind::reuse;
      } else if (*value == "dirty-writeback") {
        options.executor.scenario = ScenarioKind::dirty_writeback;
      } else if (*value == "mixed") {
        options.executor.scenario = ScenarioKind::mixed;
      } else if (*value == "budget-pressure") {
        options.executor.scenario = ScenarioKind::budget_pressure;
      } else {
        error = "--scenario is not a supported compression workload";
        return false;
      }
      continue;
    }
    if (argument == "--passes" || argument == "--warmup-passes" ||
        argument == "--measurement-passes" || argument == "--codec-slots" ||
        argument == "--codec-workers" || argument == "--staging-slots" ||
        argument == "--prefetch-distance" || argument == "--budget-poll-ms" ||
        argument == "--stall-timeout-ms") {
      const auto value = value_after(index, argument);
      std::uint32_t parsed = 0;
      if (!value.has_value() || !parse_u32(*value, parsed)) {
        if (error.empty()) {
          error = std::string(argument) + " must be an integer";
        }
        return false;
      }
      if (argument == "--passes") {
        if (parsed < 2U || parsed > 8U) {
          error = "--passes must be between 2 and 8";
          return false;
        }
        options.executor.passes = parsed;
      } else if (argument == "--warmup-passes") {
        if (parsed > 8U) {
          error = "--warmup-passes must be between 0 and 8";
          return false;
        }
        options.executor.warmup_passes = parsed;
      } else if (argument == "--measurement-passes") {
        if (parsed == 0U || parsed > 16U) {
          error = "--measurement-passes must be between 1 and 16";
          return false;
        }
        options.executor.measurement_passes = parsed;
      } else if (argument == "--codec-slots") {
        if (parsed < 2U || parsed > 8U) {
          error = "--codec-slots must be between 2 and 8";
          return false;
        }
        options.executor.codec_slots = parsed;
      } else if (argument == "--codec-workers") {
        if (parsed == 0U || parsed > 8U) {
          error = "--codec-workers must be between 1 and 8";
          return false;
        }
        options.executor.codec_workers = parsed;
      } else if (argument == "--staging-slots") {
        if (parsed < 2U || parsed > 8U) {
          error = "--staging-slots must be between 2 and 8";
          return false;
        }
        options.executor.staging_slots = parsed;
      } else if (argument == "--prefetch-distance") {
        if (parsed > 8U) {
          error = "--prefetch-distance must be between 0 and 8";
          return false;
        }
        options.executor.prefetch_distance = parsed;
      } else if (argument == "--budget-poll-ms") {
        if (parsed == 0U) {
          error = "--budget-poll-ms must be positive";
          return false;
        }
        options.executor.budget_poll_interval = std::chrono::milliseconds(parsed);
      } else {
        if (parsed == 0U) {
          error = "--stall-timeout-ms must be positive";
          return false;
        }
        options.executor.stall_timeout = std::chrono::milliseconds(parsed);
      }
      continue;
    }
    if (argument == "--timeout-seconds") {
      const auto value = value_after(index, argument);
      std::uint64_t parsed = 0;
      if (!value.has_value() || !parse_u64(*value, parsed) || parsed == 0U ||
          parsed > static_cast<std::uint64_t>(std::chrono::seconds::max().count()) ||
          parsed > static_cast<std::uint64_t>(std::chrono::milliseconds::max().count()) / 1000ULL) {
        if (error.empty()) {
          error = "--timeout-seconds must be a positive duration";
        }
        return false;
      }
      options.timeout = std::chrono::seconds(parsed);
      continue;
    }
    if (argument == "--seed") {
      const auto value = value_after(index, argument);
      if (!value.has_value() || !parse_seed(*value, options.executor.seed)) {
        if (error.empty()) {
          error = "--seed must be a hexadecimal uint64 value";
        }
        return false;
      }
      continue;
    }
    if (argument == "--trace" || argument == "--json") {
      const auto value = value_after(index, argument);
      if (!value.has_value() || value->empty()) {
        if (error.empty()) {
          error = std::string(argument) + " requires a non-empty path";
        }
        return false;
      }
      if (argument == "--trace") {
        if (*value == "-") {
          error = "--trace requires a file path; stdout is reserved for report output";
          return false;
        }
        options.trace_path = std::string(*value);
        options.executor.trace_enabled = true;
      } else {
        options.json_path = std::string(*value);
      }
      continue;
    }
    error = "unknown argument: " + std::string(argument);
    return false;
  }

  if (options.action != CliAction::run && arguments.size() != 1U) {
    error = "--help and --version cannot be combined with execution options";
    return false;
  }
  if (!options.print_text && !options.json_path.has_value() && !options.worker) {
    error = "--no-text requires --json";
    return false;
  }
  if (options.executor.trace_enabled && !options.trace_path.has_value() && !options.worker) {
    error = "--trace-enabled is reserved for the isolated worker";
    return false;
  }
  if (options.trace_path.has_value() && options.json_path.has_value() &&
      *options.json_path != "-" && paths_collide(*options.trace_path, *options.json_path)) {
    error = "--trace and --json must name different files";
    return false;
  }
  return true;
}

std::vector<std::string> worker_arguments(const CliOptions& options) {
  const ExecutorOptions& value = options.executor;
  std::vector<std::string> output{
      "--worker",
      "--device",
      std::to_string(value.device_ordinal),
      "--logical-size",
      size_argument(value.logical_bytes),
      "--chunk-size",
      size_argument(value.chunk_bytes),
      "--cache-target",
      size_argument(value.cache_target_bytes),
      "--compression-policy",
      std::string(compression_policy_name(value.compression_policy)),
      "--path",
      std::string(path_name(value.path)),
      "--codec",
      std::string(residency::compression_codec_name(value.codec)),
      "--policy",
      std::string(replacement_policy_name(value.replacement_policy)),
      "--scenario",
      std::string(scenario_name(value.scenario)),
      "--passes",
      std::to_string(value.passes),
      "--warmup-passes",
      std::to_string(value.warmup_passes),
      "--measurement-passes",
      std::to_string(value.measurement_passes),
      "--host-store-cap",
      size_argument(value.host_store_cap_bytes),
      "--host-headroom",
      size_argument(value.host_headroom_bytes),
      "--device-headroom",
      size_argument(value.device_headroom_bytes),
      "--compression-scratch-cap",
      size_argument(value.compression_scratch_cap_bytes),
      "--codec-slots",
      std::to_string(value.codec_slots),
      "--codec-workers",
      std::to_string(value.codec_workers),
      "--staging-slots",
      std::to_string(value.staging_slots),
      "--prefetch-distance",
      std::to_string(value.prefetch_distance),
      "--budget-poll-ms",
      std::to_string(value.budget_poll_interval.count()),
      "--stall-timeout-ms",
      std::to_string(value.stall_timeout.count()),
      "--timeout-seconds",
      std::to_string(options.timeout.count()),
      "--seed",
      seed_argument(value.seed),
  };
  if (!options.pretty_json) {
    output.emplace_back("--compact-json");
  }
  if (!options.print_text) {
    output.emplace_back("--no-text");
  }
  if (value.include_identifiers) {
    output.emplace_back("--include-identifiers");
  }
  if (value.trace_enabled) {
    output.emplace_back("--trace-enabled");
  }
  return output;
}

std::filesystem::path current_executable_path(const char* argv_zero) {
#ifdef _WIN32
  std::vector<wchar_t> module_path(32'768U, L'\0');
  const DWORD length =
      GetModuleFileNameW(nullptr, module_path.data(), static_cast<DWORD>(module_path.size()));
  if (length != 0U && length < module_path.size()) {
    return std::filesystem::path(std::wstring(module_path.data(), length));
  }
#elif defined(__linux__)
  std::vector<char> module_path(4'096U, '\0');
  while (module_path.size() <= 1024U * 1024U) {
    const ssize_t length = readlink("/proc/self/exe", module_path.data(), module_path.size());
    if (length >= 0 && static_cast<std::size_t>(length) < module_path.size()) {
      return std::filesystem::path(
          std::string(module_path.data(), static_cast<std::size_t>(length)));
    }
    if (length < 0) {
      break;
    }
    module_path.resize(module_path.size() * 2U);
  }
#endif
  if (argv_zero == nullptr || *argv_zero == '\0') {
    return {};
  }
  std::error_code error;
  const std::filesystem::path requested(argv_zero);
  if (requested.is_absolute() || requested.has_parent_path()) {
    const auto absolute = std::filesystem::absolute(requested, error);
    return error ? std::filesystem::path{} : absolute;
  }
#ifdef _WIN32
  char* path_value = nullptr;
  std::size_t path_value_size = 0;
  if (_dupenv_s(&path_value, &path_value_size, "PATH") != 0 || path_value == nullptr) {
    return {};
  }
  const std::string owned_search_path(path_value, path_value_size > 0U ? path_value_size - 1U : 0U);
  std::free(path_value);
  constexpr char separator = ';';
#else
  const char* path_value = std::getenv("PATH");
  if (path_value == nullptr) {
    return {};
  }
  const std::string owned_search_path(path_value);
  constexpr char separator = ':';
#endif
  const std::string_view path(owned_search_path);
  std::size_t begin = 0;
  while (begin <= path.size()) {
    const std::size_t end = path.find(separator, begin);
    const std::string_view entry =
        path.substr(begin, end == std::string_view::npos ? path.size() - begin : end - begin);
    const std::filesystem::path candidate =
        std::filesystem::path(entry.empty() ? "." : std::string(entry)) / requested;
    if (std::filesystem::is_regular_file(candidate, error) && !error) {
      const auto absolute = std::filesystem::absolute(candidate, error);
      if (!error) {
        return absolute;
      }
    }
    error.clear();
    if (end == std::string_view::npos) {
      break;
    }
    begin = end + 1U;
  }
  return {};
}

} // namespace xvram::compression
