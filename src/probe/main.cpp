#include "xvram/base/size_parser.hpp"
#include "xvram/probe/report.hpp"
#include "xvram/version.hpp"

#include <charconv>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr int exit_success = 0;
constexpr int exit_requirement_not_met = 20;
constexpr int exit_probe_failure = 21;
constexpr int exit_usage = 64;
constexpr int exit_internal = 70;
constexpr int exit_io = 74;

struct CliOptions {
  xvram::probe::ProbeOptions probe;
  std::optional<std::string> json_path;
  bool pretty_json = true;
  bool print_text = true;
  bool require_device_vmm = false;
};

void print_help(std::ostream& output) {
  output << R"(xvram-probe - inspect CUDA VMM, host-memory, and WDDM capabilities

Usage:
  xvram-probe [collect] [options]

Options:
  --device <ordinal>          Probe one device; exit 21 when it is unavailable
  --json <path|->             Write the versioned JSON report ("-" is stdout)
  --compact-json              Disable JSON indentation
  --no-text                   Suppress the human-readable report
  --skip-vmm-smoke            Skip the small VMM map/remap verification
  --benchmark                 Run bounded pinned H2D/D2H/full-duplex measurements
  --benchmark-bytes <size>    Bytes per direction (default: 64 MiB; range: 8-128 MiB)
  --benchmark-iterations <n>  Measured transfer iterations (default: 20; max: 1000)
  --include-identifiers       Include stable GPU UUID/LUID values in output
  --require-device-vmm        Exit 20 unless the selected device passes VMM map/remap
  --version                   Print xVRAM version
  -h, --help                  Show this help

The default collection is safe and bounded. It uses short-lived isolated CUDA contexts;
it does not reset primary contexts or change driver settings. The benchmark runs only
when explicitly requested.
)";
}

[[nodiscard]] bool parse_integer(const std::string_view text, std::int32_t& output) {
  const auto result = std::from_chars(text.data(), text.data() + text.size(), output, 10);
  return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

[[nodiscard]] bool parse_unsigned(const std::string_view text, std::uint32_t& output) {
  const auto result = std::from_chars(text.data(), text.data() + text.size(), output, 10);
  return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

[[nodiscard]] std::optional<CliOptions> parse_cli(const int argc, char** argv, std::string& error) {
  CliOptions options;
  std::vector<std::string_view> arguments;
  arguments.reserve(static_cast<std::size_t>(argc > 0 ? argc - 1 : 0));
  for (int index = 1; index < argc; ++index) {
    arguments.emplace_back(argv[index]);
  }
  if (!arguments.empty() && arguments.front() == "collect") {
    arguments.erase(arguments.begin());
  }

  const auto value_after = [&](std::size_t& index,
                               const std::string_view option) -> std::optional<std::string_view> {
    if (index + 1 >= arguments.size()) {
      error = std::string(option) + " requires a value";
      return std::nullopt;
    }
    ++index;
    return arguments[index];
  };

  for (std::size_t index = 0; index < arguments.size(); ++index) {
    const std::string_view argument = arguments[index];
    if (argument == "-h" || argument == "--help") {
      print_help(std::cout);
      return std::nullopt;
    }
    if (argument == "--version") {
      std::cout << XVRAM_VERSION << '\n';
      return std::nullopt;
    }
    if (argument == "--device") {
      const auto value = value_after(index, argument);
      std::int32_t ordinal = -1;
      if (!value.has_value() || !parse_integer(*value, ordinal) || ordinal < 0) {
        if (error.empty()) {
          error = "--device must be a non-negative integer";
        }
        return std::nullopt;
      }
      options.probe.device_ordinal = ordinal;
    } else if (argument == "--json") {
      const auto value = value_after(index, argument);
      if (!value.has_value()) {
        return std::nullopt;
      }
      options.json_path = std::string(*value);
    } else if (argument == "--compact-json") {
      options.pretty_json = false;
    } else if (argument == "--no-text") {
      options.print_text = false;
    } else if (argument == "--skip-vmm-smoke") {
      options.probe.run_vmm_smoke = false;
    } else if (argument == "--benchmark") {
      options.probe.run_transfer_benchmark = true;
    } else if (argument == "--benchmark-bytes") {
      const auto value = value_after(index, argument);
      if (!value.has_value()) {
        return std::nullopt;
      }
      const xvram::ParsedSize parsed = xvram::parse_size(*value);
      constexpr std::uint64_t maximum = 128ULL * 1024ULL * 1024ULL;
      constexpr std::uint64_t minimum = 8ULL * 1024ULL * 1024ULL;
      if (!parsed || *parsed.bytes < minimum || *parsed.bytes > maximum) {
        error = parsed ? "--benchmark-bytes must be between 8 MiB and 128 MiB" : parsed.error;
        return std::nullopt;
      }
      options.probe.benchmark_bytes = *parsed.bytes;
    } else if (argument == "--benchmark-iterations") {
      const auto value = value_after(index, argument);
      std::uint32_t iterations = 0;
      if (!value.has_value() || !parse_unsigned(*value, iterations) || iterations == 0 ||
          iterations > 1000) {
        if (error.empty()) {
          error = "--benchmark-iterations must be between 1 and 1000";
        }
        return std::nullopt;
      }
      options.probe.benchmark_iterations = iterations;
    } else if (argument == "--include-identifiers") {
      options.probe.include_stable_identifiers = true;
    } else if (argument == "--require-device-vmm") {
      options.require_device_vmm = true;
    } else {
      error = "unknown argument: " + std::string(argument);
      return std::nullopt;
    }
  }

  if (!options.print_text && !options.json_path.has_value()) {
    error = "--no-text requires --json";
    return std::nullopt;
  }
  return options;
}

[[nodiscard]] bool has_required_device_vmm(const xvram::probe::ProbeReport& report) {
  for (const xvram::probe::DeviceReport& device : report.cuda.devices) {
    if (device.vmm_smoke.has_value() && device.vmm_smoke->status == "completed" &&
        device.vmm_smoke->initial_copy_verified.value_or(false) &&
        device.vmm_smoke->remap_copy_verified.value_or(false) &&
        device.vmm_smoke->cleanup_complete.value_or(false)) {
      return true;
    }
  }
  return false;
}

} // namespace

int main(const int argc, char** argv) {
  try {
    std::string parse_error;
    const std::optional<CliOptions> options = parse_cli(argc, argv, parse_error);
    if (!options.has_value()) {
      if (!parse_error.empty()) {
        std::cerr << "error: " << parse_error << "\n\n";
        print_help(std::cerr);
        return exit_usage;
      }
      return exit_success;
    }

    const xvram::probe::ProbeReport report = xvram::probe::collect(options->probe);

    if (options->json_path.has_value()) {
      if (*options->json_path == "-") {
        xvram::probe::write_json(report, std::cout, options->pretty_json);
        if (options->print_text) {
          xvram::probe::write_text(report, std::cerr);
        }
      } else {
        const std::filesystem::path path(*options->json_path);
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        if (!output) {
          std::cerr << "error: could not open JSON output: " << path.string() << '\n';
          return exit_io;
        }
        xvram::probe::write_json(report, output, options->pretty_json);
        if (!output) {
          std::cerr << "error: failed while writing JSON output: " << path.string() << '\n';
          return exit_io;
        }
        if (options->print_text) {
          xvram::probe::write_text(report, std::cout);
        }
      }
    } else if (options->print_text) {
      xvram::probe::write_text(report, std::cout);
    }

    if (options->probe.device_ordinal.has_value() && report.cuda.devices.empty()) {
      return exit_probe_failure;
    }
    if (options->require_device_vmm && !has_required_device_vmm(report)) {
      return exit_requirement_not_met;
    }
    return exit_success;
  } catch (const std::exception& exception) {
    std::cerr << "fatal: " << exception.what() << '\n';
    return exit_internal;
  } catch (...) {
    std::cerr << "fatal: unknown internal error\n";
    return exit_internal;
  }
}
