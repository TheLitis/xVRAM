#include "xvram/probe/report.hpp"

#include "platform/cuda/cuda_driver.hpp"
#include "platform/nvml/nvml.hpp"
#include "platform/system_info.hpp"
#include "probe/privacy.hpp"
#include "xvram/version.hpp"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <utility>

namespace xvram::probe {
namespace {

[[nodiscard]] std::string utc_timestamp() {
  const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::tm utc{};
#ifdef _WIN32
  gmtime_s(&utc, &now);
#else
  gmtime_r(&now, &utc);
#endif
  std::ostringstream output;
  output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
  return output.str();
}

[[nodiscard]] std::string compiler_description() {
#if defined(_MSC_VER)
  return "MSVC " + std::to_string(_MSC_VER);
#elif defined(__clang__)
  return "Clang " + std::to_string(__clang_major__) + "." + std::to_string(__clang_minor__) + "." +
         std::to_string(__clang_patchlevel__);
#elif defined(__GNUC__)
  return "GCC " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__) + "." +
         std::to_string(__GNUC_PATCHLEVEL__);
#else
  return "unknown";
#endif
}

[[nodiscard]] std::string build_type() {
#ifdef NDEBUG
  return "Release";
#else
  return "Debug";
#endif
}

} // namespace

void apply_identifier_policy(ProbeReport& report, const bool include_stable_identifiers) {
  if (include_stable_identifiers) {
    return;
  }

  for (DeviceReport& device : report.cuda.devices) {
    device.uuid.reset();
    device.luid.reset();
    device.pci_bus_id.reset();
    device.attributes.erase("pci_domain_id");
    device.attributes.erase("pci_bus_id");
    device.attributes.erase("pci_device_id");
    if (device.dxgi.has_value()) {
      device.dxgi->luid.clear();
    }
  }
}

ProbeReport collect(const ProbeOptions& options) {
  ProbeReport report;
  report.generated_at_utc = utc_timestamp();
  report.build = {XVRAM_VERSION, XVRAM_GIT_COMMIT, compiler_description(), build_type(),
                  XVRAM_CUDA_HEADERS_VERSION};
  report.system = platform::collect_system_info();

  cuda::Driver driver;
  driver.collect(report.cuda, report.diagnostics, options);
  nvml::enrich(report.cuda, report.diagnostics);
  apply_identifier_policy(report, options.include_stable_identifiers);

  if (options.run_transfer_benchmark) {
    if (report.cuda.devices.empty()) {
      report.transfer_benchmark = TransferMeasurement{"skipped",
                                                      std::nullopt,
                                                      0,
                                                      options.benchmark_iterations,
                                                      std::nullopt,
                                                      std::nullopt,
                                                      std::nullopt,
                                                      "No selected CUDA device is available"};
    } else {
      const std::size_t benchmark_diagnostics_begin = report.diagnostics.size();
      report.transfer_benchmark = driver.benchmark_transfers(report.cuda.devices.front().ordinal,
                                                             options, report.diagnostics);
      for (std::size_t index = benchmark_diagnostics_begin; index < report.diagnostics.size();
           ++index) {
        if (!report.diagnostics[index].device_ordinal.has_value()) {
          report.diagnostics[index].device_ordinal = report.cuda.devices.front().ordinal;
        }
      }
    }
  }

  if (options.run_overlap_benchmark) {
    if (report.cuda.devices.empty()) {
      OverlapMeasurement skipped;
      skipped.status = "skipped";
      skipped.samples = options.overlap_samples;
      skipped.target_compute_ms = options.overlap_target_compute_ms;
      skipped.message = "No selected CUDA device is available";
      report.overlap_benchmark = std::move(skipped);
    } else {
      const std::size_t benchmark_diagnostics_begin = report.diagnostics.size();
      report.overlap_benchmark = driver.benchmark_overlap(report.cuda.devices.front().ordinal,
                                                          options, report.diagnostics);
      for (std::size_t index = benchmark_diagnostics_begin; index < report.diagnostics.size();
           ++index) {
        if (!report.diagnostics[index].device_ordinal.has_value()) {
          report.diagnostics[index].device_ordinal = report.cuda.devices.front().ordinal;
        }
      }
    }
  }
  return report;
}

} // namespace xvram::probe
