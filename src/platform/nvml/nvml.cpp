#include "platform/nvml/nvml.hpp"

#include "platform/dynamic_library.hpp"

#include <nvml.h>

#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <ShlObj.h>
#include <Windows.h>
#endif

namespace xvram::nvml {
namespace {

using Init = nvmlReturn_t (*)(void);
using Shutdown = nvmlReturn_t (*)(void);
using SystemGetDriverVersion = nvmlReturn_t (*)(char*, unsigned int);
using DeviceGetHandleByPciBusId = nvmlReturn_t (*)(const char*, nvmlDevice_t*);
using DeviceGetDriverModel = nvmlReturn_t (*)(nvmlDevice_t, nvmlDriverModel_t*, nvmlDriverModel_t*);
using ErrorString = const char* (*)(nvmlReturn_t);

template <typename Function>
[[nodiscard]] Function resolve(const platform::DynamicLibrary& library, const char* name) {
  return library.symbol<Function>(name);
}

[[nodiscard]] std::string_view driver_model_name(const nvmlDriverModel_t model) {
  switch (model) {
  case NVML_DRIVER_WDDM:
    return "wddm";
  case NVML_DRIVER_WDM:
    return "tcc";
  case NVML_DRIVER_MCDM:
    return "mcdm";
  }
  return "unknown";
}

#ifdef _WIN32
[[nodiscard]] bool open_nvml(platform::DynamicLibrary& library) {
  if (library.open_system({"nvml.dll"})) {
    return true;
  }

  PWSTR program_files = nullptr;
  const HRESULT result =
      SHGetKnownFolderPath(FOLDERID_ProgramFiles, KF_FLAG_DEFAULT, nullptr, &program_files);
  if (FAILED(result) || program_files == nullptr) {
    if (program_files != nullptr) {
      CoTaskMemFree(program_files);
    }
    return false;
  }
  const std::filesystem::path path =
      std::filesystem::path(program_files) / L"NVIDIA Corporation" / L"NVSMI" / L"nvml.dll";
  CoTaskMemFree(program_files);
  return library.open_absolute(path);
}
#endif

} // namespace

void enrich(probe::CudaReport& report, std::vector<probe::Diagnostic>& diagnostics) {
  platform::DynamicLibrary library;
#ifdef _WIN32
  report.nvml_library_loaded = open_nvml(library);
#else
  report.nvml_library_loaded = library.open_system({"libnvidia-ml.so.1", "libnvidia-ml.so"});
#endif
  if (!report.nvml_library_loaded) {
    diagnostics.push_back({probe::DiagnosticLevel::info, "nvml", "load_library",
                           "NVML is unavailable: " + library.error(), std::nullopt});
    return;
  }

  const Init init = resolve<Init>(library, "nvmlInit_v2");
  const Shutdown shutdown = resolve<Shutdown>(library, "nvmlShutdown");
  const SystemGetDriverVersion get_driver_version =
      resolve<SystemGetDriverVersion>(library, "nvmlSystemGetDriverVersion");
  DeviceGetHandleByPciBusId get_device =
      resolve<DeviceGetHandleByPciBusId>(library, "nvmlDeviceGetHandleByPciBusId_v2");
  if (get_device == nullptr) {
    get_device = resolve<DeviceGetHandleByPciBusId>(library, "nvmlDeviceGetHandleByPciBusId");
  }
  DeviceGetDriverModel get_driver_model =
      resolve<DeviceGetDriverModel>(library, "nvmlDeviceGetDriverModel_v2");
  if (get_driver_model == nullptr) {
    get_driver_model = resolve<DeviceGetDriverModel>(library, "nvmlDeviceGetDriverModel");
  }
  const ErrorString error_string = resolve<ErrorString>(library, "nvmlErrorString");

  if (init == nullptr || shutdown == nullptr || get_driver_version == nullptr) {
    diagnostics.push_back({probe::DiagnosticLevel::warning, "nvml", "resolve_symbols",
                           "NVML is missing baseline entry points", std::nullopt});
    return;
  }

  const nvmlReturn_t initialization = init();
  if (initialization != NVML_SUCCESS) {
    const char* message = error_string != nullptr ? error_string(initialization) : nullptr;
    diagnostics.push_back({probe::DiagnosticLevel::warning, "nvml", "nvmlInit_v2",
                           message != nullptr ? message : "NVML initialization failed",
                           static_cast<std::int64_t>(initialization)});
    return;
  }

  std::array<char, NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE> version{};
  const nvmlReturn_t driver_version_result =
      get_driver_version(version.data(), static_cast<unsigned int>(version.size()));
  if (driver_version_result == NVML_SUCCESS) {
    report.display_driver_version = version.data();
  } else {
    const char* message = error_string != nullptr ? error_string(driver_version_result) : nullptr;
    diagnostics.push_back({probe::DiagnosticLevel::info, "nvml", "nvmlSystemGetDriverVersion",
                           message != nullptr ? message : "NVML driver version query failed",
                           static_cast<std::int64_t>(driver_version_result)});
  }

  if (get_device != nullptr && get_driver_model != nullptr) {
    for (probe::DeviceReport& device : report.devices) {
      if (!device.pci_bus_id.has_value()) {
        continue;
      }
      nvmlDevice_t nvml_device = nullptr;
      const nvmlReturn_t get_device_result = get_device(device.pci_bus_id->c_str(), &nvml_device);
      if (get_device_result != NVML_SUCCESS) {
        const char* message = error_string != nullptr ? error_string(get_device_result) : nullptr;
        diagnostics.push_back({probe::DiagnosticLevel::info, "nvml",
                               "nvmlDeviceGetHandleByPciBusId_v2",
                               message != nullptr ? message : "NVML device correlation failed",
                               static_cast<std::int64_t>(get_device_result), device.ordinal});
        continue;
      }
      nvmlDriverModel_t current = NVML_DRIVER_WDDM;
      nvmlDriverModel_t pending = NVML_DRIVER_WDDM;
      const nvmlReturn_t driver_model_result = get_driver_model(nvml_device, &current, &pending);
      if (driver_model_result == NVML_SUCCESS) {
        device.driver_model = driver_model_name(current);
        device.pending_driver_model = driver_model_name(pending);
      } else {
        const char* message = error_string != nullptr ? error_string(driver_model_result) : nullptr;
        diagnostics.push_back({probe::DiagnosticLevel::info, "nvml", "nvmlDeviceGetDriverModel_v2",
                               message != nullptr ? message : "NVML driver model query failed",
                               static_cast<std::int64_t>(driver_model_result), device.ordinal});
      }
    }
  }

  const nvmlReturn_t shutdown_result = shutdown();
  if (shutdown_result != NVML_SUCCESS) {
    const char* message = error_string != nullptr ? error_string(shutdown_result) : nullptr;
    diagnostics.push_back({probe::DiagnosticLevel::warning, "nvml", "nvmlShutdown",
                           message != nullptr ? message : "NVML shutdown failed",
                           static_cast<std::int64_t>(shutdown_result)});
  }
}

} // namespace xvram::nvml
