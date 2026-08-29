#include "platform/system_info.hpp"

#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <winternl.h>
#else
#include <sys/sysinfo.h>
#include <sys/utsname.h>
#endif

namespace xvram::platform {

probe::SystemInfo collect_system_info() {
  probe::SystemInfo info;
  info.logical_processor_count = std::thread::hardware_concurrency();

#ifdef _WIN32
  info.os_name = "Windows";

  using RtlGetVersionFunction = LONG(WINAPI*)(PRTL_OSVERSIONINFOW);
  if (HMODULE ntdll = GetModuleHandleW(L"ntdll.dll"); ntdll != nullptr) {
    const auto rtl_get_version =
        reinterpret_cast<RtlGetVersionFunction>(GetProcAddress(ntdll, "RtlGetVersion"));
    if (rtl_get_version != nullptr) {
      RTL_OSVERSIONINFOW version{};
      version.dwOSVersionInfoSize = sizeof(version);
      if (rtl_get_version(&version) == 0) {
        info.os_version = std::to_string(version.dwMajorVersion) + "." +
                          std::to_string(version.dwMinorVersion) + "." +
                          std::to_string(version.dwBuildNumber);
      }
    }
  }

  SYSTEM_INFO system{};
  GetNativeSystemInfo(&system);
  switch (system.wProcessorArchitecture) {
  case PROCESSOR_ARCHITECTURE_AMD64:
    info.architecture = "x86_64";
    break;
  case PROCESSOR_ARCHITECTURE_ARM64:
    info.architecture = "arm64";
    break;
  default:
    info.architecture = "unknown";
    break;
  }
  if (info.logical_processor_count == 0) {
    info.logical_processor_count = system.dwNumberOfProcessors;
  }

  MEMORYSTATUSEX memory{};
  memory.dwLength = sizeof(memory);
  if (GlobalMemoryStatusEx(&memory) != FALSE) {
    info.physical_memory_bytes = memory.ullTotalPhys;
    info.available_memory_bytes = memory.ullAvailPhys;
  }
#else
  utsname system{};
  if (uname(&system) == 0) {
    info.os_name = system.sysname;
    info.os_version = system.release;
    info.architecture = system.machine;
  } else {
    info.os_name = "Unix";
    info.architecture = "unknown";
  }

  struct sysinfo memory{};
  if (sysinfo(&memory) == 0) {
    info.physical_memory_bytes = static_cast<std::uint64_t>(memory.totalram) * memory.mem_unit;
    info.available_memory_bytes = static_cast<std::uint64_t>(memory.freeram) * memory.mem_unit;
  }
#endif

  return info;
}

} // namespace xvram::platform
