#pragma once

#include "platform/cuda/cuda_abi.hpp"
#include "platform/dynamic_library.hpp"
#include "xvram/probe/report.hpp"

#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace xvram::cuda {

class Driver {
public:
  Driver() = default;

  void collect(probe::CudaReport& report, std::vector<probe::Diagnostic>& diagnostics,
               const probe::ProbeOptions& options);
  [[nodiscard]] probe::TransferMeasurement
  benchmark_transfers(std::int32_t ordinal, const probe::ProbeOptions& options,
                      std::vector<probe::Diagnostic>& diagnostics);
  [[nodiscard]] probe::OverlapMeasurement
  benchmark_overlap(std::int32_t ordinal, const probe::ProbeOptions& options,
                    std::vector<probe::Diagnostic>& diagnostics);

private:
  template <typename Function>
  bool resolve(Function& output, std::initializer_list<const char*> names) {
    for (const char* name : names) {
      if (Function candidate = library_.symbol<Function>(name); candidate != nullptr) {
        output = candidate;
        return true;
      }
    }
    return false;
  }

  [[nodiscard]] bool load(std::vector<probe::Diagnostic>& diagnostics);
  [[nodiscard]] std::string result_name(abi::Result result) const;
  [[nodiscard]] std::string result_message(abi::Result result) const;
  [[nodiscard]] std::optional<int>
  query_attribute(abi::Device device, abi::NativeDeviceAttribute attribute,
                  std::string_view report_key, std::vector<probe::Diagnostic>& diagnostics) const;
  [[nodiscard]] std::optional<std::uint64_t>
  query_free_memory(abi::Device device, std::vector<probe::Diagnostic>& diagnostics) const;
  [[nodiscard]] probe::GranularityInfo
  query_granularity(CUmemLocationType location_type, std::int32_t location_id,
                    std::string_view report_key, std::vector<probe::Diagnostic>& diagnostics) const;
  [[nodiscard]] probe::VmmSmokeResult smoke_test_vmm(abi::Device device, std::int32_t ordinal,
                                                     std::uint64_t allocation_bytes,
                                                     std::vector<probe::Diagnostic>& diagnostics);

  struct QuarantinedVmmState {
    std::int32_t device_ordinal{};
    abi::DevicePointer reservation{};
    abi::DevicePointer mapped_address{};
    std::size_t reservation_bytes{};
    std::size_t mapping_bytes{};
    std::optional<abi::GenericAllocationHandle> handle;
    bool mapping_active{};
    bool reservation_active{};
  };

  platform::DynamicLibrary library_;
  abi::Init init_ = nullptr;
  abi::DriverGetVersion driver_get_version_ = nullptr;
  abi::GetErrorName get_error_name_ = nullptr;
  abi::GetErrorString get_error_string_ = nullptr;
  abi::DeviceGetCount device_get_count_ = nullptr;
  abi::DeviceGet device_get_ = nullptr;
  abi::DeviceGetName device_get_name_ = nullptr;
  abi::DeviceTotalMem device_total_memory_ = nullptr;
  abi::DeviceGetAttribute device_get_attribute_ = nullptr;
  abi::DeviceGetUuid device_get_uuid_ = nullptr;
  abi::DeviceGetLuid device_get_luid_ = nullptr;
  abi::DeviceGetPciBusId device_get_pci_bus_id_ = nullptr;
  abi::MemGetAllocationGranularity mem_get_allocation_granularity_ = nullptr;
  abi::ContextGetCurrent context_get_current_ = nullptr;
  abi::ContextSetCurrent context_set_current_ = nullptr;
  abi::ContextCreate context_create_ = nullptr;
  abi::ContextDestroy context_destroy_ = nullptr;
  abi::MemGetInfo mem_get_info_ = nullptr;
  abi::MemAddressReserve mem_address_reserve_ = nullptr;
  abi::MemAddressFree mem_address_free_ = nullptr;
  abi::MemCreate mem_create_ = nullptr;
  abi::MemRelease mem_release_ = nullptr;
  abi::MemMap mem_map_ = nullptr;
  abi::MemUnmap mem_unmap_ = nullptr;
  abi::MemSetAccess mem_set_access_ = nullptr;
  abi::MemAlloc mem_alloc_ = nullptr;
  abi::MemFree mem_free_ = nullptr;
  abi::MemHostAlloc mem_host_alloc_ = nullptr;
  abi::MemFreeHost mem_free_host_ = nullptr;
  abi::MemcpyHtoDAsync memcpy_h2d_async_ = nullptr;
  abi::MemcpyDtoHAsync memcpy_d2h_async_ = nullptr;
  abi::StreamCreate stream_create_ = nullptr;
  abi::StreamDestroy stream_destroy_ = nullptr;
  abi::StreamSynchronize stream_synchronize_ = nullptr;
  abi::StreamWaitEvent stream_wait_event_ = nullptr;
  abi::EventCreate event_create_ = nullptr;
  abi::EventDestroy event_destroy_ = nullptr;
  abi::EventRecord event_record_ = nullptr;
  abi::EventSynchronize event_synchronize_ = nullptr;
  abi::EventElapsedTime event_elapsed_time_ = nullptr;
  abi::ModuleLoadData module_load_data_ = nullptr;
  abi::ModuleGetFunction module_get_function_ = nullptr;
  abi::ModuleUnload module_unload_ = nullptr;
  abi::LaunchKernel launch_kernel_ = nullptr;
  std::vector<QuarantinedVmmState> quarantined_vmm_;
  bool active_cuda_poisoned_ = false;
  bool load_attempted_ = false;
  bool loaded_ = false;
};

} // namespace xvram::cuda
