#pragma once

#include "platform/cuda/cuda_abi.hpp"
#include "platform/dynamic_library.hpp"

#include <initializer_list>
#include <string>

namespace xvram::cuda {

class CudaApi {
public:
  enum class LoadStatus { loaded, library_unavailable, baseline_symbols_missing };
  struct InjectedDispatch {};

  struct LoadResult {
    LoadStatus status = LoadStatus::library_unavailable;
    bool attempted_now = false;
  };

  CudaApi() = default;
  explicit CudaApi(InjectedDispatch) noexcept
      : load_status_(LoadStatus::loaded), load_attempted_(true) {}

  [[nodiscard]] LoadResult load();
  [[nodiscard]] const std::string& loaded_name() const noexcept;
  [[nodiscard]] const std::string& error() const noexcept;
  void abandon() noexcept;

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
  abi::ContextGetDevice context_get_device_ = nullptr;
  abi::ContextSetCurrent context_set_current_ = nullptr;
  abi::ContextPushCurrent context_push_current_ = nullptr;
  abi::ContextPopCurrent context_pop_current_ = nullptr;
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
  abi::MemGetAccess mem_get_access_ = nullptr;
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
  abi::StreamIsCapturing stream_is_capturing_ = nullptr;
  abi::EventCreate event_create_ = nullptr;
  abi::EventDestroy event_destroy_ = nullptr;
  abi::EventRecord event_record_ = nullptr;
  abi::EventQuery event_query_ = nullptr;
  abi::EventSynchronize event_synchronize_ = nullptr;
  abi::EventElapsedTime event_elapsed_time_ = nullptr;
  abi::ModuleLoadData module_load_data_ = nullptr;
  abi::ModuleGetFunction module_get_function_ = nullptr;
  abi::ModuleUnload module_unload_ = nullptr;
  abi::LaunchKernel launch_kernel_ = nullptr;

private:
  template <typename Function>
  bool resolve(Function& output, const std::initializer_list<const char*> names) {
    for (const char* name : names) {
      if (Function candidate = library_.symbol<Function>(name); candidate != nullptr) {
        output = candidate;
        return true;
      }
    }
    return false;
  }

  platform::DynamicLibrary library_;
  LoadStatus load_status_ = LoadStatus::library_unavailable;
  bool load_attempted_ = false;
};

} // namespace xvram::cuda
