#include "platform/cuda/cuda_api.hpp"

namespace xvram::cuda {

CudaApi::LoadResult CudaApi::load() {
  if (load_attempted_) {
    return {load_status_, false};
  }
  load_attempted_ = true;

#ifdef _WIN32
  const bool library_loaded = library_.open_system({"nvcuda.dll"});
#else
  const bool library_loaded = library_.open_system({"libcuda.so.1", "libcuda.so"});
#endif
  if (!library_loaded) {
    load_status_ = LoadStatus::library_unavailable;
    return {load_status_, true};
  }

  const bool required =
      resolve(init_, {"cuInit"}) && resolve(driver_get_version_, {"cuDriverGetVersion"}) &&
      resolve(device_get_count_, {"cuDeviceGetCount"}) && resolve(device_get_, {"cuDeviceGet"}) &&
      resolve(device_get_name_, {"cuDeviceGetName"}) &&
      resolve(device_total_memory_, {"cuDeviceTotalMem_v2"}) &&
      resolve(device_get_attribute_, {"cuDeviceGetAttribute"});

  resolve(get_error_name_, {"cuGetErrorName"});
  resolve(get_error_string_, {"cuGetErrorString"});
  resolve(device_get_uuid_, {"cuDeviceGetUuid_v2", "cuDeviceGetUuid"});
  resolve(device_get_luid_, {"cuDeviceGetLuid"});
  resolve(device_get_pci_bus_id_, {"cuDeviceGetPCIBusId"});
  resolve(mem_get_allocation_granularity_, {"cuMemGetAllocationGranularity"});
  resolve(context_get_current_, {"cuCtxGetCurrent"});
  resolve(context_get_device_, {"cuCtxGetDevice"});
  resolve(context_set_current_, {"cuCtxSetCurrent"});
  resolve(context_push_current_, {"cuCtxPushCurrent_v2", "cuCtxPushCurrent"});
  resolve(context_pop_current_, {"cuCtxPopCurrent_v2", "cuCtxPopCurrent"});
  resolve(context_create_, {"cuCtxCreate_v2"});
  resolve(context_destroy_, {"cuCtxDestroy_v2"});
  resolve(mem_get_info_, {"cuMemGetInfo_v2"});
  resolve(mem_address_reserve_, {"cuMemAddressReserve"});
  resolve(mem_address_free_, {"cuMemAddressFree"});
  resolve(mem_create_, {"cuMemCreate"});
  resolve(mem_release_, {"cuMemRelease"});
  resolve(mem_map_, {"cuMemMap"});
  resolve(mem_unmap_, {"cuMemUnmap"});
  resolve(mem_set_access_, {"cuMemSetAccess"});
  resolve(mem_get_access_, {"cuMemGetAccess"});
  resolve(mem_alloc_, {"cuMemAlloc_v2"});
  resolve(mem_free_, {"cuMemFree_v2"});
  resolve(mem_host_alloc_, {"cuMemHostAlloc"});
  resolve(mem_free_host_, {"cuMemFreeHost"});
  resolve(memcpy_h2d_async_, {"cuMemcpyHtoDAsync_v2"});
  resolve(memcpy_d2h_async_, {"cuMemcpyDtoHAsync_v2"});
  resolve(stream_create_, {"cuStreamCreate"});
  // These legacy exports are retained only where the official signatures are ABI-identical.
  resolve(stream_destroy_, {"cuStreamDestroy_v2", "cuStreamDestroy"});
  resolve(stream_synchronize_, {"cuStreamSynchronize"});
  resolve(stream_wait_event_, {"cuStreamWaitEvent"});
  resolve(event_create_, {"cuEventCreate"});
  resolve(event_destroy_, {"cuEventDestroy_v2", "cuEventDestroy"});
  resolve(event_record_, {"cuEventRecord"});
  resolve(event_query_, {"cuEventQuery"});
  resolve(event_synchronize_, {"cuEventSynchronize"});
  resolve(event_elapsed_time_, {"cuEventElapsedTime_v2", "cuEventElapsedTime"});
  resolve(module_load_data_, {"cuModuleLoadData"});
  resolve(module_get_function_, {"cuModuleGetFunction"});
  resolve(module_unload_, {"cuModuleUnload"});
  resolve(launch_kernel_, {"cuLaunchKernel"});

  load_status_ = required ? LoadStatus::loaded : LoadStatus::baseline_symbols_missing;
  return {load_status_, true};
}

const std::string& CudaApi::loaded_name() const noexcept {
  return library_.loaded_name();
}

const std::string& CudaApi::error() const noexcept {
  return library_.error();
}

void CudaApi::abandon() noexcept {
  library_.abandon();
}

} // namespace xvram::cuda
