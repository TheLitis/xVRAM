#pragma once

// The CUDA ABI comes from NVIDIA's official headers. The probe loads the driver
// dynamically, but never re-declares versioned CUDA structs or enum values.
#include <cuda.h>

#include <cstddef>
#include <cstdint>

namespace xvram::cuda::abi {

using Result = CUresult;
using Device = CUdevice;
using DevicePointer = CUdeviceptr;
using Context = CUcontext;
using Stream = CUstream;
using Event = CUevent;
using Module = CUmodule;
using Function = CUfunction;
using Uuid = CUuuid;
using MemAllocationProp = CUmemAllocationProp;
using MemLocation = CUmemLocation;
using MemAccessFlags = CUmemAccess_flags;
using GenericAllocationHandle = CUmemGenericAllocationHandle;
using NativeDeviceAttribute = CUdevice_attribute;

inline constexpr Result success = CUDA_SUCCESS;
inline constexpr auto allocation_type_pinned = CU_MEM_ALLOCATION_TYPE_PINNED;
inline constexpr auto location_device = CU_MEM_LOCATION_TYPE_DEVICE;
inline constexpr auto location_host = CU_MEM_LOCATION_TYPE_HOST;
inline constexpr auto location_host_numa = CU_MEM_LOCATION_TYPE_HOST_NUMA;
inline constexpr auto granularity_minimum = CU_MEM_ALLOC_GRANULARITY_MINIMUM;
inline constexpr auto granularity_recommended = CU_MEM_ALLOC_GRANULARITY_RECOMMENDED;
inline constexpr auto mem_access_none = CU_MEM_ACCESS_FLAGS_PROT_NONE;
inline constexpr auto mem_access_read = CU_MEM_ACCESS_FLAGS_PROT_READ;
inline constexpr auto mem_access_read_write = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

namespace attributes {
inline constexpr auto max_threads_per_block = CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK;
inline constexpr auto warp_size = CU_DEVICE_ATTRIBUTE_WARP_SIZE;
inline constexpr auto clock_rate = CU_DEVICE_ATTRIBUTE_CLOCK_RATE;
inline constexpr auto multiprocessor_count = CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
inline constexpr auto integrated = CU_DEVICE_ATTRIBUTE_INTEGRATED;
inline constexpr auto can_map_host_memory = CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY;
inline constexpr auto compute_mode = CU_DEVICE_ATTRIBUTE_COMPUTE_MODE;
inline constexpr auto concurrent_kernels = CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS;
inline constexpr auto pci_bus_id = CU_DEVICE_ATTRIBUTE_PCI_BUS_ID;
inline constexpr auto pci_device_id = CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID;
inline constexpr auto tcc_driver = CU_DEVICE_ATTRIBUTE_TCC_DRIVER;
inline constexpr auto memory_clock_rate = CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE;
inline constexpr auto global_memory_bus_width = CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH;
inline constexpr auto l2_cache_size = CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE;
inline constexpr auto async_engine_count = CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT;
inline constexpr auto unified_addressing = CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING;
inline constexpr auto pci_domain_id = CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID;
inline constexpr auto compute_capability_major = CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR;
inline constexpr auto compute_capability_minor = CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR;
inline constexpr auto managed_memory = CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY;
inline constexpr auto host_native_atomic_supported =
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED;
inline constexpr auto pageable_memory_access = CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS;
inline constexpr auto concurrent_managed_access = CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS;
inline constexpr auto compute_preemption_supported =
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED;
inline constexpr auto can_use_host_pointer_for_registered_memory =
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM;
inline constexpr auto host_register_supported = CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED;
inline constexpr auto pageable_memory_access_uses_host_page_tables =
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES;
inline constexpr auto direct_managed_memory_access_from_host =
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST;
inline constexpr auto virtual_memory_management_supported =
    CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED;
inline constexpr auto win32_handle_supported =
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED;
inline constexpr auto win32_kmt_handle_supported =
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED;
inline constexpr auto generic_compression_supported =
    CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED;
inline constexpr auto max_persisting_l2_cache_size =
    CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE;
inline constexpr auto gpu_direct_rdma_with_cuda_vmm_supported =
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED;
inline constexpr auto memory_pools_supported = CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED;
inline constexpr auto gpu_direct_rdma_supported = CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED;
inline constexpr auto mempool_supported_handle_types =
    CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES;
inline constexpr auto host_numa_virtual_memory_management_supported =
    CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED;
inline constexpr auto host_numa_memory_pools_supported =
    CU_DEVICE_ATTRIBUTE_HOST_NUMA_MEMORY_POOLS_SUPPORTED;
inline constexpr auto host_memory_pools_supported = CU_DEVICE_ATTRIBUTE_HOST_MEMORY_POOLS_SUPPORTED;
inline constexpr auto host_virtual_memory_management_supported =
    CU_DEVICE_ATTRIBUTE_HOST_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED;
inline constexpr auto host_numa_id = CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID;
} // namespace attributes

using Init = CUresult(CUDAAPI*)(unsigned int);
using DriverGetVersion = CUresult(CUDAAPI*)(int*);
using GetErrorName = CUresult(CUDAAPI*)(CUresult, const char**);
using GetErrorString = CUresult(CUDAAPI*)(CUresult, const char**);
using DeviceGetCount = CUresult(CUDAAPI*)(int*);
using DeviceGet = CUresult(CUDAAPI*)(CUdevice*, int);
using DeviceGetName = CUresult(CUDAAPI*)(char*, int, CUdevice);
using DeviceTotalMem = CUresult(CUDAAPI*)(std::size_t*, CUdevice);
using DeviceGetAttribute = CUresult(CUDAAPI*)(int*, CUdevice_attribute, CUdevice);
using DeviceGetUuid = CUresult(CUDAAPI*)(CUuuid*, CUdevice);
using DeviceGetLuid = CUresult(CUDAAPI*)(char*, unsigned int*, CUdevice);
using DeviceGetPciBusId = CUresult(CUDAAPI*)(char*, int, CUdevice);
using MemGetAllocationGranularity = CUresult(CUDAAPI*)(std::size_t*, const CUmemAllocationProp*,
                                                       CUmemAllocationGranularity_flags);
using ContextGetCurrent = CUresult(CUDAAPI*)(CUcontext*);
using ContextGetDevice = CUresult(CUDAAPI*)(CUdevice*);
using ContextSetCurrent = CUresult(CUDAAPI*)(CUcontext);
using ContextPushCurrent = CUresult(CUDAAPI*)(CUcontext);
using ContextPopCurrent = CUresult(CUDAAPI*)(CUcontext*);
using ContextCreate = CUresult(CUDAAPI*)(CUcontext*, unsigned int, CUdevice);
using ContextDestroy = CUresult(CUDAAPI*)(CUcontext);
using MemGetInfo = CUresult(CUDAAPI*)(std::size_t*, std::size_t*);
using MemAddressReserve = CUresult(CUDAAPI*)(CUdeviceptr*, std::size_t, std::size_t, CUdeviceptr,
                                             unsigned long long);
using MemAddressFree = CUresult(CUDAAPI*)(CUdeviceptr, std::size_t);
using MemCreate = CUresult(CUDAAPI*)(CUmemGenericAllocationHandle*, std::size_t,
                                     const CUmemAllocationProp*, unsigned long long);
using MemRelease = CUresult(CUDAAPI*)(CUmemGenericAllocationHandle);
using MemMap = CUresult(CUDAAPI*)(CUdeviceptr, std::size_t, std::size_t,
                                  CUmemGenericAllocationHandle, unsigned long long);
using MemUnmap = CUresult(CUDAAPI*)(CUdeviceptr, std::size_t);
using MemSetAccess = CUresult(CUDAAPI*)(CUdeviceptr, std::size_t, const CUmemAccessDesc*,
                                        std::size_t);
using MemGetAccess = CUresult(CUDAAPI*)(unsigned long long*, const CUmemLocation*, CUdeviceptr);
using MemAlloc = CUresult(CUDAAPI*)(CUdeviceptr*, std::size_t);
using MemFree = CUresult(CUDAAPI*)(CUdeviceptr);
using MemHostAlloc = CUresult(CUDAAPI*)(void**, std::size_t, unsigned int);
using MemFreeHost = CUresult(CUDAAPI*)(void*);
using MemcpyHtoDAsync = CUresult(CUDAAPI*)(CUdeviceptr, const void*, std::size_t, CUstream);
using MemcpyDtoHAsync = CUresult(CUDAAPI*)(void*, CUdeviceptr, std::size_t, CUstream);
using StreamCreate = CUresult(CUDAAPI*)(CUstream*, unsigned int);
using StreamDestroy = CUresult(CUDAAPI*)(CUstream);
using StreamSynchronize = CUresult(CUDAAPI*)(CUstream);
using StreamWaitEvent = CUresult(CUDAAPI*)(CUstream, CUevent, unsigned int);
using EventCreate = CUresult(CUDAAPI*)(CUevent*, unsigned int);
using EventDestroy = CUresult(CUDAAPI*)(CUevent);
using EventRecord = CUresult(CUDAAPI*)(CUevent, CUstream);
using EventQuery = CUresult(CUDAAPI*)(CUevent);
using EventSynchronize = CUresult(CUDAAPI*)(CUevent);
using EventElapsedTime = CUresult(CUDAAPI*)(float*, CUevent, CUevent);
using ModuleLoadData = CUresult(CUDAAPI*)(CUmodule*, const void*);
using ModuleGetFunction = CUresult(CUDAAPI*)(CUfunction*, CUmodule, const char*);
using ModuleUnload = CUresult(CUDAAPI*)(CUmodule);
using LaunchKernel = CUresult(CUDAAPI*)(CUfunction, unsigned int, unsigned int, unsigned int,
                                        unsigned int, unsigned int, unsigned int, unsigned int,
                                        CUstream, void**, void**);

} // namespace xvram::cuda::abi
