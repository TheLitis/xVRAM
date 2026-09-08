// Isolated diagnostic capability check. Optional empty-kernel permission smoke;
// no settings changes, user data, arbitrary kernel inputs or process attach.
#include "platform/dynamic_library.hpp"
#include "platform/sha256.hpp"
#include <windows.h>
#include <cuda.h>
#include <cupti_pcsampling.h>
#include <cupti_profiler_target.h>
#include <cstdio>
#include <cstdlib>
#include <array>
#include <string_view>
#include <vector>
#include <set>

namespace {
bool pinned(const wchar_t* name, const char* hash) {
  const auto module=GetModuleHandleW(name);
  wchar_t path[32768]{};
  const auto length=module ? GetModuleFileNameW(module,path,32768) : 0;
  return length && length<32768 && xvram::platform::sha256_file(path).digest==hash;
}
}

int main(int argc, char** argv) {
  const bool launch_smoke=argc==2 && std::string_view(argv[1])=="--launch-smoke";
  if(argc!=1 && !launch_smoke) return 64;
  if(!pinned(L"cupti64_2026.2.1.dll","9b10d2fafaff1a4dc9e447c4a1355fccb04ee024fa7e7d28c9e4c5ab53347faf")) return 23;
  xvram::platform::DynamicLibrary host, target;
  wchar_t executable[32768]{};
  const auto length=GetModuleFileNameW(nullptr,executable,32768);
  if(!length || length>=32768) return 23;
  const auto directory=std::filesystem::path(executable).parent_path();
  if(xvram::platform::sha256_file(directory/L"nvperf_host.dll").digest!=
       "cde4143748734948838ede61b350db59a0d7e6ac6ef451e375a5702d85174a9b" ||
     xvram::platform::sha256_file(directory/L"nvperf_target.dll").digest!=
       "049f858e512592e895b5789d0872f44ee7cbb57d3e4b8936303f4d7eb5b2b1a7" ||
     !host.open_absolute(directory/L"nvperf_host.dll") || !target.open_absolute(directory/L"nvperf_target.dll") ||
     !pinned(L"nvperf_host.dll","cde4143748734948838ede61b350db59a0d7e6ac6ef451e375a5702d85174a9b") ||
     !pinned(L"nvperf_target.dll","049f858e512592e895b5789d0872f44ee7cbb57d3e4b8936303f4d7eb5b2b1a7")) return 23;
  xvram::platform::DynamicLibrary driver;
  if (!driver.open_system({"nvcuda.dll"})) return 23;
  const auto init = driver.symbol<decltype(&cuInit)>("cuInit");
  const auto device_get = driver.symbol<decltype(&cuDeviceGet)>("cuDeviceGet");
  using Create = CUresult (CUDAAPI*)(CUcontext*, unsigned, CUdevice);
  const auto create = driver.symbol<Create>("cuCtxCreate_v2");
  const auto destroy = driver.symbol<decltype(&cuCtxDestroy)>("cuCtxDestroy_v2");
  if (!init || !device_get || !create || !destroy) return 23;
  CUdevice device = 0;
  CUcontext context = nullptr;
  if (init(0) != CUDA_SUCCESS || device_get(&device, 0) != CUDA_SUCCESS ||
      create(&context, 0, device) != CUDA_SUCCESS) return 23;
  CUpti_Profiler_Initialize_Params profiler{};
  profiler.structSize=CUpti_Profiler_Initialize_Params_STRUCT_SIZE;
  const auto initialized=cuptiProfilerInitialize(&profiler);
  CUpti_Profiler_DeviceSupported_Params support{};
  support.structSize=CUpti_Profiler_DeviceSupported_Params_STRUCT_SIZE;
  support.cuDevice=device; support.api=CUPTI_PROFILER_PC_SAMPLING;
  const auto supported=cuptiProfilerDeviceSupported(&support);
  std::printf("{\"profiler_initialize\":%d,\"device_support_result\":%d,\"device_support_level\":%d}\n",
    static_cast<int>(initialized),static_cast<int>(supported),static_cast<int>(support.isSupported));
  CUpti_PCSamplingEnableParams enable{};
  enable.size = CUpti_PCSamplingEnableParamsSize;
  enable.ctx = context;
  const auto result = cuptiPCSamplingEnable(&enable);
  std::size_t reasons=0;
  CUpti_PCSamplingGetNumStallReasonsParams count{};
  count.size=CUpti_PCSamplingGetNumStallReasonsParamsSize; count.ctx=context; count.numStallReasons=&reasons;
  const auto count_result=cuptiPCSamplingGetNumStallReasons(&count);
  int launch_result=-1, synchronize_result=-1, unload_result=-1, data_result=-1;
  if(launch_smoke && result==CUPTI_SUCCESS && count_result==CUPTI_SUCCESS && reasons>0 && reasons<=128) {
    std::vector<CUpti_PCSamplingPCData> rows(64);
    std::vector<CUpti_PCSamplingStallReason> stalls(64*reasons);
    for(std::size_t i=0;i<rows.size();++i) {
      rows[i].size=sizeof(CUpti_PCSamplingPCData); rows[i].stallReason=stalls.data()+i*reasons;
    }
    CUpti_PCSamplingData data{};
    data.size=sizeof(data); data.collectNumPcs=rows.size(); data.pPcData=rows.data();
    std::array<CUpti_PCSamplingConfigurationInfo,3> attributes{};
    attributes[0].attributeType=CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_COLLECTION_MODE;
    attributes[0].attributeData.collectionModeData.collectionMode=CUPTI_PC_SAMPLING_COLLECTION_MODE_KERNEL_SERIALIZED;
    attributes[1].attributeType=CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_DATA_BUFFER;
    attributes[1].attributeData.samplingDataBufferData.samplingDataBuffer=&data;
    attributes[2].attributeType=CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_HARDWARE_BUFFER_SIZE;
    attributes[2].attributeData.hardwareBufferSizeData.hardwareBufferSize=16*1024*1024;
    CUpti_PCSamplingConfigurationInfoParams config{};
    config.size=CUpti_PCSamplingConfigurationInfoParamsSize; config.ctx=context;
    config.numAttributes=attributes.size(); config.pPCSamplingConfigurationInfo=attributes.data();
    const auto configured=cuptiPCSamplingSetConfigurationAttribute(&config);
    const auto load=driver.symbol<decltype(&cuModuleLoadData)>("cuModuleLoadData");
    const auto function_get=driver.symbol<decltype(&cuModuleGetFunction)>("cuModuleGetFunction");
    const auto launch=driver.symbol<decltype(&cuLaunchKernel)>("cuLaunchKernel");
    const auto synchronize=driver.symbol<decltype(&cuCtxSynchronize)>("cuCtxSynchronize");
    const auto unload=driver.symbol<decltype(&cuModuleUnload)>("cuModuleUnload");
    CUmodule module=nullptr; CUfunction function=nullptr;
    constexpr const char* ptx=".version 8.0\n.target sm_86\n.address_size 64\n.visible .entry sampling_smoke() { ret; }\n";
    const bool attributes_valid=attributes[0].attributeStatus==CUPTI_SUCCESS &&
      attributes[1].attributeStatus==CUPTI_SUCCESS && attributes[2].attributeStatus==CUPTI_SUCCESS;
    if(configured==CUPTI_SUCCESS && attributes_valid && load && function_get && launch && synchronize && unload &&
       load(&module,ptx)==CUDA_SUCCESS && function_get(&function,module,"sampling_smoke")==CUDA_SUCCESS) {
      std::puts("{\"stage\":\"before_empty_kernel\"}"); std::fflush(stdout);
      launch_result=static_cast<int>(launch(function,1,1,1,32,1,1,0,nullptr,nullptr,nullptr));
      synchronize_result=static_cast<int>(synchronize());
      CUpti_PCSamplingGetDataParams get{};
      get.size=CUpti_PCSamplingGetDataParamsSize; get.ctx=context; get.pcSamplingData=&data;
      data_result=static_cast<int>(cuptiPCSamplingGetData(&get));
    }
    std::set<char*> names;
    const auto collect_names=[&] {
      if(data.totalNumPcs>rows.size()) return false;
      for(std::size_t i=0;i<data.totalNumPcs;++i) if(rows[i].functionName) names.insert(rows[i].functionName);
      return true;
    };
    bool names_valid=collect_names();
    if(module && unload) unload_result=static_cast<int>(unload(module));
    // Disable while the configured data buffers are still alive.
    CUpti_PCSamplingDisableParams disable{};
    disable.size=CUpti_PCSamplingDisableParamsSize; disable.ctx=context;
    const auto smoke_disabled=cuptiPCSamplingDisable(&disable);
    names_valid=collect_names() && names_valid;
    const auto smoke_cleanup=destroy(context);
    for(auto* name:names) std::free(name);
    std::printf("{\"launch\":%d,\"synchronize\":%d,\"data\":%d,\"unload\":%d,\"disable\":%d,\"destroy\":%d}\n",
      launch_result,synchronize_result,data_result,unload_result,static_cast<int>(smoke_disabled),static_cast<int>(smoke_cleanup));
    return names_valid && launch_result==0 && synchronize_result==0 && data_result==0 && unload_result==0 &&
      smoke_disabled==CUPTI_SUCCESS && smoke_cleanup==CUDA_SUCCESS ? 0 : 27;
  }
  int disabled = -1;
  if (result == CUPTI_SUCCESS) {
    CUpti_PCSamplingDisableParams disable{};
    disable.size = CUpti_PCSamplingDisableParamsSize;
    disable.ctx = context;
    disabled = static_cast<int>(cuptiPCSamplingDisable(&disable));
  }
  const auto cleanup = destroy(context);
  std::printf("{\"diagnostic\":\"pc_sampling_enable\",\"device\":0,"
              "\"enable_result\":%d,\"disable_result\":%d,\"context_destroy_result\":%d,"
              "\"count_result\":%d,\"stall_reasons\":%zu}\n",
              static_cast<int>(result), disabled, static_cast<int>(cleanup),static_cast<int>(count_result),reasons);
  if (cleanup != CUDA_SUCCESS || (result == CUPTI_SUCCESS && disabled != 0)) return 27;
  return !launch_smoke && result == CUPTI_SUCCESS && count_result==CUPTI_SUCCESS && reasons>0 ? 0 : 23;
}
