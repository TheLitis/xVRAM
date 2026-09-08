// Run with XVRAM_MEMORY_WITNESS_TRACE pointing to a fresh file. These are
// ordinary CPU calls into the observer, not CUDA submissions or driver loads.
#include "compat_launch_probe/memory_witness.hpp"
#include <iostream>
#include <string_view>

namespace memory=xvram::launch_probe::memory_witness;
namespace {
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
CUcontext context() { return reinterpret_cast<CUcontext>(static_cast<std::uintptr_t>(0x600000)); }
template<class T> void callback(const char* symbol, T& params, std::uint64_t id,
                               CUpti_ApiCallbackSite site, int result=0) {
  CUpti_CallbackData data{};
  data.functionName=symbol; data.functionParams=&params;
  data.context=context(); data.callbackSite=site; data.functionReturnValue=&result;
  memory::api(data,id);
}
template<class T> void pair(const char* symbol, T& params, std::uint64_t id, int result=0) {
  callback(symbol,params,id,CUPTI_API_ENTER);
  callback(symbol,params,id,CUPTI_API_EXIT,result);
}
void valid() {
  CUdeviceptr normal=0x11000000;
  cuMemAlloc_v2_params allocation{&normal,1024};
  callback("cuMemAlloc_v2",allocation,1,CUPTI_API_ENTER);
  check(!memory::resolve(normal,1,0,context()).known,"pending API must not resolve");
  callback("cuMemAlloc_v2",allocation,1,CUPTI_API_EXIT);
  const auto first=memory::resolve(normal,1024,0,context());
  check(first.known && first.readable && first.writable && first.observed_sequence>0,"allocation snapshot");
  check(!memory::resolve(normal,1024,0,nullptr).readable,"unknown context cannot claim implicit access");
  cuMemcpyHtoDAsync_v2_params transfer{normal,nullptr,1024,nullptr};
  pair("cuMemcpyHtoDAsync_v2",transfer,2);
  CUdeviceptr reserved=0x22000000;
  cuMemAddressReserve_params reserve{&reserved,4096,0,0,0};
  pair("cuMemAddressReserve",reserve,3);
  CUmemGenericAllocationHandle handle=0xABCDEF55;
  CUmemAllocationProp props{}; props.type=CU_MEM_ALLOCATION_TYPE_PINNED;
  props.location.type=CU_MEM_LOCATION_TYPE_DEVICE; props.location.id=0;
  cuMemCreate_params create{&handle,4096,&props,0};
  pair("cuMemCreate",create,4);
  cuMemMap_params map{reserved,4096,0,handle,0}; pair("cuMemMap",map,5);
  check(memory::resolve(reserved,4096).mapped && !memory::resolve(reserved,4096).readable,"mapping is not access");
  CUmemAccessDesc desc{}; desc.location=props.location; desc.flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  cuMemSetAccess_params access{reserved,4096,&desc,1}; pair("cuMemSetAccess",access,6);
  check(memory::resolve(reserved,4096).writable,"SetAccess observed");
  cuMemcpyDtoDAsync_v2_params d2d{reserved,normal,1024,nullptr}; pair("cuMemcpyDtoDAsync_v2",d2d,7);
  cuMemsetD8Async_params fill{reserved,0,4096,nullptr}; pair("cuMemsetD8Async",fill,8);
  cuMemRelease_params release{handle}; pair("cuMemRelease",release,9);
  check(memory::resolve(reserved,4096).mapped,"released handle retains mapping");
  cuMemUnmap_params unmap{reserved,4096}; pair("cuMemUnmap",unmap,10);
  cuMemAddressFree_params free_reservation{reserved,4096}; pair("cuMemAddressFree",free_reservation,11);
  cuMemFree_v2_params free{normal}; pair("cuMemFree_v2",free,12);
  pair("cuMemAlloc_v2",allocation,13);
  check(memory::resolve(normal,1024,0,context()).generation==first.generation+1,"native VA generation");
  pair("cuMemFree_v2",free,14);
  // Host-pinned calls are explicitly outside this device-only observer.
  cuMemFreeHost_params free_host{nullptr}; pair("cuMemFreeHost",free_host,15);
  check(memory::healthy(),"host scope is explicit, not accidentally unsupported");
  memory::finish();
}
void compound() {
  CUdeviceptr reserved=0x22000000;
  constexpr std::size_t chunk=4096;
  cuMemAddressReserve_params reserve{&reserved,3*chunk,0,0,0};
  std::uint64_t api=0;
  pair("cuMemAddressReserve",reserve,++api);
  CUmemAllocationProp props{}; props.type=CU_MEM_ALLOCATION_TYPE_PINNED;
  props.location.type=CU_MEM_LOCATION_TYPE_DEVICE; props.location.id=0;
  for (std::size_t i=0;i<3;++i) {
    CUmemGenericAllocationHandle handle=0xABCDEF55+i;
    cuMemCreate_params create{&handle,chunk,&props,0}; pair("cuMemCreate",create,++api);
    cuMemMap_params map{reserved+i*chunk,chunk,0,handle,0}; pair("cuMemMap",map,++api);
    cuMemRelease_params release{handle}; pair("cuMemRelease",release,++api);
  }
  CUmemAccessDesc desc{}; desc.location=props.location; desc.flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  cuMemSetAccess_params access{reserved,3*chunk,&desc,1}; pair("cuMemSetAccess",access,++api);
  const auto before=memory::resolve(reserved,3*chunk);
  check(before.mappings.size()==3 && before.writable,"three full mappings observed");
  cuMemUnmap_params unmap{reserved,3*chunk}; pair("cuMemUnmap",unmap,++api);
  const auto after=memory::resolve(reserved,3*chunk);
  check(memory::healthy() && after.known && !after.mapped && after.revision==before.revision+3,
        "single native unmap retires three original mappings");
  cuMemAddressFree_params release{reserved,3*chunk}; pair("cuMemAddressFree",release,++api);
  memory::finish();
}
void invalid(std::string_view scenario) {
  CUdeviceptr address=0x11000000;
  cuMemAlloc_v2_params allocate{&address,4096};
  if (scenario=="failed-allocation") {
    pair("cuMemAlloc_v2",allocate,1,2);
    check(memory::healthy() && !memory::resolve(address,1).known,"failed output was not admitted");
    memory::finish(); return;
  }
  pair("cuMemAlloc_v2",allocate,1);
  if (scenario=="null-free") {
    const auto before=memory::resolve(address,4096,0,context());
    cuMemFree_v2_params empty{0};
    pair("cuMemFree_v2",empty,2); pair("cuMemFree_v2",empty,3);
    pair("cuMemFree_v2",empty,4,1); // Failed calls are not observed no-ops.
    const auto after=memory::resolve(address,4096,0,context());
    check(memory::healthy() && after.known && before.allocation_id==after.allocation_id
          && before.generation==after.generation && before.revision==after.revision,
          "successful null free cannot retire or revise allocation state");
    cuMemFree_v2_params free{address}; pair("cuMemFree_v2",free,5);
    memory::finish(); return;
  }
  if (scenario=="unmapped-copy") {
    cuMemcpyHtoDAsync_v2_params transfer{0x33000000,nullptr,4096,nullptr};
    pair("cuMemcpyHtoDAsync_v2",transfer,2);
  } else if (scenario=="stale-api") pair("cuMemAlloc_v2",allocate,1);
  else if (scenario=="unknown-map") {
    cuMemMap_params map{0x33000000,4096,0,777,0}; pair("cuMemMap",map,2);
  } else if (scenario=="partial-free") {
    cuMemFree_v2_params free{address+16}; pair("cuMemFree_v2",free,2);
  } else if (scenario=="unknown-free") {
    cuMemFree_v2_params free{0x33000000}; pair("cuMemFree_v2",free,2);
  } else if (scenario=="stale-free") {
    cuMemFree_v2_params free{address}; pair("cuMemFree_v2",free,2); pair("cuMemFree_v2",free,3);
  } else if (scenario=="unsupported") {
    pair("cuMemImportFromShareableHandle",allocate,2);
  } else throw std::runtime_error("unknown test scenario");
  check(!memory::healthy() && !memory::resolve(address,1).known,"observer failure must invalidate snapshots");
  bool failed=false;
  try { memory::finish(); } catch (const std::runtime_error&) { failed=true; }
  check(failed,"failed observer cannot finish successfully");
}
}
int main(int argc, char** argv) {
  try {
    check(!memory::enabled() && !memory::resolve(1,1).known,"observer defaults off");
    memory::initialize(); check(memory::enabled(),"fresh trace env required");
    if (argc==1) valid(); else if (argc==2 && std::string_view(argv[1])=="compound-unmap") compound();
    else if (argc==2) invalid(argv[1]); else return 64;
    std::cout<<"typed memory observer CPU test passed\n";
    return 0;
  } catch (const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
