// No CUDA library is linked or loaded. Each scenario needs a fresh trace path.
#include "compat_launch_probe/teardown_witness.hpp"
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <thread>

namespace witness=xvram::launch_probe::teardown_witness;
namespace {
void check(bool value, const char* message) { if(!value) throw std::runtime_error(message); }
CUlibrary library() { return reinterpret_cast<CUlibrary>(static_cast<std::uintptr_t>(0xABCDEF001234)); }
CUcontext context() { return reinterpret_cast<CUcontext>(static_cast<std::uintptr_t>(0xABCDEF005678)); }
template<class T> void callback(const char* symbol, T* params, std::uint64_t api,
                               CUpti_ApiCallbackSite site, int result=0) {
  CUpti_CallbackData data{}; data.functionName=symbol; data.functionParams=params;
  data.functionReturnValue=&result; data.callbackSite=site;
  witness::api(data,api);
}
template<class T> void pair(const char* symbol, T& params, std::uint64_t api, int result=0) {
  callback(symbol,&params,api,CUPTI_API_ENTER);
  callback(symbol,&params,api,CUPTI_API_EXIT,result);
}
void late_atexit() noexcept {
  cuLibraryUnload_params unload{library()}; cuDevicePrimaryCtxRelease_params release{0};
  pair("cuLibraryUnload",unload,3); pair("cuDevicePrimaryCtxRelease",release,4);
  const auto result=witness::snapshot();
  if(!result.healthy || !result.checkpointed || result.after_checkpoint_pairs!=2 ||
     result.libraries_live || result.primary_references || result.pending) std::abort();
}
void atexit_setup() {
  CUcontext ctx=context(); cuDevicePrimaryCtxRetain_params retain{&ctx,0};
  CUlibrary lib=library(); cuLibraryLoadData_params load{}; load.library=&lib;
  pair("cuDevicePrimaryCtxRetain",retain,1); pair("cuLibraryLoadData",load,2);
  check(witness::healthy() && !witness::snapshot().checkpointed,"atexit not yet reached");
}
void valid() {
  CUcontext ctx=nullptr; cuDevicePrimaryCtxRetain_params retain{&ctx,0};
  callback("cuDevicePrimaryCtxRetain",&retain,1,CUPTI_API_ENTER);
  check(witness::snapshot().primary_references==0,"retain output read before EXIT");
  ctx=context(); callback("cuDevicePrimaryCtxRetain",&retain,1,CUPTI_API_EXIT);
  CUlibrary lib=nullptr; cuLibraryLoadData_params load{}; load.library=&lib;
  callback("cuLibraryLoadData",&load,2,CUPTI_API_ENTER);
  check(witness::snapshot().libraries_live==0,"load output read before EXIT");
  lib=library(); callback("cuLibraryLoadData",&load,2,CUPTI_API_EXIT);
  cuLibraryUnload_params unload{lib}; pair("cuLibraryUnload",unload,3);
  cuLibraryLoadFromFile_params from_file{}; from_file.library=&lib;
  pair("cuLibraryLoadFromFile",from_file,4); // Same raw handle, fresh known library generation.
  pair("cuDevicePrimaryCtxRetain",retain,5);
  cuDevicePrimaryCtxRelease_v2_params release_v2{0}; pair("cuDevicePrimaryCtxRelease_v2",release_v2,6);
  witness::checkpoint(); const auto cutoff=witness::snapshot();
  pair("cuLibraryUnload",unload,7);
  cuDevicePrimaryCtxRelease_params release{0}; pair("cuDevicePrimaryCtxRelease",release,8);
  const auto result=witness::snapshot();
  check(result.healthy && result.checkpointed && !result.pending,"healthy retained observer");
  check(result.sequence>cutoff.sequence && result.after_checkpoint_pairs==2,"post-checkpoint rows retained");
  check(result.libraries_live==0 && result.primary_references==0,"observed retain/library balance");
  check(result.library_loads==2 && result.library_unloads==2 && result.primary_retains==2 && result.primary_releases==2,
        "exact lifecycle accounting");
  check(result.lifecycle_pairs==8 && result.qpc_frequency>0,"bounded QPC and API count");
}
void invalid(std::string_view scenario) {
  CUcontext ctx=context(); cuDevicePrimaryCtxRetain_params retain{&ctx,0};
  CUlibrary lib=library(); cuLibraryLoadData_params load{}; load.library=&lib;
  cuLibraryUnload_params unload{lib}; cuDevicePrimaryCtxRelease_params release{0};
  pair("cuDevicePrimaryCtxRetain",retain,1);
  pair("cuLibraryLoadData",load,2);
  if(scenario=="stale-unload") { pair("cuLibraryUnload",unload,3); pair("cuLibraryUnload",unload,4); }
  else if(scenario=="live-library-aba") pair("cuLibraryLoadData",load,3);
  else if(scenario=="api-aba") pair("cuLibraryUnload",unload,2);
  else if(scenario=="failed-api") pair("cuLibraryUnload",unload,3,1);
  else if(scenario=="orphan-exit") callback("cuLibraryUnload",&unload,3,CUPTI_API_EXIT);
  else if(scenario=="input-changed") {
    callback("cuLibraryUnload",&unload,3,CUPTI_API_ENTER); unload.library=nullptr;
    callback("cuLibraryUnload",&unload,3,CUPTI_API_EXIT);
  } else if(scenario=="unknown-release") { release.dev=1; pair("cuDevicePrimaryCtxRelease",release,3); }
  else if(scenario=="reference-underflow") { pair("cuDevicePrimaryCtxRelease",release,3); pair("cuDevicePrimaryCtxRelease",release,4); }
  else if(scenario=="context-recreation") { pair("cuDevicePrimaryCtxRelease",release,3); pair("cuDevicePrimaryCtxRetain",retain,4); }
  else if(scenario=="wrong-context") { ctx=reinterpret_cast<CUcontext>(static_cast<std::uintptr_t>(0xABCDEFAAAA)); pair("cuDevicePrimaryCtxRetain",retain,3); }
  else if(scenario=="negative-device") { release.dev=-1; pair("cuDevicePrimaryCtxRelease",release,3); }
  else if(scenario=="multiple-producers") {
    std::thread other([&] { pair("cuLibraryUnload",unload,3); }); other.join();
  } else if(scenario=="null-params") callback("cuLibraryUnload",static_cast<cuLibraryUnload_params*>(nullptr),3,CUPTI_API_ENTER);
  else if(scenario=="unreadable-params") callback("cuLibraryUnload",reinterpret_cast<cuLibraryUnload_params*>(static_cast<std::uintptr_t>(1)),3,CUPTI_API_ENTER);
  else if(scenario=="unreadable-output") {
    load.library=reinterpret_cast<CUlibrary*>(static_cast<std::uintptr_t>(1)); pair("cuLibraryLoadData",load,3);
  } else if(scenario=="unreadable-result") {
    callback("cuLibraryUnload",&unload,3,CUPTI_API_ENTER);
    CUpti_CallbackData data{}; data.functionName="cuLibraryUnload"; data.functionParams=&unload;
    data.functionReturnValue=reinterpret_cast<void*>(static_cast<std::uintptr_t>(1));
    data.callbackSite=CUPTI_API_EXIT; witness::api(data,3);
  } else if(scenario=="null-output") { lib=nullptr; pair("cuLibraryLoadData",load,3); }
  else if(scenario=="output-slot-changed") {
    callback("cuLibraryLoadData",&load,3,CUPTI_API_ENTER);
    CUlibrary another=library(); load.library=&another; callback("cuLibraryLoadData",&load,3,CUPTI_API_EXIT);
  } else if(scenario=="open-checkpoint") { callback("cuLibraryUnload",&unload,3,CUPTI_API_ENTER); witness::checkpoint(); }
  else if(scenario=="overlap") { callback("cuLibraryUnload",&unload,3,CUPTI_API_ENTER); callback("cuLibraryLoadData",&load,4,CUPTI_API_ENTER); }
  else if(scenario=="unknown-reset") pair("cuDevicePrimaryCtxReset",release,3);
  else if(scenario=="late-launch") { witness::checkpoint(); pair("cuLaunchKernel",release,3); }
  else if(scenario=="late-query") { witness::checkpoint(); pair("cuCtxGetCurrent",release,3); }
  else if(scenario=="release-before-unloads") {
    witness::checkpoint(); pair("cuDevicePrimaryCtxRelease",release,3);
    const auto result=witness::snapshot();
    check(result.healthy && !result.primary_references && result.libraries_live==1 && result.after_checkpoint_pairs==1,
          "successful release remains observed; outstanding library is not declared closed");
    return;
  }
  else throw std::runtime_error("unknown CPU test scenario");
  check(!witness::healthy() && witness::snapshot().errors>0,"invalid witness must fail closed");
}
}
int main(int argc, char** argv) {
  try {
    const bool late=argc==2 && std::string_view(argv[1])=="atexit-tail";
    // Reverse registration order makes this execute AFTER the observer's real
    // atexit checkpoint. It is not a manually simulated closed-file boundary.
    if(late && std::atexit(late_atexit)!=0) throw std::runtime_error("test atexit registration");
    check(!witness::enabled(),"defaults off"); witness::initialize();
    check(witness::enabled() && witness::healthy(),"fresh trace path environment required");
    if(late) atexit_setup(); else if(argc==1) valid(); else if(argc==2) invalid(argv[1]); else return 64;
    std::cout<<"typed teardown observer CPU test passed\n"; return 0;
  } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
