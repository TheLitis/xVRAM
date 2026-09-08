// Runs entirely on the CPU. The allocation resolver is deliberately fake;
// official CUDA/CUPTI headers supply declarations, not linked driver libraries.
#include "compat_launch_probe/kernel_capture.hpp"
#include "compat_launch_probe/memory_witness.hpp"
#include "capture_catalog.hpp"
#include <windows.h>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>

namespace mw=xvram::launch_probe::memory_witness;
namespace kc=xvram::launch_probe::kernel_capture;
namespace tc=xvram::launch_probe::typed_capture;
namespace {
constexpr std::uint64_t sentinel=0x7fedcba987654321ULL;
std::size_t lookups=0;
void require(bool value) { if(!value) throw std::runtime_error("kernel_capture_native_test"); }
std::size_t count(const std::string& text,const std::string& needle) {
  std::size_t found=0,position=0;
  while((position=text.find(needle,position))!=std::string::npos) { ++found; position+=needle.size(); }
  return found;
}
}
namespace xvram::launch_probe::memory_witness {
bool enabled() noexcept { return true; }
bool healthy() noexcept { return true; }
Resolution resolve(std::uint64_t address,std::uint64_t bytes,int device,CUcontext context,std::uint64_t alignment) noexcept {
  ++lookups;
  if(address!=sentinel || bytes!=1 || device!=0 || !context || alignment!=16) return {};
  Resolution result;
  result.known=true; result.mapped=true; result.readable=true; result.writable=true;
  result.allocation_id=7; result.generation=3; result.offset_bytes=17;
  result.bytes=bytes; result.allocation_bytes=4096; result.revision=19;
  result.observed_sequence=23; result.alignment=16; result.base_mod_alignment=0;
  return result;
}
}
int main() {
  std::filesystem::path output;
  try {
    wchar_t temp[MAX_PATH]{};
    require(GetTempPathW(MAX_PATH,temp)>0);
    output=std::filesystem::path(temp)/(L"xvram-kernel-capture-"+std::to_wstring(GetCurrentProcessId())+L"-"+std::to_wstring(GetTickCount64())+L".jsonl");
    require(SetEnvironmentVariableW(L"XVRAM_KERNEL_ARGUMENT_TRACE",output.c_str())!=0);
    kc::initialize(); require(kc::enabled() && kc::healthy());
    const auto& kernel=tc::kernels[0];
    require(kernel.supported);
    xvram::launch_probe::Snapshot snapshot{kernel.symbol,{}};
    std::array<std::array<std::byte,128>,64> storage{};
    std::array<const void*,64> bank{};
    std::size_t expected_pointers=0;
    for(std::size_t i=0;i<kernel.arguments.size();++i) {
      const auto& argument=kernel.arguments[i];
      snapshot.parameters.push_back({argument.offset,argument.bytes});
      bank[i]=storage[i].data();
      for(const auto& field:argument.fields) {
        if(field.type==tc::ValueType::ptr) {
          std::memcpy(storage[i].data()+field.offset,&sentinel,sizeof(sentinel)); ++expected_pointers;
        }
      }
    }
    const kc::Geometry geometry{{2,3,4},{32,2,1},128};
    const auto context=reinterpret_cast<CUcontext>(static_cast<std::uintptr_t>(1));
    kc::launch(1,snapshot,geometry,bank.data(),nullptr,0,context); kc::returned(1,0);
    require(lookups==expected_pointers && kc::healthy());
    auto mismatch=snapshot; ++mismatch.parameters[0].bytes;
    const auto inaccessible=reinterpret_cast<void*>(static_cast<std::uintptr_t>(1));
    kc::launch(2,mismatch,geometry,inaccessible,nullptr,0,context); kc::returned(2,0);
    const auto& opaque=tc::kernels.back(); require(!opaque.supported);
    kc::launch(3,{opaque.symbol,{}},geometry,inaccessible,nullptr,0,context); kc::returned(3,0);
    kc::launch(4,{"unknown-symbol",{}},geometry,inaccessible,nullptr,0,context); kc::returned(4,0);
    kc::launch(5,snapshot,geometry,inaccessible,inaccessible,0,context); kc::returned(5,0);
    require(lookups==expected_pointers && kc::healthy());
    // Actual Windows fault-aware host read must not crash or emit any fields.
    kc::launch(6,snapshot,geometry,inaccessible,nullptr,0,context); kc::returned(6,0);
    require(lookups==expected_pointers && !kc::healthy());
    kc::finish();
    std::ifstream stream(output,std::ios::binary);
    const std::string trace{std::istreambuf_iterator<char>(stream),{}};
    require(trace.find(std::to_string(sentinel))==std::string::npos);
    require(trace.find("7fedcba9")==std::string::npos);
    require(trace.find("\"allocation_id\":7,\"generation\":3")!=std::string::npos);
    require(trace.find("\"grid_x\":2,\"grid_y\":3,\"grid_z\":4")!=std::string::npos);
    require(trace.find("\"memory_bounds_proven\":true")==std::string::npos);
    require(count(trace,"\"kind\":\"begin\"")==6 && count(trace,"\"kind\":\"return\"")==6);
    require(count(trace,"\"status\":\"captured\"")==1);
    require(count(trace,"\"status\":\"unsupported_abi\"")==1);
    require(count(trace,"\"status\":\"unsupported_opaque_library\"")==1);
    require(count(trace,"\"status\":\"unsupported_symbol\"")==1);
    require(count(trace,"\"status\":\"unsupported_argument_bank\"")==1);
    require(count(trace,"\"status\":\"capture_fault\"")==1);
    require(trace.find("\"capture_faults\":1")!=std::string::npos);
    std::uint64_t sequence=0;
    for(std::size_t position=0;(position=trace.find("\"sequence\":",position))!=std::string::npos;) {
      position+=11; require(std::stoull(trace.substr(position))==++sequence);
    }
    stream.close();
    require(std::filesystem::remove(output));
    std::cout<<"kernel capture exact ABI, fault-aware reads, geometry, generation IDs, privacy: PASS\n";
    return 0;
  } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
