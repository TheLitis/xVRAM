#include "compat_launch_probe/typed_capture.hpp"
#include "compat_launch_probe/activity_drain.hpp"
#include "capture_catalog.hpp"
#include <cstdlib>
#include <iostream>
#include <vector>
namespace tc=xvram::launch_probe::typed_capture;
static_assert(tc::kernels.size()==39 && tc::kernels[0].supported && !tc::kernels[38].supported);
using xvram::launch_probe::Snapshot;
namespace {
void require(bool condition) { if(!condition) throw std::runtime_error("typed_capture_test"); }
template<class F> void rejects(F&& f) { bool caught=false; try { f(); } catch(const std::runtime_error&) { caught=true; } require(caught); }
struct Sink {
  std::vector<std::uint64_t> pointers, scalars;
  void pointer(std::size_t,const char*,const char*,std::uint64_t value) { pointers.push_back(value); }
  void scalar(std::size_t,const char*,const char*,tc::ValueType,std::uint64_t value) { scalars.push_back(value); }
};
}
int main() {
  try {
    const std::array activity_kinds{1,2,3};
    std::vector<int> activity_calls;
    for (int fault=-1; fault<4; ++fault) {
      activity_calls.clear();
      const auto disable=[&](int kind) { activity_calls.push_back(kind); return kind==fault ? 1 : 0; };
      const auto flush=[&] { activity_calls.push_back(4); return fault==0 ? 1 : 0; };
      require(!xvram::launch_probe::retire_activity(false,true,activity_kinds,disable,flush));
      require(!xvram::launch_probe::retire_activity(true,false,activity_kinds,disable,flush));
      require(activity_calls.empty());
      require(xvram::launch_probe::retire_activity(true,true,activity_kinds,disable,flush)==(fault==-1));
      require(activity_calls==std::vector<int>({1,2,3,4}));
    }
    constexpr tc::Field pointer[]{ {"value",tc::ValueType::ptr,0} };
    constexpr tc::Field scalar[]{ {"value",tc::ValueType::u32,0} };
    const tc::Argument arguments[]{ {"input",0,8,pointer}, {"count",8,4,scalar} };
    const tc::Kernel kernel{1,"reviewed",arguments,true};
    Snapshot snapshot{"reviewed",{{0,8},{8,4}}};
    std::uint64_t device=0x1234567887654321ULL;
    std::uint32_t count=19;
    const void* bank[]{&device,&count};
    std::size_t reads=0;
    const auto reader=[&](const void* source,void* target,std::size_t size) { ++reads; std::memcpy(target,source,size); };
    Sink sink;
    tc::capture(kernel,snapshot,bank,nullptr,reader,sink);
    require(sink.pointers==std::vector<std::uint64_t>{device} && sink.scalars==std::vector<std::uint64_t>{19} && reads==3);
    require(device==0x1234567887654321ULL && count==19);
    reads=0;
    snapshot.parameters[1].bytes=8;
    rejects([&] { tc::capture(kernel,snapshot,bank,nullptr,reader,sink); }); require(reads==0);
    snapshot.parameters[1].bytes=4;
    rejects([&] { tc::capture(kernel,snapshot,bank,bank,reader,sink); }); require(reads==0);
    snapshot.name="opaque";
    rejects([&] { tc::capture(kernel,snapshot,bank,nullptr,reader,sink); }); require(reads==0);
    snapshot.name="reviewed";
    Sink partial;
    rejects([&] { tc::capture(kernel,snapshot,bank,nullptr,[](const void*,void*,std::size_t) { throw std::runtime_error("injected_read_fault"); },partial); });
    require(partial.pointers.empty() && partial.scalars.empty());
    const tc::Field fields[]{ {"left",tc::ValueType::u32,0}, {"right",tc::ValueType::u32,12} };
    const tc::Argument padded[]{ {"pair",0,16,fields} };
    const tc::Kernel padded_kernel{2,"padding",padded,true};
    std::array<std::uint32_t,4> data{7,0xdeadbeefU,0xdeadbeefU,9};
    const void* padded_bank[]{data.data()};
    Sink safe;
    tc::capture(padded_kernel,Snapshot{"padding",{{0,16}}},padded_bank,nullptr,reader,safe);
    require(safe.scalars==std::vector<std::uint64_t>({7,9}) && safe.pointers.empty());
    std::cout<<"typed capture validation, fault boundary, pointer routing, padding: PASS\n";
    return 0;
  } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
