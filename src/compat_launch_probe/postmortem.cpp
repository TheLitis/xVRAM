#include "postmortem.hpp"
#include <windows.h>
#include <stdexcept>

namespace xvram::launch_probe::postmortem {
namespace {
// Keep the mapping and view alive until Windows reclaims the owned process.
// Closing it at atexit would reopen the late-producer observation gap.
volatile LONG64* ledger=nullptr;
constexpr LONG64 magic=0x3154504D415658LL;
enum Field : unsigned { Magic, Version, Bytes, Armed, Sealed, Finished,
  CallbacksOpen, ApisOpen, LateEntries, Errors, Entered, Exited, GpuDrained,
  ActivityDrained, CensusClosed, RetainedNames, SamplerDisabled,
  ActivityCallbacksOpen, ActivityOutstanding, LateActivity, Count };
static_assert(Count==20);
LONG64 read(Field key) noexcept { return InterlockedCompareExchange64(ledger+key,0,0); }
void set(Field key, LONG64 value) noexcept { InterlockedExchange64(ledger+key,value); }
}
void initialize() {
  wchar_t name[160]{};
  const auto n=GetEnvironmentVariableW(L"XVRAM_AUDIT_POSTMORTEM",name,160);
  if(!n) return; // Evidence without the controller ledger cannot prove cutoff.
  if(n>=160 || ledger) throw std::runtime_error("postmortem_name_or_reinitialize");
  const auto mapping=OpenFileMappingW(FILE_MAP_ALL_ACCESS,FALSE,name);
  if(!mapping) throw std::runtime_error("postmortem_mapping_missing");
  auto* view=MapViewOfFile(mapping,FILE_MAP_ALL_ACCESS,0,0,Count*sizeof(LONG64));
  if(!view) { CloseHandle(mapping); throw std::runtime_error("postmortem_map_failed"); }
  ledger=static_cast<volatile LONG64*>(view);
  if(read(Magic)!=magic || read(Version)!=1 || read(Bytes)!=Count*8 || read(Armed)!=0)
    throw std::runtime_error("postmortem_header_or_reuse");
  for(unsigned i=Sealed;i<Count;++i)
    if(InterlockedCompareExchange64(ledger+i,0,0)!=0) throw std::runtime_error("postmortem_dirty_mapping");
  set(Armed,1);
}
void error() noexcept { if(ledger) InterlockedIncrement64(ledger+Errors); }
void callback_begin(bool api, bool entering, bool terminal_sync) noexcept {
  if(!ledger) return;
  InterlockedIncrement64(ledger+CallbacksOpen);
  if(api && entering) {
    InterlockedIncrement64(ledger+ApisOpen);
    InterlockedIncrement64(ledger+Entered);
    if(read(Sealed) && !terminal_sync) InterlockedIncrement64(ledger+LateEntries);
  }
}
void callback_end(bool api, bool entering) noexcept {
  if(!ledger) return;
  if(api && !entering) {
    InterlockedIncrement64(ledger+Exited);
    if(InterlockedDecrement64(ledger+ApisOpen)<0) error();
  }
  if(InterlockedDecrement64(ledger+CallbacksOpen)<0) error();
}
void activity_begin(bool request) noexcept {
  if(!ledger) return;
  InterlockedIncrement64(ledger+ActivityCallbacksOpen);
  if(read(ActivityDrained)) InterlockedIncrement64(ledger+LateActivity);
  if(request) InterlockedIncrement64(ledger+ActivityOutstanding);
}
void activity_cancel() noexcept {
  if(ledger && InterlockedDecrement64(ledger+ActivityOutstanding)<0) error();
}
void activity_end(bool completed) noexcept {
  if(!ledger) return;
  if(completed) activity_cancel();
  if(InterlockedDecrement64(ledger+ActivityCallbacksOpen)<0) error();
}
void seal() {
  if(!ledger) return;
  if(InterlockedExchange64(ledger+Sealed,1)!=0 || read(CallbacksOpen) || read(ApisOpen) || read(Errors)) {
    error(); throw std::runtime_error("postmortem_seal_open");
  }
}
void drained(bool gpu, bool activity) noexcept {
  if(!ledger) return;
  set(GpuDrained,gpu ? 1 : 0); set(ActivityDrained,activity ? 1 : 0);
  if(!gpu || !activity || read(ActivityCallbacksOpen) || read(ActivityOutstanding)) error();
}
void census_closed() noexcept { if(ledger) set(CensusClosed,1); }
void sampler_disabled(std::uint64_t names) noexcept {
  if(!ledger) return;
  if(names>65536) { error(); return; }
  set(RetainedNames,static_cast<LONG64>(names)); set(SamplerDisabled,1);
}
void finish() noexcept { if(ledger) set(Finished,1); }
}
