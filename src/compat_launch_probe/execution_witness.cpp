// Typed observational callback data only; no CUDA queries or argument changes.
#include "execution_witness.hpp"
#include "compat_audit_collector/trace_core.hpp"
#include <windows.h>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <map>

namespace xvram::launch_probe::execution_witness {
namespace {
struct Entry {
  std::uint64_t thread=0, context=0, stream=0, event=0;
  unsigned flags=0;
  bool tool=false;
  std::string symbol, stream_kind="none";
};
struct State {
  HANDLE file=INVALID_HANDLE_VALUE;
  std::mutex mutex;
  audit::Registry ids;
  std::map<std::uint64_t, Entry> pending;
  std::uint64_t sequence=0, bytes=0;
  std::atomic_uint64_t errors{0};
  bool sealed=false, finished=false;
  template<class T> std::uint64_t id(const char* category, T native) {
    return native ? ids.identity(category,reinterpret_cast<std::uintptr_t>(native)) : 0;
  }
  audit::JsonLine line(const char* kind) {
    audit::JsonLine value;
    value.number("schema_version",1); value.string("record_type","xvram.cuda_order_witness");
    value.number("sequence",++sequence); value.string("kind",kind); return value;
  }
  void emit(const audit::JsonLine& value) {
    const auto text=value.finish();
    if(text.size()>256ULL*1024*1024-bytes) throw std::runtime_error("order_trace_capacity");
    DWORD written=0;
    if(!WriteFile(file,text.data(),static_cast<DWORD>(text.size()),&written,nullptr) || written!=text.size())
      throw std::runtime_error("order_trace_write");
    bytes+=text.size();
  }
};
State* state=nullptr;
std::atomic_uint64_t thread_sequence{0};
thread_local const std::uint64_t thread_token=++thread_sequence;
bool relevant(std::string_view name) {
  return name.starts_with("cuLaunch") || name.starts_with("cuMemcpy") || name.starts_with("cuMemset") ||
    name.starts_with("cuStream") || name.starts_with("cuEvent") || name.starts_with("cuCtx") ||
    name.starts_with("cuDevicePrimaryCtx") || name.starts_with("cuGraph") ||
    name=="cuMemAllocAsync" || name=="cuMemFreeAsync";
}
void inputs(Entry& entry, const void* p) {
  const auto& s=entry.symbol;
  if(!p) {
    if(s=="cuCtxSynchronize") return; // legacy no-argument API has no parameter block
    throw std::runtime_error("order_parameters_missing");
  }
  CUstream stream=nullptr;
  CUevent event=nullptr;
  if(s=="cuLaunchKernel") stream=static_cast<const cuLaunchKernel_params*>(p)->hStream;
  else if(s=="cuCtxSetCurrent") entry.context=state->id("context",static_cast<const cuCtxSetCurrent_params*>(p)->ctx);
  else if(s=="cuMemcpyHtoDAsync_v2") stream=static_cast<const cuMemcpyHtoDAsync_v2_params*>(p)->hStream;
  else if(s=="cuMemcpyDtoHAsync_v2") stream=static_cast<const cuMemcpyDtoHAsync_v2_params*>(p)->hStream;
  else if(s=="cuMemcpyDtoDAsync_v2") stream=static_cast<const cuMemcpyDtoDAsync_v2_params*>(p)->hStream;
  else if(s=="cuMemsetD8Async") stream=static_cast<const cuMemsetD8Async_params*>(p)->hStream;
  else if(s=="cuStreamSynchronize") stream=static_cast<const cuStreamSynchronize_params*>(p)->hStream;
  else if(s=="cuStreamDestroy_v2") stream=static_cast<const cuStreamDestroy_v2_params*>(p)->hStream;
  else if(s=="cuStreamCreate") entry.flags=static_cast<const cuStreamCreate_params*>(p)->Flags;
  else if(s=="cuEventCreate") entry.flags=static_cast<const cuEventCreate_params*>(p)->Flags;
  else if(s=="cuEventDestroy_v2") event=static_cast<const cuEventDestroy_v2_params*>(p)->hEvent;
  else if(s=="cuEventRecord") { const auto* a=static_cast<const cuEventRecord_params*>(p); event=a->hEvent; stream=a->hStream; }
  else if(s=="cuEventQuery") event=static_cast<const cuEventQuery_params*>(p)->hEvent;
  else if(s=="cuEventSynchronize") event=static_cast<const cuEventSynchronize_params*>(p)->hEvent;
  else if(s=="cuStreamWaitEvent") { const auto* a=static_cast<const cuStreamWaitEvent_params*>(p); event=a->hEvent; stream=a->hStream; entry.flags=a->Flags; }
  const bool has_stream=s=="cuLaunchKernel" || s=="cuMemcpyHtoDAsync_v2" || s=="cuMemcpyDtoHAsync_v2" ||
    s=="cuMemcpyDtoDAsync_v2" || s=="cuMemsetD8Async" || s=="cuStreamSynchronize" ||
    s=="cuStreamDestroy_v2" || s=="cuEventRecord" || s=="cuStreamWaitEvent";
  if(has_stream) {
    if(stream==CU_STREAM_PER_THREAD) entry.stream_kind="per_thread";
    else if(!stream || stream==CU_STREAM_LEGACY) entry.stream_kind="legacy";
    else entry.stream_kind="explicit";
    entry.stream=entry.stream_kind=="legacy" ? state->ids.identity("legacy_stream",entry.context) : state->id("stream",stream);
  }
  entry.event=state->id("event",event);
}
void footer() noexcept {
  if(!state || !state->mutex.try_lock()) return;
  try {
    if(!state->finished) {
      auto line=state->line("incomplete"); line.number("errors",state->errors);
      line.number("open_calls",state->pending.size()); state->emit(line);
    }
  } catch(...) { ++state->errors; }
  state->mutex.unlock();
}
}
void initialize() {
  wchar_t path[32768]{};
  const auto n=GetEnvironmentVariableW(L"XVRAM_EXECUTION_WITNESS_TRACE",path,32768);
  if(!n) return;
  if(n>=32768 || state) throw std::runtime_error("order_path_or_reinitialize");
  state=new State;
  state->file=CreateFileW(path,GENERIC_WRITE,FILE_SHARE_READ,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
  if(state->file==INVALID_HANDLE_VALUE) throw std::runtime_error("order_trace_create");
  state->emit(state->line("session"));
  if(std::atexit(footer)!=0) throw std::runtime_error("order_atexit");
}
bool enabled() noexcept { return state!=nullptr; }
void api(const CUpti_CallbackData& data, std::uint64_t api_id) noexcept {
  if(!state || !data.functionName) return;
  try {
    std::lock_guard lock(state->mutex);
    const bool exit=data.callbackSite==CUPTI_API_EXIT;
    if(!exit && state->sealed && !terminal_tool) {
      ++state->errors;
      auto late=state->line("late_producer"); late.number("api_id",api_id); state->emit(late);
    }
    if(!relevant(data.functionName)) return;
    Entry entry;
    int result=0;
    if(!exit) {
      entry.symbol=data.functionName;
      entry.thread=thread_token; // OS thread IDs may be reused after thread exit.
      entry.context=state->id("context",data.context); entry.tool=terminal_tool;
      inputs(entry,data.functionParams);
      if(state->pending.size()>=4096 || !state->pending.emplace(api_id,entry).second)
        throw std::runtime_error("order_pending_limit_or_duplicate");
    } else {
      const auto found=state->pending.find(api_id);
      if(found==state->pending.end() || !data.functionReturnValue) throw std::runtime_error("order_orphan_exit");
      entry=found->second; state->pending.erase(found);
      if(entry.symbol!=data.functionName) throw std::runtime_error("order_exit_symbol");
      result=*static_cast<const int*>(data.functionReturnValue);
      if(result==0) {
        const auto* p=data.functionParams;
        if(entry.symbol=="cuStreamCreate") {
          entry.stream=state->id("stream",*static_cast<const cuStreamCreate_params*>(p)->phStream);
          entry.stream_kind="explicit";
        }
        else if(entry.symbol=="cuEventCreate") entry.event=state->id("event",*static_cast<const cuEventCreate_params*>(p)->phEvent);
        else if(entry.symbol=="cuDevicePrimaryCtxRetain") entry.context=state->id("context",*static_cast<const cuDevicePrimaryCtxRetain_params*>(p)->pctx);
        else if(entry.symbol=="cuStreamGetGreenCtx") entry.flags=*static_cast<const cuStreamGetGreenCtx_params*>(p)->phCtx ? 1U : 0U;
      }
    }
    auto line=state->line(exit ? "api_exit" : "api_enter");
    line.number("api_id",api_id); line.number("thread_id",entry.thread); line.number("context_id",entry.context);
    line.number("stream_id",entry.stream); line.number("event_id",entry.event); line.number("flags",entry.flags);
    line.string("stream_kind",entry.stream_kind);
    line.string("symbol",entry.symbol); line.boolean("terminal_tool",entry.tool); line.integer("result",result);
    state->emit(line);
  } catch(...) { ++state->errors; }
}
void seal() {
  if(!state) return;
  std::lock_guard lock(state->mutex);
  if(state->sealed || !state->pending.empty() || state->errors) throw std::runtime_error("order_not_quiescent");
  state->sealed=true;
  auto line=state->line("seal"); line.number("open_calls",state->pending.size()); state->emit(line);
}
void finish(bool gpu_drained, bool census_drained) {
  if(!state) return;
  std::lock_guard lock(state->mutex);
  auto line=state->line("summary"); line.boolean("gpu_drained",gpu_drained);
  line.boolean("census_drained",census_drained); line.boolean("sealed",state->sealed);
  line.number("errors",state->errors); line.number("open_calls",state->pending.size());
  state->emit(line); state->finished=true;
  if(!FlushFileBuffers(state->file)) throw std::runtime_error("order_flush_failed");
  if(state->errors || !state->pending.empty() || !state->sealed || !gpu_drained || !census_drained)
    throw std::runtime_error("order_terminal_failure");
}
}
