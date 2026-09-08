// Diagnostic host observations only: no CUDA/CUPTI calls, no argument mutation.
#include "teardown_witness.hpp"
#include "compat_audit_collector/trace_core.hpp"
#include <windows.h>
#include <atomic>
#include <cstdlib>
#include <map>
#include <mutex>
#include <optional>
#include <string_view>
#include <type_traits>

namespace xvram::launch_probe::teardown_witness {
namespace {
constexpr std::uint64_t max_api=2000000, max_objects=100000, max_records=200000;
constexpr std::uint64_t byte_cap=64ULL*1024*1024;
enum class Operation { Ignore, LoadData, LoadFile, Unload, Retain, Release, ReleaseV2, Unsupported };
struct Entry {
  std::uint64_t api=0, producer=0;
  Operation operation=Operation::Ignore;
  CUlibrary library=nullptr;
  CUlibrary* library_output=nullptr;
  CUcontext* context_output=nullptr;
  CUdevice device=0;
};
struct Library { std::uint64_t id=0, generation=0; bool live=false; };
struct Primary { CUcontext native=nullptr; std::uint64_t id=0, generation=1, references=0; };
struct State {
  HANDLE file=INVALID_HANDLE_VALUE;
  std::mutex mutex;
  std::atomic_uint64_t errors{0};
  std::map<std::uintptr_t, Library> libraries;
  std::map<int, Primary> primaries;
  std::optional<Entry> pending;
  Snapshot counts;
  std::uint64_t bytes=0, last_ticks=0, last_entered_api=0, last_seen_api=0, producer=0;
  std::uint64_t next_library=1, next_context=1;
  audit::JsonLine line(const char* kind) {
    LARGE_INTEGER now{};
    if(!QueryPerformanceCounter(&now) || now.QuadPart<0 ||
       static_cast<std::uint64_t>(now.QuadPart)<last_ticks || counts.sequence>=max_records)
      throw std::runtime_error("teardown_clock_or_record_limit");
    last_ticks=static_cast<std::uint64_t>(now.QuadPart);
    audit::JsonLine row;
    row.number("schema_version",1); row.string("record_type","xvram.cuda_teardown_witness");
    row.number("sequence",++counts.sequence); row.string("kind",kind);
    row.number("qpc_ticks",last_ticks); return row;
  }
  void emit(const audit::JsonLine& row) {
    const auto text=row.finish();
    if(text.size()>65536 || bytes>byte_cap || text.size()>byte_cap-bytes)
      throw std::runtime_error("teardown_trace_capacity");
    DWORD written=0;
    if(!WriteFile(file,text.data(),static_cast<DWORD>(text.size()),&written,nullptr) || written!=text.size())
      throw std::runtime_error("teardown_trace_write");
    bytes+=text.size();
  }
};
// Neither state nor file is destroyed by a C++ static destructor. The controller
// reads the retained file after owned worker reap; no self-issued final exists.
State* state=nullptr;
std::atomic_uint64_t next_producer{0};
thread_local const std::uint64_t producer_token=++next_producer;

template<class T> T copy(const T* source) {
  static_assert(std::is_trivially_copyable_v<T> && sizeof(T)<=128);
  T value{}; SIZE_T read=0;
  if(!source || !ReadProcessMemory(GetCurrentProcess(),source,&value,sizeof(value),&read) || read!=sizeof(value))
    throw std::runtime_error("teardown_host_snapshot");
  return value;
}
Operation operation(std::string_view name) {
  if(name=="cuLibraryLoadData") return Operation::LoadData;
  if(name=="cuLibraryLoadFromFile") return Operation::LoadFile;
  if(name=="cuLibraryUnload") return Operation::Unload;
  if(name=="cuDevicePrimaryCtxRetain") return Operation::Retain;
  if(name=="cuDevicePrimaryCtxRelease") return Operation::Release;
  if(name=="cuDevicePrimaryCtxRelease_v2") return Operation::ReleaseV2;
  if(name.starts_with("cuLibraryLoad") || name.starts_with("cuLibraryUnload") ||
     name.starts_with("cuDevicePrimaryCtxRetain") || name.starts_with("cuDevicePrimaryCtxRelease") ||
     name.starts_with("cuDevicePrimaryCtxReset") || name.starts_with("cuCtxDestroy")) return Operation::Unsupported;
  return Operation::Ignore;
}
const char* symbol(Operation op) {
  switch(op) {
    case Operation::LoadData: return "cuLibraryLoadData";
    case Operation::LoadFile: return "cuLibraryLoadFromFile";
    case Operation::Unload: return "cuLibraryUnload";
    case Operation::Retain: return "cuDevicePrimaryCtxRetain";
    case Operation::Release: return "cuDevicePrimaryCtxRelease";
    case Operation::ReleaseV2: return "cuDevicePrimaryCtxRelease_v2";
    default: throw std::runtime_error("teardown_unknown_operation");
  }
}
Entry inputs(Operation op, const void* params) {
  Entry entry; entry.operation=op;
  switch(op) {
    case Operation::LoadData: entry.library_output=copy(static_cast<const cuLibraryLoadData_params*>(params)).library; break;
    case Operation::LoadFile: entry.library_output=copy(static_cast<const cuLibraryLoadFromFile_params*>(params)).library; break;
    case Operation::Unload: entry.library=copy(static_cast<const cuLibraryUnload_params*>(params)).library; break;
    case Operation::Retain: {
      const auto value=copy(static_cast<const cuDevicePrimaryCtxRetain_params*>(params));
      entry.context_output=value.pctx; entry.device=value.dev; break;
    }
    case Operation::Release: entry.device=copy(static_cast<const cuDevicePrimaryCtxRelease_params*>(params)).dev; break;
    case Operation::ReleaseV2: entry.device=copy(static_cast<const cuDevicePrimaryCtxRelease_v2_params*>(params)).dev; break;
    default: throw std::runtime_error("teardown_untyped_operation");
  }
  if(entry.device<0 || entry.device>127) throw std::runtime_error("teardown_device_range");
  if((op==Operation::LoadData || op==Operation::LoadFile) && !entry.library_output)
    throw std::runtime_error("teardown_library_output_missing");
  if(op==Operation::Unload && !entry.library) throw std::runtime_error("teardown_null_library");
  if(op==Operation::Retain && !entry.context_output) throw std::runtime_error("teardown_context_output_missing");
  return entry;
}
bool same_inputs(const Entry& a, const Entry& b) {
  return a.operation==b.operation && a.library==b.library && a.library_output==b.library_output &&
         a.context_output==b.context_output && a.device==b.device;
}
const char* reason(std::string_view message) noexcept {
  if(message=="teardown_host_snapshot") return "host_snapshot_failed";
  if(message=="teardown_failed_native_api") return "native_api_failed";
  if(message=="teardown_unmatched_primary_release") return "primary_reference_unmatched";
  if(message=="teardown_unproved_context_recreation" || message=="teardown_context_device_alias")
    return "context_generation_unproved";
  if(message=="teardown_stale_library" || message=="teardown_library_reuse") return "library_generation_invalid";
  if(message=="teardown_multiple_producers") return "multiple_producers";
  if(message=="teardown_overlapping_lifecycles") return "overlapping_lifecycles";
  if(message=="teardown_unknown_lifecycle_or_tail") return "unsupported_lifecycle_or_tail";
  if(message=="teardown_api_aba" || message=="teardown_exit_identity") return "api_identity_invalid";
  if(message=="teardown_device_range") return "device_out_of_profile";
  if(message=="teardown_callback_header") return "callback_header_invalid";
  if(message=="teardown_null_library_output" || message=="teardown_null_context_output" ||
     message=="teardown_library_output_missing" || message=="teardown_context_output_missing" ||
     message=="teardown_null_library") return "required_native_value_missing";
  return "resource_or_internal_failure";
}
void error_record(std::uint64_t api_id, const char* category) noexcept {
  ++state->errors;
  try {
    auto row=state->line("error"); row.number("api_id",api_id);
    row.number("errors",state->errors); row.string("reason",category);
    state->emit(row);
  } catch(...) { ++state->errors; }
}
void complete(const Entry& entry) {
  std::uint64_t lib_id=0, lib_generation=0, ctx_id=0, ctx_generation=0;
  const auto op=entry.operation;
  if(op==Operation::LoadData || op==Operation::LoadFile) {
    const auto native=copy(entry.library_output);
    if(!native) throw std::runtime_error("teardown_null_library_output");
    const auto key=reinterpret_cast<std::uintptr_t>(native);
    auto found=state->libraries.find(key);
    if(found==state->libraries.end()) {
      if(state->libraries.size()>=max_objects) throw std::runtime_error("teardown_library_capacity");
      found=state->libraries.emplace(key,Library{state->next_library++,0,false}).first;
    }
    auto& lib=found->second;
    if(lib.live || lib.generation>=max_objects) throw std::runtime_error("teardown_library_reuse");
    ++lib.generation; lib.live=true; lib_id=lib.id; lib_generation=lib.generation;
    ++state->counts.library_loads; ++state->counts.libraries_live;
  } else if(op==Operation::Unload) {
    const auto found=state->libraries.find(reinterpret_cast<std::uintptr_t>(entry.library));
    if(found==state->libraries.end() || !found->second.live) throw std::runtime_error("teardown_stale_library");
    lib_id=found->second.id; lib_generation=found->second.generation; found->second.live=false;
    ++state->counts.library_unloads; --state->counts.libraries_live;
  } else if(op==Operation::Retain) {
    const auto native=copy(entry.context_output);
    if(!native) throw std::runtime_error("teardown_null_context_output");
    auto found=state->primaries.find(entry.device);
    if(found==state->primaries.end()) {
      for(const auto& [device, primary]:state->primaries) {
        (void)device;
        if(primary.native==native) throw std::runtime_error("teardown_context_device_alias");
      }
      found=state->primaries.emplace(entry.device,Primary{native,state->next_context++,1,0}).first;
    } else if(found->second.native!=native || !found->second.references) {
      // Reference balance reaching zero did not prove context destruction.
      // No invented generation on either same-address or changed-address retain.
      throw std::runtime_error("teardown_unproved_context_recreation");
    }
    auto& primary=found->second;
    if(primary.references>=max_api) throw std::runtime_error("teardown_reference_capacity");
    ++primary.references; ctx_id=primary.id; ctx_generation=primary.generation;
    ++state->counts.primary_retains; ++state->counts.primary_references;
  } else {
    const auto found=state->primaries.find(entry.device);
    if(found==state->primaries.end() || !found->second.references)
      throw std::runtime_error("teardown_unmatched_primary_release");
    // A successful release is observable even when library generations remain
    // loaded. Report the true reference balance; offline policy must reject
    // missing library retirement rather than suppress this real API result.
    auto& primary=found->second; --primary.references;
    ctx_id=primary.id; ctx_generation=primary.generation;
    ++state->counts.primary_releases; --state->counts.primary_references;
  }
  auto row=state->line("lifecycle"); row.number("api_id",entry.api);
  row.number("errors",state->errors);
  row.number("producer_id",entry.producer); row.string("symbol",symbol(op)); row.integer("result",0);
  row.number("library_id",lib_id); row.number("library_generation",lib_generation);
  row.number("context_id",ctx_id); row.number("context_generation",ctx_generation);
  row.integer("device",entry.device); state->emit(row);
  ++state->counts.lifecycle_pairs;
  if(state->counts.checkpointed) ++state->counts.after_checkpoint_pairs;
}
}

void initialize() {
  wchar_t path[32768]{};
  const auto n=GetEnvironmentVariableW(L"XVRAM_TEARDOWN_WITNESS_TRACE",path,32768);
  if(!n) return;
  if(n>=32768 || state) throw std::runtime_error("teardown_path_or_reinitialize");
  auto* created=new State;
  LARGE_INTEGER frequency{};
  if(!QueryPerformanceFrequency(&frequency) || frequency.QuadPart<=0 || frequency.QuadPart>1000000000000LL) {
    delete created; throw std::runtime_error("teardown_qpc_frequency");
  }
  created->counts.qpc_frequency=static_cast<std::uint64_t>(frequency.QuadPart);
  created->file=CreateFileW(path,GENERIC_WRITE,FILE_SHARE_READ,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
  if(created->file==INVALID_HANDLE_VALUE) { delete created; throw std::runtime_error("teardown_trace_create"); }
  state=created;
  auto row=state->line("session"); row.number("qpc_frequency",state->counts.qpc_frequency);
  row.boolean("terminal_complete",false); row.boolean("context_destruction_proven",false);
  state->emit(row);
  if(std::atexit(checkpoint)!=0) throw std::runtime_error("teardown_atexit");
}
bool enabled() noexcept { return state!=nullptr; }
bool healthy() noexcept { return state && state->errors==0; }
void api(const CUpti_CallbackData& data, std::uint64_t api_id) noexcept {
  if(!state) return;
  // A callback can occur while the loader detaches DLLs. Never wait on a mutex
  // held by another (possibly terminated) producer. Missing typed rows fail the
  // controller's census reconciliation even if a later error row cannot flush.
  std::unique_lock lock(state->mutex,std::try_to_lock);
  if(!lock.owns_lock()) { ++state->errors; return; }
  try {
    if(!api_id || api_id>max_api || !data.functionName ||
       (data.callbackSite!=CUPTI_API_ENTER && data.callbackSite!=CUPTI_API_EXIT))
      throw std::runtime_error("teardown_callback_header");
    const auto op=operation(data.functionName);
    const bool enter=data.callbackSite==CUPTI_API_ENTER;
    if(enter) {
      if(api_id<=state->last_entered_api) throw std::runtime_error("teardown_api_aba");
      state->last_entered_api=api_id;
    }
    if(api_id>state->last_seen_api) state->last_seen_api=api_id;
    if(op==Operation::Unsupported || (state->counts.checkpointed && op!=Operation::Unload &&
                                    op!=Operation::Release && op!=Operation::ReleaseV2))
      throw std::runtime_error("teardown_unknown_lifecycle_or_tail");
    if(op==Operation::Ignore) return;
    if(!producer_token || producer_token>max_objects || (state->producer && producer_token!=state->producer))
      throw std::runtime_error("teardown_multiple_producers");
    state->producer=producer_token;
    auto entry=inputs(op,data.functionParams);
    if(enter) {
      if(state->pending) throw std::runtime_error("teardown_overlapping_lifecycles");
      entry.api=api_id; entry.producer=producer_token; state->pending=entry;
    } else {
      if(!state->pending || state->pending->api!=api_id ||
         state->pending->producer!=producer_token || !same_inputs(*state->pending,entry))
        throw std::runtime_error("teardown_exit_identity");
      if(copy(static_cast<const int*>(data.functionReturnValue))!=0)
        throw std::runtime_error("teardown_failed_native_api");
      complete(*state->pending); state->pending.reset();
    }
  } catch(const std::exception& error) { error_record(api_id<=max_api ? api_id : 0,reason(error.what())); }
  catch(...) { error_record(api_id<=max_api ? api_id : 0,"resource_or_internal_failure"); }
}
void checkpoint() noexcept {
  if(!state) return;
  std::unique_lock lock(state->mutex,std::try_to_lock);
  if(!lock.owns_lock()) { ++state->errors; return; }
  if(state->counts.checkpointed) return;
  try {
    if(state->pending) ++state->errors;
    state->counts.checkpointed=true;
    auto row=state->line("checkpoint"); row.number("last_seen_driver_api_id",state->last_seen_api);
    row.number("lifecycle_pairs",state->counts.lifecycle_pairs); row.number("pending_lifecycles",state->pending ? 1U : 0U);
    row.number("errors",state->errors); row.boolean("terminal_complete",false); state->emit(row);
    // File/state stay live. No Flush/Close/Wait/CUDA call under DLL teardown.
  } catch(...) { ++state->errors; }
}
Snapshot snapshot() noexcept {
  Snapshot result;
  if(!state) return result;
  result.enabled=true;
  std::unique_lock lock(state->mutex,std::try_to_lock);
  if(!lock.owns_lock()) { result.errors=state->errors+1; return result; }
  result=state->counts; result.enabled=true; result.errors=state->errors;
  result.healthy=result.errors==0; result.pending=state->pending ? 1U : 0U; return result;
}
}
