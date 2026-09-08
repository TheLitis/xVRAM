#include "kernel_capture.hpp"
#include "capture_catalog.hpp"
#include "memory_witness.hpp"
#include "compat_audit_collector/trace_core.hpp"
#include <windows.h>
#include <bit>
#include <cstdlib>
#include <map>
#include <mutex>

namespace xvram::launch_probe::kernel_capture {
namespace {
namespace tc=typed_capture;
constexpr std::uint64_t byte_cap=512ULL*1024*1024;
constexpr std::uint64_t call_cap=100000, field_cap=4000000;
struct State {
  HANDLE file=INVALID_HANDLE_VALUE;
  std::mutex mutex;
  std::uint64_t sequence=0, bytes=0, last_call=0, calls=0, fields=0;
  std::uint64_t unsupported=0, faults=0, unresolved=0, errors=0, returns=0;
  bool finished=false;
  std::map<std::uint64_t,bool> pending;
  audit::JsonLine line(const char* kind, std::uint64_t call=0) {
    audit::JsonLine line;
    line.number("schema_version",1);
    line.string("record_type","xvram.cuda_kernel_arguments");
    line.number("sequence",++sequence); line.string("kind",kind);
    line.number("call",call);
    return line;
  }
  void emit(const audit::JsonLine& line) {
    const auto text=line.finish();
    if(text.size()>byte_cap-bytes) throw std::runtime_error("kernel_capture_capacity");
    DWORD written=0;
    if(!WriteFile(file,text.data(),static_cast<DWORD>(text.size()),&written,nullptr) || written!=text.size())
      throw std::runtime_error("kernel_capture_write");
    bytes+=text.size();
  }
};
State* state=nullptr;
void read_host(const void* source, void* target, std::size_t bytes) {
  SIZE_T copied=0;
  if(!ReadProcessMemory(GetCurrentProcess(),source,target,bytes,&copied) || copied!=bytes)
    throw std::runtime_error("kernel_capture_host_read_fault");
}
const char* type_name(tc::ValueType type) {
  switch(type) {
    case tc::ValueType::ptr: return "ptr";
    case tc::ValueType::i64: return "i64";
    case tc::ValueType::u64: return "u64";
    case tc::ValueType::i32: return "i32";
    case tc::ValueType::u32: return "u32";
    case tc::ValueType::f32: return "f32";
    case tc::ValueType::boolean: return "bool";
  }
  throw std::runtime_error("kernel_capture_type");
}
struct Sink {
  std::uint64_t call;
  int device;
  CUcontext context;
  std::vector<audit::JsonLine> rows;
  std::uint64_t unknown=0;
  audit::JsonLine field(std::size_t ordinal,const char* argument,const char* name,tc::ValueType type) {
    auto line=state->line("field",call);
    line.number("ordinal",ordinal); line.string("argument",argument);
    line.string("field",name); line.string("value_type",type_name(type));
    return line;
  }
  void pointer(std::size_t ordinal,const char* argument,const char* name,std::uint64_t native) {
    auto line=field(ordinal,argument,name,tc::ValueType::ptr);
    // This lookup establishes identity at a single byte only. Tensor extents,
    // access modes, indirect indices and cubin semantics remain independent.
    const auto resolved=native && context && device>=0 ? memory_witness::resolve(native,1,device,context) : memory_witness::Resolution{};
    const bool is_null=native==0;
    line.string("resolution",is_null ? "null" : resolved.known ? "allocation_identity" : "unresolved");
    line.boolean("memory_bounds_proven",false);
    if(resolved.known) {
      line.number("allocation_id",resolved.allocation_id); line.number("generation",resolved.generation);
      line.number("offset_bytes",resolved.offset_bytes); line.number("allocation_bytes",resolved.allocation_bytes);
      line.number("lookup_bytes",1); line.number("memory_revision",resolved.revision);
      line.number("memory_sequence",resolved.observed_sequence);
      line.number("alignment",resolved.alignment); line.number("base_mod_alignment",resolved.base_mod_alignment);
      line.boolean("mapped",resolved.mapped); line.boolean("readable",resolved.readable);
      line.boolean("writable",resolved.writable);
    } else if(!is_null) ++unknown;
    rows.push_back(std::move(line));
  }
  void scalar(std::size_t ordinal,const char* argument,const char* name,tc::ValueType type,std::uint64_t bits) {
    auto line=field(ordinal,argument,name,type);
    switch(type) {
      case tc::ValueType::i64: line.integer("value",std::bit_cast<std::int64_t>(bits)); break;
      case tc::ValueType::i32: line.integer("value",std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(bits))); break;
      // Exact IEEE-754 storage, including non-finite values, without invalid
      // JSON numbers or locale-sensitive conversion. These are reviewed f32
      // fields, never reinterpretations of pointers or struct padding.
      case tc::ValueType::f32: line.number("bits_u32",static_cast<std::uint32_t>(bits)); break;
      case tc::ValueType::boolean: line.boolean("value",bits!=0); break;
      case tc::ValueType::u32: case tc::ValueType::u64: line.number("value",bits); break;
      case tc::ValueType::ptr: throw std::runtime_error("kernel_capture_pointer_scalar");
    }
    rows.push_back(std::move(line));
  }
};
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
  const auto size=GetEnvironmentVariableW(L"XVRAM_KERNEL_ARGUMENT_TRACE",path,32768);
  if(!size) return;
  if(size>=32768 || state) throw std::runtime_error("kernel_capture_path_or_reinitialize");
  state=new State;
  state->file=CreateFileW(path,GENERIC_WRITE,FILE_SHARE_READ,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
  if(state->file==INVALID_HANDLE_VALUE) throw std::runtime_error("kernel_capture_create");
  auto line=state->line("session");
  line.string("catalog_sha256",tc::catalog_sha256); line.number("byte_cap",byte_cap);
  line.number("call_cap",call_cap); line.number("field_cap",field_cap);
  line.boolean("memory_observer_enabled",memory_witness::enabled());
  line.boolean("memory_bounds_proven",false); line.boolean("source_semantics_bound_to_cubin",false);
  state->emit(line);
  if(std::atexit(footer)!=0) throw std::runtime_error("kernel_capture_atexit");
}
bool enabled() noexcept { return state!=nullptr; }
bool healthy() noexcept {
  if(!state) return true;
  try { std::lock_guard lock(state->mutex); return state->errors==0 && state->faults==0; }
  catch(...) { return false; }
}
void launch(std::uint64_t call,const Snapshot& snapshot,const Geometry& geometry,
            const void* parameters,const void* extra,int device,CUcontext context) {
  if(!state) return;
  std::lock_guard lock(state->mutex);
  try {
    if(state->finished || !call || call<=state->last_call || state->calls>=call_cap || !state->pending.empty())
      throw std::runtime_error("kernel_capture_call_order_or_limit");
    state->last_call=call; ++state->calls; state->pending.emplace(call,true);
    const tc::Kernel* kernel=nullptr;
    for(const auto& candidate:tc::kernels) if(snapshot.name==candidate.symbol) { kernel=&candidate; break; }
    auto begin=state->line("begin",call);
    begin.number("catalog_id",kernel ? kernel->id : 0);
    begin.number("grid_x",geometry.grid[0]); begin.number("grid_y",geometry.grid[1]); begin.number("grid_z",geometry.grid[2]);
    begin.number("block_x",geometry.block[0]); begin.number("block_y",geometry.block[1]); begin.number("block_z",geometry.block[2]);
    begin.number("shared_bytes",geometry.shared_bytes); begin.integer("device",device);
    begin.boolean("memory_observer_healthy",memory_witness::healthy());
    state->emit(begin);
    const char* status="captured";
    if(!kernel) status="unsupported_symbol";
    else if(!kernel->supported) status="unsupported_opaque_library";
    else if(!tc::matches(*kernel,snapshot)) status="unsupported_abi";
    else if(!parameters || extra) status="unsupported_argument_bank";
    Sink sink{call,device,context,{},0};
    if(std::string_view(status)=="captured") {
      // Validation/host reads/identity resolutions build a private batch first.
      // No partial field snapshot is published on host read failure.
      const auto before=state->sequence;
      try { tc::capture(*kernel,snapshot,parameters,extra,read_host,sink); }
      catch(...) { state->sequence=before; sink.rows.clear(); sink.unknown=0; ++state->faults; status="capture_fault"; }
      if(sink.rows.size()>field_cap-state->fields) throw std::runtime_error("kernel_capture_field_limit");
      for(const auto& row:sink.rows) state->emit(row);
      state->fields+=sink.rows.size(); state->unresolved+=sink.unknown;
    } else ++state->unsupported;
    auto end=state->line("capture_end",call); end.string("status",status);
    end.number("fields",sink.rows.size()); end.number("unresolved_pointers",sink.unknown);
    end.boolean("memory_bounds_proven",false); state->emit(end);
  } catch(...) { ++state->errors; throw; }
}
void returned(std::uint64_t call,int result) {
  if(!state) return;
  std::lock_guard lock(state->mutex);
  try {
    if(state->finished || state->pending.erase(call)!=1) throw std::runtime_error("kernel_capture_orphan_return");
    auto line=state->line("return",call); line.integer("result",result); state->emit(line); ++state->returns;
  } catch(...) { ++state->errors; throw; }
}
void finish() {
  if(!state) return;
  std::lock_guard lock(state->mutex);
  try {
    if(state->finished || !state->pending.empty()) throw std::runtime_error("kernel_capture_finish_state");
    auto line=state->line("summary"); line.number("calls",state->calls); line.number("returns",state->returns);
    line.number("fields",state->fields); line.number("unsupported",state->unsupported);
    line.number("capture_faults",state->faults); line.number("unresolved_pointers",state->unresolved);
    line.number("errors",state->errors); line.boolean("terminal_complete",false);
    line.boolean("memory_bounds_proven",false); line.boolean("source_semantics_bound_to_cubin",false);
    state->emit(line);
    if(!FlushFileBuffers(state->file)) throw std::runtime_error("kernel_capture_flush");
    if(!CloseHandle(state->file)) throw std::runtime_error("kernel_capture_close");
    state->file=INVALID_HANDLE_VALUE;
    state->finished=true;
  } catch(...) { ++state->errors; throw; }
}
}
