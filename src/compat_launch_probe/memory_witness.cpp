#include "memory_witness.hpp"
#include "compat_audit_collector/trace_core.hpp"
#include <windows.h>
#include <array>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <string>

namespace xvram::launch_probe::memory_witness {
namespace {
constexpr Word trace_cap=256ULL*1024*1024, record_cap=4000000, api_cap=2000000;
enum class Operation { allocate, free, reserve, free_reservation, create, release,
                       map, unmap, access, h2d, d2h, d2d, memset };
struct Parameters {
  Operation operation=Operation::allocate;
  Word address=0, source=0, bytes=0, handle=0, offset=0, flags=0, context=0;
  int device=-1;
  void* output=nullptr; // Callback-lifetime host output, never serialized.
  std::array<CUmemAccessDesc,16> access{};
  std::size_t access_count=0;
  std::string symbol;
};
struct State {
  HANDLE file=INVALID_HANDLE_VALUE;
  std::mutex mutex;
  MemoryRegistry registry;
  std::map<Word,Parameters> pending;
  Word sequence=0, bytes=0, api_count=0, failures=0, last_api=0;
  std::atomic_uint64_t errors{0};
  bool closed=false;
  audit::JsonLine line(const char* kind) {
    if (sequence>=record_cap) throw std::runtime_error("memory_record_capacity");
    audit::JsonLine line;
    line.number("schema_version",1); line.string("record_type","xvram.cuda_memory_witness");
    line.number("sequence",++sequence); line.string("kind",kind);
    return line;
  }
  void emit(const audit::JsonLine& line) {
    const auto value=line.finish();
    if (closed || bytes>trace_cap || value.size()>trace_cap-bytes)
      throw std::runtime_error("memory_trace_capacity");
    DWORD written=0;
    if (!WriteFile(file,value.data(),static_cast<DWORD>(value.size()),&written,nullptr)
        || written!=value.size()) throw std::runtime_error("memory_trace_write");
    bytes+=value.size();
  }
};
State* state=nullptr;

bool memory_mutation(std::string_view symbol) {
  // Host-pinned allocation tracking is a separate unresolved audit surface.
  // A device pointer derived from host mapping still resolves unknown here.
  if (symbol=="cuMemFreeHost" || symbol.starts_with("cuMemAllocHost")) return false;
  return symbol.starts_with("cuMemAlloc") || symbol.starts_with("cuMemFree")
    || symbol.starts_with("cuMemAddress") || symbol.starts_with("cuMemCreate")
    || symbol.starts_with("cuMemRelease") || symbol.starts_with("cuMemMap")
    || symbol.starts_with("cuMemUnmap") || symbol.starts_with("cuMemSetAccess")
    || symbol.starts_with("cuMemImport") || symbol.starts_with("cuMemExport")
    || symbol.starts_with("cuMemRetain") || symbol.starts_with("cuMemcpy")
    || symbol.starts_with("cuMemset");
}
Parameters parameters(const CUpti_CallbackData& data) {
  if (!data.functionParams) throw std::runtime_error("memory_parameters_missing");
  Parameters out;
  out.symbol=data.functionName;
  out.context=reinterpret_cast<std::uintptr_t>(data.context);
  const auto* p=data.functionParams;
  const auto& s=out.symbol;
  if (s=="cuMemAlloc_v2") {
    const auto& a=*static_cast<const cuMemAlloc_v2_params*>(p);
    out.operation=Operation::allocate; out.output=a.dptr; out.bytes=a.bytesize;
  } else if (s=="cuMemFree_v2") {
    out.operation=Operation::free; out.address=static_cast<const cuMemFree_v2_params*>(p)->dptr;
  } else if (s=="cuMemAddressReserve") {
    const auto& a=*static_cast<const cuMemAddressReserve_params*>(p);
    out.operation=Operation::reserve; out.output=a.ptr; out.bytes=a.size; out.flags=a.flags;
  } else if (s=="cuMemAddressFree") {
    const auto& a=*static_cast<const cuMemAddressFree_params*>(p);
    out.operation=Operation::free_reservation; out.address=a.ptr; out.bytes=a.size;
  } else if (s=="cuMemCreate") {
    const auto& a=*static_cast<const cuMemCreate_params*>(p);
    if (!a.prop || a.prop->type!=CU_MEM_ALLOCATION_TYPE_PINNED
        || a.prop->location.type!=CU_MEM_LOCATION_TYPE_DEVICE || a.prop->location.id<0
        || a.prop->allocFlags.usage) throw std::runtime_error("memory_create_profile");
    out.operation=Operation::create; out.output=a.handle; out.bytes=a.size;
    out.flags=a.flags; out.device=a.prop->location.id;
  } else if (s=="cuMemRelease") {
    out.operation=Operation::release; out.handle=static_cast<const cuMemRelease_params*>(p)->handle;
  } else if (s=="cuMemMap") {
    const auto& a=*static_cast<const cuMemMap_params*>(p);
    out.operation=Operation::map; out.address=a.ptr; out.bytes=a.size;
    out.offset=a.offset; out.handle=a.handle; out.flags=a.flags;
  } else if (s=="cuMemUnmap") {
    const auto& a=*static_cast<const cuMemUnmap_params*>(p);
    out.operation=Operation::unmap; out.address=a.ptr; out.bytes=a.size;
  } else if (s=="cuMemSetAccess") {
    const auto& a=*static_cast<const cuMemSetAccess_params*>(p);
    if (!a.desc || !a.count || a.count>out.access.size()) throw std::runtime_error("memory_access_count");
    out.operation=Operation::access; out.address=a.ptr; out.bytes=a.size; out.access_count=a.count;
    for (std::size_t i=0;i<a.count;++i) {
      out.access[i]=a.desc[i];
      const auto& desc=out.access[i];
      if (desc.location.type!=CU_MEM_LOCATION_TYPE_DEVICE || desc.location.id<0
          || (desc.flags!=CU_MEM_ACCESS_FLAGS_PROT_NONE && desc.flags!=CU_MEM_ACCESS_FLAGS_PROT_READ
              && desc.flags!=CU_MEM_ACCESS_FLAGS_PROT_READWRITE)) throw std::runtime_error("memory_access_profile");
      for (std::size_t j=0;j<i;++j)
        if (out.access[j].location.id==desc.location.id) throw std::runtime_error("memory_duplicate_access_device");
    }
  } else if (s=="cuMemcpyHtoD_v2" || s=="cuMemcpyHtoD_v2_ptds") {
    const auto capture=[&](const auto& a) { out.operation=Operation::h2d; out.address=a.dstDevice; out.bytes=a.ByteCount; };
    if (s=="cuMemcpyHtoD_v2") capture(*static_cast<const cuMemcpyHtoD_v2_params*>(p));
    else capture(*static_cast<const cuMemcpyHtoD_v2_ptds_params*>(p));
  } else if (s=="cuMemcpyHtoDAsync_v2" || s=="cuMemcpyHtoDAsync_v2_ptsz") {
    const auto capture=[&](const auto& a) { out.operation=Operation::h2d; out.address=a.dstDevice; out.bytes=a.ByteCount; };
    if (s=="cuMemcpyHtoDAsync_v2") capture(*static_cast<const cuMemcpyHtoDAsync_v2_params*>(p));
    else capture(*static_cast<const cuMemcpyHtoDAsync_v2_ptsz_params*>(p));
  } else if (s=="cuMemcpyDtoH_v2" || s=="cuMemcpyDtoH_v2_ptds") {
    const auto capture=[&](const auto& a) { out.operation=Operation::d2h; out.address=a.srcDevice; out.bytes=a.ByteCount; };
    if (s=="cuMemcpyDtoH_v2") capture(*static_cast<const cuMemcpyDtoH_v2_params*>(p));
    else capture(*static_cast<const cuMemcpyDtoH_v2_ptds_params*>(p));
  } else if (s=="cuMemcpyDtoHAsync_v2" || s=="cuMemcpyDtoHAsync_v2_ptsz") {
    const auto capture=[&](const auto& a) { out.operation=Operation::d2h; out.address=a.srcDevice; out.bytes=a.ByteCount; };
    if (s=="cuMemcpyDtoHAsync_v2") capture(*static_cast<const cuMemcpyDtoHAsync_v2_params*>(p));
    else capture(*static_cast<const cuMemcpyDtoHAsync_v2_ptsz_params*>(p));
  } else if (s=="cuMemcpyDtoD_v2" || s=="cuMemcpyDtoD_v2_ptds") {
    const auto capture=[&](const auto& a) { out.operation=Operation::d2d; out.address=a.dstDevice; out.source=a.srcDevice; out.bytes=a.ByteCount; };
    if (s=="cuMemcpyDtoD_v2") capture(*static_cast<const cuMemcpyDtoD_v2_params*>(p));
    else capture(*static_cast<const cuMemcpyDtoD_v2_ptds_params*>(p));
  } else if (s=="cuMemcpyDtoDAsync_v2" || s=="cuMemcpyDtoDAsync_v2_ptsz") {
    const auto capture=[&](const auto& a) { out.operation=Operation::d2d; out.address=a.dstDevice; out.source=a.srcDevice; out.bytes=a.ByteCount; };
    if (s=="cuMemcpyDtoDAsync_v2") capture(*static_cast<const cuMemcpyDtoDAsync_v2_params*>(p));
    else capture(*static_cast<const cuMemcpyDtoDAsync_v2_ptsz_params*>(p));
  } else if (s=="cuMemsetD8_v2" || s=="cuMemsetD8_v2_ptds") {
    const auto capture=[&](const auto& a) { out.operation=Operation::memset; out.address=a.dstDevice; out.bytes=a.N; };
    if (s=="cuMemsetD8_v2") capture(*static_cast<const cuMemsetD8_v2_params*>(p));
    else capture(*static_cast<const cuMemsetD8_v2_ptds_params*>(p));
  } else if (s=="cuMemsetD8Async" || s=="cuMemsetD8Async_ptsz") {
    const auto capture=[&](const auto& a) { out.operation=Operation::memset; out.address=a.dstDevice; out.bytes=a.N; };
    if (s=="cuMemsetD8Async") capture(*static_cast<const cuMemsetD8Async_params*>(p));
    else capture(*static_cast<const cuMemsetD8Async_ptsz_params*>(p));
  } else if (s=="cuMemsetD16_v2" || s=="cuMemsetD16_v2_ptds") {
    const auto capture=[&](const auto& a) { out.operation=Operation::memset; out.address=a.dstDevice; out.bytes=sum(a.N,a.N); };
    if (s=="cuMemsetD16_v2") capture(*static_cast<const cuMemsetD16_v2_params*>(p));
    else capture(*static_cast<const cuMemsetD16_v2_ptds_params*>(p));
  } else if (s=="cuMemsetD16Async" || s=="cuMemsetD16Async_ptsz") {
    const auto capture=[&](const auto& a) { out.operation=Operation::memset; out.address=a.dstDevice; out.bytes=sum(a.N,a.N); };
    if (s=="cuMemsetD16Async") capture(*static_cast<const cuMemsetD16Async_params*>(p));
    else capture(*static_cast<const cuMemsetD16Async_ptsz_params*>(p));
  } else if (s=="cuMemsetD32_v2" || s=="cuMemsetD32_v2_ptds") {
    const auto capture=[&](const auto& a) { out.operation=Operation::memset; out.address=a.dstDevice; out.bytes=sum(sum(a.N,a.N),sum(a.N,a.N)); };
    if (s=="cuMemsetD32_v2") capture(*static_cast<const cuMemsetD32_v2_params*>(p));
    else capture(*static_cast<const cuMemsetD32_v2_ptds_params*>(p));
  } else if (s=="cuMemsetD32Async" || s=="cuMemsetD32Async_ptsz") {
    const auto capture=[&](const auto& a) { out.operation=Operation::memset; out.address=a.dstDevice; out.bytes=sum(sum(a.N,a.N),sum(a.N,a.N)); };
    if (s=="cuMemsetD32Async") capture(*static_cast<const cuMemsetD32Async_params*>(p));
    else capture(*static_cast<const cuMemsetD32Async_ptsz_params*>(p));
  } else throw std::runtime_error("memory_unsupported_api");
  return out;
}
void add_resolution(audit::JsonLine& line, const Resolution& value, std::string_view prefix={}) {
  const auto p=std::string(prefix);
  line.number(p+"allocation_id",value.allocation_id); line.number(p+"generation",value.generation);
  line.number(p+"offset_bytes",value.offset_bytes); line.number(p+"allocation_bytes",value.allocation_bytes);
  line.boolean(p+"known",value.known); line.boolean(p+"mapped",value.mapped);
  line.boolean(p+"readable",value.readable); line.boolean(p+"writable",value.writable);
}
void add_identity(audit::JsonLine& line, Identity value) {
  line.number("object_id",value.id); line.number("generation",value.generation); line.number("bytes",value.bytes);
}
void apply(const Parameters& p, audit::JsonLine& line) {
  auto& registry=state->registry;
  if (p.flags) throw std::runtime_error("memory_reserved_flags");
  switch (p.operation) {
  case Operation::allocate: case Operation::reserve:
    if (!p.output) throw std::runtime_error("memory_output_missing");
    add_identity(line,registry.allocate(*static_cast<const CUdeviceptr*>(p.output),p.bytes,
                                       p.operation==Operation::reserve,p.context)); break;
  case Operation::free: case Operation::free_reservation:
    add_identity(line,registry.release(p.address,p.bytes,p.operation==Operation::free_reservation)); break;
  case Operation::create:
    if (!p.output) throw std::runtime_error("memory_handle_output_missing");
    add_identity(line,registry.create(*static_cast<const CUmemGenericAllocationHandle*>(p.output),p.bytes));
    line.integer("device",p.device); break;
  case Operation::release: add_identity(line,registry.release_handle(p.handle)); break;
  case Operation::map: case Operation::unmap: {
    const auto range=registry.resolve(p.address,p.bytes,0,p.context);
    const auto mapping=p.operation==Operation::map ? registry.map(p.address,p.bytes,p.handle,p.offset)
                                                 : registry.unmap(p.address,p.bytes);
    line.number("allocation_id",range.allocation_id); line.number("generation",range.generation);
    line.number("offset_bytes",mapping.offset_bytes); line.number("bytes",mapping.bytes);
    line.number("mapping_id",mapping.mapping_id); line.number("physical_id",mapping.physical_id);
    line.number("physical_generation",mapping.physical_generation);
    line.number("physical_offset_bytes",mapping.physical_offset_bytes); break;
  }
  case Operation::access: {
    const auto range=registry.resolve(p.address,p.bytes,0,p.context);
    for (std::size_t i=0;i<p.access_count;++i)
      registry.set_access(p.address,p.bytes,p.access[i].location.id,static_cast<unsigned>(p.access[i].flags));
    line.number("allocation_id",range.allocation_id); line.number("generation",range.generation);
    line.number("offset_bytes",range.offset_bytes); line.number("bytes",p.bytes);
    line.number("descriptor_count",p.access_count);
    // Each bounded descriptor is a separate normalized record, emitted by api().
    break;
  }
  case Operation::h2d: case Operation::d2h: case Operation::d2d: case Operation::memset: {
    line.number("bytes",p.bytes);
    if (!p.bytes) { line.boolean("zero_bytes",true); break; }
    const auto range=registry.resolve(p.address,p.bytes,0,p.context);
    if (!range.known || !range.mapped) throw std::runtime_error("memory_transfer_unobserved_range");
    add_resolution(line,range); line.boolean("zero_bytes",false);
    if (p.operation==Operation::d2d) {
      const auto source=registry.resolve(p.source,p.bytes,0,p.context);
      if (!source.known || !source.mapped) throw std::runtime_error("memory_transfer_unobserved_source");
      add_resolution(line,source,"source_");
    }
    break;
  }
  }
}
void summary() {
  auto line=state->line("summary");
  const auto counts=state->registry.counts();
  line.number("errors",state->errors); line.number("api_pairs",state->api_count);
  line.number("failed_apis",state->failures); line.number("open_calls",state->pending.size());
  line.number("revision",state->registry.revision());
  line.number("live_allocations",counts.allocations); line.number("live_reservations",counts.reservations);
  line.number("live_handles",counts.handles); line.number("live_mappings",counts.mappings);
  line.number("allocation_bytes",counts.allocation_bytes); line.number("reservation_bytes",counts.reservation_bytes);
  line.number("physical_bytes",counts.physical_bytes); line.number("mapped_bytes",counts.mapped_bytes);
  line.boolean("tensor_bounds_proven",false); line.boolean("terminal_complete",false);
  state->emit(line); state->closed=true;
}
void footer() noexcept {
  if (!state || !state->mutex.try_lock()) return;
  try { if (!state->closed) summary(); } catch (...) { ++state->errors; }
  state->mutex.unlock();
}
}
void initialize() {
  wchar_t path[32768]{};
  const auto n=GetEnvironmentVariableW(L"XVRAM_MEMORY_WITNESS_TRACE",path,32768);
  if (!n) return;
  if (n>=32768 || state) throw std::runtime_error("memory_path_or_reinitialize");
  state=new State;
  state->file=CreateFileW(path,GENERIC_WRITE,FILE_SHARE_READ,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
  if (state->file==INVALID_HANDLE_VALUE) throw std::runtime_error("memory_trace_create");
  auto line=state->line("session"); line.boolean("tensor_bounds_proven",false);
  line.string("scope","device_allocations_and_vmm_only"); line.boolean("host_pinned_observed",false);
  line.boolean("terminal_complete",false); state->emit(line);
  if (std::atexit(footer)!=0) throw std::runtime_error("memory_atexit");
}
bool enabled() noexcept { return state!=nullptr; }
bool healthy() noexcept { return state && !state->errors; }
void api(const CUpti_CallbackData& data, Word api_id) noexcept {
  if (!state || !data.functionName || !memory_mutation(data.functionName)) return;
  try {
    std::lock_guard lock(state->mutex);
    if (state->closed || state->errors) { ++state->errors; return; }
    if (!api_id || api_id>api_cap) throw std::runtime_error("memory_api_capacity");
    if (data.callbackSite==CUPTI_API_ENTER) {
      if (api_id<=state->last_api) throw std::runtime_error("memory_api_reuse");
      // This observer's first profile is serial. Overlapping memory API calls
      // would make an EXIT-time allocation-generation join ambiguous.
      if (!state->pending.empty()) throw std::runtime_error("memory_overlapping_api");
      if (state->pending.size()>=4096 || !state->pending.emplace(api_id,parameters(data)).second)
        throw std::runtime_error("memory_pending_capacity_or_duplicate");
      state->last_api=api_id;
      return;
    }
    if (data.callbackSite!=CUPTI_API_EXIT || !data.functionReturnValue)
      throw std::runtime_error("memory_callback_site_or_result");
    auto found=state->pending.find(api_id);
    if (found==state->pending.end() || found->second.symbol!=data.functionName
        || found->second.context!=reinterpret_cast<std::uintptr_t>(data.context))
      throw std::runtime_error("memory_api_pair");
    const auto params=found->second;
    state->pending.erase(found);
    const int result=*static_cast<const int*>(data.functionReturnValue);
    auto line=state->line("operation"); line.number("api_id",api_id);
    line.string("symbol",params.symbol); line.integer("result",result);
    if (result==0) apply(params,line); else ++state->failures;
    line.number("revision",state->registry.revision()); state->emit(line); ++state->api_count;
    if (result==0 && params.operation==Operation::access) {
      const auto range=state->registry.resolve(params.address,params.bytes,0,params.context);
      for (std::size_t i=0;i<params.access_count;++i) {
        auto desc=state->line("access_descriptor"); desc.number("api_id",api_id);
        desc.number("allocation_id",range.allocation_id); desc.number("generation",range.generation);
        desc.number("offset_bytes",range.offset_bytes); desc.number("bytes",params.bytes);
        desc.integer("device",params.access[i].location.id);
        desc.number("flags",static_cast<unsigned>(params.access[i].flags));
        desc.number("revision",state->registry.revision()); state->emit(desc);
      }
    }
  } catch (...) {
    ++state->errors;
    try {
      std::lock_guard lock(state->mutex);
      if (!state->closed) {
        auto line=state->line("error"); line.number("api_id",api_id);
        line.string("reason","observer_state_or_unsupported_api"); state->emit(line);
      }
    } catch (...) { ++state->errors; }
  }
}
Resolution resolve(Word address, Word bytes, int device, CUcontext context, Word alignment) noexcept {
  if (!state) return {};
  try {
    std::lock_guard lock(state->mutex);
    if (state->errors || state->closed || !state->pending.empty()) return {};
    auto result=state->registry.resolve(address,bytes,device,reinterpret_cast<std::uintptr_t>(context),alignment);
    result.observed_sequence=state->sequence;
    return result;
  } catch (...) { ++state->errors; return {}; }
}
void finish() {
  if (!state) return;
  std::lock_guard lock(state->mutex);
  if (state->closed) throw std::runtime_error("memory_double_finish");
  summary();
  if (!FlushFileBuffers(state->file)) throw std::runtime_error("memory_flush_failed");
  if (state->errors || !state->pending.empty()) throw std::runtime_error("memory_incomplete");
}
}
