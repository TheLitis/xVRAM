// Opt-in, bounded diagnostic. No production residency or argument changes.
// Diagnostic only: serialized PC samples are identity evidence, NOT evidence
// of the application's unperturbed scheduling. No CUDA calls in CRC callbacks.
#include "sampling.hpp"
#include "postmortem.hpp"
#include "compat_audit_collector/module_evidence.hpp"
#include "platform/sha256.hpp"
#include "platform/dynamic_library.hpp"
#include <cupti_pcsampling.h>
#include <cupti_activity.h>
#include <cupti_profiler_target.h>
#include <windows.h>
#include <array>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <set>
#include <tuple>

namespace xvram::launch_probe::sampling {
namespace {
constexpr std::size_t pcs = 4096, max_stalls = 128;
constexpr std::size_t hardware_bytes=512ULL*1024*1024;
constexpr std::uint32_t sampling_period=11;
struct State {
  HANDLE file = INVALID_HANDLE_VALUE;
  std::mutex mutex;
  audit::ModuleCopies copies;
  std::uint64_t sequence = 0, bytes = 0;
  std::map<std::uint64_t, std::string> hashes;
  std::set<char*> allocated_names;
  platform::DynamicLibrary host, target, driver;
  std::atomic_uint64_t errors{0};
  CUcontext context = nullptr;
  CUpti_PCSamplingData data{};
  CUpti_PCSamplingData configured_data{};
  std::vector<CUpti_PCSamplingPCData> rows;
  std::vector<CUpti_PCSamplingPCData> configured_rows;
  std::vector<CUpti_PCSamplingStallReason> stalls;
  std::vector<CUpti_PCSamplingStallReason> configured_stalls;
  bool closed = false;
  audit::JsonLine line(const char* kind) {
    audit::JsonLine value;
    value.number("schema_version", 1); value.string("record_type", "xvram.cuda_pc_witness");
    value.number("sequence", ++sequence); value.string("kind", kind); return value;
  }
  void emit(const audit::JsonLine& line) {
    const auto text = line.finish();
    if (closed || text.size() > 64ULL*1024*1024-bytes) throw std::runtime_error("sampling_capacity");
    DWORD written=0;
    if (!WriteFile(file,text.data(),static_cast<DWORD>(text.size()),&written,nullptr) || written!=text.size())
      throw std::runtime_error("sampling_write");
    bytes += text.size();
  }
};
State* state=nullptr;
void check(CUptiResult result) {
  if(result != CUPTI_SUCCESS) {
    if(state) {
      std::lock_guard lock(state->mutex);
      auto line=state->line("error"); line.number("cupti_result",result); state->emit(line);
      ++state->errors;
    }
    throw std::runtime_error("sampling_cupti_failure");
  }
}
void CUPTIAPI cubin(const void* bytes, std::size_t length, std::uint64_t* cookie) noexcept {
  // CUPTI's documented custom-CRC callback supplies the exact image bytes.
  // Deterministic FNV-1a is only a transport key. Every observed key is also
  // bound to SHA-256 outside this callback; any key/hash collision fails closed.
  if(!state || !cookie) return;
  *cookie=0;
  try {
    std::lock_guard lock(state->mutex);
    if (!bytes || !length || length > 64ULL*1024*1024) throw std::runtime_error("sampling_image_limit");
    std::uint64_t id=14695981039346656037ULL;
    for(std::size_t i=0;i<length;++i) { id ^= static_cast<const unsigned char*>(bytes)[i]; id *= 1099511628211ULL; }
    if(!id || !state->copies.push(bytes,length,{0,id,1,state->sequence,{}}))
      throw std::runtime_error("sampling_cubin_capacity");
    *cookie=id;
  } catch(...) {
    ++state->errors;
    try {
      std::lock_guard lock(state->mutex);
      auto line=state->line("copy_rejected"); line.number("bytes",length);
      line.number("active_bytes",state->copies.active_bytes); state->emit(line);
    } catch(...) {}
  }
}
void drain_copies() {
  for (;;) {
    audit::ModuleCopy copy;
    { std::lock_guard lock(state->mutex); if(!state->copies.pop(copy)) return; }
    // Owned bytes outlive the callback; CPU hashing is outside it.
    const auto sha=platform::sha256_bytes(copy.bytes);
    std::lock_guard lock(state->mutex);
    const auto [entry, inserted]=state->hashes.emplace(copy.module,sha);
    if(!inserted && entry->second!=sha) throw std::runtime_error("sampling_crc_collision");
    auto line=state->line("cubin"); line.number("cubin_cookie",copy.module);
    line.number("bytes",copy.bytes.size()); line.string("sha256",sha); state->emit(line);
    state->copies.retire(copy.bytes.size());
  }
}
void footer() noexcept {
  if(!state || !state->mutex.try_lock()) return;
  try {
    if(state->closed) { state->mutex.unlock(); return; }
    auto line=state->line("summary"); line.number("errors",state->errors);
    line.number("copies_submitted",state->copies.submitted); line.number("copies_retired",state->copies.retired);
    line.boolean("terminal_complete",false); line.boolean("sampler_cleanup_proven",false);
    state->emit(line); state->closed=true;
  } catch(...) { ++state->errors; }
  state->mutex.unlock();
}
}
bool enabled() noexcept { return state!=nullptr; }
void initialize() {
  wchar_t path[32768]{};
  const auto length=GetEnvironmentVariableW(L"XVRAM_PC_WITNESS_TRACE",path,32768);
  if(!length) return;
  if(length>=32768 || state) throw std::runtime_error("sampling_path_or_reinitialize");
  state=new State;
  if(!state->driver.open_system({"nvcuda.dll"})) throw std::runtime_error("sampling_driver_missing");
  state->file=CreateFileW(path,GENERIC_WRITE,FILE_SHARE_READ,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
  if(state->file==INVALID_HANDLE_VALUE) throw std::runtime_error("sampling_create");
  auto line=state->line("session"); line.boolean("serialized_diagnostic",true);
  line.number("sampling_period",sampling_period); line.number("hardware_buffer_bytes",hardware_bytes);
  line.number("configured_buffer_pcs",pcs); line.number("result_buffer_pcs",pcs);
  line.boolean("terminal_complete",false); state->emit(line);
  wchar_t loaded_path[32768]{};
  const auto loaded=GetModuleHandleW(L"cupti64_2026.2.1.dll");
  const auto loaded_length=loaded ? GetModuleFileNameW(loaded,loaded_path,32768) : 0;
  if(!loaded_length || loaded_length>=32768) throw std::runtime_error("sampling_cupti_missing");
  const auto directory=std::filesystem::path(loaded_path).parent_path();
  const auto open_pinned=[&](platform::DynamicLibrary& library, const wchar_t* filename, const char* hash) {
    const auto file=directory/filename;
    if(platform::sha256_file(file).digest!=hash || !library.open_absolute(file))
      throw std::runtime_error("sampling_profiler_pin");
    wchar_t actual[32768]{};
    const auto module=GetModuleHandleW(filename);
    const auto n=module ? GetModuleFileNameW(module,actual,32768) : 0;
    if(!n || n>=32768 || platform::sha256_file(actual).digest!=hash)
      throw std::runtime_error("sampling_loaded_profiler_pin");
  };
  open_pinned(state->host,L"nvperf_host.dll","cde4143748734948838ede61b350db59a0d7e6ac6ef451e375a5702d85174a9b");
  open_pinned(state->target,L"nvperf_target.dll","049f858e512592e895b5789d0872f44ee7cbb57d3e4b8936303f4d7eb5b2b1a7");
  CUpti_Profiler_Initialize_Params profiler{};
  profiler.structSize=CUpti_Profiler_Initialize_Params_STRUCT_SIZE;
  check(cuptiProfilerInitialize(&profiler));
  check(cuptiRegisterComputeCrcCallback(cubin));
  if(std::atexit(footer)!=0) throw std::runtime_error("sampling_atexit");
}
void before(CUcontext context) {
  if(!state) return;
  if(!state->context) {
    std::lock_guard lock(state->mutex);
    auto line=state->line("preflight"); line.boolean("context_present",context!=nullptr);
    line.number("errors",state->errors); line.number("copies_submitted",state->copies.submitted);
    state->emit(line);
  }
  if(!context || state->errors || (state->context && state->context!=context))
    throw std::runtime_error("sampling_context_or_error");
  if(state->context) return;
  state->context=context;
  const auto memory=state->driver.symbol<decltype(&cuMemGetInfo)>("cuMemGetInfo_v2");
  std::size_t free_bytes=0, total_bytes=0;
  if(!memory || memory(&free_bytes,&total_bytes)!=CUDA_SUCCESS || free_bytes<hardware_bytes+512ULL*1024*1024)
    throw std::runtime_error("sampling_device_headroom");
  CUpti_PCSamplingEnableParams enable{};
  enable.size=CUpti_PCSamplingEnableParamsSize; enable.ctx=context;
  check(cuptiPCSamplingEnable(&enable));
  std::size_t reasons=0;
  CUpti_PCSamplingGetNumStallReasonsParams count{};
  count.size=CUpti_PCSamplingGetNumStallReasonsParamsSize; count.ctx=context; count.numStallReasons=&reasons;
  check(cuptiPCSamplingGetNumStallReasons(&count));
  {
    std::lock_guard lock(state->mutex);
    auto line=state->line("enabled"); line.number("stall_reasons",reasons); state->emit(line);
  }
  if(!reasons || reasons>max_stalls) throw std::runtime_error("sampling_stall_limit");
  state->rows.resize(pcs); state->stalls.resize(pcs*reasons);
  state->configured_rows.resize(pcs); state->configured_stalls.resize(pcs*reasons);
  for(std::size_t i=0;i<pcs;++i) {
    state->rows[i].size=sizeof(CUpti_PCSamplingPCData);
    state->rows[i].stallReason=state->stalls.data()+i*reasons;
    state->configured_rows[i].size=sizeof(CUpti_PCSamplingPCData);
    state->configured_rows[i].stallReason=state->configured_stalls.data()+i*reasons;
  }
  state->data.size=sizeof(CUpti_PCSamplingData); state->data.collectNumPcs=pcs;
  state->data.pPcData=state->rows.data();
  state->configured_data.size=sizeof(CUpti_PCSamplingData); state->configured_data.collectNumPcs=pcs;
  state->configured_data.pPcData=state->configured_rows.data();
  std::array<CUpti_PCSamplingConfigurationInfo,4> attributes{};
  attributes[0].attributeType=CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_DATA_BUFFER;
  attributes[0].attributeData.samplingDataBufferData.samplingDataBuffer=&state->configured_data;
  attributes[1].attributeType=CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_COLLECTION_MODE;
  attributes[1].attributeData.collectionModeData.collectionMode=CUPTI_PC_SAMPLING_COLLECTION_MODE_KERNEL_SERIALIZED;
  attributes[2].attributeType=CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_HARDWARE_BUFFER_SIZE;
  attributes[2].attributeData.hardwareBufferSizeData.hardwareBufferSize=hardware_bytes;
  attributes[3].attributeType=CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_PERIOD;
  attributes[3].attributeData.samplingPeriodData.samplingPeriod=sampling_period;
  CUpti_PCSamplingConfigurationInfoParams config{};
  config.size=CUpti_PCSamplingConfigurationInfoParamsSize; config.ctx=context;
  config.numAttributes=attributes.size(); config.pPCSamplingConfigurationInfo=attributes.data();
  check(cuptiPCSamplingSetConfigurationAttribute(&config));
  for(const auto& attribute:attributes) check(attribute.attributeStatus);
  drain_copies();
}
void after(std::uint64_t call, CUcontext context, CUresult synchronized) {
  if(!state) return;
  if(context!=state->context || synchronized!=CUDA_SUCCESS || state->errors)
    throw std::runtime_error("sampling_completion");
  drain_copies();
  std::set<std::tuple<std::uint32_t,std::uint64_t,std::uint32_t,std::string>> unique;
  std::uint64_t total=0, dropped=0;
  for(std::size_t batch=0;;++batch) {
    if(batch>=64) throw std::runtime_error("sampling_batch_limit");
    CUpti_PCSamplingGetDataParams get{};
    get.size=CUpti_PCSamplingGetDataParamsSize; get.ctx=context; get.pcSamplingData=&state->data;
    check(cuptiPCSamplingGetData(&get));
    if(state->data.totalNumPcs>pcs || state->data.hardwareBufferFull) throw std::runtime_error("sampling_data_limit");
    total+=state->data.totalNumPcs; dropped+=state->data.droppedSamples;
    for(std::size_t i=0;i<state->data.totalNumPcs;++i) {
      const auto& row=state->rows[i];
      if(!row.functionName || !row.correlationId || !row.cubinCrc) throw std::runtime_error("sampling_identity_missing");
      state->allocated_names.insert(row.functionName);
      if(state->allocated_names.size()>65536) throw std::runtime_error("sampling_name_pool_limit");
      std::size_t size=0; while(size<=4096 && row.functionName[size]) ++size;
      if(size>4096) throw std::runtime_error("sampling_name_limit");
      unique.emplace(row.correlationId,row.cubinCrc,row.functionIndex,std::string(row.functionName,size));
    }
    if(!state->data.remainingNumPcs) break;
  }
  std::lock_guard lock(state->mutex);
  for(const auto& [correlation,cookie,index,name]:unique) {
    auto line=state->line("sample_identity"); line.number("observed_after_call",call);
    line.number("correlation_id",correlation); line.number("cubin_cookie",cookie);
    line.number("function_index",index); line.string("name",name); state->emit(line);
  }
  auto line=state->line("collection"); line.number("observed_after_call",call);
  line.number("pc_records",total); line.number("dropped_samples",dropped); state->emit(line);
}
void finish(CUresult synchronized) {
  if(!state) return;
  if(!state->context || synchronized!=CUDA_SUCCESS || state->errors || state->closed)
    throw std::runtime_error("sampling_terminal_state");
  // Called before RtlExitUserProcess, never from CUPTI/CRC callbacks, atexit or
  // DllMain. The controller deadline still owns any blocked driver operation.
  if(state->configured_data.totalNumPcs || state->configured_data.remainingNumPcs)
    throw std::runtime_error("sampling_terminal_uncollected");
  CUpti_PCSamplingDisableParams disable{};
  disable.size=CUpti_PCSamplingDisableParamsSize; disable.ctx=state->context;
  check(cuptiPCSamplingDisable(&disable));
  if(state->configured_data.totalNumPcs || state->configured_data.remainingNumPcs)
    throw std::runtime_error("sampling_late_records");
  drain_copies();
  // CUPTI can share these names with Activity kernel records, including ones
  // delivered later. Retain the bounded name pool until the owned worker dies;
  // only the controller's post-reap ledger may confirm its reclamation.
  postmortem::sampler_disabled(state->allocated_names.size());
  std::lock_guard lock(state->mutex);
  auto line=state->line("summary"); line.number("errors",state->errors);
  line.number("copies_submitted",state->copies.submitted); line.number("copies_retired",state->copies.retired);
  line.boolean("terminal_complete",false); line.boolean("sampler_cleanup_proven",false);
  state->emit(line); state->closed=true;
  if(!FlushFileBuffers(state->file)) throw std::runtime_error("sampling_flush_failed");
}
}
