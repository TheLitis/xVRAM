// Diagnostic-only CUPTI subscriber. No CUDA runtime/driver calls, code hooks,
// application argument changes, tensor-bound inference, or GPU submissions.
// Lifecycle follows NVIDIA CUDA 13.3 cupti_trace_injection, except its Windows
// process-exit Detours hook is deliberately omitted (see README.md).
#include "trace_core.hpp"
#include "module_evidence.hpp"
#include "platform/sha256.hpp"

#include <cupti.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <string_view>
#include <thread>
#include <type_traits>
#ifdef _WIN32
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace xvram::audit {
namespace {
constexpr std::size_t activity_buffer_bytes = 1024U * 1024U;
constexpr std::size_t maximum_buffers = 16;
constexpr std::uint64_t maximum_records = 1000000;
constexpr std::array activities{CUPTI_ACTIVITY_KIND_DRIVER, CUPTI_ACTIVITY_KIND_RUNTIME,
    CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL, CUPTI_ACTIVITY_KIND_MEMCPY,
    CUPTI_ACTIVITY_KIND_MEMSET, CUPTI_ACTIVITY_KIND_MEMORY2, CUPTI_ACTIVITY_KIND_FUNCTION};

template <typename T> std::uint64_t native_key(T value) {
  if constexpr (std::is_pointer_v<T>) return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(value));
  else return static_cast<std::uint64_t>(value);
}
std::uint64_t now() {
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
}
std::string_view bounded_name(const char * text) {
  if (text == nullptr) return {};
  std::size_t count = 0;
  while (count <= 4096 && text[count] != '\0') ++count;
  if (count > 4096) throw std::length_error("name limit");
  return {text, count};
}
bool named(std::string_view actual, std::string_view expected) {
  if (actual.ends_with("_ptsz") || actual.ends_with("_ptds")) actual.remove_suffix(5);
  return actual == expected;
}

class Sink {
public:
  bool open(const std::filesystem::path & path) {
#ifdef _WIN32
    file_ = CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ,
                        nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    return file_ != INVALID_HANDLE_VALUE;
#else
    file_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    return file_ >= 0;
#endif
  }
  bool write(std::string_view text) noexcept {
    while (!text.empty()) {
#ifdef _WIN32
      DWORD written = 0;
      if (file_ == INVALID_HANDLE_VALUE || !WriteFile(file_, text.data(), static_cast<DWORD>(text.size()), &written, nullptr) || written == 0) return false;
      text.remove_prefix(written);
#else
      const auto written = ::write(file_, text.data(), text.size());
      if (written < 0 && errno == EINTR) continue;
      if (written <= 0) return false;
      text.remove_prefix(static_cast<std::size_t>(written));
#endif
    }
    return true;
  }
private:
#ifdef _WIN32
  HANDLE file_ = INVALID_HANDLE_VALUE;
#else
  int file_ = -1;
#endif
};

struct State {
  std::mutex mutex;
  std::mutex wait_mutex;
  std::condition_variable wake;
  Sink sink;
  Registry registry;
  ModuleCopies module_copies;
  ModuleLifetimes cupti_modules, native_modules;
  // Native function ownership is scoped by context and exact module generation.
  struct FunctionOwner { std::uint64_t module=0, generation=0, function_generation=0; };
  std::map<std::pair<std::uint64_t,std::uint64_t>,FunctionOwner> function_owners;
  std::uint64_t next_function_generation=0, module_hash_records=0;
  std::map<std::thread::id, std::uint64_t> threads;
  CUpti_SubscriberHandle subscriber = nullptr;
  std::thread flusher;
  std::atomic<bool> stop{false}, finalizing{false}, finalized{false};
  std::atomic<std::uint64_t> dropped{0}, errors{0}, serialization_errors{0};
  std::atomic<std::uint64_t> buffers_requested{0}, buffers_completed{0}, buffers_outstanding{0};
  std::uint64_t sequence = 0, callback_records = 0, activity_records = 0, resource_records = 0;
  std::uint64_t unknown_callback_details = 0, unknown_activity_kinds = 0, incomplete_activities = 0;
  std::int64_t api_inflight = 0;
  bool initialized = false;
  unsigned trace_schema_version = 1;

  JsonLine begin(std::string_view kind) {
    JsonLine line;
    line.number("schema_version", trace_schema_version);
    line.string("report_type", "xvram.cuda_compat_audit_trace");
    line.number("sequence", sequence + 1);
    line.string("kind", kind);
    line.number("timestamp_ns", now());
    return line;
  }
  bool emit(const JsonLine & line, bool terminal = false) {
    if(finalized && !terminal) { ++dropped; return false; }
    if (sequence >= maximum_records && !terminal) { ++dropped; return false; }
    const auto data = line.finish();
    if (!sink.write(data)) { ++serialization_errors; return false; }
    ++sequence;
    return true;
  }
  std::uint64_t id(std::string_view kind, std::uint64_t value) { return registry.identity(kind, value); }
  std::uint64_t thread_id() {
    const auto native = std::this_thread::get_id();
    if (auto found = threads.find(native); found != threads.end()) return found->second;
    if (threads.size() >= max_identities) throw std::length_error("thread limit");
    return threads.emplace(native, threads.size() + 1).first->second;
  }
  void gap(std::string_view reason, std::uint64_t count = 0) {
    auto line = begin("gap"); line.string("reason", reason); line.number("lost_records", count);
    static_cast<void>(emit(line));
  }
  void summary(bool safe) {
    auto line = begin("summary");
    line.number("callback_records", callback_records);
    line.number("activity_records", activity_records);
    line.number("resource_records", resource_records);
    line.number("dropped_records", dropped.load());
    line.number("unknown_callback_details", unknown_callback_details);
    line.number("unknown_activity_kinds", unknown_activity_kinds);
    line.number("serialization_errors", serialization_errors.load());
    line.number("collector_errors", errors.load());
    line.number("buffers_requested", buffers_requested.load());
    line.number("buffers_completed", buffers_completed.load());
    line.number("incomplete_activities", incomplete_activities);
    if (trace_schema_version == 3) {
      line.number("module_copies_submitted",module_copies.submitted);
      line.number("module_copies_retired",module_copies.retired);
      line.number("module_copy_active_bytes",module_copies.active_bytes);
      line.number("module_copy_total_bytes",module_copies.total_bytes);
      line.number("module_copy_peak_bytes",module_copies.peak_bytes);
      line.number("module_hash_records",module_hash_records);
    }
    line.boolean("safely_finalized", safe);
    line.string("terminal_checkpoint", safe ? "explicit_finalize" : "process_exit_unflushed");
    const bool complete = safe && api_inflight == 0 && dropped == 0 && errors == 0 &&
        serialization_errors == 0 && unknown_callback_details == 0 && unknown_activity_kinds == 0 &&
        incomplete_activities == 0 && buffers_requested == buffers_completed &&
        (trace_schema_version != 3 || (module_copies.submitted == module_copies.retired &&
          module_copies.retired == module_hash_records && module_copies.active_bytes == 0));
    line.boolean("complete", complete);
    static_cast<void>(emit(line, true));
  }
};
// Intentionally process-lifetime: no CUPTI call or thread join from DLL teardown.
State * state = nullptr;
std::mutex initialization_mutex;
thread_local bool inside_callback = false;

void account(CUptiResult result) noexcept {
  if (result != CUPTI_SUCCESS && state != nullptr) ++state->errors;
}
std::uint64_t stream_id(State & s, CUcontext context, CUstream stream, bool per_thread = false) {
  std::uint32_t value = 0, context_value = 0;
  if (cuptiGetStreamIdEx(context, stream, per_thread ? 1 : 0, &value) != CUPTI_SUCCESS) {
    ++s.errors; return 0;
  }
  if(context==nullptr || cuptiGetContextId(context,&context_value)!=CUPTI_SUCCESS) {
    ++s.errors; return 0;
  }
  return s.id("cupti_stream", (static_cast<std::uint64_t>(context_value)<<32) | value);
}
std::uint64_t context_id(State & s, CUcontext context) {
  if (context == nullptr) return 0;
  std::uint32_t value = 0;
  if (cuptiGetContextId(context, &value) != CUPTI_SUCCESS) { ++s.errors; return 0; }
  return s.id("cupti_context", value);
}
void geometry(JsonLine & line, std::uint64_t gx, std::uint64_t gy, std::uint64_t gz,
              std::uint64_t bx, std::uint64_t by, std::uint64_t bz, std::uint64_t shared) {
  line.number("grid_x", gx); line.number("grid_y", gy); line.number("grid_z", gz);
  line.number("block_x", bx); line.number("block_y", by); line.number("block_z", bz);
  line.number("shared_bytes", shared);
}

// Only fixed, official CUDA API metadata is read. Kernel argument arrays,
// application buffers and opaque private ABI data are never read. V3 separately
// copies only CUPTI-owned module resource bytes while their callback is valid.
bool api_detail(State & s, JsonLine & line, const CUpti_CallbackData & data,
                bool runtime, bool exited, bool success) {
  const auto name = bounded_name(data.functionName);
  const bool mutate = exited && success;
  const auto address = [&](std::uint64_t value, std::uint64_t bytes, std::string_view prefix = {}) {
    add_range(line, s.registry.lookup(value, bytes), prefix);
  };
  const auto alloc = [&](std::uint64_t value, std::uint64_t bytes, std::string_view kind) {
    line.string("memory_kind", kind); line.number("size_bytes", bytes);
    if (mutate) add_range(line, s.registry.allocate(value, bytes));
  };
  const auto free = [&](std::uint64_t value) {
    add_range(line, mutate ? s.registry.release(value) : s.registry.lookup(value, 0));
  };
  const auto copy = [&](std::uint64_t dst, std::uint64_t src, std::uint64_t bytes) {
    line.number("size_bytes", bytes); address(src, bytes, "src_"); address(dst, bytes, "dst_");
  };
#define META(TYPE) const auto & p = *static_cast<const TYPE *>(data.functionParams); line.number("parameter_bytes", sizeof(TYPE))
#define OP(TEXT) line.string("op", TEXT)
  if (data.functionParams == nullptr) { OP("other"); return false; }
  // Version 1 retains its original metadata and unknown-detail behavior. Version
  // 2 observes only official resolver inputs/outputs; the function pointer is
  // never called, replaced, dereferenced as code, or serialized as an address.
  if (s.trace_schema_version >= 2) {
    const auto resolver = [&](const char * symbol, std::uint64_t flags, void ** output,
                              bool has_version, std::int64_t version,
                              const auto * query_status) {
      OP("other");
      const auto requested = bounded_name(symbol);
      if (!resolver_symbol(requested)) return false;
      line.string("requested_symbol", requested);
      line.number("resolver_flags", flags);
      if (has_version) line.integer("requested_version", version);
      if (mutate && query_status) line.integer("query_status", static_cast<std::int64_t>(*query_status));
      if (mutate && output && *output && (!query_status || static_cast<int>(*query_status) == 0))
        line.number("entry_point_id", s.id("resolver_entry_point", native_key(*output)));
      return true;
    };
    if (!runtime && name == "cuGetProcAddress") {
      META(cuGetProcAddress_params);
      return resolver(p.symbol, p.flags, p.pfn, true, p.cudaVersion,
                      static_cast<const CUdriverProcAddressQueryResult *>(nullptr));
    }
    if (!runtime && name == "cuGetProcAddress_v2") {
      META(cuGetProcAddress_v2_params);
      return resolver(p.symbol, p.flags, p.pfn, true, p.cudaVersion, p.symbolStatus);
    }
    if (runtime && named(name, "cudaGetDriverEntryPoint")) {
      META(cudaGetDriverEntryPoint_v11030_params);
      return resolver(p.symbol, p.flags, p.funcPtr, false, 0, p.driverStatus);
    }
    if (runtime && named(name, "cudaGetDriverEntryPointByVersion")) {
      META(cudaGetDriverEntryPointByVersion_v12050_params);
      return resolver(p.symbol, p.flags, p.funcPtr, true, p.cudaVersion, p.driverStatus);
    }
  }
  if (runtime) {
    if (named(name, "cudaMalloc")) {
      META(cudaMalloc_v3020_params); OP("allocate"); alloc(mutate && p.devPtr ? native_key(*p.devPtr) : 0, p.size, "device"); return true;
    }
    if (named(name, "cudaMallocHost")) {
      META(cudaMallocHost_v3020_params); OP("host_allocate"); alloc(mutate && p.ptr ? native_key(*p.ptr) : 0, p.size, "pinned_host"); return true;
    }
    if (named(name, "cudaHostAlloc")) {
      META(cudaHostAlloc_v3020_params); OP("host_allocate"); alloc(mutate && p.pHost ? native_key(*p.pHost) : 0, p.size, "pinned_host"); return true;
    }
    if (named(name, "cudaMallocManaged")) {
      META(cudaMallocManaged_v6000_params); OP("allocate"); alloc(mutate && p.devPtr ? native_key(*p.devPtr) : 0, p.size, "managed"); return true;
    }
    if (named(name, "cudaHostRegister")) {
      META(cudaHostRegister_v4000_params); OP("host_register"); alloc(native_key(p.ptr), p.size, "registered_host"); return true;
    }
    if (named(name, "cudaHostUnregister")) {
      META(cudaHostUnregister_v4000_params); OP("host_unregister"); free(native_key(p.ptr)); return true;
    }
    if (named(name, "cudaFree")) { META(cudaFree_v3020_params); OP("free"); free(native_key(p.devPtr)); return true; }
    if (named(name, "cudaFreeHost")) { META(cudaFreeHost_v3020_params); OP("host_free"); free(native_key(p.ptr)); return true; }
    if (named(name, "cudaMemcpy")) { META(cudaMemcpy_v3020_params); OP("copy"); copy(native_key(p.dst), native_key(p.src), p.count); return true; }
    if (named(name, "cudaMemcpyAsync")) {
      META(cudaMemcpyAsync_v3020_params); OP("copy"); copy(native_key(p.dst), native_key(p.src), p.count);
      line.number("stream_id", stream_id(s, data.context, reinterpret_cast<CUstream>(p.stream), name.ends_with("_ptsz"))); return true;
    }
    if (named(name, "cudaMemcpy2D") || named(name, "cudaMemcpy2DAsync")) {
      OP("copy");
      std::uint64_t width = 0, height = 0, sp = 0, dp = 0, src = 0, dst = 0;
      if (named(name, "cudaMemcpy2D")) { META(cudaMemcpy2D_v3020_params); width=p.width; height=p.height; sp=p.spitch; dp=p.dpitch; src=native_key(p.src); dst=native_key(p.dst); }
      else { META(cudaMemcpy2DAsync_v3020_params); width=p.width; height=p.height; sp=p.spitch; dp=p.dpitch; src=native_key(p.src); dst=native_key(p.dst); line.number("stream_id", stream_id(s,data.context,reinterpret_cast<CUstream>(p.stream),name.ends_with("_ptsz"))); }
      const auto max = std::numeric_limits<std::uint64_t>::max();
      if ((height && width > max/height) || (height > 1 && (sp > (max-width)/(height-1) || dp > (max-width)/(height-1)))) return false;
      line.number("size_bytes", width*height); line.number("width_bytes",width); line.number("height",height);
      line.number("src_pitch_bytes",sp); line.number("dst_pitch_bytes",dp);
      address(src,height ? (height-1)*sp+width : 0,"src_"); address(dst,height ? (height-1)*dp+width : 0,"dst_"); return true;
    }
    if (named(name, "cudaMemset")) { META(cudaMemset_v3020_params); OP("memset"); line.number("size_bytes",p.count); address(native_key(p.devPtr),p.count); return true; }
    if (named(name, "cudaMemsetAsync")) { META(cudaMemsetAsync_v3020_params); OP("memset"); line.number("size_bytes",p.count); address(native_key(p.devPtr),p.count); line.number("stream_id",stream_id(s,data.context,reinterpret_cast<CUstream>(p.stream),name.ends_with("_ptsz"))); return true; }
    if (named(name, "cudaLaunchKernel")) {
      META(cudaLaunchKernel_v7000_params); OP("launch"); line.number("function_id",s.id("runtime_function",native_key(p.func)));
      geometry(line,p.gridDim.x,p.gridDim.y,p.gridDim.z,p.blockDim.x,p.blockDim.y,p.blockDim.z,p.sharedMem);
      line.number("stream_id",stream_id(s,data.context,reinterpret_cast<CUstream>(p.stream),name.ends_with("_ptsz"))); return true;
    }
    if (named(name, "cudaLaunchKernelExC")) {
      META(cudaLaunchKernelExC_v11060_params); OP("launch"); if (!p.config) return false;
      line.number("function_id",s.id("runtime_function",native_key(p.func)));
      const auto & c = *p.config;
      geometry(line,c.gridDim.x,c.gridDim.y,c.gridDim.z,c.blockDim.x,c.blockDim.y,c.blockDim.z,c.dynamicSmemBytes);
      line.number("stream_id",stream_id(s,data.context,reinterpret_cast<CUstream>(c.stream),name.ends_with("_ptsz"))); return true;
    }
    if (named(name,"cudaEventRecord")) { META(cudaEventRecord_v3020_params); OP("event"); line.number("event_id",s.id("event",native_key(p.event))); line.number("stream_id",stream_id(s,data.context,reinterpret_cast<CUstream>(p.stream),name.ends_with("_ptsz"))); return true; }
    if (named(name,"cudaStreamWaitEvent")) { META(cudaStreamWaitEvent_v3020_params); OP("event"); line.number("event_id",s.id("event",native_key(p.event))); line.number("stream_id",stream_id(s,data.context,reinterpret_cast<CUstream>(p.stream),name.ends_with("_ptsz"))); return true; }
    if (named(name,"cudaEventDestroy")) { META(cudaEventDestroy_v3020_params); OP("event"); line.number("event_id",s.id("event",native_key(p.event))); if(mutate) s.registry.retire_identity("event",native_key(p.event)); return true; }
    if (named(name,"cudaEventSynchronize")) { META(cudaEventSynchronize_v3020_params); OP("event"); line.number("event_id",s.id("event",native_key(p.event))); return true; }
    if (named(name,"cudaEventQuery")) { META(cudaEventQuery_v3020_params); OP("event"); line.number("event_id",s.id("event",native_key(p.event))); return true; }
    if (named(name,"cudaStreamSynchronize")) { META(cudaStreamSynchronize_v3020_params); OP("stream"); line.number("stream_id",stream_id(s,data.context,reinterpret_cast<CUstream>(p.stream),name.ends_with("_ptsz"))); return true; }
  } else {
    if (s.trace_schema_version == 3) {
      const auto context=context_id(s,data.context);
      const auto module_identity=[&](CUmodule module) {
        return s.id("native_module_"+std::to_string(context),native_key(module));
      };
      const auto function_identity=[&](CUfunction function) {
        return s.id("driver_function_"+std::to_string(context),native_key(function));
      };
      if (name=="cuModuleLoad" || name=="cuModuleLoadData" || name=="cuModuleLoadDataEx" || name=="cuModuleLoadFatBinary") {
        CUmodule * output=nullptr;
        if (name=="cuModuleLoad") { META(cuModuleLoad_params); output=p.module; }
        else if (name=="cuModuleLoadData") { META(cuModuleLoadData_params); output=p.module; }
        else if (name=="cuModuleLoadDataEx") { META(cuModuleLoadDataEx_params); output=p.module; }
        else { META(cuModuleLoadFatBinary_params); output=p.module; }
        OP("other");
        if (mutate && output && *output) {
          const auto module=module_identity(*output);
          const auto life=s.native_modules.load(context,module);
          line.number("module_id",module); line.number("module_generation",life.generation);
        }
        return true;
      }
      if (name=="cuModuleUnload") {
        META(cuModuleUnload_params); OP("other");
        const auto module=module_identity(p.hmod);
        const auto life=mutate ? s.native_modules.unload(context,module) : s.native_modules.lookup(context,module);
        line.number("module_id",module); line.number("module_generation",life.generation);
        return true;
      }
      if (name=="cuModuleGetFunction") {
        META(cuModuleGetFunction_params); OP("module_function");
        const auto module=module_identity(p.hmod);
        const auto life=s.native_modules.lookup(context,module);
        line.number("module_id",module); line.number("module_generation",life.live ? life.generation : 0);
        line.string("kernel_name",bounded_name(p.name));
        if (mutate && p.hfunc && *p.hfunc) {
          const auto function=function_identity(*p.hfunc);
          const auto key=std::pair{context,function};
          if (!s.function_owners.contains(key) && s.function_owners.size()>=max_identities) throw std::length_error("function limit");
          auto & owner=s.function_owners[key];
          if (!life.live || owner.module!=module || owner.generation!=life.generation || owner.function_generation==0)
            owner={module,life.live ? life.generation : 0,++s.next_function_generation};
          line.number("function_id",function); line.number("function_generation",owner.function_generation);
        }
        return true;
      }
      if (named(name,"cuLaunchKernel")) {
        META(cuLaunchKernel_params); OP("launch");
        const auto function=function_identity(p.f);
        line.number("function_id",function);
        const auto owner=s.function_owners.find({context,function});
        if (owner!=s.function_owners.end()) {
          const auto life=s.native_modules.lookup(context,owner->second.module);
          if (life.live && life.generation==owner->second.generation) {
            line.number("module_id",owner->second.module); line.number("module_generation",life.generation);
            line.number("function_generation",owner->second.function_generation);
          }
        }
        geometry(line,p.gridDimX,p.gridDimY,p.gridDimZ,p.blockDimX,p.blockDimY,p.blockDimZ,p.sharedMemBytes);
        line.number("stream_id",stream_id(s,data.context,p.hStream,name.ends_with("_ptsz"))); return true;
      }
    }
    if (name=="cuMemAlloc_v2") { META(cuMemAlloc_v2_params); OP("allocate"); alloc(mutate && p.dptr ? *p.dptr : 0,p.bytesize,"device"); return true; }
    if (name=="cuMemFree_v2") { META(cuMemFree_v2_params); OP("free"); free(p.dptr); return true; }
    if (name=="cuMemAddressReserve") { META(cuMemAddressReserve_params); OP("vmm_reserve"); alloc(mutate && p.ptr ? *p.ptr : 0,p.size,"virtual_reservation"); return true; }
    if (name=="cuMemAddressFree") { META(cuMemAddressFree_params); OP("vmm_free"); line.number("size_bytes",p.size); free(p.ptr); return true; }
    if (name=="cuMemCreate") { META(cuMemCreate_params); OP("vmm_create"); line.number("size_bytes",p.size); if(mutate && p.handle) line.number("physical_allocation_id",s.id("physical",*p.handle)); return true; }
    if (name=="cuMemRelease") { META(cuMemRelease_params); OP("vmm_release"); line.number("physical_allocation_id",s.id("physical",p.handle)); if(mutate) s.registry.retire_identity("physical",p.handle); return true; }
    if (name=="cuMemMap") { META(cuMemMap_params); OP("vmm_map"); line.number("size_bytes",p.size); line.number("handle_offset_bytes",p.offset); line.number("physical_allocation_id",s.id("physical",p.handle)); address(p.ptr,p.size); return true; }
    if (name=="cuMemUnmap") { META(cuMemUnmap_params); OP("vmm_unmap"); line.number("size_bytes",p.size); address(p.ptr,p.size); return true; }
    if (name=="cuMemSetAccess") { META(cuMemSetAccess_params); OP("vmm_access"); line.number("size_bytes",p.size); address(p.ptr,p.size); return true; }
    if (name=="cuMemcpyHtoD_v2") { META(cuMemcpyHtoD_v2_params); OP("copy"); copy(p.dstDevice,native_key(p.srcHost),p.ByteCount); return true; }
    if (name=="cuMemcpyDtoH_v2") { META(cuMemcpyDtoH_v2_params); OP("copy"); copy(native_key(p.dstHost),p.srcDevice,p.ByteCount); return true; }
    if (name=="cuMemcpyDtoD_v2") { META(cuMemcpyDtoD_v2_params); OP("copy"); copy(p.dstDevice,p.srcDevice,p.ByteCount); return true; }
    if (named(name,"cuLaunchKernel")) {
      META(cuLaunchKernel_params); OP("launch"); line.number("function_id",s.id("driver_function",native_key(p.f)));
      geometry(line,p.gridDimX,p.gridDimY,p.gridDimZ,p.blockDimX,p.blockDimY,p.blockDimZ,p.sharedMemBytes);
      line.number("stream_id",stream_id(s,data.context,p.hStream,name.ends_with("_ptsz"))); return true;
    }
    if (name=="cuModuleGetFunction") { META(cuModuleGetFunction_params); OP("module_function"); line.number("module_id",s.id("native_module",native_key(p.hmod))); line.string("kernel_name",bounded_name(p.name)); if(mutate && p.hfunc) line.number("function_id",s.id("driver_function",native_key(*p.hfunc))); return true; }
  }
  OP("other"); return false;
#undef META
#undef OP
}

void resource(State & s, CUpti_CallbackId cbid, const CUpti_ResourceData & data) {
  auto line=s.begin("resource"); line.number("callback_id",cbid);
  line.number("context_id",context_id(s,data.context));
  bool known=true;
  if(cbid==CUPTI_CBID_RESOURCE_CONTEXT_CREATED || cbid==CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING) {
    line.string("resource_kind","context"); line.string("operation",cbid==CUPTI_CBID_RESOURCE_CONTEXT_CREATED ? "create" : "destroy");
  } else if(cbid==CUPTI_CBID_RESOURCE_STREAM_CREATED || cbid==CUPTI_CBID_RESOURCE_STREAM_DESTROY_STARTING) {
    line.string("resource_kind","stream"); line.string("operation",cbid==CUPTI_CBID_RESOURCE_STREAM_CREATED ? "create" : "destroy");
    line.number("stream_id",stream_id(s,data.context,data.resourceHandle.stream));
    // CUPTI stream identifiers are unique for a context lifetime; native handles are not emitted.
  } else if(cbid==CUPTI_CBID_RESOURCE_MODULE_LOADED || cbid==CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING) {
    line.string("resource_kind","module"); line.string("operation",cbid==CUPTI_CBID_RESOURCE_MODULE_LOADED ? "load" : "unload");
    if(data.resourceDescriptor) {
      const auto & module=*static_cast<const CUpti_ModuleResourceData *>(data.resourceDescriptor);
      const auto context=context_id(s,data.context);
      const auto id=s.id(s.trace_schema_version==3 ? "cupti_module_"+std::to_string(context) : "cupti_module",module.moduleId);
      line.number("module_id",id); line.number("size_bytes",module.cubinSize);
      if (s.trace_schema_version==3) {
        const bool loading=cbid==CUPTI_CBID_RESOURCE_MODULE_LOADED;
        const auto life=loading ? s.cupti_modules.load(context,id) : s.cupti_modules.unload(context,id);
        line.number("module_generation",life.generation);
        if (loading) {
          // No parser, hash, file I/O, CUDA call or queue-space wait here.
          const bool copied=s.module_copies.push(module.pCubin,module.cubinSize,
                                                 ModuleCopy{context,id,life.generation,s.sequence+1,{}});
          line.boolean("module_copy_queued",copied);
          if (!copied) { ++s.dropped; known=false; }
        } else if (life.generation==0) known=false;
      }
    }
    else known=false;
  } else { line.string("resource_kind","other"); line.string("operation","other"); known=false; }
  line.boolean("detail_known",known);
  if(s.emit(line)) ++s.resource_records;
}

void CUPTIAPI callback(void *, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void * raw) noexcept {
  auto * s=state;
  if(s==nullptr || s->finalized || raw==nullptr) return;
  if(inside_callback) { ++s->dropped; return; }
  inside_callback=true;
  try {
    std::lock_guard lock(s->mutex);
    if(domain==CUPTI_CB_DOMAIN_RUNTIME_API || domain==CUPTI_CB_DOMAIN_DRIVER_API) {
      const auto & data=*static_cast<const CUpti_CallbackData *>(raw);
      const bool exited=data.callbackSite==CUPTI_API_EXIT;
      s->api_inflight += exited ? -1 : 1;
      const std::int64_t status=exited && data.functionReturnValue ? *static_cast<const int *>(data.functionReturnValue) : -1;
      auto line=s->begin(exited ? "api_exit" : "api_enter");
      line.string("domain",domain==CUPTI_CB_DOMAIN_RUNTIME_API ? "runtime" : "driver");
      line.number("callback_id",cbid); line.number("correlation_id",data.correlationId);
      line.number("thread_id",s->thread_id()); line.number("context_id",s->id("cupti_context",data.contextUid));
      line.string("symbol",bounded_name(data.functionName));
      if(exited) line.integer("status",status);
      if(data.symbolName && !named(bounded_name(data.functionName),"cuModuleGetFunction"))
        line.string("kernel_name",bounded_name(data.symbolName));
      const bool known=api_detail(*s,line,data,domain==CUPTI_CB_DOMAIN_RUNTIME_API,exited,status==0);
      line.boolean("detail_known",known);
      if(!known) ++s->unknown_callback_details;
      if(s->emit(line)) ++s->callback_records;
    } else if(domain==CUPTI_CB_DOMAIN_RESOURCE) resource(*s,cbid,*static_cast<const CUpti_ResourceData *>(raw));
    else { ++s->errors; s->gap("cupti_state_or_unknown_callback"); }
  } catch(...) { ++s->serialization_errors; ++s->dropped; }
  inside_callback=false;
}

void activity(State & s, const CUpti_Activity & raw) {
  auto line=s.begin("activity"); line.number("activity_kind_id",static_cast<std::uint64_t>(raw.kind));
  bool known=true;
  const auto timing=[&](std::uint64_t start,std::uint64_t end,std::uint32_t context,std::uint32_t stream,std::uint32_t correlation) {
    line.number("start_ns",start); line.number("end_ns",end);
    line.number("context_id",s.id("cupti_context",context));
    line.number("stream_id",s.id("cupti_stream",(static_cast<std::uint64_t>(context)<<32) | stream));
    line.number("correlation_id",correlation); if(start==0 || end==0 || end<start) ++s.incomplete_activities;
  };
  switch(raw.kind) {
  case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
    const auto & k=reinterpret_cast<const CUpti_ActivityKernel12 &>(raw);
    line.string("activity_kind","kernel"); line.string("name",bounded_name(k.name));
    timing(k.start,k.end,k.contextId,k.streamId,k.correlationId);
    if(k.gridX<0 || k.gridY<0 || k.gridZ<0 || k.blockX<0 || k.blockY<0 || k.blockZ<0 || k.dynamicSharedMemory<0) { known=false; ++s.incomplete_activities; }
    else geometry(line,static_cast<std::uint64_t>(k.gridX),static_cast<std::uint64_t>(k.gridY),static_cast<std::uint64_t>(k.gridZ),static_cast<std::uint64_t>(k.blockX),static_cast<std::uint64_t>(k.blockY),static_cast<std::uint64_t>(k.blockZ),static_cast<std::uint64_t>(k.dynamicSharedMemory));
    break;
  }
  case CUPTI_ACTIVITY_KIND_RUNTIME:
  case CUPTI_ACTIVITY_KIND_DRIVER: {
    const auto & a=reinterpret_cast<const CUpti_ActivityAPI &>(raw);
    line.string("activity_kind",raw.kind==CUPTI_ACTIVITY_KIND_RUNTIME ? "api_runtime" : "api_driver");
    line.number("start_ns",a.start); line.number("end_ns",a.end); line.number("correlation_id",a.correlationId);
    if(a.start==0 || a.end==0 || a.end<a.start) ++s.incomplete_activities;
    break;
  }
  case CUPTI_ACTIVITY_KIND_MEMCPY: {
    const auto & a=reinterpret_cast<const CUpti_ActivityMemcpy6 &>(raw);
    line.string("activity_kind","memcpy"); line.number("size_bytes",a.bytes); timing(a.start,a.end,a.contextId,a.streamId,a.correlationId); break;
  }
  case CUPTI_ACTIVITY_KIND_MEMSET: {
    const auto & a=reinterpret_cast<const CUpti_ActivityMemset4 &>(raw);
    line.string("activity_kind","memset"); line.number("size_bytes",a.bytes); timing(a.start,a.end,a.contextId,a.streamId,a.correlationId); break;
  }
  case CUPTI_ACTIVITY_KIND_MEMORY2: {
    const auto & a=reinterpret_cast<const CUpti_ActivityMemory4 &>(raw);
    line.string("activity_kind","memory"); line.number("size_bytes",a.bytes); line.number("correlation_id",a.correlationId);
    line.number("memory_kind_id",static_cast<std::uint64_t>(a.memoryKind)); line.number("operation_id",static_cast<std::uint64_t>(a.memoryOperationType));
    // Activities can arrive after API free/reuse: never reconstruct identity by address alone.
    add_range(line,{}); known=false; break;
  }
  case CUPTI_ACTIVITY_KIND_FUNCTION: {
    const auto & a=reinterpret_cast<const CUpti_ActivityFunction &>(raw);
    line.string("activity_kind","function"); line.string("name",bounded_name(a.name));
    const auto context=s.id("cupti_context",a.contextId);
    line.number("context_id",context);
    line.number("module_id",s.id(s.trace_schema_version==3 ? "cupti_module_"+std::to_string(context) : "cupti_module",a.moduleId));
    line.number("function_id",s.id(s.trace_schema_version==3 ? "cupti_function_"+std::to_string(context) : "cupti_function",a.id));
    if (s.trace_schema_version==3) {
      line.number("function_index",a.functionIndex);
      // Buffered activities do not establish which reused module generation ran.
      line.number("module_generation",0);
    }
    break;
  }
  default: line.string("activity_kind","unknown"); ++s.unknown_activity_kinds; known=false; break;
  }
  line.boolean("detail_known",known); if(s.emit(line)) ++s.activity_records;
}

void CUPTIAPI buffer_requested(std::uint8_t ** buffer,std::size_t * size,std::size_t * max_records,CUpti_BufferCallbackRequestInfo *) noexcept {
  *buffer=nullptr; *size=0; *max_records=0;
  auto * s=state; if(!s) return;
  if(s->buffers_outstanding.fetch_add(1)>=maximum_buffers) { --s->buffers_outstanding; ++s->dropped; return; }
  // malloc meets the activity API's eight-byte alignment requirement on x64.
  *buffer=static_cast<std::uint8_t *>(std::malloc(activity_buffer_bytes));
  if(!*buffer) { --s->buffers_outstanding; ++s->dropped; return; }
  *size=activity_buffer_bytes; ++s->buffers_requested;
}
void CUPTIAPI buffer_completed(std::uint8_t * buffer,std::size_t,std::size_t valid,CUpti_BufferCallbackCompleteInfo *) noexcept {
  auto * s=state;
  if(s) {
    try {
      CUpti_Activity * record=nullptr;
      while(valid) {
        const auto result=cuptiActivityGetNextRecord_v2(s->subscriber,buffer,valid,&record);
        if(result==CUPTI_ERROR_MAX_LIMIT_REACHED) break;
        if(result!=CUPTI_SUCCESS || record==nullptr) { ++s->errors; break; }
        std::lock_guard lock(s->mutex); activity(*s,*record);
      }
      std::size_t dropped=0;
      const auto result=cuptiActivityGetNumDroppedRecords_v2(s->subscriber,nullptr,0,&dropped);
      account(result); s->dropped+=dropped;
    } catch(...) { ++s->serialization_errors; ++s->dropped; }
    ++s->buffers_completed; --s->buffers_outstanding;
  }
  std::free(buffer);
}
void exit_footer() noexcept {
  auto * s=state; if(!s || s->finalized) return;
  s->stop=true;
  // Never block on a potentially terminated thread, or call CUPTI during exit.
  if(s->mutex.try_lock()) {
    try { s->summary(false); s->finalized=true; } catch(...) { ++s->serialization_errors; }
    s->mutex.unlock();
  }
}
void hash_modules(State & s) noexcept {
  for (;;) {
    ModuleCopy item;
    {
      std::lock_guard lock(s.mutex);
      if (s.finalized || !s.module_copies.pop(item)) return;
    }
    try {
      const auto bytes=item.bytes.size();
      const auto digest=platform::sha256_bytes(item.bytes);
      // Free the owned copy before releasing its admission credit.
      std::vector<std::byte>().swap(item.bytes);
      std::lock_guard lock(s.mutex);
      s.module_copies.retire(bytes);
      auto line=s.begin("module_hash");
      line.number("context_id",item.context); line.number("module_id",item.module);
      line.number("module_generation",item.generation); line.number("source_sequence",item.source_sequence);
      line.number("size_bytes",bytes); line.string("sha256",digest);
      if (s.emit(line)) ++s.module_hash_records;
    } catch (...) { ++s.errors; ++s.dropped; return; }
  }
}
int initialize() noexcept {
  std::lock_guard lock(initialization_mutex);
  if(state) return state->initialized ? 1 : 0;
  try {
    auto * s=new State;
    state=s;
#ifdef _WIN32
    char version_value[3]{};
    SetLastError(ERROR_SUCCESS);
    const DWORD version_length=GetEnvironmentVariableA("XVRAM_AUDIT_TRACE_VERSION",version_value,
                                                       static_cast<DWORD>(sizeof(version_value)));
    if(version_length>=sizeof(version_value)) throw std::invalid_argument("unsupported trace version");
    s->trace_schema_version=trace_version(version_length==0 && GetLastError()==ERROR_ENVVAR_NOT_FOUND ? nullptr : version_value);
#else
    s->trace_schema_version=trace_version(std::getenv("XVRAM_AUDIT_TRACE_VERSION"));
#endif
    std::filesystem::path trace_path;
#ifdef _WIN32
    const DWORD needed=GetEnvironmentVariableW(L"XVRAM_AUDIT_TRACE",nullptr,0);
    if(needed==0 || needed>32768) return 0;
    std::wstring path(needed,L'\0');
    const DWORD copied=GetEnvironmentVariableW(L"XVRAM_AUDIT_TRACE",path.data(),needed);
    if(copied==0 || copied>=needed) return 0;
    path.resize(copied); trace_path=path;
#else
    const char * path=std::getenv("XVRAM_AUDIT_TRACE");
    if(path==nullptr || *path=='\0') return 0;
    trace_path=path;
#endif
    if(!s->sink.open(trace_path)) return 0;
    {
      std::lock_guard trace_lock(s->mutex);
      auto line=s->begin("session"); line.string("collector_version","0.1");
#ifdef _WIN32
      line.number("process_id",GetCurrentProcessId());
#else
      line.number("process_id",static_cast<std::uint64_t>(getpid()));
#endif
      line.number("max_record_bytes",max_record_bytes); line.boolean("complete",false);
      line.boolean("kernel_arguments_captured",false); line.boolean("tensor_bounds_known",false);
      line.boolean("runtime_parameter_bytes_not_kernel_arguments",true);
      line.string("cublas_api_visibility","unavailable"); line.string("timestamp_clock","steady_clock"); line.string("activity_clock","cupti");
      if(!s->emit(line)) return 0;
      s->gap("cublas_api_not_observed_by_cupti");
    }
    if(std::atexit(exit_footer)!=0) { ++s->errors; return 0; }
    CUpti_SubscriberParams params={sizeof(CUpti_SubscriberParams),nullptr,nullptr,0,0};
    if(cuptiSubscribe_v2(&s->subscriber,callback,nullptr,&params)!=CUPTI_SUCCESS) { ++s->errors; return 0; }
    account(cuptiActivityRegisterCallbacks_v2(s->subscriber,buffer_requested,buffer_completed));
    account(cuptiEnableDomain(1,s->subscriber,CUPTI_CB_DOMAIN_RUNTIME_API));
    account(cuptiEnableDomain(1,s->subscriber,CUPTI_CB_DOMAIN_DRIVER_API));
    account(cuptiEnableDomain(1,s->subscriber,CUPTI_CB_DOMAIN_RESOURCE));
    account(cuptiEnableCallback(1,s->subscriber,CUPTI_CB_DOMAIN_STATE,CUPTI_CBID_STATE_FATAL_ERROR));
    for(const auto kind:activities) account(cuptiActivityEnable_v2(s->subscriber,kind,nullptr));
    s->flusher=std::thread([s] {
      std::unique_lock wait_lock(s->wait_mutex);
      while(!s->stop) {
        s->wake.wait_for(wait_lock,std::chrono::milliseconds(200),[s]{return s->stop.load();});
        if(s->stop) break;
        wait_lock.unlock(); account(cuptiActivityFlushAll(0));
        if (s->trace_schema_version==3) hash_modules(*s);
        wait_lock.lock();
      }
    });
    s->initialized=true;
    return 1;
  } catch(...) { if(state) ++state->errors; return 0; }
}
int finalize() noexcept {
  auto * s=state;
  if(!s || !s->initialized || inside_callback) return 0;
  if(s->finalized) return 1;
  if(s->finalizing.exchange(true)) return 0;
  try {
    // Caller contract: all application CUDA activity is already synchronized and
    // no thread may enter CUDA again. This function itself never synchronizes CUDA.
    s->stop=true; s->wake.notify_all();
    if(s->flusher.joinable()) s->flusher.join();
    for(const auto kind:activities) account(cuptiActivityDisable_v2(s->subscriber,kind,nullptr));
    account(cuptiActivityFlushAll(1));
    std::size_t dropped=0; account(cuptiActivityGetNumDroppedRecords_v2(s->subscriber,nullptr,0,&dropped)); s->dropped+=dropped;
    account(cuptiEnableAllDomains(0,s->subscriber));
    account(cuptiUnsubscribe(s->subscriber));
    if (s->trace_schema_version==3) hash_modules(*s);
    std::lock_guard trace_lock(s->mutex);
    s->summary(true); s->finalized=true;
    return s->errors==0 && s->serialization_errors==0 ? 1 : 0;
  } catch(...) { ++s->errors; return 0; }
}
} // namespace
} // namespace xvram::audit

#ifdef _WIN32
#define XVRAM_AUDIT_EXPORT extern "C" __declspec(dllexport)
#else
#define XVRAM_AUDIT_EXPORT extern "C" __attribute__((visibility("default")))
#endif
XVRAM_AUDIT_EXPORT int InitializeInjection() noexcept { return xvram::audit::initialize(); }
XVRAM_AUDIT_EXPORT int XvramAuditFinalize() noexcept { return xvram::audit::finalize(); }
