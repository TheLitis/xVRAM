// Active diagnostic boundary, loaded by CUDA's documented InitializeInjection
// mechanism ONLY in the pinned, controller-owned llama-completion process.
// Public Driver launch entry points only. The optional typed witness reads
// reviewed host argument fields; no device-memory dereference or VMM work.
#include "core.hpp"
#include "census.hpp"
#include "witness.hpp"
#ifdef XVRAM_PROBE_TEARDOWN_WITNESS
#include "teardown_witness.hpp"
#endif
#ifdef XVRAM_PROBE_MEMORY_WITNESS
#include "memory_witness.hpp"
#endif
#ifdef XVRAM_PROBE_KERNEL_CAPTURE
#include "kernel_capture.hpp"
#endif
#ifdef XVRAM_PROBE_PC_WITNESS
#include "sampling.hpp"
#include "execution_witness.hpp"
#include "postmortem.hpp"
#endif
#include "compat_audit_collector/trace_core.hpp"
#include "platform/dynamic_library.hpp"
#include "platform/sha256.hpp"
#include <cuda.h>
#include <windows.h>
#include <tlhelp32.h>
#include <detours.h>
#include <filesystem>
#include <atomic>
#include <cstring>
#include <mutex>

namespace {
using Launch = decltype(&cuLaunchKernel);
using xvram::audit::JsonLine;
constexpr std::uint64_t trace_cap = 128ULL * 1024 * 1024;
#ifdef XVRAM_PROBE_LEGACY_RESOLVER
constexpr unsigned trace_version = 2, hook_count = 3;
#else
constexpr unsigned trace_version = 1, hook_count = 2;
#endif
thread_local bool inside = false;
struct State {
  xvram::platform::DynamicLibrary driver;
  std::recursive_mutex calls_mutex;
  HANDLE trace = INVALID_HANDLE_VALUE;
  Launch legacy = nullptr, per_thread = nullptr, resolved_legacy = nullptr;
  bool metadata = false;
  std::atomic_bool poisoned{false};
  std::uint64_t sequence = 0, call = 0, bytes = 0;
  template<class F> F get(const char* name) {
    const auto result = driver.symbol<F>(name);
    if (!result) throw std::runtime_error("driver_export_missing");
    return result;
  }
  JsonLine line(const char* kind, std::uint64_t id) {
    JsonLine value;
    value.number("schema_version", trace_version); value.string("record_type", "xvram.cuda_launch_probe");
    value.number("sequence", ++sequence); value.string("kind", kind); value.number("call_id", id);
    return value;
  }
  void emit(const JsonLine& line) {
    const auto data = line.finish();
    if (data.size() > trace_cap - bytes) throw std::runtime_error("trace_capacity");
    DWORD written = 0;
    if (!WriteFile(trace, data.data(), static_cast<DWORD>(data.size()), &written, nullptr) || written != data.size())
      throw std::runtime_error("trace_write_failed");
    bytes += data.size();
  }
};
std::atomic<State*> current{nullptr}; // process lifetime; never CUDA teardown under loader lock
std::mutex initialization;
void checked(CUresult code) { if (code != CUDA_SUCCESS) throw std::runtime_error("driver_query_failed"); }
#ifdef XVRAM_PROBE_LEGACY_RESOLVER
struct ContextlessQuery {
  State& state;
  CUkernel kernel;
  CUstream stream;
  CUcontext current() {
    CUcontext result = nullptr;
    checked(state.get<decltype(&cuCtxGetCurrent)>("cuCtxGetCurrent")(&result));
    return result;
  }
  CUcontext stream_context(CUcontext context) {
    if (!stream) return context;
    CUcontext result = nullptr;
    checked(state.get<decltype(&cuStreamGetCtx)>("cuStreamGetCtx")(stream, &result));
    return result;
  }
  CUfunction function() {
    CUfunction result = nullptr;
    checked(state.get<decltype(&cuKernelGetFunction)>("cuKernelGetFunction")(&result, kernel));
    return result;
  }
};
#endif
#ifdef XVRAM_PROBE_WITNESS
struct LibraryQuery {
  State& state;
  CUkernel kernel;
  CUlibrary library() {
    CUlibrary result=nullptr;
    checked(state.get<decltype(&cuKernelGetLibrary)>("cuKernelGetLibrary")(&result,kernel));
    return result;
  }
  CUmodule module(CUlibrary library) {
    CUmodule result=nullptr;
    checked(state.get<decltype(&cuLibraryGetModule)>("cuLibraryGetModule")(&result,library));
    return result;
  }
};
#endif
struct Query {
  State& state;
  CUfunction function;
  bool contextless = false;
  CUstream stream = nullptr;
  CUmodule resolved_module = nullptr;
  CUlibrary resolved_library = nullptr;
  CUkernel original_kernel = nullptr;
  bool library_module_equal = false;
  bool resolve() {
    if (!function) return false;
#ifdef XVRAM_PROBE_LEGACY_RESOLVER
    if (contextless) {
      // Explicit pinned-profile contract, never an INVALID_HANDLE retry or a
      // guess at the handle's internal representation. Leave launch f unchanged.
      original_kernel = reinterpret_cast<CUkernel>(function);
      ContextlessQuery query{state, original_kernel, stream};
      function = xvram::launch_probe::resolve_contextless(query);
    }
#endif
    checked(state.get<decltype(&cuFuncGetModule)>("cuFuncGetModule")(&resolved_module, function));
#ifdef XVRAM_PROBE_WITNESS
    if(contextless) {
      // The original CUkernel remains unchanged for the native forward.
      LibraryQuery query{state,original_kernel};
      resolved_library=xvram::launch_probe::resolve_library(query,resolved_module);
      library_module_equal=true;
    }
#endif
    return resolved_module != nullptr;
  }
  std::string name() {
    const char* name = nullptr;
    checked(state.get<decltype(&cuFuncGetName)>("cuFuncGetName")(&name, function));
    if (!name) throw std::runtime_error("null_function_name");
    std::size_t size = 0;
    while (size <= 4096 && name[size]) ++size;
    if (size > 4096) throw std::runtime_error("function_name_limit");
    return {name, size};
  }
  std::size_t count() {
    std::size_t count = 0;
    checked(state.get<decltype(&cuFuncGetParamCount)>("cuFuncGetParamCount")(function, &count));
    return count;
  }
  xvram::launch_probe::Parameter parameter(std::size_t index) {
    xvram::launch_probe::Parameter result;
    checked(state.get<decltype(&cuFuncGetParamInfo)>("cuFuncGetParamInfo")(
      function, index, &result.offset, &result.bytes));
    return result;
  }
};

CUresult dispatch(unsigned route, CUfunction function, unsigned gx, unsigned gy, unsigned gz,
                  unsigned bx, unsigned by, unsigned bz, unsigned shared, CUstream stream,
                  void** parameters, void** extra) noexcept {
  auto* s = current.load();
  if (!s) return CUDA_ERROR_UNKNOWN;
  const auto native = route == 2 ? s->resolved_legacy : route == 1 ? s->per_thread : s->legacy;
  if (!native) return CUDA_ERROR_UNKNOWN;
  // Public entry points can delegate to one another. Preserve their original
  // forwarding chain without recursively running diagnostics. This is NOT a
  // residency interceptor and never claims complete per-GPU-launch accounting.
  if (inside) return native(
    function, gx, gy, gz, bx, by, bz, shared, stream, parameters, extra);
  inside = true;
  struct Guard { ~Guard() { inside = false; } } guard;
  try {
    std::lock_guard lock(s->calls_mutex);
    if (s->poisoned) return CUDA_ERROR_UNKNOWN;
    try {
      const auto id = ++s->call;
#ifdef XVRAM_PROBE_PC_WITNESS
      CUcontext sampling_context=nullptr;
      if(xvram::launch_probe::sampling::enabled()) {
        checked(s->get<decltype(&cuCtxGetCurrent)>("cuCtxGetCurrent")(&sampling_context));
        xvram::launch_probe::sampling::before(sampling_context);
      }
#endif
      const auto result = xvram::launch_probe::invoke([&] {
        if (!s->metadata) return xvram::launch_probe::Snapshot{};
        Query query{*s, function, route == 2, stream};
        auto snapshot=xvram::launch_probe::inspect(query);
#ifdef XVRAM_PROBE_WITNESS
        xvram::launch_probe::witness::launch(id,query.resolved_library,query.resolved_module,query.library_module_equal);
#endif
        return snapshot;
      }, [&](const auto& snapshot) {
#ifdef XVRAM_PROBE_KERNEL_CAPTURE
        if(xvram::launch_probe::kernel_capture::enabled()) {
          CUcontext context=nullptr;
          CUdevice device=-1;
          checked(s->get<decltype(&cuCtxGetCurrent)>("cuCtxGetCurrent")(&context));
          if(!context) throw std::runtime_error("kernel_capture_context_missing");
          checked(s->get<decltype(&cuCtxGetDevice)>("cuCtxGetDevice")(&device));
          if(stream) {
            CUcontext stream_context=nullptr;
            checked(s->get<decltype(&cuStreamGetCtx)>("cuStreamGetCtx")(stream,&stream_context));
            if(stream_context!=context) throw std::runtime_error("kernel_capture_stream_context_mismatch");
          }
          xvram::launch_probe::kernel_capture::launch(id,snapshot,
            {{gx,gy,gz},{bx,by,bz},shared},parameters,extra,device,context);
        }
#endif
        auto begin = s->line("begin", id);
        begin.string("api", route == 2 ? "cuLaunchKernel_resolved_legacy" : route == 1 ? "cuLaunchKernel_resolved_ptsz" : "cuLaunchKernel");
        begin.boolean("metadata", s->metadata); begin.boolean("module_resolved", s->metadata);
#ifdef XVRAM_PROBE_LEGACY_RESOLVER
        begin.string("handle_kind", route == 2 ? "contextless_kernel" : "function");
#endif
        begin.string("kernel_name", snapshot.name); begin.number("parameter_count", snapshot.parameters.size());
        s->emit(begin);
        for (std::size_t i = 0; i < snapshot.parameters.size(); ++i) {
          auto parameter = s->line("parameter", id);
          parameter.number("index", i); parameter.number("offset_bytes", snapshot.parameters[i].offset);
          parameter.number("size_bytes", snapshot.parameters[i].bytes); s->emit(parameter);
        }
      }, [&] {
        xvram::launch_probe::census::Scope marked_forward(id);
        return static_cast<int>(native(
          function, gx, gy, gz, bx, by, bz, shared, stream, parameters, extra)); },
      [&](int code) {
#ifdef XVRAM_PROBE_KERNEL_CAPTURE
        xvram::launch_probe::kernel_capture::returned(id,code);
#endif
#ifdef XVRAM_PROBE_PC_WITNESS
        if(xvram::launch_probe::sampling::enabled()) {
          if(code!=CUDA_SUCCESS) throw std::runtime_error("sampling_launch_failed");
          xvram::launch_probe::sampling::after(id,sampling_context,
            s->get<decltype(&cuCtxSynchronize)>("cuCtxSynchronize")());
        }
#endif
        auto end = s->line("end", id); end.integer("result", code); s->emit(end);
      });
      return static_cast<CUresult>(result);
    } catch (...) { s->poisoned = true; return CUDA_ERROR_UNKNOWN; }
  } catch (...) { return CUDA_ERROR_UNKNOWN; }
}
CUresult CUDAAPI legacy(CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by,
                      unsigned bz, unsigned shared, CUstream stream, void** params, void** extra) {
  return dispatch(0, f, gx, gy, gz, bx, by, bz, shared, stream, params, extra);
}
CUresult CUDAAPI per_thread(CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by,
                          unsigned bz, unsigned shared, CUstream stream, void** params, void** extra) {
  return dispatch(1, f, gx, gy, gz, bx, by, bz, shared, stream, params, extra);
}
#ifdef XVRAM_PROBE_LEGACY_RESOLVER
CUresult CUDAAPI resolved_legacy(CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by,
                               unsigned bz, unsigned shared, CUstream stream, void** params, void** extra) {
  return dispatch(2, f, gx, gy, gz, bx, by, bz, shared, stream, params, extra);
}
#endif

struct Transaction {
  bool active = false;
  std::vector<HANDLE> threads;
  ~Transaction() {
    if (active) DetourTransactionAbort();
    for (const auto thread : threads) CloseHandle(thread);
  }
  void begin() {
    threads.reserve(256); // allocate the ownership ledger before suspending any thread
    if (DetourTransactionBegin() != NO_ERROR) throw std::runtime_error("hook_transaction_failed");
    active = true;
    // Detours updates PCs of the enrolled threads atomically. Never target a
    // different process; unknown/enrollment failures abort the whole transaction.
    const HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snapshot == INVALID_HANDLE_VALUE) throw std::runtime_error("thread_snapshot_failed");
    struct Close { HANDLE h; ~Close() { CloseHandle(h); } } closer{snapshot};
    THREADENTRY32 entry{}; entry.dwSize = sizeof(entry);
    if (!Thread32First(snapshot, &entry)) throw std::runtime_error("thread_snapshot_failed");
    do {
      if (entry.th32OwnerProcessID != GetCurrentProcessId()) continue;
      if (threads.size() >= 256) throw std::runtime_error("thread_limit");
      if (entry.th32ThreadID == GetCurrentThreadId()) continue;
      const HANDLE thread = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_SET_CONTEXT |
                                       THREAD_QUERY_INFORMATION, FALSE, entry.th32ThreadID);
      if (!thread) throw std::runtime_error("thread_open_failed");
      if (GetProcessIdOfThread(thread) != GetCurrentProcessId()) {
        CloseHandle(thread); throw std::runtime_error("thread_owner_changed");
      }
      threads.push_back(thread);
      if (DetourUpdateThread(thread) != NO_ERROR) throw std::runtime_error("thread_enrollment_failed");
    } while (Thread32Next(snapshot, &entry));
    if (GetLastError() != ERROR_NO_MORE_FILES) throw std::runtime_error("thread_snapshot_truncated");
  }
  void commit() {
    const auto code = DetourTransactionCommit();
    active = false;
    if (code != NO_ERROR) throw std::runtime_error("hook_commit_failed");
  }
};
#ifdef XVRAM_PROBE_PC_WITNESS
using ProcessExit = void (WINAPI*)(unsigned long);
ProcessExit native_process_exit=nullptr;
void WINAPI diagnostic_process_exit(unsigned long code) noexcept {
  if(auto* s=current.load()) {
    try {
      std::lock_guard lock(s->calls_mutex);
      if(xvram::launch_probe::sampling::enabled() || xvram::launch_probe::execution_witness::enabled()) {
        xvram::launch_probe::census::stop_flusher();
        xvram::launch_probe::postmortem::seal();
        xvram::launch_probe::execution_witness::seal();
        xvram::launch_probe::execution_witness::terminal_tool=true;
        const auto result=s->get<decltype(&cuCtxSynchronize)>("cuCtxSynchronize")();
        xvram::launch_probe::sampling::finish(result);
        const bool drained=xvram::launch_probe::census::drain(result==CUDA_SUCCESS);
        xvram::launch_probe::postmortem::drained(result==CUDA_SUCCESS,drained);
        xvram::launch_probe::execution_witness::finish(result==CUDA_SUCCESS,drained);
#ifdef XVRAM_PROBE_TEARDOWN_WITNESS
        xvram::launch_probe::teardown_witness::checkpoint();
#endif
#ifdef XVRAM_PROBE_KERNEL_CAPTURE
        xvram::launch_probe::kernel_capture::finish();
#endif
        if(result!=CUDA_SUCCESS || !drained) throw std::runtime_error("diagnostic_terminal_drain_failed");
        xvram::launch_probe::postmortem::finish();
      }
    } catch(...) { xvram::launch_probe::postmortem::error(); s->poisoned=true; code=27; }
  }
  xvram::launch_probe::execution_witness::terminal_tool=false;
  native_process_exit(code);
}
#endif
} // namespace

extern "C" __declspec(dllexport) int InitializeInjection() noexcept {
  try {
    std::lock_guard lock(initialization);
    if (const auto* existing = current.load()) return existing->poisoned ? 0 : 1;
    wchar_t executable[32768]{}, output[32768]{}, mode[32]{};
    const auto length = GetModuleFileNameW(nullptr, executable, 32768);
    if (!length || length >= 32768) return 0;
    const std::filesystem::path path(executable);
    if (path.filename() != L"llama-completion.exe" ||
        xvram::platform::sha256_file(path).digest != "10678184d600ff60f6b3962771b6b2bef48477755236a02e2f7f7f00c0d1fa8d" ||
        xvram::platform::sha256_file(path.parent_path() / "ggml-cuda.dll").digest !=
          "88350839e27a43212a52cf6686562b2b6ef1498c59391dfb8cda49f3ebd89a62") return 0;
    const auto n = GetEnvironmentVariableW(L"XVRAM_LAUNCH_PROBE_TRACE", output, 32768);
    const auto m = GetEnvironmentVariableW(L"XVRAM_LAUNCH_PROBE_MODE", mode, 32);
    if (!n || n >= 32768 || !m || m >= 32 || !std::filesystem::path(output).is_absolute()) return 0;
    if (std::wstring_view(mode) != L"routing" && std::wstring_view(mode) != L"metadata") return 0;
    auto* s = new State;
    current = s;
    s->metadata = std::wstring_view(mode) == L"metadata";
#ifdef XVRAM_PROBE_WITNESS
    if(!s->metadata) throw std::runtime_error("witness_requires_metadata");
    xvram::launch_probe::witness::initialize();
#endif
#ifdef XVRAM_PROBE_PC_WITNESS
    xvram::launch_probe::execution_witness::initialize();
    xvram::launch_probe::postmortem::initialize();
#endif
#ifdef XVRAM_PROBE_CENSUS
#ifdef XVRAM_PROBE_TEARDOWN_WITNESS
    xvram::launch_probe::teardown_witness::initialize();
#endif
#ifdef XVRAM_PROBE_MEMORY_WITNESS
    // Observe allocations from the first enabled Driver callback, including
    // startup queries. This observer never submits CUDA work in callbacks.
    xvram::launch_probe::memory_witness::initialize();
#endif
#ifdef XVRAM_PROBE_KERNEL_CAPTURE
    xvram::launch_probe::kernel_capture::initialize();
#endif
    xvram::launch_probe::census::initialize();
#endif
#ifdef XVRAM_PROBE_PC_WITNESS
    xvram::launch_probe::sampling::initialize();
#endif
    if (!s->driver.open_system({"nvcuda.dll"})) throw std::runtime_error("driver_unavailable");
    s->legacy = s->get<Launch>("cuLaunchKernel");
    // The saved resolver trace requests version 7000 with per-thread-default
    // stream semantics. Its returned entry point is not assumed to equal the
    // exported cuLaunchKernel_ptsz stub. No private export-table layout is read.
    using Resolve = CUresult (CUDAAPI*)(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*);
    void* resolved = nullptr;
    CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    checked(s->get<Resolve>("cuGetProcAddress_v2")("cuLaunchKernel", &resolved, 7000,
      CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM, &status));
    if (!resolved || status != CU_GET_PROC_ADDRESS_SUCCESS) throw std::runtime_error("launch_resolution_failed");
    static_assert(sizeof(s->per_thread) == sizeof(resolved));
    std::memcpy(&s->per_thread, &resolved, sizeof(resolved));
    if (s->legacy == s->per_thread) throw std::runtime_error("unexpected_aliased_launch_exports");
#ifdef XVRAM_PROBE_LEGACY_RESOLVER
    resolved = nullptr; status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    checked(s->get<Resolve>("cuGetProcAddress_v2")("cuLaunchKernel", &resolved, 7000,
      CU_GET_PROC_ADDRESS_LEGACY_STREAM, &status));
    if (!resolved || status != CU_GET_PROC_ADDRESS_SUCCESS) throw std::runtime_error("legacy_launch_resolution_failed");
    std::memcpy(&s->resolved_legacy, &resolved, sizeof(resolved));
    if (s->resolved_legacy == s->legacy || s->resolved_legacy == s->per_thread)
      throw std::runtime_error("unexpected_aliased_legacy_entry");
#endif
    s->trace = CreateFileW(output, GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (s->trace == INVALID_HANDLE_VALUE) throw std::runtime_error("trace_create_failed");
    std::lock_guard calls_lock(s->calls_mutex);
    Transaction transaction;
    transaction.begin();
#ifdef XVRAM_PROBE_PC_WITNESS
    // Official CUPTI Windows injection sample uses this pre-loader-teardown
    // boundary. It does NOT by itself assert that all application producers
    // were quiescent, or that the census was completely delivered.
    const auto ntdll=GetModuleHandleW(L"ntdll.dll");
    native_process_exit=ntdll ? reinterpret_cast<ProcessExit>(GetProcAddress(ntdll,"RtlExitUserProcess")) : nullptr;
    if(!native_process_exit || DetourAttach(reinterpret_cast<PVOID*>(&native_process_exit),
       reinterpret_cast<PVOID>(diagnostic_process_exit))!=NO_ERROR)
      throw std::runtime_error("diagnostic_exit_hook_failed");
#endif
    if (DetourAttach(reinterpret_cast<PVOID*>(&s->legacy), reinterpret_cast<PVOID>(legacy)) != NO_ERROR ||
        DetourAttach(reinterpret_cast<PVOID*>(&s->per_thread), reinterpret_cast<PVOID>(per_thread)) != NO_ERROR)
      throw std::runtime_error("hook_attach_failed");
#ifdef XVRAM_PROBE_LEGACY_RESOLVER
    if (DetourAttach(reinterpret_cast<PVOID*>(&s->resolved_legacy), reinterpret_cast<PVOID>(resolved_legacy)) != NO_ERROR)
      throw std::runtime_error("legacy_hook_attach_failed");
#endif
    transaction.commit();
    auto setup = s->line("setup", 0); setup.number("hooks", hook_count); s->emit(setup);
    return 1;
  } catch (...) { if (auto* failed = current.load()) failed->poisoned = true; return 0; }
}
