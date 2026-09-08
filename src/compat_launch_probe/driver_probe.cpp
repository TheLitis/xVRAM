// Active diagnostic boundary, loaded by CUDA's documented InitializeInjection
// mechanism ONLY in the pinned, controller-owned llama-completion process.
// Public Driver launch entry points only. No kernel argument reads or VMM work.
#include "core.hpp"
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
thread_local bool inside = false;
struct State {
  xvram::platform::DynamicLibrary driver;
  std::recursive_mutex calls_mutex;
  HANDLE trace = INVALID_HANDLE_VALUE;
  Launch legacy = nullptr, per_thread = nullptr;
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
    value.number("schema_version", 1); value.string("record_type", "xvram.cuda_launch_probe");
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
struct Query {
  State& state;
  CUfunction function;
  bool resolve() {
    if (!function) return false;
    CUmodule module = nullptr;
    checked(state.get<decltype(&cuFuncGetModule)>("cuFuncGetModule")(&module, function));
    return module != nullptr;
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

CUresult dispatch(bool per_thread, CUfunction function, unsigned gx, unsigned gy, unsigned gz,
                  unsigned bx, unsigned by, unsigned bz, unsigned shared, CUstream stream,
                  void** parameters, void** extra) noexcept {
  auto* s = current.load();
  if (!s) return CUDA_ERROR_UNKNOWN;
  // Public entry points can delegate to one another. Preserve their original
  // forwarding chain without recursively running diagnostics. This is NOT a
  // residency interceptor and never claims complete per-GPU-launch accounting.
  if (inside) return (per_thread ? s->per_thread : s->legacy)(
    function, gx, gy, gz, bx, by, bz, shared, stream, parameters, extra);
  inside = true;
  struct Guard { ~Guard() { inside = false; } } guard;
  try {
    std::lock_guard lock(s->calls_mutex);
    if (s->poisoned) return CUDA_ERROR_UNKNOWN;
    try {
      const auto id = ++s->call;
      const auto result = xvram::launch_probe::invoke([&] {
        if (!s->metadata) return xvram::launch_probe::Snapshot{};
        Query query{*s, function};
        return xvram::launch_probe::inspect(query);
      }, [&](const auto& snapshot) {
        auto begin = s->line("begin", id);
        begin.string("api", per_thread ? "cuLaunchKernel_resolved_ptsz" : "cuLaunchKernel");
        begin.boolean("metadata", s->metadata); begin.boolean("module_resolved", s->metadata);
        begin.string("kernel_name", snapshot.name); begin.number("parameter_count", snapshot.parameters.size());
        s->emit(begin);
        for (std::size_t i = 0; i < snapshot.parameters.size(); ++i) {
          auto parameter = s->line("parameter", id);
          parameter.number("index", i); parameter.number("offset_bytes", snapshot.parameters[i].offset);
          parameter.number("size_bytes", snapshot.parameters[i].bytes); s->emit(parameter);
        }
      }, [&] { return static_cast<int>((per_thread ? s->per_thread : s->legacy)(
          function, gx, gy, gz, bx, by, bz, shared, stream, parameters, extra)); },
      [&](int code) { auto end = s->line("end", id); end.integer("result", code); s->emit(end); });
      return static_cast<CUresult>(result);
    } catch (...) { s->poisoned = true; return CUDA_ERROR_UNKNOWN; }
  } catch (...) { return CUDA_ERROR_UNKNOWN; }
}
CUresult CUDAAPI legacy(CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by,
                      unsigned bz, unsigned shared, CUstream stream, void** params, void** extra) {
  return dispatch(false, f, gx, gy, gz, bx, by, bz, shared, stream, params, extra);
}
CUresult CUDAAPI per_thread(CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by,
                          unsigned bz, unsigned shared, CUstream stream, void** params, void** extra) {
  return dispatch(true, f, gx, gy, gz, bx, by, bz, shared, stream, params, extra);
}

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
    s->trace = CreateFileW(output, GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (s->trace == INVALID_HANDLE_VALUE) throw std::runtime_error("trace_create_failed");
    std::lock_guard calls_lock(s->calls_mutex);
    Transaction transaction;
    transaction.begin();
    if (DetourAttach(reinterpret_cast<PVOID*>(&s->legacy), reinterpret_cast<PVOID>(legacy)) != NO_ERROR ||
        DetourAttach(reinterpret_cast<PVOID*>(&s->per_thread), reinterpret_cast<PVOID>(per_thread)) != NO_ERROR)
      throw std::runtime_error("hook_attach_failed");
    transaction.commit();
    auto setup = s->line("setup", 0); setup.number("hooks", 2); s->emit(setup);
    return 1;
  } catch (...) { if (auto* failed = current.load()) failed->poisoned = true; return 0; }
}
