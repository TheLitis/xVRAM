// Windows-only, opt-in diagnostic CUDA Runtime proxy. Not a memory interceptor.
// No DllMain, patching, injection, kernel-argument reads or CUPTI dependency.
#include "core.hpp"
#include "compat_audit_collector/trace_core.hpp"
#include "platform/dynamic_library.hpp"
#include "platform/sha256.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <windows.h>

#include <chrono>
#include <filesystem>
#include <mutex>
#include <string>

namespace {
constexpr auto native_hash = "b00ca6f53699120da815bf3e06e2e4285fae2f201235b883dcbb50eec51e2a2a";
constexpr std::uint64_t trace_cap = 128ULL * 1024 * 1024;
char module_anchor;
thread_local bool active = false;
using xvram::audit::JsonLine;

struct State {
  std::recursive_mutex mutex;
  xvram::platform::DynamicLibrary runtime, driver;
  HANDLE trace = INVALID_HANDLE_VALUE;
  std::uint64_t sequence = 0, bytes = 0, calls = 0;
  bool initialized = false, poisoned = false, metadata = false;

  void initialize() {
    if (initialized) return;
    HMODULE self = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(&module_anchor), &self))
      throw std::runtime_error("proxy_location_failed");
    wchar_t module_path[32768]{};
    const auto length = GetModuleFileNameW(self, module_path, 32768);
    if (!length || length >= 32768) throw std::runtime_error("proxy_location_failed");
    const auto native = std::filesystem::path(module_path).parent_path() / L"xvram_native_cudart64_13.dll";
    if (xvram::platform::sha256_file(native).digest != native_hash)
      throw std::runtime_error("native_runtime_hash_mismatch");
    if (!runtime.open_absolute(native) || !driver.open_system({"nvcuda.dll"}))
      throw std::runtime_error("native_library_unavailable");
    wchar_t output[32768]{}, mode[32]{};
    const auto output_length = GetEnvironmentVariableW(L"XVRAM_LAUNCH_PROBE_TRACE", output, 32768);
    const auto mode_length = GetEnvironmentVariableW(L"XVRAM_LAUNCH_PROBE_MODE", mode, 32);
    if (!output_length || output_length >= 32768 || !mode_length || mode_length >= 32)
      throw std::runtime_error("missing_probe_configuration");
    const std::wstring selected(mode);
    if (selected != L"routing" && selected != L"metadata") throw std::runtime_error("invalid_probe_mode");
    metadata = selected == L"metadata";
    const std::filesystem::path path(output);
    if (!path.is_absolute()) throw std::runtime_error("trace_path_not_absolute");
    // The controller owns a fresh directory. Refuse replacement even in an
    // incorrectly configured direct invocation; never fall back to stdout.
    trace = CreateFileW(output, GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_NEW,
                       FILE_ATTRIBUTE_NORMAL, nullptr);
    if (trace == INVALID_HANDLE_VALUE) throw std::runtime_error("trace_create_failed");
    initialized = true;
  }
  template<class F> F rt(const char* name) {
    const auto result = runtime.symbol<F>(name);
    if (!result) throw std::runtime_error("runtime_export_missing");
    return result;
  }
  template<class F> F drv(const char* name) {
    const auto result = driver.symbol<F>(name);
    if (!result) throw std::runtime_error("driver_export_missing");
    return result;
  }
  JsonLine line(const char* kind, std::uint64_t call) {
    JsonLine result;
    result.number("schema_version", 1); result.string("record_type", "xvram.cuda_launch_probe");
    result.number("sequence", ++sequence); result.string("kind", kind); result.number("call_id", call);
    return result;
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
State& state() {
  // Do not unload CUDA or run arbitrary teardown while the Windows loader lock
  // is held. The owning controller reaps the process; this is not SDK cleanup.
  static auto* instance = new State;
  return *instance;
}
void checked(CUresult value) {
  if (value != CUDA_SUCCESS) throw std::runtime_error("driver_query_failed");
}
struct Query {
  State& s;
  const void* symbol;
  CUfunction function = nullptr;
  bool resolve() {
    cudaFunction_t resolved = nullptr;
    const auto get = s.rt<decltype(&cudaGetFuncBySymbol)>("cudaGetFuncBySymbol");
    if (get(&resolved, symbol) != cudaSuccess || !resolved) return false;
    function = static_cast<CUfunction>(resolved); // documented Runtime/Driver shared type
    CUmodule module = nullptr;
    checked(s.drv<decltype(&cuFuncGetModule)>("cuFuncGetModule")(&module, function));
    return module != nullptr;
  }
  std::string name() {
    const char* value = nullptr;
    checked(s.drv<decltype(&cuFuncGetName)>("cuFuncGetName")(&value, function));
    if (!value) throw std::runtime_error("null_function_name");
    std::size_t size = 0;
    while (size <= 4096 && value[size]) ++size;
    if (size > 4096) throw std::runtime_error("function_name_limit");
    return {value, size};
  }
  std::size_t count() {
    std::size_t value = 0;
    checked(s.drv<decltype(&cuFuncGetParamCount)>("cuFuncGetParamCount")(function, &value));
    return value;
  }
  xvram::launch_probe::Parameter parameter(std::size_t index) {
    xvram::launch_probe::Parameter result;
    checked(s.drv<decltype(&cuFuncGetParamInfo)>("cuFuncGetParamInfo")(
      function, index, &result.offset, &result.bytes));
    return result;
  }
};

template<class Forward>
cudaError_t launch(const char* api, const void* symbol, Forward forward) noexcept {
  if (active) return cudaErrorUnknown; // never recurse from a profiling callback
  active = true;
  struct Guard { ~Guard() { active = false; } } guard;
  try {
    auto& s = state();
    std::lock_guard lock(s.mutex);
    if (s.poisoned) return cudaErrorUnknown;
    try {
      s.initialize();
      const auto call = ++s.calls;
      const auto result = xvram::launch_probe::invoke([&] {
        if (!s.metadata) return xvram::launch_probe::Snapshot{};
        // A diagnostic query can perturb CUDA error/lazy-loading state. Refuse
        // metadata inspection when a pre-existing Runtime error is present.
        if (s.rt<decltype(&cudaPeekAtLastError)>("cudaPeekAtLastError")() != cudaSuccess)
          throw std::runtime_error("preexisting_runtime_error");
        Query query{s, symbol};
        return xvram::launch_probe::inspect(query);
      }, [&](const auto& snapshot) {
        auto begin = s.line("begin", call);
        begin.string("api", api); begin.boolean("metadata", s.metadata);
        begin.string("kernel_name", snapshot.name); begin.number("parameter_count", snapshot.parameters.size());
        begin.boolean("module_resolved", s.metadata);
        s.emit(begin);
        for (std::size_t i = 0; i < snapshot.parameters.size(); ++i) {
          auto parameter = s.line("parameter", call);
          parameter.number("index", i); parameter.number("offset_bytes", snapshot.parameters[i].offset);
          parameter.number("size_bytes", snapshot.parameters[i].bytes); s.emit(parameter);
        }
      }, [&] { return static_cast<int>(forward(s)); }, [&](int value) {
        auto end = s.line("end", call); end.integer("result", value); s.emit(end);
      });
      return static_cast<cudaError_t>(result);
    } catch (...) { s.poisoned = true; return cudaErrorUnknown; }
  } catch (...) { return cudaErrorUnknown; }
}
} // namespace

extern "C" cudaError_t CUDARTAPI xvram_probe_launch(
  const void* function, dim3 grid, dim3 block, void** args, std::size_t shared, cudaStream_t stream) {
  return launch("cudaLaunchKernel", function, [&](State& s) {
    return s.rt<decltype(&cudaLaunchKernel)>("cudaLaunchKernel")(function, grid, block, args, shared, stream);
  });
}
extern "C" cudaError_t CUDARTAPI xvram_probe_launch_ex(
  const cudaLaunchConfig_t* config, const void* function, void** args) {
  return launch("cudaLaunchKernelExC", function, [&](State& s) {
    return s.rt<decltype(&cudaLaunchKernelExC)>("cudaLaunchKernelExC")(config, function, args);
  });
}
