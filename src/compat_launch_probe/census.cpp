// Opt-in simultaneous census. No CUDA calls in callbacks or shutdown hooks.
#include "census.hpp"
#include "compat_audit_collector/trace_core.hpp"
#include "platform/sha256.hpp"
#include <cupti.h>
#include <windows.h>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <thread>

namespace xvram::launch_probe::census {
namespace {
using audit::JsonLine;
#ifdef XVRAM_PROBE_LEGACY_RESOLVER
constexpr std::uint64_t cap = 256ULL * 1024 * 1024;
constexpr unsigned trace_version = 2;
#else
constexpr std::uint64_t cap = 128ULL * 1024 * 1024;
constexpr unsigned trace_version = 1;
#endif
constexpr std::size_t buffer_bytes = 1024 * 1024;
struct State {
  HANDLE file = INVALID_HANDLE_VALUE;
  CUpti_SubscriberHandle subscriber = nullptr;
  std::mutex mutex;
  std::atomic_uint64_t errors{0}, dropped{0}, outstanding{0};
  std::uint64_t sequence = 0, bytes = 0, api_id = 0;
  bool closed = false;
  JsonLine line(const char* kind) {
    JsonLine line;
    line.number("schema_version", trace_version); line.string("record_type", "xvram.cuda_launch_census");
    line.number("sequence", ++sequence); line.string("kind", kind); return line;
  }
  void emit(const JsonLine& line) {
    const auto data = line.finish();
    if (closed || data.size() > cap - bytes) throw std::runtime_error("census_trace_capacity");
    DWORD written = 0;
    if (!WriteFile(file, data.data(), static_cast<DWORD>(data.size()), &written, nullptr) || written != data.size())
      throw std::runtime_error("census_trace_write");
    bytes += data.size();
  }
};
State* state = nullptr; // owned until process exit; no loader-lock teardown
thread_local bool in_callback = false;
thread_local ApiStack api_stack;
std::string name(const char* value) {
  if (!value) throw std::runtime_error("census_null_name");
  std::size_t n = 0;
  while (n <= 4096 && value[n]) ++n;
  if (n > 4096) throw std::runtime_error("census_name_limit");
  return {value, n};
}
void check(CUptiResult result) {
  if (result != CUPTI_SUCCESS) throw std::runtime_error("census_cupti_failure");
}
void CUPTIAPI callback(void*, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void* raw) noexcept {
  if (!state || !raw) return;
  if (in_callback) { ++state->dropped; return; }
  in_callback = true;
  try {
    std::lock_guard lock(state->mutex);
    if (!state->closed) {
      if (domain != CUPTI_CB_DOMAIN_RUNTIME_API && domain != CUPTI_CB_DOMAIN_DRIVER_API)
        throw std::runtime_error("census_callback_domain");
      const auto& data = *static_cast<const CUpti_CallbackData*>(raw);
      const bool exit = data.callbackSite == CUPTI_API_EXIT;
      ApiFrame frame;
      if (!exit) {
        frame = api_stack.enter(++state->api_id, current_call, static_cast<std::uint32_t>(domain), cbid, data.correlationId);
      } else {
        frame = api_stack.exit(current_call, static_cast<std::uint32_t>(domain), cbid, data.correlationId);
      }
      auto line = state->line(exit ? "api_exit" : "api_enter");
      line.number("api_id", frame.id); line.number("parent_api_id", frame.parent);
      line.string("domain", domain == CUPTI_CB_DOMAIN_RUNTIME_API ? "runtime" : "driver");
      line.number("correlation_id", data.correlationId);
      line.number("probe_call_id", current_call);
      line.string("symbol", name(data.functionName));
      if (exit) {
        if (!data.functionReturnValue) throw std::runtime_error("census_missing_result");
        line.integer("result", *static_cast<const int*>(data.functionReturnValue));
      }
      state->emit(line);
    }
  } catch (...) { ++state->errors; }
  in_callback = false;
}
void CUPTIAPI requested(std::uint8_t** buffer, std::size_t* size, std::size_t* records,
                       CUpti_BufferCallbackRequestInfo*) noexcept {
  *buffer = nullptr; *size = 0; *records = 0;
  if (!state) return;
  if (state->outstanding.fetch_add(1) >= 8) { --state->outstanding; ++state->dropped; return; }
  *buffer = static_cast<std::uint8_t*>(std::malloc(buffer_bytes));
  if (!*buffer) { --state->outstanding; ++state->errors; return; }
  *size = buffer_bytes;
}
void CUPTIAPI completed(std::uint8_t* buffer, std::size_t, std::size_t valid,
                       CUpti_BufferCallbackCompleteInfo*) noexcept {
  if (state) {
    try {
      CUpti_Activity* record = nullptr;
      while (valid) {
        const auto result = cuptiActivityGetNextRecord_v2(state->subscriber, buffer, valid, &record);
        if (result == CUPTI_ERROR_MAX_LIMIT_REACHED) break;
        check(result);
        if (!record) throw std::runtime_error("census_null_activity");
        std::lock_guard lock(state->mutex);
        if (state->closed) break;
        if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
          const auto& kernel = *reinterpret_cast<const CUpti_ActivityKernel12*>(record);
          auto line = state->line("kernel"); line.number("correlation_id", kernel.correlationId);
          line.string("name", name(kernel.name)); line.number("start_ns", kernel.start);
          line.number("end_ns", kernel.end); state->emit(line);
        } else if (record->kind != CUPTI_ACTIVITY_KIND_RUNTIME && record->kind != CUPTI_ACTIVITY_KIND_DRIVER) {
          throw std::runtime_error("census_unknown_activity");
        }
      }
      std::size_t dropped = 0;
      check(cuptiActivityGetNumDroppedRecords_v2(state->subscriber, nullptr, 0, &dropped));
      state->dropped += dropped;
    } catch (...) { ++state->errors; }
    --state->outstanding;
  }
  std::free(buffer);
}
void footer() noexcept {
  // No synchronize, CUPTI flush, join, or waiting for another thread at process
  // teardown. This terminal snapshot explicitly does NOT prove completeness.
  if (!state || !state->mutex.try_lock()) return;
  try {
    auto line = state->line("summary");
    line.number("errors", state->errors); line.number("dropped", state->dropped);
    line.number("buffers_outstanding", state->outstanding);
    line.boolean("terminal_complete", false); state->emit(line); state->closed = true;
  } catch (...) { ++state->errors; }
  state->mutex.unlock();
}
} // namespace
void initialize() {
  if (state) throw std::runtime_error("census_reinitialize");
  wchar_t path[32768]{};
  const auto n = GetEnvironmentVariableW(L"XVRAM_LAUNCH_CENSUS_TRACE", path, 32768);
  if (!n || n >= 32768) throw std::runtime_error("census_path_missing");
  state = new State;
  // Validate the actually loaded diagnostic dependency, not only a PATH entry.
  const auto cupti = GetModuleHandleW(L"cupti64_2026.2.1.dll");
  wchar_t loaded[32768]{};
  const auto length = cupti ? GetModuleFileNameW(cupti, loaded, 32768) : 0;
  constexpr const char* cupti_hash = "9b10d2fafaff1a4dc9e447c4a1355fccb04ee024fa7e7d28c9e4c5ab53347faf";
  if (!length || length >= 32768 || platform::sha256_file(loaded).digest != cupti_hash)
    throw std::runtime_error("census_cupti_pin");
  state->file = CreateFileW(path, GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (state->file == INVALID_HANDLE_VALUE) throw std::runtime_error("census_trace_create");
  {
    std::lock_guard lock(state->mutex);
    auto line = state->line("session"); line.boolean("terminal_complete", false);
    line.string("cupti_sha256", cupti_hash); state->emit(line);
  }
  if (std::atexit(footer) != 0) throw std::runtime_error("census_atexit");
  CUpti_SubscriberParams params = {sizeof(CUpti_SubscriberParams), nullptr, nullptr, 0, 0};
  check(cuptiSubscribe_v2(&state->subscriber, callback, nullptr, &params));
  check(cuptiActivityRegisterCallbacks_v2(state->subscriber, requested, completed));
  check(cuptiEnableDomain(1, state->subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
  check(cuptiEnableDomain(1, state->subscriber, CUPTI_CB_DOMAIN_DRIVER_API));
  check(cuptiEnableCallback(1, state->subscriber, CUPTI_CB_DOMAIN_STATE, CUPTI_CBID_STATE_FATAL_ERROR));
  for (const auto kind : {CUPTI_ACTIVITY_KIND_RUNTIME, CUPTI_ACTIVITY_KIND_DRIVER, CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL})
    check(cuptiActivityEnable_v2(state->subscriber, kind, nullptr));
  std::thread([] {
    for (;;) {
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      // CUPTI flushing occurs on an independent tool thread, never a callback.
      if (cuptiActivityFlushAll(0) != CUPTI_SUCCESS) ++state->errors;
    }
  }).detach();
}
} // namespace xvram::launch_probe::census
