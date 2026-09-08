// Diagnostic sidecar only. Library outputs are copied while CUPTI data is valid.
// No CUDA calls or submissions from callbacks; no serialized native values.
#include "witness.hpp"
#include "compat_audit_collector/trace_core.hpp"
#include "compat_audit_collector/module_evidence.hpp"
#include <windows.h>
#include <atomic>
#include <cstdlib>
#include <mutex>

namespace xvram::launch_probe::witness {
namespace {
struct State {
  HANDLE file = INVALID_HANDLE_VALUE;
  std::mutex mutex;
  audit::Registry ids;
  audit::ModuleLifetimes libraries;
  std::uint64_t sequence = 0, bytes = 0;
  std::atomic_uint64_t errors{0};
  bool closed = false;
  audit::JsonLine line(const char* kind) {
    audit::JsonLine line;
    line.number("schema_version", 1); line.string("record_type", "xvram.cuda_identity_witness");
    line.number("sequence", ++sequence); line.string("kind", kind);
    return line;
  }
  void emit(const audit::JsonLine& line) {
    const auto data = line.finish();
    constexpr std::uint64_t cap = 64ULL * 1024 * 1024;
    if (closed || data.size() > cap-bytes) throw std::runtime_error("witness_capacity");
    DWORD written = 0;
    if (!WriteFile(file, data.data(), static_cast<DWORD>(data.size()), &written, nullptr) || written != data.size())
      throw std::runtime_error("witness_write");
    bytes += data.size();
  }
  std::uint64_t library(CUlibrary lib) { return ids.identity("library", reinterpret_cast<std::uintptr_t>(lib)); }
};
State* state = nullptr; // process lifetime, no blocking/CUDA teardown under loader lock
void footer() noexcept {
  if (!state || !state->mutex.try_lock()) return;
  try {
    auto line=state->line("summary"); line.number("errors",state->errors);
    line.boolean("terminal_complete",false); state->emit(line); state->closed=true;
  } catch (...) { ++state->errors; }
  state->mutex.unlock();
}
}
void initialize() {
  if (state) throw std::runtime_error("witness_reinitialize");
  wchar_t path[32768]{};
  const auto n=GetEnvironmentVariableW(L"XVRAM_IDENTITY_WITNESS_TRACE",path,32768);
  if (!n || n>=32768) throw std::runtime_error("witness_path_missing");
  state=new State;
  state->file=CreateFileW(path,GENERIC_WRITE,FILE_SHARE_READ,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
  if(state->file==INVALID_HANDLE_VALUE) throw std::runtime_error("witness_create");
  auto line=state->line("session"); line.boolean("terminal_complete",false); state->emit(line);
  if(std::atexit(footer)!=0) throw std::runtime_error("witness_atexit");
}
void api(const CUpti_CallbackData& data, std::uint64_t api_id) noexcept {
  if(!state || !data.functionName || data.callbackSite!=CUPTI_API_EXIT) return;
  const std::string_view symbol(data.functionName);
  if(symbol!="cuLibraryLoadData" && symbol!="cuLibraryLoadFromFile" && symbol!="cuLibraryUnload") return;
  try {
    if(!data.functionParams || !data.functionReturnValue) throw std::runtime_error("witness_callback_data");
    const auto status=*static_cast<const int*>(data.functionReturnValue);
    std::lock_guard lock(state->mutex);
    auto line=state->line("library_api"); line.number("api_id",api_id); line.string("symbol",symbol);
    line.integer("result",status);
    if(status==0) {
      CUlibrary lib=nullptr;
      if(symbol=="cuLibraryUnload") lib=static_cast<const cuLibraryUnload_params*>(data.functionParams)->library;
      else {
        const auto output=symbol=="cuLibraryLoadData" ?
          static_cast<const cuLibraryLoadData_params*>(data.functionParams)->library :
          static_cast<const cuLibraryLoadFromFile_params*>(data.functionParams)->library;
        if(output) lib=*output;
      }
      if(!lib) throw std::runtime_error("witness_null_library");
      const auto id=state->library(lib);
      const auto life=symbol=="cuLibraryUnload" ? state->libraries.unload(0,id) : state->libraries.load(0,id);
      if(!life.live || !life.generation) throw std::runtime_error("witness_library_lifetime");
      line.number("library_id",id); line.number("library_generation",life.generation);
    } else { line.number("library_id",0); line.number("library_generation",0); }
    state->emit(line);
  } catch (...) { ++state->errors; }
}
void launch(std::uint64_t call, CUlibrary library, CUmodule module, bool equal) {
  if(!state || !module) throw std::runtime_error("witness_launch_missing");
  std::lock_guard lock(state->mutex);
  if(state->errors) throw std::runtime_error("witness_poisoned");
  auto line=state->line("launch_identity"); line.number("call_id",call);
  const auto id=library ? state->library(library) : 0;
  const auto life=state->libraries.lookup(0,id);
  line.number("library_id",id); line.number("library_generation",life.live ? life.generation : 0);
  // A module observation ID is not an authenticated module generation or cubin ID.
  line.number("module_observation_id",state->ids.identity("module_observation",reinterpret_cast<std::uintptr_t>(module)));
  line.boolean("library_module_equal",equal); line.boolean("cubin_binding_proven",false);
  state->emit(line);
}
}
