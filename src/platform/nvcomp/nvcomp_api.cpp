#include "platform/nvcomp/nvcomp_api.hpp"

#include <array>
#include <filesystem>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

namespace xvram::nvcomp {
namespace {

[[nodiscard]] std::filesystem::path module_directory() {
#ifdef _WIN32
  HMODULE module = nullptr;
  if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                             GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                         reinterpret_cast<LPCWSTR>(&module_directory), &module) == FALSE) {
    return {};
  }
  std::array<wchar_t, 32'768> path{};
  const DWORD length = GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
  return length == 0 || length >= path.size()
             ? std::filesystem::path{}
             : std::filesystem::path(std::wstring(path.data(), length)).parent_path();
#else
  Dl_info info{};
  if (dladdr(reinterpret_cast<const void*>(&module_directory), &info) == 0 ||
      info.dli_fname == nullptr) {
    return {};
  }
  return std::filesystem::path(info.dli_fname).parent_path();
#endif
}

[[nodiscard]] bool open_app_local(platform::DynamicLibrary& library) {
  const std::filesystem::path directory = module_directory();
  if (directory.empty()) {
    return false;
  }
#ifdef _WIN32
  constexpr std::array names{L"nvcomp64_5.dll"};
#else
  constexpr std::array names{"libnvcomp.so.5", "libnvcomp.so"};
#endif
  for (const auto* name : names) {
    const std::filesystem::path candidate = directory / name;
    if (std::filesystem::is_regular_file(candidate) && library.open_absolute(candidate)) {
      return true;
    }
  }
  return false;
}

} // namespace

NvcompApi::NvcompApi(NvcompDispatch dispatch, std::string loaded_name)
    : dispatch_(dispatch), source_(NvcompLibrarySource::injected), load_attempted_(true),
      loaded_name_(std::move(loaded_name)) {
  validate_dispatch();
  validate_version();
}

NvcompLoadResult NvcompApi::load(const NvcompLoadOptions& options) {
  if (load_attempted_) {
    return {status_, false};
  }
  load_attempted_ = true;
  if (!options.library.empty() && !options.library.is_absolute()) {
    status_ = NvcompLoadStatus::invalid_path;
    error_ = "an explicit nvCOMP library path must be absolute";
    return {status_, true};
  }

  const bool explicit_path = !options.library.empty();
  bool opened = explicit_path
                    ? library_.open_absolute(options.library)
#ifdef _WIN32
                    : library_.open_system({"nvcomp64_5.dll"});
#else
                    : library_.open_system({"libnvcomp.so.5", "libnvcomp.so"});
#endif
  if (opened) {
    source_ = explicit_path ? NvcompLibrarySource::explicit_path : NvcompLibrarySource::system;
  } else if (!explicit_path && open_app_local(library_)) {
    opened = true;
    source_ = NvcompLibrarySource::app_local;
  }
  if (!opened) {
    status_ = NvcompLoadStatus::library_unavailable;
    error_ = library_.error();
    return {status_, true};
  }

  loaded_name_ = library_.loaded_name();
  resolve_symbols();
  validate_dispatch();
  validate_version();
  return {status_, true};
}

const NvcompDispatch& NvcompApi::dispatch() const noexcept {
  return dispatch_;
}

NvcompLoadStatus NvcompApi::status() const noexcept {
  return status_;
}

NvcompLibrarySource NvcompApi::library_source() const noexcept {
  return source_;
}

const std::string& NvcompApi::loaded_name() const noexcept {
  return loaded_name_;
}

const std::string& NvcompApi::error() const noexcept {
  return error_;
}

const std::vector<std::string>& NvcompApi::missing_symbols() const noexcept {
  return missing_symbols_;
}

const nvcompProperties_t& NvcompApi::properties() const noexcept {
  return properties_;
}

bool NvcompApi::has_lz4() const noexcept {
  return status_ == NvcompLoadStatus::loaded;
}

void NvcompApi::abandon() noexcept {
  library_.abandon();
  dispatch_ = {};
}

void NvcompApi::resolve_symbols() {
  missing_symbols_.clear();
  require(dispatch_.get_properties, "nvcompGetProperties");
  require(dispatch_.get_status_string, "nvcompGetStatusString");
  require(dispatch_.lz4_compress_get_required_alignments,
          "nvcompBatchedLZ4CompressGetRequiredAlignments");
  require(dispatch_.lz4_compress_get_temp_size, "nvcompBatchedLZ4CompressGetTempSizeAsync");
  require(dispatch_.lz4_compress_get_max_output_size,
          "nvcompBatchedLZ4CompressGetMaxOutputChunkSize");
  require(dispatch_.lz4_compress_async, "nvcompBatchedLZ4CompressAsync");
  require(dispatch_.lz4_decompress_get_required_alignments,
          "nvcompBatchedLZ4DecompressGetRequiredAlignments");
  require(dispatch_.lz4_decompress_get_temp_size,
          "nvcompBatchedLZ4DecompressGetTempSizeAsync");
  require(dispatch_.lz4_decompress_async, "nvcompBatchedLZ4DecompressAsync");
}

void NvcompApi::validate_dispatch() {
  missing_symbols_.clear();
  const auto check = [&](const auto function, const char* name) {
    if (function == nullptr) {
      missing_symbols_.emplace_back(name);
    }
  };
  check(dispatch_.get_properties, "nvcompGetProperties");
  check(dispatch_.get_status_string, "nvcompGetStatusString");
  check(dispatch_.lz4_compress_get_required_alignments,
        "nvcompBatchedLZ4CompressGetRequiredAlignments");
  check(dispatch_.lz4_compress_get_temp_size, "nvcompBatchedLZ4CompressGetTempSizeAsync");
  check(dispatch_.lz4_compress_get_max_output_size,
        "nvcompBatchedLZ4CompressGetMaxOutputChunkSize");
  check(dispatch_.lz4_compress_async, "nvcompBatchedLZ4CompressAsync");
  check(dispatch_.lz4_decompress_get_required_alignments,
        "nvcompBatchedLZ4DecompressGetRequiredAlignments");
  check(dispatch_.lz4_decompress_get_temp_size,
        "nvcompBatchedLZ4DecompressGetTempSizeAsync");
  check(dispatch_.lz4_decompress_async, "nvcompBatchedLZ4DecompressAsync");
  if (!missing_symbols_.empty()) {
    status_ = NvcompLoadStatus::symbols_missing;
    error_ = "nvCOMP is missing required LZ4 batch symbols";
  } else {
    status_ = NvcompLoadStatus::loaded;
    error_.clear();
  }
}

void NvcompApi::validate_version() {
  if (status_ != NvcompLoadStatus::loaded) {
    return;
  }
  properties_ = {};
  const nvcompStatus_t result = dispatch_.get_properties(&properties_);
  if (result != nvcompSuccess || properties_.version < 5300U || properties_.version >= 6000U) {
    status_ = NvcompLoadStatus::incompatible_version;
    error_ = "xVRAM requires nvCOMP 5.3 or newer within major version 5";
  }
}

const char* nvcomp_load_status_name(const NvcompLoadStatus status) noexcept {
  switch (status) {
  case NvcompLoadStatus::loaded:
    return "loaded";
  case NvcompLoadStatus::library_unavailable:
    return "library_unavailable";
  case NvcompLoadStatus::symbols_missing:
    return "symbols_missing";
  case NvcompLoadStatus::incompatible_version:
    return "incompatible_version";
  case NvcompLoadStatus::invalid_path:
    return "invalid_path";
  }
  return "library_unavailable";
}

const char* nvcomp_library_source_name(const NvcompLibrarySource source) noexcept {
  switch (source) {
  case NvcompLibrarySource::unknown:
    return "unknown";
  case NvcompLibrarySource::system:
    return "system";
  case NvcompLibrarySource::app_local:
    return "app_local";
  case NvcompLibrarySource::explicit_path:
    return "explicit_path";
  case NvcompLibrarySource::injected:
    return "injected";
  }
  return "unknown";
}

} // namespace xvram::nvcomp
