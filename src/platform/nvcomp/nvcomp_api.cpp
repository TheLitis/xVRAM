#include "platform/nvcomp/nvcomp_api.hpp"

#include "platform/sha256.hpp"

#include <array>
#include <cctype>
#include <filesystem>
#include <string_view>
#include <system_error>
#include <utility>

#include <nvcomp/version.h>

#ifndef XVRAM_NVCOMP_LIBRARY_SHA256
#error "XVRAM_NVCOMP_LIBRARY_SHA256 must pin the staged nvCOMP GPU shared library"
#endif

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

constexpr std::string_view pinned_library_sha256{XVRAM_NVCOMP_LIBRARY_SHA256};
constexpr std::string_view pinned_package_version{"5.3.0.16"};

[[nodiscard]] bool is_sha256(const std::string_view value) noexcept {
  if (value.size() != 64U) {
    return false;
  }
  for (const char character : value) {
    if (!std::isxdigit(static_cast<unsigned char>(character))) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] std::string lowercase(std::string value) {
  for (char& character : value) {
    character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
  }
  return value;
}

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

[[nodiscard]] bool find_app_local(std::filesystem::path& loaded_path) {
  std::filesystem::path directory = module_directory();
  if (directory.empty()) {
    return false;
  }
  if (directory.is_relative()) {
    std::error_code error;
    directory = std::filesystem::absolute(directory, error);
    if (error) {
      return false;
    }
  }
#ifdef _WIN32
  constexpr std::array names{L"nvcomp64_5.dll"};
  const std::array directories{directory};
#else
  constexpr std::array names{"libnvcomp.so.5", "libnvcomp.so"};
  // Executables are installed in bin while shared libraries live in the sibling lib directory.
  // Shared xVRAM libraries continue to find nvCOMP directly beside themselves.
  const std::array directories{directory, directory.parent_path() / "lib"};
#endif
  for (const std::filesystem::path& search_directory : directories) {
    for (const auto* name : names) {
      const std::filesystem::path candidate = search_directory / name;
      std::error_code error;
      if (std::filesystem::is_regular_file(candidate, error) && !error) {
        loaded_path = candidate;
        return true;
      }
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
  std::filesystem::path loaded_path = options.library;
  if (!explicit_path && !find_app_local(loaded_path)) {
    status_ = NvcompLoadStatus::library_unavailable;
    error_ = "app-local nvCOMP runtime was not found beside xVRAM";
    return {status_, true};
  }

  // Verify before invoking the OS loader so a modified binary cannot run its module initializer,
  // then verify the same path again after loading so telemetry describes the file actually chosen
  // by the app-local resolver.
  if (!validate_integrity(loaded_path, options)) {
    return {status_, true};
  }
  const bool opened = library_.open_absolute(loaded_path);
  if (opened) {
    source_ = explicit_path ? NvcompLibrarySource::explicit_path : NvcompLibrarySource::app_local;
  }
  if (!opened) {
    integrity_verified_ = false;
    library_sha256_.clear();
    status_ = NvcompLoadStatus::library_unavailable;
    error_ = library_.error();
    if (error_.empty()) {
      error_ = "app-local nvCOMP runtime was not found beside xVRAM";
    }
    return {status_, true};
  }

  loaded_name_ = library_.loaded_name();
  if (!validate_integrity(loaded_path, options)) {
    library_.close();
    return {status_, true};
  }
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

const std::string& NvcompApi::library_sha256() const noexcept {
  return library_sha256_;
}

const std::string& NvcompApi::version_string() const noexcept {
  return version_string_;
}

bool NvcompApi::integrity_verified() const noexcept {
  return integrity_verified_;
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
  require(dispatch_.lz4_decompress_get_temp_size, "nvcompBatchedLZ4DecompressGetTempSizeAsync");
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
  check(dispatch_.lz4_decompress_get_temp_size, "nvcompBatchedLZ4DecompressGetTempSizeAsync");
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
  if (result != nvcompSuccess || properties_.version != NVCOMP_VER) {
    status_ = NvcompLoadStatus::incompatible_version;
    error_ = "xVRAM requires exact nvCOMP semantic version 5.3.0";
    version_string_.clear();
    return;
  }
  version_string_ =
      source_ == NvcompLibrarySource::injected ? "5.3.0" : std::string(pinned_package_version);
}

bool NvcompApi::validate_integrity(const std::filesystem::path& loaded_path,
                                   const NvcompLoadOptions& options) {
  integrity_verified_ = false;
  const std::string expected =
      lowercase(options.expected_library_sha256.value_or(std::string(pinned_library_sha256)));
  if (!is_sha256(expected)) {
    status_ = NvcompLoadStatus::integrity_failure;
    error_ = "the configured nvCOMP shared-library SHA-256 pin is malformed";
    return false;
  }

  const platform::FileSha256Result actual = platform::sha256_file(loaded_path);
  if (!actual) {
    status_ = NvcompLoadStatus::integrity_failure;
    error_ = "nvCOMP shared-library SHA-256 verification failed: " + actual.error;
    return false;
  }
  library_sha256_ = actual.digest;
  if (library_sha256_ != expected) {
    status_ = NvcompLoadStatus::integrity_failure;
    error_ = "the loaded nvCOMP shared library does not match its SHA-256 pin";
    return false;
  }
  integrity_verified_ = true;
  return true;
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
  case NvcompLoadStatus::integrity_failure:
    return "integrity_failure";
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
