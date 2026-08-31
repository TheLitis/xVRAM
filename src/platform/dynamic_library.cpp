#include "platform/dynamic_library.hpp"

#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>

#else
#include <dlfcn.h>
#endif

namespace xvram::platform {
namespace {

#ifdef _WIN32
[[nodiscard]] std::string windows_error_message(const DWORD error) {
  if (error == ERROR_SUCCESS) {
    return {};
  }

  wchar_t* buffer = nullptr;
  const DWORD length = FormatMessageW(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
      nullptr, error, 0, reinterpret_cast<wchar_t*>(&buffer), 0, nullptr);
  if (length == 0 || buffer == nullptr) {
    return "Windows error " + std::to_string(error);
  }

  int utf8_length = WideCharToMultiByte(CP_UTF8, 0, buffer, static_cast<int>(length), nullptr, 0,
                                        nullptr, nullptr);
  std::string result;
  if (utf8_length > 0) {
    result.resize(static_cast<std::size_t>(utf8_length));
    WideCharToMultiByte(CP_UTF8, 0, buffer, static_cast<int>(length), result.data(), utf8_length,
                        nullptr, nullptr);
  }
  LocalFree(buffer);
  while (!result.empty() &&
         (result.back() == '\r' || result.back() == '\n' || result.back() == ' ')) {
    result.pop_back();
  }
  return result;
}

#endif

} // namespace

DynamicLibrary::~DynamicLibrary() {
  close();
}

DynamicLibrary::DynamicLibrary(DynamicLibrary&& other) noexcept
    : handle_(std::exchange(other.handle_, nullptr)), loaded_name_(std::move(other.loaded_name_)),
      error_(std::move(other.error_)) {}

DynamicLibrary& DynamicLibrary::operator=(DynamicLibrary&& other) noexcept {
  if (this != &other) {
    close();
    handle_ = std::exchange(other.handle_, nullptr);
    loaded_name_ = std::move(other.loaded_name_);
    error_ = std::move(other.error_);
  }
  return *this;
}

bool DynamicLibrary::open_system(const std::initializer_list<std::string_view> candidates) {
  close();
  error_.clear();

  for (const std::string_view candidate : candidates) {
#ifdef _WIN32
    const std::wstring wide_candidate(candidate.begin(), candidate.end());
    HMODULE module = LoadLibraryExW(wide_candidate.c_str(), nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (module != nullptr) {
      handle_ = module;
      loaded_name_ = std::string(candidate);
      error_.clear();
      return true;
    }
    error_ = std::string(candidate) + ": " + windows_error_message(GetLastError());
#else
    void* module = dlopen(std::string(candidate).c_str(), RTLD_NOW | RTLD_LOCAL);
    if (module != nullptr) {
      handle_ = module;
      loaded_name_ = std::string(candidate);
      error_.clear();
      return true;
    }
    if (const char* message = dlerror(); message != nullptr) {
      error_ = std::string(candidate) + ": " + message;
    }
#endif
  }
  return false;
}

bool DynamicLibrary::open_absolute(const std::filesystem::path& path) {
  close();
  error_.clear();
  if (!path.is_absolute()) {
    error_ = "Dynamic library path is not absolute";
    return false;
  }

#ifdef _WIN32
  HMODULE module = LoadLibraryExW(path.c_str(), nullptr,
                                  LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
  if (module == nullptr) {
    error_ = path.string() + ": " + windows_error_message(GetLastError());
    return false;
  }
#else
  void* module = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (module == nullptr) {
    const char* message = dlerror();
    error_ = path.string() + ": " + (message != nullptr ? message : "dlopen failed");
    return false;
  }
#endif

  handle_ = module;
  loaded_name_ = path.filename().string();
  return true;
}

void DynamicLibrary::close() noexcept {
  if (handle_ == nullptr) {
    return;
  }
#ifdef _WIN32
  FreeLibrary(static_cast<HMODULE>(handle_));
#else
  dlclose(handle_);
#endif
  handle_ = nullptr;
  loaded_name_.clear();
}

void DynamicLibrary::abandon() noexcept {
  handle_ = nullptr;
  loaded_name_.clear();
}

void* DynamicLibrary::symbol_address(const char* const name) const noexcept {
  if (handle_ == nullptr || name == nullptr) {
    return nullptr;
  }
#ifdef _WIN32
  return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle_), name));
#else
  return dlsym(handle_, name);
#endif
}

} // namespace xvram::platform
