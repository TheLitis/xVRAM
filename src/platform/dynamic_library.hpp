#pragma once

#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <string>
#include <string_view>
#include <type_traits>

namespace xvram::platform {

class DynamicLibrary {
public:
  DynamicLibrary() = default;
  ~DynamicLibrary();

  DynamicLibrary(const DynamicLibrary&) = delete;
  DynamicLibrary& operator=(const DynamicLibrary&) = delete;
  DynamicLibrary(DynamicLibrary&& other) noexcept;
  DynamicLibrary& operator=(DynamicLibrary&& other) noexcept;

  [[nodiscard]] bool open_system(std::initializer_list<std::string_view> candidates);
  [[nodiscard]] bool open_absolute(const std::filesystem::path& path);
  void close() noexcept;
  // Deliberately retain the OS module reference until process exit. Quarantine paths use this
  // when a live library-owned GPU handle cannot be synchronized or destroyed safely.
  void abandon() noexcept;

  template <typename Function> [[nodiscard]] Function symbol(const char* name) const noexcept {
    static_assert(std::is_pointer_v<Function> && std::is_trivially_copyable_v<Function>);
    const void* address = symbol_address(name);
    static_assert(sizeof(Function) == sizeof(address));
    Function function = nullptr;
    std::memcpy(&function, &address, sizeof(function));
    return function;
  }

  [[nodiscard]] bool is_open() const noexcept {
    return handle_ != nullptr;
  }
  [[nodiscard]] const std::string& loaded_name() const noexcept {
    return loaded_name_;
  }
  [[nodiscard]] const std::string& error() const noexcept {
    return error_;
  }

private:
  [[nodiscard]] void* symbol_address(const char* name) const noexcept;

  void* handle_ = nullptr;
  std::string loaded_name_;
  std::string error_;
};

} // namespace xvram::platform
