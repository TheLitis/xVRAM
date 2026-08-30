#pragma once

#include "xvram/xvram.h"

#include <cstdint>
#include <string>
#include <string_view>

namespace xvram::sdk {

struct Error {
  xvram_status status = XVRAM_STATUS_SUCCESS;
  xvram_native_error_domain native_domain = XVRAM_NATIVE_ERROR_NONE;
  std::int64_t native_code = 0;
  std::string stage;
  std::string operation;
  std::string native_name;
  std::string message;

  [[nodiscard]] explicit operator bool() const noexcept {
    return status != XVRAM_STATUS_SUCCESS;
  }
};

[[nodiscard]] Error make_error(xvram_status status, std::string_view stage,
                               std::string_view operation, std::string_view message);
[[nodiscard]] Error make_native_error(xvram_status status, xvram_native_error_domain native_domain,
                                      std::int64_t native_code, std::string_view stage,
                                      std::string_view operation, std::string_view native_name,
                                      std::string_view message);
[[nodiscard]] Error exception_error(std::string_view operation) noexcept;
[[nodiscard]] const char* status_name(xvram_status status) noexcept;

void write_error_info(const Error& error, xvram_error_info_v1& output) noexcept;

} // namespace xvram::sdk
