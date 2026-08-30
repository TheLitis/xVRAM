#include "sdk/status.hpp"

#include <algorithm>
#include <cstring>
#include <exception>
#include <new>

namespace xvram::sdk {
namespace {

template <std::size_t Size>
void copy_text(char (&destination)[Size], const std::string& source) noexcept {
  static_assert(Size > 0);
  const std::size_t count = std::min(source.size(), Size - 1U);
  if (count != 0U) {
    std::memcpy(destination, source.data(), count);
  }
  destination[count] = '\0';
  if (count + 1U < Size) {
    std::memset(destination + count + 1U, 0, Size - count - 1U);
  }
}

} // namespace

Error make_error(const xvram_status status, const std::string_view stage,
                 const std::string_view operation, const std::string_view message) {
  Error error;
  error.status = status;
  error.stage.assign(stage);
  error.operation.assign(operation);
  error.message.assign(message);
  return error;
}

Error make_native_error(const xvram_status status, const xvram_native_error_domain native_domain,
                        const std::int64_t native_code, const std::string_view stage,
                        const std::string_view operation, const std::string_view native_name,
                        const std::string_view message) {
  Error error = make_error(status, stage, operation, message);
  error.native_domain = native_domain;
  error.native_code = native_code;
  error.native_name.assign(native_name);
  return error;
}

Error exception_error(const std::string_view operation) noexcept {
  try {
    throw;
  } catch (const std::bad_alloc&) {
    Error error;
    error.status = XVRAM_STATUS_HOST_OUT_OF_MEMORY;
    try {
      error.stage = "sdk";
      error.operation.assign(operation);
      error.message = "host allocation failed";
    } catch (...) {
    }
    return error;
  } catch (const std::exception& exception) {
    Error error;
    error.status = XVRAM_STATUS_INTERNAL;
    try {
      error.stage = "sdk";
      error.operation.assign(operation);
      error.message = exception.what();
    } catch (...) {
    }
    return error;
  } catch (...) {
    Error error;
    error.status = XVRAM_STATUS_INTERNAL;
    try {
      error.stage = "sdk";
      error.operation.assign(operation);
      error.message = "unknown C++ exception at ABI boundary";
    } catch (...) {
    }
    return error;
  }
}

const char* status_name(const xvram_status status) noexcept {
  switch (status) {
  case XVRAM_STATUS_SUCCESS:
    return "success";
  case XVRAM_STATUS_INVALID_ARGUMENT:
    return "invalid_argument";
  case XVRAM_STATUS_INCOMPATIBLE_ABI:
    return "incompatible_abi";
  case XVRAM_STATUS_UNSUPPORTED:
    return "unsupported";
  case XVRAM_STATUS_UNAVAILABLE:
    return "unavailable";
  case XVRAM_STATUS_HOST_OUT_OF_MEMORY:
    return "host_out_of_memory";
  case XVRAM_STATUS_DEVICE_OUT_OF_MEMORY:
    return "device_out_of_memory";
  case XVRAM_STATUS_BUDGET_PRESSURE:
    return "budget_pressure";
  case XVRAM_STATUS_BUSY:
    return "busy";
  case XVRAM_STATUS_NOT_READY:
    return "not_ready";
  case XVRAM_STATUS_TIMEOUT:
    return "timeout";
  case XVRAM_STATUS_CORRUPTION:
    return "corruption";
  case XVRAM_STATUS_CALLBACK_FAILED:
    return "callback_failed";
  case XVRAM_STATUS_CUDA_ERROR:
    return "cuda_error";
  case XVRAM_STATUS_CUBLAS_ERROR:
    return "cublas_error";
  case XVRAM_STATUS_POISONED:
    return "poisoned";
  case XVRAM_STATUS_CLEANUP_FAILED:
    return "cleanup_failed";
  case XVRAM_STATUS_CLOSED:
    return "closed";
  case XVRAM_STATUS_CANCELLED:
    return "cancelled";
  case XVRAM_STATUS_INTERNAL:
    return "internal";
  default:
    return "unknown";
  }
}

void write_error_info(const Error& error, xvram_error_info_v1& output) noexcept {
  const std::uint32_t caller_size = output.struct_size;
  output = XVRAM_ERROR_INFO_V1_INIT;
  output.struct_size = caller_size;
  output.status = error.status;
  output.native_domain = error.native_domain;
  output.native_code = error.native_code;
  copy_text(output.stage, error.stage);
  copy_text(output.operation, error.operation);
  copy_text(output.native_name, error.native_name);
  copy_text(output.message, error.message);
}

} // namespace xvram::sdk
