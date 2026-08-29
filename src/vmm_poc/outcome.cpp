#include "vmm_poc/outcome.hpp"

#include "vmm_poc/executor.hpp"

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>

namespace xvram::vmm_poc {
namespace {

constexpr int exit_prerequisite = 23;
constexpr int exit_corruption = 24;
constexpr int exit_oom = 25;
constexpr int exit_timeout = 26;

[[nodiscard]] std::string lowercase(std::string_view value) {
  std::string output(value);
  std::transform(output.begin(), output.end(), output.begin(), [](const unsigned char character) {
    return static_cast<char>(std::tolower(character));
  });
  return output;
}

[[nodiscard]] bool contains(const std::string_view value, const std::string_view needle) {
  return value.find(needle) != std::string_view::npos;
}

} // namespace

std::string normalize_outcome_reason(const ExecutorResult& execution) {
  if (execution.exit_code == exit_corruption) {
    return "data_mismatch";
  }
  if (execution.exit_code == exit_timeout) {
    return "timeout";
  }

  const std::string source = lowercase(execution.reason.value_or(""));
  const std::string stage =
      execution.failure.has_value() ? lowercase(execution.failure->stage) : std::string{};
  const std::string operation =
      execution.failure.has_value() ? lowercase(execution.failure->operation) : std::string{};

  if (contains(source, "device_ordinal") || contains(source, "driver") ||
      contains(source, "initialization")) {
    return "device_unavailable";
  }
  if (contains(source, "vmm") || contains(source, "uva")) {
    return "vmm_unsupported";
  }
  if (source == "insufficient_host_memory") {
    return "insufficient_host_memory";
  }
  if (source == "insufficient_device_budget" || contains(source, "budget")) {
    return "insufficient_device_budget";
  }
  if (execution.exit_code == exit_prerequisite) {
    return "invalid_configuration";
  }
  if (execution.exit_code == exit_oom) {
    if (stage == "budget" || contains(operation, "budget") || contains(operation, "wddm")) {
      return "insufficient_device_budget";
    }
    if (stage == "host_backing" || contains(operation, "host") ||
        contains(operation, "pageable_backing")) {
      return "host_oom";
    }
    return "device_oom";
  }
  if (execution.cleanup.complete.has_value() && !*execution.cleanup.complete) {
    return "cleanup_error";
  }
  if (execution.failure.has_value() && execution.failure->native_code.has_value()) {
    return "cuda_error";
  }
  return "internal_error";
}

std::string normalize_outcome_stage(const ExecutionFailure& failure) {
  const std::string stage = lowercase(failure.stage);
  const std::string operation = lowercase(failure.operation);

  if (stage == "configuration" || stage == "preflight" || stage == "planning" ||
      stage == "budget") {
    return "planning";
  }
  if (stage == "cleanup") {
    return "cleanup";
  }
  if (stage == "mapping") {
    return contains(operation, "unmap") ? "unmapping" : "mapping";
  }
  if (stage == "verification" || stage == "proof") {
    return "verification";
  }
  if (contains(operation, "host") || contains(operation, "pageable") ||
      contains(operation, "backing")) {
    return "host_backing";
  }
  if (stage == "safety" || stage == "measurement") {
    return "kernel";
  }
  if (stage == "execution") {
    if (contains(operation, "htod") || contains(operation, "h2d")) {
      return "transfer_h2d";
    }
    if (contains(operation, "dtoh") || contains(operation, "d2h")) {
      return "transfer_d2h";
    }
    return "kernel";
  }
  if (stage == "setup") {
    if (contains(operation, "memcreate")) {
      return "physical_allocation";
    }
    return "reserve";
  }
  return "watchdog";
}

} // namespace xvram::vmm_poc
