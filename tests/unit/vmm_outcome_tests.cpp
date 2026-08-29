#include "vmm_poc/executor.hpp"
#include "vmm_poc/outcome.hpp"

#include <iostream>
#include <string>
#include <utility>

namespace {

int failures = 0;

void check(const bool condition, const char* expression, const int line) {
  if (!condition) {
    std::cerr << "line " << line << ": check failed: " << expression << '\n';
    ++failures;
  }
}

#define CHECK(expression) check(static_cast<bool>(expression), #expression, __LINE__)

[[nodiscard]] xvram::vmm_poc::ExecutorResult
failed_execution(const int exit_code, std::string stage, std::string operation) {
  xvram::vmm_poc::ExecutorResult result;
  result.exit_code = exit_code;
  result.failure = xvram::vmm_poc::ExecutionFailure{
      std::move(stage), std::move(operation), "fixture failure", {}, {}, {}, {}, {}};
  return result;
}

void outcome_reason_tests() {
  using xvram::vmm_poc::normalize_outcome_reason;

  CHECK(normalize_outcome_reason(failed_execution(25, "setup", "allocate_pageable_backing")) ==
        "host_oom");
  CHECK(normalize_outcome_reason(failed_execution(25, "setup", "cuMemHostAlloc")) == "host_oom");
  CHECK(normalize_outcome_reason(failed_execution(25, "host_backing", "VirtualAlloc")) ==
        "host_oom");
  CHECK(normalize_outcome_reason(failed_execution(25, "budget", "WDDM pressure")) ==
        "insufficient_device_budget");
  CHECK(normalize_outcome_reason(failed_execution(25, "setup", "cuMemCreate")) == "device_oom");

  xvram::vmm_poc::ExecutorResult planned_budget =
      failed_execution(25, "planning", "make_workload_plan");
  planned_budget.reason = "insufficient_device_budget";
  CHECK(normalize_outcome_reason(planned_budget) == "insufficient_device_budget");

  xvram::vmm_poc::ExecutorResult unavailable = failed_execution(23, "preflight", "cuDeviceGet");
  unavailable.reason = "device_ordinal_unavailable";
  CHECK(normalize_outcome_reason(unavailable) == "device_unavailable");

  xvram::vmm_poc::ExecutorResult unsupported =
      failed_execution(23, "preflight", "cuDeviceGetAttribute");
  unsupported.reason = "VMM is unsupported";
  CHECK(normalize_outcome_reason(unsupported) == "vmm_unsupported");

  CHECK(normalize_outcome_reason(failed_execution(24, "verification", "compare_tile")) ==
        "data_mismatch");
  CHECK(normalize_outcome_reason(failed_execution(26, "watchdog", "worker_deadline")) == "timeout");

  xvram::vmm_poc::ExecutorResult cleanup = failed_execution(27, "cleanup", "cuMemUnmap");
  cleanup.cleanup.complete = false;
  CHECK(normalize_outcome_reason(cleanup) == "cleanup_error");

  xvram::vmm_poc::ExecutorResult cuda = failed_execution(27, "kernel", "cuLaunchKernel");
  cuda.failure->native_code = 700;
  CHECK(normalize_outcome_reason(cuda) == "cuda_error");
}

void outcome_stage_tests() {
  using xvram::vmm_poc::normalize_outcome_stage;

  CHECK(normalize_outcome_stage(*failed_execution(25, "setup", "host_allocation").failure) ==
        "host_backing");
  CHECK(normalize_outcome_stage(*failed_execution(25, "execution", "host_allocation").failure) ==
        "host_backing");
  CHECK(normalize_outcome_stage(*failed_execution(25, "setup", "cuMemHostAlloc").failure) ==
        "host_backing");
  CHECK(normalize_outcome_stage(*failed_execution(25, "setup", "cuMemCreate").failure) ==
        "physical_allocation");
  CHECK(normalize_outcome_stage(*failed_execution(27, "mapping", "cuMemUnmap").failure) ==
        "unmapping");
  CHECK(normalize_outcome_stage(*failed_execution(27, "mapping", "cuMemMap").failure) == "mapping");
  CHECK(normalize_outcome_stage(*failed_execution(27, "execution", "cuMemcpyHtoDAsync").failure) ==
        "transfer_h2d");
  CHECK(normalize_outcome_stage(*failed_execution(27, "execution", "cuMemcpyDtoHAsync").failure) ==
        "transfer_d2h");
  CHECK(normalize_outcome_stage(*failed_execution(24, "verification", "compare_tile").failure) ==
        "verification");
  CHECK(normalize_outcome_stage(*failed_execution(27, "cleanup", "cuMemUnmap").failure) ==
        "cleanup");
}

} // namespace

int main() {
  outcome_reason_tests();
  outcome_stage_tests();
  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
    return 1;
  }
  std::cout << "all VMM outcome tests passed\n";
  return 0;
}
