#pragma once

#include <string>

namespace xvram::vmm_poc {

struct ExecutorResult;
struct ExecutionFailure;

[[nodiscard]] std::string normalize_outcome_reason(const ExecutorResult& execution);
[[nodiscard]] std::string normalize_outcome_stage(const ExecutionFailure& failure);

} // namespace xvram::vmm_poc
