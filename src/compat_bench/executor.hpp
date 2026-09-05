#pragma once
#include "compat_bench/report.hpp"
#include <functional>
#include <optional>
#include <string_view>

namespace xvram::compat_bench {
using ExecutionTrace = std::function<void(std::string_view transition, std::uint64_t operation,
                                          std::optional<std::uint64_t> allocation,
                                          std::uint64_t bytes, std::string_view reason)>;
[[nodiscard]] Report run_executor(const Options& options, const ExecutionTrace& trace = {});
} // namespace xvram::compat_bench
