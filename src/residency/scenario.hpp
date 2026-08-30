#pragma once

#include "residency/core.hpp"

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace xvram::residency {

enum class ScenarioKind {
  suite,
  sequential,
  reuse,
  random,
  read_only,
  write_heavy,
  budget_pressure,
};

[[nodiscard]] std::string_view scenario_kind_name(ScenarioKind scenario) noexcept;

enum class ScenarioOperationKind {
  access,
  pressure_acquire,
  pressure_release,
};

struct ScenarioOperation {
  std::uint64_t sequence = 0;
  ScenarioOperationKind kind = ScenarioOperationKind::access;
  std::optional<AccessRange> access;
  std::uint64_t pressure_bytes = 0;
  std::uint32_t pass_index = 0;
  bool warmup = false;
  bool sequential_one_touch = false;
};

struct ScenarioTrace {
  ScenarioKind scenario = ScenarioKind::sequential;
  std::vector<ScenarioOperation> operations;
};

struct ScenarioInput {
  AllocationId allocation_id{1};
  std::uint64_t logical_bytes = 0;
  std::uint64_t chunk_bytes = 0;
  std::uint64_t cache_target_bytes = 0;
  std::optional<std::uint64_t> pressure_bytes;
  std::uint32_t passes = 2;
  std::uint64_t seed = 0x585652414D503032ULL;
};

enum class ScenarioTraceError {
  none,
  invalid_configuration,
  arithmetic_overflow,
  suite_requires_suite_generator,
};

[[nodiscard]] std::string_view scenario_trace_error_name(ScenarioTraceError error) noexcept;

struct ScenarioTraceResult {
  ScenarioTraceError error = ScenarioTraceError::none;
  std::optional<ScenarioTrace> trace;

  [[nodiscard]] explicit operator bool() const noexcept {
    return error == ScenarioTraceError::none && trace.has_value();
  }
};

struct ScenarioSuiteResult {
  ScenarioTraceError error = ScenarioTraceError::none;
  std::vector<ScenarioTrace> traces;

  [[nodiscard]] explicit operator bool() const noexcept {
    return error == ScenarioTraceError::none;
  }
};

[[nodiscard]] ScenarioTraceResult generate_scenario_trace(ScenarioKind scenario,
                                                          const ScenarioInput& input);
[[nodiscard]] ScenarioSuiteResult generate_scenario_suite(const ScenarioInput& input);

} // namespace xvram::residency
