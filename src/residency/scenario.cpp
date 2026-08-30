#include "residency/scenario.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <numeric>
#include <utility>

namespace xvram::residency {
namespace {

class SplitMix64 {
public:
  explicit SplitMix64(const std::uint64_t seed) noexcept : state_(seed) {}

  [[nodiscard]] std::uint64_t next() noexcept {
    state_ += 0x9E3779B97F4A7C15ULL;
    std::uint64_t value = state_;
    value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
  }

private:
  std::uint64_t state_;
};

[[nodiscard]] std::optional<std::uint64_t> checked_multiply(const std::uint64_t left,
                                                            const std::uint64_t right) noexcept {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
    return std::nullopt;
  }
  return left * right;
}

struct GeneratorContext {
  const ScenarioInput& input;
  std::uint64_t chunk_count = 0;
  std::uint64_t cache_chunks = 0;
  ScenarioTrace trace;
};

[[nodiscard]] AccessRange make_chunk_access(const GeneratorContext& context,
                                            const std::uint64_t chunk_index,
                                            const AccessMode mode) noexcept {
  const std::uint64_t offset = chunk_index * context.input.chunk_bytes;
  const std::uint64_t length =
      std::min(context.input.chunk_bytes, context.input.logical_bytes - offset);
  return AccessRange{context.input.allocation_id, offset, length, mode};
}

void append_access(GeneratorContext& context, const std::uint64_t chunk_index,
                   const AccessMode mode, const std::uint32_t pass_index, const bool warmup,
                   const bool sequential_one_touch) {
  const std::uint64_t sequence = static_cast<std::uint64_t>(context.trace.operations.size());
  context.trace.operations.push_back(ScenarioOperation{
      sequence, ScenarioOperationKind::access, make_chunk_access(context, chunk_index, mode), 0,
      pass_index, warmup, sequential_one_touch});
}

void append_pressure(GeneratorContext& context, const ScenarioOperationKind kind,
                     const std::uint64_t pressure_bytes) {
  const std::uint64_t sequence = static_cast<std::uint64_t>(context.trace.operations.size());
  context.trace.operations.push_back(
      ScenarioOperation{sequence, kind, std::nullopt, pressure_bytes});
}

void append_alternating_scan(GeneratorContext& context, const AccessMode mode,
                             const std::uint32_t passes, const bool one_touch) {
  for (std::uint32_t pass = 0; pass < passes; ++pass) {
    const bool forward = (pass & 1U) == 0U;
    for (std::uint64_t position = 0; position < context.chunk_count; ++position) {
      const std::uint64_t chunk = forward ? position : context.chunk_count - 1U - position;
      append_access(context, chunk, mode, pass, false, one_touch);
    }
  }
}

[[nodiscard]] ScenarioTraceError reserve_operations(ScenarioTrace& trace,
                                                    const std::uint64_t count) {
  if (count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return ScenarioTraceError::arithmetic_overflow;
  }
  trace.operations.reserve(static_cast<std::size_t>(count));
  return ScenarioTraceError::none;
}

[[nodiscard]] ScenarioTraceError generate_sequential(GeneratorContext& context) {
  const std::optional<std::uint64_t> count =
      checked_multiply(context.chunk_count, static_cast<std::uint64_t>(context.input.passes));
  if (!count.has_value()) {
    return ScenarioTraceError::arithmetic_overflow;
  }
  const ScenarioTraceError reserve_error = reserve_operations(context.trace, *count);
  if (reserve_error != ScenarioTraceError::none) {
    return reserve_error;
  }
  append_alternating_scan(context, AccessMode::read_write, context.input.passes, true);
  return ScenarioTraceError::none;
}

[[nodiscard]] ScenarioTraceError generate_reuse(GeneratorContext& context) {
  constexpr std::uint32_t cycles = 8;
  const std::uint64_t hot_chunks =
      std::max<std::uint64_t>(1, std::min(context.chunk_count, context.cache_chunks / 2U));
  const std::optional<std::uint64_t> count =
      checked_multiply(hot_chunks, static_cast<std::uint64_t>(cycles));
  if (!count.has_value()) {
    return ScenarioTraceError::arithmetic_overflow;
  }
  const ScenarioTraceError reserve_error = reserve_operations(context.trace, *count);
  if (reserve_error != ScenarioTraceError::none) {
    return reserve_error;
  }
  for (std::uint32_t cycle = 0; cycle < cycles; ++cycle) {
    for (std::uint64_t chunk = 0; chunk < hot_chunks; ++chunk) {
      append_access(context, chunk, AccessMode::read, cycle, cycle == 0, false);
    }
  }
  return ScenarioTraceError::none;
}

[[nodiscard]] ScenarioTraceError generate_random(GeneratorContext& context) {
  const std::optional<std::uint64_t> count = checked_multiply(context.chunk_count, 4);
  if (!count.has_value()) {
    return ScenarioTraceError::arithmetic_overflow;
  }
  const ScenarioTraceError reserve_error = reserve_operations(context.trace, *count);
  if (reserve_error != ScenarioTraceError::none) {
    return reserve_error;
  }
  SplitMix64 random(context.input.seed ^ 0x72616E646F6D0002ULL);
  for (std::uint64_t operation = 0; operation < *count; ++operation) {
    const std::uint64_t chunk = random.next() % context.chunk_count;
    const AccessMode mode = (operation & 1ULL) == 0 ? AccessMode::read : AccessMode::read_write;
    append_access(context, chunk, mode, 0, false, false);
  }
  return ScenarioTraceError::none;
}

[[nodiscard]] ScenarioTraceError generate_read_only(GeneratorContext& context) {
  const std::optional<std::uint64_t> count =
      checked_multiply(context.chunk_count, static_cast<std::uint64_t>(context.input.passes));
  if (!count.has_value()) {
    return ScenarioTraceError::arithmetic_overflow;
  }
  const ScenarioTraceError reserve_error = reserve_operations(context.trace, *count);
  if (reserve_error != ScenarioTraceError::none) {
    return reserve_error;
  }
  append_alternating_scan(context, AccessMode::read, context.input.passes, true);
  return ScenarioTraceError::none;
}

[[nodiscard]] ScenarioTraceError generate_write_heavy(GeneratorContext& context) {
  if (context.chunk_count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return ScenarioTraceError::arithmetic_overflow;
  }
  const std::optional<std::uint64_t> count =
      checked_multiply(context.chunk_count, static_cast<std::uint64_t>(context.input.passes));
  if (!count.has_value()) {
    return ScenarioTraceError::arithmetic_overflow;
  }
  const ScenarioTraceError reserve_error = reserve_operations(context.trace, *count);
  if (reserve_error != ScenarioTraceError::none) {
    return reserve_error;
  }

  std::vector<AccessMode> modes(static_cast<std::size_t>(*count), AccessMode::read_write);
  const std::size_t read_count = modes.size() / 5U;
  const std::size_t write_only_count = modes.size() / 5U;
  std::fill_n(modes.begin(), read_count, AccessMode::read);
  std::fill_n(modes.begin() + static_cast<std::ptrdiff_t>(read_count), write_only_count,
              AccessMode::write_only);
  SplitMix64 mode_random(context.input.seed ^ 0x6D6F646568760002ULL);
  for (std::size_t remaining = modes.size(); remaining > 1U; --remaining) {
    const std::size_t selected =
        static_cast<std::size_t>(mode_random.next() % static_cast<std::uint64_t>(remaining));
    std::swap(modes[remaining - 1U], modes[selected]);
  }

  std::vector<std::uint64_t> order(static_cast<std::size_t>(context.chunk_count));
  std::iota(order.begin(), order.end(), std::uint64_t{0});
  SplitMix64 random(context.input.seed ^ 0x7772697465687602ULL);
  std::size_t operation_index = 0;
  for (std::uint32_t pass = 0; pass < context.input.passes; ++pass) {
    for (std::size_t remaining = order.size(); remaining > 1U; --remaining) {
      const std::size_t selected =
          static_cast<std::size_t>(random.next() % static_cast<std::uint64_t>(remaining));
      std::swap(order[remaining - 1U], order[selected]);
    }
    for (const std::uint64_t chunk : order) {
      append_access(context, chunk, modes[operation_index], pass, false, false);
      ++operation_index;
    }
  }
  return ScenarioTraceError::none;
}

[[nodiscard]] ScenarioTraceError generate_budget_pressure(GeneratorContext& context) {
  const std::uint64_t hot_chunks = std::min(context.chunk_count, context.cache_chunks);
  const std::optional<std::uint64_t> access_count = checked_multiply(hot_chunks, 3);
  if (!access_count.has_value() || *access_count > std::numeric_limits<std::uint64_t>::max() - 2) {
    return ScenarioTraceError::arithmetic_overflow;
  }
  const ScenarioTraceError reserve_error = reserve_operations(context.trace, *access_count + 2);
  if (reserve_error != ScenarioTraceError::none) {
    return reserve_error;
  }
  const std::optional<std::uint64_t> default_pressure =
      checked_multiply(context.input.chunk_bytes, 4);
  if (!default_pressure.has_value()) {
    return ScenarioTraceError::arithmetic_overflow;
  }
  const std::uint64_t pressure = context.input.pressure_bytes.value_or(*default_pressure);
  if (pressure == 0) {
    return ScenarioTraceError::invalid_configuration;
  }

  for (std::uint64_t chunk = 0; chunk < hot_chunks; ++chunk) {
    append_access(context, chunk, AccessMode::read, 0, true, false);
  }
  append_pressure(context, ScenarioOperationKind::pressure_acquire, pressure);
  for (std::uint64_t chunk = 0; chunk < hot_chunks; ++chunk) {
    append_access(context, chunk, AccessMode::read, 1, false, false);
  }
  append_pressure(context, ScenarioOperationKind::pressure_release, pressure);
  for (std::uint64_t chunk = 0; chunk < hot_chunks; ++chunk) {
    append_access(context, chunk, AccessMode::read, 2, false, false);
  }
  return ScenarioTraceError::none;
}

} // namespace

std::string_view scenario_kind_name(const ScenarioKind scenario) noexcept {
  switch (scenario) {
  case ScenarioKind::suite:
    return "suite";
  case ScenarioKind::sequential:
    return "sequential";
  case ScenarioKind::reuse:
    return "reuse";
  case ScenarioKind::random:
    return "random";
  case ScenarioKind::read_only:
    return "read-only";
  case ScenarioKind::write_heavy:
    return "write-heavy";
  case ScenarioKind::budget_pressure:
    return "budget-pressure";
  }
  return "invalid";
}

std::string_view scenario_trace_error_name(const ScenarioTraceError error) noexcept {
  switch (error) {
  case ScenarioTraceError::none:
    return "none";
  case ScenarioTraceError::invalid_configuration:
    return "invalid_configuration";
  case ScenarioTraceError::arithmetic_overflow:
    return "arithmetic_overflow";
  case ScenarioTraceError::suite_requires_suite_generator:
    return "suite_requires_suite_generator";
  }
  return "invalid";
}

ScenarioTraceResult generate_scenario_trace(const ScenarioKind scenario,
                                            const ScenarioInput& input) {
  if (scenario == ScenarioKind::suite) {
    return {ScenarioTraceError::suite_requires_suite_generator, std::nullopt};
  }
  const std::optional<std::uint64_t> minimum_cache = checked_multiply(input.chunk_bytes, 2);
  if (!input.allocation_id || input.logical_bytes == 0 || input.chunk_bytes == 0 ||
      !minimum_cache.has_value() || input.cache_target_bytes < *minimum_cache ||
      input.cache_target_bytes % input.chunk_bytes != 0 || input.passes < 2U || input.passes > 8U) {
    return {ScenarioTraceError::invalid_configuration, std::nullopt};
  }

  const std::uint64_t chunk_count = ((input.logical_bytes - 1U) / input.chunk_bytes) + 1U;
  GeneratorContext context{input, chunk_count, input.cache_target_bytes / input.chunk_bytes,
                           ScenarioTrace{scenario, {}}};
  ScenarioTraceError error = ScenarioTraceError::invalid_configuration;
  switch (scenario) {
  case ScenarioKind::sequential:
    error = generate_sequential(context);
    break;
  case ScenarioKind::reuse:
    error = generate_reuse(context);
    break;
  case ScenarioKind::random:
    error = generate_random(context);
    break;
  case ScenarioKind::read_only:
    error = generate_read_only(context);
    break;
  case ScenarioKind::write_heavy:
    error = generate_write_heavy(context);
    break;
  case ScenarioKind::budget_pressure:
    error = generate_budget_pressure(context);
    break;
  case ScenarioKind::suite:
    break;
  }
  if (error != ScenarioTraceError::none) {
    return {error, std::nullopt};
  }
  return {ScenarioTraceError::none, std::move(context.trace)};
}

ScenarioSuiteResult generate_scenario_suite(const ScenarioInput& input) {
  constexpr std::array scenarios{ScenarioKind::sequential, ScenarioKind::reuse,
                                 ScenarioKind::random, ScenarioKind::read_only,
                                 ScenarioKind::write_heavy};
  ScenarioSuiteResult result;
  result.traces.reserve(scenarios.size());
  for (const ScenarioKind scenario : scenarios) {
    ScenarioTraceResult trace = generate_scenario_trace(scenario, input);
    if (!trace) {
      result.error = trace.error;
      result.traces.clear();
      return result;
    }
    result.traces.push_back(std::move(*trace.trace));
  }
  return result;
}

} // namespace xvram::residency
