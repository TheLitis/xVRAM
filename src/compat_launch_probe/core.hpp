#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace xvram::launch_probe {
inline constexpr std::size_t max_parameters = 256;
inline constexpr std::size_t max_bank_bytes = 65536;

struct Parameter {
  std::size_t offset = 0;
  std::size_t bytes = 0;
};
struct Snapshot {
  std::string name;
  std::vector<Parameter> parameters;
};

// Context-less kernel resolution is explicit, not an invalid-handle fallback.
// The resolved function is used only for metadata, never to replace launch f.
template<class Q> auto resolve_contextless(Q& query) {
  const auto context = query.current();
  if (!context || query.stream_context(context) != context)
    throw std::runtime_error("kernel_context_mismatch");
  const auto function = query.function();
  if (!function) throw std::runtime_error("kernel_function_missing");
  return function;
}

// An independently queried library/module edge must agree with cuFuncGetModule.
// It authenticates neither CUPTI module IDs nor the bytes of a cubin.
template<class Q, class Module> auto resolve_library(Q& query, Module expected) {
  if (!expected) throw std::runtime_error("witness_function_module_missing");
  const auto library = query.library();
  if (!library || query.module(library) != expected)
    throw std::runtime_error("witness_module_disagreement");
  return library;
}

// Metadata is a new snapshot on EVERY call, never a cache keyed by a reusable
// native pointer. No borrowed parameter/name storage escapes this operation.
// Q is injected by tests; the native implementation calls only documented APIs.
template<class Q> Snapshot inspect(Q& query) {
  if (!query.resolve()) throw std::runtime_error("function_resolution_failed");
  const auto name = query.name();
  if (name.empty() || name.size() > 4096) throw std::runtime_error("function_name_limit");
  const auto count = query.count();
  if (count > max_parameters) throw std::runtime_error("parameter_count_limit");
  Snapshot result{name, {}};
  result.parameters.reserve(count);
  std::size_t end = 0;
  for (std::size_t i = 0; i < count; ++i) {
    const auto parameter = query.parameter(i);
    if (parameter.bytes == 0 || parameter.offset < end ||
        parameter.offset > max_bank_bytes || parameter.bytes > max_bank_bytes - parameter.offset)
      throw std::runtime_error("invalid_parameter_layout");
    end = parameter.offset + parameter.bytes;
    result.parameters.push_back(parameter);
  }
  return result;
}

// Admission and all metadata writes must succeed BEFORE the one native forward.
// Retirement here means host API return, NOT GPU event completion.
template<class Inspect, class Begin, class Forward, class End>
int invoke(Inspect inspect_call, Begin begin, Forward forward, End end) {
  const auto snapshot = inspect_call();
  begin(snapshot);
  const int result = forward();
  end(result);
  return result;
}
} // namespace xvram::launch_probe
