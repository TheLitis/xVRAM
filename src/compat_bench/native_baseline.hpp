#pragma once
#include "compat_bench/options.hpp"
#include "compat_bench/pattern.hpp"
#include <string>
#include <vector>

namespace xvram::compat_bench {
struct NativeBaseline {
  bool completed = false;
  bool cleanup_complete = false;
  double milliseconds = 0;
  std::vector<float> c;
  std::string error;
};
[[nodiscard]] NativeBaseline native_baseline(const Options& options, const Shape& shape);
} // namespace xvram::compat_bench
