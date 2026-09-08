#pragma once
#include <array>

namespace xvram::launch_probe {
// An Activity flush is not a GPU fence. Only stop collection after the caller
// has separately observed successful device completion and stopped its flusher.
// Try every disable and the final flush even when one cleanup stage fails.
template<class Kind, class Disable, class Flush>
bool retire_activity(bool gpu_drained, bool flusher_stopped,
                     const std::array<Kind, 3>& kinds, Disable&& disable, Flush&& flush) {
  if (!gpu_drained || !flusher_stopped) return false;
  bool success = true;
  for (const auto kind : kinds) {
    if (disable(kind) != 0) success = false;
  }
  if (flush() != 0) success = false;
  return success;
}
}
