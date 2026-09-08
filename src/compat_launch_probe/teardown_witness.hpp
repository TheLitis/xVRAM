#pragma once
#include <cupti.h>
#include <cstdint>

namespace xvram::launch_probe::teardown_witness {
struct Snapshot {
  bool enabled=false, healthy=false, checkpointed=false;
  std::uint64_t errors=0, sequence=0, pending=0, qpc_frequency=0;
  std::uint64_t libraries_live=0, primary_references=0;
  std::uint64_t library_loads=0, library_unloads=0, primary_retains=0, primary_releases=0;
  std::uint64_t lifecycle_pairs=0, after_checkpoint_pairs=0;
};
void initialize();
bool enabled() noexcept;
bool healthy() noexcept;
void api(const CUpti_CallbackData&, std::uint64_t api_id) noexcept;
// Execution checkpoint only, never terminal closure. atexit calls this too.
void checkpoint() noexcept;
Snapshot snapshot() noexcept;
}
