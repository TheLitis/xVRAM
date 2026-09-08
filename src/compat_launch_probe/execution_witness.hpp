#pragma once
#include <cupti.h>
#include <cstdint>
namespace xvram::launch_probe::execution_witness {
inline thread_local bool terminal_tool=false;
void initialize();
bool enabled() noexcept;
void api(const CUpti_CallbackData&, std::uint64_t) noexcept;
void seal();
void finish(bool gpu_drained, bool census_drained);
}
