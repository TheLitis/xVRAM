#pragma once
#include <cuda.h>
#include <cstdint>
namespace xvram::launch_probe::sampling {
void initialize();
bool enabled() noexcept;
void before(CUcontext);
void after(std::uint64_t call, CUcontext, CUresult synchronization);
void finish(CUresult synchronization);
}
