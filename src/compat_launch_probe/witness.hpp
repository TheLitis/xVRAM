#pragma once
#include <cstdint>
#ifdef XVRAM_PROBE_WITNESS
#include <cuda.h>
#include <cupti.h>
namespace xvram::launch_probe::witness {
void initialize();
void api(const CUpti_CallbackData&, std::uint64_t api_id) noexcept;
void launch(std::uint64_t call, CUlibrary library, CUmodule module, bool library_module_equal);
}
#endif
