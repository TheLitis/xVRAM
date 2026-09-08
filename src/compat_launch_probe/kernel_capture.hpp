#pragma once
#include "core.hpp"
#include <cuda.h>
#include <array>

namespace xvram::launch_probe::kernel_capture {
struct Geometry {
  std::array<unsigned,3> grid{}, block{};
  unsigned shared_bytes=0;
};
// Diagnostic host-parameter observation only. Must be called outside CUPTI
// callbacks. The caller queries metadata/context and forwards the original
// argument bank unchanged; no CUDA operation is issued by this component.
void initialize();
bool enabled() noexcept;
bool healthy() noexcept;
void launch(std::uint64_t call, const Snapshot&, const Geometry&,
            const void* parameters, const void* extra, int device, CUcontext);
void returned(std::uint64_t call, int result);
// This is a host-observer checkpoint, NOT a GPU/producer retirement proof.
void finish();
}
