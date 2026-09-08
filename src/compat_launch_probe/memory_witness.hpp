#pragma once
#include "memory_registry.hpp"
#include <cupti.h>

namespace xvram::launch_probe::memory_witness {
void initialize();
bool enabled() noexcept;
bool healthy() noexcept;
void api(const CUpti_CallbackData& data, std::uint64_t api_id) noexcept;
// Pure observer snapshot: neither launches work nor changes residency. Unknown
// state (including an overlapping memory API) returns known=false. A one-byte
// lookup is NOT proof of a later tensor access spanning more than one byte.
Resolution resolve(std::uint64_t address, std::uint64_t bytes, int device=0,
                   CUcontext context=nullptr, std::uint64_t alignment=16) noexcept;
// Called outside callbacks/loader lock after callback quiescence. This records
// memory-observer completion only, never GPU retirement or tensor proof.
void finish();
}
