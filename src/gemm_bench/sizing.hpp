#pragma once

#include <cstdint>
#include <optional>

namespace xvram::gemm_bench {

struct HostSizingInput {
  std::uint64_t physical_memory_bytes = 0;
  std::uint64_t available_memory_bytes = 0;
  std::uint64_t chunk_bytes = 0;
  std::uint32_t staging_slots = 0;
  std::uint64_t service_reserve_bytes = 256ULL * 1024ULL * 1024ULL;
};

struct AutoGemmShape {
  std::uint64_t m = 0;
  std::uint64_t n = 0;
  std::uint64_t k = 0;
  std::uint64_t logical_bytes = 0;
};

// Leaves max(4 GiB, 25% physical RAM), the declared pinned staging window, and a service
// reserve outside pageable GEMM backing. Arithmetic failure and an exhausted reserve both
// return nullopt.
[[nodiscard]] std::optional<std::uint64_t>
safe_host_logical_limit(const HostSizingInput& input) noexcept;

// Chooses a square M/N problem with K=256 whose three physical matrices fit both the 1.5x VRAM
// target and the safe host limit. The returned logical byte count is always strictly larger than
// VRAM; otherwise automatic oversubscription is unavailable.
[[nodiscard]] std::optional<AutoGemmShape>
choose_auto_gemm_shape(std::uint64_t total_vram_bytes, std::uint64_t element_bytes,
                       std::uint64_t safe_host_limit_bytes) noexcept;

} // namespace xvram::gemm_bench
