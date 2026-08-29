#pragma once

#include "xvram/probe/report.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace xvram::platform {

using AdapterLuid = std::array<std::uint8_t, 8>;

[[nodiscard]] std::optional<probe::DxgiAdapterInfo>
query_dxgi_memory(const AdapterLuid& luid, std::uint32_t node_mask,
                  std::vector<probe::Diagnostic>& diagnostics);

} // namespace xvram::platform
