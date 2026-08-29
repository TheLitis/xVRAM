#include "platform/dxgi_memory.hpp"

namespace xvram::platform {

std::optional<probe::DxgiAdapterInfo> query_dxgi_memory(const AdapterLuid&, const std::uint32_t,
                                                        std::vector<probe::Diagnostic>&) {
  return std::nullopt;
}

} // namespace xvram::platform
