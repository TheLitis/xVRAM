#include <cstdint>
#include <string_view>
#include <type_traits>

#include <xvram/xvram.h>

static_assert(std::is_standard_layout_v<xvram_api_v1>);
static_assert(std::is_standard_layout_v<xvram_session_config_v1>);
static_assert(sizeof(void*) == 8U);

int main() {
  xvram_api_v1 api{};
  api.struct_size = static_cast<std::uint32_t>(sizeof(api));
  const auto status =
      xvram_get_api(XVRAM_ABI_VERSION_1, static_cast<std::uint32_t>(sizeof(api)), &api);
  if (status != XVRAM_STATUS_SUCCESS || api.abi_version != XVRAM_ABI_VERSION_1) {
    return 1;
  }
  if (api.status_name == nullptr || api.gemm_execute == nullptr || api.operation_wait == nullptr ||
      api.session_close == nullptr) {
    return 2;
  }
  return std::string_view(api.status_name(XVRAM_STATUS_NOT_READY)) == "not_ready" ? 0 : 3;
}
