#pragma once

#include "xvram/probe/report.hpp"

namespace xvram::platform {

[[nodiscard]] probe::SystemInfo collect_system_info();

} // namespace xvram::platform
