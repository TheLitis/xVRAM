#pragma once

#include "xvram/probe/report.hpp"

#include <vector>

namespace xvram::nvml {

void enrich(probe::CudaReport& report, std::vector<probe::Diagnostic>& diagnostics);

} // namespace xvram::nvml
