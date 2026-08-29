#pragma once

namespace xvram::probe {

struct ProbeReport;

void apply_identifier_policy(ProbeReport& report, bool include_stable_identifiers);

} // namespace xvram::probe
