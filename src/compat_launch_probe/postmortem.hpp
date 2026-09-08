#pragma once
#include <cstdint>
namespace xvram::launch_probe::postmortem {
// Controller-owned shared counters survive even an abrupt worker death. No
// CUDA calls, native addresses, waits, or argument changes in this observer.
void initialize();
void callback_begin(bool api, bool entering, bool terminal_sync) noexcept;
void callback_end(bool api, bool entering) noexcept;
void activity_begin(bool request) noexcept;
void activity_end(bool completed) noexcept;
void activity_cancel() noexcept;
void error() noexcept;
void seal();
void drained(bool gpu, bool activity) noexcept;
void census_closed() noexcept;
void sampler_disabled(std::uint64_t retained_names) noexcept;
void finish() noexcept;
}
