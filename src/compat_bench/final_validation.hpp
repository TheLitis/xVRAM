#pragma once
#include <cstdint>
#include <string>
#include <string_view>

namespace xvram::compat_bench {
// Parse the complete document, rejecting duplicate keys, non-finite numbers and excessive
// nesting. Validate the typed final envelope and amend termination only after controller reap.
[[nodiscard]] bool finalize_json(std::string_view input, std::int32_t exit_code, bool pretty,
                                 std::string& output, std::string& error);
} // namespace xvram::compat_bench
