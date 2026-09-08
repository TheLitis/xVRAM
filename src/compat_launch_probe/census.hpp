#pragma once
#include <cstdint>
#include <array>
#include <stdexcept>

namespace xvram::launch_probe::census {
// Only the actual native forward is marked; metadata queries are outside this
// scope. CUPTI callbacks read this TLS integer, never call CUDA or inspect args.
inline thread_local std::uint64_t current_call = 0;
struct ApiFrame {
  std::uint64_t id = 0, parent = 0, marker = 0;
  std::uint32_t domain = 0, callback = 0, correlation = 0;
};
class ApiStack {
public:
  ApiFrame enter(std::uint64_t id, std::uint64_t marker, std::uint32_t domain,
                 std::uint32_t callback, std::uint32_t correlation) {
    if (!id || !correlation || depth_ == frames_.size()) throw std::runtime_error("census_api_depth_or_id");
    const ApiFrame frame{id, depth_ ? frames_[depth_-1].id : 0, marker, domain, callback, correlation};
    frames_[depth_++] = frame;
    return frame;
  }
  ApiFrame exit(std::uint64_t marker, std::uint32_t domain, std::uint32_t callback,
                std::uint32_t correlation) {
    if (!depth_) throw std::runtime_error("census_orphan_exit");
    const auto frame = frames_[depth_-1];
    if (frame.domain != domain || frame.callback != callback || frame.correlation != correlation || frame.marker != marker)
      throw std::runtime_error("census_api_exit_order");
    --depth_;
    return frame;
  }
private:
  std::array<ApiFrame, 64> frames_{};
  std::size_t depth_ = 0;
};
class Scope {
public:
  explicit Scope(std::uint64_t id) {
    if (id == 0 || current_call != 0) throw std::logic_error("census_scope_overlap");
    current_call = id;
  }
  ~Scope() { current_call = 0; }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;
};
#ifdef XVRAM_PROBE_CENSUS
void initialize();
void stop_flusher();
bool drain(bool gpu_drained);
#endif
} // namespace xvram::launch_probe::census
