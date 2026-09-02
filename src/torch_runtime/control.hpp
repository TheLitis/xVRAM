#pragma once

#include "xvram/internal/torch_runtime.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace xvram::torch_runtime {

struct PreparedView;

class ViewTicket final {
public:
  ~ViewTicket();

  ViewTicket(const ViewTicket&) = delete;
  ViewTicket& operator=(const ViewTicket&) = delete;
  ViewTicket(ViewTicket&&) = delete;
  ViewTicket& operator=(ViewTicket&&) = delete;

private:
  struct State;
  explicit ViewTicket(std::shared_ptr<State> state) noexcept;

  std::shared_ptr<State> state_;

  friend xvram_torch_runtime_status
  prepare_view(xvram_torch_runtime_session session, xvram_torch_runtime_lease lease,
               xvram_torch_runtime_allocation allocation, std::uint64_t byte_offset,
               std::uint64_t required_bytes, std::uint64_t alignment,
               PreparedView& output) noexcept;
};

struct PreparedView {
  void* address = nullptr;
  std::uint64_t byte_length = 0;
  std::int32_t device_ordinal = -1;
  std::shared_ptr<ViewTicket> ticket;
};

/*
 * Used by the Stable-ABI custom op after tensor metadata has been validated.
 * The returned ticket is the only owner-facing lifetime signal: destroying it
 * decrements the lease's live-view count, but never releases CUDA memory.
 */
[[nodiscard]] xvram_torch_runtime_status
prepare_view(xvram_torch_runtime_session session, xvram_torch_runtime_lease lease,
             xvram_torch_runtime_allocation allocation, std::uint64_t byte_offset,
             std::uint64_t required_bytes, std::uint64_t alignment, PreparedView& output) noexcept;

[[nodiscard]] std::string last_error_message();

} // namespace xvram::torch_runtime
