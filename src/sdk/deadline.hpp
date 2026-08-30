#pragma once

#include "sdk/status.hpp"

#include <chrono>
#include <cstdint>
#include <limits>

namespace xvram::sdk {

// A descriptor deadline is relative to the instant at which an operation is submitted. Keeping
// the absolute time point in one small value object makes queue and between-unit checks use
// exactly the same boundary (now >= deadline) and keeps duration arithmetic overflow-safe.
class SubmissionDeadline {
public:
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;

  [[nodiscard]] static bool create(const std::uint64_t deadline_ms, const TimePoint submitted_at,
                                   SubmissionDeadline& output) noexcept {
    output = SubmissionDeadline{};
    output.submitted_at_ = submitted_at;
    if (deadline_ms == 0) {
      return true;
    }

    const Clock::duration remaining = TimePoint::max() - submitted_at;
    const auto remaining_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(remaining).count();
    if (remaining_ms < 0 || deadline_ms > static_cast<std::uint64_t>(remaining_ms)) {
      return false;
    }

    const auto milliseconds =
        std::chrono::milliseconds(static_cast<std::chrono::milliseconds::rep>(deadline_ms));
    output.deadline_at_ = submitted_at + std::chrono::duration_cast<Clock::duration>(milliseconds);
    output.has_deadline_ = true;
    return true;
  }

  [[nodiscard]] bool has_deadline() const noexcept {
    return has_deadline_;
  }

  [[nodiscard]] bool expired(const TimePoint now = Clock::now()) const noexcept {
    return has_deadline_ && now >= deadline_at_;
  }

  [[nodiscard]] TimePoint submitted_at() const noexcept {
    return submitted_at_;
  }

  [[nodiscard]] Error timeout_error(const char* stage, const char* operation) const {
    return make_error(XVRAM_STATUS_TIMEOUT, stage, operation,
                      "operation missed its submission-relative deadline before a safe launch "
                      "boundary");
  }

private:
  TimePoint submitted_at_{};
  TimePoint deadline_at_{};
  bool has_deadline_ = false;
};

[[nodiscard]] inline Error validate_submission_deadline(const std::uint64_t deadline_ms,
                                                        const char* operation) {
  SubmissionDeadline ignored;
  if (!SubmissionDeadline::create(deadline_ms, SubmissionDeadline::Clock::now(), ignored)) {
    return make_error(XVRAM_STATUS_INVALID_ARGUMENT, "deadline", operation,
                      "deadline_ms cannot be represented by the monotonic clock");
  }
  return {};
}

} // namespace xvram::sdk
