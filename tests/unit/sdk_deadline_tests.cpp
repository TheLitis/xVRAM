#include "sdk/deadline.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>

namespace {

int failures = 0;

#define CHECK(expression)                                                                          \
  do {                                                                                             \
    if (!(expression)) {                                                                           \
      std::cerr << __FILE__ << ':' << __LINE__ << ": check failed: " #expression << '\n';          \
      ++failures;                                                                                  \
    }                                                                                              \
  } while (false)

using Deadline = xvram::sdk::SubmissionDeadline;

void zero_deadline_test() {
  Deadline deadline;
  const Deadline::TimePoint submitted{std::chrono::milliseconds(123)};
  CHECK(Deadline::create(0, submitted, deadline));
  CHECK(!deadline.has_deadline());
  CHECK(!deadline.expired(Deadline::TimePoint::max()));
}

void overflow_test() {
  Deadline deadline;
  CHECK(!Deadline::create(std::numeric_limits<std::uint64_t>::max(), Deadline::Clock::now(),
                          deadline));
}

void queued_expiry_test() {
  const Deadline::TimePoint submitted{std::chrono::milliseconds(100)};
  Deadline deadline;
  CHECK(Deadline::create(5, submitted, deadline));
  CHECK(!deadline.expired(submitted + std::chrono::milliseconds(4)));
  CHECK(deadline.expired(submitted + std::chrono::milliseconds(5)));

  const xvram::sdk::Error error = deadline.timeout_error("operation", "queue_deadline");
  CHECK(error.status == XVRAM_STATUS_TIMEOUT);
  CHECK(error.stage == "operation");
  CHECK(error.operation == "queue_deadline");
}

void between_unit_expiry_test() {
  const Deadline::TimePoint submitted{std::chrono::milliseconds(200)};
  Deadline deadline;
  CHECK(Deadline::create(10, submitted, deadline));

  std::uint64_t completed_units = 0;
  if (!deadline.expired(submitted + std::chrono::milliseconds(6))) {
    ++completed_units;
  }
  // The first unit retired safely while the deadline elapsed. The next launch is rejected at the
  // boundary, and completed progress is not rolled back.
  const bool second_launch_allowed = !deadline.expired(submitted + std::chrono::milliseconds(10));
  if (second_launch_allowed) {
    ++completed_units;
  }
  CHECK(!second_launch_allowed);
  CHECK(completed_units == 1U);
  CHECK(deadline.timeout_error("gemm", "tile_deadline").status == XVRAM_STATUS_TIMEOUT);
}

} // namespace

int main() {
  zero_deadline_test();
  overflow_test();
  queued_expiry_test();
  between_unit_expiry_test();
  if (failures != 0) {
    std::cerr << failures << " SDK deadline test(s) failed\n";
    return 1;
  }
  return 0;
}
