// No CUDA/CUPTI dependency: exercise the actual shared native observer across
// an owned child-process boundary, with deterministic concurrency handshakes.
#include "compat_launch_probe/postmortem.hpp"
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <string_view>
#include <thread>

namespace observer = xvram::launch_probe::postmortem;
namespace {
void api(bool terminal = false) {
  observer::callback_begin(true, true, terminal);
  observer::callback_end(true, true);
  observer::callback_begin(true, false, terminal);
  observer::callback_end(true, false);
}
void terminal() {
  api(true); // Exact terminal synchronization exemption supplied by caller.
  observer::drained(true, true);
  observer::census_closed();
  observer::finish();
}
int open_at_seal(bool callback_open) {
  std::atomic_bool active{false}, release{false};
  std::thread producer([&] {
    observer::callback_begin(true, true, false);
    if (!callback_open) observer::callback_end(true, true);
    active.store(true); active.notify_one();
    release.wait(false);
    if (callback_open) observer::callback_end(true, true);
    observer::callback_begin(true, false, false);
    observer::callback_end(true, false);
  });
  active.wait(false);
  bool rejected = false;
  try { observer::seal(); } catch (...) { rejected = true; }
  release.store(true); release.notify_one(); producer.join();
  terminal();
  return rejected ? 0 : 70;
}
}

int main(int argc, char** argv) {
  if (argc != 2) return 64;
  const std::string_view scenario{argv[1]};
  try {
    observer::initialize();
    api();
    if (scenario == "early-exit") return 0;
    if (scenario == "crash") std::_Exit(27);
    if (scenario == "hang") {
      std::mutex mutex;
      std::unique_lock lock(mutex);
      std::condition_variable never;
      never.wait(lock, [] { return false; });
    }
    if (scenario == "api-open-at-seal") return open_at_seal(false);
    if (scenario == "callback-open-at-seal") return open_at_seal(true);
    if (scenario == "activity-outstanding-at-drain") {
      observer::activity_begin(true); observer::activity_end(false);
      observer::seal(); terminal();
      observer::activity_begin(false); observer::activity_end(true);
      return 0;
    }
    if (scenario == "activity-callback-open-at-drain") {
      observer::activity_begin(true);
      observer::seal(); terminal();
      observer::activity_end(false);
      observer::activity_begin(false); observer::activity_end(true);
      return 0;
    }
    observer::seal();
    if (scenario == "sampled") observer::sampler_disabled(39);
    terminal();
    if (scenario == "clean" || scenario == "sampled") return 0;
    if (scenario == "late-api-thread") {
      std::thread producer([] { api(); }); producer.join();
    } else if (scenario == "late-fatal") {
      observer::callback_begin(false, false, false);
      observer::error();
      observer::callback_end(false, false);
    } else if (scenario == "late-activity") {
      observer::activity_begin(true); observer::activity_end(false);
      observer::activity_begin(false); observer::activity_end(true);
    } else if (scenario == "open-api-after-footer") {
      observer::callback_begin(true, true, false);
      observer::callback_end(true, true);
    } else if (scenario == "open-callback-after-footer") {
      observer::callback_begin(false, false, false);
    } else if (scenario == "activity-underflow") {
      observer::activity_begin(false); observer::activity_end(true);
    } else if (scenario == "api-underflow") {
      observer::callback_begin(true, false, false);
      observer::callback_end(true, false);
    } else {
      return 64;
    }
    return 0;
  } catch (...) {
    observer::error();
    return 27;
  }
}
