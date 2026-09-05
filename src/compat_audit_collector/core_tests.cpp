#include "trace_core.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void check(bool value) { if (!value) throw std::runtime_error("collector core test failed"); }
}
int main() {
  using namespace xvram::audit;
  check(trace_version(nullptr) == 1 && trace_version("1") == 1 && trace_version("2") == 2);
  for (const char * invalid : {"", "0", "3", "02", "2 ", "-1"}) {
    bool invalid_version = false;
    try { static_cast<void>(trace_version(invalid)); } catch (const std::invalid_argument &) { invalid_version = true; }
    check(invalid_version);
  }
  check(resolver_symbol("cuMemAlloc_v2") && resolver_symbol("cudaLaunchKernel_ptsz"));
  for (const auto invalid : {"", "C:\\secret", "bad name", "123symbol", "name_0x123abc", "bad\nname"})
    check(!resolver_symbol(invalid));
  check(!resolver_symbol(std::string(4097, 'x')));
  Registry registry;
  const auto first = registry.allocate(0x123400, 128);
  check(first.known && first.generation == 1);
  check(registry.allocate(0x123400, 128).allocation_id == first.allocation_id);
  check(registry.lookup(0x123407, 121).offset_bytes == 7);
  check(!registry.lookup(0x123407, 122).known);
  check(!registry.lookup(0x1233ff, 1).known);
  check(!registry.allocate(std::numeric_limits<std::uint64_t>::max() - 5, 8).known);
  check(registry.release(0x123400).allocation_id == first.allocation_id);
  check(!registry.lookup(0x123400, 1).known);
  const auto next = registry.allocate(0x123400, 128);
  check(next.allocation_id != first.allocation_id && next.generation == 2);
  bool rejected = false;
  try { static_cast<void>(registry.allocate(0x123420,128)); } catch (const std::runtime_error &) { rejected = true; }
  check(rejected);
  // Dead bases must not hide a later larger live interval at an earlier base.
  static_cast<void>(registry.release(0x123400));
  static_cast<void>(registry.allocate(0x123300,512));
  check(registry.lookup(0x123407,1).known);
  check(registry.identity("stream", 123) == registry.identity("stream", 123));
  check(registry.identity("stream", 123) != registry.identity("event", 123));
  const auto old_event=registry.identity("event",123);
  registry.retire_identity("event",123);
  check(old_event!=registry.identity("event",123));
  JsonLine line;
  line.string("name", "test\n\"\\");
  add_range(line, registry.lookup(0x123407, 1));
  const auto json = line.finish();
  check(json.find("test\\u000a\\\"\\\\") != std::string::npos);
  check(json.find("123400") == std::string::npos);
  check(json.find("\"offset_bytes\":263") != std::string::npos);
  rejected = false;
  try { line.string("large", std::string(4097, 'x')); } catch (const std::length_error &) { rejected = true; }
  check(rejected);
  Registry limited(1);
  static_cast<void>(limited.identity("stream", 1));
  rejected = false;
  try { static_cast<void>(limited.identity("stream", 2)); } catch (const std::length_error &) { rejected = true; }
  check(rejected);
  std::cout << "collector core tests passed\n";
}
