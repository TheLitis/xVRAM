#include "trace_core.hpp"
#include "module_evidence.hpp"
#include "platform/sha256.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void check(bool value) { if (!value) throw std::runtime_error("collector core test failed"); }
}
int main() {
  using namespace xvram::audit;
  namespace platform = xvram::platform;
  check(trace_version(nullptr) == 1 && trace_version("1") == 1 && trace_version("2") == 2);
  check(trace_version("3") == 3);
  for (const char * invalid : {"", "0", "4", "02", "2 ", "-1"}) {
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
  ModuleLifetimes modules;
  check(modules.load(1,7).generation==1);
  check(modules.load(2,7).generation==1);
  rejected=false;
  try { static_cast<void>(modules.load(1,7)); } catch(const std::logic_error &) { rejected=true; }
  check(rejected);
  check(modules.unload(1,7).generation==1 && !modules.lookup(1,7).live);
  check(modules.unload(1,7).generation==0);
  check(modules.load(1,7).generation==2);
  ModuleCopies copies(4,6,10);
  char data[]="abcd";
  check(!copies.push(nullptr,4,{}) && !copies.push(data,5,{}) && !copies.push(data,0,{}));
  check(copies.push(data,4,ModuleCopy{1,7,1,42,{}}));
  data[0]='z';
  ModuleCopy item;
  check(copies.pop(item) && item.bytes[0]==std::byte{'a'} && item.source_sequence==42);
  check(!copies.push(data,4,{}) && copies.active_bytes==4);
  check(platform::sha256_bytes(item.bytes)=="88d4266fd4e6338d13b845fcf289579d209c897823b9217da3e161936f031589");
  item.bytes.clear(); copies.retire(4);
  check(copies.push(data,4,{})); check(copies.pop(item)); copies.retire(4);
  check(!copies.push(data,4,{}) && copies.push(data,2,{}));
  check(copies.pop(item)); copies.retire(2);
  check(copies.active_bytes==0 && copies.total_bytes==10 && copies.peak_bytes==4);
  check(copies.submitted==3 && copies.retired==3 && !copies.pop(item));
  check(platform::sha256_bytes({})=="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  ModuleCopies failing(4,6,10,[](std::vector<std::byte> &,const void *,std::size_t) { throw std::bad_alloc(); });
  rejected=false;
  try { static_cast<void>(failing.push(data,4,{})); } catch(const std::bad_alloc &) { rejected=true; }
  check(rejected && failing.submitted==0 && failing.active_bytes==0 && failing.total_bytes==0);
  check(!failing.pop(item));
  std::cout << "collector core tests passed\n";
}
