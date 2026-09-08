#include "compat_launch_probe/memory_registry.hpp"
#include <iostream>
#include <string_view>

using namespace xvram::launch_probe::memory_witness;
namespace {
void check(bool value, std::string_view message) {
  if (!value) throw std::runtime_error(std::string(message));
}
template<class F> void rejected(F operation) {
  bool failed=false;
  try { operation(); } catch (const std::runtime_error&) { failed=true; }
  check(failed,"expected rejection");
}
void ordinary() {
  MemoryRegistry model;
  const auto first=model.allocate(4097,64,false,9);
  const auto range=model.resolve(4098,63,0,9,16);
  check(range.known && range.mapped && range.readable && range.writable,"ordinary context access");
  check(range.allocation_id==first.id && range.generation==1 && range.offset_bytes==1,"ordinary identity");
  check(range.base_mod_alignment==1 && range.alignment==16,"alignment residue");
  check(!model.resolve(4098,63,0,10).readable,"other context is not implicit peer access");
  check(!model.resolve(4098,64,0,9).known,"tail cannot escape");
  check(!model.resolve(4161,1,0,9).known,"one past allocation");
  rejected([&]{ model.release(4098,0,false); });
  rejected([&]{ model.allocate(4100,4,false); });
  model.release(4097,0,false);
  check(!model.resolve(4097,1,0,9).known,"freed range");
  rejected([&]{ model.release(4097,0,false); });
  const auto second=model.allocate(4097,32,false,9);
  check(second.id!=first.id && second.generation==2,"VA ABA has fresh id and generation");
  model.release(4097,0,false);
  check(model.counts().allocation_bytes==0 && model.counts().allocations==0,"ordinary accounting");
}
void mapping_and_access() {
  MemoryRegistry model;
  const auto allocation=model.allocate(65536,8192,true);
  model.create(5,4096); model.create(6,4096);
  check(model.resolve(65536,1,0).known && !model.resolve(65536,1,0).mapped,"reserved is not mapped");
  const auto first=model.map(65536,4096,5);
  check(model.resolve(65536,4096,0).mapped && !model.resolve(65536,4096,0).readable,"map needs SetAccess");
  check(!model.resolve(65536,8192,0).mapped,"mapping hole");
  rejected([&]{ model.set_access(65536,8192,0,3); });
  model.map(69632,4096,6);
  model.set_access(65536,8192,0,3);
  auto all=model.resolve(65536,8192,0);
  check(all.known && all.readable && all.writable && all.mappings.size()==2,"multi-map coverage");
  check(all.allocation_id==allocation.id && all.mappings[0].mapping_id==first.mapping_id,"mapping identities");
  check(!model.resolve(65536,8192,1).readable,"device accessibility separate");
  model.set_access(67584,4096,0,1);
  check(model.resolve(67584,4096,0).readable && !model.resolve(67584,4096,0).writable,"partial read-only replacement");
  check(model.resolve(65536,2048,0).writable && model.resolve(71680,2048,0).writable,"untouched access intervals");
  model.set_access(67584,4096,1,3);
  check(model.resolve(67584,4096,1).writable && !model.resolve(67584,4096,0).writable,"independent device descriptors");
  model.set_access(67584,4096,0,0);
  check(!model.resolve(67584,1,0).readable,"revocation not access");
  rejected([&]{ model.unmap(65536,2048); });
  rejected([&]{ model.unmap(65536,8192); });
  rejected([&]{ model.release(65536,8192,true); });
  model.unmap(65536,4096);
  const auto remap=model.map(65536,4096,5);
  check(remap.mapping_id!=first.mapping_id && !model.resolve(65536,4096,0).readable,"remap resets access generation");
  model.unmap(65536,4096); model.unmap(69632,4096);
  model.release_handle(5); model.release_handle(6);
  rejected([&]{ model.release(65536,4096,true); });
  model.release(65536,8192,true);
  const auto counts=model.counts();
  check(!counts.reservations && !counts.handles && !counts.mappings && !counts.physical_bytes
        && !counts.mapped_bytes && !counts.reservation_bytes,"complete ledger accounting");
}
void aliases_and_handle_aba() {
  MemoryRegistry model;
  model.allocate(65536,4096,true); model.allocate(131072,4096,true);
  const auto physical=model.create(7,4096);
  model.map(65536,4096,7); model.map(131072,4096,7);
  const auto a=model.resolve(65536,1,0), b=model.resolve(131072,1,0);
  check(a.allocation_id!=b.allocation_id && a.mappings[0].physical_id==b.mappings[0].physical_id,"explicit physical aliases");
  model.release_handle(7); // Legal: mappings keep the allocation alive.
  check(model.counts().physical_bytes==4096 && model.counts().handles==0,"release does not drop mapped physical authority");
  rejected([&]{ model.release_handle(7); });
  const auto reused=model.create(7,4096);
  check(reused.id!=physical.id && reused.generation==2,"handle ABA");
  check(model.resolve(65536,1,0).mappings[0].physical_generation==1,"old mapping preserves old physical generation");
  model.unmap(65536,4096);
  check(model.counts().physical_bytes==8192,"alias still retains old generation");
  model.unmap(131072,4096);
  check(model.counts().physical_bytes==4096,"last alias retires released generation");
  model.release_handle(7);
  check(model.counts().physical_bytes==0,"new generation independent");
}
void rejection_and_limits() {
  MemoryRegistry model;
  rejected([&]{ model.allocate(0,1,false); });
  rejected([&]{ model.allocate(1,0,false); });
  rejected([&]{ model.allocate(std::numeric_limits<Word>::max(),1,false); });
  rejected([&]{ (void)model.resolve(1,0,0); });
  rejected([&]{ (void)model.resolve(1,1,0,0,3); });
  rejected([&]{ (void)model.resolve(1,1,0,0,0); });
  rejected([&]{ model.map(65536,4096,1); });
  model.allocate(65536,8192,true);
  rejected([&]{ model.map(65536,4096,1); });
  model.create(1,4096);
  rejected([&]{ model.create(1,4096); });
  rejected([&]{ model.map(65536,8192,1); });
  rejected([&]{ model.map(65536,4096,1,4096); });
  model.map(65536,4096,1);
  rejected([&]{ model.map(65536,4096,1); });
  rejected([&]{ model.set_access(65536,4096,-1,3); });
  rejected([&]{ model.set_access(65536,4096,0,2); });
  MemoryRegistry budget(32,32,4096);
  budget.allocate(65536,4096,false);
  rejected([&]{ budget.allocate(131072,1,false); });
  check(budget.counts().allocation_bytes==4096,"budget rejection before mutation");
  MemoryRegistry objects(1);
  objects.allocate(1,1,false);
  rejected([&]{ objects.create(1,1); });
  MemoryRegistry access(32,1);
  access.allocate(65536,4096,true); access.create(1,4096); access.map(65536,4096,1);
  access.set_access(65536,4096,0,3);
  rejected([&]{ access.set_access(66560,1024,0,1); });
  check(access.resolve(65536,4096,0).writable,"access admission is atomic");
}
}
int main() {
  try {
    ordinary(); mapping_and_access(); aliases_and_handle_aba(); rejection_and_limits();
    std::cout<<"memory registry: allocation, VMM, access, alias, ABA and limit checks passed\n";
    return 0;
  } catch (const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
