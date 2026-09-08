#include "core.hpp"
#include "census.hpp"
#include <iostream>
#include <limits>
#include <atomic>
#include <thread>

using namespace xvram::launch_probe;
namespace {
void require(bool value) { if (!value) throw std::runtime_error("test assertion"); }
template<class F> void fails(F f) {
  bool failed = false;
  try { f(); } catch (const std::runtime_error&) { failed = true; }
  require(failed);
}
struct Fake {
  std::vector<Parameter> layout{{0, 8}, {16, 4}};
  std::string label = "kernel";
  int calls = 0, fault = -1;
  void step() { if (calls++ == fault) throw std::runtime_error("injected"); }
  bool resolve() { step(); return true; }
  std::string name() { step(); return label; }
  std::size_t count() { step(); return layout.size(); }
  Parameter parameter(std::size_t index) { step(); return layout.at(index); }
};
struct FakeContextless {
  int calls = 0, fault = -1, context = 1, owner = 1, result = 2;
  void step() { if (calls++ == fault) throw std::runtime_error("injected resolution"); }
  int current() { step(); return context; }
  int stream_context(int) { step(); return owner; }
  int function() { step(); return result; }
};
}
int main() {
  try {
    FakeContextless resolved;
    require(resolve_contextless(resolved) == 2 && resolved.calls == 3);
    for (int fault = 0; fault < 3; ++fault) {
      FakeContextless broken; broken.fault = fault;
      fails([&] { (void)resolve_contextless(broken); });
      require(broken.calls == fault + 1); // no retry or fallback
    }
    FakeContextless wrong_context; wrong_context.owner = 2;
    fails([&] { (void)resolve_contextless(wrong_context); });
    require(wrong_context.calls == 2); // never resolves in the wrong context
    FakeContextless missing_context; missing_context.context = 0;
    fails([&] { (void)resolve_contextless(missing_context); });
    require(missing_context.calls == 1);
    FakeContextless missing_function; missing_function.result = 0;
    fails([&] { (void)resolve_contextless(missing_function); });
    require(census::current_call == 0);
    try { census::Scope invalid(0); require(false); } catch (const std::logic_error&) {}
    {
      census::Scope scope(7);
      require(census::current_call == 7);
      try { census::Scope overlapping(8); require(false); } catch (const std::logic_error&) {}
      require(census::current_call == 7);
      bool independent = false;
      std::thread other([&] { independent = census::current_call == 0; census::Scope local(8); });
      other.join(); require(independent && census::current_call == 7);
    }
    require(census::current_call == 0);
    try { census::Scope unwind(9); throw std::runtime_error("unwind"); }
    catch (const std::runtime_error&) { require(census::current_call == 0); }
    census::ApiStack stack;
    fails([&] { (void)stack.exit(0, 1, 1, 1); });
    fails([&] { (void)stack.enter(0, 0, 1, 1, 1); });
    require(stack.enter(1, 0, 1, 2, 7).parent == 0);
    require(stack.enter(2, 5, 2, 3, 7).parent == 1); // shared CUPTI correlation, distinct call
    fails([&] { (void)stack.exit(0, 2, 3, 7); });
    fails([&] { (void)stack.exit(5, 1, 3, 7); });
    fails([&] { (void)stack.exit(5, 2, 2, 7); });
    fails([&] { (void)stack.exit(5, 2, 3, 8); });
    require(stack.exit(5, 2, 3, 7).id == 2);
    require(stack.exit(0, 1, 2, 7).id == 1);
    for (std::uint64_t i = 1; i <= 64; ++i) (void)stack.enter(i, 0, 1, 1, 1);
    fails([&] { (void)stack.enter(65, 0, 1, 1, 1); });
    for (std::uint64_t i = 64; i; --i) require(stack.exit(0, 1, 1, 1).id == i);
    Fake query;
    const auto first = inspect(query);
    require(first.parameters.size() == 2 && first.parameters[1].offset == 16);
    query.label = "new_generation";
    query.layout = {{0, 4}};
    const auto second = inspect(query);
    require(first.name == "kernel" && second.name == "new_generation" && second.parameters.size() == 1);
    for (int fault = 0; fault < 5; ++fault) {
      Fake broken; broken.fault = fault;
      fails([&] { (void)inspect(broken); });
    }
    for (const auto& invalid : std::vector<std::vector<Parameter>>{
           {{0, 0}}, {{0, 8}, {4, 8}}, {{65536, 1}},
           {{std::numeric_limits<std::size_t>::max(), 2}}, {{0, 65537}}}) {
      Fake broken; broken.layout = invalid;
      fails([&] { (void)inspect(broken); });
    }
    query.layout.clear(); require(inspect(query).parameters.empty());
    query.layout.resize(257); fails([&] { (void)inspect(query); });
    query.layout.clear(); query.label.clear(); fails([&] { (void)inspect(query); });
    query.label.assign(4097, 'a'); fails([&] { (void)inspect(query); });
    std::string order;
    const auto result = invoke([&] { order += 'Q'; return Snapshot{}; },
      [&](const Snapshot&) { order += 'B'; }, [&] { order += 'F'; return 17; },
      [&](int r) { require(r == 17); order += 'E'; });
    require(result == 17 && order == "QBFE");
    int forwards = 0;
    std::atomic_bool fault_enabled{true};
    fails([&] { (void)invoke([] { return Snapshot{}; },
      [&](const Snapshot&) { if (fault_enabled.load()) throw std::runtime_error("admission"); },
      [&] { ++forwards; return 0; }, [](int) {}); });
    require(forwards == 0);
    fails([&] { (void)invoke([] { return Snapshot{}; }, [](const Snapshot&) {},
      [&] { ++forwards; return 0; }, [&](int) { if (fault_enabled.load()) throw std::runtime_error("write"); }); });
    require(forwards == 1); // no replay after an uncertain trace write
    std::cout << "launch probe core passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
