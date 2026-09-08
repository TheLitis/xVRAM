#include "core.hpp"
#include <iostream>
#include <limits>
#include <atomic>

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
}
int main() {
  try {
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
