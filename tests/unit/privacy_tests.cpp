#include "probe/privacy.hpp"
#include "xvram/probe/report.hpp"

#include <cstdint>
#include <iostream>

namespace {

int failures = 0;

void check(const bool condition, const char* expression, const int line) {
  if (!condition) {
    std::cerr << "line " << line << ": check failed: " << expression << '\n';
    ++failures;
  }
}

#define CHECK(expression) check(static_cast<bool>(expression), #expression, __LINE__)

[[nodiscard]] xvram::probe::ProbeReport populated_report() {
  xvram::probe::ProbeReport report;
  xvram::probe::DeviceReport device;
  device.ordinal = 3;
  device.name = "Privacy fixture GPU";
  device.uuid = "GPU-00112233-4455-6677-8899-aabbccddeeff";
  device.luid = "0102030405060708";
  device.node_mask = 4;
  device.pci_bus_id = "00000000:65:00.0";
  device.attributes.emplace("pci_domain_id", 0);
  device.attributes.emplace("pci_bus_id", 0x65);
  device.attributes.emplace("pci_device_id", 0);
  device.attributes.emplace("multiprocessor_count", 46);

  xvram::probe::DxgiAdapterInfo dxgi;
  dxgi.name = "Privacy fixture GPU";
  dxgi.luid = "0102030405060708";
  device.dxgi = dxgi;

  report.cuda.devices.push_back(device);
  return report;
}

void default_policy_redacts_identifiers() {
  xvram::probe::ProbeReport report = populated_report();
  xvram::probe::apply_identifier_policy(report, false);

  const xvram::probe::DeviceReport& device = report.cuda.devices.front();
  CHECK(!device.uuid.has_value());
  CHECK(!device.luid.has_value());
  CHECK(!device.pci_bus_id.has_value());
  CHECK(device.attributes.find("pci_domain_id") == device.attributes.end());
  CHECK(device.attributes.find("pci_bus_id") == device.attributes.end());
  CHECK(device.attributes.find("pci_device_id") == device.attributes.end());
  CHECK(device.attributes.at("multiprocessor_count") == 46);
  CHECK(device.node_mask == std::uint32_t{4});
  CHECK(device.dxgi.has_value());
  CHECK(device.dxgi->luid.empty());
}

void opt_in_policy_retains_identifiers() {
  xvram::probe::ProbeReport report = populated_report();
  xvram::probe::apply_identifier_policy(report, true);

  const xvram::probe::DeviceReport& device = report.cuda.devices.front();
  CHECK(device.uuid == "GPU-00112233-4455-6677-8899-aabbccddeeff");
  CHECK(device.luid == "0102030405060708");
  CHECK(device.pci_bus_id == "00000000:65:00.0");
  CHECK(device.attributes.at("pci_domain_id") == 0);
  CHECK(device.attributes.at("pci_bus_id") == 0x65);
  CHECK(device.attributes.at("pci_device_id") == 0);
  CHECK(device.dxgi.has_value());
  CHECK(device.dxgi->luid == "0102030405060708");
}

} // namespace

int main() {
  default_policy_redacts_identifiers();
  opt_in_policy_retains_identifiers();
  if (failures != 0) {
    std::cerr << failures << " privacy test(s) failed\n";
    return 1;
  }
  std::cout << "all privacy tests passed\n";
  return 0;
}
