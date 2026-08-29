#pragma once

#include "platform/cuda/cuda_api.hpp"
#include "xvram/probe/report.hpp"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace xvram::cuda {

class Driver {
public:
  Driver() = default;

  void collect(probe::CudaReport& report, std::vector<probe::Diagnostic>& diagnostics,
               const probe::ProbeOptions& options);
  [[nodiscard]] probe::TransferMeasurement
  benchmark_transfers(std::int32_t ordinal, const probe::ProbeOptions& options,
                      std::vector<probe::Diagnostic>& diagnostics);
  [[nodiscard]] probe::OverlapMeasurement
  benchmark_overlap(std::int32_t ordinal, const probe::ProbeOptions& options,
                    std::vector<probe::Diagnostic>& diagnostics);

private:
  [[nodiscard]] bool load(std::vector<probe::Diagnostic>& diagnostics);
  [[nodiscard]] std::string result_name(abi::Result result) const;
  [[nodiscard]] std::string result_message(abi::Result result) const;
  [[nodiscard]] std::optional<int>
  query_attribute(abi::Device device, abi::NativeDeviceAttribute attribute,
                  std::string_view report_key, std::vector<probe::Diagnostic>& diagnostics) const;
  [[nodiscard]] std::optional<std::uint64_t>
  query_free_memory(abi::Device device, std::vector<probe::Diagnostic>& diagnostics) const;
  [[nodiscard]] probe::GranularityInfo
  query_granularity(CUmemLocationType location_type, std::int32_t location_id,
                    std::string_view report_key, std::vector<probe::Diagnostic>& diagnostics) const;
  [[nodiscard]] probe::VmmSmokeResult smoke_test_vmm(abi::Device device, std::int32_t ordinal,
                                                     std::uint64_t allocation_bytes,
                                                     std::vector<probe::Diagnostic>& diagnostics);

  struct QuarantinedVmmState {
    std::int32_t device_ordinal{};
    abi::DevicePointer reservation{};
    abi::DevicePointer mapped_address{};
    std::size_t reservation_bytes{};
    std::size_t mapping_bytes{};
    std::optional<abi::GenericAllocationHandle> handle;
    bool mapping_active{};
    bool reservation_active{};
  };

  CudaApi api_;
  std::vector<QuarantinedVmmState> quarantined_vmm_;
  bool active_cuda_poisoned_ = false;
};

} // namespace xvram::cuda
