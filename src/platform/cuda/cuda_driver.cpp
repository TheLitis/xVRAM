#include "platform/cuda/cuda_driver.hpp"

#include "platform/dxgi_memory.hpp"
#include "xvram/synthetic_overlap_ptx.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string_view>

namespace xvram::cuda {
namespace {

using probe::Diagnostic;
using probe::DiagnosticLevel;

[[nodiscard]] std::string bytes_to_hex(const std::uint8_t* bytes, const std::size_t count) {
  static constexpr char hex[] = "0123456789abcdef";
  std::string result;
  result.reserve(count * 2);
  for (std::size_t index = 0; index < count; ++index) {
    result.push_back(hex[(bytes[index] >> 4U) & 0xFU]);
    result.push_back(hex[bytes[index] & 0xFU]);
  }
  return result;
}

[[nodiscard]] std::string format_uuid(const abi::Uuid& uuid) {
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(uuid.bytes);
  const std::string raw = bytes_to_hex(bytes, std::size(uuid.bytes));
  return raw.substr(0, 8) + "-" + raw.substr(8, 4) + "-" + raw.substr(12, 4) + "-" +
         raw.substr(16, 4) + "-" + raw.substr(20);
}

[[nodiscard]] bool capability_value(const std::map<std::string, std::optional<bool>>& capabilities,
                                    const std::string& key) {
  const auto iterator = capabilities.find(key);
  return iterator != capabilities.end() && iterator->second.value_or(false);
}

} // namespace

bool Driver::load(std::vector<Diagnostic>& diagnostics) {
  const CudaApi::LoadResult load_result = api_.load();
  if (load_result.attempted_now && load_result.status == CudaApi::LoadStatus::library_unavailable) {
    diagnostics.push_back({DiagnosticLevel::warning, "cuda", "load_driver",
                           "CUDA Driver API library was not loaded: " + api_.error(),
                           std::nullopt});
  } else if (load_result.attempted_now &&
             load_result.status == CudaApi::LoadStatus::baseline_symbols_missing) {
    diagnostics.push_back({DiagnosticLevel::error, "cuda", "resolve_symbols",
                           "The CUDA driver is missing one or more baseline symbols",
                           std::nullopt});
  }
  return load_result.status == CudaApi::LoadStatus::loaded;
}

std::string Driver::result_name(const abi::Result result) const {
  const char* name = nullptr;
  if (api_.get_error_name_ != nullptr && api_.get_error_name_(result, &name) == abi::success &&
      name != nullptr) {
    return name;
  }
  return "CUDA_ERROR_" + std::to_string(result);
}

std::string Driver::result_message(const abi::Result result) const {
  const char* message = nullptr;
  if (api_.get_error_string_ != nullptr &&
      api_.get_error_string_(result, &message) == abi::success && message != nullptr) {
    return message;
  }
  return result_name(result);
}

std::optional<int> Driver::query_attribute(const abi::Device device,
                                           const abi::NativeDeviceAttribute attribute,
                                           const std::string_view report_key,
                                           std::vector<Diagnostic>& diagnostics) const {
  int value = 0;
  if (api_.device_get_attribute_ == nullptr) {
    return std::nullopt;
  }
  const abi::Result result = api_.device_get_attribute_(&value, attribute, device);
  if (result != abi::success) {
    diagnostics.push_back({DiagnosticLevel::info, "cuda",
                           "cuDeviceGetAttribute(" + std::string(report_key) + ")",
                           result_message(result), result});
    return std::nullopt;
  }
  return value;
}

std::optional<std::uint64_t> Driver::query_free_memory(const abi::Device device,
                                                       std::vector<Diagnostic>& diagnostics) const {
  if (api_.context_create_ == nullptr || api_.context_destroy_ == nullptr ||
      api_.context_get_current_ == nullptr || api_.context_set_current_ == nullptr ||
      api_.mem_get_info_ == nullptr) {
    return std::nullopt;
  }

  abi::Context previous = nullptr;
  abi::Result result = api_.context_get_current_(&previous);
  if (result != abi::success) {
    return std::nullopt;
  }
  abi::Context context = nullptr;
  result = api_.context_create_(&context, CU_CTX_SCHED_AUTO, device);
  if (result != abi::success) {
    diagnostics.push_back(
        {DiagnosticLevel::warning, "cuda", "cuCtxCreate", result_message(result), result});
    return std::nullopt;
  }

  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  result = api_.mem_get_info_(&free_bytes, &total_bytes);
  const abi::Result destroy_result = api_.context_destroy_(context);
  const abi::Result restore_result = api_.context_set_current_(previous);
  if (destroy_result != abi::success || restore_result != abi::success) {
    const abi::Result cleanup_result =
        destroy_result != abi::success ? destroy_result : restore_result;
    diagnostics.push_back({DiagnosticLevel::warning, "cuda", "cleanup_probe_context",
                           "CUDA probe context cleanup reported an error", cleanup_result});
  }
  if (result != abi::success) {
    diagnostics.push_back(
        {DiagnosticLevel::warning, "cuda", "cuMemGetInfo", result_message(result), result});
    return std::nullopt;
  }
  return static_cast<std::uint64_t>(free_bytes);
}

probe::GranularityInfo Driver::query_granularity(const CUmemLocationType location_type,
                                                 const std::int32_t location_id,
                                                 const std::string_view report_key,
                                                 std::vector<Diagnostic>& diagnostics) const {
  probe::GranularityInfo info;
  if (api_.mem_get_allocation_granularity_ == nullptr) {
    info.error = "cuMemGetAllocationGranularity is not exported";
    return info;
  }

  abi::MemAllocationProp properties{};
  properties.type = abi::allocation_type_pinned;
  properties.location.type = location_type;
  properties.location.id = location_id;

  std::size_t value = 0;
  abi::Result result =
      api_.mem_get_allocation_granularity_(&value, &properties, abi::granularity_minimum);
  if (result == abi::success) {
    info.minimum_bytes = static_cast<std::uint64_t>(value);
  } else {
    info.error = result_name(result) + ": " + result_message(result);
    diagnostics.push_back({DiagnosticLevel::info, "cuda",
                           "cuMemGetAllocationGranularity(" + std::string(report_key) + ",minimum)",
                           result_message(result), result});
  }

  value = 0;
  result = api_.mem_get_allocation_granularity_(&value, &properties, abi::granularity_recommended);
  if (result == abi::success) {
    info.recommended_bytes = static_cast<std::uint64_t>(value);
  } else {
    if (!info.error.has_value()) {
      info.error = result_name(result) + ": " + result_message(result);
    }
    diagnostics.push_back(
        {DiagnosticLevel::info, "cuda",
         "cuMemGetAllocationGranularity(" + std::string(report_key) + ",recommended)",
         result_message(result), result});
  }
  return info;
}

probe::VmmSmokeResult Driver::smoke_test_vmm(const abi::Device device, const std::int32_t ordinal,
                                             const std::uint64_t allocation_bytes,
                                             std::vector<Diagnostic>& diagnostics) {
  probe::VmmSmokeResult smoke;
  smoke.status = "failed";
  smoke.allocation_bytes = allocation_bytes;

  if (api_.context_create_ == nullptr || api_.context_destroy_ == nullptr ||
      api_.context_get_current_ == nullptr || api_.context_set_current_ == nullptr ||
      api_.mem_address_reserve_ == nullptr || api_.mem_address_free_ == nullptr ||
      api_.mem_create_ == nullptr || api_.mem_release_ == nullptr || api_.mem_map_ == nullptr ||
      api_.mem_unmap_ == nullptr || api_.mem_set_access_ == nullptr ||
      api_.mem_host_alloc_ == nullptr || api_.mem_free_host_ == nullptr ||
      api_.memcpy_h2d_async_ == nullptr || api_.memcpy_d2h_async_ == nullptr ||
      api_.stream_create_ == nullptr || api_.stream_destroy_ == nullptr ||
      api_.stream_synchronize_ == nullptr) {
    smoke.message = "The CUDA driver is missing one or more VMM smoke-test entry points";
    return smoke;
  }
  if (allocation_bytes == 0 || allocation_bytes > std::numeric_limits<std::size_t>::max() / 2) {
    smoke.message = "Invalid VMM granularity for the smoke test";
    return smoke;
  }

  const std::size_t bytes = static_cast<std::size_t>(allocation_bytes);
  abi::Context previous = nullptr;
  abi::Context context = nullptr;
  abi::DevicePointer reservation = 0;
  abi::DevicePointer mapped_address = 0;
  abi::GenericAllocationHandle handle{};
  bool reservation_created = false;
  bool handle_created = false;
  bool mapped = false;
  void* source = nullptr;
  void* destination = nullptr;
  abi::Stream stream = nullptr;
  bool previous_captured = false;
  bool stream_synchronization_failed = false;

  abi::Result result = api_.context_get_current_(&previous);
  if (result == abi::success) {
    previous_captured = true;
    result = api_.context_create_(&context, CU_CTX_SCHED_AUTO, device);
  }

  const auto cleanup = [&]() -> bool {
    bool complete = true;
    const auto record_cleanup_error = [&](const char* operation, const abi::Result error) {
      if (error == abi::success) {
        return true;
      }
      complete = false;
      diagnostics.push_back(
          {DiagnosticLevel::warning, "vmm_smoke", operation, result_message(error), error});
      return false;
    };

    if (stream != nullptr) {
      const abi::Result synchronize_result = api_.stream_synchronize_(stream);
      if (synchronize_result != abi::success) {
        stream_synchronization_failed = true;
      }
      record_cleanup_error("cuStreamSynchronize(cleanup)", synchronize_result);
    }

    if (mapped) {
      if (record_cleanup_error("cuMemUnmap(cleanup)", api_.mem_unmap_(mapped_address, bytes))) {
        mapped = false;
      }
    }
    // CUDA explicitly permits releasing a generic allocation handle while mappings still exist.
    // The mapping retains the allocation until it can be unmapped, so never lose our handle merely
    // because cuMemUnmap failed.
    if (handle_created) {
      if (record_cleanup_error("cuMemRelease(cleanup)", api_.mem_release_(handle))) {
        handle_created = false;
      }
    }
    if (reservation_created && !mapped) {
      if (record_cleanup_error("cuMemAddressFree(cleanup)",
                               api_.mem_address_free_(reservation, bytes * 2))) {
        reservation_created = false;
      }
    }
    if (stream != nullptr) {
      if (record_cleanup_error("cuStreamDestroy(cleanup)", api_.stream_destroy_(stream))) {
        stream = nullptr;
      }
    }
    if (destination != nullptr) {
      if (record_cleanup_error("cuMemFreeHost(destination)", api_.mem_free_host_(destination))) {
        destination = nullptr;
      }
    }
    if (source != nullptr) {
      if (record_cleanup_error("cuMemFreeHost(source)", api_.mem_free_host_(source))) {
        source = nullptr;
      }
    }

    bool context_destroyed = context == nullptr;
    if (context != nullptr) {
      if (record_cleanup_error("cuCtxDestroy(cleanup)", api_.context_destroy_(context))) {
        context = nullptr;
        context_destroyed = true;
      }
    }
    if (stream_synchronization_failed && context_destroyed) {
      diagnostics.push_back(
          {DiagnosticLevel::warning, "vmm_smoke", "cleanup_after_stream_error",
           "VMM teardown was attempted and the isolated context was destroyed after stream "
           "synchronization reported an error",
           std::nullopt});
      complete = false;
    }

    if (mapped || reservation_created || handle_created) {
      quarantined_vmm_.push_back(
          {ordinal, reservation, mapped_address, bytes * 2, bytes,
           handle_created ? std::optional<abi::GenericAllocationHandle>(handle) : std::nullopt,
           mapped, reservation_created});
      mapped = false;
      reservation_created = false;
      handle_created = false;
      diagnostics.push_back(
          {DiagnosticLevel::error, "vmm_smoke", "quarantine_vmm_resources",
           "Unresolved VMM state was quarantined for the remaining Driver lifetime; further "
           "active CUDA work is disabled",
           std::nullopt});
      complete = false;
    }
    if (previous_captured) {
      record_cleanup_error("cuCtxSetCurrent(restore)", api_.context_set_current_(previous));
    }
    if (!complete || stream_synchronization_failed) {
      active_cuda_poisoned_ = true;
    }
    return complete;
  };

  const auto fail = [&](const char* operation, const abi::Result error) {
    smoke.message =
        std::string(operation) + ": " + result_name(error) + ": " + result_message(error);
    diagnostics.push_back(
        {DiagnosticLevel::warning, "vmm_smoke", operation, result_message(error), error});
    smoke.cleanup_complete = cleanup();
  };

  if (result != abi::success) {
    fail(previous_captured ? "cuCtxCreate" : "cuCtxGetCurrent", result);
    return smoke;
  }

  abi::MemAllocationProp properties{};
  properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  properties.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
  properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  properties.location.id = ordinal;

  result = api_.mem_address_reserve_(&reservation, bytes * 2, 0, 0, 0);
  if (result != abi::success) {
    fail("cuMemAddressReserve", result);
    return smoke;
  }
  reservation_created = true;
  result = api_.mem_create_(&handle, bytes, &properties, 0);
  if (result != abi::success) {
    fail("cuMemCreate", result);
    return smoke;
  }
  handle_created = true;

  mapped_address = reservation;
  result = api_.mem_map_(mapped_address, bytes, 0, handle, 0);
  if (result != abi::success) {
    fail("cuMemMap(first)", result);
    return smoke;
  }
  mapped = true;

  CUmemAccessDesc access{};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = ordinal;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  if ((result = api_.mem_set_access_(mapped_address, bytes, &access, 1)) != abi::success ||
      (result = api_.mem_host_alloc_(&source, bytes, 0)) != abi::success ||
      (result = api_.mem_host_alloc_(&destination, bytes, 0)) != abi::success ||
      (result = api_.stream_create_(&stream, CU_STREAM_NON_BLOCKING)) != abi::success) {
    fail("prepare_first_mapping", result);
    return smoke;
  }

  auto* source_bytes = static_cast<std::uint8_t*>(source);
  auto* destination_bytes = static_cast<std::uint8_t*>(destination);
  for (std::size_t index = 0; index < bytes; ++index) {
    source_bytes[index] = static_cast<std::uint8_t>((index * 131U + 17U) & 0xFFU);
  }
  std::memset(destination, 0, bytes);

  if ((result = api_.memcpy_h2d_async_(mapped_address, source, bytes, stream)) != abi::success ||
      (result = api_.memcpy_d2h_async_(destination, mapped_address, bytes, stream)) !=
          abi::success) {
    fail("copy_first_mapping", result);
    return smoke;
  }
  result = api_.stream_synchronize_(stream);
  if (result != abi::success) {
    stream_synchronization_failed = true;
    fail("cuStreamSynchronize(first_mapping)", result);
    return smoke;
  }
  smoke.initial_copy_verified = std::memcmp(source, destination, bytes) == 0;
  if (!*smoke.initial_copy_verified) {
    smoke.message = "Data verification failed before remapping";
    diagnostics.push_back({DiagnosticLevel::error, "vmm_smoke", "verify_first_mapping",
                           *smoke.message, std::nullopt});
    smoke.cleanup_complete = cleanup();
    return smoke;
  }

  result = api_.mem_unmap_(mapped_address, bytes);
  if (result != abi::success) {
    fail("cuMemUnmap(first)", result);
    return smoke;
  }
  mapped = false;
  mapped_address = reservation + static_cast<abi::DevicePointer>(bytes);
  result = api_.mem_map_(mapped_address, bytes, 0, handle, 0);
  if (result == abi::success) {
    mapped = true;
    result = api_.mem_set_access_(mapped_address, bytes, &access, 1);
  }
  if (result != abi::success) {
    fail("map_second_address", result);
    return smoke;
  }

  std::memset(destination_bytes, 0, bytes);
  if ((result = api_.memcpy_d2h_async_(destination, mapped_address, bytes, stream)) !=
      abi::success) {
    fail("copy_second_mapping", result);
    return smoke;
  }
  result = api_.stream_synchronize_(stream);
  if (result != abi::success) {
    stream_synchronization_failed = true;
    fail("cuStreamSynchronize(second_mapping)", result);
    return smoke;
  }
  smoke.remap_copy_verified = std::memcmp(source, destination, bytes) == 0;
  if (!*smoke.remap_copy_verified) {
    smoke.message = "The physical allocation did not preserve data across remapping";
    diagnostics.push_back({DiagnosticLevel::error, "vmm_smoke", "verify_second_mapping",
                           *smoke.message, std::nullopt});
    smoke.cleanup_complete = cleanup();
    return smoke;
  }

  smoke.cleanup_complete = cleanup();
  if (!*smoke.cleanup_complete) {
    smoke.message = "VMM data verification passed, but cleanup reported an error";
    return smoke;
  }
  smoke.status = "completed";
  smoke.message = "Physical allocation preserved data after remapping to a second virtual address";
  return smoke;
}

void Driver::collect(probe::CudaReport& report, std::vector<Diagnostic>& diagnostics,
                     const probe::ProbeOptions& options) {
  report.library_loaded = load(diagnostics);
  report.library_name = api_.loaded_name();
  if (!report.library_loaded) {
    return;
  }

  const abi::Result initialization = api_.init_(0);
  report.initialization_code = initialization;
  report.initialization_name = result_name(initialization);
  report.initialization_message = result_message(initialization);
  if (initialization != abi::success) {
    diagnostics.push_back(
        {DiagnosticLevel::error, "cuda", "cuInit", result_message(initialization), initialization});
    return;
  }

  int raw_driver_version = 0;
  const abi::Result version_result = api_.driver_get_version_(&raw_driver_version);
  if (version_result == abi::success) {
    report.driver_version = probe::DriverVersion{raw_driver_version, raw_driver_version / 1000,
                                                 (raw_driver_version % 1000) / 10};
  } else {
    diagnostics.push_back({DiagnosticLevel::warning, "cuda", "cuDriverGetVersion",
                           result_message(version_result), version_result});
  }

  int device_count = 0;
  const abi::Result count_result = api_.device_get_count_(&device_count);
  if (count_result != abi::success) {
    diagnostics.push_back({DiagnosticLevel::error, "cuda", "cuDeviceGetCount",
                           result_message(count_result), count_result});
    return;
  }

  const std::array capability_attributes = {
      std::pair{"can_map_host_memory", abi::attributes::can_map_host_memory},
      std::pair{"concurrent_kernels", abi::attributes::concurrent_kernels},
      std::pair{"unified_addressing", abi::attributes::unified_addressing},
      std::pair{"managed_memory", abi::attributes::managed_memory},
      std::pair{"host_native_atomic", abi::attributes::host_native_atomic_supported},
      std::pair{"pageable_memory_access", abi::attributes::pageable_memory_access},
      std::pair{"concurrent_managed_access", abi::attributes::concurrent_managed_access},
      std::pair{"compute_preemption", abi::attributes::compute_preemption_supported},
      std::pair{"same_host_registered_pointer",
                abi::attributes::can_use_host_pointer_for_registered_memory},
      std::pair{"host_register", abi::attributes::host_register_supported},
      std::pair{"pageable_access_uses_host_page_tables",
                abi::attributes::pageable_memory_access_uses_host_page_tables},
      std::pair{"direct_managed_access_from_host",
                abi::attributes::direct_managed_memory_access_from_host},
      std::pair{"virtual_memory_management", abi::attributes::virtual_memory_management_supported},
      std::pair{"win32_shareable_handle", abi::attributes::win32_handle_supported},
      std::pair{"win32_kmt_shareable_handle", abi::attributes::win32_kmt_handle_supported},
      std::pair{"generic_compression", abi::attributes::generic_compression_supported},
      std::pair{"gpudirect_rdma_with_vmm",
                abi::attributes::gpu_direct_rdma_with_cuda_vmm_supported},
      std::pair{"device_memory_pools", abi::attributes::memory_pools_supported},
      std::pair{"gpudirect_rdma", abi::attributes::gpu_direct_rdma_supported},
      std::pair{"host_numa_vmm", abi::attributes::host_numa_virtual_memory_management_supported},
      std::pair{"host_numa_memory_pools", abi::attributes::host_numa_memory_pools_supported},
      std::pair{"host_memory_pools", abi::attributes::host_memory_pools_supported},
      std::pair{"host_vmm", abi::attributes::host_virtual_memory_management_supported},
  };

  const std::array numeric_attributes = {
      std::pair{"compute_capability_major", abi::attributes::compute_capability_major},
      std::pair{"compute_capability_minor", abi::attributes::compute_capability_minor},
      std::pair{"multiprocessor_count", abi::attributes::multiprocessor_count},
      std::pair{"warp_size", abi::attributes::warp_size},
      std::pair{"async_engine_count", abi::attributes::async_engine_count},
      std::pair{"clock_rate_khz", abi::attributes::clock_rate},
      std::pair{"memory_clock_rate_khz", abi::attributes::memory_clock_rate},
      std::pair{"memory_bus_width_bits", abi::attributes::global_memory_bus_width},
      std::pair{"l2_cache_bytes", abi::attributes::l2_cache_size},
      std::pair{"max_persisting_l2_cache_bytes", abi::attributes::max_persisting_l2_cache_size},
      std::pair{"max_threads_per_block", abi::attributes::max_threads_per_block},
      std::pair{"integrated", abi::attributes::integrated},
      std::pair{"tcc_driver", abi::attributes::tcc_driver},
      std::pair{"compute_mode", abi::attributes::compute_mode},
      std::pair{"pci_domain_id", abi::attributes::pci_domain_id},
      std::pair{"pci_bus_id", abi::attributes::pci_bus_id},
      std::pair{"pci_device_id", abi::attributes::pci_device_id},
      std::pair{"mempool_supported_handle_types", abi::attributes::mempool_supported_handle_types},
      std::pair{"host_numa_id", abi::attributes::host_numa_id},
  };

  for (int ordinal = 0; ordinal < device_count; ++ordinal) {
    if (options.device_ordinal.has_value() && ordinal != *options.device_ordinal) {
      continue;
    }

    const std::size_t device_diagnostics_begin = diagnostics.size();
    abi::Device device = 0;
    const abi::Result get_result = api_.device_get_(&device, ordinal);
    if (get_result != abi::success) {
      diagnostics.push_back({DiagnosticLevel::warning, "cuda", "cuDeviceGet",
                             result_message(get_result), get_result, ordinal});
      continue;
    }

    probe::DeviceReport output;
    output.ordinal = ordinal;
    std::array<char, 256> name{};
    const abi::Result name_result =
        api_.device_get_name_(name.data(), static_cast<int>(name.size()), device);
    if (name_result == abi::success) {
      output.name = name.data();
    } else {
      output.name = "CUDA device " + std::to_string(ordinal);
      diagnostics.push_back({DiagnosticLevel::info, "cuda", "cuDeviceGetName",
                             result_message(name_result), name_result});
    }

    std::size_t total_memory = 0;
    const abi::Result total_memory_result = api_.device_total_memory_(&total_memory, device);
    if (total_memory_result == abi::success) {
      output.total_memory_bytes = static_cast<std::uint64_t>(total_memory);
    } else {
      diagnostics.push_back({DiagnosticLevel::info, "cuda", "cuDeviceTotalMem_v2",
                             result_message(total_memory_result), total_memory_result});
    }

    if (api_.device_get_uuid_ != nullptr) {
      abi::Uuid uuid{};
      const abi::Result uuid_result = api_.device_get_uuid_(&uuid, device);
      if (uuid_result == abi::success) {
        if (options.include_stable_identifiers) {
          output.uuid = format_uuid(uuid);
        }
      } else {
        diagnostics.push_back({DiagnosticLevel::info, "cuda", "cuDeviceGetUuid",
                               result_message(uuid_result), uuid_result});
      }
    }

    std::optional<platform::AdapterLuid> raw_luid;
    const std::optional<int> tcc_driver =
        query_attribute(device, abi::attributes::tcc_driver, "tcc_driver", diagnostics);
#ifdef _WIN32
    if (tcc_driver.has_value() && *tcc_driver == 0 && api_.device_get_luid_ != nullptr) {
      platform::AdapterLuid luid{};
      unsigned int node_mask = 0;
      const abi::Result luid_result =
          api_.device_get_luid_(reinterpret_cast<char*>(luid.data()), &node_mask, device);
      if (luid_result == abi::success) {
        output.node_mask = node_mask;
        const bool meaningful_luid = std::any_of(luid.begin(), luid.end(),
                                                 [](const std::uint8_t byte) { return byte != 0; });
        if (std::has_single_bit(node_mask) && meaningful_luid) {
          if (options.include_stable_identifiers) {
            output.luid = bytes_to_hex(luid.data(), luid.size());
          }
          raw_luid = luid;
        } else {
          diagnostics.push_back(
              {DiagnosticLevel::warning, "cuda", "validate_wddm_luid",
               "CUDA returned an empty LUID or a non-single-node mask; DXGI correlation skipped",
               static_cast<std::int64_t>(node_mask)});
        }
      } else {
        diagnostics.push_back({DiagnosticLevel::info, "cuda", "cuDeviceGetLuid",
                               result_message(luid_result), luid_result});
      }
    } else if (!tcc_driver.has_value()) {
      diagnostics.push_back({DiagnosticLevel::info, "cuda", "gate_dxgi_luid",
                             "CUDA driver mode was unavailable; WDDM LUID correlation skipped",
                             std::nullopt});
    } else if (*tcc_driver != 0) {
      diagnostics.push_back({DiagnosticLevel::info, "cuda", "gate_dxgi_luid",
                             "The device is in TCC mode; WDDM LUID correlation skipped",
                             std::nullopt});
    } else {
      diagnostics.push_back({DiagnosticLevel::info, "cuda", "resolve_cuDeviceGetLuid",
                             "cuDeviceGetLuid is not exported; DXGI correlation skipped",
                             std::nullopt});
    }
#endif

    if (api_.device_get_pci_bus_id_ != nullptr) {
      std::array<char, 32> pci{};
      const abi::Result pci_result =
          api_.device_get_pci_bus_id_(pci.data(), static_cast<int>(pci.size()), device);
      if (pci_result == abi::success) {
        output.pci_bus_id = pci.data();
      } else {
        diagnostics.push_back({DiagnosticLevel::info, "cuda", "cuDeviceGetPCIBusId",
                               result_message(pci_result), pci_result});
      }
    }

    for (const auto& [key, attribute] : capability_attributes) {
      const std::optional<int> value = query_attribute(device, attribute, key, diagnostics);
      output.capabilities.emplace(key, value.has_value() ? std::optional<bool>(*value != 0)
                                                         : std::nullopt);
    }
    for (const auto& [key, attribute] : numeric_attributes) {
      const std::optional<int> value = std::string_view(key) == "tcc_driver"
                                           ? tcc_driver
                                           : query_attribute(device, attribute, key, diagnostics);
      output.attributes.emplace(key, value.has_value() ? std::optional<std::int64_t>(*value)
                                                       : std::optional<std::int64_t>{});
    }

    if (!active_cuda_poisoned_) {
      output.free_memory_bytes = query_free_memory(device, diagnostics);
    } else {
      diagnostics.push_back(
          {DiagnosticLevel::warning, "cuda", "active_work_refused",
           "Free-memory context probe was skipped because an earlier active CUDA cleanup failed",
           std::nullopt});
    }

    if (capability_value(output.capabilities, "virtual_memory_management")) {
      output.allocation_granularity.emplace(
          "device", query_granularity(abi::location_device, ordinal, "device", diagnostics));
    }
    if (capability_value(output.capabilities, "host_vmm")) {
      output.allocation_granularity.emplace(
          "host", query_granularity(abi::location_host, 0, "host", diagnostics));
    }
    if (capability_value(output.capabilities, "host_numa_vmm")) {
      const auto host_numa = output.attributes.find("host_numa_id");
      if (host_numa != output.attributes.end() && host_numa->second.has_value() &&
          *host_numa->second >= 0 &&
          *host_numa->second <= std::numeric_limits<std::int32_t>::max()) {
        const auto host_numa_id = static_cast<std::int32_t>(*host_numa->second);
        const std::string key = "host_numa_" + std::to_string(host_numa_id);
        output.allocation_granularity.emplace(
            key, query_granularity(abi::location_host_numa, host_numa_id, key, diagnostics));
      } else {
        diagnostics.push_back(
            {DiagnosticLevel::info, "cuda", "select_host_numa_location",
             "The closest host NUMA node is unavailable; HOST_NUMA granularity skipped",
             std::nullopt});
      }
    }

    if (options.run_vmm_smoke && active_cuda_poisoned_) {
      output.vmm_smoke = probe::VmmSmokeResult{
          "skipped",    0,
          std::nullopt, std::nullopt,
          std::nullopt, "VMM smoke test refused because an earlier active CUDA cleanup failed"};
    } else if (options.run_vmm_smoke) {
      const auto granularity = output.allocation_granularity.find("device");
      if (granularity != output.allocation_granularity.end() &&
          granularity->second.minimum_bytes.has_value()) {
        output.vmm_smoke =
            smoke_test_vmm(device, ordinal, *granularity->second.minimum_bytes, diagnostics);
      } else {
        output.vmm_smoke = probe::VmmSmokeResult{
            "skipped",
            0,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            "Device VMM is unavailable or allocation granularity could not be queried"};
      }
    }

    if (raw_luid.has_value()) {
      output.dxgi =
          platform::query_dxgi_memory(*raw_luid, output.node_mask.value_or(0), diagnostics);
      if (output.dxgi.has_value() && !options.include_stable_identifiers) {
        output.dxgi->luid.clear();
      }
    }

    for (std::size_t index = device_diagnostics_begin; index < diagnostics.size(); ++index) {
      if (!diagnostics[index].device_ordinal.has_value()) {
        diagnostics[index].device_ordinal = ordinal;
      }
    }
    report.devices.push_back(std::move(output));
  }

  if (options.device_ordinal.has_value() && report.devices.empty()) {
    diagnostics.push_back({DiagnosticLevel::error, "cuda", "select_device",
                           "Requested CUDA device ordinal does not exist", *options.device_ordinal,
                           *options.device_ordinal});
  }
}

probe::TransferMeasurement Driver::benchmark_transfers(const std::int32_t ordinal,
                                                       const probe::ProbeOptions& options,
                                                       std::vector<Diagnostic>& diagnostics) {
  probe::TransferMeasurement measurement;
  measurement.status = "failed";
  measurement.device_ordinal = ordinal;
  measurement.iterations = options.benchmark_iterations;

  if (active_cuda_poisoned_) {
    measurement.message =
        "Transfer benchmark refused because an earlier active CUDA cleanup failed";
    diagnostics.push_back({DiagnosticLevel::error, "benchmark", "active_work_refused",
                           *measurement.message, std::nullopt});
    return measurement;
  }

  if (!load(diagnostics) || api_.init_(0) != abi::success) {
    measurement.message = "CUDA driver is not available";
    return measurement;
  }
  if (api_.mem_alloc_ == nullptr || api_.mem_free_ == nullptr || api_.mem_host_alloc_ == nullptr ||
      api_.mem_free_host_ == nullptr || api_.memcpy_h2d_async_ == nullptr ||
      api_.memcpy_d2h_async_ == nullptr || api_.stream_create_ == nullptr ||
      api_.stream_destroy_ == nullptr || api_.stream_synchronize_ == nullptr ||
      api_.event_create_ == nullptr || api_.event_destroy_ == nullptr ||
      api_.event_record_ == nullptr || api_.event_synchronize_ == nullptr ||
      api_.event_elapsed_time_ == nullptr || api_.context_create_ == nullptr ||
      api_.context_destroy_ == nullptr || api_.context_get_current_ == nullptr ||
      api_.context_set_current_ == nullptr || api_.mem_get_info_ == nullptr) {
    measurement.message = "CUDA driver is missing transfer benchmark entry points";
    return measurement;
  }

  abi::Device device = 0;
  abi::Result result = api_.device_get_(&device, ordinal);
  if (result != abi::success) {
    measurement.message = result_name(result) + ": " + result_message(result);
    return measurement;
  }

  abi::Context previous = nullptr;
  abi::Context context = nullptr;
  if (api_.context_get_current_(&previous) != abi::success ||
      api_.context_create_(&context, CU_CTX_SCHED_AUTO, device) != abi::success) {
    measurement.message = "Could not create an isolated CUDA benchmark context";
    return measurement;
  }

  const auto cleanup_context = [&] {
    abi::Result destroy_result = abi::success;
    if (context != nullptr) {
      destroy_result = api_.context_destroy_(context);
      if (destroy_result == abi::success) {
        context = nullptr;
      }
    }
    const abi::Result restore_result = api_.context_set_current_(previous);
    if (destroy_result != abi::success || restore_result != abi::success) {
      const abi::Result cleanup_result =
          destroy_result != abi::success ? destroy_result : restore_result;
      diagnostics.push_back({DiagnosticLevel::warning, "benchmark", "cleanup_context",
                             "CUDA benchmark context cleanup reported an error", cleanup_result});
      return false;
    }
    return true;
  };

  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  result = api_.mem_get_info_(&free_bytes, &total_bytes);
  if (result != abi::success) {
    cleanup_context();
    measurement.message = result_name(result) + ": " + result_message(result);
    return measurement;
  }

  constexpr std::uint64_t minimum_benchmark_bytes = 8ULL * 1024ULL * 1024ULL;
  const std::uint64_t safe_device_limit = static_cast<std::uint64_t>(free_bytes) / 4ULL;
  const std::uint64_t requested = std::min<std::uint64_t>(
      options.benchmark_bytes, static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()));
  const std::uint64_t bytes64 = std::min(requested, safe_device_limit);
  if (bytes64 < minimum_benchmark_bytes) {
    cleanup_context();
    measurement.status = "skipped";
    measurement.message = "Insufficient free VRAM for a safe transfer benchmark";
    return measurement;
  }
  const std::size_t bytes = static_cast<std::size_t>(bytes64);
  measurement.bytes_per_direction = bytes64;

  abi::DevicePointer device_a = 0;
  abi::DevicePointer device_b = 0;
  void* host_a = nullptr;
  void* host_b = nullptr;
  abi::Stream h2d_stream = nullptr;
  abi::Stream d2h_stream = nullptr;
  abi::Event start_event = nullptr;
  abi::Event stop_event = nullptr;

  const auto cleanup = [&] {
    bool complete = true;
    const auto record_cleanup_error = [&](const char* operation, const abi::Result error) {
      if (error == abi::success) {
        return true;
      }
      complete = false;
      diagnostics.push_back(
          {DiagnosticLevel::warning, "benchmark", operation, result_message(error), error});
      return false;
    };

    bool streams_idle = true;
    if (h2d_stream != nullptr) {
      streams_idle = record_cleanup_error("cuStreamSynchronize(h2d_cleanup)",
                                          api_.stream_synchronize_(h2d_stream)) &&
                     streams_idle;
    }
    if (d2h_stream != nullptr) {
      streams_idle = record_cleanup_error("cuStreamSynchronize(d2h_cleanup)",
                                          api_.stream_synchronize_(d2h_stream)) &&
                     streams_idle;
    }

    if (streams_idle) {
      if (stop_event != nullptr &&
          record_cleanup_error("cuEventDestroy(stop)", api_.event_destroy_(stop_event))) {
        stop_event = nullptr;
      }
      if (start_event != nullptr &&
          record_cleanup_error("cuEventDestroy(start)", api_.event_destroy_(start_event))) {
        start_event = nullptr;
      }
      if (d2h_stream != nullptr &&
          record_cleanup_error("cuStreamDestroy(d2h)", api_.stream_destroy_(d2h_stream))) {
        d2h_stream = nullptr;
      }
      if (h2d_stream != nullptr &&
          record_cleanup_error("cuStreamDestroy(h2d)", api_.stream_destroy_(h2d_stream))) {
        h2d_stream = nullptr;
      }
      if (host_b != nullptr &&
          record_cleanup_error("cuMemFreeHost(host_b)", api_.mem_free_host_(host_b))) {
        host_b = nullptr;
      }
      if (host_a != nullptr &&
          record_cleanup_error("cuMemFreeHost(host_a)", api_.mem_free_host_(host_a))) {
        host_a = nullptr;
      }
      if (device_b != 0 && record_cleanup_error("cuMemFree(device_b)", api_.mem_free_(device_b))) {
        device_b = 0;
      }
      if (device_a != 0 && record_cleanup_error("cuMemFree(device_a)", api_.mem_free_(device_a))) {
        device_a = 0;
      }
    }

    const bool context_cleanup_complete = cleanup_context();
    complete = context_cleanup_complete && complete;
    if (!streams_idle) {
      if (context == nullptr) {
        start_event = nullptr;
        stop_event = nullptr;
        h2d_stream = nullptr;
        d2h_stream = nullptr;
        device_a = 0;
        device_b = 0;
        if (host_b != nullptr && record_cleanup_error("cuMemFreeHost(host_b_after_context)",
                                                      api_.mem_free_host_(host_b))) {
          host_b = nullptr;
        }
        if (host_a != nullptr && record_cleanup_error("cuMemFreeHost(host_a_after_context)",
                                                      api_.mem_free_host_(host_a))) {
          host_a = nullptr;
        }
      }
      diagnostics.push_back(
          {DiagnosticLevel::warning, "benchmark", "cleanup_after_stream_error",
           "The isolated context was destroyed before releasing transfer resources because "
           "stream synchronization failed",
           std::nullopt});
      complete = false;
    }
    return complete;
  };

  const auto fail = [&](const char* operation, const abi::Result error) {
    measurement.message =
        std::string(operation) + ": " + result_name(error) + ": " + result_message(error);
    diagnostics.push_back(
        {DiagnosticLevel::warning, "benchmark", operation, result_message(error), error});
  };

  if ((result = api_.mem_alloc_(&device_a, bytes)) != abi::success ||
      (result = api_.mem_alloc_(&device_b, bytes)) != abi::success ||
      (result = api_.mem_host_alloc_(&host_a, bytes, 0)) != abi::success ||
      (result = api_.mem_host_alloc_(&host_b, bytes, 0)) != abi::success ||
      (result = api_.stream_create_(&h2d_stream, CU_STREAM_NON_BLOCKING)) != abi::success ||
      (result = api_.stream_create_(&d2h_stream, CU_STREAM_NON_BLOCKING)) != abi::success ||
      (result = api_.event_create_(&start_event, 0U)) != abi::success ||
      (result = api_.event_create_(&stop_event, 0U)) != abi::success) {
    fail("allocate_benchmark_resources", result);
    cleanup();
    return measurement;
  }

  std::memset(host_a, 0xA5, bytes);
  std::memset(host_b, 0, bytes);

  if ((result = api_.memcpy_h2d_async_(device_a, host_a, bytes, h2d_stream)) != abi::success ||
      (result = api_.memcpy_h2d_async_(device_b, host_a, bytes, h2d_stream)) != abi::success ||
      (result = api_.stream_synchronize_(h2d_stream)) != abi::success) {
    fail("warm_up_transfers", result);
    cleanup();
    return measurement;
  }

  const auto measure_one_direction = [&](const bool host_to_device) -> std::optional<double> {
    abi::Result call_result =
        api_.event_record_(start_event, host_to_device ? h2d_stream : d2h_stream);
    for (std::uint32_t index = 0;
         call_result == abi::success && index < options.benchmark_iterations; ++index) {
      call_result = host_to_device ? api_.memcpy_h2d_async_(device_a, host_a, bytes, h2d_stream)
                                   : api_.memcpy_d2h_async_(host_b, device_b, bytes, d2h_stream);
    }
    if (call_result == abi::success) {
      call_result = api_.event_record_(stop_event, host_to_device ? h2d_stream : d2h_stream);
    }
    if (call_result == abi::success) {
      call_result = api_.event_synchronize_(stop_event);
    }
    float milliseconds = 0.0F;
    if (call_result == abi::success) {
      call_result = api_.event_elapsed_time_(&milliseconds, start_event, stop_event);
    }
    if (call_result != abi::success || milliseconds <= 0.0F) {
      return std::nullopt;
    }
    const double gib =
        static_cast<double>(bytes64) * options.benchmark_iterations / (1024.0 * 1024.0 * 1024.0);
    return gib / (static_cast<double>(milliseconds) / 1000.0);
  };

  measurement.h2d_gib_per_second = measure_one_direction(true);
  measurement.d2h_gib_per_second = measure_one_direction(false);
  if (!measurement.h2d_gib_per_second.has_value() || !measurement.d2h_gib_per_second.has_value()) {
    measurement.message = "A timed unidirectional transfer failed";
    cleanup();
    return measurement;
  }

  api_.stream_synchronize_(h2d_stream);
  api_.stream_synchronize_(d2h_stream);
  const auto duplex_start = std::chrono::steady_clock::now();
  for (std::uint32_t index = 0; index < options.benchmark_iterations; ++index) {
    result = api_.memcpy_h2d_async_(device_a, host_a, bytes, h2d_stream);
    if (result == abi::success) {
      result = api_.memcpy_d2h_async_(host_b, device_b, bytes, d2h_stream);
    }
    if (result != abi::success) {
      break;
    }
  }
  if (result == abi::success) {
    result = api_.stream_synchronize_(h2d_stream);
  }
  if (result == abi::success) {
    result = api_.stream_synchronize_(d2h_stream);
  }
  const auto duplex_stop = std::chrono::steady_clock::now();
  if (result != abi::success) {
    fail("full_duplex_transfer", result);
    cleanup();
    return measurement;
  }

  const double duplex_seconds = std::chrono::duration<double>(duplex_stop - duplex_start).count();
  if (duplex_seconds <= 0.0) {
    measurement.message = "Full-duplex transfer timing was not positive";
    diagnostics.push_back({DiagnosticLevel::error, "benchmark", "time_full_duplex",
                           *measurement.message, std::nullopt});
    cleanup();
    return measurement;
  }
  const double aggregate_gib = 2.0 * static_cast<double>(bytes64) * options.benchmark_iterations /
                               (1024.0 * 1024.0 * 1024.0);
  measurement.full_duplex_aggregate_gib_per_second = aggregate_gib / duplex_seconds;
  if (std::memcmp(host_a, host_b, bytes) != 0) {
    measurement.message = "D2H/full-duplex benchmark data verification failed";
    diagnostics.push_back({DiagnosticLevel::error, "benchmark", "verify_transfer_data",
                           *measurement.message, std::nullopt});
    cleanup();
    return measurement;
  }
  std::memset(host_b, 0, bytes);
  result = api_.memcpy_d2h_async_(host_b, device_a, bytes, d2h_stream);
  if (result == abi::success) {
    result = api_.stream_synchronize_(d2h_stream);
  }
  if (result != abi::success) {
    fail("verify_h2d_round_trip", result);
    cleanup();
    return measurement;
  }
  if (std::memcmp(host_a, host_b, bytes) != 0) {
    measurement.message = "H2D round-trip benchmark data verification failed";
    diagnostics.push_back({DiagnosticLevel::error, "benchmark", "verify_h2d_round_trip",
                           *measurement.message, std::nullopt});
    cleanup();
    return measurement;
  }
  measurement.status = "completed";
  if (!cleanup()) {
    measurement.status = "failed";
    measurement.message = "Transfer completed, but CUDA benchmark context cleanup failed";
  }
  return measurement;
}

probe::OverlapMeasurement Driver::benchmark_overlap(const std::int32_t ordinal,
                                                    const probe::ProbeOptions& options,
                                                    std::vector<Diagnostic>& diagnostics) {
  probe::OverlapMeasurement measurement;
  measurement.status = "failed";
  measurement.device_ordinal = ordinal;
  measurement.module_version = synthetic_overlap::module_version;
  measurement.module_sha256 = std::string(synthetic_overlap::module_sha256);
  measurement.samples = options.overlap_samples;
  measurement.target_compute_ms = options.overlap_target_compute_ms;

  const auto diagnose = [&](const DiagnosticLevel level, const char* operation,
                            const std::string& message,
                            const std::optional<abi::Result> code = std::nullopt) {
    diagnostics.push_back({level, "overlap", operation, message,
                           code.has_value()
                               ? std::optional<std::int64_t>(static_cast<std::int64_t>(*code))
                               : std::nullopt});
  };

  if (active_cuda_poisoned_) {
    measurement.message = "Overlap benchmark refused because earlier active CUDA cleanup failed";
    diagnose(DiagnosticLevel::error, "active_work_refused", *measurement.message);
    return measurement;
  }
  if (!load(diagnostics) || api_.init_(0) != abi::success) {
    measurement.message = "CUDA driver is not available";
    return measurement;
  }
  if (api_.mem_alloc_ == nullptr || api_.mem_free_ == nullptr || api_.mem_host_alloc_ == nullptr ||
      api_.mem_free_host_ == nullptr || api_.memcpy_h2d_async_ == nullptr ||
      api_.memcpy_d2h_async_ == nullptr || api_.stream_create_ == nullptr ||
      api_.stream_destroy_ == nullptr || api_.stream_synchronize_ == nullptr ||
      api_.stream_wait_event_ == nullptr || api_.event_create_ == nullptr ||
      api_.event_destroy_ == nullptr || api_.event_record_ == nullptr ||
      api_.event_synchronize_ == nullptr || api_.event_elapsed_time_ == nullptr ||
      api_.module_load_data_ == nullptr || api_.module_get_function_ == nullptr ||
      api_.module_unload_ == nullptr || api_.launch_kernel_ == nullptr ||
      api_.context_create_ == nullptr || api_.context_destroy_ == nullptr ||
      api_.context_get_current_ == nullptr || api_.context_set_current_ == nullptr ||
      api_.device_get_attribute_ == nullptr || api_.mem_get_info_ == nullptr) {
    measurement.message = "CUDA driver is missing overlap benchmark entry points";
    return measurement;
  }

  abi::Device device = 0;
  abi::Result result = api_.device_get_(&device, ordinal);
  if (result != abi::success) {
    measurement.message = result_name(result) + ": " + result_message(result);
    return measurement;
  }

  int multiprocessor_count = 0;
  int maximum_threads_per_block = 0;
  int warp_size = 0;
  if ((result = api_.device_get_attribute_(
           &multiprocessor_count, abi::attributes::multiprocessor_count, device)) != abi::success ||
      (result = api_.device_get_attribute_(&maximum_threads_per_block,
                                           abi::attributes::max_threads_per_block, device)) !=
          abi::success ||
      (result = api_.device_get_attribute_(&warp_size, abi::attributes::warp_size, device)) !=
          abi::success ||
      multiprocessor_count <= 0 || maximum_threads_per_block <= 0 || warp_size <= 0) {
    measurement.message = result == abi::success
                              ? "Invalid CUDA launch limits for the overlap workload"
                              : result_name(result) + ": " + result_message(result);
    return measurement;
  }

  const int bounded_threads = std::min(maximum_threads_per_block, 256);
  const int rounded_threads = (bounded_threads / warp_size) * warp_size;
  if (rounded_threads <= 0 ||
      multiprocessor_count > static_cast<int>(std::numeric_limits<std::uint32_t>::max() / 4U)) {
    measurement.message = "CUDA launch geometry cannot be represented safely";
    return measurement;
  }
  measurement.block_threads = static_cast<std::uint32_t>(rounded_threads);
  measurement.grid_blocks = static_cast<std::uint32_t>(multiprocessor_count) * 4U;

  abi::Context previous = nullptr;
  abi::Context context = nullptr;
  bool previous_captured = false;
  result = api_.context_get_current_(&previous);
  if (result == abi::success) {
    previous_captured = true;
    result = api_.context_create_(&context, CU_CTX_SCHED_AUTO, device);
  }
  if (result != abi::success) {
    measurement.message = previous_captured ? "Could not create an isolated overlap context"
                                            : "Could not capture the current CUDA context";
    return measurement;
  }

  const auto cleanup_context = [&]() {
    bool complete = true;
    if (context != nullptr) {
      const abi::Result destroy_result = api_.context_destroy_(context);
      if (destroy_result == abi::success) {
        context = nullptr;
      } else {
        complete = false;
        diagnose(DiagnosticLevel::warning, "cuCtxDestroy(cleanup)", result_message(destroy_result),
                 destroy_result);
      }
    }
    if (previous_captured) {
      const abi::Result restore_result = api_.context_set_current_(previous);
      if (restore_result != abi::success) {
        complete = false;
        diagnose(DiagnosticLevel::warning, "cuCtxSetCurrent(restore)",
                 result_message(restore_result), restore_result);
      }
    }
    return complete;
  };

  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  result = api_.mem_get_info_(&free_bytes, &total_bytes);
  if (result != abi::success) {
    measurement.message = result_name(result) + ": " + result_message(result);
    measurement.cleanup_complete = cleanup_context();
    return measurement;
  }

  const std::uint64_t thread_count =
      static_cast<std::uint64_t>(measurement.grid_blocks) * measurement.block_threads;
  if (thread_count > std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t)) {
    measurement.message = "Synthetic output allocation exceeds the host size type";
    measurement.cleanup_complete = cleanup_context();
    return measurement;
  }
  const std::size_t output_bytes = static_cast<std::size_t>(thread_count) * sizeof(std::uint32_t);
  constexpr std::uint64_t minimum_overlap_bytes = 8ULL * 1024ULL * 1024ULL;
  constexpr std::uint64_t device_headroom_bytes = 32ULL * 1024ULL * 1024ULL;
  const std::uint64_t free64 = static_cast<std::uint64_t>(free_bytes);
  const std::uint64_t reserved = static_cast<std::uint64_t>(output_bytes) + device_headroom_bytes;
  const std::uint64_t safe_device_limit = free64 > reserved ? (free64 - reserved) / 4ULL : 0ULL;
  const std::uint64_t requested = std::min(
      options.overlap_bytes, static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()));
  const std::uint64_t bytes64 = std::min(requested, safe_device_limit);
  if (bytes64 < minimum_overlap_bytes) {
    measurement.status = "skipped";
    measurement.message = "Insufficient free VRAM for a safe overlap benchmark";
    measurement.cleanup_complete = cleanup_context();
    return measurement;
  }
  const std::size_t bytes = static_cast<std::size_t>(bytes64);
  measurement.bytes_per_copy = bytes64;

  abi::DevicePointer transfer_buffer = 0;
  abi::DevicePointer compute_output = 0;
  void* host_source = nullptr;
  void* host_destination = nullptr;
  abi::Stream control_stream = nullptr;
  abi::Stream compute_stream = nullptr;
  abi::Stream copy_stream = nullptr;
  abi::Event start_event = nullptr;
  abi::Event stop_event = nullptr;
  abi::Event compute_done_event = nullptr;
  abi::Event copy_done_event = nullptr;
  abi::Module module = nullptr;
  abi::Function function = nullptr;

  const auto cleanup = [&]() {
    bool complete = true;
    const auto record_cleanup_error = [&](const char* operation, const abi::Result error) {
      if (error == abi::success) {
        return true;
      }
      complete = false;
      diagnose(DiagnosticLevel::warning, operation, result_message(error), error);
      return false;
    };

    bool streams_idle = true;
    for (const auto& [name, stream] : std::array<std::pair<const char*, abi::Stream>, 3>{
             std::pair{"cuStreamSynchronize(control_cleanup)", control_stream},
             std::pair{"cuStreamSynchronize(compute_cleanup)", compute_stream},
             std::pair{"cuStreamSynchronize(copy_cleanup)", copy_stream}}) {
      if (stream != nullptr) {
        streams_idle = record_cleanup_error(name, api_.stream_synchronize_(stream)) && streams_idle;
      }
    }

    if (streams_idle) {
      for (auto& [name, event] : std::array<std::pair<const char*, abi::Event*>, 4>{
               std::pair{"cuEventDestroy(copy_done)", &copy_done_event},
               std::pair{"cuEventDestroy(compute_done)", &compute_done_event},
               std::pair{"cuEventDestroy(stop)", &stop_event},
               std::pair{"cuEventDestroy(start)", &start_event}}) {
        if (*event != nullptr && record_cleanup_error(name, api_.event_destroy_(*event))) {
          *event = nullptr;
        }
      }
      for (auto& [name, stream] : std::array<std::pair<const char*, abi::Stream*>, 3>{
               std::pair{"cuStreamDestroy(copy)", &copy_stream},
               std::pair{"cuStreamDestroy(compute)", &compute_stream},
               std::pair{"cuStreamDestroy(control)", &control_stream}}) {
        if (*stream != nullptr && record_cleanup_error(name, api_.stream_destroy_(*stream))) {
          *stream = nullptr;
        }
      }
      if (module != nullptr &&
          record_cleanup_error("cuModuleUnload", api_.module_unload_(module))) {
        module = nullptr;
        function = nullptr;
      }
      if (compute_output != 0 &&
          record_cleanup_error("cuMemFree(compute_output)", api_.mem_free_(compute_output))) {
        compute_output = 0;
      }
      if (transfer_buffer != 0 &&
          record_cleanup_error("cuMemFree(transfer_buffer)", api_.mem_free_(transfer_buffer))) {
        transfer_buffer = 0;
      }
      if (host_destination != nullptr &&
          record_cleanup_error("cuMemFreeHost(destination)",
                               api_.mem_free_host_(host_destination))) {
        host_destination = nullptr;
      }
      if (host_source != nullptr &&
          record_cleanup_error("cuMemFreeHost(source)", api_.mem_free_host_(host_source))) {
        host_source = nullptr;
      }
    }

    complete = cleanup_context() && complete;
    if (!streams_idle) {
      if (context == nullptr) {
        control_stream = nullptr;
        compute_stream = nullptr;
        copy_stream = nullptr;
        start_event = nullptr;
        stop_event = nullptr;
        compute_done_event = nullptr;
        copy_done_event = nullptr;
        module = nullptr;
        function = nullptr;
        transfer_buffer = 0;
        compute_output = 0;
        if (host_destination != nullptr &&
            record_cleanup_error("cuMemFreeHost(destination_after_context)",
                                 api_.mem_free_host_(host_destination))) {
          host_destination = nullptr;
        }
        if (host_source != nullptr && record_cleanup_error("cuMemFreeHost(source_after_context)",
                                                           api_.mem_free_host_(host_source))) {
          host_source = nullptr;
        }
      }
      diagnose(DiagnosticLevel::warning, "cleanup_after_stream_error",
               "The isolated overlap context was destroyed before releasing context-owned "
               "resources because stream synchronization failed");
      complete = false;
    }
    if (!complete) {
      active_cuda_poisoned_ = true;
    }
    return complete;
  };

  const auto fail = [&](const char* operation, const abi::Result error) {
    measurement.message =
        std::string(operation) + ": " + result_name(error) + ": " + result_message(error);
    diagnose(DiagnosticLevel::warning, operation, result_message(error), error);
  };

  if ((result = api_.mem_alloc_(&transfer_buffer, bytes)) != abi::success ||
      (result = api_.mem_alloc_(&compute_output, output_bytes)) != abi::success ||
      (result = api_.mem_host_alloc_(&host_source, bytes, 0)) != abi::success ||
      (result = api_.mem_host_alloc_(&host_destination, bytes, 0)) != abi::success ||
      (result = api_.stream_create_(&control_stream, CU_STREAM_NON_BLOCKING)) != abi::success ||
      (result = api_.stream_create_(&compute_stream, CU_STREAM_NON_BLOCKING)) != abi::success ||
      (result = api_.stream_create_(&copy_stream, CU_STREAM_NON_BLOCKING)) != abi::success ||
      (result = api_.event_create_(&start_event, 0U)) != abi::success ||
      (result = api_.event_create_(&stop_event, 0U)) != abi::success ||
      (result = api_.event_create_(&compute_done_event, 0U)) != abi::success ||
      (result = api_.event_create_(&copy_done_event, 0U)) != abi::success ||
      (result = api_.module_load_data_(&module, synthetic_overlap::ptx)) != abi::success ||
      (result = api_.module_get_function_(&function, module,
                                          synthetic_overlap::kernel_name.data())) != abi::success) {
    fail("prepare_overlap_resources", result);
    measurement.cleanup_complete = cleanup();
    return measurement;
  }

  auto* source_bytes = static_cast<std::uint8_t*>(host_source);
  for (std::size_t index = 0; index < bytes; ++index) {
    source_bytes[index] = static_cast<std::uint8_t>((index * 29U + 0x5BU) & 0xFFU);
  }
  std::memset(host_destination, 0, bytes);
  result = api_.memcpy_h2d_async_(transfer_buffer, host_source, bytes, copy_stream);
  if (result == abi::success) {
    result = api_.stream_synchronize_(copy_stream);
  }
  if (result != abi::success) {
    fail("initialize_transfer_buffer", result);
    measurement.cleanup_complete = cleanup();
    return measurement;
  }

  const auto launch_compute = [&](const std::uint32_t iterations,
                                  const abi::Stream stream) -> abi::Result {
    abi::DevicePointer output_argument = compute_output;
    std::uint32_t iteration_argument = iterations;
    void* parameters[] = {&output_argument, &iteration_argument};
    return api_.launch_kernel_(function, measurement.grid_blocks, 1U, 1U, measurement.block_threads,
                               1U, 1U, 0U, stream, parameters, nullptr);
  };

  const auto timed_stream_sample = [&](const char* operation, const abi::Stream stream,
                                       auto&& enqueue) -> std::optional<double> {
    abi::Result call_result = api_.event_record_(start_event, stream);
    if (call_result == abi::success) {
      call_result = enqueue();
    }
    if (call_result == abi::success) {
      call_result = api_.event_record_(stop_event, stream);
    }
    if (call_result == abi::success) {
      call_result = api_.event_synchronize_(stop_event);
    }
    float milliseconds = 0.0F;
    if (call_result == abi::success) {
      call_result = api_.event_elapsed_time_(&milliseconds, start_event, stop_event);
    }
    if (call_result != abi::success) {
      fail(operation, call_result);
      return std::nullopt;
    }
    if (!(milliseconds > 0.0F) || !std::isfinite(milliseconds)) {
      measurement.message = std::string(operation) + " produced a non-positive timing";
      diagnose(DiagnosticLevel::error, operation, *measurement.message);
      return std::nullopt;
    }
    return static_cast<double>(milliseconds);
  };

  const auto median_samples = [&](auto&& sample) -> std::optional<double> {
    std::vector<double> values;
    values.reserve(options.overlap_samples);
    for (std::uint32_t index = 0; index < options.overlap_samples; ++index) {
      const std::optional<double> value = sample(index);
      if (!value.has_value()) {
        return std::nullopt;
      }
      values.push_back(*value);
    }
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2U;
    if ((values.size() & 1U) != 0U) {
      return values[middle];
    }
    return (values[middle - 1U] + values[middle]) / 2.0;
  };

  constexpr std::uint32_t minimum_kernel_iterations = 256U;
  constexpr std::uint32_t maximum_kernel_iterations = 50'000'000U;
  std::uint32_t kernel_iterations = 16'384U;
  for (std::uint32_t attempt = 0; attempt < 8U; ++attempt) {
    const std::optional<double> elapsed =
        timed_stream_sample("calibrate_compute", compute_stream,
                            [&] { return launch_compute(kernel_iterations, compute_stream); });
    if (!elapsed.has_value()) {
      measurement.cleanup_complete = cleanup();
      return measurement;
    }
    const double target = static_cast<double>(options.overlap_target_compute_ms);
    if (*elapsed >= target * 0.85 && *elapsed <= target * 1.15) {
      break;
    }
    const double scale = std::clamp(target / *elapsed, 0.25, 8.0);
    const double scaled = static_cast<double>(kernel_iterations) * scale;
    const auto next = static_cast<std::uint32_t>(
        std::clamp(std::llround(scaled), static_cast<long long>(minimum_kernel_iterations),
                   static_cast<long long>(maximum_kernel_iterations)));
    if (next == kernel_iterations) {
      break;
    }
    kernel_iterations = next;
  }
  measurement.kernel_iterations = kernel_iterations;

  measurement.compute_only_ms = median_samples([&](const std::uint32_t) {
    return timed_stream_sample("measure_compute", compute_stream,
                               [&] { return launch_compute(kernel_iterations, compute_stream); });
  });
  if (!measurement.compute_only_ms.has_value()) {
    measurement.cleanup_complete = cleanup();
    return measurement;
  }

  std::uint32_t observed = 0;
  result = api_.memcpy_d2h_async_(host_destination, compute_output, sizeof(observed), copy_stream);
  if (result == abi::success) {
    result = api_.stream_synchronize_(copy_stream);
  }
  if (result != abi::success) {
    fail("read_compute_verification", result);
    measurement.cleanup_complete = cleanup();
    return measurement;
  }
  std::memcpy(&observed, host_destination, sizeof(observed));
  std::uint32_t expected = 0x9E3779B9U;
  for (std::uint32_t index = 0; index < kernel_iterations; ++index) {
    expected = expected * 1'664'525U + 1'013'904'223U;
  }
  measurement.compute_verified = observed == expected;
  if (!*measurement.compute_verified) {
    measurement.message = "Synthetic compute result verification failed";
    diagnose(DiagnosticLevel::error, "verify_compute", *measurement.message);
    measurement.cleanup_complete = cleanup();
    return measurement;
  }

  const auto measure_copy = [&](const bool host_to_device,
                                const std::uint32_t repetitions) -> std::optional<double> {
    return median_samples([&](const std::uint32_t) {
      return timed_stream_sample(
          host_to_device ? "measure_h2d" : "measure_d2h", copy_stream, [&]() -> abi::Result {
            abi::Result copy_result = abi::success;
            for (std::uint32_t index = 0; copy_result == abi::success && index < repetitions;
                 ++index) {
              copy_result =
                  host_to_device
                      ? api_.memcpy_h2d_async_(transfer_buffer, host_source, bytes, copy_stream)
                      : api_.memcpy_d2h_async_(host_destination, transfer_buffer, bytes,
                                               copy_stream);
            }
            return copy_result;
          });
    });
  };

  const std::optional<double> h2d_single = measure_copy(true, 1U);
  const std::optional<double> d2h_single = measure_copy(false, 1U);
  if (!h2d_single.has_value() || !d2h_single.has_value()) {
    measurement.cleanup_complete = cleanup();
    return measurement;
  }
  const auto balanced_repetitions = [&](const double single_copy_ms) {
    const double ratio = *measurement.compute_only_ms / single_copy_ms;
    return static_cast<std::uint32_t>(
        std::clamp(std::llround(ratio), 1LL, static_cast<long long>(64U)));
  };
  const std::uint32_t h2d_repetitions = balanced_repetitions(*h2d_single);
  const std::uint32_t d2h_repetitions = balanced_repetitions(*d2h_single);
  const std::optional<double> h2d_copy_ms = measure_copy(true, h2d_repetitions);
  const std::optional<double> d2h_copy_ms = measure_copy(false, d2h_repetitions);
  if (!h2d_copy_ms.has_value() || !d2h_copy_ms.has_value()) {
    measurement.cleanup_complete = cleanup();
    return measurement;
  }

  const auto concurrent_sample = [&](const bool host_to_device, const std::uint32_t repetitions,
                                     const std::uint32_t sample_index) -> std::optional<double> {
    abi::Result call_result = api_.event_record_(start_event, control_stream);
    if (call_result == abi::success) {
      call_result = api_.stream_wait_event_(compute_stream, start_event, 0U);
    }
    if (call_result == abi::success) {
      call_result = api_.stream_wait_event_(copy_stream, start_event, 0U);
    }

    const auto enqueue_compute = [&]() {
      if (call_result == abi::success) {
        call_result = launch_compute(kernel_iterations, compute_stream);
      }
      if (call_result == abi::success) {
        call_result = api_.event_record_(compute_done_event, compute_stream);
      }
    };
    const auto enqueue_copy = [&]() {
      for (std::uint32_t index = 0; call_result == abi::success && index < repetitions; ++index) {
        call_result =
            host_to_device
                ? api_.memcpy_h2d_async_(transfer_buffer, host_source, bytes, copy_stream)
                : api_.memcpy_d2h_async_(host_destination, transfer_buffer, bytes, copy_stream);
      }
      if (call_result == abi::success) {
        call_result = api_.event_record_(copy_done_event, copy_stream);
      }
    };

    if ((sample_index & 1U) == 0U) {
      enqueue_compute();
      enqueue_copy();
    } else {
      enqueue_copy();
      enqueue_compute();
    }
    if (call_result == abi::success) {
      call_result = api_.stream_wait_event_(control_stream, compute_done_event, 0U);
    }
    if (call_result == abi::success) {
      call_result = api_.stream_wait_event_(control_stream, copy_done_event, 0U);
    }
    if (call_result == abi::success) {
      call_result = api_.event_record_(stop_event, control_stream);
    }
    if (call_result == abi::success) {
      call_result = api_.event_synchronize_(stop_event);
    }
    float milliseconds = 0.0F;
    if (call_result == abi::success) {
      call_result = api_.event_elapsed_time_(&milliseconds, start_event, stop_event);
    }
    if (call_result != abi::success) {
      fail(host_to_device ? "measure_h2d_overlap" : "measure_d2h_overlap", call_result);
      return std::nullopt;
    }
    if (!(milliseconds > 0.0F) || !std::isfinite(milliseconds)) {
      measurement.message = "Concurrent overlap timing was not positive";
      diagnose(DiagnosticLevel::error,
               host_to_device ? "measure_h2d_overlap" : "measure_d2h_overlap",
               *measurement.message);
      return std::nullopt;
    }
    return static_cast<double>(milliseconds);
  };

  const std::optional<double> h2d_concurrent = median_samples(
      [&](const std::uint32_t index) { return concurrent_sample(true, h2d_repetitions, index); });
  const std::optional<double> d2h_concurrent = median_samples(
      [&](const std::uint32_t index) { return concurrent_sample(false, d2h_repetitions, index); });
  if (!h2d_concurrent.has_value() || !d2h_concurrent.has_value()) {
    measurement.cleanup_complete = cleanup();
    return measurement;
  }

  const auto direction = [&](const std::uint32_t repetitions, const double copy_only_ms,
                             const double concurrent_ms) {
    probe::OverlapDirectionMeasurement output;
    output.copy_repetitions = repetitions;
    output.copy_only_ms = copy_only_ms;
    output.concurrent_ms = concurrent_ms;
    output.serial_ms = *measurement.compute_only_ms + copy_only_ms;
    output.ideal_ms = std::max(*measurement.compute_only_ms, copy_only_ms);
    output.speedup = *output.serial_ms / concurrent_ms;
    const double overlap_window = std::min(*measurement.compute_only_ms, copy_only_ms);
    output.overlap_efficiency = (*output.serial_ms - concurrent_ms) / overlap_window;
    return output;
  };
  measurement.h2d = direction(h2d_repetitions, *h2d_copy_ms, *h2d_concurrent);
  measurement.d2h = direction(d2h_repetitions, *d2h_copy_ms, *d2h_concurrent);

  std::memset(host_destination, 0, bytes);
  result = api_.memcpy_d2h_async_(host_destination, transfer_buffer, bytes, copy_stream);
  if (result == abi::success) {
    result = api_.stream_synchronize_(copy_stream);
  }
  if (result != abi::success) {
    fail("verify_overlap_transfer", result);
    measurement.cleanup_complete = cleanup();
    return measurement;
  }
  measurement.transfer_verified = std::memcmp(host_source, host_destination, bytes) == 0;
  if (!*measurement.transfer_verified) {
    measurement.message = "Overlap transfer data verification failed";
    diagnose(DiagnosticLevel::error, "verify_overlap_transfer", *measurement.message);
    measurement.cleanup_complete = cleanup();
    return measurement;
  }

  measurement.cleanup_complete = cleanup();
  if (!*measurement.cleanup_complete) {
    measurement.message = "Overlap measurement completed, but cleanup reported an error";
    return measurement;
  }
  measurement.status = "completed";
  measurement.message = "Calibrated synthetic compute overlapped with independent H2D and D2H "
                        "transfer batches";
  return measurement;
}

} // namespace xvram::cuda
