#include "platform/dxgi_memory.hpp"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <bit>
#include <cstring>
#include <string>

namespace xvram::platform {
namespace {

using Microsoft::WRL::ComPtr;

[[nodiscard]] std::string wide_to_utf8(const wchar_t* input) {
  if (input == nullptr || *input == L'\0') {
    return {};
  }
  const int input_length = static_cast<int>(wcslen(input));
  const int output_length =
      WideCharToMultiByte(CP_UTF8, 0, input, input_length, nullptr, 0, nullptr, nullptr);
  if (output_length <= 0) {
    return {};
  }
  std::string output(static_cast<std::size_t>(output_length), '\0');
  WideCharToMultiByte(CP_UTF8, 0, input, input_length, output.data(), output_length, nullptr,
                      nullptr);
  return output;
}

[[nodiscard]] std::uint32_t node_index_from_mask(const std::uint32_t node_mask) {
  return static_cast<std::uint32_t>(std::countr_zero(node_mask));
}

[[nodiscard]] std::optional<probe::VideoMemoryInfo>
query_segment(IDXGIAdapter3& adapter, const std::uint32_t node_index,
              const DXGI_MEMORY_SEGMENT_GROUP group, const char* group_name,
              std::vector<probe::Diagnostic>& diagnostics) {
  DXGI_QUERY_VIDEO_MEMORY_INFO raw{};
  const HRESULT result = adapter.QueryVideoMemoryInfo(node_index, group, &raw);
  if (FAILED(result)) {
    diagnostics.push_back({probe::DiagnosticLevel::warning, "dxgi",
                           std::string("QueryVideoMemoryInfo(") + group_name + ")",
                           "DXGI could not query the process video-memory budget",
                           static_cast<std::int64_t>(result)});
    return std::nullopt;
  }
  return probe::VideoMemoryInfo{raw.Budget, raw.CurrentUsage, raw.AvailableForReservation,
                                raw.CurrentReservation};
}

} // namespace

std::optional<probe::DxgiAdapterInfo>
query_dxgi_memory(const AdapterLuid& luid, const std::uint32_t node_mask,
                  std::vector<probe::Diagnostic>& diagnostics) {
  if (!std::has_single_bit(node_mask)) {
    diagnostics.push_back(
        {probe::DiagnosticLevel::warning, "dxgi", "validate_node_mask",
         "DXGI video-memory queries require exactly one CUDA node in the node mask",
         static_cast<std::int64_t>(node_mask)});
    return std::nullopt;
  }

  ComPtr<IDXGIFactory1> factory;
  const HRESULT factory_result = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
  if (FAILED(factory_result)) {
    diagnostics.push_back({probe::DiagnosticLevel::warning, "dxgi", "CreateDXGIFactory1",
                           "DXGI factory creation failed",
                           static_cast<std::int64_t>(factory_result)});
    return std::nullopt;
  }

  for (UINT index = 0;; ++index) {
    ComPtr<IDXGIAdapter1> adapter;
    const HRESULT enumerate_result = factory->EnumAdapters1(index, &adapter);
    if (enumerate_result == DXGI_ERROR_NOT_FOUND) {
      break;
    }
    if (FAILED(enumerate_result)) {
      diagnostics.push_back({probe::DiagnosticLevel::warning, "dxgi", "EnumAdapters1",
                             "DXGI adapter enumeration failed",
                             static_cast<std::int64_t>(enumerate_result)});
      break;
    }

    DXGI_ADAPTER_DESC1 description{};
    if (FAILED(adapter->GetDesc1(&description))) {
      continue;
    }
    static_assert(sizeof(description.AdapterLuid) == std::tuple_size_v<AdapterLuid>);
    if (std::memcmp(&description.AdapterLuid, luid.data(), luid.size()) != 0) {
      continue;
    }

    probe::DxgiAdapterInfo result;
    result.name = wide_to_utf8(description.Description);
    result.node_index = node_index_from_mask(node_mask);
    result.dedicated_video_memory_bytes = description.DedicatedVideoMemory;
    result.dedicated_system_memory_bytes = description.DedicatedSystemMemory;
    result.shared_system_memory_bytes = description.SharedSystemMemory;
    result.software_adapter = (description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0;

    const auto* raw_luid = reinterpret_cast<const std::uint8_t*>(&description.AdapterLuid);
    static constexpr char hex[] = "0123456789abcdef";
    result.luid.reserve(16);
    for (std::size_t byte_index = 0; byte_index < luid.size(); ++byte_index) {
      const std::uint8_t byte = raw_luid[byte_index];
      result.luid.push_back(hex[(byte >> 4U) & 0xFU]);
      result.luid.push_back(hex[byte & 0xFU]);
    }

    ComPtr<IDXGIAdapter3> adapter3;
    const HRESULT query_result = adapter.As(&adapter3);
    if (FAILED(query_result)) {
      diagnostics.push_back({probe::DiagnosticLevel::warning, "dxgi",
                             "QueryInterface(IDXGIAdapter3)",
                             "The matched adapter does not expose video-memory budgets",
                             static_cast<std::int64_t>(query_result)});
      return result;
    }

    result.local = query_segment(*adapter3.Get(), result.node_index,
                                 DXGI_MEMORY_SEGMENT_GROUP_LOCAL, "local", diagnostics);
    result.non_local = query_segment(*adapter3.Get(), result.node_index,
                                     DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, "non_local", diagnostics);
    return result;
  }

  diagnostics.push_back({probe::DiagnosticLevel::warning, "dxgi", "match_cuda_luid",
                         "No DXGI adapter matched the CUDA device LUID", std::nullopt});
  return std::nullopt;
}

} // namespace xvram::platform
