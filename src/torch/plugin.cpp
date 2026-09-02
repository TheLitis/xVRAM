#include "torch/allocator.hpp"
#include "xvram/torch_allocator.h"

#include <algorithm>
#include <cstring>
#include <new>

namespace {

struct PluginState {
  xvram::cuda::CudaApi api;
  xvram::torch_allocator::SegmentAllocator allocator;

  PluginState() noexcept : allocator(api) {}
};

[[nodiscard]] PluginState* plugin_state() noexcept {
  // Intentionally process-lifetime. Running CUDA cleanup from a DLL finalizer can
  // execute under the platform loader lock after PyTorch has torn its contexts down.
  static PluginState* state = new (std::nothrow) PluginState();
  return state;
}

template <typename Structure>
[[nodiscard]] xvram_torch_allocator_status copy_versioned_structure(
    Structure* output, const std::size_t output_size, const Structure& value) noexcept {
  constexpr std::size_t prefix_size = sizeof(std::uint32_t) * 2U;
  if (output == nullptr || output_size < prefix_size || output->struct_size < prefix_size) {
    return XVRAM_TORCH_ALLOCATOR_INCOMPATIBLE_ABI;
  }
  const std::size_t copied =
      std::min({output_size, static_cast<std::size_t>(output->struct_size), sizeof(Structure)});
  std::memcpy(output, &value, copied);
  return XVRAM_TORCH_ALLOCATOR_SUCCESS;
}

} // namespace

extern "C" void* XVRAM_TORCH_CALL xvram_torch_alloc(const std::size_t bytes, const int device,
                                                      void* stream) {
  PluginState* state = plugin_state();
  if (state == nullptr) {
    return nullptr;
  }
  return state->allocator.allocate(bytes, device,
                                   reinterpret_cast<xvram::cuda::abi::Stream>(stream));
}

extern "C" void XVRAM_TORCH_CALL xvram_torch_free(void* pointer, const std::size_t bytes,
                                                   const int device, void* stream) {
  PluginState* state = plugin_state();
  if (state == nullptr) {
    return;
  }
  state->allocator.deallocate(pointer, bytes, device,
                              reinterpret_cast<xvram::cuda::abi::Stream>(stream));
}

extern "C" xvram_torch_allocator_status XVRAM_TORCH_CALL xvram_torch_get_stats(
    xvram_torch_allocator_stats_v1* output, const std::size_t output_size) {
  PluginState* state = plugin_state();
  if (state == nullptr) {
    return XVRAM_TORCH_ALLOCATOR_OUT_OF_MEMORY;
  }
  return copy_versioned_structure(output, output_size, state->allocator.stats());
}

extern "C" xvram_torch_allocator_status XVRAM_TORCH_CALL xvram_torch_reset_stats() {
  PluginState* state = plugin_state();
  if (state == nullptr) {
    return XVRAM_TORCH_ALLOCATOR_OUT_OF_MEMORY;
  }
  state->allocator.reset_stats();
  return XVRAM_TORCH_ALLOCATOR_SUCCESS;
}

extern "C" xvram_torch_allocator_status XVRAM_TORCH_CALL xvram_torch_get_last_error(
    xvram_torch_allocator_error_v1* output, const std::size_t output_size) {
  constexpr std::size_t prefix_size = sizeof(std::uint32_t) * 2U;
  if (output == nullptr || output_size < prefix_size || output->struct_size < prefix_size) {
    return XVRAM_TORCH_ALLOCATOR_INCOMPATIBLE_ABI;
  }
  PluginState* state = plugin_state();
  if (state == nullptr) {
    return XVRAM_TORCH_ALLOCATOR_OUT_OF_MEMORY;
  }

  const xvram::torch_allocator::LastError source = state->allocator.last_error();
  xvram_torch_allocator_error_v1 value = XVRAM_TORCH_ALLOCATOR_ERROR_V1_INIT;
  value.status = source.status;
  value.native_code = source.native_code;
  std::memcpy(value.stage, source.stage.data(), source.stage.size());
  std::memcpy(value.operation, source.operation.data(), source.operation.size());
  std::memcpy(value.message, source.message.data(), source.message.size());
  const std::size_t copied =
      std::min({output_size, static_cast<std::size_t>(output->struct_size), sizeof(value)});
  std::memcpy(output, &value, copied);
  return XVRAM_TORCH_ALLOCATOR_SUCCESS;
}
