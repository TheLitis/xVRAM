#ifndef TORCH_TARGET_VERSION
#define TORCH_TARGET_VERSION 0x020B000000000000ULL
#endif

#include "torch_runtime/control.hpp"

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

static_assert(TORCH_FEATURE_VERSION == TORCH_VERSION_2_11_0,
              "xvram_torch_runtime must target the PyTorch 2.11 Stable ABI");

namespace xvram::torch_runtime {
namespace {

[[nodiscard]] std::uint64_t checked_add(const std::uint64_t left, const std::uint64_t right,
                                        const char* message) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    throw std::invalid_argument(message);
  }
  return left + right;
}

[[nodiscard]] std::uint64_t checked_multiply(const std::uint64_t left, const std::uint64_t right,
                                             const char* message) {
  if (left != 0U && right > std::numeric_limits<std::uint64_t>::max() / left) {
    throw std::invalid_argument(message);
  }
  return left * right;
}

[[nodiscard]] std::uint64_t scalar_size(const std::int64_t scalar_type) {
  switch (scalar_type) {
  case 0:  // Byte
  case 1:  // Char
  case 11: // Bool
    return 1U;
  case 2:  // Short
  case 5:  // Half
  case 15: // BFloat16
    return 2U;
  case 3: // Int
  case 6: // Float
  case 8: // ComplexHalf
    return 4U;
  case 4: // Long
  case 7: // Double
  case 9: // ComplexFloat
    return 8U;
  case 10: // ComplexDouble
    return 16U;
  default:
    throw std::invalid_argument(
        "xVRAM Stable-ABI view uses an unsupported or version-unstable scalar type");
  }
}

struct TensorExtent {
  std::uint64_t bytes = 0;
  std::uint64_t alignment = 0;
};

[[nodiscard]] TensorExtent tensor_extent(const torch::headeronly::IntHeaderOnlyArrayRef sizes,
                                         const torch::headeronly::IntHeaderOnlyArrayRef strides,
                                         const std::int64_t scalar_type) {
  if (sizes.size() != strides.size() || sizes.size() > 64U) {
    throw std::invalid_argument("xVRAM tensor sizes and strides have incompatible ranks");
  }
  const std::uint64_t item_bytes = scalar_size(scalar_type);
  std::uint64_t maximum_element = 0;
  bool empty = false;
  for (std::size_t index = 0; index < sizes.size(); ++index) {
    const std::int64_t size = sizes[index];
    const std::int64_t stride = strides[index];
    if (size < 0 || stride < 0) {
      throw std::invalid_argument("xVRAM Stable-ABI views reject negative sizes and strides");
    }
    if (size == 0) {
      empty = true;
      continue;
    }
    maximum_element = checked_add(maximum_element,
                                  checked_multiply(static_cast<std::uint64_t>(size - 1),
                                                   static_cast<std::uint64_t>(stride),
                                                   "xVRAM tensor stride extent overflowed"),
                                  "xVRAM tensor extent overflowed");
  }
  // Empty tensors cannot dereference the pointer. One element is retained as a
  // conservative storage bound and keeps the Stable-ABI from_blob contract simple.
  const std::uint64_t elements =
      empty ? 1U : checked_add(maximum_element, 1U, "xVRAM tensor size overflowed");
  return {checked_multiply(elements, item_bytes, "xVRAM tensor byte extent overflowed"),
          item_bytes};
}

} // namespace

torch::stable::Tensor wrap_resolved_v1(const std::int64_t session_id, const std::int64_t lease_id,
                                       const std::int64_t allocation_id,
                                       const std::int64_t byte_offset,
                                       const torch::headeronly::IntHeaderOnlyArrayRef sizes,
                                       const torch::headeronly::IntHeaderOnlyArrayRef strides,
                                       const std::int64_t scalar_type) {
  if (session_id <= 0 || lease_id <= 0 || allocation_id <= 0 || byte_offset < 0) {
    throw std::invalid_argument(
        "xVRAM session, lease, allocation, and byte offset metadata are invalid");
  }
  const TensorExtent extent = tensor_extent(sizes, strides, scalar_type);
  PreparedView prepared{};
  const xvram_torch_runtime_status status = prepare_view(
      static_cast<xvram_torch_runtime_session>(session_id),
      static_cast<xvram_torch_runtime_lease>(lease_id),
      static_cast<xvram_torch_runtime_allocation>(allocation_id),
      static_cast<std::uint64_t>(byte_offset), extent.bytes, extent.alignment, prepared);
  if (status != XVRAM_TORCH_RUNTIME_SUCCESS) {
    throw std::runtime_error("xVRAM cannot wrap the logical allocation: " + last_error_message());
  }
  if (prepared.device_ordinal < 0 ||
      prepared.device_ordinal > std::numeric_limits<torch::stable::DeviceIndex>::max()) {
    throw std::runtime_error("xVRAM resolved an invalid CUDA device ordinal");
  }

  const auto dtype = static_cast<torch::headeronly::ScalarType>(scalar_type);
  const torch::stable::Device device(
      torch::headeronly::DeviceType::CUDA,
      static_cast<torch::stable::DeviceIndex>(prepared.device_ordinal));
  auto ticket = std::move(prepared.ticket);
  return torch::stable::from_blob(
      prepared.address, sizes, strides, device, dtype,
      [ticket = std::move(ticket)](void*) mutable noexcept { ticket.reset(); });
}

} // namespace xvram::torch_runtime

STABLE_TORCH_LIBRARY(xvram_internal, library) {
  library.def("_wrap_resolved_v1(int session, int lease, int allocation, int offset, int[] sizes, "
              "int[] strides, int scalar_type) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(xvram_internal, CompositeExplicitAutograd, library) {
  library.impl("_wrap_resolved_v1", TORCH_BOX(&xvram::torch_runtime::wrap_resolved_v1));
}
