#pragma once

#include "sdk/status.hpp"
#include "xvram/xvram_v2.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace xvram::sdk {

class BackendAllocation;
class BackendOperation;
class BackendGemmPlan;
class BackendSession;

struct AccessRequest {
  std::shared_ptr<BackendAllocation> allocation;
  std::uint64_t offset_bytes = 0;
  std::uint64_t length_bytes = 0;
  xvram_access_mode mode = XVRAM_ACCESS_READ;
};

struct PrefetchRequest {
  std::vector<AccessRequest> ranges;
  std::uint32_t flags = 0;
  std::uint64_t deadline_ms = 0;
};

struct TransactionRequest {
  std::vector<AccessRequest> ranges;
  std::uint32_t flags = 0;
  std::uint64_t workspace_bytes = 0;
  std::uint64_t deadline_ms = 0;
  xvram_transaction_callback_v1 callback = nullptr;
  void* user_data = nullptr;
};

struct MatrixRequest {
  std::shared_ptr<BackendAllocation> allocation;
  std::uint64_t offset_bytes = 0;
  std::uint64_t rows = 0;
  std::uint64_t columns = 0;
  std::uint64_t leading_dimension = 0;
  xvram_data_type data_type = XVRAM_DATA_FP32;
  xvram_matrix_layout layout = XVRAM_MATRIX_ROW_MAJOR;
};

struct GemmRequest {
  MatrixRequest a;
  MatrixRequest b;
  MatrixRequest c;
  std::uint32_t flags = 0;
  xvram_matrix_operation operation_a = XVRAM_MATRIX_OP_N;
  xvram_matrix_operation operation_b = XVRAM_MATRIX_OP_N;
  xvram_compute_mode compute_mode = XVRAM_COMPUTE_AUTO;
  double alpha = 1.0;
  double beta = 0.0;
  std::uint64_t workspace_cap_bytes = 0;
  std::uint64_t deadline_ms = 0;
  std::uint64_t tile_m_hint = 0;
  std::uint64_t tile_n_hint = 0;
  std::uint64_t tile_k_hint = 0;
};

class BackendAllocation {
public:
  virtual ~BackendAllocation() = default;

  [[nodiscard]] virtual Error info(xvram_allocation_info_v1& output) const = 0;
  [[nodiscard]] virtual Error write(std::uint64_t offset_bytes, const void* source,
                                    std::uint64_t size_bytes) = 0;
  [[nodiscard]] virtual Error read(std::uint64_t offset_bytes, void* destination,
                                   std::uint64_t size_bytes) = 0;
  [[nodiscard]] virtual Error release() = 0;
};

class BackendOperation {
public:
  virtual ~BackendOperation() = default;

  [[nodiscard]] virtual Error poll(xvram_operation_info_v1& output) = 0;
  [[nodiscard]] virtual Error wait(std::uint64_t timeout_ms, xvram_operation_info_v1& output) = 0;
  [[nodiscard]] virtual Error cancel() = 0;
  [[nodiscard]] virtual Error error() const = 0;
};

class BackendGemmPlan {
public:
  virtual ~BackendGemmPlan() = default;

  [[nodiscard]] virtual Error info(xvram_gemm_plan_info_v1& output) const = 0;
  [[nodiscard]] virtual Error submit(std::shared_ptr<BackendOperation>& output) = 0;
};

class BackendSession {
public:
  virtual ~BackendSession() = default;

  [[nodiscard]] virtual std::uint64_t chunk_size_bytes() const noexcept = 0;

  [[nodiscard]] virtual Error allocate(const xvram_allocation_desc_v1& desc,
                                       std::shared_ptr<BackendAllocation>& output) = 0;
  [[nodiscard]] virtual Error prefetch(const PrefetchRequest& request,
                                       std::shared_ptr<BackendOperation>& output) = 0;
  [[nodiscard]] virtual Error submit_transaction(const TransactionRequest& request,
                                                 std::shared_ptr<BackendOperation>& output) = 0;
  [[nodiscard]] virtual Error create_gemm_plan(const GemmRequest& request,
                                               std::shared_ptr<BackendGemmPlan>& output) = 0;
  [[nodiscard]] virtual Error telemetry(xvram_session_telemetry_v1& output) const = 0;
  [[nodiscard]] virtual Error telemetry_v2(xvram_session_telemetry_v2& output) const {
    output.v1 = {};
    output.v1.struct_size = sizeof(output.v1);
    if (Error error = telemetry(output.v1); error) {
      return error;
    }
    // The default adapter describes an ABI-v1 raw session. Compression-aware
    // backends override this method with authoritative representation counters.
    output.logical_h2d_bytes = output.v1.bytes_h2d;
    output.pcie_h2d_bytes = output.v1.bytes_h2d;
    output.logical_d2h_bytes = output.v1.bytes_d2h;
    output.pcie_d2h_bytes = output.v1.bytes_d2h;
    return {};
  }
  [[nodiscard]] virtual Error drain(std::uint64_t timeout_ms) = 0;
  [[nodiscard]] virtual Error close(std::uint64_t timeout_ms) = 0;
};

class BackendFactory {
public:
  virtual ~BackendFactory() = default;

  [[nodiscard]] virtual Error create_session(const xvram_session_config_v1& config,
                                             std::shared_ptr<BackendSession>& output) = 0;
  [[nodiscard]] virtual Error create_session_v2(const xvram_session_config_v2& config,
                                                std::shared_ptr<BackendSession>& output) {
    if (config.compression_mode != XVRAM_COMPRESSION_DISABLED) {
      return make_error(XVRAM_STATUS_UNSUPPORTED, "sdk", "session_create_v2",
                        "backend does not implement compressed host backing");
    }
    return create_session(config.v1, output);
  }
};

/*
 * runtime.cpp installs a provider before the public entry point is used. The
 * provider and returned factory must remain alive until the SDK shared library is
 * unloaded. Keeping this hook internal avoids adding a second exported C symbol.
 */
using BackendFactoryProvider = BackendFactory* (*)() noexcept;
void install_backend_factory(BackendFactoryProvider provider) noexcept;

} // namespace xvram::sdk
