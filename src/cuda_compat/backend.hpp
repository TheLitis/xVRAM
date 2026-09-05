#pragma once

#include "gemm/planner.hpp"
#include "sdk/status.hpp"
#include "xvram/cuda_compat.h"

#include <functional>
#include <memory>

namespace xvram::cuda_compat {

struct Allocation {
  residency::AllocationId id;
  std::uint64_t address = 0;
  std::uint64_t bytes = 0;
};

struct BackendResult {
  sdk::Error error;
  // True when failed work may have changed state or submitted GPU work. The adapter
  // admits no further data operations after such a failure; close remains available.
  bool poison = false;
};

// Only the serial adapter worker calls Backend methods. Snapshot callbacks can happen during
// a long call; their input owns no pointers and the adapter copies it under a separate lock.
// Backend fills runtime, device/host geometry, host backing, retired VA and cleanup fields.
using SnapshotSink = std::function<void(const xvram_cuda_compat_telemetry_v1&)>;

class Backend {
public:
  virtual ~Backend() = default;
  virtual sdk::Error initialize(const xvram_cuda_compat_config_v1&, SnapshotSink) = 0;
  virtual BackendResult allocate(std::uint64_t bytes, Allocation& output) = 0;
  virtual BackendResult release(residency::AllocationId id) = 0;
  virtual BackendResult write(residency::AllocationId id, std::uint64_t offset, const void* source,
                              std::uint64_t bytes) = 0;
  virtual BackendResult read(residency::AllocationId id, std::uint64_t offset, void* destination,
                             std::uint64_t bytes) = 0;
  virtual BackendResult gemm(const gemm::GemmProblem& problem) = 0;
  virtual BackendResult synchronize() = 0;
  virtual BackendResult close() = 0;
};

using BackendFactory = std::function<std::unique_ptr<Backend>()>;
[[nodiscard]] std::unique_ptr<Backend> make_runtime_backend();

// Per-instance seam lets fault tests exercise lifecycle/ownership without creating CUDA contexts.
class Adapter {
public:
  explicit Adapter(BackendFactory factory = make_runtime_backend);
  ~Adapter();
  Adapter(const Adapter&) = delete;
  Adapter& operator=(const Adapter&) = delete;
  xvram_status initialize(const xvram_cuda_compat_config_v1* config) noexcept;
  xvram_status shutdown() noexcept;
  xvram_status get_error(xvram_error_info_v1* output) const noexcept;
  xvram_status get_telemetry(xvram_cuda_compat_telemetry_v1* output) const noexcept;
  xvram_status malloc_device(void** output, std::uint64_t bytes) noexcept;
  xvram_status free_device(void* pointer) noexcept;
  xvram_status memcpy(void* destination, const void* source, std::uint64_t bytes,
                      std::uint32_t kind) noexcept;
  xvram_status get_device(std::int32_t* output) noexcept;
  xvram_status get_device_count(std::int32_t* output) noexcept;
  xvram_status set_device(std::int32_t device) noexcept;
  xvram_status synchronize() noexcept;
  xvram_status mem_get_info(std::uint64_t* free_bytes, std::uint64_t* total_bytes) noexcept;
  xvram_status blas_create(void** output) noexcept;
  xvram_status blas_destroy(void* handle) noexcept;
  xvram_status blas_get_version(void* handle, std::int32_t* output) noexcept;
  xvram_status blas_set_stream(void* handle, std::uint64_t stream) noexcept;
  xvram_status blas_get_stream(void* handle, std::uint64_t* output) noexcept;
  xvram_status blas_set_pointer_mode(void* handle, std::uint32_t mode) noexcept;
  xvram_status blas_get_pointer_mode(void* handle, std::uint32_t* output) noexcept;
  xvram_status blas_set_math_mode(void* handle, std::uint32_t mode) noexcept;
  xvram_status blas_get_math_mode(void* handle, std::uint32_t* output) noexcept;
  xvram_status blas_sgemm(void* handle, std::uint32_t op_a, std::uint32_t op_b, std::int32_t m,
                          std::int32_t n, std::int32_t k, const float* alpha, const float* a,
                          std::int32_t lda, const float* b, std::int32_t ldb, const float* beta,
                          float* c, std::int32_t ldc) noexcept;
  // Internal fault seam, not part of the exported C ABI.
  void fail_next_enqueue_for_testing() noexcept;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace xvram::cuda_compat
