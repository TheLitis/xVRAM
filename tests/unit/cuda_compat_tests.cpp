#include "cuda_compat/backend.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <future>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using namespace xvram;
using cuda_compat::Adapter;
using cuda_compat::BackendResult;
int failures = 0;
#define CHECK(value)                                                                               \
  do {                                                                                             \
    if (!(value)) {                                                                                \
      std::cerr << "line " << __LINE__ << ": " #value "\n";                                        \
      ++failures;                                                                                  \
    }                                                                                              \
  } while (false)
struct State {
  std::map<std::uint64_t, std::vector<std::byte>> memory;
  std::thread::id worker;
  std::thread::id destroyed;
  std::uint64_t next_id = 1;
  std::uint64_t backend_calls = 0;
  std::uint64_t gemm_calls = 0;
  bool wrong_thread = false;
  bool fail_initialize = false;
  bool fail_allocate = false;
  bool fail_write = false;
  bool throw_write = false;
  bool throw_close = false;
  bool closed = false;
  std::mutex gate_mutex;
  std::condition_variable gate;
  bool block_gemm = false;
  bool gemm_started = false;
  bool release_gemm = false;
};
class Fake final : public cuda_compat::Backend {
public:
  explicit Fake(std::shared_ptr<State> state) : s_(std::move(state)) {}
  ~Fake() override {
    s_->destroyed = std::this_thread::get_id();
  }
  void check_thread() {
    if (s_->worker != std::this_thread::get_id())
      s_->wrong_thread = true;
    ++s_->backend_calls;
  }
  void publish() {
    ++t_.progress_sequence;
    if (sink_)
      sink_(t_);
  }
  sdk::Error initialize(const xvram_cuda_compat_config_v1& config,
                        cuda_compat::SnapshotSink sink) override {
    s_->worker = std::this_thread::get_id();
    sink_ = std::move(sink);
    check_thread();
    t_.struct_size = sizeof(t_);
    t_.runtime.struct_size = sizeof(t_.runtime);
    t_.device_ordinal = config.session.device_ordinal;
    t_.device_count = 1;
    t_.effective_chunk_bytes = config.session.chunk_size_bytes;
    t_.total_vram_bytes = 8ULL << 30;
    t_.host_physical_bytes = 32ULL << 30;
    t_.host_available_bytes = 24ULL << 30;
    t_.host_store_cap_bytes = 16ULL << 30;
    t_.runtime.cublas_version = 130600;
    t_.runtime.cuda_driver_version = 13030;
    std::memcpy(t_.device_name, "fake GPU", 9);
    publish();
    return s_->fail_initialize ? sdk::make_error(XVRAM_STATUS_UNAVAILABLE, "preflight",
                                                 "initialize", "fake driver unavailable")
                               : sdk::Error{};
  }
  BackendResult allocate(std::uint64_t bytes, cuda_compat::Allocation& output) override {
    check_thread();
    if (s_->fail_allocate)
      return {sdk::make_error(XVRAM_STATUS_HOST_OUT_OF_MEMORY, "preflight", "allocate",
                              "fake host admission"),
              false};
    const auto id = s_->next_id++;
    s_->memory[id].resize(static_cast<std::size_t>(bytes));
    output = {{id}, 0x100000000ULL + id * (1ULL << 20), bytes};
    publish();
    return {};
  }
  BackendResult release(residency::AllocationId id) override {
    check_thread();
    s_->memory.erase(id.value);
    ++t_.retired_va_reservations;
    t_.retired_va_bytes += t_.effective_chunk_bytes;
    publish();
    return {};
  }
  BackendResult write(residency::AllocationId id, std::uint64_t offset, const void* source,
                      std::uint64_t bytes) override {
    check_thread();
    if (s_->throw_write)
      throw std::runtime_error("write threw after admission");
    if (s_->fail_write)
      return {sdk::make_native_error(XVRAM_STATUS_CUDA_ERROR, XVRAM_NATIVE_ERROR_CUDA, 999, "copy",
                                     "fake_write", "CUDA_ERROR_UNKNOWN",
                                     "fake possible submission"),
              true};
    std::memcpy(s_->memory.at(id.value).data() + offset, source, static_cast<std::size_t>(bytes));
    publish();
    return {};
  }
  BackendResult read(residency::AllocationId id, std::uint64_t offset, void* destination,
                     std::uint64_t bytes) override {
    check_thread();
    std::memcpy(destination, s_->memory.at(id.value).data() + offset,
                static_cast<std::size_t>(bytes));
    publish();
    return {};
  }
  BackendResult gemm(const gemm::GemmProblem& p) override {
    check_thread();
    ++s_->gemm_calls;
    ++t_.tiles_submitted;
    publish();
    {
      std::unique_lock lock(s_->gate_mutex);
      s_->gemm_started = true;
      s_->gate.notify_all();
      if (s_->block_gemm)
        s_->gate.wait(lock, [this] { return s_->release_gemm; });
    }
    const auto get = [this](const gemm::MatrixView& v, std::uint64_t row, std::uint64_t col) {
      float value = 0;
      const auto& memory = s_->memory.at(v.allocation_id.value);
      std::memcpy(&value, memory.data() + v.offset_bytes + (col * v.leading_dimension + row) * 4,
                  4);
      return value;
    };
    for (std::uint64_t j = 0; j < p.n; ++j)
      for (std::uint64_t i = 0; i < p.m; ++i) {
        float sum = 0;
        for (std::uint64_t kk = 0; kk < p.k; ++kk)
          sum +=
              (p.a_operation == gemm::MatrixOperation::none ? get(p.a, i, kk) : get(p.a, kk, i)) *
              (p.b_operation == gemm::MatrixOperation::none ? get(p.b, kk, j) : get(p.b, j, kk));
        const float result =
            static_cast<float>(p.alpha) * sum + static_cast<float>(p.beta) * get(p.c, i, j);
        std::memcpy(s_->memory.at(p.c.allocation_id.value).data() + p.c.offset_bytes +
                        (j * p.c.leading_dimension + i) * 4,
                    &result, 4);
      }
    ++t_.tiles_retired;
    publish();
    return {};
  }
  BackendResult synchronize() override {
    check_thread();
    return {};
  }
  BackendResult close() override {
    check_thread();
    s_->closed = true;
    if (s_->throw_close)
      throw std::runtime_error("fake close failed");
    s_->memory.clear();
    t_.retired_va_reservations_freed += t_.retired_va_reservations;
    t_.retired_va_bytes_freed += t_.retired_va_bytes;
    t_.retired_va_reservations = 0;
    t_.retired_va_bytes = 0;
    t_.cleanup_operations_drained = 1;
    t_.cleanup_events_drained = 1;
    t_.cleanup_completed = 1;
    publish();
    return {};
  }

private:
  std::shared_ptr<State> s_;
  cuda_compat::SnapshotSink sink_;
  xvram_cuda_compat_telemetry_v1 t_{};
};
cuda_compat::BackendFactory factory(const std::shared_ptr<State>& s) {
  return [s] { return std::make_unique<Fake>(s); };
}
xvram_cuda_compat_config_v1 config() {
  xvram_cuda_compat_config_v1 c = XVRAM_CUDA_COMPAT_CONFIG_V1_INIT;
  c.session.chunk_size_bytes = 65536;
  return c;
}
xvram_cuda_compat_telemetry_v1 telemetry(Adapter& adapter) {
  xvram_cuda_compat_telemetry_v1 t{};
  t.struct_size = sizeof(t);
  CHECK(adapter.get_telemetry(&t) == XVRAM_STATUS_SUCCESS);
  return t;
}
void lifecycle_and_ranges() {
  auto s = std::make_shared<State>();
  Adapter a(factory(s));
  auto c = config();
  void* pointer = reinterpret_cast<void*>(0x1234);
  CHECK(a.malloc_device(&pointer, 64) == XVRAM_STATUS_NOT_READY);
  CHECK(pointer == reinterpret_cast<void*>(0x1234));
  c.session.max_transaction_ms = 251;
  CHECK(a.initialize(&c) == XVRAM_STATUS_INVALID_ARGUMENT);
  c = config();
  c.flags = 1;
  CHECK(a.initialize(&c) == XVRAM_STATUS_INVALID_ARGUMENT);
  c = config();
  c.cublas_library[0] = 'x';
  CHECK(a.initialize(&c) == XVRAM_STATUS_INVALID_ARGUMENT);
  c = config();
  CHECK(a.initialize(&c) == XVRAM_STATUS_SUCCESS);
  CHECK(a.initialize(&c) == XVRAM_STATUS_CLOSED);
  CHECK(a.malloc_device(&pointer, 0) == XVRAM_STATUS_INVALID_ARGUMENT);
  CHECK(pointer == reinterpret_cast<void*>(0x1234));
  CHECK(a.malloc_device(&pointer, 64) == XVRAM_STATUS_SUCCESS);
  auto* interior = reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(pointer) + 4);
  const auto before = s->backend_calls;
  CHECK(a.free_device(interior) == XVRAM_STATUS_INVALID_ARGUMENT);
  CHECK(a.malloc_device(reinterpret_cast<void**>(pointer), 64) == XVRAM_STATUS_INVALID_ARGUMENT);
  CHECK(a.blas_create(reinterpret_cast<void**>(pointer)) == XVRAM_STATUS_INVALID_ARGUMENT);
  CHECK(a.memcpy(pointer, pointer, 4, XVRAM_CUDA_COMPAT_H2D) == XVRAM_STATUS_INVALID_ARGUMENT);
  CHECK(a.memcpy(pointer, pointer, 4, 3) == XVRAM_STATUS_UNSUPPORTED);
  CHECK(s->backend_calls == before);
  const float input[3] = {1, 2, 3};
  float output[3] = {-1, -1, -1};
  CHECK(a.memcpy(interior, input, sizeof(input), XVRAM_CUDA_COMPAT_H2D) == XVRAM_STATUS_SUCCESS);
  CHECK(a.memcpy(output, interior, sizeof(output), XVRAM_CUDA_COMPAT_D2H) == XVRAM_STATUS_SUCCESS);
  CHECK(std::memcmp(input, output, sizeof(input)) == 0);
  CHECK(a.memcpy(output, interior, 64, XVRAM_CUDA_COMPAT_D2H) == XVRAM_STATUS_INVALID_ARGUMENT);
  CHECK(a.free_device(pointer) == XVRAM_STATUS_SUCCESS);
  CHECK(a.free_device(pointer) == XVRAM_STATUS_INVALID_ARGUMENT);
  CHECK(a.memcpy(output, pointer, 4, XVRAM_CUDA_COMPAT_D2H) == XVRAM_STATUS_INVALID_ARGUMENT);
  void* next = nullptr;
  CHECK(a.malloc_device(&next, 64) == XVRAM_STATUS_SUCCESS);
  CHECK(next != pointer);
  // A retired managed pointer is not accepted as the host side of a later copy.
  CHECK(a.memcpy(next, pointer, 4, XVRAM_CUDA_COMPAT_H2D) == XVRAM_STATUS_INVALID_ARGUMENT);
  void* h = nullptr;
  CHECK(a.blas_create(&h) == XVRAM_STATUS_SUCCESS);
  CHECK(a.blas_set_stream(h, 1) == XVRAM_STATUS_UNSUPPORTED);
  CHECK(a.blas_set_pointer_mode(h, 1) == XVRAM_STATUS_UNSUPPORTED);
  CHECK(a.blas_set_math_mode(h, 3) == XVRAM_STATUS_UNSUPPORTED);
  CHECK(a.blas_set_math_mode(h, 2) == XVRAM_STATUS_SUCCESS);
  uint32_t mode = 9;
  CHECK(a.blas_get_math_mode(h, &mode) == XVRAM_STATUS_SUCCESS);
  CHECK(mode == 2);
  CHECK(a.blas_get_math_mode(h, reinterpret_cast<uint32_t*>(next)) ==
        XVRAM_STATUS_INVALID_ARGUMENT);
  CHECK(a.blas_destroy(h) == XVRAM_STATUS_SUCCESS);
  CHECK(a.blas_destroy(h) == XVRAM_STATUS_INVALID_ARGUMENT);
  mode = 9;
  CHECK(a.blas_get_math_mode(h, &mode) == XVRAM_STATUS_INVALID_ARGUMENT);
  CHECK(mode == 9);
  void* h2 = nullptr;
  CHECK(a.blas_create(&h2) == XVRAM_STATUS_SUCCESS);
  CHECK(h2 != h);
  const auto progress = telemetry(a).progress_sequence;
  int32_t device = -1;
  CHECK(a.get_device(&device) == XVRAM_STATUS_SUCCESS);
  CHECK(device == 0);
  CHECK(a.set_device(0) == XVRAM_STATUS_SUCCESS);
  CHECK(a.blas_get_math_mode(h2, &mode) == XVRAM_STATUS_SUCCESS);
  CHECK(telemetry(a).progress_sequence == progress);
  CHECK(a.shutdown() == XVRAM_STATUS_SUCCESS);
  CHECK(s->closed);
  CHECK(!s->wrong_thread);
  CHECK(s->destroyed == s->worker);
  const auto t = telemetry(a);
  CHECK(t.state == XVRAM_CUDA_COMPAT_CLOSED);
  CHECK(t.live_allocations == 0);
  CHECK(t.live_handles == 0);
  CHECK(t.calls_attempted == t.calls_completed + t.calls_rejected);
  CHECK(a.initialize(&c) == XVRAM_STATUS_CLOSED);
  CHECK(a.malloc_device(&pointer, 64) == XVRAM_STATUS_CLOSED);
}
void gemm_validation_and_concurrency() {
  auto s = std::make_shared<State>();
  Adapter a(factory(s));
  auto c = config();
  CHECK(a.initialize(&c) == 0);
  void* ap = nullptr;
  void* bp = nullptr;
  void* cp = nullptr;
  void* h = nullptr;
  CHECK(a.malloc_device(&ap, 64) == 0);
  CHECK(a.malloc_device(&bp, 64) == 0);
  CHECK(a.malloc_device(&cp, 64) == 0);
  CHECK(a.blas_create(&h) == 0);
  const float matrix[4] = {1, 2, 3, 4};
  const float alpha = 1, beta = 1;
  for (void* p : {ap, bp, cp})
    CHECK(a.memcpy(p, matrix, sizeof(matrix), 1) == 0);
  const auto run = [&](uint32_t op, int32_t m, int32_t lda, const float* scalar, float* output) {
    return a.blas_sgemm(h, op, 0, m, 2, 2, scalar, static_cast<float*>(ap), lda,
                        static_cast<float*>(bp), 2, &beta, output, 2);
  };
  CHECK(run(2, 2, 2, &alpha, static_cast<float*>(cp)) == XVRAM_STATUS_UNSUPPORTED);
  CHECK(run(0, 0, 2, &alpha, static_cast<float*>(cp)) == XVRAM_STATUS_UNSUPPORTED);
  CHECK(run(0, 2, 1, &alpha, static_cast<float*>(cp)) == XVRAM_STATUS_INVALID_ARGUMENT);
  CHECK(run(0, 2, 2, static_cast<float*>(ap), static_cast<float*>(cp)) ==
        XVRAM_STATUS_INVALID_ARGUMENT);
  CHECK(run(0, 2, 2, &alpha, static_cast<float*>(ap)) == XVRAM_STATUS_INVALID_ARGUMENT);
  CHECK(run(0, 2, 200, &alpha, static_cast<float*>(cp)) == XVRAM_STATUS_INVALID_ARGUMENT);
  CHECK(s->gemm_calls == 0);
  s->block_gemm = true;
  auto running =
      std::async(std::launch::async, [&] { return run(0, 2, 2, &alpha, static_cast<float*>(cp)); });
  {
    std::unique_lock lock(s->gate_mutex);
    s->gate.wait(lock, [&] { return s->gemm_started; });
  }
  auto t = telemetry(a);
  CHECK(t.tiles_submitted == 1);
  CHECK(t.tiles_retired == 0);
  // The queued call must own scalar values before it waits behind the blocked first call.
  float queued_alpha = 0;
  const auto attempted_before = t.calls_attempted;
  auto queued = std::async(std::launch::async,
                           [&] { return run(0, 2, 2, &queued_alpha, static_cast<float*>(cp)); });
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (telemetry(a).calls_attempted == attempted_before &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  const bool captured = telemetry(a).calls_attempted > attempted_before;
  CHECK(captured);
  if (captured)
    queued_alpha = 100; // Capture precedes the observed attempted counter.
  {
    std::lock_guard lock(s->gate_mutex);
    s->release_gemm = true;
  }
  s->gate.notify_all();
  CHECK(running.get() == 0);
  CHECK(queued.get() == 0);
  t = telemetry(a);
  CHECK(t.tiles_retired == 2);
  float output[4]{};
  CHECK(a.memcpy(output, cp, sizeof(output), 2) == 0);
  const float expected[4] = {8, 12, 18, 26};
  CHECK(std::memcmp(output, expected, sizeof(output)) == 0);
  std::vector<std::future<bool>> callers;
  for (int i = 0; i < 8; ++i)
    callers.push_back(std::async(std::launch::async, [&] {
      void* p = nullptr;
      return a.malloc_device(&p, 32) == 0 && a.free_device(p) == 0;
    }));
  for (auto& caller : callers)
    CHECK(caller.get());
  CHECK(!s->wrong_thread);
  CHECK(a.shutdown() == 0);
  t = telemetry(a);
  CHECK(t.calls_attempted == t.calls_completed + t.calls_rejected);
}
void failures_and_cleanup() {
  for (int mode = 0; mode < 4; ++mode) {
    auto s = std::make_shared<State>();
    s->fail_initialize = mode == 0;
    Adapter a(factory(s));
    auto c = config();
    if (mode == 0) {
      CHECK(a.initialize(&c) == XVRAM_STATUS_UNAVAILABLE);
      CHECK(a.initialize(&c) == XVRAM_STATUS_CLOSED);
      CHECK(s->destroyed == s->worker);
      continue;
    }
    CHECK(a.initialize(&c) == 0);
    void* p = nullptr;
    s->fail_allocate = true;
    CHECK(a.malloc_device(&p, 32) == XVRAM_STATUS_HOST_OUT_OF_MEMORY);
    CHECK(p == nullptr);
    CHECK(telemetry(a).state == XVRAM_CUDA_COMPAT_READY);
    s->fail_allocate = false;
    CHECK(a.malloc_device(&p, 32) == 0);
    s->fail_write = mode == 1;
    s->throw_write = mode == 2;
    s->throw_close = mode == 3;
    if (mode != 3) {
      int input = 5;
      CHECK(a.memcpy(p, &input, sizeof(input), 1) != 0);
      CHECK(telemetry(a).state == XVRAM_CUDA_COMPAT_POISONED);
      CHECK(a.synchronize() == XVRAM_STATUS_POISONED);
      CHECK(a.free_device(p) == XVRAM_STATUS_POISONED);
      xvram_error_info_v1 error = XVRAM_ERROR_INFO_V1_INIT;
      CHECK(a.get_error(&error) == 0);
      CHECK(error.status == XVRAM_STATUS_POISONED);
    }
    CHECK(a.shutdown() ==
          static_cast<xvram_status>(mode == 3 ? XVRAM_STATUS_INTERNAL : XVRAM_STATUS_SUCCESS));
    CHECK(s->destroyed == s->worker);
    CHECK(!s->wrong_thread);
  }
  auto s = std::make_shared<State>();
  Adapter a(factory(s));
  auto c = config();
  CHECK(a.initialize(&c) == XVRAM_STATUS_SUCCESS);
  void* live = nullptr;
  void* handle = nullptr;
  CHECK(a.malloc_device(&live, 64) == XVRAM_STATUS_SUCCESS);
  CHECK(a.blas_create(&handle) == XVRAM_STATUS_SUCCESS);
  a.fail_next_enqueue_for_testing();
  CHECK(a.shutdown() == XVRAM_STATUS_HOST_OUT_OF_MEMORY);
  CHECK(s->closed);
  CHECK(s->destroyed == s->worker);
  CHECK(!s->wrong_thread);
  const auto t = telemetry(a);
  CHECK(t.cleanup_completed == 1);
  CHECK(t.live_allocations == 0);
  CHECK(t.live_handles == 0);
  CHECK(t.logical_bytes == 0);
  CHECK(t.allocations_created == t.allocations_released);
  CHECK(t.handles_created == t.handles_destroyed);
}
void transposed_strided_gemm() {
  auto s = std::make_shared<State>();
  Adapter a(factory(s));
  auto cfg = config();
  CHECK(a.initialize(&cfg) == 0);
  void* handle = nullptr;
  CHECK(a.blas_create(&handle) == 0);
  constexpr int m = 3, n = 2, k = 4, ldc = 5;
  constexpr std::size_t prefix = 2, elements = 128;
  constexpr float sentinel = -777;
  for (std::uint32_t oa = 0; oa < 2; ++oa)
    for (std::uint32_t ob = 0; ob < 2; ++ob) {
      const int lda = (oa == 0 ? m : k) + 2, ldb = (ob == 0 ? k : n) + 1;
      std::vector<float> av(elements, sentinel), bv(elements, sentinel), cv(elements, sentinel);
      for (int i = 0; i < m; ++i)
        for (int kk = 0; kk < k; ++kk) {
          const auto index = static_cast<std::size_t>(oa == 0 ? kk * lda + i : i * lda + kk);
          av[prefix + index] = static_cast<float>(i + kk + 1);
        }
      for (int j = 0; j < n; ++j)
        for (int kk = 0; kk < k; ++kk) {
          const auto index = static_cast<std::size_t>(ob == 0 ? j * ldb + kk : kk * ldb + j);
          bv[prefix + index] = static_cast<float>(2 * kk - j - 1);
        }
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
          cv[prefix + static_cast<std::size_t>(j * ldc + i)] = 3;
      auto expected = cv;
      const float alpha = 2, beta = oa == ob ? 0.0F : 1.0F;
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
          float sum = 0;
          for (int kk = 0; kk < k; ++kk)
            sum += static_cast<float>((i + kk + 1) * (2 * kk - j - 1));
          expected[prefix + static_cast<std::size_t>(j * ldc + i)] = alpha * sum + beta * 3;
        }
      void* ap = nullptr;
      void* bp = nullptr;
      void* cp = nullptr;
      const auto bytes = elements * sizeof(float);
      CHECK(a.malloc_device(&ap, bytes) == 0);
      CHECK(a.malloc_device(&bp, bytes) == 0);
      CHECK(a.malloc_device(&cp, bytes) == 0);
      CHECK(a.memcpy(ap, av.data(), bytes, 1) == 0);
      CHECK(a.memcpy(bp, bv.data(), bytes, 1) == 0);
      CHECK(a.memcpy(cp, cv.data(), bytes, 1) == 0);
      const auto offset = [](void* p) {
        return reinterpret_cast<float*>(reinterpret_cast<std::uintptr_t>(p) +
                                        prefix * sizeof(float));
      };
      CHECK(a.blas_sgemm(handle, oa, ob, m, n, k, &alpha, offset(ap), lda, offset(bp), ldb, &beta,
                         offset(cp), ldc) == 0);
      CHECK(a.memcpy(cv.data(), cp, bytes, 2) == 0);
      CHECK(cv == expected);
      const auto count = s->backend_calls;
      const float nonfinite = std::numeric_limits<float>::infinity();
      CHECK(a.blas_sgemm(handle, oa, ob, m, n, k, &nonfinite, offset(ap), lda, offset(bp), ldb,
                         &beta, offset(cp), ldc) == XVRAM_STATUS_UNSUPPORTED);
      const auto misaligned =
          reinterpret_cast<float*>(reinterpret_cast<std::uintptr_t>(offset(ap)) + 1);
      CHECK(a.blas_sgemm(handle, oa, ob, m, n, k, &alpha, misaligned, lda, offset(bp), ldb, &beta,
                         offset(cp), ldc) == XVRAM_STATUS_INVALID_ARGUMENT);
      CHECK(s->backend_calls == count);
      CHECK(a.memcpy(cv.data(), cp, bytes, 2) == 0);
      CHECK(cv == expected);
      CHECK(a.free_device(ap) == 0);
      CHECK(a.free_device(bp) == 0);
      CHECK(a.free_device(cp) == 0);
    }
  CHECK(a.shutdown() == 0);
}
} // namespace
int main() {
  lifecycle_and_ranges();
  gemm_validation_and_concurrency();
  failures_and_cleanup();
  transposed_strided_gemm();
  if (failures)
    std::cerr << failures << " CUDA compatibility tests failed\n";
  return failures ? 1 : 0;
}
