#include "cuda_compat/backend.hpp"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <filesystem>
#include <future>
#include <limits>
#include <map>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace xvram::cuda_compat {
namespace {
sdk::Error invalid(const char* operation, const char* message) {
  return sdk::make_error(XVRAM_STATUS_INVALID_ARGUMENT, "compat_preflight", operation, message);
}
sdk::Error unsupported(const char* operation, const char* message) {
  return sdk::make_error(XVRAM_STATUS_UNSUPPORTED, "compat_preflight", operation, message);
}
template <class T> bool has_size(const T* value) {
  return value && value->struct_size >= sizeof(T);
}
template <class T, std::size_t N> bool zero(const T (&value)[N]) {
  return std::all_of(std::begin(value), std::end(value), [](T item) { return item == 0; });
}
bool valid_config(const xvram_cuda_compat_config_v1& value) {
  const auto& s = value.session;
  return value.flags == 0 && zero(value.reserved) && s.struct_size >= sizeof(s) && s.flags == 0 &&
         s.device_ordinal >= 0 && s.context_mode == XVRAM_CONTEXT_ISOLATED &&
         s.chunk_size_bytes > 0 && (s.chunk_size_bytes & (s.chunk_size_bytes - 1)) == 0 &&
         s.staging_slots >= 2 && s.staging_slots <= 8 && s.prefetch_distance <= 8 &&
         s.budget_poll_ms > 0 && s.cache_policy <= XVRAM_CACHE_POLICY_LRU && s.reserved0 == 0 &&
         zero(s.reserved) &&
         s.budget_poll_ms <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) &&
         s.max_transaction_ms <= 250 && value.stall_timeout_ms > 0 &&
         value.stall_timeout_ms <=
             static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
}
bool terminated(const char* value, std::size_t bytes) {
  return std::memchr(value, 0, bytes) != nullptr;
}
std::filesystem::path utf8_path(const char* value) {
  const auto* begin = reinterpret_cast<const char8_t*>(value);
  return std::filesystem::path(std::u8string(begin, begin + std::strlen(value)));
}
} // namespace

class Adapter::Impl {
public:
  explicit Impl(BackendFactory factory) : factory_(std::move(factory)) {
    telemetry_.struct_size = sizeof(telemetry_);
    telemetry_.runtime.struct_size = sizeof(telemetry_.runtime);
  }
  struct Entry {
    Allocation allocation;
    std::uint64_t reserved_bytes = 0;
    bool live = true;
  };
  struct Handle {
    std::uint32_t math_mode = 0;
    bool live = true;
  };

  xvram_status record(const sdk::Error& error, bool rejected = false) noexcept {
    if (!error)
      return XVRAM_STATUS_SUCCESS;
    try {
      std::lock_guard lock(snapshot_mutex_);
      error_ = error;
      if (rejected)
        ++telemetry_.calls_rejected;
    } catch (...) {
    }
    return error.status;
  }
  xvram_status reject_initialize(const sdk::Error& error) noexcept {
    {
      std::lock_guard lock(snapshot_mutex_);
      ++telemetry_.calls_attempted;
    }
    return record(error, true);
  }
  void state(std::uint32_t value) {
    std::lock_guard lock(snapshot_mutex_);
    telemetry_.state = value;
  }
  std::uint32_t state() const {
    std::lock_guard lock(snapshot_mutex_);
    return telemetry_.state;
  }
  void snapshot(const xvram_cuda_compat_telemetry_v1& source) {
    std::lock_guard lock(snapshot_mutex_);
    telemetry_.runtime = source.runtime;
    telemetry_.device_ordinal = source.device_ordinal;
    telemetry_.device_count = source.device_count;
    std::copy(std::begin(source.device_name), std::end(source.device_name), telemetry_.device_name);
    telemetry_.total_vram_bytes = source.total_vram_bytes;
    telemetry_.host_physical_bytes = source.host_physical_bytes;
    telemetry_.host_available_bytes = source.host_available_bytes;
    telemetry_.host_store_cap_bytes = source.host_store_cap_bytes;
    telemetry_.host_headroom_bytes = source.host_headroom_bytes;
    telemetry_.effective_chunk_bytes = source.effective_chunk_bytes;
    telemetry_.tiles_submitted = source.tiles_submitted;
    telemetry_.tiles_retired = source.tiles_retired;
    telemetry_.retired_va_reservations = source.retired_va_reservations;
    telemetry_.retired_va_reservations_freed = source.retired_va_reservations_freed;
    telemetry_.retired_va_bytes = source.retired_va_bytes;
    telemetry_.retired_va_bytes_freed = source.retired_va_bytes_freed;
    telemetry_.host_backing_bytes = source.host_backing_bytes;
    telemetry_.host_backing_peak_bytes = source.host_backing_peak_bytes;
    telemetry_.host_budget_bytes = source.host_budget_bytes;
    telemetry_.cleanup_operations_drained = source.cleanup_operations_drained;
    telemetry_.cleanup_events_drained = source.cleanup_events_drained;
    telemetry_.cleanup_completed = source.cleanup_completed;
    telemetry_.quarantined = source.quarantined;
    // A repeated telemetry poll is not progress. Only backend work changes its sequence.
    if (source.progress_sequence > backend_progress_) {
      telemetry_.progress_sequence += source.progress_sequence - backend_progress_;
      backend_progress_ = source.progress_sequence;
    }
  }
  template <class F>
  xvram_status call(const char* operation, F&& function, bool ready = true) noexcept {
    try {
      {
        std::lock_guard lock(snapshot_mutex_);
        ++telemetry_.calls_attempted;
      }
      auto promise = std::make_shared<std::promise<xvram_status>>();
      auto future = promise->get_future();
      {
        std::lock_guard lock(queue_mutex_);
        if (!accepting_) {
          const auto current = state();
          return record(
              sdk::make_error(
                  current == XVRAM_CUDA_COMPAT_UNINITIALIZED ? XVRAM_STATUS_NOT_READY
                                                             : XVRAM_STATUS_CLOSED,
                  "compat", operation,
                  "explicit initialization is required and a closed adapter cannot reopen"),
              true);
        }
        if (fail_next_enqueue_) {
          fail_next_enqueue_ = false;
          throw std::bad_alloc();
        }
        jobs_.emplace_back([this, operation, ready, promise,
                            fn = std::forward<F>(function)]() mutable {
          xvram_status result = XVRAM_STATUS_INTERNAL;
          try {
            const auto current = state();
            if (ready && current != XVRAM_CUDA_COMPAT_READY) {
              result = record(
                  sdk::make_error(current == XVRAM_CUDA_COMPAT_POISONED ? XVRAM_STATUS_POISONED
                                                                        : XVRAM_STATUS_CLOSED,
                                  "compat", operation,
                                  "adapter does not admit data operations in this lifecycle state"),
                  true);
            } else {
              {
                std::lock_guard lock(snapshot_mutex_);
                ++telemetry_.calls_submitted;
              }
              const sdk::Error error = fn();
              result = record(error, static_cast<bool>(error));
              if (!error) {
                std::lock_guard lock(snapshot_mutex_);
                ++telemetry_.calls_completed;
              }
            }
          } catch (...) {
            result = record(sdk::exception_error(operation), true);
            // Backend exceptions have an unknown mutation boundary; metadata/preflight
            // allocation failures have not started backend work.
            if (mutation_active_)
              state(XVRAM_CUDA_COMPAT_POISONED);
            mutation_active_ = false;
          }
          promise->set_value(result);
        });
      }
      queue_condition_.notify_one();
      return future.get();
    } catch (...) {
      return record(sdk::exception_error(operation), true);
    }
  }
  sdk::Error result(BackendResult value) {
    if (value.error && value.poison) {
      state(XVRAM_CUDA_COMPAT_POISONED);
    }
    return value.error;
  }
  template <class F> sdk::Error perform(F&& fn) {
    mutation_active_ = true;
    auto value = fn();
    mutation_active_ = false;
    return result(std::move(value));
  }
  sdk::Error close_backend() noexcept {
    sdk::Error error;
    try {
      if (backend_)
        error = result(backend_->close());
    } catch (...) {
      error = sdk::exception_error("backend_close");
    }
    // This helper is only dispatched on the worker, including when close throws.
    backend_.reset();
    return error;
  }
  void finish_close(const sdk::Error& error) {
    {
      std::lock_guard ownership(ownership_mutex_);
      for (auto& [address, entry] : allocations_) {
        (void)address;
        entry.live = false;
      }
      for (auto& h : handles_)
        h->live = false;
    }
    std::lock_guard lock(snapshot_mutex_);
    if (!error) {
      telemetry_.allocations_released = telemetry_.allocations_created;
      telemetry_.handles_destroyed = telemetry_.handles_created;
      telemetry_.live_allocations = 0;
      telemetry_.live_handles = 0;
      telemetry_.logical_bytes = 0;
      telemetry_.cleanup_completed = 1;
    }
    telemetry_.state = XVRAM_CUDA_COMPAT_CLOSED;
  }
  void worker_main() {
    for (;;) {
      std::function<void()> job;
      {
        std::unique_lock lock(queue_mutex_);
        queue_condition_.wait(lock, [this] { return stopping_ || !jobs_.empty(); });
        if (jobs_.empty() && stopping_)
          break;
        job = std::move(jobs_.front());
        jobs_.pop_front();
      }
      job();
    }
    // Cleanup must not depend on allocating another promise/queue node. Even a persistent
    // host allocation failure while scheduling shutdown leaves destruction on this worker.
    if (backend_) {
      const auto error = close_backend();
      (void)record(error);
      finish_close(error);
    }
  }
  void stop_worker() {
    {
      std::lock_guard lock(queue_mutex_);
      accepting_ = false;
      stopping_ = true;
    }
    queue_condition_.notify_all();
    if (worker_.joinable())
      worker_.join();
  }
  Entry* lookup(const void* pointer, std::uint64_t length) {
    std::lock_guard ownership(ownership_mutex_);
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(pointer));
    auto entry = allocations_.upper_bound(address);
    if (entry == allocations_.begin())
      return nullptr;
    --entry;
    const auto offset = address - entry->first;
    return entry->second.live && offset <= entry->second.allocation.bytes &&
                   length <= entry->second.allocation.bytes - offset
               ? &entry->second
               : nullptr;
  }
  bool touches_managed(const void* pointer, std::uint64_t length) const {
    std::lock_guard ownership(ownership_mutex_);
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(pointer));
    const auto extent = std::max<std::uint64_t>(1, length);
    if (extent > UINT64_MAX - address)
      return true;
    const auto end = address + extent;
    for (const auto& [base, entry] : allocations_) {
      if (base >= end)
        break;
      if (base + entry.reserved_bytes > address)
        return true;
    }
    return false;
  }
  Handle* handle(void* pointer) const {
    std::lock_guard ownership(ownership_mutex_);
    const auto found = std::find_if(handles_.begin(), handles_.end(),
                                    [pointer](const auto& h) { return h.get() == pointer; });
    return found != handles_.end() && (*found)->live ? found->get() : nullptr;
  }
  bool host(const void* pointer, std::uint64_t bytes) const {
    std::lock_guard ownership(ownership_mutex_);
    if (!pointer || touches_managed(pointer, bytes))
      return false;
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    const auto extent = std::max<std::uint64_t>(bytes, 1);
    if (extent > UINTPTR_MAX - begin)
      return false;
    for (const auto& h : handles_) {
      const auto value = reinterpret_cast<std::uintptr_t>(h.get());
      if (begin < value + sizeof(Handle) && value < begin + extent)
        return false;
    }
    return true;
  }
  bool capture_scalars(const float* alpha, const float* beta, float& a, float& b) const noexcept {
    try {
      std::lock_guard ownership(ownership_mutex_);
      if (!host(alpha, sizeof(float)) || !host(beta, sizeof(float)))
        return false;
      std::memcpy(&a, alpha, sizeof(a));
      std::memcpy(&b, beta, sizeof(b));
      return true;
    } catch (...) {
      return false;
    }
  }
  BackendFactory factory_;
  std::unique_ptr<Backend> backend_;
  xvram_cuda_compat_config_v1 config_ = XVRAM_CUDA_COMPAT_CONFIG_V1_INIT;
  mutable std::mutex snapshot_mutex_;
  xvram_cuda_compat_telemetry_v1 telemetry_{};
  sdk::Error error_;
  std::uint64_t backend_progress_ = 0;
  std::mutex lifecycle_mutex_;
  std::mutex queue_mutex_;
  std::condition_variable queue_condition_;
  std::deque<std::function<void()>> jobs_;
  std::thread worker_;
  bool attempted_ = false;
  bool accepting_ = false;
  bool stopping_ = false;
  bool fail_next_enqueue_ = false;
  bool mutation_active_ = false;
  std::map<std::uint64_t, Entry> allocations_;
  std::vector<std::unique_ptr<Handle>> handles_;
  mutable std::recursive_mutex ownership_mutex_;
};

Adapter::Adapter(BackendFactory factory) : impl_(std::make_unique<Impl>(std::move(factory))) {}
Adapter::~Adapter() {
  (void)shutdown();
}
void Adapter::fail_next_enqueue_for_testing() noexcept {
  std::lock_guard lock(impl_->queue_mutex_);
  impl_->fail_next_enqueue_ = true;
}

xvram_status Adapter::initialize(const xvram_cuda_compat_config_v1* config) noexcept {
  try {
    if (!has_size(config))
      return impl_->reject_initialize(invalid("initialize", "configuration prefix is missing"));
    if (!valid_config(*config))
      return impl_->reject_initialize(
          invalid("initialize", "unsupported configuration or nonzero reserved fields"));
    if (!terminated(config->cublas_library, sizeof(config->cublas_library)) ||
        !terminated(config->cublas_lt_library, sizeof(config->cublas_lt_library)))
      return impl_->reject_initialize(
          invalid("initialize", "library paths must be NUL terminated"));
    const bool core = config->cublas_library[0] != 0, lt = config->cublas_lt_library[0] != 0;
    if (core != lt || (core && (!utf8_path(config->cublas_library).is_absolute() ||
                                !utf8_path(config->cublas_lt_library).is_absolute())))
      return impl_->reject_initialize(
          invalid("initialize", "library paths must be an absolute core/Lt pair"));
    std::lock_guard lifecycle(impl_->lifecycle_mutex_);
    if (impl_->attempted_)
      return impl_->reject_initialize(
          sdk::make_error(XVRAM_STATUS_CLOSED, "compat", "initialize",
                          "only one initialization attempt is permitted"));
    impl_->attempted_ = true;
    impl_->config_ = *config;
    impl_->state(XVRAM_CUDA_COMPAT_INITIALIZING);
    impl_->worker_ = std::thread([this] { impl_->worker_main(); });
    {
      std::lock_guard lock(impl_->queue_mutex_);
      impl_->accepting_ = true;
    }
    const auto status = impl_->call(
        "initialize",
        [this] {
          impl_->backend_ = impl_->factory_();
          if (!impl_->backend_)
            return sdk::make_error(XVRAM_STATUS_INTERNAL, "compat", "initialize",
                                   "backend factory returned null");
          auto error = impl_->backend_->initialize(
              impl_->config_, [this](const auto& value) { impl_->snapshot(value); });
          if (!error)
            impl_->state(XVRAM_CUDA_COMPAT_READY);
          else {
            (void)impl_->close_backend();
            impl_->state(XVRAM_CUDA_COMPAT_CLOSED);
          }
          return error;
        },
        false);
    if (status != XVRAM_STATUS_SUCCESS) {
      // Cleanup/destruction must remain confined to the CUDA worker, including exceptions.
      (void)impl_->call(
          "initialize_cleanup",
          [this] {
            (void)impl_->close_backend();
            impl_->state(XVRAM_CUDA_COMPAT_CLOSED);
            return sdk::Error{};
          },
          false);
      impl_->stop_worker();
    }
    return status;
  } catch (...) {
    impl_->state(XVRAM_CUDA_COMPAT_CLOSED);
    return impl_->record(sdk::exception_error("initialize"), true);
  }
}

xvram_status Adapter::shutdown() noexcept {
  try {
    std::lock_guard lifecycle(impl_->lifecycle_mutex_);
    bool accepting = false;
    {
      std::lock_guard lock(impl_->queue_mutex_);
      accepting = impl_->accepting_;
    }
    if (!accepting)
      return XVRAM_STATUS_SUCCESS;
    const auto status = impl_->call(
        "shutdown",
        [this] {
          auto error = impl_->close_backend();
          impl_->finish_close(error);
          return error;
        },
        false);
    impl_->stop_worker();
    return status;
  } catch (...) {
    return impl_->record(sdk::exception_error("shutdown"));
  }
}
xvram_status Adapter::get_error(xvram_error_info_v1* output) const noexcept {
  if (!has_size(output))
    return XVRAM_STATUS_INVALID_ARGUMENT;
  try {
    std::lock_guard lock(impl_->snapshot_mutex_);
    sdk::write_error_info(impl_->error_, *output);
    return XVRAM_STATUS_SUCCESS;
  } catch (...) {
    return XVRAM_STATUS_INTERNAL;
  }
}
xvram_status Adapter::get_telemetry(xvram_cuda_compat_telemetry_v1* output) const noexcept {
  if (!has_size(output))
    return XVRAM_STATUS_INVALID_ARGUMENT;
  try {
    std::lock_guard lock(impl_->snapshot_mutex_);
    const auto size = output->struct_size;
    *output = impl_->telemetry_;
    output->struct_size = size;
    return XVRAM_STATUS_SUCCESS;
  } catch (...) {
    return XVRAM_STATUS_INTERNAL;
  }
}
xvram_status Adapter::malloc_device(void** output, std::uint64_t bytes) noexcept {
  return impl_->call("cudaMalloc", [this, output, bytes] {
    if (!impl_->host(output, sizeof(*output)) || bytes == 0 ||
        bytes > UINT64_MAX - (impl_->config_.session.chunk_size_bytes - 1))
      return invalid("cudaMalloc",
                     "a host output pointer and positive representable size are required");
    Allocation allocation;
    if (auto error = impl_->perform([&] { return impl_->backend_->allocate(bytes, allocation); });
        error)
      return error;
    std::uint64_t chunk = 0;
    {
      std::lock_guard lock(impl_->snapshot_mutex_);
      chunk = impl_->telemetry_.effective_chunk_bytes;
    }
    if (chunk == 0)
      chunk = impl_->config_.session.chunk_size_bytes;
    if (bytes > UINT64_MAX - (chunk - 1)) {
      (void)impl_->perform([&] { return impl_->backend_->release(allocation.id); });
      return invalid("cudaMalloc", "allocation reservation rounding overflows");
    }
    const auto reserved = ((bytes + chunk - 1) / chunk) * chunk;
    if (allocation.address == 0 || allocation.bytes != bytes ||
        reserved > UINT64_MAX - allocation.address ||
        impl_->touches_managed(
            reinterpret_cast<void*>(static_cast<std::uintptr_t>(allocation.address)), reserved)) {
      (void)impl_->perform([&] { return impl_->backend_->release(allocation.id); });
      impl_->state(XVRAM_CUDA_COMPAT_POISONED);
      return sdk::make_error(XVRAM_STATUS_CORRUPTION, "compat", "cudaMalloc",
                             "backend reused an owned or retired virtual range");
    }
    try {
      std::lock_guard ownership(impl_->ownership_mutex_);
      impl_->allocations_.emplace(allocation.address, Impl::Entry{allocation, reserved, true});
    } catch (...) {
      (void)impl_->perform([&] { return impl_->backend_->release(allocation.id); });
      return sdk::exception_error("allocation_registry");
    }
    {
      std::lock_guard lock(impl_->snapshot_mutex_);
      ++impl_->telemetry_.allocations_created;
      ++impl_->telemetry_.live_allocations;
      impl_->telemetry_.logical_bytes += bytes;
      impl_->telemetry_.logical_bytes_peak =
          std::max(impl_->telemetry_.logical_bytes_peak, impl_->telemetry_.logical_bytes);
    }
    void* address = reinterpret_cast<void*>(static_cast<std::uintptr_t>(allocation.address));
    std::memcpy(output, &address, sizeof(address));
    return sdk::Error{};
  });
}
xvram_status Adapter::free_device(void* pointer) noexcept {
  return impl_->call("cudaFree", [this, pointer] {
    if (!pointer)
      return sdk::Error{};
    auto* entry = impl_->lookup(pointer, 1);
    if (!entry || entry->allocation.address != reinterpret_cast<std::uintptr_t>(pointer))
      return invalid("cudaFree", "pointer is not a live allocation base");
    if (auto error = impl_->perform([&] { return impl_->backend_->release(entry->allocation.id); });
        error)
      return error;
    {
      std::lock_guard ownership(impl_->ownership_mutex_);
      entry->live = false;
    }
    {
      std::lock_guard lock(impl_->snapshot_mutex_);
      ++impl_->telemetry_.allocations_released;
      --impl_->telemetry_.live_allocations;
      impl_->telemetry_.logical_bytes -= entry->allocation.bytes;
    }
    return sdk::Error{};
  });
}
xvram_status Adapter::memcpy(void* destination, const void* source, std::uint64_t bytes,
                             std::uint32_t kind) noexcept {
  return impl_->call("cudaMemcpy", [this, destination, source, bytes, kind] {
    if (kind != XVRAM_CUDA_COMPAT_H2D && kind != XVRAM_CUDA_COMPAT_D2H)
      return unsupported("cudaMemcpy",
                         "only explicit host-to-device and device-to-host copies are supported");
    if (bytes == 0)
      return sdk::Error{};
    const void* device = kind == XVRAM_CUDA_COMPAT_H2D ? destination : source;
    const void* host = kind == XVRAM_CUDA_COMPAT_H2D ? source : destination;
    auto* entry = impl_->lookup(device, bytes);
    if (!entry || !impl_->host(host, bytes))
      return invalid("cudaMemcpy", "copy must pair one live managed range with a host range");
    const auto offset = reinterpret_cast<std::uintptr_t>(device) - entry->allocation.address;
    auto error = impl_->perform([&] {
      return kind == XVRAM_CUDA_COMPAT_H2D
                 ? impl_->backend_->write(entry->allocation.id, offset, source, bytes)
                 : impl_->backend_->read(entry->allocation.id, offset, destination, bytes);
    });
    if (error)
      return error;
    {
      std::lock_guard lock(impl_->snapshot_mutex_);
      (kind == XVRAM_CUDA_COMPAT_H2D ? impl_->telemetry_.h2d_bytes : impl_->telemetry_.d2h_bytes) +=
          bytes;
    }
    return sdk::Error{};
  });
}
xvram_status Adapter::get_device(std::int32_t* output) noexcept {
  return impl_->call("cudaGetDevice", [this, output] {
    if (!impl_->host(output, sizeof(*output)))
      return invalid("cudaGetDevice", "host output required");
    *output = impl_->config_.session.device_ordinal;
    return sdk::Error{};
  });
}
xvram_status Adapter::get_device_count(std::int32_t* output) noexcept {
  return impl_->call("cudaGetDeviceCount", [this, output] {
    if (!impl_->host(output, sizeof(*output)))
      return invalid("cudaGetDeviceCount", "host output required");
    std::lock_guard lock(impl_->snapshot_mutex_);
    *output = static_cast<std::int32_t>(impl_->telemetry_.device_count);
    return sdk::Error{};
  });
}
xvram_status Adapter::set_device(std::int32_t device) noexcept {
  return impl_->call("cudaSetDevice", [this, device] {
    return device == impl_->config_.session.device_ordinal
               ? sdk::Error{}
               : unsupported("cudaSetDevice", "only the initialized device may be selected");
  });
}
xvram_status Adapter::synchronize() noexcept {
  return impl_->call("cudaDeviceSynchronize", [this] {
    return impl_->perform([&] { return impl_->backend_->synchronize(); });
  });
}
xvram_status Adapter::mem_get_info(std::uint64_t* free_bytes, std::uint64_t* total_bytes) noexcept {
  return impl_->call("cudaMemGetInfo", [this, free_bytes, total_bytes] {
    if (!impl_->host(free_bytes, sizeof(*free_bytes)) ||
        !impl_->host(total_bytes, sizeof(*total_bytes)) || free_bytes == total_bytes)
      return invalid("cudaMemGetInfo", "distinct host outputs required");
    std::lock_guard lock(impl_->snapshot_mutex_);
    *free_bytes = impl_->telemetry_.runtime.cuda_free_bytes_end;
    *total_bytes = impl_->telemetry_.total_vram_bytes;
    return sdk::Error{};
  });
}
xvram_status Adapter::blas_create(void** output) noexcept {
  return impl_->call("cublasCreate", [this, output] {
    if (!impl_->host(output, sizeof(*output)))
      return invalid("cublasCreate", "host output required");
    auto value = std::make_unique<Impl::Handle>();
    auto* pointer = value.get();
    {
      std::lock_guard ownership(impl_->ownership_mutex_);
      impl_->handles_.push_back(std::move(value));
    }
    {
      std::lock_guard lock(impl_->snapshot_mutex_);
      ++impl_->telemetry_.handles_created;
      ++impl_->telemetry_.live_handles;
    }
    void* opaque = pointer;
    std::memcpy(output, &opaque, sizeof(opaque));
    return sdk::Error{};
  });
}
xvram_status Adapter::blas_destroy(void* pointer) noexcept {
  return impl_->call("cublasDestroy", [this, pointer] {
    auto* handle = impl_->handle(pointer);
    if (!handle)
      return invalid("cublasDestroy", "unknown or destroyed handle");
    {
      std::lock_guard ownership(impl_->ownership_mutex_);
      handle->live = false;
    }
    {
      std::lock_guard lock(impl_->snapshot_mutex_);
      ++impl_->telemetry_.handles_destroyed;
      --impl_->telemetry_.live_handles;
    }
    return sdk::Error{};
  });
}
xvram_status Adapter::blas_get_version(void* pointer, std::int32_t* output) noexcept {
  return impl_->call("cublasGetVersion", [this, pointer, output] {
    if (!impl_->handle(pointer) || !impl_->host(output, sizeof(*output)))
      return invalid("cublasGetVersion", "live handle and host output required");
    std::lock_guard lock(impl_->snapshot_mutex_);
    *output = static_cast<std::int32_t>(impl_->telemetry_.runtime.cublas_version);
    return sdk::Error{};
  });
}
xvram_status Adapter::blas_set_stream(void* pointer, std::uint64_t stream) noexcept {
  return impl_->call("cublasSetStream", [this, pointer, stream] {
    if (!impl_->handle(pointer))
      return invalid("cublasSetStream", "live handle required");
    return stream == 0 ? sdk::Error{}
                       : unsupported("cublasSetStream", "only the default stream is supported");
  });
}
xvram_status Adapter::blas_get_stream(void* pointer, std::uint64_t* output) noexcept {
  return impl_->call("cublasGetStream", [this, pointer, output] {
    if (!impl_->handle(pointer) || !impl_->host(output, sizeof(*output)))
      return invalid("cublasGetStream", "live handle and host output required");
    const std::uint64_t zero_value = 0;
    std::memcpy(output, &zero_value, sizeof(zero_value));
    return sdk::Error{};
  });
}
xvram_status Adapter::blas_set_pointer_mode(void* pointer, std::uint32_t mode) noexcept {
  return impl_->call("cublasSetPointerMode", [this, pointer, mode] {
    if (!impl_->handle(pointer))
      return invalid("cublasSetPointerMode", "live handle required");
    return mode == 0 ? sdk::Error{}
                     : unsupported("cublasSetPointerMode", "only host pointer mode is supported");
  });
}
xvram_status Adapter::blas_get_pointer_mode(void* pointer, std::uint32_t* output) noexcept {
  return impl_->call("cublasGetPointerMode", [this, pointer, output] {
    if (!impl_->handle(pointer) || !impl_->host(output, sizeof(*output)))
      return invalid("cublasGetPointerMode", "live handle and host output required");
    const std::uint32_t mode = 0;
    std::memcpy(output, &mode, sizeof(mode));
    return sdk::Error{};
  });
}
xvram_status Adapter::blas_set_math_mode(void* pointer, std::uint32_t mode) noexcept {
  return impl_->call("cublasSetMathMode", [this, pointer, mode] {
    auto* h = impl_->handle(pointer);
    if (!h)
      return invalid("cublasSetMathMode", "live handle required");
    if (mode != 0 && mode != 2)
      return unsupported("cublasSetMathMode",
                         "only default and pedantic FP32 math modes are supported");
    h->math_mode = mode;
    return sdk::Error{};
  });
}
xvram_status Adapter::blas_get_math_mode(void* pointer, std::uint32_t* output) noexcept {
  return impl_->call("cublasGetMathMode", [this, pointer, output] {
    auto* h = impl_->handle(pointer);
    if (!h || !impl_->host(output, sizeof(*output)))
      return invalid("cublasGetMathMode", "live handle and host output required");
    std::memcpy(output, &h->math_mode, sizeof(h->math_mode));
    return sdk::Error{};
  });
}
xvram_status Adapter::blas_sgemm(void* pointer, std::uint32_t op_a, std::uint32_t op_b,
                                 std::int32_t m, std::int32_t n, std::int32_t k, const float* alpha,
                                 const float* a, std::int32_t lda, const float* b, std::int32_t ldb,
                                 const float* beta, float* c, std::int32_t ldc) noexcept {
  // Host pointer mode captures scalar values on the calling thread. Registry checks are
  // synchronized with allocation/handle publication so managed/tombstoned values are never read.
  const bool initialized = impl_->state() == XVRAM_CUDA_COMPAT_READY;
  float alpha_value = 0, beta_value = 0;
  const bool host_scalars =
      initialized && impl_->capture_scalars(alpha, beta, alpha_value, beta_value);
  return impl_->call("cublasSgemm", [=, this] {
    if (!initialized)
      return sdk::make_error(XVRAM_STATUS_NOT_READY, "compat_preflight", "cublasSgemm",
                             "initialization must finish before submitting a GEMM");
    if (!impl_->handle(pointer))
      return invalid("cublasSgemm", "live adapter handle required");
    if (op_a > 1 || op_b > 1 || m <= 0 || n <= 0 || k <= 0)
      return unsupported("cublasSgemm", "positive dimensions and N/T operations are required");
    if (!host_scalars)
      return invalid("cublasSgemm", "alpha and beta must be host pointers");
    if (lda < (op_a == 0 ? m : k) || ldb < (op_b == 0 ? k : n) || ldc < m)
      return invalid("cublasSgemm", "leading dimension is too small");
    const auto* ae = impl_->lookup(a, sizeof(float));
    const auto* be = impl_->lookup(b, sizeof(float));
    const auto* ce = impl_->lookup(c, sizeof(float));
    if (!ae || !be || !ce)
      return invalid("cublasSgemm", "all matrices must be live adapter allocations");
    if ((reinterpret_cast<std::uintptr_t>(a) | reinterpret_cast<std::uintptr_t>(b) |
         reinterpret_cast<std::uintptr_t>(c)) %
            alignof(float) !=
        0)
      return invalid("cublasSgemm", "FP32 matrices must be aligned");
    gemm::GemmProblem problem;
    problem.m = static_cast<std::uint64_t>(m);
    problem.n = static_cast<std::uint64_t>(n);
    problem.k = static_cast<std::uint64_t>(k);
    const auto matrix = [](const Impl::Entry& e, const void* p, std::uint64_t rows,
                           std::uint64_t cols, std::int32_t ld) {
      return gemm::MatrixView{e.allocation.id,
                              e.allocation.bytes,
                              reinterpret_cast<std::uintptr_t>(p) - e.allocation.address,
                              rows,
                              cols,
                              static_cast<std::uint64_t>(ld),
                              gemm::MatrixLayout::column_major,
                              gemm::ElementType::fp32};
    };
    problem.a =
        matrix(*ae, a, op_a == 0 ? problem.m : problem.k, op_a == 0 ? problem.k : problem.m, lda);
    problem.b =
        matrix(*be, b, op_b == 0 ? problem.k : problem.n, op_b == 0 ? problem.n : problem.k, ldb);
    problem.c = matrix(*ce, c, problem.m, problem.n, ldc);
    problem.a_operation =
        op_a == 0 ? gemm::MatrixOperation::none : gemm::MatrixOperation::transpose;
    problem.b_operation =
        op_b == 0 ? gemm::MatrixOperation::none : gemm::MatrixOperation::transpose;
    problem.compute_mode = gemm::ComputeMode::strict_fp32;
    // Both supported cuBLAS math modes select exact FP32; implicit TF32 is never enabled.
    problem.alpha = alpha_value;
    problem.beta = beta_value;
    if (!std::isfinite(problem.alpha) || !std::isfinite(problem.beta))
      return unsupported("cublasSgemm", "finite host scalars are required");
    const auto validation = gemm::validate_problem(problem);
    if (!validation)
      return sdk::make_error(XVRAM_STATUS_INVALID_ARGUMENT, "compat_preflight", "cublasSgemm",
                             gemm::problem_error_name(validation.error));
    if (auto error = impl_->perform([&] { return impl_->backend_->gemm(problem); }); error)
      return error;
    {
      std::lock_guard lock(impl_->snapshot_mutex_);
      ++impl_->telemetry_.gemm_calls;
    }
    return sdk::Error{};
  });
}
} // namespace xvram::cuda_compat
