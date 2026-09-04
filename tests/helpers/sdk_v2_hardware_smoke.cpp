#include "xvram/xvram_v2.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

constexpr std::uint64_t mib = UINT64_C(1024) * UINT64_C(1024);
constexpr std::uint64_t chunk_bytes = UINT64_C(64) * mib;
// The configured target can hold at most ten chunks even if every codec reserve is reclaimed.
// Twelve chunks therefore guarantee a bounded oversubscription and physical-handle reuse without
// relying on an implementation-specific current frame count.
constexpr std::uint64_t chunk_count = 12;
constexpr std::uint64_t allocation_bytes = chunk_bytes * chunk_count;
constexpr std::uint64_t operation_timeout_ms = UINT64_C(120000);

#if defined(_WIN32)
using CudaMemsetD8Async = int(__stdcall*)(unsigned long long, unsigned char, std::size_t, void*);
#else
using CudaMemsetD8Async = int (*)(unsigned long long, unsigned char, std::size_t, void*);
#endif

class CudaDriver final {
public:
  CudaDriver() = default;
  CudaDriver(const CudaDriver&) = delete;
  CudaDriver& operator=(const CudaDriver&) = delete;

  ~CudaDriver() {
#if defined(_WIN32)
    if (module_ != nullptr) {
      (void)FreeLibrary(module_);
    }
#else
    if (module_ != nullptr) {
      (void)dlclose(module_);
    }
#endif
  }

  [[nodiscard]] bool load() noexcept {
#if defined(_WIN32)
    module_ = LoadLibraryW(L"nvcuda.dll");
    if (module_ == nullptr) {
      return false;
    }
    FARPROC symbol = GetProcAddress(module_, "cuMemsetD8Async_ptsz");
    static_assert(sizeof(symbol) == sizeof(memset_d8_async_));
    std::memcpy(&memset_d8_async_, &symbol, sizeof(memset_d8_async_));
    if (memset_d8_async_ == nullptr) {
      symbol = GetProcAddress(module_, "cuMemsetD8Async");
      std::memcpy(&memset_d8_async_, &symbol, sizeof(memset_d8_async_));
    }
#else
    module_ = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (module_ == nullptr) {
      return false;
    }
    void* symbol = dlsym(module_, "cuMemsetD8Async_ptsz");
    static_assert(sizeof(symbol) == sizeof(memset_d8_async_));
    std::memcpy(&memset_d8_async_, &symbol, sizeof(memset_d8_async_));
    if (memset_d8_async_ == nullptr) {
      symbol = dlsym(module_, "cuMemsetD8Async");
      std::memcpy(&memset_d8_async_, &symbol, sizeof(memset_d8_async_));
    }
#endif
    return memset_d8_async_ != nullptr;
  }

  [[nodiscard]] int memset_async(const std::uint64_t address, const unsigned char value,
                                 const std::uint64_t bytes,
                                 const std::uint64_t stream) const noexcept {
    if (memset_d8_async_ == nullptr || bytes > std::numeric_limits<std::size_t>::max()) {
      return -1;
    }
    return memset_d8_async_(static_cast<unsigned long long>(address), value,
                            static_cast<std::size_t>(bytes),
                            reinterpret_cast<void*>(static_cast<std::uintptr_t>(stream)));
  }

private:
#if defined(_WIN32)
  HMODULE module_ = nullptr;
#else
  void* module_ = nullptr;
#endif
  CudaMemsetD8Async memset_d8_async_ = nullptr;
};

struct Mutation {
  const CudaDriver* driver = nullptr;
  std::uint64_t expected_allocation_id = 0;
  unsigned char value = 0;
};

template <std::size_t Size> void copy_text(char (&destination)[Size], const char* source) noexcept {
  const std::size_t length = std::min<std::size_t>(std::strlen(source), Size - 1U);
  std::copy_n(source, length, destination);
  destination[length] = '\0';
}

void set_callback_error(xvram_error_info_v1* error, const int native_code,
                        const char* message) noexcept {
  if (error == nullptr || error->struct_size < sizeof(*error)) {
    return;
  }
  error->status = XVRAM_STATUS_CUDA_ERROR;
  error->native_domain = XVRAM_NATIVE_ERROR_CUDA;
  error->native_code = native_code;
  copy_text(error->stage, "sdk_v2_smoke");
  copy_text(error->operation, "cuMemsetD8Async");
  copy_text(error->native_name, "CUDA_ERROR");
  copy_text(error->message, message);
}

xvram_status XVRAM_CALL mutate_callback(const xvram_transaction_context_v1* context,
                                        void* user_data,
                                        xvram_error_info_v1* callback_error) noexcept {
  const auto* mutation = static_cast<const Mutation*>(user_data);
  if (context == nullptr || context->struct_size < sizeof(*context) || mutation == nullptr ||
      mutation->driver == nullptr || context->range_count != 1U || context->ranges == nullptr ||
      context->native_stream == 0U) {
    set_callback_error(callback_error, -1, "invalid transaction callback context");
    return XVRAM_STATUS_INVALID_ARGUMENT;
  }
  const xvram_resolved_range_v1& range = context->ranges[0];
  if (range.allocation_id != mutation->expected_allocation_id ||
      range.length_bytes != chunk_bytes || range.device_address == 0U ||
      range.mode != XVRAM_ACCESS_READ_WRITE) {
    set_callback_error(callback_error, -1, "resolved range does not match the submitted chunk");
    return XVRAM_STATUS_INVALID_ARGUMENT;
  }
  const int result = mutation->driver->memset_async(range.device_address, mutation->value,
                                                    range.length_bytes, context->native_stream);
  if (result != 0) {
    set_callback_error(callback_error, result, "CUDA memset submission failed");
    return XVRAM_STATUS_CUDA_ERROR;
  }
  return XVRAM_STATUS_SUCCESS;
}

xvram_status XVRAM_CALL observe_callback(const xvram_transaction_context_v1* context, void*,
                                         xvram_error_info_v1*) noexcept {
  return context != nullptr && context->struct_size >= sizeof(*context) &&
                 context->range_count == 1U && context->ranges != nullptr &&
                 context->native_stream != 0U
             ? XVRAM_STATUS_SUCCESS
             : XVRAM_STATUS_INVALID_ARGUMENT;
}

[[nodiscard]] bool parse_device(const std::string_view text, std::int32_t& output) noexcept {
  if (text.empty()) {
    return false;
  }
  std::int32_t parsed = 0;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || parsed < 0) {
    return false;
  }
  output = parsed;
  return true;
}

[[nodiscard]] bool api_table_is_complete(const xvram_api_v2& api) noexcept {
  return api.v1.struct_size == sizeof(api) && api.v1.abi_version == XVRAM_ABI_VERSION_2 &&
         api.v1.status_name != nullptr && api.v1.session_close != nullptr &&
         api.v1.session_release != nullptr && api.v1.allocation_create != nullptr &&
         api.v1.allocation_write != nullptr && api.v1.allocation_read != nullptr &&
         api.v1.allocation_release != nullptr && api.v1.transaction_execute != nullptr &&
         api.v1.transaction_submit != nullptr && api.v1.operation_poll != nullptr &&
         api.v1.operation_release != nullptr && api.v1.session_drain != nullptr &&
         api.session_create_v2 != nullptr && api.session_get_telemetry_v2 != nullptr;
}

[[nodiscard]] bool verify_fill(const std::vector<std::byte>& bytes,
                               const unsigned char expected) noexcept {
  return std::all_of(bytes.begin(), bytes.end(), [expected](const std::byte value) {
    return std::to_integer<unsigned char>(value) == expected;
  });
}

[[nodiscard]] bool consistent_h2d_snapshot(const xvram_session_telemetry_v2& value) noexcept {
  // The production runtime increments these together, and publishes snapshots at
  // retired work-item boundaries. The prefix and extension must share that epoch.
  // D2H is intentionally excluded: rejected compressed candidates add logical
  // D2H attempts without adding another completed v1 writeback.
  return value.v1.bytes_h2d == value.logical_h2d_bytes;
}

struct TelemetrySamples {
  std::uint64_t total = 0;
  std::uint64_t while_pending = 0;
};

[[nodiscard]] constexpr bool operation_is_terminal(const xvram_operation_state state) noexcept {
  return state == XVRAM_OPERATION_COMPLETED || state == XVRAM_OPERATION_CANCELLED ||
         state == XVRAM_OPERATION_FAILED;
}

[[nodiscard]] constexpr int quarantine_exit_code(const xvram_status status) noexcept {
  return status == XVRAM_STATUS_TIMEOUT ? 26 : 27;
}

[[noreturn]] void quarantine_pending_operation(const char* reason,
                                               const xvram_status status) noexcept {
  // operation_release only drops the public handle; it neither cancels nor drains the worker.
  // Do not unwind Mutation, unload its CUDA driver, or invoke session cleanup while a callback
  // may still use that state. This standalone hardware helper is the quarantine boundary.
  const int exit_code = quarantine_exit_code(status);
  (void)std::fprintf(stderr,
                     "SDK v2 smoke quarantine: %s (status=%d, exit=%d); operation completion "
                     "is unknown, terminating without callback-state cleanup\n",
                     reason, static_cast<int>(status), exit_code);
  (void)std::fflush(stderr);
  std::_Exit(exit_code);
}

[[nodiscard]] xvram_status execute_with_live_telemetry(const xvram_api_v2& api,
                                                       const xvram_session session,
                                                       const xvram_transaction_desc_v1& transaction,
                                                       TelemetrySamples& samples) {
  xvram_operation operation = nullptr;
  const xvram_status submitted = api.v1.transaction_submit(session, &transaction, &operation);
  if (submitted != XVRAM_STATUS_SUCCESS) {
    return submitted;
  }
  struct OperationGuard {
    const xvram_api_v2& api;
    xvram_operation operation;
    ~OperationGuard() {
      api.v1.operation_release(operation);
    }
  } guard{api, operation};
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(operation_timeout_ms);
  xvram_status telemetry_error = XVRAM_STATUS_SUCCESS;
  for (;;) {
    xvram_operation_info_v1 info = XVRAM_OPERATION_INFO_V1_INIT;
    const xvram_status polled = api.v1.operation_poll(operation, &info);
    if (polled != XVRAM_STATUS_SUCCESS) {
      quarantine_pending_operation("operation_poll failed", polled);
    }
    const bool pending =
        info.state == XVRAM_OPERATION_QUEUED || info.state == XVRAM_OPERATION_RUNNING;
    if (!pending && !operation_is_terminal(info.state)) {
      quarantine_pending_operation("operation_poll returned an unknown state",
                                   XVRAM_STATUS_INTERNAL);
    }
    xvram_session_telemetry_v2 snapshot{};
    snapshot.struct_size = sizeof(snapshot);
    const xvram_status observed = api.session_get_telemetry_v2(session, &snapshot);
    if (observed != XVRAM_STATUS_SUCCESS) {
      telemetry_error = observed;
    } else {
      ++samples.total;
      samples.while_pending += pending ? 1U : 0U;
      if (!consistent_h2d_snapshot(snapshot)) {
        telemetry_error = XVRAM_STATUS_INTERNAL;
      }
    }
    // A failed consistency observation must not return while the callback still
    // uses the caller's Mutation object. Retire the operation before failing.
    if (!pending) {
      return info.result == XVRAM_STATUS_SUCCESS ? telemetry_error : info.result;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      quarantine_pending_operation("operation exceeded its 120-second deadline",
                                   XVRAM_STATUS_TIMEOUT);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

void print_session_error(const xvram_api_v2& api, const xvram_session session,
                         const char* operation, const xvram_status status) {
  std::cerr << operation << " failed: " << api.v1.status_name(status);
  if (session != nullptr) {
    xvram_error_info_v1 error = XVRAM_ERROR_INFO_V1_INIT;
    if (api.v1.session_get_error(session, &error) == XVRAM_STATUS_SUCCESS &&
        error.status != XVRAM_STATUS_SUCCESS) {
      std::cerr << " [" << error.stage << '/' << error.operation << "] " << error.message;
      if (error.native_code != 0) {
        std::cerr << " (native " << error.native_code << ')';
      }
    }
  }
  std::cerr << '\n';
}

struct Handles {
  const xvram_api_v2* api = nullptr;
  xvram_session session = nullptr;
  xvram_allocation allocation = nullptr;

  ~Handles() {
    if (api == nullptr) {
      return;
    }
    if (allocation != nullptr) {
      (void)api->v1.allocation_release(allocation);
      allocation = nullptr;
    }
    if (session != nullptr) {
      (void)api->v1.session_close(session, operation_timeout_ms);
      api->v1.session_release(session);
      session = nullptr;
    }
  }
};

[[nodiscard]] int self_test() {
  static_assert(operation_is_terminal(XVRAM_OPERATION_COMPLETED));
  static_assert(operation_is_terminal(XVRAM_OPERATION_CANCELLED));
  static_assert(operation_is_terminal(XVRAM_OPERATION_FAILED));
  static_assert(!operation_is_terminal(XVRAM_OPERATION_QUEUED));
  static_assert(!operation_is_terminal(XVRAM_OPERATION_RUNNING));
  static_assert(!operation_is_terminal(static_cast<xvram_operation_state>(99)));
  static_assert(quarantine_exit_code(XVRAM_STATUS_TIMEOUT) == 26);
  static_assert(quarantine_exit_code(XVRAM_STATUS_CUDA_ERROR) == 27);
  xvram_api_v2 api{};
  if (xvram_get_api(XVRAM_ABI_VERSION_2, static_cast<std::uint32_t>(sizeof(api)), &api) !=
          XVRAM_STATUS_SUCCESS ||
      !api_table_is_complete(api)) {
    std::cerr << "SDK v2 API table self-test failed\n";
    return 1;
  }
  xvram_api_v1 api_v1{};
  if (xvram_get_api(XVRAM_ABI_VERSION_1, static_cast<std::uint32_t>(sizeof(api_v1)), &api_v1) !=
          XVRAM_STATUS_SUCCESS ||
      api_v1.struct_size != sizeof(api_v1) || api_v1.abi_version != XVRAM_ABI_VERSION_1 ||
      api.v1.status_name != api_v1.status_name || api.v1.session_create != api_v1.session_create ||
      api.v1.transaction_execute != api_v1.transaction_execute ||
      api.v1.allocation_read != api_v1.allocation_read ||
      api.v1.allocation_write != api_v1.allocation_write ||
      api.v1.session_release != api_v1.session_release ||
      !std::all_of(std::begin(api.reserved), std::end(api.reserved),
                   [](const std::uint64_t value) { return value == 0U; })) {
    std::cerr << "SDK v2 did not preserve the v1 table prefix\n";
    return 1;
  }
  std::vector<std::byte> fixture(257U, std::byte{0x5A});
  if (!verify_fill(fixture, 0x5A)) {
    return 1;
  }
  fixture[128] = std::byte{0xA5};
  if (verify_fill(fixture, 0x5A)) {
    return 1;
  }
  xvram_session_telemetry_v2 coherent{};
  coherent.v1.bytes_h2d = chunk_bytes;
  coherent.logical_h2d_bytes = chunk_bytes;
  coherent.v1.bytes_d2h = chunk_bytes;
  coherent.logical_d2h_bytes = 2U * chunk_bytes;
  if (!consistent_h2d_snapshot(coherent)) {
    return 1;
  }
  coherent.logical_h2d_bytes += chunk_bytes;
  if (consistent_h2d_snapshot(coherent)) {
    return 1;
  }
  std::cout << "xvram SDK v2 smoke contract: ok\n";
  return 0;
}

[[nodiscard]] int hardware_test(const std::int32_t device_ordinal) {
  xvram_api_v2 api{};
  const xvram_status get_api_status =
      xvram_get_api(XVRAM_ABI_VERSION_2, static_cast<std::uint32_t>(sizeof(api)), &api);
  if (get_api_status != XVRAM_STATUS_SUCCESS || !api_table_is_complete(api)) {
    std::cerr << "xvram_get_api(v2) failed\n";
    return 1;
  }

  CudaDriver cuda;
  if (!cuda.load()) {
    std::cerr << "CUDA driver or cuMemsetD8Async is unavailable\n";
    return 1;
  }

  Handles handles{&api};
  xvram_session_config_v2 config = XVRAM_SESSION_CONFIG_V2_INIT;
  config.v1.device_ordinal = device_ordinal;
  config.v1.chunk_size_bytes = chunk_bytes;
  config.v1.cache_target_bytes = UINT64_C(640) * mib;
  config.v1.device_headroom_bytes = UINT64_C(512) * mib;
  config.v1.workspace_cap_bytes = UINT64_C(4) * mib;
  config.v1.staging_slots = 4U;
  config.v1.prefetch_distance = 0U;
  config.compression_mode = XVRAM_COMPRESSION_CAPACITY;
  config.compression_codec = XVRAM_COMPRESSION_CODEC_LZ4;
  // Export/verification slots are admitted against the same exact host budget as authoritative
  // blobs. Two GiB keeps the bounded 768-MiB workload plus transient immutable generations safe.
  config.host_store_cap_bytes = UINT64_C(2048) * mib;
  config.host_headroom_bytes = UINT64_C(512) * mib;
  config.compression_workspace_cap_bytes = UINT64_C(256) * mib;
  config.codec_slots = 2U;
  config.codec_workers = 2U;

  xvram_status status = api.session_create_v2(&config, &handles.session);
  if (status != XVRAM_STATUS_SUCCESS) {
    print_session_error(api, handles.session, "session_create_v2", status);
    return 1;
  }

  xvram_allocation_desc_v1 allocation_desc = XVRAM_ALLOCATION_DESC_V1_INIT;
  allocation_desc.flags = XVRAM_ALLOCATION_FLAG_ZERO_INITIALIZE;
  allocation_desc.size_bytes = allocation_bytes;
  allocation_desc.priority = XVRAM_ALLOCATION_STREAMING;
  status = api.v1.allocation_create(handles.session, &allocation_desc, &handles.allocation);
  if (status != XVRAM_STATUS_SUCCESS) {
    print_session_error(api, handles.session, "allocation_create", status);
    return 1;
  }

  xvram_allocation_info_v1 allocation_info{};
  allocation_info.struct_size = sizeof(allocation_info);
  status = api.v1.allocation_get_info(handles.allocation, &allocation_info);
  if (status != XVRAM_STATUS_SUCCESS || allocation_info.size_bytes != allocation_bytes ||
      allocation_info.stable_virtual_address == 0U) {
    print_session_error(api, handles.session, "allocation_get_info", status);
    return 1;
  }

  std::vector<std::byte> host_chunk(static_cast<std::size_t>(chunk_bytes));
  TelemetrySamples telemetry_samples;
  for (std::uint64_t chunk = 0; chunk < chunk_count; ++chunk) {
    const unsigned char initial = static_cast<unsigned char>(0x10U + chunk);
    std::fill(host_chunk.begin(), host_chunk.end(), static_cast<std::byte>(initial));
    status = api.v1.allocation_write(handles.allocation, chunk * chunk_bytes, host_chunk.data(),
                                     chunk_bytes);
    if (status != XVRAM_STATUS_SUCCESS) {
      print_session_error(api, handles.session, "allocation_write", status);
      return 1;
    }
  }

  for (std::uint64_t chunk = 0; chunk < chunk_count; ++chunk) {
    xvram_access_range_v1 range{handles.allocation, chunk * chunk_bytes, chunk_bytes,
                                XVRAM_ACCESS_READ_WRITE, 0U};
    Mutation mutation{&cuda, allocation_info.allocation_id,
                      static_cast<unsigned char>(0x80U + chunk)};
    xvram_transaction_desc_v1 transaction{};
    transaction.struct_size = sizeof(transaction);
    transaction.ranges = &range;
    transaction.range_count = 1U;
    transaction.callback = &mutate_callback;
    transaction.user_data = &mutation;
    status = execute_with_live_telemetry(api, handles.session, transaction, telemetry_samples);
    if (status != XVRAM_STATUS_SUCCESS) {
      print_session_error(api, handles.session, "transaction_execute(write)", status);
      return 1;
    }
  }

  // A reverse read pass forces the cache to revisit compressed dirty evictions while preserving
  // the GPU-produced bytes. The callback intentionally submits no work; the runtime still owns and
  // records the event boundary that guards lease retirement.
  for (std::uint64_t ordinal = 0; ordinal < chunk_count; ++ordinal) {
    const std::uint64_t chunk = chunk_count - ordinal - 1U;
    xvram_access_range_v1 range{handles.allocation, chunk * chunk_bytes, chunk_bytes,
                                XVRAM_ACCESS_READ, 0U};
    xvram_transaction_desc_v1 transaction{};
    transaction.struct_size = sizeof(transaction);
    transaction.ranges = &range;
    transaction.range_count = 1U;
    transaction.callback = &observe_callback;
    status = execute_with_live_telemetry(api, handles.session, transaction, telemetry_samples);
    if (status != XVRAM_STATUS_SUCCESS) {
      print_session_error(api, handles.session, "transaction_execute(read)", status);
      return 1;
    }
  }

  status = api.v1.session_drain(handles.session, operation_timeout_ms);
  if (status != XVRAM_STATUS_SUCCESS) {
    print_session_error(api, handles.session, "session_drain", status);
    return 1;
  }

  for (std::uint64_t chunk = 0; chunk < chunk_count; ++chunk) {
    status = api.v1.allocation_read(handles.allocation, chunk * chunk_bytes, host_chunk.data(),
                                    chunk_bytes);
    if (status != XVRAM_STATUS_SUCCESS) {
      print_session_error(api, handles.session, "allocation_read", status);
      return 1;
    }
    const unsigned char expected = static_cast<unsigned char>(0x80U + chunk);
    if (!verify_fill(host_chunk, expected)) {
      std::cerr << "allocation verification failed for chunk " << chunk << '\n';
      return 1;
    }
  }

  xvram_session_telemetry_v2 active{};
  active.struct_size = sizeof(active);
  status = api.session_get_telemetry_v2(handles.session, &active);
  if (status != XVRAM_STATUS_SUCCESS) {
    print_session_error(api, handles.session, "session_get_telemetry_v2(active)", status);
    return 1;
  }
  const std::uint64_t compressed_decisions =
      active.cpu_lz4_gpu_decode_decisions + active.gpu_lz4_decisions;
  const bool active_proof =
      active.logical_bytes == allocation_bytes && active.host_compressed_bytes > 0U &&
      active.compression_attempts > 0U && active.compression_commits > 0U &&
      compressed_decisions > 0U && active.v1.cache_misses >= chunk_count &&
      active.v1.dirty_evictions > 0U && active.v1.writebacks_completed > 0U &&
      active.v1.handle_reuses > 0U && active.logical_h2d_bytes > active.pcie_h2d_bytes &&
      active.logical_d2h_bytes > 0U && active.v1.unsafe_remaps == 0U &&
      active.v1.unsafe_transitions == 0U && active.v1.poisoned == 0U &&
      telemetry_samples.while_pending > 0U && consistent_h2d_snapshot(active) &&
      (active.v1.flags & XVRAM_TELEMETRY_STABLE_VIRTUAL_ADDRESSES) != 0U &&
      (active.v1.flags & XVRAM_TELEMETRY_NO_PHYSICAL_ALIASES) != 0U;
  if (!active_proof) {
    std::cerr << "active SDK v2 telemetry did not prove compression and cache reuse: "
              << "logical=" << active.logical_bytes
              << ", host_compressed=" << active.host_compressed_bytes
              << ", attempts=" << active.compression_attempts
              << ", commits=" << active.compression_commits
              << ", compressed_decisions=" << compressed_decisions
              << ", misses=" << active.v1.cache_misses
              << ", dirty_evictions=" << active.v1.dirty_evictions
              << ", writebacks=" << active.v1.writebacks_completed
              << ", reuses=" << active.v1.handle_reuses
              << ", logical_h2d=" << active.logical_h2d_bytes
              << ", pcie_h2d=" << active.pcie_h2d_bytes
              << ", logical_d2h=" << active.logical_d2h_bytes
              << ", unsafe_remaps=" << active.v1.unsafe_remaps
              << ", unsafe_transitions=" << active.v1.unsafe_transitions
              << ", poisoned=" << active.v1.poisoned << '\n';
    return 1;
  }

  status = api.v1.allocation_release(handles.allocation);
  if (status != XVRAM_STATUS_SUCCESS) {
    print_session_error(api, handles.session, "allocation_release", status);
    return 1;
  }
  handles.allocation = nullptr;
  status = api.v1.session_close(handles.session, operation_timeout_ms);
  if (status != XVRAM_STATUS_SUCCESS) {
    print_session_error(api, handles.session, "session_close", status);
    return 1;
  }

  xvram_session_telemetry_v2 final{};
  final.struct_size = sizeof(final);
  status = api.session_get_telemetry_v2(handles.session, &final);
  if (status != XVRAM_STATUS_SUCCESS) {
    print_session_error(api, handles.session, "session_get_telemetry_v2(final)", status);
    return 1;
  }
  const bool cleanup_proof =
      final.v1.resident_bytes == 0U && final.v1.mappings > 0U &&
      final.v1.mappings == final.v1.set_access_calls && final.v1.mappings == final.v1.unmaps &&
      final.v1.physical_handles_created == final.v1.physical_handles_released &&
      final.codec_workspace_bytes == 0U && final.codec_slot_bytes == 0U &&
      final.spill_reserved_bytes == 0U && final.v1.unsafe_remaps == 0U &&
      final.v1.unsafe_transitions == 0U && final.v1.poisoned == 0U &&
      consistent_h2d_snapshot(final);
  if (!cleanup_proof) {
    std::cerr << "final SDK v2 telemetry did not prove complete cleanup\n";
    return 1;
  }

  std::cout << "{\"report_type\":\"xvram.sdk_v2_hardware_smoke\",\"status\":\"completed\""
            << ",\"device\":" << device_ordinal << ",\"logical_bytes\":" << allocation_bytes
            << ",\"mappings\":" << final.v1.mappings << ",\"unmaps\":" << final.v1.unmaps
            << ",\"handle_reuses\":" << final.v1.handle_reuses
            << ",\"dirty_evictions\":" << final.v1.dirty_evictions
            << ",\"compression_commits\":" << final.compression_commits
            << ",\"logical_h2d_bytes\":" << final.logical_h2d_bytes
            << ",\"pcie_h2d_bytes\":" << final.pcie_h2d_bytes
            << ",\"telemetry_samples\":" << telemetry_samples.total
            << ",\"pending_telemetry_samples\":" << telemetry_samples.while_pending << "}\n";

  api.v1.session_release(handles.session);
  handles.session = nullptr;
  return 0;
}

} // namespace

int main(const int argc, char** argv) {
  if (argc == 2 && std::string_view{argv[1]} == "--self-test") {
    return self_test();
  }
  std::int32_t device_ordinal = 0;
  if (argc == 3 && std::string_view{argv[1]} == "--device") {
    if (!parse_device(argv[2], device_ordinal)) {
      std::cerr << "invalid --device ordinal\n";
      return 64;
    }
  } else if (argc != 1) {
    std::cerr << "usage: xvram-sdk-v2-hardware-smoke [--device <ordinal>|--self-test]\n";
    return 64;
  }
  return hardware_test(device_ordinal);
}
