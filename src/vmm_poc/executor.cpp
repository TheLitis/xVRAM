#include "vmm_poc/executor.hpp"

#include "platform/dxgi_memory.hpp"
#include "platform/system_info.hpp"
#include "xvram/vmm_transform_ptx.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <deque>
#include <exception>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#else
#include <sys/mman.h>
#endif

namespace xvram::vmm_poc {
namespace {

constexpr int exit_completed = 0;
constexpr int exit_prerequisite = 23;
constexpr int exit_corruption = 24;
constexpr int exit_oom = 25;
constexpr int exit_timeout = 26;
constexpr int exit_failure = 27;
constexpr auto maximum_kernel_duration = std::chrono::milliseconds(250);

using Clock = std::chrono::steady_clock;

constexpr std::uint64_t parallel_partition_words = 64ULL * 1024ULL * 1024ULL;
constexpr std::uint64_t parallel_chunk_words = 4ULL * 1024ULL * 1024ULL;
constexpr unsigned maximum_cpu_workers = 8U;

[[nodiscard]] unsigned cpu_worker_count(const std::uint64_t word_count) noexcept {
  if (word_count <= parallel_partition_words) {
    return 1U;
  }
  const std::uint64_t useful_workers = ((word_count - 1ULL) / parallel_partition_words) + 1ULL;
  const unsigned hardware_workers = std::max(2U, std::thread::hardware_concurrency());
  return static_cast<unsigned>(
      std::min<std::uint64_t>({useful_workers, maximum_cpu_workers, hardware_workers}));
}

template <typename Work, typename Heartbeat>
void parallel_word_chunks(const std::uint64_t word_count, Work&& work, Heartbeat&& heartbeat) {
  const unsigned worker_count = cpu_worker_count(word_count);
  if (worker_count == 1U) {
    for (std::uint64_t begin = 0; begin < word_count; begin += parallel_chunk_words) {
      work(begin, std::min(word_count, begin + parallel_chunk_words));
      heartbeat();
    }
    return;
  }

  std::atomic<std::uint64_t> next_word{0};
  std::atomic<unsigned> active_workers{0U};
  std::mutex worker_exception_mutex;
  std::exception_ptr worker_exception;
  std::vector<std::jthread> workers;
  try {
    workers.reserve(worker_count);
  } catch (const std::bad_alloc&) {
    for (std::uint64_t begin = 0; begin < word_count; begin += parallel_chunk_words) {
      work(begin, std::min(word_count, begin + parallel_chunk_words));
      heartbeat();
    }
    return;
  }
  for (unsigned worker = 0; worker < worker_count; ++worker) {
    active_workers.fetch_add(1U, std::memory_order_relaxed);
    try {
      workers.emplace_back([&]() noexcept {
        try {
          for (;;) {
            const std::uint64_t begin =
                next_word.fetch_add(parallel_chunk_words, std::memory_order_relaxed);
            if (begin >= word_count) {
              break;
            }
            work(begin, std::min(word_count, begin + parallel_chunk_words));
          }
        } catch (...) {
          {
            const std::lock_guard lock(worker_exception_mutex);
            if (worker_exception == nullptr) {
              worker_exception = std::current_exception();
            }
          }
          next_word.store(word_count, std::memory_order_relaxed);
        }
        active_workers.fetch_sub(1U, std::memory_order_release);
      });
    } catch (...) {
      active_workers.fetch_sub(1U, std::memory_order_relaxed);
      break;
    }
  }

  if (workers.empty()) {
    for (std::uint64_t begin = 0; begin < word_count; begin += parallel_chunk_words) {
      work(begin, std::min(word_count, begin + parallel_chunk_words));
      heartbeat();
    }
    return;
  }

  while (active_workers.load(std::memory_order_acquire) != 0U) {
    heartbeat();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  workers.clear();
  if (worker_exception != nullptr) {
    std::rethrow_exception(worker_exception);
  }
}

class PageableBacking {
public:
  PageableBacking() = default;
  PageableBacking(const PageableBacking&) = delete;
  PageableBacking& operator=(const PageableBacking&) = delete;

  ~PageableBacking() {
    (void)release();
  }

  [[nodiscard]] bool allocate(const std::uint64_t bytes) {
    if (data_ != nullptr || bytes == 0 ||
        bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
      return false;
    }
#ifdef _WIN32
    data_ = static_cast<std::uint32_t*>(VirtualAlloc(nullptr, static_cast<std::size_t>(bytes),
                                                     MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (data_ == nullptr) {
      return false;
    }
#else
    void* mapping = mmap(nullptr, static_cast<std::size_t>(bytes), PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapping == MAP_FAILED) {
      return false;
    }
    data_ = static_cast<std::uint32_t*>(mapping);
#endif
    bytes_ = static_cast<std::size_t>(bytes);
    return true;
  }

  [[nodiscard]] bool release() noexcept {
    if (data_ == nullptr) {
      return true;
    }
#ifdef _WIN32
    const bool released = VirtualFree(data_, 0, MEM_RELEASE) != FALSE;
#else
    const bool released = munmap(data_, bytes_) == 0;
#endif
    if (released) {
      data_ = nullptr;
      bytes_ = 0;
    }
    return released;
  }

  [[nodiscard]] std::uint32_t* data() noexcept {
    return data_;
  }

  [[nodiscard]] const std::uint32_t* data() const noexcept {
    return data_;
  }

private:
  std::uint32_t* data_ = nullptr;
  std::size_t bytes_ = 0;
};

[[nodiscard]] double milliseconds_between(const Clock::time_point begin,
                                          const Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

[[nodiscard]] std::string cuda_result_name(cuda::CudaApi& api, const cuda::abi::Result result) {
  const char* name = nullptr;
  if (api.get_error_name_ != nullptr && api.get_error_name_(result, &name) == cuda::abi::success &&
      name != nullptr) {
    return name;
  }
  return "CUDA_ERROR_" + std::to_string(static_cast<std::int64_t>(result));
}

void append_diagnostic_noexcept(ExecutorResult& result, const std::string_view message) noexcept {
  try {
    result.diagnostics.emplace_back(message);
  } catch (...) {
  }
}

[[nodiscard]] std::string cuda_result_message(cuda::CudaApi& api, const cuda::abi::Result result) {
  const char* message = nullptr;
  if (api.get_error_string_ != nullptr &&
      api.get_error_string_(result, &message) == cuda::abi::success && message != nullptr) {
    return message;
  }
  return cuda_result_name(api, result);
}

[[nodiscard]] int classify_cuda_failure(const cuda::abi::Result result) {
  return result == CUDA_ERROR_OUT_OF_MEMORY ? exit_oom : exit_failure;
}

void record_failure(ExecutorResult& result, const int exit_code, std::string stage,
                    std::string operation, std::string message,
                    const std::optional<cuda::abi::Result> native = std::nullopt,
                    const std::optional<std::uint32_t> pass = std::nullopt,
                    const std::optional<std::uint64_t> tile = std::nullopt,
                    const std::optional<std::uint64_t> byte_offset = std::nullopt,
                    cuda::CudaApi* api = nullptr) {
  if (result.failure.has_value()) {
    return;
  }
  result.exit_code = exit_code;
  result.status = exit_code == exit_prerequisite ? "skipped" : "failed";
  ExecutionFailure failure;
  failure.stage = std::move(stage);
  failure.operation = std::move(operation);
  failure.message = std::move(message);
  failure.pass_index = pass;
  failure.tile_index = tile;
  failure.logical_byte_offset = byte_offset;
  if (native.has_value()) {
    failure.native_code = static_cast<std::int64_t>(*native);
    if (api != nullptr) {
      failure.native_name = cuda_result_name(*api, *native);
    }
  }
  result.failure = std::move(failure);
}

[[nodiscard]] bool require_cuda(ExecutorResult& result, cuda::CudaApi& api,
                                const cuda::abi::Result cuda_result, const char* stage,
                                const char* operation,
                                const std::optional<std::uint32_t> pass = std::nullopt,
                                const std::optional<std::uint64_t> tile = std::nullopt,
                                const std::optional<std::uint64_t> byte_offset = std::nullopt) {
  if (cuda_result == cuda::abi::success) {
    return true;
  }
  record_failure(result, classify_cuda_failure(cuda_result), stage, operation,
                 cuda_result_message(api, cuda_result), cuda_result, pass, tile, byte_offset, &api);
  return false;
}

[[nodiscard]] std::string digest_string(const Digest128Value digest) {
  std::ostringstream output;
  output << std::hex << std::setfill('0') << std::setw(16) << digest.high << std::setw(16)
         << digest.low;
  return output.str();
}

[[nodiscard]] std::string bytes_as_uuid(const unsigned char* bytes) {
  static constexpr std::array<std::size_t, 4> hyphens{4, 6, 8, 10};
  static constexpr char hex[] = "0123456789abcdef";
  std::string output;
  output.reserve(36);
  for (std::size_t index = 0; index < 16U; ++index) {
    if (std::find(hyphens.begin(), hyphens.end(), index) != hyphens.end()) {
      output.push_back('-');
    }
    output.push_back(hex[(bytes[index] >> 4U) & 0x0FU]);
    output.push_back(hex[bytes[index] & 0x0FU]);
  }
  return output;
}

#ifdef _WIN32
[[nodiscard]] std::string bytes_as_hex(const unsigned char* bytes, const std::size_t size) {
  static constexpr char hex[] = "0123456789abcdef";
  std::string output;
  output.reserve(size * 2U);
  for (std::size_t index = 0; index < size; ++index) {
    output.push_back(hex[(bytes[index] >> 4U) & 0x0FU]);
    output.push_back(hex[bytes[index] & 0x0FU]);
  }
  return output;
}
#endif

struct WddmIdentity {
  platform::AdapterLuid luid{};
  std::uint32_t node_mask = 0;
};

#ifdef _WIN32
struct WddmObservation {
  std::uint64_t budget = 0;
  std::uint64_t usage = 0;
  std::uint64_t available = 0;
};

[[nodiscard]] std::optional<WddmObservation> query_wddm(const WddmIdentity& identity,
                                                        std::vector<std::string>& diagnostics) {
  std::vector<probe::Diagnostic> dxgi_diagnostics;
  const auto adapter =
      platform::query_dxgi_memory(identity.luid, identity.node_mask, dxgi_diagnostics);
  for (const auto& diagnostic : dxgi_diagnostics) {
    diagnostics.push_back(diagnostic.operation + ": " + diagnostic.message);
  }
  if (!adapter.has_value() || !adapter->local.has_value()) {
    return std::nullopt;
  }
  const auto& local = *adapter->local;
  return WddmObservation{local.budget_bytes, local.current_usage_bytes,
                         local.budget_bytes > local.current_usage_bytes
                             ? local.budget_bytes - local.current_usage_bytes
                             : 0};
}
#endif

struct SlotResources {
  cuda::abi::GenericAllocationHandle handle{};
  void* pinned = nullptr;
  cuda::abi::Event h2d_start = nullptr;
  cuda::abi::Event h2d_done = nullptr;
  cuda::abi::Event kernel_start = nullptr;
  cuda::abi::Event kernel_done = nullptr;
  cuda::abi::Event d2h_start = nullptr;
  cuda::abi::Event slot_done = nullptr;
  cuda::abi::DevicePointer mapped_address = 0;
  std::uint64_t mapped_bytes = 0;
  std::uint64_t tile_index = 0;
  std::uint64_t valid_bytes = 0;
  std::uint64_t global_word_start = 0;
  std::uint32_t pass_index = 0;
  std::uint64_t mapping_generation = 0;
  std::uint64_t mode_mapping_generation = 0;
  Clock::time_point launched_at{};
  bool mapped = false;
  bool busy = false;
  bool work_submitted = false;
  bool slot_done_recorded = false;
  bool completion_boundary = false;
};

struct Resources {
  cuda::CudaApi* api = nullptr;
  cuda::abi::Device device = 0;
  cuda::abi::Context previous_context = nullptr;
  cuda::abi::Context context = nullptr;
  cuda::abi::DevicePointer reservation = 0;
  std::uint64_t reservation_bytes = 0;
  cuda::abi::DevicePointer logical_base = 0;
  std::uint64_t mapped_logical_bytes = 0;
  cuda::abi::Module module = nullptr;
  cuda::abi::Function kernel = nullptr;
  cuda::abi::Stream reference_stream = nullptr;
  cuda::abi::Stream h2d_stream = nullptr;
  cuda::abi::Stream compute_stream = nullptr;
  cuda::abi::Stream d2h_stream = nullptr;
  std::vector<SlotResources> slots;
  std::vector<cuda::abi::DevicePointer> stable_addresses;
  bool reservation_quarantined = false;
};

[[nodiscard]] bool have_executor_symbols(const cuda::CudaApi& api) {
  return api.init_ != nullptr && api.device_get_count_ != nullptr && api.device_get_ != nullptr &&
         api.device_get_name_ != nullptr && api.device_total_memory_ != nullptr &&
         api.device_get_attribute_ != nullptr && api.context_get_current_ != nullptr &&
         api.context_set_current_ != nullptr && api.context_create_ != nullptr &&
         api.context_destroy_ != nullptr && api.mem_get_info_ != nullptr &&
         api.mem_get_allocation_granularity_ != nullptr && api.mem_address_reserve_ != nullptr &&
         api.mem_address_free_ != nullptr && api.mem_create_ != nullptr &&
         api.mem_release_ != nullptr && api.mem_map_ != nullptr && api.mem_unmap_ != nullptr &&
         api.mem_set_access_ != nullptr && api.mem_host_alloc_ != nullptr &&
         api.mem_free_host_ != nullptr && api.memcpy_h2d_async_ != nullptr &&
         api.memcpy_d2h_async_ != nullptr && api.stream_create_ != nullptr &&
         api.stream_destroy_ != nullptr && api.stream_synchronize_ != nullptr &&
         api.stream_wait_event_ != nullptr && api.event_create_ != nullptr &&
         api.event_destroy_ != nullptr && api.event_record_ != nullptr &&
         api.event_query_ != nullptr && api.event_synchronize_ != nullptr &&
         api.event_elapsed_time_ != nullptr && api.module_load_data_ != nullptr &&
         api.module_get_function_ != nullptr && api.module_unload_ != nullptr &&
         api.launch_kernel_ != nullptr;
}

[[nodiscard]] std::uint64_t valid_bytes_for_tile(const WorkloadPlan& plan,
                                                 const std::uint64_t tile_index) {
  return tile_index + 1ULL == plan.logical_chunk_count ? plan.tail_chunk_bytes
                                                       : plan.effective_chunk_bytes;
}

[[nodiscard]] bool create_event(cuda::CudaApi& api, ExecutorResult& result,
                                cuda::abi::Event& event) {
  return require_cuda(result, api, api.event_create_(&event, CU_EVENT_DEFAULT), "setup",
                      "cuEventCreate");
}

[[nodiscard]] bool create_resources(Resources& resources, ExecutorResult& result,
                                    const WorkloadPlan& plan, const std::uint32_t slot_count) {
  cuda::CudaApi& api = *resources.api;
  const auto setup_started = Clock::now();

  if (!require_cuda(result, api, api.module_load_data_(&resources.module, vmm_transform_ptx.data()),
                    "setup", "cuModuleLoadData") ||
      !require_cuda(
          result, api,
          api.module_get_function_(&resources.kernel, resources.module, "xvram_vmm_transform_v1"),
          "setup", "cuModuleGetFunction")) {
    return false;
  }

  const auto mapped_bytes = checked_multiply(plan.logical_chunk_count, plan.effective_chunk_bytes);
  const auto reservation_bytes = mapped_bytes.has_value()
                                     ? checked_add(*mapped_bytes, plan.effective_alignment_bytes)
                                     : std::nullopt;
  if (!mapped_bytes.has_value() || !reservation_bytes.has_value()) {
    record_failure(result, exit_prerequisite, "planning", "reserve_va",
                   "virtual-address reservation size overflowed");
    return false;
  }
  resources.mapped_logical_bytes = *mapped_bytes;
  resources.reservation_bytes = *reservation_bytes;
  if (!require_cuda(result, api,
                    api.mem_address_reserve_(&resources.reservation,
                                             static_cast<std::size_t>(resources.reservation_bytes),
                                             0, 0, 0),
                    "setup", "cuMemAddressReserve")) {
    return false;
  }
  const std::uint64_t raw = static_cast<std::uint64_t>(resources.reservation);
  const auto logical_base = checked_align_up(raw, plan.effective_alignment_bytes);
  const auto logical_end = logical_base.has_value()
                               ? checked_add(*logical_base, resources.mapped_logical_bytes)
                               : std::nullopt;
  const auto reservation_end = checked_add(raw, resources.reservation_bytes);
  if (!logical_base.has_value() || !logical_end.has_value() || !reservation_end.has_value() ||
      *logical_end > *reservation_end) {
    record_failure(result, exit_failure, "setup", "align_logical_base",
                   "reserved virtual-address range cannot contain the aligned logical range");
    return false;
  }
  resources.logical_base = static_cast<cuda::abi::DevicePointer>(*logical_base);
  resources.stable_addresses.assign(static_cast<std::size_t>(plan.logical_chunk_count), 0);

  CUmemAllocationProp allocation{};
  allocation.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  allocation.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  allocation.location.id = resources.device;
  allocation.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;

  resources.slots.resize(slot_count);
  for (auto& slot : resources.slots) {
    if (!require_cuda(result, api,
                      api.mem_create_(&slot.handle,
                                      static_cast<std::size_t>(plan.effective_chunk_bytes),
                                      &allocation, 0),
                      "setup", "cuMemCreate") ||
        !require_cuda(result, api,
                      api.mem_host_alloc_(&slot.pinned,
                                          static_cast<std::size_t>(plan.effective_chunk_bytes), 0),
                      "setup", "cuMemHostAlloc") ||
        !create_event(api, result, slot.h2d_start) || !create_event(api, result, slot.h2d_done) ||
        !create_event(api, result, slot.kernel_start) ||
        !create_event(api, result, slot.kernel_done) ||
        !create_event(api, result, slot.d2h_start) || !create_event(api, result, slot.slot_done)) {
      return false;
    }
  }

  const auto create_stream = [&](cuda::abi::Stream& stream) {
    return require_cuda(result, api, api.stream_create_(&stream, CU_STREAM_NON_BLOCKING), "setup",
                        "cuStreamCreate");
  };
  if (!create_stream(resources.reference_stream) || !create_stream(resources.h2d_stream) ||
      !create_stream(resources.compute_stream) || !create_stream(resources.d2h_stream)) {
    return false;
  }
  const double setup_ms = milliseconds_between(setup_started, Clock::now());
  result.reference.timings.setup_ms = setup_ms;
  result.pipeline.timings.setup_ms = setup_ms;
  return true;
}

[[nodiscard]] bool map_slot(Resources& resources, ExecutorResult& result, const WorkloadPlan& plan,
                            SlotResources& slot, const std::uint64_t tile_index,
                            ModeStatistics& statistics) {
  if (slot.mapped || slot.busy) {
    ++statistics.unsafe_remaps;
    record_failure(result, exit_failure, "mapping", "slot_state",
                   "attempted to map a slot that was still mapped or busy", std::nullopt,
                   std::nullopt, tile_index, tile_index * plan.effective_chunk_bytes);
    return false;
  }
  if (slot.mapping_generation > 0 && !slot.completion_boundary) {
    ++statistics.unsafe_remaps;
    record_failure(result, exit_failure, "mapping", "event_boundary",
                   "attempted to reuse a handle without a completed event boundary", std::nullopt,
                   std::nullopt, tile_index, tile_index * plan.effective_chunk_bytes);
    return false;
  }

  const std::uint64_t byte_offset = tile_index * plan.effective_chunk_bytes;
  const auto address = static_cast<cuda::abi::DevicePointer>(
      static_cast<std::uint64_t>(resources.logical_base) + byte_offset);
  auto& stable = resources.stable_addresses[static_cast<std::size_t>(tile_index)];
  if (stable == 0) {
    stable = address;
  } else if (stable != address) {
    statistics.stable_addresses = false;
    record_failure(result, exit_failure, "mapping", "stable_address",
                   "a logical tile did not retain its virtual address", std::nullopt, std::nullopt,
                   tile_index, byte_offset);
    return false;
  }

  cuda::CudaApi& api = *resources.api;
  const auto remap_started = Clock::now();
  const auto map_result = api.mem_map_(
      address, static_cast<std::size_t>(plan.effective_chunk_bytes), 0, slot.handle, 0);
  if (map_result != cuda::abi::success) {
    return require_cuda(result, api, map_result, "mapping", "cuMemMap", std::nullopt, tile_index,
                        byte_offset);
  }
  slot.mapped = true;
  slot.mapped_address = address;
  slot.mapped_bytes = plan.effective_chunk_bytes;
  slot.work_submitted = false;
  slot.slot_done_recorded = false;
  slot.completion_boundary = false;
  ++statistics.mappings;

  CUmemAccessDesc access{};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = resources.device;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  if (!require_cuda(result, api,
                    api.mem_set_access_(
                        address, static_cast<std::size_t>(plan.effective_chunk_bytes), &access, 1),
                    "mapping", "cuMemSetAccess", std::nullopt, tile_index, byte_offset)) {
    // No device work was submitted, so setup-failure cleanup may safely remove this mapping.
    slot.completion_boundary = true;
    return false;
  }
  ++statistics.set_access_calls;
  if (slot.mode_mapping_generation > 0) {
    ++statistics.remappings;
    ++statistics.handle_reuses;
    statistics.remap_samples_ms.push_back(milliseconds_between(remap_started, Clock::now()));
  }
  ++slot.mapping_generation;
  ++slot.mode_mapping_generation;
  return true;
}

[[nodiscard]] bool launch_tile(Resources& resources, ExecutorResult& result,
                               const WorkloadPlan& plan, SlotResources& slot,
                               const std::uint64_t tile_index, const std::uint32_t pass_index,
                               const std::uint64_t seed, const bool pipeline,
                               const std::uint32_t blocks, const std::uint32_t threads,
                               ModeStatistics& statistics, const std::uint32_t* backing) {
  if (!map_slot(resources, result, plan, slot, tile_index, statistics)) {
    return false;
  }
  const std::uint64_t valid_bytes = valid_bytes_for_tile(plan, tile_index);
  const std::uint64_t byte_offset = tile_index * plan.effective_chunk_bytes;
  const std::uint64_t global_word_start = byte_offset / word_bytes;
  std::memcpy(slot.pinned, backing + static_cast<std::size_t>(global_word_start),
              static_cast<std::size_t>(valid_bytes));

  slot.tile_index = tile_index;
  slot.valid_bytes = valid_bytes;
  slot.global_word_start = global_word_start;
  slot.pass_index = pass_index;
  slot.busy = true;
  slot.launched_at = Clock::now();

  cuda::CudaApi& api = *resources.api;
  const auto stream = pipeline ? resources.h2d_stream : resources.reference_stream;
  const auto compute_stream = pipeline ? resources.compute_stream : resources.reference_stream;
  const auto d2h_stream = pipeline ? resources.d2h_stream : resources.reference_stream;
  const auto fail = [&](const cuda::abi::Result cuda_result, const char* operation) {
    return require_cuda(result, api, cuda_result, "execution", operation, pass_index, tile_index,
                        byte_offset);
  };

  if (!fail(api.event_record_(slot.h2d_start, stream), "cuEventRecord(h2d_start)")) {
    return false;
  }
  const auto h2d_result = api.memcpy_h2d_async_(slot.mapped_address, slot.pinned,
                                                static_cast<std::size_t>(valid_bytes), stream);
  if (h2d_result == cuda::abi::success) {
    slot.work_submitted = true;
    statistics.h2d_bytes += valid_bytes;
  }
  if (!fail(h2d_result, "cuMemcpyHtoDAsync") ||
      !fail(api.event_record_(slot.h2d_done, stream), "cuEventRecord(h2d_done)")) {
    return false;
  }
  if (pipeline && !fail(api.stream_wait_event_(compute_stream, slot.h2d_done, 0),
                        "cuStreamWaitEvent(h2d_done)")) {
    return false;
  }
  if (!fail(api.event_record_(slot.kernel_start, compute_stream), "cuEventRecord(kernel_start)")) {
    return false;
  }

  std::uint64_t word_count = valid_bytes / word_bytes;
  cuda::abi::DevicePointer kernel_address = slot.mapped_address;
  std::uint64_t kernel_global_start = global_word_start;
  std::uint32_t kernel_pass = pass_index;
  std::uint64_t kernel_seed = seed;
  void* kernel_parameters[] = {&kernel_address, &word_count, &kernel_global_start, &kernel_pass,
                               &kernel_seed};
  if (!fail(api.launch_kernel_(resources.kernel, blocks, 1, 1, threads, 1, 1, 0, compute_stream,
                               kernel_parameters, nullptr),
            "cuLaunchKernel") ||
      !fail(api.event_record_(slot.kernel_done, compute_stream), "cuEventRecord(kernel_done)")) {
    return false;
  }
  if (pipeline && !fail(api.stream_wait_event_(d2h_stream, slot.kernel_done, 0),
                        "cuStreamWaitEvent(kernel_done)")) {
    return false;
  }
  if (!fail(api.event_record_(slot.d2h_start, d2h_stream), "cuEventRecord(d2h_start)") ||
      !fail(api.memcpy_d2h_async_(slot.pinned, slot.mapped_address,
                                  static_cast<std::size_t>(valid_bytes), d2h_stream),
            "cuMemcpyDtoHAsync")) {
    return false;
  }
  statistics.d2h_bytes += valid_bytes;
  const auto slot_done_result = api.event_record_(slot.slot_done, d2h_stream);
  if (slot_done_result == cuda::abi::success) {
    slot.slot_done_recorded = true;
  }
  if (!fail(slot_done_result, "cuEventRecord(slot_done)")) {
    return false;
  }
  return true;
}

[[nodiscard]] bool wait_for_kernel_launch_safety(Resources& resources, ExecutorResult& result,
                                                 SlotResources& slot,
                                                 const ExecutorOptions& options) {
  cuda::CudaApi& api = *resources.api;
  const auto deadline = Clock::now() + options.stall_timeout;
  for (;;) {
    const cuda::abi::Result query = api.event_query_(slot.kernel_done);
    if (query == cuda::abi::success) {
      break;
    }
    if (query != CUDA_ERROR_NOT_READY) {
      record_failure(result, exit_failure, "safety", "cuEventQuery(kernel_done)",
                     "kernel event query returned an unexpected CUDA result", query,
                     slot.pass_index, slot.tile_index, slot.tile_index * slot.mapped_bytes, &api);
      return false;
    }
    if (Clock::now() >= deadline) {
      record_failure(result, exit_timeout, "safety", "cuEventQuery(kernel_done)",
                     "kernel did not complete before the worker polling deadline", std::nullopt,
                     slot.pass_index, slot.tile_index, slot.tile_index * slot.mapped_bytes);
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  float kernel_ms = 0.0F;
  if (!require_cuda(result, api,
                    api.event_elapsed_time_(&kernel_ms, slot.kernel_start, slot.kernel_done),
                    "safety", "cuEventElapsedTime(kernel_launch_gate)", slot.pass_index,
                    slot.tile_index, slot.tile_index * slot.mapped_bytes)) {
    return false;
  }
  if (kernel_ms > static_cast<float>(maximum_kernel_duration.count())) {
    record_failure(result, exit_failure, "safety", "kernel_duration",
                   "kernel tile duration exceeded 250 ms; the next launch was suppressed",
                   std::nullopt, slot.pass_index, slot.tile_index,
                   slot.tile_index * slot.mapped_bytes);
    return false;
  }
  return true;
}

[[nodiscard]] bool wait_for_slot(Resources& resources, ExecutorResult& result, SlotResources& slot,
                                 const ExecutorOptions& options) {
  cuda::CudaApi& api = *resources.api;
  if (!slot.slot_done_recorded) {
    record_failure(result, exit_failure, "execution", "cuEventQuery(slot_done)",
                   "slot completion event was not recorded for this mapping generation",
                   std::nullopt, slot.pass_index, slot.tile_index,
                   slot.tile_index * slot.mapped_bytes);
    return false;
  }
  const auto deadline = Clock::now() + options.stall_timeout;
  for (;;) {
    const cuda::abi::Result query = api.event_query_(slot.slot_done);
    if (query == cuda::abi::success) {
      slot.completion_boundary = true;
      return true;
    }
    if (query != CUDA_ERROR_NOT_READY) {
      record_failure(result, exit_failure, "execution", "cuEventQuery(slot_done)",
                     "cuEventQuery returned neither CUDA_SUCCESS nor CUDA_ERROR_NOT_READY", query,
                     slot.pass_index, slot.tile_index, slot.tile_index * slot.mapped_bytes, &api);
      return false;
    }
    if (Clock::now() >= deadline) {
      record_failure(result, exit_timeout, "execution", "cuEventQuery(slot_done)",
                     "GPU tile did not complete before the worker polling deadline", std::nullopt,
                     slot.pass_index, slot.tile_index, slot.tile_index * slot.mapped_bytes);
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

[[nodiscard]] bool collect_event_timing(Resources& resources, ExecutorResult& result,
                                        SlotResources& slot, ModeStatistics& statistics) {
  cuda::CudaApi& api = *resources.api;
  const auto add_elapsed = [&](cuda::abi::Event start, cuda::abi::Event end, double& total,
                               const char* operation, float& latest) {
    latest = 0.0F;
    const auto cuda_result = api.event_elapsed_time_(&latest, start, end);
    if (!require_cuda(result, api, cuda_result, "measurement", operation, slot.pass_index,
                      slot.tile_index, slot.tile_index * slot.mapped_bytes)) {
      return false;
    }
    total += static_cast<double>(latest);
    return true;
  };
  float h2d = 0.0F;
  float kernel = 0.0F;
  float d2h = 0.0F;
  if (!add_elapsed(slot.h2d_start, slot.h2d_done, statistics.timings.h2d_ms,
                   "cuEventElapsedTime(h2d)", h2d) ||
      !add_elapsed(slot.kernel_start, slot.kernel_done, statistics.timings.kernel_ms,
                   "cuEventElapsedTime(kernel)", kernel) ||
      !add_elapsed(slot.d2h_start, slot.slot_done, statistics.timings.d2h_ms,
                   "cuEventElapsedTime(d2h)", d2h)) {
    return false;
  }
  if (kernel > static_cast<float>(maximum_kernel_duration.count())) {
    record_failure(result, exit_failure, "safety", "kernel_duration",
                   "kernel tile duration exceeded the 250 ms launch limit", std::nullopt,
                   slot.pass_index, slot.tile_index, slot.tile_index * slot.mapped_bytes);
    return false;
  }
  return true;
}

[[nodiscard]] bool verify_tile(ExecutorResult& result, const ExecutorOptions& options,
                               const WorkloadPlan& plan, SlotResources& slot,
                               ModeStatistics& statistics, std::uint32_t* backing) {
  const auto verification_started = Clock::now();
  const auto* words = static_cast<const std::uint32_t*>(slot.pinned);
  const std::uint64_t word_count = slot.valid_bytes / word_bytes;
  for (std::uint64_t index = 0; index < word_count; ++index) {
    const std::uint64_t global_index = slot.global_word_start + index;
    const std::uint32_t expected = transform_word(backing[static_cast<std::size_t>(global_index)],
                                                  global_index, slot.pass_index, options.seed);
    if (words[static_cast<std::size_t>(index)] != expected) {
      ++statistics.mismatch_count;
      if (!statistics.first_mismatch_byte_offset.has_value()) {
        statistics.first_mismatch_byte_offset = global_index * word_bytes;
      }
    }
  }
  statistics.words_verified += word_count;
  std::memcpy(backing + static_cast<std::size_t>(slot.global_word_start), slot.pinned,
              static_cast<std::size_t>(slot.valid_bytes));
  statistics.timings.verification_ms += milliseconds_between(verification_started, Clock::now());
  if (statistics.mismatch_count != 0) {
    record_failure(result, exit_corruption, "verification", "tile_reference",
                   "GPU tile does not match the CPU reference", std::nullopt, slot.pass_index,
                   slot.tile_index, *statistics.first_mismatch_byte_offset);
    return false;
  }
  (void)plan;
  return true;
}

[[nodiscard]] bool unmap_slot(Resources& resources, ExecutorResult& result, SlotResources& slot,
                              ModeStatistics& statistics) {
  if (!slot.mapped) {
    slot.busy = false;
    return true;
  }
  if (!slot.completion_boundary) {
    ++statistics.unsafe_remaps;
    record_failure(result, exit_failure, "mapping", "event_boundary",
                   "refusing to unmap a slot before its completion event", std::nullopt,
                   slot.pass_index, slot.tile_index, slot.tile_index * slot.mapped_bytes);
    return false;
  }
  const auto cuda_result =
      resources.api->mem_unmap_(slot.mapped_address, static_cast<std::size_t>(slot.mapped_bytes));
  if (!require_cuda(result, *resources.api, cuda_result, "mapping", "cuMemUnmap", slot.pass_index,
                    slot.tile_index, slot.tile_index * slot.mapped_bytes)) {
    resources.reservation_quarantined = true;
    return false;
  }
  slot.mapped = false;
  slot.busy = false;
  ++statistics.unmappings;
  ++statistics.event_boundaries;
  return true;
}

[[nodiscard]] bool observe_budget(ExecutorResult& result, DeviceSnapshot& device,
                                  const std::optional<WddmIdentity>& identity,
                                  const WorkloadPlan& plan, const ExecutorOptions& options,
                                  const ExecutorEnvironment* environment) {
  std::optional<BudgetSnapshot> budget;
  if (environment != nullptr) {
    budget = environment->query_device_budget ? environment->query_device_budget()
                                              : environment->initial_device_budget;
    if (!budget.has_value()) {
      record_failure(result, exit_failure, "budget", "injected_budget_query",
                     "the injected recurring device-budget query failed");
      return false;
    }
  }
#ifdef _WIN32
  if (environment == nullptr) {
    if (!identity.has_value()) {
      record_failure(result, exit_failure, "budget", "DXGI",
                     "the Windows worker has no CUDA LUID for recurring budget checks");
      return false;
    }
    const auto observation = query_wddm(*identity, result.diagnostics);
    if (!observation.has_value()) {
      record_failure(result, exit_failure, "budget", "QueryVideoMemoryInfo",
                     "the recurring WDDM budget query failed");
      return false;
    }
    budget = BudgetSnapshot{observation->budget, observation->usage};
  }
#else
  (void)identity;
#endif
  if (!budget.has_value()) {
    return true;
  }
  const std::uint64_t available = budget->available_bytes();
  device.wddm_available_bytes_end = available;
  if (!device.wddm_available_bytes_minimum.has_value() ||
      available < *device.wddm_available_bytes_minimum) {
    device.wddm_available_bytes_minimum = available;
  }
  const std::uint64_t required = plan.resident_physical_bytes + options.device_headroom_bytes;
  if (available < required) {
    record_failure(result, exit_oom, "budget", "WDDM pressure",
                   "device-memory budget fell below the physical window plus headroom");
    return false;
  }
  return true;
}

[[nodiscard]] bool final_verify(ExecutorResult& result, const ExecutorOptions& options,
                                const WorkloadPlan& plan, ModeStatistics& statistics,
                                const std::uint32_t* backing, const ProgressCallback& progress,
                                const std::string_view mode) {
  const auto started = Clock::now();
  auto next_heartbeat = Clock::now() + options.progress_heartbeat;
  const auto heartbeat = [&]() {
    if (progress && Clock::now() >= next_heartbeat) {
      progress(mode, statistics, plan.tile_visit_count);
      next_heartbeat = Clock::now() + options.progress_heartbeat;
    }
  };

  std::atomic<std::uint64_t> final_mismatch_count{0};
  std::atomic<std::uint64_t> final_first_mismatch{std::numeric_limits<std::uint64_t>::max()};
  parallel_word_chunks(
      plan.logical_element_count,
      [&](const std::uint64_t begin, const std::uint64_t end) {
        std::uint64_t local_mismatches = 0;
        std::uint64_t local_first = std::numeric_limits<std::uint64_t>::max();
        for (std::uint64_t index = begin; index < end; ++index) {
          const std::uint32_t actual = backing[static_cast<std::size_t>(index)];
          const std::uint32_t expected = expected_word(index, options.passes, options.seed);
          if (actual != expected) {
            ++local_mismatches;
            local_first = std::min(local_first, index);
          }
        }
        final_mismatch_count.fetch_add(local_mismatches, std::memory_order_relaxed);
        std::uint64_t observed = final_first_mismatch.load(std::memory_order_relaxed);
        while (local_first < observed &&
               !final_first_mismatch.compare_exchange_weak(
                   observed, local_first, std::memory_order_relaxed, std::memory_order_relaxed)) {
        }
      },
      heartbeat);

  const std::uint64_t mismatches = final_mismatch_count.load(std::memory_order_relaxed);
  statistics.mismatch_count += mismatches;
  const std::uint64_t first_mismatch = final_first_mismatch.load(std::memory_order_relaxed);
  if (first_mismatch != std::numeric_limits<std::uint64_t>::max() &&
      !statistics.first_mismatch_byte_offset.has_value()) {
    statistics.first_mismatch_byte_offset = first_mismatch * word_bytes;
  }

  Digest128Accumulator actual_digest;
  for (std::uint64_t begin = 0; begin < plan.logical_element_count; begin += parallel_chunk_words) {
    const std::uint64_t end = std::min(plan.logical_element_count, begin + parallel_chunk_words);
    for (std::uint64_t index = begin; index < end; ++index) {
      actual_digest.update(backing[static_cast<std::size_t>(index)], index);
    }
    heartbeat();
  }
  statistics.words_verified += plan.logical_element_count;
  statistics.digest128 = digest_string(actual_digest.value());
  // Every word was independently compared with the CPU reference above, so the successful
  // output digest is also the CPU-reference digest. Only corruption needs the slower second
  // digest pass in order to retain both values in the failure report.
  if (statistics.mismatch_count == 0) {
    statistics.expected_digest128 = statistics.digest128;
  } else {
    Digest128Accumulator expected_digest;
    for (std::uint64_t begin = 0; begin < plan.logical_element_count;
         begin += parallel_chunk_words) {
      const std::uint64_t end = std::min(plan.logical_element_count, begin + parallel_chunk_words);
      for (std::uint64_t index = begin; index < end; ++index) {
        expected_digest.update(expected_word(index, options.passes, options.seed), index);
      }
      heartbeat();
    }
    statistics.expected_digest128 = digest_string(expected_digest.value());
  }
  statistics.full_verification = true;
  statistics.matches_cpu =
      statistics.mismatch_count == 0 && statistics.digest128 == statistics.expected_digest128;
  statistics.timings.verification_ms += milliseconds_between(started, Clock::now());
  if (!statistics.matches_cpu) {
    record_failure(result, exit_corruption, "verification", "full_reference",
                   "final pageable backing does not match the CPU reference", std::nullopt,
                   std::nullopt, std::nullopt, statistics.first_mismatch_byte_offset);
    return false;
  }
  return true;
}

void reset_backing(const ExecutorOptions& options, const WorkloadPlan& plan, std::uint32_t* backing,
                   const ProgressCallback& progress, const std::string_view mode,
                   const ModeStatistics& statistics) {
  auto next_heartbeat = Clock::now() + options.progress_heartbeat;
  parallel_word_chunks(
      plan.logical_element_count,
      [&](const std::uint64_t begin, const std::uint64_t end) {
        for (std::uint64_t index = begin; index < end; ++index) {
          backing[static_cast<std::size_t>(index)] = initial_word(index, options.seed);
        }
      },
      [&]() {
        if (progress && Clock::now() >= next_heartbeat) {
          progress(mode, statistics, plan.tile_visit_count);
          next_heartbeat = Clock::now() + options.progress_heartbeat;
        }
      });
}

[[nodiscard]] bool retire_slot(Resources& resources, ExecutorResult& result,
                               const ExecutorOptions& options, const WorkloadPlan& plan,
                               SlotResources& slot, ModeStatistics& statistics,
                               std::uint32_t* backing, const ProgressCallback& progress,
                               const std::string_view mode) {
  if (!wait_for_slot(resources, result, slot, options)) {
    return false;
  }
  const bool timing_ok = collect_event_timing(resources, result, slot, statistics);
  const bool verification_ok = verify_tile(result, options, plan, slot, statistics, backing);
  const bool unmap_ok = unmap_slot(resources, result, slot, statistics);
  if (unmap_ok) {
    ++statistics.tiles_retired;
    if (progress) {
      progress(mode, statistics, plan.tile_visit_count);
    }
  }
  return timing_ok && verification_ok && unmap_ok;
}

[[nodiscard]] bool run_reference_mode(Resources& resources, ExecutorResult& result,
                                      DeviceSnapshot& device, const ExecutorOptions& options,
                                      const WorkloadPlan& plan,
                                      const std::optional<WddmIdentity>& wddm_identity,
                                      const ExecutorEnvironment* environment,
                                      std::uint32_t* backing, const ProgressCallback& progress,
                                      const std::uint32_t blocks, const std::uint32_t threads) {
  ModeStatistics& statistics = result.reference;
  statistics.status = "running";
  statistics.slots = 1;
  for (auto& slot : resources.slots) {
    slot.mode_mapping_generation = 0;
  }
  reset_backing(options, plan, backing, progress, "reference", statistics);
  const auto started = Clock::now();

  for (std::uint32_t pass = 0; pass < options.passes; ++pass) {
    for (std::uint64_t within_pass = 0; within_pass < plan.logical_chunk_count; ++within_pass) {
      const std::uint64_t tile =
          (pass & 1U) == 0U ? within_pass : plan.logical_chunk_count - 1ULL - within_pass;
      SlotResources& slot = resources.slots.front();
      const bool launched = launch_tile(resources, result, plan, slot, tile, pass, options.seed,
                                        false, blocks, threads, statistics, backing);
      if (!launched) {
        return false;
      }
      if (!retire_slot(resources, result, options, plan, slot, statistics, backing, progress,
                       "reference")) {
        return false;
      }
      if (statistics.tiles_retired % 32ULL == 0 &&
          !observe_budget(result, device, wddm_identity, plan, options, environment)) {
        return false;
      }
    }
    statistics.passes_completed = pass + 1U;
  }
  if (!final_verify(result, options, plan, statistics, backing, progress, "reference")) {
    return false;
  }
  statistics.timings.total_ms = milliseconds_between(started, Clock::now());
  statistics.status = "completed";
  if (progress) {
    progress("reference", statistics, plan.tile_visit_count);
  }
  return true;
}

[[nodiscard]] bool run_pipeline_mode(Resources& resources, ExecutorResult& result,
                                     DeviceSnapshot& device, const ExecutorOptions& options,
                                     const WorkloadPlan& plan,
                                     const std::optional<WddmIdentity>& wddm_identity,
                                     const ExecutorEnvironment* environment, std::uint32_t* backing,
                                     const ProgressCallback& progress, const std::uint32_t blocks,
                                     const std::uint32_t threads) {
  ModeStatistics& statistics = result.pipeline;
  statistics.status = "running";
  statistics.slots = options.window_slots;
  for (auto& slot : resources.slots) {
    slot.mode_mapping_generation = 0;
  }
  reset_backing(options, plan, backing, progress, "pipeline", statistics);
  const auto started = Clock::now();
  std::vector<std::size_t> free_slots;
  free_slots.reserve(resources.slots.size());
  for (std::size_t index = resources.slots.size(); index > 0; --index) {
    free_slots.push_back(index - 1U);
  }
  std::deque<std::size_t> fifo;
  std::optional<std::size_t> previous_kernel_slot;

  for (std::uint32_t pass = 0; pass < options.passes; ++pass) {
    std::uint64_t next = 0;
    bool stop_launching = false;
    while (next < plan.logical_chunk_count || !fifo.empty()) {
      while (!stop_launching && next < plan.logical_chunk_count && !free_slots.empty()) {
        if (previous_kernel_slot.has_value() &&
            !wait_for_kernel_launch_safety(resources, result,
                                           resources.slots[*previous_kernel_slot], options)) {
          stop_launching = true;
          break;
        }
        const std::size_t slot_index = free_slots.back();
        free_slots.pop_back();
        const std::uint64_t tile =
            (pass & 1U) == 0U ? next : plan.logical_chunk_count - 1ULL - next;
        if (!launch_tile(resources, result, plan, resources.slots[slot_index], tile, pass,
                         options.seed, true, blocks, threads, statistics, backing)) {
          stop_launching = true;
          free_slots.push_back(slot_index);
          break;
        }
        fifo.push_back(slot_index);
        previous_kernel_slot = slot_index;
        ++next;
      }

      if (fifo.empty()) {
        if (stop_launching || result.failure.has_value()) {
          return false;
        }
        continue;
      }
      const std::size_t retired_index = fifo.front();
      fifo.pop_front();
      const bool retired =
          retire_slot(resources, result, options, plan, resources.slots[retired_index], statistics,
                      backing, progress, "pipeline");
      free_slots.push_back(retired_index);
      if (!retired) {
        stop_launching = true;
      }
      if (retired && statistics.tiles_retired % 32ULL == 0 &&
          !observe_budget(result, device, wddm_identity, plan, options, environment)) {
        stop_launching = true;
      }
      if (stop_launching) {
        // Existing work is drained in FIFO order; no additional tile may be submitted.
        while (!fifo.empty()) {
          const std::size_t active_index = fifo.front();
          fifo.pop_front();
          (void)retire_slot(resources, result, options, plan, resources.slots[active_index],
                            statistics, backing, progress, "pipeline");
          free_slots.push_back(active_index);
        }
        return false;
      }
    }
    statistics.passes_completed = pass + 1U;
  }

  if (!final_verify(result, options, plan, statistics, backing, progress, "pipeline")) {
    return false;
  }
  statistics.timings.total_ms = milliseconds_between(started, Clock::now());
  statistics.status = "completed";
  if (progress) {
    progress("pipeline", statistics, plan.tile_visit_count);
  }
  return true;
}

void cleanup_resources(Resources& resources, ExecutorResult& result) noexcept {
  CleanupLedger ledger;
  ledger.events_drained = true;
  ledger.mappings_removed = true;
  ledger.handles_released = true;
  ledger.reservation_released = true;
  ledger.events_destroyed = true;
  ledger.streams_destroyed = true;
  ledger.module_unloaded = true;
  ledger.pinned_buffers_released = true;
  ledger.context_destroyed = true;
  if (resources.api == nullptr) {
    ledger.complete = true;
    result.cleanup = ledger;
    return;
  }
  cuda::CudaApi& api = *resources.api;
  const auto cleanup_error = [&](const char* operation,
                                 const cuda::abi::Result cuda_result) noexcept {
    try {
      result.diagnostics.push_back(std::string("cleanup ") + operation + ": " +
                                   cuda_result_name(api, cuda_result) + ": " +
                                   cuda_result_message(api, cuda_result));
    } catch (...) {
    }
  };

  bool partial_generation = false;
  for (auto& slot : resources.slots) {
    if (!slot.busy) {
      continue;
    }
    if (!slot.slot_done_recorded) {
      if (slot.work_submitted) {
        *ledger.events_drained = false;
        resources.reservation_quarantined = true;
        partial_generation = true;
        append_diagnostic_noexcept(
            result, "cleanup quarantined a mapping whose final completion event was not recorded");
      } else {
        slot.completion_boundary = true;
        slot.busy = false;
      }
      continue;
    }
    const auto cuda_result = api.event_synchronize_(slot.slot_done);
    if (cuda_result == cuda::abi::success) {
      slot.completion_boundary = true;
      slot.busy = false;
    } else {
      *ledger.events_drained = false;
      partial_generation = true;
      resources.reservation_quarantined = true;
      cleanup_error("cuEventSynchronize(slot_done)", cuda_result);
    }
  }
  bool dma_quarantine = false;
  if (partial_generation) {
    const std::array<cuda::abi::Stream, 4> streams{resources.reference_stream, resources.h2d_stream,
                                                   resources.compute_stream, resources.d2h_stream};
    for (const auto stream : streams) {
      if (stream == nullptr) {
        continue;
      }
      const auto cuda_result = api.stream_synchronize_(stream);
      if (cuda_result != cuda::abi::success) {
        dma_quarantine = true;
        cleanup_error("cuStreamSynchronize(partial_generation)", cuda_result);
      }
    }
    if (!dma_quarantine) {
      for (auto& slot : resources.slots) {
        if (slot.work_submitted && !slot.slot_done_recorded) {
          slot.busy = false;
        }
      }
    }
  }
  for (auto& slot : resources.slots) {
    if (!slot.mapped) {
      continue;
    }
    if (!slot.completion_boundary) {
      *ledger.mappings_removed = false;
      resources.reservation_quarantined = true;
      append_diagnostic_noexcept(
          result, "cleanup refused cuMemUnmap without a completed slot event boundary");
      continue;
    }
    const auto cuda_result =
        api.mem_unmap_(slot.mapped_address, static_cast<std::size_t>(slot.mapped_bytes));
    if (cuda_result == cuda::abi::success) {
      slot.mapped = false;
    } else {
      *ledger.mappings_removed = false;
      resources.reservation_quarantined = true;
      cleanup_error("cuMemUnmap", cuda_result);
    }
  }
  for (auto& slot : resources.slots) {
    if (slot.handle == 0) {
      continue;
    }
    const auto cuda_result = api.mem_release_(slot.handle);
    if (cuda_result != cuda::abi::success) {
      *ledger.handles_released = false;
      cleanup_error("cuMemRelease", cuda_result);
    }
    slot.handle = 0;
  }
  if (resources.reservation != 0) {
    if (resources.reservation_quarantined || !*ledger.mappings_removed) {
      *ledger.reservation_released = false;
    } else {
      const auto cuda_result = api.mem_address_free_(
          resources.reservation, static_cast<std::size_t>(resources.reservation_bytes));
      if (cuda_result != cuda::abi::success) {
        *ledger.reservation_released = false;
        cleanup_error("cuMemAddressFree", cuda_result);
      }
      resources.reservation = 0;
    }
  }

  bool context_destroyed_early = false;
  if (dma_quarantine && resources.context != nullptr) {
    const auto cuda_result = api.context_destroy_(resources.context);
    if (cuda_result == cuda::abi::success) {
      context_destroyed_early = true;
      resources.context = nullptr;
      for (auto& slot : resources.slots) {
        slot.h2d_start = nullptr;
        slot.h2d_done = nullptr;
        slot.kernel_start = nullptr;
        slot.kernel_done = nullptr;
        slot.d2h_start = nullptr;
        slot.slot_done = nullptr;
      }
      resources.reference_stream = nullptr;
      resources.h2d_stream = nullptr;
      resources.compute_stream = nullptr;
      resources.d2h_stream = nullptr;
      resources.module = nullptr;
      if (api.context_set_current_(resources.previous_context) != cuda::abi::success) {
        *ledger.context_destroyed = false;
        append_diagnostic_noexcept(result, "cleanup cuCtxSetCurrent(previous) failed");
      }
    } else {
      *ledger.context_destroyed = false;
      *ledger.events_destroyed = false;
      *ledger.streams_destroyed = false;
      *ledger.module_unloaded = false;
      cleanup_error("cuCtxDestroy(quarantine)", cuda_result);
    }
    *ledger.pinned_buffers_released = false;
    append_diagnostic_noexcept(
        result,
        "cleanup retained pinned staging until worker exit because DMA drain was not confirmed");
  }

  const auto destroy_event = [&](cuda::abi::Event& event) {
    if (event == nullptr) {
      return;
    }
    const auto cuda_result = api.event_destroy_(event);
    if (cuda_result != cuda::abi::success) {
      *ledger.events_destroyed = false;
      cleanup_error("cuEventDestroy", cuda_result);
    }
    event = nullptr;
  };
  if (!dma_quarantine) {
    for (auto& slot : resources.slots) {
      destroy_event(slot.h2d_start);
      destroy_event(slot.h2d_done);
      destroy_event(slot.kernel_start);
      destroy_event(slot.kernel_done);
      destroy_event(slot.d2h_start);
      destroy_event(slot.slot_done);
    }
  }
  const auto destroy_stream = [&](cuda::abi::Stream& stream) {
    if (stream == nullptr) {
      return;
    }
    const auto cuda_result = api.stream_destroy_(stream);
    if (cuda_result != cuda::abi::success) {
      *ledger.streams_destroyed = false;
      cleanup_error("cuStreamDestroy", cuda_result);
    }
    stream = nullptr;
  };
  if (!dma_quarantine) {
    destroy_stream(resources.reference_stream);
    destroy_stream(resources.h2d_stream);
    destroy_stream(resources.compute_stream);
    destroy_stream(resources.d2h_stream);
  }
  if (!dma_quarantine && resources.module != nullptr) {
    const auto cuda_result = api.module_unload_(resources.module);
    if (cuda_result != cuda::abi::success) {
      *ledger.module_unloaded = false;
      cleanup_error("cuModuleUnload", cuda_result);
    }
    resources.module = nullptr;
  }
  if (!dma_quarantine) {
    for (auto& slot : resources.slots) {
      if (slot.pinned == nullptr) {
        continue;
      }
      const auto cuda_result = api.mem_free_host_(slot.pinned);
      if (cuda_result != cuda::abi::success) {
        *ledger.pinned_buffers_released = false;
        cleanup_error("cuMemFreeHost", cuda_result);
      }
      slot.pinned = nullptr;
    }
  }
  if (!dma_quarantine && !context_destroyed_early && resources.context != nullptr) {
    const auto cuda_result = api.context_destroy_(resources.context);
    if (cuda_result != cuda::abi::success) {
      *ledger.context_destroyed = false;
      cleanup_error("cuCtxDestroy", cuda_result);
    }
    resources.context = nullptr;
    if (api.context_set_current_(resources.previous_context) != cuda::abi::success) {
      *ledger.context_destroyed = false;
      append_diagnostic_noexcept(result, "cleanup cuCtxSetCurrent(previous) failed");
    }
  }

  ledger.host_backing_released = result.cleanup.host_backing_released;
  ledger.complete = *ledger.events_drained && *ledger.mappings_removed &&
                    *ledger.handles_released && *ledger.reservation_released &&
                    *ledger.events_destroyed && *ledger.streams_destroyed &&
                    *ledger.module_unloaded && *ledger.pinned_buffers_released &&
                    ledger.host_backing_released.value_or(false) && *ledger.context_destroyed;
  result.cleanup = ledger;
  if (!*ledger.complete) {
    result.exit_code = exit_failure;
    try {
      result.status = "failed";
    } catch (...) {
    }
    if (!result.failure.has_value()) {
      try {
        record_failure(result, exit_failure, "cleanup", "resource_ledger",
                       "one or more cleanup stages failed");
      } catch (...) {
      }
    }
  }
}

[[nodiscard]] bool get_attribute(cuda::CudaApi& api, ExecutorResult& result,
                                 const cuda::abi::Device device,
                                 const cuda::abi::NativeDeviceAttribute attribute,
                                 const char* operation, std::uint32_t& output) {
  int value = 0;
  if (!require_cuda(result, api, api.device_get_attribute_(&value, attribute, device), "preflight",
                    operation)) {
    return false;
  }
  if (value < 0) {
    record_failure(result, exit_failure, "preflight", operation,
                   "CUDA returned a negative device attribute");
    return false;
  }
  output = static_cast<std::uint32_t>(value);
  return true;
}

} // namespace

ExecutorResult run_executor(cuda::CudaApi& api, const ExecutorOptions& options,
                            const ProgressCallback& progress,
                            const ExecutorEnvironment* environment) {
  static_assert(std::is_nothrow_move_constructible_v<ExecutorResult>);
  ExecutorResult result;
  result.exit_code = exit_failure;
  result.status = "failed";
  result.cleanup.host_backing_released = true;
  Resources resources;
  resources.api = &api;
  PageableBacking backing;

  const auto finish = [&]() noexcept -> ExecutorResult {
    try {
      if (progress) {
        if (result.pipeline.status != "not_requested") {
          progress("pipeline", result.pipeline,
                   result.plan.has_value() ? result.plan->tile_visit_count : 0);
        } else if (result.reference.status != "not_requested") {
          progress("reference", result.reference,
                   result.plan.has_value() ? result.plan->tile_visit_count : 0);
        }
      }
    } catch (...) {
      if (!result.failure.has_value()) {
        try {
          record_failure(result, exit_failure, "reporting", "progress_callback",
                         "the worker progress callback raised an exception");
        } catch (...) {
          result.exit_code = exit_failure;
          try {
            result.status = "failed";
          } catch (...) {
          }
        }
      }
    }
    result.cleanup.host_backing_released = backing.release();
    if (!result.cleanup.host_backing_released.value_or(false)) {
      append_diagnostic_noexcept(result, "cleanup failed to release pageable host backing");
    }
    cleanup_resources(resources, result);
    if (result.reference.status == "running") {
      try {
        result.reference.status = "failed";
      } catch (...) {
      }
    }
    if (result.pipeline.status == "running") {
      try {
        result.pipeline.status = "failed";
      } catch (...) {
      }
    }
    return std::move(result);
  };

  if (options.chunk_bytes == 0 || options.window_slots == 0 || options.window_slots > 8U ||
      options.passes < 2U || options.passes > 8U || options.stall_timeout.count() <= 0 ||
      options.progress_heartbeat.count() <= 0 ||
      (options.logical_bytes.has_value() &&
       (*options.logical_bytes == 0 || *options.logical_bytes % word_bytes != 0)) ||
      (options.mode != RequestedMode::reference && options.window_slots < 2U)) {
    record_failure(result, exit_prerequisite, "configuration", "validate",
                   "the requested Phase 1 configuration is unsafe or invalid");
    return finish();
  }

  const auto load = api.load();
  if (load.status != cuda::CudaApi::LoadStatus::loaded) {
    result.reason = "cuda_driver_unavailable";
    record_failure(result, exit_prerequisite, "preflight", "load_cuda_driver",
                   api.error().empty() ? "CUDA Driver API is unavailable" : api.error());
    return finish();
  }
  if (!have_executor_symbols(api)) {
    result.reason = "cuda_vmm_symbols_unavailable";
    record_failure(result, exit_prerequisite, "preflight", "resolve_cuda_symbols",
                   "the CUDA driver does not export every function required by Phase 1");
    return finish();
  }
  const auto initialization = api.init_(0);
  if (initialization != cuda::abi::success) {
    result.reason = "cuda_initialization_failed";
    record_failure(result, exit_prerequisite, "preflight", "cuInit",
                   cuda_result_message(api, initialization), initialization, std::nullopt,
                   std::nullopt, std::nullopt, &api);
    return finish();
  }
  int device_count = 0;
  if (!require_cuda(result, api, api.device_get_count_(&device_count), "preflight",
                    "cuDeviceGetCount")) {
    return finish();
  }
  if (options.device_ordinal < 0 || options.device_ordinal >= device_count) {
    result.reason = "device_ordinal_unavailable";
    record_failure(result, exit_prerequisite, "preflight", "select_device",
                   "the requested CUDA device ordinal is unavailable");
    return finish();
  }
  if (!require_cuda(result, api, api.device_get_(&resources.device, options.device_ordinal),
                    "preflight", "cuDeviceGet")) {
    return finish();
  }

  DeviceSnapshot device;
  device.ordinal = options.device_ordinal;
  std::array<char, 256> device_name{};
  if (!require_cuda(result, api,
                    api.device_get_name_(device_name.data(), static_cast<int>(device_name.size()),
                                         resources.device),
                    "preflight", "cuDeviceGetName")) {
    return finish();
  }
  device.name = device_name.data();
  if (options.include_identifiers && api.device_get_uuid_ != nullptr) {
    cuda::abi::Uuid uuid{};
    if (api.device_get_uuid_(&uuid, resources.device) == cuda::abi::success) {
      device.uuid = bytes_as_uuid(reinterpret_cast<const unsigned char*>(uuid.bytes));
    }
  }
  if (options.include_identifiers && api.device_get_pci_bus_id_ != nullptr) {
    std::array<char, 32> pci_bus_id{};
    if (api.device_get_pci_bus_id_(pci_bus_id.data(), static_cast<int>(pci_bus_id.size()),
                                   resources.device) == cuda::abi::success) {
      device.pci_bus_id = pci_bus_id.data();
    }
  }
  int tcc_driver = 0;
  if (api.device_get_attribute_(&tcc_driver, cuda::abi::attributes::tcc_driver, resources.device) ==
      cuda::abi::success) {
#ifdef _WIN32
    device.driver_model = tcc_driver != 0 ? "tcc" : "wddm";
#else
    device.driver_model = tcc_driver != 0 ? "tcc" : "unknown";
#endif
  }
  std::size_t total_memory = 0;
  if (!require_cuda(result, api, api.device_total_memory_(&total_memory, resources.device),
                    "preflight", "cuDeviceTotalMem")) {
    return finish();
  }
  device.total_memory_bytes = static_cast<std::uint64_t>(total_memory);
  std::uint32_t vmm = 0;
  std::uint32_t uva = 0;
  if (!get_attribute(api, result, resources.device,
                     cuda::abi::attributes::virtual_memory_management_supported,
                     "CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED", vmm) ||
      !get_attribute(api, result, resources.device, cuda::abi::attributes::unified_addressing,
                     "CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING", uva) ||
      !get_attribute(api, result, resources.device, cuda::abi::attributes::multiprocessor_count,
                     "CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT", device.multiprocessor_count) ||
      !get_attribute(api, result, resources.device, cuda::abi::attributes::warp_size,
                     "CU_DEVICE_ATTRIBUTE_WARP_SIZE", device.warp_size) ||
      !get_attribute(api, result, resources.device, cuda::abi::attributes::max_threads_per_block,
                     "CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK",
                     device.maximum_threads_per_block)) {
    return finish();
  }
  device.vmm_supported = vmm != 0;
  device.unified_addressing = uva != 0;
  if (!device.vmm_supported || !device.unified_addressing) {
    result.device = device;
    result.reason = "vmm_or_uva_not_supported";
    record_failure(result, exit_prerequisite, "preflight", "capabilities",
                   "the selected device must support CUDA VMM and unified addressing");
    return finish();
  }
  if (device.multiprocessor_count == 0 || device.warp_size == 0 ||
      device.maximum_threads_per_block < device.warp_size) {
    record_failure(result, exit_failure, "preflight", "kernel_geometry",
                   "the selected device reported invalid launch geometry attributes");
    return finish();
  }

  if (!require_cuda(result, api, api.context_get_current_(&resources.previous_context), "preflight",
                    "cuCtxGetCurrent") ||
      !require_cuda(
          result, api,
          api.context_create_(&resources.context, CU_CTX_SCHED_BLOCKING_SYNC, resources.device),
          "preflight", "cuCtxCreate")) {
    return finish();
  }

  std::size_t free_memory = 0;
  std::size_t context_total = 0;
  if (!require_cuda(result, api, api.mem_get_info_(&free_memory, &context_total), "preflight",
                    "cuMemGetInfo")) {
    return finish();
  }
  device.free_memory_bytes_start = static_cast<std::uint64_t>(free_memory);

  CUmemAllocationProp allocation{};
  allocation.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  allocation.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  allocation.location.id = resources.device;
  allocation.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
  std::size_t minimum_granularity = 0;
  std::size_t recommended_granularity = 0;
  if (!require_cuda(result, api,
                    api.mem_get_allocation_granularity_(&minimum_granularity, &allocation,
                                                        CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                    "preflight", "cuMemGetAllocationGranularity(minimum)") ||
      !require_cuda(result, api,
                    api.mem_get_allocation_granularity_(&recommended_granularity, &allocation,
                                                        CU_MEM_ALLOC_GRANULARITY_RECOMMENDED),
                    "preflight", "cuMemGetAllocationGranularity(recommended)")) {
    return finish();
  }
  device.minimum_granularity_bytes = static_cast<std::uint64_t>(minimum_granularity);
  device.recommended_granularity_bytes = static_cast<std::uint64_t>(recommended_granularity);

  std::optional<WddmIdentity> wddm_identity;
  std::optional<std::uint64_t> available_device_budget;
#ifdef _WIN32
  if (environment == nullptr) {
    if (api.device_get_luid_ == nullptr) {
      result.device = device;
      result.reason = "cuda_luid_unavailable";
      record_failure(result, exit_prerequisite, "preflight", "cuDeviceGetLuid",
                     "Windows safety preflight requires a CUDA adapter LUID");
      return finish();
    }
    WddmIdentity identity;
    std::array<char, 8> luid{};
    if (!require_cuda(result, api,
                      api.device_get_luid_(luid.data(), &identity.node_mask, resources.device),
                      "preflight", "cuDeviceGetLuid")) {
      return finish();
    }
    std::memcpy(identity.luid.data(), luid.data(), identity.luid.size());
    if (options.include_identifiers) {
      device.luid = bytes_as_hex(identity.luid.data(), identity.luid.size());
    }
    wddm_identity = identity;
    const auto observation = query_wddm(identity, result.diagnostics);
    if (!observation.has_value()) {
      result.device = device;
      result.reason = "wddm_budget_unavailable";
      record_failure(result, exit_prerequisite, "preflight", "QueryVideoMemoryInfo",
                     "Windows safety preflight requires the DXGI local-memory budget");
      return finish();
    }
    device.wddm_budget_bytes_start = observation->budget;
    device.wddm_usage_bytes_start = observation->usage;
    device.wddm_available_bytes_start = observation->available;
    device.wddm_available_bytes_minimum = observation->available;
    device.wddm_available_bytes_end = observation->available;
    available_device_budget = observation->available;
  }
#endif
  if (environment != nullptr) {
    if (!environment->initial_device_budget.has_value()) {
#ifdef _WIN32
      result.device = device;
      result.reason = "injected_budget_unavailable";
      record_failure(result, exit_prerequisite, "preflight", "injected_device_budget",
                     "Windows safety preflight requires an injected device budget");
      return finish();
#endif
    } else {
      const auto budget = *environment->initial_device_budget;
      const std::uint64_t available = budget.available_bytes();
      device.wddm_budget_bytes_start = budget.budget_bytes;
      device.wddm_usage_bytes_start = budget.usage_bytes;
      device.wddm_available_bytes_start = available;
      device.wddm_available_bytes_minimum = available;
      device.wddm_available_bytes_end = available;
      available_device_budget = available;
    }
  }
  result.device = device;

  const probe::SystemInfo system = platform::collect_system_info();
  const std::uint64_t physical_host_bytes = environment != nullptr
                                                ? environment->physical_host_bytes
                                                : system.physical_memory_bytes.value_or(0);
  const std::uint64_t available_host_bytes = environment != nullptr
                                                 ? environment->available_host_bytes
                                                 : system.available_memory_bytes.value_or(0);
  if (physical_host_bytes == 0 || available_host_bytes == 0) {
    result.reason = "host_memory_observation_unavailable";
    record_failure(result, exit_prerequisite, "preflight", "host_memory",
                   "safe logical sizing requires physical and available host memory");
    return finish();
  }
  const std::uint32_t resource_slots =
      options.mode == RequestedMode::reference ? 1U : options.window_slots;
  PlanningInput planning;
  planning.requested_logical_bytes = options.logical_bytes;
  planning.requested_chunk_bytes = options.chunk_bytes;
  planning.passes = options.passes;
  planning.window_slots = resource_slots;
  planning.total_device_bytes = device.total_memory_bytes;
  planning.free_device_bytes = device.free_memory_bytes_start;
  planning.available_device_budget_bytes = available_device_budget;
  planning.minimum_granularity_bytes = device.minimum_granularity_bytes;
  planning.recommended_granularity_bytes = device.recommended_granularity_bytes;
  planning.physical_host_bytes = physical_host_bytes;
  planning.available_host_bytes = available_host_bytes;
  planning.device_headroom_bytes = options.device_headroom_bytes;
  const PlanningResult planned = make_workload_plan(planning);
  if (!planned) {
    result.reason = std::string(plan_error_name(planned.error));
    const int code =
        planned.error == PlanError::insufficient_device_budget ? exit_oom : exit_prerequisite;
    record_failure(result, code, "planning", "make_workload_plan",
                   "the requested oversubscription proof cannot be executed safely: " +
                       std::string(plan_error_name(planned.error)));
    return finish();
  }
  result.plan = *planned.plan;
  const WorkloadPlan& plan = *result.plan;

  if (!backing.allocate(plan.backing_bytes)) {
    record_failure(result, exit_oom, "setup", "allocate_pageable_backing",
                   "pageable host backing allocation failed");
    return finish();
  }
  result.cleanup.host_backing_released = false;
  bool resources_created = false;
  try {
    resources_created = create_resources(resources, result, plan, resource_slots);
  } catch (const std::bad_alloc&) {
    try {
      record_failure(result, exit_oom, "setup", "host_allocation",
                     "host allocation failed while creating executor resources");
    } catch (...) {
      result.exit_code = exit_oom;
      result.status = "failed";
    }
  } catch (const std::exception& exception) {
    try {
      record_failure(result, exit_failure, "setup", "internal_exception", exception.what());
    } catch (...) {
      result.exit_code = exit_failure;
      result.status = "failed";
    }
  } catch (...) {
    try {
      record_failure(result, exit_failure, "setup", "internal_exception",
                     "unknown exception while creating executor resources");
    } catch (...) {
      result.exit_code = exit_failure;
      result.status = "failed";
    }
  }
  if (!resources_created) {
    return finish();
  }

  const std::uint32_t threads =
      (std::min<std::uint32_t>(256U, device.maximum_threads_per_block) / device.warp_size) *
      device.warp_size;
  const std::uint32_t blocks = device.multiprocessor_count * 4U;
  bool modes_ok = true;
  try {
    if (options.mode == RequestedMode::reference || options.mode == RequestedMode::both) {
      modes_ok = run_reference_mode(resources, result, *result.device, options, plan, wddm_identity,
                                    environment, backing.data(), progress, blocks, threads);
    }
    if (modes_ok &&
        (options.mode == RequestedMode::pipeline || options.mode == RequestedMode::both)) {
      modes_ok = run_pipeline_mode(resources, result, *result.device, options, plan, wddm_identity,
                                   environment, backing.data(), progress, blocks, threads);
    }
  } catch (const std::bad_alloc&) {
    try {
      record_failure(result, exit_oom, "execution", "host_allocation",
                     "host allocation failed while executing the proof workload");
    } catch (...) {
      result.exit_code = exit_oom;
      result.status = "failed";
    }
    modes_ok = false;
  } catch (const std::exception& exception) {
    try {
      record_failure(result, exit_failure, "execution", "internal_exception", exception.what());
    } catch (...) {
      result.exit_code = exit_failure;
      result.status = "failed";
    }
    modes_ok = false;
  } catch (...) {
    try {
      record_failure(result, exit_failure, "execution", "internal_exception",
                     "unknown exception while executing the proof workload");
    } catch (...) {
      result.exit_code = exit_failure;
      result.status = "failed";
    }
    modes_ok = false;
  }
  if (modes_ok && options.mode == RequestedMode::both &&
      result.reference.digest128 != result.pipeline.digest128) {
    record_failure(result, exit_corruption, "verification", "mode_digest",
                   "reference and pipeline modes produced different final digests");
    modes_ok = false;
  }
  if (modes_ok) {
    const auto valid_mode = [](const ModeStatistics& mode) {
      return mode.status == "completed" && mode.handle_reuses > 0 && mode.unsafe_remaps == 0 &&
             mode.mappings == mode.set_access_calls && mode.mappings == mode.unmappings &&
             mode.unmappings == mode.event_boundaries && mode.stable_addresses &&
             mode.full_verification && mode.matches_cpu;
    };
    const bool reference_required =
        options.mode == RequestedMode::reference || options.mode == RequestedMode::both;
    const bool pipeline_required =
        options.mode == RequestedMode::pipeline || options.mode == RequestedMode::both;
    const bool proof_invariants = plan.effective_logical_bytes > device.total_memory_bytes &&
                                  plan.resident_physical_bytes < plan.effective_logical_bytes &&
                                  (!reference_required || valid_mode(result.reference)) &&
                                  (!pipeline_required || valid_mode(result.pipeline));
    if (!proof_invariants) {
      record_failure(result, exit_failure, "proof", "invariants",
                     "one or more resident-only VMM proof invariants were not satisfied");
    } else {
      result.exit_code = exit_completed;
      result.status = "completed";
      result.reason.reset();
    }
  }
  return finish();
}

} // namespace xvram::vmm_poc
