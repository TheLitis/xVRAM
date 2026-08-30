#include "residency/executor.hpp"

#include "platform/dxgi_memory.hpp"
#include "platform/pageable_memory.hpp"
#include "platform/system_info.hpp"
#include "residency/core.hpp"
#include "residency/metrics.hpp"
#include "residency/policy.hpp"
#include "residency/workload.hpp"
#include "xvram/residency_workload_ptx.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
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
#include <span>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#endif

namespace xvram::residency {
namespace {

constexpr int exit_completed = 0;
constexpr int exit_prerequisite = 23;
constexpr int exit_corruption = 24;
constexpr int exit_oom = 25;
constexpr int exit_failure = 27;
constexpr std::uint64_t word_bytes = sizeof(std::uint32_t);
constexpr std::uint64_t service_host_bytes = 256ULL * 1024ULL * 1024ULL;
constexpr auto maximum_kernel_duration = std::chrono::milliseconds(250);
constexpr std::uint64_t parallel_partition_words = 64ULL * 1024ULL * 1024ULL;
constexpr std::uint64_t parallel_chunk_words = 4ULL * 1024ULL * 1024ULL;
constexpr unsigned maximum_cpu_workers = 8U;

using Clock = std::chrono::steady_clock;

[[nodiscard]] double milliseconds_between(const Clock::time_point begin,
                                          const Clock::time_point end) noexcept {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

[[nodiscard]] unsigned cpu_worker_count(const std::uint64_t word_count) noexcept {
  if (word_count <= parallel_partition_words) {
    return 1U;
  }
  const std::uint64_t useful = ((word_count - 1ULL) / parallel_partition_words) + 1ULL;
  const unsigned hardware = std::max(2U, std::thread::hardware_concurrency());
  return static_cast<unsigned>(
      std::min<std::uint64_t>({useful, maximum_cpu_workers, hardware}));
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

  std::atomic<std::uint64_t> next{0};
  std::atomic<unsigned> active{0};
  std::mutex exception_mutex;
  std::exception_ptr exception;
  std::vector<std::jthread> workers;
  workers.reserve(worker_count);
  for (unsigned index = 0; index < worker_count; ++index) {
    active.fetch_add(1U, std::memory_order_relaxed);
    workers.emplace_back([&]() noexcept {
      try {
        for (;;) {
          const std::uint64_t begin =
              next.fetch_add(parallel_chunk_words, std::memory_order_relaxed);
          if (begin >= word_count) {
            break;
          }
          work(begin, std::min(word_count, begin + parallel_chunk_words));
        }
      } catch (...) {
        const std::lock_guard lock(exception_mutex);
        if (exception == nullptr) {
          exception = std::current_exception();
        }
        next.store(word_count, std::memory_order_relaxed);
      }
      active.fetch_sub(1U, std::memory_order_release);
    });
  }
  while (active.load(std::memory_order_acquire) != 0U) {
    heartbeat();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  workers.clear();
  if (exception != nullptr) {
    std::rethrow_exception(exception);
  }
}

[[nodiscard]] std::string cuda_name(cuda::CudaApi& api, const cuda::abi::Result code) {
  const char* name = nullptr;
  if (api.get_error_name_ != nullptr && api.get_error_name_(code, &name) == cuda::abi::success &&
      name != nullptr) {
    return name;
  }
  return "CUDA_ERROR_" + std::to_string(static_cast<std::int64_t>(code));
}

[[nodiscard]] std::string cuda_message(cuda::CudaApi& api, const cuda::abi::Result code) {
  const char* message = nullptr;
  if (api.get_error_string_ != nullptr &&
      api.get_error_string_(code, &message) == cuda::abi::success && message != nullptr) {
    return message;
  }
  return cuda_name(api, code);
}

[[nodiscard]] int classify_cuda(const cuda::abi::Result code) noexcept {
  return code == CUDA_ERROR_OUT_OF_MEMORY ? exit_oom : exit_failure;
}

void append_diagnostic(ExecutorResult& result, const probe::DiagnosticLevel level,
                       std::string operation, std::string message,
                       const std::optional<std::int64_t> code = std::nullopt) noexcept {
  try {
    result.diagnostics.emplace_back(level, "residency", std::move(operation),
                                    std::move(message), code);
  } catch (...) {
  }
}

void record_failure(ExecutorResult& result, const int exit_code, std::string stage,
                    std::string operation, std::string message,
                    const std::optional<cuda::abi::Result> native = std::nullopt,
                    cuda::CudaApi* api = nullptr,
                    const std::optional<ChunkKey> key = std::nullopt,
                    const std::optional<std::uint64_t> byte_offset = std::nullopt,
                    const std::optional<std::string> policy = std::nullopt,
                    const std::optional<std::string> scenario = std::nullopt) {
  if (result.failure.has_value()) {
    return;
  }
  result.exit_code = exit_code;
  result.status = exit_code == exit_prerequisite ? "skipped" : "failed";
  ExecutionFailure failure;
  failure.stage = std::move(stage);
  failure.operation = std::move(operation);
  failure.message = std::move(message);
  failure.logical_byte_offset = byte_offset;
  failure.policy = policy;
  failure.scenario = scenario;
  if (key.has_value()) {
    failure.allocation_id = key->allocation_id.value;
    failure.chunk_index = key->chunk_index;
  }
  if (native.has_value()) {
    failure.native_code = static_cast<std::int64_t>(*native);
    if (api != nullptr) {
      failure.native_name = cuda_name(*api, *native);
    }
  }
  result.failure = std::move(failure);
}

[[nodiscard]] bool require_cuda(ExecutorResult& result, cuda::CudaApi& api,
                                const cuda::abi::Result code, const char* stage,
                                const char* operation,
                                const std::optional<ChunkKey> key = std::nullopt) {
  if (code == cuda::abi::success) {
    return true;
  }
  record_failure(result, classify_cuda(code), stage, operation, cuda_message(api, code), code,
                 &api, key);
  return false;
}

[[nodiscard]] bool have_executor_symbols(const cuda::CudaApi& api) noexcept {
  return api.init_ != nullptr && api.device_get_count_ != nullptr && api.device_get_ != nullptr &&
         api.device_get_name_ != nullptr && api.device_total_memory_ != nullptr &&
         api.device_get_attribute_ != nullptr && api.context_get_current_ != nullptr &&
         api.context_get_device_ != nullptr && api.context_set_current_ != nullptr &&
         api.context_create_ != nullptr && api.context_destroy_ != nullptr &&
         api.mem_get_info_ != nullptr && api.mem_get_allocation_granularity_ != nullptr &&
         api.mem_address_reserve_ != nullptr && api.mem_address_free_ != nullptr &&
         api.mem_create_ != nullptr && api.mem_release_ != nullptr && api.mem_map_ != nullptr &&
         api.mem_unmap_ != nullptr && api.mem_set_access_ != nullptr &&
         api.mem_get_access_ != nullptr && api.mem_alloc_ != nullptr && api.mem_free_ != nullptr &&
         api.mem_host_alloc_ != nullptr && api.mem_free_host_ != nullptr &&
         api.memcpy_h2d_async_ != nullptr && api.memcpy_d2h_async_ != nullptr &&
         api.stream_create_ != nullptr && api.stream_destroy_ != nullptr &&
         api.stream_synchronize_ != nullptr && api.event_create_ != nullptr &&
         api.event_destroy_ != nullptr && api.event_record_ != nullptr &&
         api.event_query_ != nullptr && api.event_elapsed_time_ != nullptr &&
         api.module_load_data_ != nullptr && api.module_get_function_ != nullptr &&
         api.module_unload_ != nullptr && api.launch_kernel_ != nullptr;
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

[[nodiscard]] bool get_attribute(cuda::CudaApi& api, ExecutorResult& result,
                                 const cuda::abi::Device device,
                                 const cuda::abi::NativeDeviceAttribute attribute,
                                 const char* operation, std::uint32_t& output) {
  int value = 0;
  if (!require_cuda(result, api, api.device_get_attribute_(&value, attribute, device),
                    "preflight", operation)) {
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

struct WddmIdentity {
  platform::AdapterLuid luid{};
  std::uint32_t node_mask = 0;
};

[[nodiscard]] std::optional<BudgetSnapshot>
query_wddm(const WddmIdentity& identity, ExecutorResult& result) {
#ifdef _WIN32
  std::vector<probe::Diagnostic> diagnostics;
  const auto adapter = platform::query_dxgi_memory(identity.luid, identity.node_mask, diagnostics);
  for (auto& diagnostic : diagnostics) {
    result.diagnostics.push_back(std::move(diagnostic));
  }
  if (!adapter.has_value() || !adapter->local.has_value()) {
    return std::nullopt;
  }
  return BudgetSnapshot{adapter->local->budget_bytes, adapter->local->current_usage_bytes};
#else
  (void)identity;
  (void)result;
  return std::nullopt;
#endif
}

struct Allocation {
  AllocationId id;
  std::uint64_t logical_bytes = 0;
  std::uint64_t mapped_bytes = 0;
  cuda::abi::DevicePointer reservation = 0;
  std::uint64_t reservation_bytes = 0;
  cuda::abi::DevicePointer logical_base = 0;
  platform::PageableMemory backing;
  std::vector<ChunkRecord> chunks;
  bool reservation_quarantined = false;
};

struct Frame {
  cuda::abi::GenericAllocationHandle handle{};
  std::optional<ChunkKey> key;
  cuda::abi::DevicePointer address = 0;
  std::uint64_t mapped_bytes = 0;
  std::uint64_t mapping_generations = 0;
  bool mapped = false;
};

struct TransferSlot {
  void* pinned = nullptr;
  cuda::abi::Event started = nullptr;
  cuda::abi::Event done = nullptr;
  std::optional<ChunkKey> key;
  std::uint64_t valid_bytes = 0;
  std::uint64_t generation = 0;
  std::uint64_t operation_id = 0;
  Clock::time_point submitted_at{};
  bool busy = false;
  bool speculative = false;
  bool evict_after = false;
};

struct ComputeSlot {
  cuda::abi::Event started = nullptr;
  cuda::abi::Event done = nullptr;
  std::optional<ChunkKey> key;
  std::uint64_t generation = 0;
  std::uint64_t operation_id = 0;
  AccessMode mode = AccessMode::read;
  Clock::time_point submitted_at{};
  bool busy = false;
};

struct ManagerResources {
  cuda::CudaApi* api = nullptr;
  cuda::abi::Device device = 0;
  cuda::abi::Module module = nullptr;
  cuda::abi::Function kernel = nullptr;
  cuda::abi::Stream h2d_stream = nullptr;
  cuda::abi::Stream compute_stream = nullptr;
  cuda::abi::Stream d2h_stream = nullptr;
  cuda::abi::DevicePointer token_device = 0;
  std::uint64_t token_bytes = 0;
  std::vector<TransferSlot> h2d_slots;
  std::vector<TransferSlot> d2h_slots;
  ComputeSlot compute;
  std::vector<Frame> frames;
};

[[nodiscard]] std::uint64_t valid_chunk_bytes(const Allocation& allocation,
                                              const std::uint64_t chunk_bytes,
                                              const std::uint64_t chunk_index) noexcept {
  const std::uint64_t offset = chunk_index * chunk_bytes;
  return std::min(chunk_bytes, allocation.logical_bytes - offset);
}

[[nodiscard]] bool key_less(const ChunkKey& left, const ChunkKey& right) noexcept {
  if (left.allocation_id.value != right.allocation_id.value) {
    return left.allocation_id.value < right.allocation_id.value;
  }
  return left.chunk_index < right.chunk_index;
}

class CacheManager {
public:
  CacheManager(cuda::CudaApi& api, ExecutorResult& result, const ExecutorOptions& options,
               const cuda::abi::Device device, const std::uint64_t chunk_bytes,
               const std::uint64_t initial_target_bytes, const std::uint64_t configured_cap_bytes,
               const std::uint32_t blocks, const std::uint32_t threads,
               std::function<std::optional<BudgetSnapshot>()> budget_provider,
               TraceCallback trace)
      : api_(api), result_(result), options_(options), device_(device), chunk_bytes_(chunk_bytes),
        configured_cap_bytes_(configured_cap_bytes), target_state_{initial_target_bytes, 0},
        blocks_(blocks), threads_(threads), budget_provider_(std::move(budget_provider)),
        trace_(std::move(trace)) {
    resources_.api = &api_;
    resources_.device = device_;
    metrics_.staging_slots_capacity = options_.staging_slots;
    metrics_.target_peak_bytes = initial_target_bytes;
    target_minimum_bytes_ = initial_target_bytes;
    target_maximum_bytes_ = initial_target_bytes;
  }

  CacheManager(const CacheManager&) = delete;
  CacheManager& operator=(const CacheManager&) = delete;

  [[nodiscard]] bool setup();
  [[nodiscard]] std::optional<AllocationId> allocate(std::uint64_t bytes);
  [[nodiscard]] bool release(AllocationId allocation_id);
  [[nodiscard]] bool enqueue_transaction(const ScenarioOperation& operation,
                                         std::span<std::uint32_t> expected_tokens,
                                         std::uint64_t& expected_token);
  [[nodiscard]] bool prefetch(std::span<const ChunkKey> keys, std::uint64_t operation_id);
  [[nodiscard]] bool poll();
  [[nodiscard]] bool drain(bool release_mappings);
  void close() noexcept;

  void set_policy(const RequestedPolicy policy);
  void set_workload_context(std::string policy, std::string scenario) {
    current_policy_ = std::move(policy);
    current_scenario_ = std::move(scenario);
  }
  [[nodiscard]] bool reset_backing(AllocationId allocation_id,
                                   const std::function<void()>& heartbeat);
  [[nodiscard]] std::optional<std::uint32_t*> backing_words(AllocationId allocation_id);
  [[nodiscard]] bool prepare_tokens(std::uint64_t count);
  [[nodiscard]] bool collect_tokens(std::span<std::uint32_t> output);
  [[nodiscard]] bool observe_budget(bool force);
  [[nodiscard]] bool acquire_pressure(std::uint64_t bytes);
  [[nodiscard]] bool release_pressure();
  [[nodiscard]] bool verify_and_flush();

  [[nodiscard]] const CacheMetrics& metrics() const noexcept {
    return metrics_;
  }
  [[nodiscard]] std::uint64_t unsafe_transitions() const noexcept {
    return unsafe_transitions_;
  }
  [[nodiscard]] std::uint64_t event_boundaries() const noexcept {
    return event_boundaries_;
  }
  [[nodiscard]] std::uint64_t target_bytes() const noexcept {
    return target_state_.current_target_bytes;
  }
  [[nodiscard]] std::uint64_t target_minimum_bytes() const noexcept {
    return target_minimum_bytes_;
  }
  [[nodiscard]] std::uint64_t target_maximum_bytes() const noexcept {
    return target_maximum_bytes_;
  }
  [[nodiscard]] std::uint64_t target_shrinks() const noexcept {
    return target_shrinks_;
  }
  [[nodiscard]] std::uint64_t target_grows() const noexcept {
    return target_grows_;
  }
  [[nodiscard]] std::uint64_t target_oom_retries() const noexcept {
    return target_oom_retries_;
  }
  [[nodiscard]] std::uint64_t trace_records() const noexcept {
    return trace_sequence_;
  }
  [[nodiscard]] std::uint64_t budget_samples() const noexcept {
    return budget_samples_;
  }
  [[nodiscard]] std::optional<std::uint64_t> cuda_free_minimum() const noexcept {
    return cuda_free_minimum_;
  }
  [[nodiscard]] std::optional<std::uint64_t> cuda_free_end() const noexcept {
    return cuda_free_end_;
  }
  [[nodiscard]] std::optional<std::uint64_t> wddm_available_minimum() const noexcept {
    return wddm_available_minimum_;
  }
  [[nodiscard]] std::optional<std::uint64_t> wddm_available_end() const noexcept {
    return wddm_available_end_;
  }
  [[nodiscard]] std::uint64_t resident_samples() const noexcept {
    return resident_samples_;
  }
  [[nodiscard]] long double resident_ratio_sum() const noexcept {
    return resident_ratio_sum_;
  }
  [[nodiscard]] const std::vector<double>& remap_samples() const noexcept {
    return remap_samples_;
  }
  [[nodiscard]] const std::vector<double>& h2d_samples() const noexcept {
    return h2d_samples_;
  }
  [[nodiscard]] const std::vector<double>& kernel_samples() const noexcept {
    return kernel_samples_;
  }
  [[nodiscard]] const std::vector<double>& d2h_samples() const noexcept {
    return d2h_samples_;
  }
  [[nodiscard]] const std::vector<double>& writeback_samples() const noexcept {
    return writeback_samples_;
  }
  [[nodiscard]] bool stable_addresses() const noexcept {
    return stable_addresses_;
  }
  [[nodiscard]] bool set_access_verified() const noexcept {
    return set_access_verified_;
  }
  [[nodiscard]] bool no_aliases() const noexcept {
    return no_aliases_;
  }
  [[nodiscard]] bool dirty_writeback_verified() const noexcept {
    return dirty_writeback_verified_;
  }
  [[nodiscard]] std::optional<std::uint64_t> first_mismatch_offset() const noexcept {
    return first_mismatch_offset_;
  }
  [[nodiscard]] std::uint64_t mismatch_count() const noexcept {
    return mismatch_count_;
  }
  [[nodiscard]] bool closed() const noexcept {
    return closed_;
  }

private:
  [[nodiscard]] Allocation* allocation(AllocationId id) noexcept;
  [[nodiscard]] const Allocation* allocation(AllocationId id) const noexcept;
  [[nodiscard]] ChunkRecord* chunk(ChunkKey key) noexcept;
  [[nodiscard]] const ChunkRecord* chunk(ChunkKey key) const noexcept;
  [[nodiscard]] Frame* frame_for(ChunkKey key) noexcept;
  [[nodiscard]] std::optional<std::size_t> frame_index_for(ChunkKey key) const noexcept;
  [[nodiscard]] bool transition(ChunkRecord& record, ChunkState target, std::string reason,
                                std::optional<std::uint64_t> operation_id = std::nullopt);
  void emit_trace(const ChunkRecord* record, std::string event, std::string reason,
                  std::optional<ChunkState> from = std::nullopt,
                  std::optional<std::uint64_t> operation_id = std::nullopt,
                  std::optional<std::uint64_t> bytes = std::nullopt,
                  std::optional<std::uint64_t> generation = std::nullopt,
                  bool speculative = false) noexcept;
  [[nodiscard]] bool create_event(cuda::abi::Event& event);
  [[nodiscard]] bool create_handle(Frame& frame, bool allow_retry);
  [[nodiscard]] std::optional<std::size_t> acquire_frame(ChunkKey incoming,
                                                         std::uint64_t operation_id);
  [[nodiscard]] bool map_chunk(ChunkKey key, std::size_t frame_index, bool speculative,
                               bool requires_h2d, std::uint64_t operation_id,
                               bool wait_for_completion);
  [[nodiscard]] bool finish_h2d(TransferSlot& slot, bool wait);
  [[nodiscard]] bool finish_d2h(TransferSlot& slot, bool wait);
  [[nodiscard]] bool finish_compute(bool wait);
  [[nodiscard]] bool query_event(cuda::abi::Event event, Clock::time_point submitted,
                                 bool wait, const char* operation, bool& complete);
  [[nodiscard]] bool evict_chunk(ChunkKey key, bool wait, std::uint64_t operation_id);
  [[nodiscard]] bool unmap_clean(ChunkKey key, std::uint64_t operation_id);
  [[nodiscard]] bool schedule_writeback(ChunkKey key, bool evict_after,
                                        std::uint64_t operation_id);
  [[nodiscard]] TransferSlot* free_h2d_slot();
  [[nodiscard]] TransferSlot* free_d2h_slot();
  [[nodiscard]] bool schedule_pending_prefetches();
  [[nodiscard]] bool promote_prefetch(ChunkKey key, std::uint64_t operation_id);
  [[nodiscard]] bool ensure_resident(const ChunkAccessPlan& plan, std::uint64_t operation_id,
                                     bool& hit);
  [[nodiscard]] bool launch_kernel(const ChunkAccessPlan& plan, std::uint64_t operation_id,
                                   std::uint32_t operation_index);
  [[nodiscard]] bool apply_cpu_reference(const ChunkAccessPlan& plan,
                                         std::uint32_t operation_index,
                                         std::uint64_t& token);
  [[nodiscard]] bool shrink_to(std::uint64_t bytes);
  [[nodiscard]] bool release_excess_handles();
  [[nodiscard]] bool validate_context();
  [[nodiscard]] std::uint64_t live_handle_bytes() const noexcept;
  [[nodiscard]] std::uint64_t resident_bytes() const noexcept;
  [[nodiscard]] std::vector<VictimCandidate> victim_candidates() const;
  [[nodiscard]] std::optional<ChunkKey> select_victim();
  void poison(std::string operation, std::string message,
              std::optional<ChunkKey> key = std::nullopt) noexcept;

  cuda::CudaApi& api_;
  ExecutorResult& result_;
  const ExecutorOptions& options_;
  cuda::abi::Device device_ = 0;
  std::uint64_t chunk_bytes_ = 0;
  std::uint64_t configured_cap_bytes_ = 0;
  TargetHysteresisState target_state_;
  std::uint32_t blocks_ = 0;
  std::uint32_t threads_ = 0;
  std::function<std::optional<BudgetSnapshot>()> budget_provider_;
  TraceCallback trace_;
  ManagerResources resources_;
  std::unordered_map<std::uint64_t, std::unique_ptr<Allocation>> allocations_;
  AllocationId next_allocation_id_{1};
  std::unique_ptr<VictimPolicy> policy_;
  std::deque<ChunkKey> pending_prefetch_;
  CacheMetrics metrics_;
  std::string current_policy_ = "clock";
  std::string current_scenario_ = "unknown";
  std::uint64_t sequence_ = 0;
  std::uint64_t trace_sequence_ = 0;
  std::uint64_t unsafe_transitions_ = 0;
  std::uint64_t event_boundaries_ = 0;
  std::uint64_t target_minimum_bytes_ = 0;
  std::uint64_t target_maximum_bytes_ = 0;
  std::uint64_t target_shrinks_ = 0;
  std::uint64_t target_grows_ = 0;
  std::uint64_t target_oom_retries_ = 0;
  std::uint64_t budget_samples_ = 0;
  std::optional<std::uint64_t> cuda_free_minimum_;
  std::optional<std::uint64_t> cuda_free_end_;
  std::optional<std::uint64_t> wddm_available_minimum_;
  std::optional<std::uint64_t> wddm_available_end_;
  Clock::time_point last_budget_poll_{};
  cuda::abi::DevicePointer pressure_allocation_ = 0;
  std::uint64_t pressure_bytes_ = 0;
  std::uint64_t resident_samples_ = 0;
  long double resident_ratio_sum_ = 0.0L;
  std::vector<double> remap_samples_;
  std::vector<double> h2d_samples_;
  std::vector<double> kernel_samples_;
  std::vector<double> d2h_samples_;
  std::vector<double> writeback_samples_;
  bool stable_addresses_ = true;
  bool set_access_verified_ = true;
  bool no_aliases_ = true;
  bool dirty_writeback_verified_ = true;
  bool poisoned_ = false;
  bool closed_ = false;
  bool quarantine_ = false;
  std::uint64_t mismatch_count_ = 0;
  std::optional<std::uint64_t> first_mismatch_offset_;
};

Allocation* CacheManager::allocation(const AllocationId id) noexcept {
  const auto found = allocations_.find(id.value);
  return found == allocations_.end() ? nullptr : found->second.get();
}

const Allocation* CacheManager::allocation(const AllocationId id) const noexcept {
  const auto found = allocations_.find(id.value);
  return found == allocations_.end() ? nullptr : found->second.get();
}

ChunkRecord* CacheManager::chunk(const ChunkKey key) noexcept {
  Allocation* owner = allocation(key.allocation_id);
  if (owner == nullptr || key.chunk_index >= owner->chunks.size()) {
    return nullptr;
  }
  return &owner->chunks[static_cast<std::size_t>(key.chunk_index)];
}

const ChunkRecord* CacheManager::chunk(const ChunkKey key) const noexcept {
  const Allocation* owner = allocation(key.allocation_id);
  if (owner == nullptr || key.chunk_index >= owner->chunks.size()) {
    return nullptr;
  }
  return &owner->chunks[static_cast<std::size_t>(key.chunk_index)];
}

std::optional<std::size_t> CacheManager::frame_index_for(const ChunkKey key) const noexcept {
  const ChunkRecord* record = chunk(key);
  if (record == nullptr || !record->frame_index.has_value()) {
    return std::nullopt;
  }
  const std::uint64_t index = *record->frame_index;
  if (index >= resources_.frames.size()) {
    return std::nullopt;
  }
  return static_cast<std::size_t>(index);
}

Frame* CacheManager::frame_for(const ChunkKey key) noexcept {
  const auto index = frame_index_for(key);
  return index.has_value() ? &resources_.frames[*index] : nullptr;
}

void CacheManager::emit_trace(const ChunkRecord* record, std::string event, std::string reason,
                              const std::optional<ChunkState> from,
                              const std::optional<std::uint64_t> operation_id,
                              const std::optional<std::uint64_t> bytes,
                              const std::optional<std::uint64_t> generation,
                              const bool speculative) noexcept {
  if (!options_.trace_enabled || !trace_) {
    return;
  }
  try {
    TraceRecord entry;
    entry.sequence = ++trace_sequence_;
    entry.monotonic_time_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
            .count());
    entry.event = std::move(event);
    entry.reason = std::move(reason);
    entry.operation_id = operation_id;
    entry.policy = current_policy_;
    entry.speculative = speculative;
    entry.bytes = bytes;
    entry.generation = generation;
    if (record != nullptr) {
      entry.allocation_id = record->key.allocation_id.value;
      entry.chunk_index = record->key.chunk_index;
      entry.to_state = std::string(chunk_state_name(record->state));
      if (from.has_value()) {
        entry.from_state = std::string(chunk_state_name(*from));
      }
    }
    trace_(entry);
  } catch (...) {
    append_diagnostic(result_, probe::DiagnosticLevel::error, "trace_callback",
                      "trace callback raised an exception");
  }
}

void CacheManager::poison(std::string operation, std::string message,
                          const std::optional<ChunkKey> key) noexcept {
  poisoned_ = true;
  if (key.has_value()) {
    if (ChunkRecord* record = chunk(*key); record != nullptr) {
      const ChunkState from = record->state;
      record->state = ChunkState::poisoned;
      emit_trace(record, "transition", "poison", from);
    }
  }
  try {
    record_failure(result_, exit_failure, "execution", std::move(operation), std::move(message),
                   std::nullopt, nullptr, key, std::nullopt, current_policy_, current_scenario_);
  } catch (...) {
  }
}

bool CacheManager::transition(ChunkRecord& record, const ChunkState target, std::string reason,
                              const std::optional<std::uint64_t> operation_id) {
  const ChunkState from = record.state;
  if (transition_chunk_state(record.state, target) != StateTransitionResult::success) {
    ++unsafe_transitions_;
    poison("state_transition",
           "illegal transition from " + std::string(chunk_state_name(from)) + " to " +
               std::string(chunk_state_name(target)),
           record.key);
    return false;
  }
  emit_trace(&record, "transition", std::move(reason), from, operation_id, std::nullopt,
             record.event_generation, record.speculative);
  return true;
}

bool CacheManager::create_event(cuda::abi::Event& event) {
  return require_cuda(result_, api_, api_.event_create_(&event, CU_EVENT_DEFAULT), "setup",
                      "cuEventCreate");
}

void CacheManager::set_policy(const RequestedPolicy policy) {
  if (policy == RequestedPolicy::lru) {
    policy_ = std::make_unique<LruPolicy>();
  } else {
    policy_ = std::make_unique<ClockPolicy>();
  }
  current_policy_ = std::string(policy_->name());
}

bool CacheManager::setup() {
  if (!require_cuda(result_, api_,
                    api_.module_load_data_(&resources_.module, residency_workload_ptx.data()),
                    "setup", "cuModuleLoadData") ||
      !require_cuda(result_, api_,
                    api_.module_get_function_(&resources_.kernel, resources_.module,
                                              "xvram_residency_workload_v1"),
                    "setup", "cuModuleGetFunction") ||
      !require_cuda(result_, api_, api_.stream_create_(&resources_.h2d_stream,
                                                       CU_STREAM_NON_BLOCKING),
                    "setup", "cuStreamCreate(h2d)") ||
      !require_cuda(result_, api_, api_.stream_create_(&resources_.compute_stream,
                                                       CU_STREAM_NON_BLOCKING),
                    "setup", "cuStreamCreate(compute)") ||
      !require_cuda(result_, api_, api_.stream_create_(&resources_.d2h_stream,
                                                       CU_STREAM_NON_BLOCKING),
                    "setup", "cuStreamCreate(d2h)")) {
    return false;
  }

  const std::uint32_t h2d_count = options_.staging_slots / 2U;
  const std::uint32_t d2h_count = options_.staging_slots - h2d_count;
  resources_.h2d_slots.resize(h2d_count);
  resources_.d2h_slots.resize(d2h_count);
  const auto create_slots = [&](std::vector<TransferSlot>& slots) -> bool {
    for (TransferSlot& slot : slots) {
      if (!require_cuda(result_, api_,
                        api_.mem_host_alloc_(&slot.pinned, static_cast<std::size_t>(chunk_bytes_),
                                             CU_MEMHOSTALLOC_PORTABLE),
                        "setup", "cuMemHostAlloc") ||
          !create_event(slot.started) || !create_event(slot.done)) {
        return false;
      }
    }
    return true;
  };
  if (!create_slots(resources_.h2d_slots) || !create_slots(resources_.d2h_slots) ||
      !create_event(resources_.compute.started) || !create_event(resources_.compute.done)) {
    return false;
  }
  set_policy(options_.policy == RequestedPolicy::lru ? RequestedPolicy::lru
                                                      : RequestedPolicy::clock);
  result_.cleanup.pinned_staging_released = false;
  result_.cleanup.events_destroyed = false;
  result_.cleanup.streams_destroyed = false;
  result_.cleanup.module_unloaded = false;
  return validate_context();
}

bool CacheManager::validate_context() {
  cuda::abi::Device current = -1;
  if (!require_cuda(result_, api_, api_.context_get_device_(&current), "setup",
                    "cuCtxGetDevice")) {
    return false;
  }
  if (current != device_) {
    poison("cuCtxGetDevice", "active CUDA context belongs to a different device");
    return false;
  }
  return true;
}

std::optional<AllocationId> CacheManager::allocate(const std::uint64_t bytes) {
  if (bytes == 0 || bytes % word_bytes != 0 ||
      bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
      !next_allocation_id_) {
    record_failure(result_, exit_prerequisite, "configuration", "allocate",
                   "logical allocation size or AllocationId is invalid");
    return std::nullopt;
  }
  const auto chunk_count = checked_add((bytes - 1ULL) / chunk_bytes_, 1ULL);
  const auto mapped_bytes = chunk_count.has_value()
                                ? checked_multiply(*chunk_count, chunk_bytes_)
                                : std::nullopt;
  const auto reservation_bytes = mapped_bytes.has_value()
                                     ? checked_add(*mapped_bytes, chunk_bytes_)
                                     : std::nullopt;
  if (!chunk_count.has_value() || !mapped_bytes.has_value() ||
      !reservation_bytes.has_value() ||
      *chunk_count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    record_failure(result_, exit_prerequisite, "planning", "allocate",
                   "logical allocation layout overflowed");
    return std::nullopt;
  }

  auto owner = std::make_unique<Allocation>();
  owner->id = next_allocation_id_;
  owner->logical_bytes = bytes;
  owner->mapped_bytes = *mapped_bytes;
  owner->reservation_bytes = *reservation_bytes;
  if (!owner->backing.allocate(bytes)) {
    record_failure(result_, exit_oom, "setup", "allocate_pageable_backing",
                   "pageable host backing allocation failed");
    return std::nullopt;
  }
  if (!require_cuda(result_, api_,
                    api_.mem_address_reserve_(&owner->reservation,
                                              static_cast<std::size_t>(owner->reservation_bytes),
                                              0, 0, 0),
                    "setup", "cuMemAddressReserve")) {
    (void)owner->backing.release();
    return std::nullopt;
  }
  const auto aligned = checked_align_up(static_cast<std::uint64_t>(owner->reservation),
                                        chunk_bytes_);
  const auto aligned_end = aligned.has_value() ? checked_add(*aligned, owner->mapped_bytes)
                                                : std::nullopt;
  const auto reservation_end =
      checked_add(static_cast<std::uint64_t>(owner->reservation), owner->reservation_bytes);
  if (!aligned.has_value() || !aligned_end.has_value() || !reservation_end.has_value() ||
      *aligned_end > *reservation_end) {
    record_failure(result_, exit_failure, "setup", "align_logical_base",
                   "padded CUDA VA reservation cannot contain the logical heap");
    (void)api_.mem_address_free_(owner->reservation,
                                static_cast<std::size_t>(owner->reservation_bytes));
    (void)owner->backing.release();
    return std::nullopt;
  }
  owner->logical_base = static_cast<cuda::abi::DevicePointer>(*aligned);
  owner->chunks.resize(static_cast<std::size_t>(*chunk_count));
  for (std::uint64_t index = 0; index < *chunk_count; ++index) {
    owner->chunks[static_cast<std::size_t>(index)].key = ChunkKey{owner->id, index};
  }
  const AllocationId id = owner->id;
  allocations_.emplace(id.value, std::move(owner));
  if (next_allocation_id_.value == std::numeric_limits<std::uint64_t>::max()) {
    next_allocation_id_ = {};
  } else {
    ++next_allocation_id_.value;
  }
  result_.cleanup.host_backing_released = false;
  result_.cleanup.virtual_reservations_released = false;
  emit_trace(nullptr, "allocation", "allocate", std::nullopt, std::nullopt, bytes);
  return id;
}

std::optional<std::uint32_t*> CacheManager::backing_words(const AllocationId id) {
  Allocation* owner = allocation(id);
  if (owner == nullptr) {
    return std::nullopt;
  }
  return static_cast<std::uint32_t*>(owner->backing.data());
}

bool CacheManager::reset_backing(const AllocationId id,
                                 const std::function<void()>& heartbeat) {
  Allocation* owner = allocation(id);
  if (owner == nullptr || !drain(true)) {
    return false;
  }
  auto* words = static_cast<std::uint32_t*>(owner->backing.data());
  const std::uint64_t count = owner->logical_bytes / word_bytes;
  parallel_word_chunks(
      count,
      [&](const std::uint64_t begin, const std::uint64_t end) {
        for (std::uint64_t index = begin; index < end; ++index) {
          words[index] = initial_word(index, options_.seed);
        }
      },
      heartbeat);
  return true;
}

std::uint64_t CacheManager::live_handle_bytes() const noexcept {
  if (metrics_.live_handles > std::numeric_limits<std::uint64_t>::max() / chunk_bytes_) {
    return std::numeric_limits<std::uint64_t>::max();
  }
  return metrics_.live_handles * chunk_bytes_;
}

std::uint64_t CacheManager::resident_bytes() const noexcept {
  if (metrics_.active_mappings > std::numeric_limits<std::uint64_t>::max() / chunk_bytes_) {
    return std::numeric_limits<std::uint64_t>::max();
  }
  return metrics_.active_mappings * chunk_bytes_;
}

bool CacheManager::create_handle(Frame& frame, const bool allow_retry) {
  CUmemAllocationProp property{};
  property.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  property.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  property.location.id = device_;
  property.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
  cuda::abi::Result code = api_.mem_create_(&frame.handle, static_cast<std::size_t>(chunk_bytes_),
                                             &property, 0);
  if (code == CUDA_ERROR_OUT_OF_MEMORY && allow_retry) {
    ++target_oom_retries_;
    if (!observe_budget(true) || !release_excess_handles()) {
      return false;
    }
    code = api_.mem_create_(&frame.handle, static_cast<std::size_t>(chunk_bytes_), &property, 0);
  }
  if (!require_cuda(result_, api_, code, "execution", "cuMemCreate")) {
    return false;
  }
  ++metrics_.handles_created;
  ++metrics_.live_handles;
  metrics_.resident_peak_bytes = std::max(metrics_.resident_peak_bytes, resident_bytes());
  result_.cleanup.physical_handles_released = false;
  return true;
}

std::vector<VictimCandidate> CacheManager::victim_candidates() const {
  std::vector<VictimCandidate> candidates;
  candidates.reserve(static_cast<std::size_t>(metrics_.active_mappings));
  for (const auto& [id, owner] : allocations_) {
    (void)id;
    for (const ChunkRecord& record : owner->chunks) {
      if (!record.frame_index.has_value()) {
        continue;
      }
      candidates.push_back(VictimCandidate{record.key, record.state, record.pin_count,
                                           chunk_has_in_flight_work(record),
                                           record.in_current_working_set});
    }
  }
  std::sort(candidates.begin(), candidates.end(), [](const VictimCandidate& left,
                                                     const VictimCandidate& right) {
    return key_less(left.key, right.key);
  });
  return candidates;
}

std::optional<ChunkKey> CacheManager::select_victim() {
  if (policy_ == nullptr) {
    return std::nullopt;
  }
  const std::vector<VictimCandidate> candidates = victim_candidates();
  const std::optional<ChunkKey> selected = policy_->select_victim(candidates);
  if (!selected.has_value()) {
    return std::nullopt;
  }
  const ChunkRecord* record = chunk(*selected);
  if (record == nullptr || !is_victim_eligible(*record) || record->in_current_working_set) {
    poison("select_victim", "policy selected a pinned, in-flight, or working-set chunk",
           selected);
    return std::nullopt;
  }
  return selected;
}

std::optional<std::size_t> CacheManager::acquire_frame(const ChunkKey incoming,
                                                       const std::uint64_t operation_id) {
  for (std::size_t index = 0; index < resources_.frames.size(); ++index) {
    Frame& frame = resources_.frames[index];
    if (frame.handle != 0 && !frame.mapped) {
      return index;
    }
  }

  const std::uint64_t frame_limit = target_state_.current_target_bytes / chunk_bytes_;
  if (metrics_.live_handles < frame_limit) {
    for (std::size_t index = 0; index < resources_.frames.size(); ++index) {
      if (resources_.frames[index].handle == 0) {
        if (!create_handle(resources_.frames[index], true)) {
          return std::nullopt;
        }
        return index;
      }
    }
    resources_.frames.emplace_back();
    if (!create_handle(resources_.frames.back(), true)) {
      resources_.frames.pop_back();
      return std::nullopt;
    }
    metrics_.resident_peak_bytes = std::max(metrics_.resident_peak_bytes, resident_bytes());
    return resources_.frames.size() - 1U;
  }

  emit_trace(nullptr, "victim_select", "cache_full", std::nullopt, operation_id,
             resident_bytes());
  const std::optional<ChunkKey> victim = select_victim();
  if (!victim.has_value()) {
    record_failure(result_, exit_oom, "execution", "acquire_frame",
                   "cache target has no event-safe victim", std::nullopt, nullptr, incoming,
                   std::nullopt, current_policy_, current_scenario_);
    return std::nullopt;
  }
  const auto victim_frame = frame_index_for(*victim);
  emit_trace(chunk(*victim), "victim_selected", "policy", std::nullopt, operation_id);
  if (!victim_frame.has_value() || !evict_chunk(*victim, true, operation_id)) {
    return std::nullopt;
  }
  return victim_frame;
}

TransferSlot* CacheManager::free_h2d_slot() {
  for (TransferSlot& slot : resources_.h2d_slots) {
    if (!slot.busy) {
      return &slot;
    }
  }
  return nullptr;
}

TransferSlot* CacheManager::free_d2h_slot() {
  for (TransferSlot& slot : resources_.d2h_slots) {
    if (!slot.busy) {
      return &slot;
    }
  }
  return nullptr;
}

bool CacheManager::query_event(const cuda::abi::Event event, const Clock::time_point submitted,
                               const bool wait, const char* operation, bool& complete) {
  complete = false;
  for (;;) {
    const cuda::abi::Result code = api_.event_query_(event);
    if (code == cuda::abi::success) {
      complete = true;
      return true;
    }
    if (code != CUDA_ERROR_NOT_READY) {
      poison(operation, "cuEventQuery returned " + cuda_name(api_, code));
      return false;
    }
    if (!wait) {
      return true;
    }
    if (Clock::now() - submitted > options_.stall_timeout) {
      record_failure(result_, exit_failure, "execution", operation,
                     "CUDA event did not complete before the worker stall timeout",
                     std::nullopt, nullptr, std::nullopt, std::nullopt, current_policy_,
                     current_scenario_);
      poisoned_ = true;
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

bool CacheManager::map_chunk(const ChunkKey key, const std::size_t frame_index,
                             const bool speculative, const bool requires_h2d,
                             const std::uint64_t operation_id, const bool wait_for_completion) {
  Allocation* owner = allocation(key.allocation_id);
  ChunkRecord* record = chunk(key);
  if (owner == nullptr || record == nullptr || frame_index >= resources_.frames.size()) {
    poison("map_chunk", "invalid allocation, chunk, or frame", key);
    return false;
  }
  Frame& frame = resources_.frames[frame_index];
  if (frame.handle == 0 || frame.mapped || record->frame_index.has_value()) {
    ++metrics_.unsafe_remaps;
    no_aliases_ = false;
    poison("map_chunk", "physical frame is already mapped or chunk already owns a frame", key);
    return false;
  }
  for (const Frame& other : resources_.frames) {
    if (&other != &frame && other.handle != 0 && other.handle == frame.handle) {
      ++metrics_.unsafe_remaps;
      no_aliases_ = false;
      poison("map_chunk", "physical handle alias detected", key);
      return false;
    }
  }
  if (record->state != ChunkState::host_clean &&
      record->state != ChunkState::prefetch_queued) {
    poison("map_chunk", "chunk is not host-clean or prefetch-queued", key);
    return false;
  }
  record->speculative = speculative;
  if (!transition(*record, ChunkState::mapping, speculative ? "speculative_map" : "demand_map",
                  operation_id)) {
    return false;
  }

  const auto chunk_offset = checked_multiply(key.chunk_index, chunk_bytes_);
  const auto address_value = chunk_offset.has_value()
                                 ? checked_add(static_cast<std::uint64_t>(owner->logical_base),
                                               *chunk_offset)
                                 : std::nullopt;
  if (!address_value.has_value()) {
    poison("map_chunk", "stable logical CUDA address overflowed", key);
    return false;
  }
  const cuda::abi::DevicePointer address =
      static_cast<cuda::abi::DevicePointer>(*address_value);
  const auto remap_started = Clock::now();
  if (!require_cuda(result_, api_,
                    api_.mem_map_(address, static_cast<std::size_t>(chunk_bytes_), 0, frame.handle,
                                  0),
                    "execution", "cuMemMap", key)) {
    return false;
  }
  frame.key = key;
  frame.address = address;
  frame.mapped_bytes = chunk_bytes_;
  frame.mapped = true;
  ++frame.mapping_generations;
  if (frame.mapping_generations > 1) {
    ++metrics_.handle_reuses;
  }
  record->frame_index = static_cast<std::uint64_t>(frame_index);
  ++metrics_.mappings_completed;
  ++metrics_.active_mappings;
  metrics_.resident_peak_bytes = std::max(metrics_.resident_peak_bytes, resident_bytes());
  result_.cleanup.mappings_removed = false;
  CUmemAccessDesc access{};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = device_;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  if (!require_cuda(result_, api_,
                    api_.mem_set_access_(address, static_cast<std::size_t>(chunk_bytes_), &access,
                                         1),
                    "execution", "cuMemSetAccess", key)) {
    quarantine_ = true;
    return false;
  }
  unsigned long long access_flags = 0;
  const CUmemLocation location{CU_MEM_LOCATION_TYPE_DEVICE, device_};
  if (!require_cuda(result_, api_, api_.mem_get_access_(&access_flags, &location, address),
                    "execution", "cuMemGetAccess", key)) {
    quarantine_ = true;
    return false;
  }
  if ((access_flags & static_cast<unsigned long long>(CU_MEM_ACCESS_FLAGS_PROT_READWRITE)) == 0) {
    set_access_verified_ = false;
    poison("cuMemGetAccess", "mapped chunk is not read-write accessible", key);
    quarantine_ = true;
    return false;
  }
  remap_samples_.push_back(milliseconds_between(remap_started, Clock::now()));
  ++metrics_.set_access_completed;
  const cuda::abi::DevicePointer expected = owner->logical_base + key.chunk_index * chunk_bytes_;
  if (address != expected) {
    stable_addresses_ = false;
    poison("map_chunk", "chunk did not map at its stable logical CUDA address", key);
    return false;
  }

  TransferSlot* slot = free_h2d_slot();
  if (slot == nullptr) {
    for (TransferSlot& candidate : resources_.h2d_slots) {
      if (candidate.busy && !finish_h2d(candidate, true)) {
        return false;
      }
      if (!candidate.busy) {
        slot = &candidate;
        break;
      }
    }
  }
  if (slot == nullptr || begin_event_generation(*record) != EventGenerationResult::success) {
    poison("h2d_generation", "no H2D staging slot or event generation overflow", key);
    return false;
  }
  const std::uint64_t valid = valid_chunk_bytes(*owner, chunk_bytes_, key.chunk_index);
  if (requires_h2d) {
    const auto* source = static_cast<const std::byte*>(owner->backing.data()) +
                         key.chunk_index * chunk_bytes_;
    std::memcpy(slot->pinned, source, static_cast<std::size_t>(valid));
  }
  record->staging_slot = static_cast<std::uint32_t>(slot - resources_.h2d_slots.data());
  if (!transition(*record, ChunkState::h2d_in_flight,
                  requires_h2d ? "h2d_submit" : "full_write_only_h2d_skip", operation_id)) {
    return false;
  }
  if (!require_cuda(result_, api_, api_.event_record_(slot->started, resources_.h2d_stream),
                    "execution", "cuEventRecord(h2d_start)", key) ||
      (requires_h2d &&
       !require_cuda(result_, api_,
                     api_.memcpy_h2d_async_(address, slot->pinned,
                                            static_cast<std::size_t>(valid),
                                            resources_.h2d_stream),
                     "execution", "cuMemcpyHtoDAsync", key)) ||
      !require_cuda(result_, api_, api_.event_record_(slot->done, resources_.h2d_stream),
                    "execution", "cuEventRecord(h2d_done)", key)) {
    return false;
  }
  slot->key = key;
  slot->valid_bytes = valid;
  slot->generation = record->event_generation;
  slot->operation_id = operation_id;
  slot->submitted_at = Clock::now();
  slot->busy = true;
  slot->speculative = speculative;
  if (requires_h2d) {
    metrics_.h2d_bytes += valid;
  }
  const auto busy_staging = static_cast<std::uint32_t>(
      std::count_if(resources_.h2d_slots.begin(), resources_.h2d_slots.end(),
                    [](const TransferSlot& item) { return item.busy; }) +
      std::count_if(resources_.d2h_slots.begin(), resources_.d2h_slots.end(),
                    [](const TransferSlot& item) { return item.busy; }));
  metrics_.staging_slots_peak = std::max(metrics_.staging_slots_peak, busy_staging);
  return !wait_for_completion || finish_h2d(*slot, true);
}

bool CacheManager::finish_h2d(TransferSlot& slot, const bool wait) {
  if (!slot.busy || !slot.key.has_value()) {
    return true;
  }
  bool complete = false;
  if (!query_event(slot.done, slot.submitted_at, wait, "cuEventQuery(h2d_done)", complete) ||
      !complete) {
    return !poisoned_;
  }
  ChunkRecord* record = chunk(*slot.key);
  if (record == nullptr || record->state != ChunkState::h2d_in_flight ||
      complete_event_generation(*record, slot.generation) != EventGenerationResult::success) {
    ++unsafe_transitions_;
    poison("h2d_retire", "stale H2D event generation or invalid chunk state", slot.key);
    return false;
  }
  float elapsed = 0.0F;
  if (!require_cuda(result_, api_, api_.event_elapsed_time_(&elapsed, slot.started, slot.done),
                    "execution", "cuEventElapsedTime(h2d)", slot.key)) {
    return false;
  }
  h2d_samples_.push_back(static_cast<double>(elapsed));
  record->staging_slot.reset();
  if (!transition(*record, ChunkState::resident_clean, "h2d_retire", slot.operation_id)) {
    return false;
  }
  if (policy_ != nullptr) {
    policy_->insert(record->key, ++sequence_, slot.speculative, record->sequential_one_touch);
  }
  void* const pinned = slot.pinned;
  const cuda::abi::Event started = slot.started;
  const cuda::abi::Event done = slot.done;
  slot = {};
  slot.pinned = pinned;
  slot.started = started;
  slot.done = done;
  return true;
}

bool CacheManager::schedule_writeback(const ChunkKey key, const bool evict_after,
                                      const std::uint64_t operation_id) {
  emit_trace(chunk(key), "writeback_prepare", evict_after ? "eviction" : "drain",
             std::nullopt, operation_id);
  ChunkRecord* record = chunk(key);
  Allocation* owner = allocation(key.allocation_id);
  Frame* frame = frame_for(key);
  if (record == nullptr || owner == nullptr || frame == nullptr || !frame->mapped ||
      record->state != ChunkState::resident_dirty || !is_victim_eligible(*record)) {
    poison("schedule_writeback", "dirty chunk is not event-safe for write-back", key);
    return false;
  }
  TransferSlot* slot = free_d2h_slot();
  if (slot == nullptr) {
    for (TransferSlot& candidate : resources_.d2h_slots) {
      if (candidate.busy && !finish_d2h(candidate, true)) {
        return false;
      }
      if (!candidate.busy) {
        slot = &candidate;
        break;
      }
    }
  }
  if (slot == nullptr || begin_event_generation(*record) != EventGenerationResult::success) {
    poison("d2h_generation", "no D2H staging slot or event generation overflow", key);
    return false;
  }
  if (!transition(*record, ChunkState::writeback_queued, "dirty_writeback_queue", operation_id)) {
    return false;
  }
  record->staging_slot = static_cast<std::uint32_t>(slot - resources_.d2h_slots.data());
  if (!transition(*record, ChunkState::d2h_in_flight, "dirty_writeback_submit",
                  operation_id)) {
    return false;
  }
  const std::uint64_t valid = valid_chunk_bytes(*owner, chunk_bytes_, key.chunk_index);
  if (!require_cuda(result_, api_, api_.event_record_(slot->started, resources_.d2h_stream),
                    "execution", "cuEventRecord(d2h_start)", key) ||
      !require_cuda(result_, api_,
                    api_.memcpy_d2h_async_(slot->pinned, frame->address,
                                           static_cast<std::size_t>(valid),
                                           resources_.d2h_stream),
                    "execution", "cuMemcpyDtoHAsync", key) ||
      !require_cuda(result_, api_, api_.event_record_(slot->done, resources_.d2h_stream),
                    "execution", "cuEventRecord(d2h_done)", key)) {
    return false;
  }
  slot->key = key;
  slot->valid_bytes = valid;
  slot->generation = record->event_generation;
  slot->operation_id = operation_id;
  slot->submitted_at = Clock::now();
  slot->busy = true;
  slot->evict_after = evict_after;
  metrics_.d2h_bytes += valid;
  ++metrics_.dirty_evictions;
  const auto busy_staging = static_cast<std::uint32_t>(
      std::count_if(resources_.h2d_slots.begin(), resources_.h2d_slots.end(),
                    [](const TransferSlot& item) { return item.busy; }) +
      std::count_if(resources_.d2h_slots.begin(), resources_.d2h_slots.end(),
                    [](const TransferSlot& item) { return item.busy; }));
  metrics_.staging_slots_peak = std::max(metrics_.staging_slots_peak, busy_staging);
  return true;
}

bool CacheManager::finish_d2h(TransferSlot& slot, const bool wait) {
  if (!slot.busy || !slot.key.has_value()) {
    return true;
  }
  bool complete = false;
  if (!query_event(slot.done, slot.submitted_at, wait, "cuEventQuery(d2h_done)", complete) ||
      !complete) {
    return !poisoned_;
  }
  ChunkRecord* record = chunk(*slot.key);
  Allocation* owner = allocation(slot.key->allocation_id);
  if (record == nullptr || owner == nullptr || record->state != ChunkState::d2h_in_flight ||
      complete_event_generation(*record, slot.generation) != EventGenerationResult::success) {
    ++unsafe_transitions_;
    poison("d2h_retire", "stale D2H event generation or invalid chunk state", slot.key);
    return false;
  }
  float elapsed = 0.0F;
  if (!require_cuda(result_, api_, api_.event_elapsed_time_(&elapsed, slot.started, slot.done),
                    "execution", "cuEventElapsedTime(d2h)", slot.key)) {
    return false;
  }
  d2h_samples_.push_back(static_cast<double>(elapsed));
  const auto compare_started = Clock::now();
  auto* destination = static_cast<std::byte*>(owner->backing.data()) +
                      slot.key->chunk_index * chunk_bytes_;
  const auto* expected = reinterpret_cast<const std::uint32_t*>(destination);
  const auto* actual = static_cast<const std::uint32_t*>(slot.pinned);
  const std::uint64_t words = slot.valid_bytes / word_bytes;
  for (std::uint64_t index = 0; index < words; ++index) {
    if (expected[index] != actual[index]) {
      ++mismatch_count_;
      dirty_writeback_verified_ = false;
      if (!first_mismatch_offset_.has_value()) {
        first_mismatch_offset_ = slot.key->chunk_index * chunk_bytes_ + index * word_bytes;
      }
    }
  }
  if (mismatch_count_ != 0) {
    record_failure(result_, exit_corruption, "verification", "dirty_writeback",
                   "D2H write-back differs from the CPU reference", std::nullopt, nullptr,
                   slot.key, first_mismatch_offset_, current_policy_, current_scenario_);
    return false;
  }
  std::memcpy(destination, slot.pinned, static_cast<std::size_t>(slot.valid_bytes));
  writeback_samples_.push_back(milliseconds_between(compare_started, Clock::now()));
  ++metrics_.eviction_writebacks_completed;
  ++metrics_.writebacks_completed;
  record->staging_slot.reset();
  if (!transition(*record, ChunkState::resident_clean, "dirty_writeback_retire",
                  slot.operation_id)) {
    return false;
  }
  const ChunkKey key = *slot.key;
  const bool evict_after = slot.evict_after;
  const std::uint64_t operation_id = slot.operation_id;
  void* const pinned = slot.pinned;
  const cuda::abi::Event started = slot.started;
  const cuda::abi::Event done = slot.done;
  slot = {};
  slot.pinned = pinned;
  slot.started = started;
  slot.done = done;
  return !evict_after || unmap_clean(key, operation_id);
}

bool CacheManager::unmap_clean(const ChunkKey key, const std::uint64_t operation_id) {
  ChunkRecord* record = chunk(key);
  Allocation* owner = allocation(key.allocation_id);
  const auto frame_index = frame_index_for(key);
  if (record == nullptr || owner == nullptr || !frame_index.has_value()) {
    poison("unmap", "chunk has no mapped frame", key);
    return false;
  }
  Frame& frame = resources_.frames[*frame_index];
  if (record->state != ChunkState::resident_clean || !is_victim_eligible(*record) ||
      record->event_generation == 0) {
    ++metrics_.unsafe_remaps;
    poison("unmap", "chunk lacks a completed event boundary", key);
    return false;
  }
  if (!transition(*record, ChunkState::evicting, "evict", operation_id) ||
      !is_safe_to_unmap(*record)) {
    ++metrics_.unsafe_remaps;
    poison("unmap", "chunk failed the event-safe unmap invariant", key);
    return false;
  }
  const cuda::abi::Result code =
      api_.mem_unmap_(frame.address, static_cast<std::size_t>(frame.mapped_bytes));
  if (code != cuda::abi::success) {
    quarantine_ = true;
    owner->reservation_quarantined = true;
    ++metrics_.unsafe_remaps;
    const cuda::abi::Result release_code = api_.mem_release_(frame.handle);
    if (release_code == cuda::abi::success) {
      frame.handle = 0;
      ++metrics_.handles_released;
      --metrics_.live_handles;
    } else {
      append_diagnostic(result_, probe::DiagnosticLevel::error, "cuMemRelease(quarantine)",
                        cuda_message(api_, release_code),
                        static_cast<std::int64_t>(release_code));
    }
    record_failure(result_, exit_failure, "cleanup", "cuMemUnmap", cuda_message(api_, code),
                   code, &api_, key, std::nullopt, current_policy_, current_scenario_);
    return false;
  }
  ++metrics_.unmaps_completed;
  --metrics_.active_mappings;
  ++event_boundaries_;
  if (record->speculative) {
    ++metrics_.prefetch_wasted;
    if (metrics_.prefetch_in_flight != 0) {
      --metrics_.prefetch_in_flight;
    }
  }
  record->speculative = false;
  if (policy_ != nullptr) {
    (void)policy_->erase(key);
  }
  frame.key.reset();
  frame.address = 0;
  frame.mapped_bytes = 0;
  frame.mapped = false;
  record->frame_index.reset();
  if (!transition(*record, ChunkState::host_clean, "unmap_complete", operation_id)) {
    return false;
  }
  return true;
}

bool CacheManager::evict_chunk(const ChunkKey key, const bool wait,
                               const std::uint64_t operation_id) {
  ChunkRecord* record = chunk(key);
  if (record == nullptr || !is_victim_eligible(*record)) {
    poison("evict", "selected chunk is not event-safe for eviction", key);
    return false;
  }
  if (record->state == ChunkState::resident_dirty) {
    if (!schedule_writeback(key, true, operation_id)) {
      return false;
    }
    TransferSlot* slot = nullptr;
    for (TransferSlot& candidate : resources_.d2h_slots) {
      if (candidate.busy && candidate.key == key) {
        slot = &candidate;
        break;
      }
    }
    return !wait || (slot != nullptr && finish_d2h(*slot, true));
  }
  ++metrics_.clean_evictions;
  return unmap_clean(key, operation_id);
}

bool CacheManager::finish_compute(const bool wait) {
  ComputeSlot& compute = resources_.compute;
  if (!compute.busy || !compute.key.has_value()) {
    return true;
  }
  bool complete = false;
  if (!query_event(compute.done, compute.submitted_at, wait, "cuEventQuery(kernel_done)",
                   complete) ||
      !complete) {
    return !poisoned_;
  }
  ChunkRecord* record = chunk(*compute.key);
  if (record == nullptr || record->pin_count == 0 ||
      complete_event_generation(*record, compute.generation) != EventGenerationResult::success) {
    ++unsafe_transitions_;
    poison("kernel_retire", "stale kernel event generation or invalid pin ownership",
           compute.key);
    return false;
  }
  float elapsed = 0.0F;
  if (!require_cuda(result_, api_,
                    api_.event_elapsed_time_(&elapsed, compute.started, compute.done),
                    "execution", "cuEventElapsedTime(kernel)", compute.key)) {
    return false;
  }
  kernel_samples_.push_back(static_cast<double>(elapsed));
  --record->pin_count;
  record->in_current_working_set = false;
  if (policy_ != nullptr) {
    policy_->touch(record->key, ++sequence_, record->sequential_one_touch);
  }
  const ChunkKey key = *compute.key;
  compute.key.reset();
  compute.busy = false;
  emit_trace(record, "kernel_retire", "compute_complete", std::nullopt,
             compute.operation_id, std::nullopt, compute.generation, false);
  if (static_cast<double>(elapsed) >
      static_cast<double>(maximum_kernel_duration.count())) {
    record_failure(result_, exit_failure, "execution", "kernel_duration",
                   "kernel tile exceeded the 250 ms safety limit", std::nullopt, nullptr, key,
                   std::nullopt, current_policy_, current_scenario_);
    return false;
  }
  return true;
}

bool CacheManager::launch_kernel(const ChunkAccessPlan& plan, const std::uint64_t operation_id,
                                 const std::uint32_t operation_index) {
  if (!finish_compute(true)) {
    return false;
  }
  ChunkRecord* record = chunk(plan.key);
  Frame* frame = frame_for(plan.key);
  if (record == nullptr || frame == nullptr || !frame->mapped ||
      (record->state != ChunkState::resident_clean &&
       record->state != ChunkState::resident_dirty) ||
      record->pin_count != 0 || begin_event_generation(*record) != EventGenerationResult::success) {
    poison("launch_kernel", "chunk is not resident or event-safe for compute", plan.key);
    return false;
  }
  ++record->pin_count;
  record->in_current_working_set = true;
  if (!require_cuda(result_, api_,
                    api_.event_record_(resources_.compute.started, resources_.compute_stream),
                    "execution", "cuEventRecord(kernel_start)", plan.key)) {
    return false;
  }
  for (const ChunkAccessSpan& span : plan.spans) {
    cuda::abi::DevicePointer data = frame->address + span.offset_bytes;
    std::uint64_t word_count = span.length_bytes / word_bytes;
    std::uint64_t global_start =
        (plan.allocation_offset_bytes + span.offset_bytes) / word_bytes;
    std::uint32_t mutable_operation_index = operation_index;
    std::uint64_t seed = options_.seed;
    std::uint32_t access_mode = static_cast<std::uint32_t>(span.mode);
    cuda::abi::DevicePointer tokens = resources_.token_device;
    void* parameters[] = {&data, &word_count, &global_start, &mutable_operation_index,
                          &seed, &access_mode, &tokens};
    if (!require_cuda(result_, api_,
                      api_.launch_kernel_(resources_.kernel, blocks_, 1, 1, threads_, 1, 1, 0,
                                          resources_.compute_stream, parameters, nullptr),
                      "execution", "cuLaunchKernel", plan.key)) {
      return false;
    }
  }
  if (!require_cuda(result_, api_,
                    api_.event_record_(resources_.compute.done, resources_.compute_stream),
                    "execution", "cuEventRecord(kernel_done)", plan.key)) {
    return false;
  }
  if (plan.marks_dirty && record->state == ChunkState::resident_clean &&
      !transition(*record, ChunkState::resident_dirty, "write_kernel_submitted", operation_id)) {
    return false;
  }
  resources_.compute.key = plan.key;
  resources_.compute.generation = record->event_generation;
  resources_.compute.operation_id = operation_id;
  resources_.compute.mode = plan.spans.front().mode;
  resources_.compute.submitted_at = Clock::now();
  resources_.compute.busy = true;
  emit_trace(record, "kernel_submit", "transaction", std::nullopt, operation_id,
             plan.valid_bytes, record->event_generation, false);
  return true;
}

bool CacheManager::prepare_tokens(const std::uint64_t count) {
  if (!drain(false) || count == 0 || count > std::numeric_limits<std::uint64_t>::max() / 4ULL) {
    record_failure(result_, exit_prerequisite, "configuration", "prepare_tokens",
                   "verification-token count is invalid");
    return false;
  }
  const std::uint64_t bytes = count * 4ULL;
  if (bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    record_failure(result_, exit_prerequisite, "configuration", "prepare_tokens",
                   "verification-token buffer exceeds the platform size limit");
    return false;
  }
  if (resources_.token_device != 0) {
    if (!require_cuda(result_, api_, api_.mem_free_(resources_.token_device), "setup",
                      "cuMemFree(tokens)")) {
      return false;
    }
    resources_.token_device = 0;
    resources_.token_bytes = 0;
  }
  if (!require_cuda(result_, api_,
                    api_.mem_alloc_(&resources_.token_device, static_cast<std::size_t>(bytes)),
                    "setup", "cuMemAlloc(tokens)")) {
    return false;
  }
  resources_.token_bytes = bytes;
  TransferSlot& staging = resources_.h2d_slots.front();
  std::uint64_t copied = 0;
  while (copied < bytes) {
    const std::uint64_t batch = std::min(chunk_bytes_, bytes - copied);
    std::memset(staging.pinned, 0, static_cast<std::size_t>(batch));
    if (!require_cuda(result_, api_,
                      api_.memcpy_h2d_async_(resources_.token_device + copied, staging.pinned,
                                             static_cast<std::size_t>(batch),
                                             resources_.h2d_stream),
                      "setup", "cuMemcpyHtoDAsync(tokens)")) {
      return false;
    }
    copied += batch;
  }
  return require_cuda(result_, api_, api_.stream_synchronize_(resources_.h2d_stream), "setup",
                      "cuStreamSynchronize(tokens)");
}

bool CacheManager::collect_tokens(const std::span<std::uint32_t> output) {
  const std::uint64_t bytes = static_cast<std::uint64_t>(output.size()) * 4ULL;
  if (!drain(false) || resources_.token_device == 0 || bytes > resources_.token_bytes) {
    poison("collect_tokens", "verification-token buffer is unavailable or undersized");
    return false;
  }
  TransferSlot& staging = resources_.d2h_slots.front();
  auto* destination = reinterpret_cast<std::byte*>(output.data());
  std::uint64_t copied = 0;
  while (copied < bytes) {
    const std::uint64_t batch = std::min(chunk_bytes_, bytes - copied);
    if (!require_cuda(result_, api_,
                      api_.memcpy_d2h_async_(staging.pinned, resources_.token_device + copied,
                                             static_cast<std::size_t>(batch),
                                             resources_.d2h_stream),
                      "verification", "cuMemcpyDtoHAsync(tokens)") ||
        !require_cuda(result_, api_, api_.stream_synchronize_(resources_.d2h_stream),
                      "verification", "cuStreamSynchronize(tokens)")) {
      return false;
    }
    std::memcpy(destination + copied, staging.pinned, static_cast<std::size_t>(batch));
    copied += batch;
  }
  return true;
}

bool CacheManager::promote_prefetch(const ChunkKey key, const std::uint64_t operation_id) {
  ChunkRecord* record = chunk(key);
  if (record == nullptr) {
    return false;
  }
  if (record->state == ChunkState::prefetch_queued) {
    const auto found = std::find(pending_prefetch_.begin(), pending_prefetch_.end(), key);
    if (found != pending_prefetch_.end()) {
      pending_prefetch_.erase(found);
    }
    ++metrics_.prefetch_promoted;
    emit_trace(record, "prefetch_promote", "demand", std::nullopt, operation_id);
    return true;
  }
  if (record->state == ChunkState::h2d_in_flight) {
    for (TransferSlot& slot : resources_.h2d_slots) {
      if (slot.busy && slot.key == key) {
        ++metrics_.prefetch_promoted;
        if (!finish_h2d(slot, true)) {
          return false;
        }
        record->speculative = false;
        ++metrics_.prefetch_useful;
        if (metrics_.prefetch_in_flight != 0) {
          --metrics_.prefetch_in_flight;
        }
        emit_trace(record, "prefetch_useful", "demand", std::nullopt, operation_id);
        return true;
      }
    }
    poison("prefetch_promote", "in-flight prefetch has no H2D staging owner", key);
    return false;
  }
  if ((record->state == ChunkState::resident_clean ||
       record->state == ChunkState::resident_dirty) &&
      record->speculative) {
    record->speculative = false;
    ++metrics_.prefetch_useful;
    if (metrics_.prefetch_in_flight != 0) {
      --metrics_.prefetch_in_flight;
    }
    emit_trace(record, "prefetch_useful", "demand", std::nullopt, operation_id);
  }
  return true;
}

bool CacheManager::ensure_resident(const ChunkAccessPlan& plan,
                                   const std::uint64_t operation_id, bool& hit) {
  ChunkRecord* record = chunk(plan.key);
  if (record == nullptr) {
    poison("ensure_resident", "transaction references an unknown chunk", plan.key);
    return false;
  }
  const bool was_prefetch = record->state == ChunkState::prefetch_queued ||
                            record->state == ChunkState::h2d_in_flight || record->speculative;
  if (!promote_prefetch(plan.key, operation_id)) {
    return false;
  }
  record = chunk(plan.key);
  if (record->state == ChunkState::resident_clean ||
      record->state == ChunkState::resident_dirty) {
    hit = true;
    ++metrics_.cache_hits;
    ++metrics_.demand_accesses;
    record->sequential_one_touch = plan.spans.size() == 1U;
    if (policy_ != nullptr) {
      policy_->touch(plan.key, ++sequence_, record->sequential_one_touch);
    }
    return true;
  }
  if (record->state != ChunkState::host_clean &&
      record->state != ChunkState::prefetch_queued) {
    poison("ensure_resident", "chunk did not reach a demand-mappable state", plan.key);
    return false;
  }
  hit = false;
  ++metrics_.cache_misses;
  ++metrics_.demand_accesses;
  const auto frame_index = acquire_frame(plan.key, operation_id);
  if (!frame_index.has_value() ||
      !map_chunk(plan.key, *frame_index, false, plan.requires_h2d, operation_id, true)) {
    return false;
  }
  record = chunk(plan.key);
  record->sequential_one_touch = plan.spans.size() == 1U;
  if (was_prefetch) {
    record->speculative = false;
    ++metrics_.prefetch_useful;
    if (metrics_.prefetch_in_flight != 0) {
      --metrics_.prefetch_in_flight;
    }
  }
  return true;
}

bool CacheManager::apply_cpu_reference(const ChunkAccessPlan& plan,
                                       const std::uint32_t operation_index,
                                       std::uint64_t& token) {
  Allocation* owner = allocation(plan.key.allocation_id);
  if (owner == nullptr) {
    return false;
  }
  auto* chunk_words = reinterpret_cast<std::uint32_t*>(
      static_cast<std::byte*>(owner->backing.data()) + plan.allocation_offset_bytes);
  for (const ChunkAccessSpan& span : plan.spans) {
    const std::uint64_t span_word_offset = span.offset_bytes / word_bytes;
    const std::uint64_t span_word_count = span.length_bytes / word_bytes;
    auto values = std::span<std::uint32_t>(chunk_words + span_word_offset,
                                          static_cast<std::size_t>(span_word_count));
    const std::uint64_t global_start =
        (plan.allocation_offset_bytes + span.offset_bytes) / word_bytes;
    const auto expected = expected_verification_token(values, global_start, operation_index,
                                                      options_.seed, span.mode);
    if (!expected.has_value()) {
      poison("cpu_reference", "verification token range overflowed", plan.key);
      return false;
    }
    token ^= *expected;
    if (span.mode != AccessMode::read) {
      for (std::uint64_t index = 0; index < span_word_count; ++index) {
        values[static_cast<std::size_t>(index)] =
            transform_word(values[static_cast<std::size_t>(index)], global_start + index,
                           operation_index, options_.seed, span.mode);
      }
    }
  }
  return true;
}

bool CacheManager::enqueue_transaction(const ScenarioOperation& operation,
                                       const std::span<std::uint32_t> expected_tokens,
                                       std::uint64_t& expected_token) {
  if (operation.kind != ScenarioOperationKind::access || !operation.access.has_value() ||
      operation.sequence > std::numeric_limits<std::uint32_t>::max() ||
      operation.sequence >= expected_tokens.size()) {
    record_failure(result_, exit_prerequisite, "configuration", "enqueue_transaction",
                   "access operation or verification-token index is invalid");
    return false;
  }
  std::vector<AllocationLayout> layouts;
  layouts.reserve(allocations_.size());
  for (const auto& [id, owner] : allocations_) {
    (void)id;
    layouts.push_back(AllocationLayout{owner->id, owner->logical_bytes});
  }
  const std::array ranges{*operation.access};
  const AccessPlanResult plan = normalize_and_split_accesses(layouts, ranges, chunk_bytes_);
  if (!plan) {
    record_failure(result_, exit_prerequisite, "configuration", "normalize_accesses",
                   "transaction range is invalid: " + std::string(access_plan_error_name(plan.error)));
    return false;
  }
  std::uint32_t token = 0;
  for (const ChunkAccessPlan& chunk_plan : plan.chunks) {
    bool hit = false;
    if (!ensure_resident(chunk_plan, operation.sequence, hit) ||
        !launch_kernel(chunk_plan, operation.sequence,
                       static_cast<std::uint32_t>(operation.sequence)) ||
        !apply_cpu_reference(chunk_plan, static_cast<std::uint32_t>(operation.sequence),
                             expected_token) ||
        !finish_compute(true)) {
      return false;
    }
    token = static_cast<std::uint32_t>(expected_token);
  }
  expected_tokens[static_cast<std::size_t>(operation.sequence)] = token;
  return true;
}

bool CacheManager::schedule_pending_prefetches() {
  while (!pending_prefetch_.empty() && free_h2d_slot() != nullptr) {
    const ChunkKey key = pending_prefetch_.front();
    ChunkRecord* record = chunk(key);
    if (record == nullptr || record->state != ChunkState::prefetch_queued) {
      pending_prefetch_.pop_front();
      continue;
    }
    const bool have_free_frame = std::any_of(
        resources_.frames.begin(), resources_.frames.end(),
        [](const Frame& frame) { return frame.handle != 0 && !frame.mapped; });
    const bool can_create = metrics_.live_handles < target_state_.current_target_bytes / chunk_bytes_;
    const bool have_victim = std::any_of(
        allocations_.begin(), allocations_.end(), [](const auto& pair) {
          return std::any_of(pair.second->chunks.begin(), pair.second->chunks.end(),
                             [](const ChunkRecord& candidate) {
                               return is_victim_eligible(candidate);
                             });
        });
    if (!have_free_frame && !can_create && !have_victim) {
      break;
    }
    const auto frame_index = acquire_frame(key, 0);
    if (!frame_index.has_value()) {
      return false;
    }
    pending_prefetch_.pop_front();
    if (!map_chunk(key, *frame_index, true, true, 0, false)) {
      return false;
    }
  }
  return true;
}

bool CacheManager::prefetch(const std::span<const ChunkKey> keys,
                            const std::uint64_t operation_id) {
  std::unordered_set<ChunkKey, ChunkKeyHash> desired;
  desired.reserve(keys.size());
  for (const ChunkKey key : keys) {
    desired.insert(key);
  }
  for (auto iterator = pending_prefetch_.begin(); iterator != pending_prefetch_.end();) {
    if (desired.contains(*iterator)) {
      ++iterator;
      continue;
    }
    if (ChunkRecord* record = chunk(*iterator);
        record != nullptr && record->state == ChunkState::prefetch_queued) {
      record->speculative = false;
      if (!transition(*record, ChunkState::host_clean, "stale_prefetch_cancel", operation_id)) {
        return false;
      }
      ++metrics_.prefetch_cancelled;
      if (metrics_.prefetch_in_flight != 0) {
        --metrics_.prefetch_in_flight;
      }
    }
    iterator = pending_prefetch_.erase(iterator);
  }
  const std::size_t limit = resources_.h2d_slots.size();
  for (const ChunkKey key : keys) {
    const std::size_t active = pending_prefetch_.size() +
                               static_cast<std::size_t>(std::count_if(
                                   resources_.h2d_slots.begin(), resources_.h2d_slots.end(),
                                   [](const TransferSlot& slot) {
                                     return slot.busy && slot.speculative;
                                   }));
    if (active >= limit) {
      break;
    }
    ChunkRecord* record = chunk(key);
    if (record == nullptr || record->state != ChunkState::host_clean) {
      continue;
    }
    record->speculative = true;
    if (!transition(*record, ChunkState::prefetch_queued, "prefetch_queue", operation_id)) {
      return false;
    }
    pending_prefetch_.push_back(key);
    ++metrics_.prefetch_issued;
    ++metrics_.prefetch_in_flight;
  }
  return schedule_pending_prefetches();
}

bool CacheManager::poll() {
  if (!finish_compute(false)) {
    return false;
  }
  for (TransferSlot& slot : resources_.h2d_slots) {
    if (!finish_h2d(slot, false)) {
      return false;
    }
  }
  for (TransferSlot& slot : resources_.d2h_slots) {
    if (!finish_d2h(slot, false)) {
      return false;
    }
  }
  if (!observe_budget(false) || !schedule_pending_prefetches()) {
    return false;
  }
  const std::uint64_t target = target_state_.current_target_bytes;
  const std::uint64_t resident = resident_bytes();
  if (target != 0) {
    resident_ratio_sum_ += static_cast<long double>(resident) / static_cast<long double>(target);
    ++resident_samples_;
  }
  return true;
}

bool CacheManager::observe_budget(const bool force) {
  const Clock::time_point now = Clock::now();
  if (!force && last_budget_poll_ != Clock::time_point{} &&
      now - last_budget_poll_ < options_.budget_poll_interval) {
    return true;
  }
  last_budget_poll_ = now;
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (!require_cuda(result_, api_, api_.mem_get_info_(&free_bytes, &total_bytes), "budget",
                    "cuMemGetInfo")) {
    return false;
  }
  (void)total_bytes;
  const std::uint64_t free_value = static_cast<std::uint64_t>(free_bytes);
  cuda_free_end_ = free_value;
  cuda_free_minimum_ =
      std::min(cuda_free_minimum_.value_or(free_value), free_value);

  std::optional<BudgetSnapshot> external;
  if (budget_provider_) {
    external = budget_provider_();
#ifdef _WIN32
    if (!external.has_value()) {
      record_failure(result_, exit_failure, "budget", "QueryVideoMemoryInfo",
                     "live WDDM budget observation failed", std::nullopt, nullptr,
                     std::nullopt, std::nullopt, current_policy_, current_scenario_);
      return false;
    }
#endif
  }
  std::optional<WddmBudgetObservation> wddm;
  if (external.has_value()) {
    wddm = WddmBudgetObservation{external->budget_bytes, external->usage_bytes};
    const std::uint64_t available = external->available_bytes();
    wddm_available_end_ = available;
    wddm_available_minimum_ =
        std::min(wddm_available_minimum_.value_or(available), available);
  }
  ++budget_samples_;
  const BudgetTarget target = calculate_budget_target(BudgetTargetInput{
      configured_cap_bytes_, free_value, live_handle_bytes(), wddm,
      options_.device_headroom_bytes, chunk_bytes_, chunk_bytes_});
  if (!target) {
    const int code = target.error == BudgetTargetError::below_minimum ? exit_oom
                                                                      : exit_failure;
    record_failure(result_, code, "budget", "calculate_target",
                   "live cache target is unsafe: " +
                       std::string(budget_target_error_name(target.error)),
                   std::nullopt, nullptr, std::nullopt, std::nullopt, current_policy_,
                   current_scenario_);
    return false;
  }
  TargetHysteresisConfig hysteresis;
  hysteresis.chunk_bytes = chunk_bytes_;
  const TargetDecision decision =
      observe_safe_target(target_state_, target.target_bytes, target.required_minimum_bytes,
                          hysteresis);
  if (decision.action == TargetAction::budget_pressure ||
      decision.action == TargetAction::invalid_configuration) {
    record_failure(result_, decision.action == TargetAction::budget_pressure ? exit_oom
                                                                              : exit_failure,
                   "budget", "target_hysteresis",
                   "dynamic budget fell below the next declared working set", std::nullopt,
                   nullptr, std::nullopt, std::nullopt, current_policy_, current_scenario_);
    return false;
  }
  if (decision.action == TargetAction::shrink) {
    ++target_shrinks_;
    if (!shrink_to(decision.effective_target_bytes)) {
      return false;
    }
  } else if (decision.action == TargetAction::grow) {
    ++target_grows_;
  }
  target_minimum_bytes_ = std::min(target_minimum_bytes_, target_state_.current_target_bytes);
  target_maximum_bytes_ = std::max(target_maximum_bytes_, target_state_.current_target_bytes);
  metrics_.target_peak_bytes = std::max(metrics_.target_peak_bytes,
                                        target_state_.current_target_bytes);
  emit_trace(nullptr, "budget_sample", std::string(target_action_name(decision.action)),
             std::nullopt, std::nullopt, target_state_.current_target_bytes);
  return true;
}

bool CacheManager::release_excess_handles() {
  const std::uint64_t limit = target_state_.current_target_bytes / chunk_bytes_;
  for (auto iterator = resources_.frames.rbegin();
       metrics_.live_handles > limit && iterator != resources_.frames.rend(); ++iterator) {
    Frame& frame = *iterator;
    if (frame.handle == 0 || frame.mapped) {
      continue;
    }
    const cuda::abi::Result code = api_.mem_release_(frame.handle);
    if (!require_cuda(result_, api_, code, "budget", "cuMemRelease(shrink)")) {
      return false;
    }
    frame.handle = 0;
    ++metrics_.handles_released;
    --metrics_.live_handles;
  }
  if (metrics_.live_handles > limit) {
    record_failure(result_, exit_oom, "budget", "release_frames",
                   "unable to shrink physical frame pool to the live target", std::nullopt,
                   nullptr, std::nullopt, std::nullopt, current_policy_, current_scenario_);
    return false;
  }
  return true;
}

bool CacheManager::shrink_to(const std::uint64_t bytes) {
  if (!finish_compute(true)) {
    return false;
  }
  for (TransferSlot& slot : resources_.h2d_slots) {
    if (!finish_h2d(slot, true)) {
      return false;
    }
  }
  for (TransferSlot& slot : resources_.d2h_slots) {
    if (!finish_d2h(slot, true)) {
      return false;
    }
  }
  while (resident_bytes() > bytes) {
    const std::optional<ChunkKey> victim = select_victim();
    if (!victim.has_value() || !evict_chunk(*victim, true, 0)) {
      record_failure(result_, exit_oom, "budget", "shrink",
                     "cache target shrink has no event-safe victim", std::nullopt, nullptr,
                     victim, std::nullopt, current_policy_, current_scenario_);
      return false;
    }
  }
  return release_excess_handles();
}

bool CacheManager::acquire_pressure(const std::uint64_t bytes) {
  if (bytes == 0 || pressure_allocation_ != 0 ||
      bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    record_failure(result_, exit_prerequisite, "configuration", "pressure_acquire",
                   "controlled pressure size is invalid");
    return false;
  }
  if (!require_cuda(result_, api_,
                    api_.mem_alloc_(&pressure_allocation_, static_cast<std::size_t>(bytes)),
                    "budget", "cuMemAlloc(pressure)")) {
    return false;
  }
  pressure_bytes_ = bytes;
  emit_trace(nullptr, "pressure_acquire", "scenario", std::nullopt, std::nullopt, bytes);
  return observe_budget(true);
}

bool CacheManager::release_pressure() {
  if (pressure_allocation_ == 0) {
    return true;
  }
  if (!require_cuda(result_, api_, api_.mem_free_(pressure_allocation_), "budget",
                    "cuMemFree(pressure)")) {
    return false;
  }
  pressure_allocation_ = 0;
  const std::uint64_t released = pressure_bytes_;
  pressure_bytes_ = 0;
  emit_trace(nullptr, "pressure_release", "scenario", std::nullopt, std::nullopt, released);
  for (std::uint32_t sample = 0; sample < 10U; ++sample) {
    if (!observe_budget(true)) {
      return false;
    }
  }
  return true;
}

bool CacheManager::drain(const bool release_mappings) {
  if (!finish_compute(true)) {
    return false;
  }
  for (TransferSlot& slot : resources_.h2d_slots) {
    if (!finish_h2d(slot, true)) {
      return false;
    }
  }
  for (TransferSlot& slot : resources_.d2h_slots) {
    if (!finish_d2h(slot, true)) {
      return false;
    }
  }
  while (!pending_prefetch_.empty()) {
    const ChunkKey key = pending_prefetch_.front();
    pending_prefetch_.pop_front();
    if (ChunkRecord* record = chunk(key);
        record != nullptr && record->state == ChunkState::prefetch_queued) {
      record->speculative = false;
      if (!transition(*record, ChunkState::host_clean, "drain_prefetch_cancel")) {
        return false;
      }
      ++metrics_.prefetch_cancelled;
      if (metrics_.prefetch_in_flight != 0) {
        --metrics_.prefetch_in_flight;
      }
    }
  }
  if (!release_mappings) {
    return true;
  }
  std::vector<ChunkKey> mapped;
  mapped.reserve(static_cast<std::size_t>(metrics_.active_mappings));
  for (const auto& [id, owner] : allocations_) {
    (void)id;
    for (const ChunkRecord& record : owner->chunks) {
      if (record.frame_index.has_value()) {
        mapped.push_back(record.key);
      }
    }
  }
  std::sort(mapped.begin(), mapped.end(), key_less);
  for (const ChunkKey key : mapped) {
    ChunkRecord* record = chunk(key);
    if (record == nullptr) {
      return false;
    }
    if (record->state == ChunkState::resident_dirty) {
      if (!evict_chunk(key, true, 0)) {
        return false;
      }
    } else if (record->state == ChunkState::resident_clean) {
      ++metrics_.clean_evictions;
      if (!unmap_clean(key, 0)) {
        return false;
      }
    } else {
      poison("drain", "mapped chunk remained in an in-flight state", key);
      return false;
    }
  }
  return true;
}

bool CacheManager::verify_and_flush() {
  return drain(true) && mismatch_count_ == 0;
}

bool CacheManager::release(const AllocationId id) {
  Allocation* owner = allocation(id);
  if (owner == nullptr || !drain(true)) {
    return false;
  }
  bool success = true;
  if (owner->reservation != 0) {
    if (owner->reservation_quarantined || quarantine_) {
      success = false;
      append_diagnostic(result_, probe::DiagnosticLevel::error, "cuMemAddressFree",
                        "reservation retained by the worker quarantine boundary");
    } else {
      const cuda::abi::Result code = api_.mem_address_free_(
          owner->reservation, static_cast<std::size_t>(owner->reservation_bytes));
      if (code != cuda::abi::success) {
        success = false;
        append_diagnostic(result_, probe::DiagnosticLevel::error, "cuMemAddressFree",
                          cuda_message(api_, code), static_cast<std::int64_t>(code));
      } else {
        owner->reservation = 0;
      }
    }
  }
  if (!owner->backing.release()) {
    success = false;
    append_diagnostic(result_, probe::DiagnosticLevel::error, "release_pageable_backing",
                      "pageable backing release failed");
  }
  allocations_.erase(id.value);
  if (allocations_.empty()) {
    result_.cleanup.host_backing_released = success;
    result_.cleanup.virtual_reservations_released = success && !quarantine_;
  }
  return success;
}

void CacheManager::close() noexcept {
  if (closed_) {
    return;
  }
  closed_ = true;
  bool transactions_drained = false;
  try {
    transactions_drained = drain(true);
  } catch (...) {
    append_diagnostic(result_, probe::DiagnosticLevel::error, "drain",
                      "exception during cache drain");
  }
  result_.cleanup.transactions_drained = transactions_drained;
  result_.cleanup.prefetch_drained = transactions_drained && pending_prefetch_.empty();
  result_.cleanup.writebacks_completed =
      transactions_drained &&
      std::none_of(resources_.d2h_slots.begin(), resources_.d2h_slots.end(),
                   [](const TransferSlot& slot) { return slot.busy; });
  result_.cleanup.events_drained = transactions_drained;

  if (pressure_allocation_ != 0) {
    const cuda::abi::Result code = api_.mem_free_(pressure_allocation_);
    if (code != cuda::abi::success) {
      append_diagnostic(result_, probe::DiagnosticLevel::error, "cuMemFree(pressure)",
                        cuda_message(api_, code), static_cast<std::int64_t>(code));
      quarantine_ = true;
    }
    pressure_allocation_ = 0;
  }
  if (resources_.token_device != 0) {
    const cuda::abi::Result code = api_.mem_free_(resources_.token_device);
    if (code != cuda::abi::success) {
      append_diagnostic(result_, probe::DiagnosticLevel::error, "cuMemFree(tokens)",
                        cuda_message(api_, code), static_cast<std::int64_t>(code));
      quarantine_ = true;
    }
    resources_.token_device = 0;
  }

  bool mappings_removed = metrics_.active_mappings == 0;
  bool handles_released = true;
  for (Frame& frame : resources_.frames) {
    if (frame.handle == 0) {
      continue;
    }
    const cuda::abi::Result code = api_.mem_release_(frame.handle);
    if (code != cuda::abi::success) {
      handles_released = false;
      append_diagnostic(result_, probe::DiagnosticLevel::error, "cuMemRelease",
                        cuda_message(api_, code), static_cast<std::int64_t>(code));
    } else {
      ++metrics_.handles_released;
      if (metrics_.live_handles != 0) {
        --metrics_.live_handles;
      }
      frame.handle = 0;
    }
    if (frame.mapped) {
      mappings_removed = false;
      quarantine_ = true;
    }
  }
  result_.cleanup.mappings_removed = mappings_removed;
  result_.cleanup.physical_handles_released = handles_released;

  bool reservations_released = mappings_removed && !quarantine_;
  bool backing_released = true;
  for (auto& [id, owner] : allocations_) {
    (void)id;
    if (owner->reservation != 0) {
      if (owner->reservation_quarantined || quarantine_ || !mappings_removed) {
        reservations_released = false;
      } else {
        const cuda::abi::Result code = api_.mem_address_free_(
            owner->reservation, static_cast<std::size_t>(owner->reservation_bytes));
        if (code != cuda::abi::success) {
          reservations_released = false;
          append_diagnostic(result_, probe::DiagnosticLevel::error, "cuMemAddressFree",
                            cuda_message(api_, code), static_cast<std::int64_t>(code));
        } else {
          owner->reservation = 0;
        }
      }
    }
    if (!owner->backing.release()) {
      backing_released = false;
    }
  }
  allocations_.clear();
  result_.cleanup.virtual_reservations_released = reservations_released;
  result_.cleanup.host_backing_released = backing_released;

  bool streams_drained = true;
  for (const cuda::abi::Stream stream :
       {resources_.h2d_stream, resources_.compute_stream, resources_.d2h_stream}) {
    if (stream == nullptr) {
      continue;
    }
    const cuda::abi::Result code = api_.stream_synchronize_(stream);
    if (code != cuda::abi::success) {
      streams_drained = false;
      append_diagnostic(result_, probe::DiagnosticLevel::error, "cuStreamSynchronize(cleanup)",
                        cuda_message(api_, code), static_cast<std::int64_t>(code));
    }
  }
  if (!streams_drained) {
    quarantine_ = true;
    result_.cleanup.events_drained = false;
  }

  bool events_destroyed = true;
  const auto destroy_event = [&](cuda::abi::Event& event) {
    if (event == nullptr) {
      return;
    }
    const cuda::abi::Result code = api_.event_destroy_(event);
    if (code != cuda::abi::success) {
      events_destroyed = false;
      append_diagnostic(result_, probe::DiagnosticLevel::error, "cuEventDestroy",
                        cuda_message(api_, code), static_cast<std::int64_t>(code));
    }
    event = nullptr;
  };
  for (TransferSlot& slot : resources_.h2d_slots) {
    destroy_event(slot.started);
    destroy_event(slot.done);
  }
  for (TransferSlot& slot : resources_.d2h_slots) {
    destroy_event(slot.started);
    destroy_event(slot.done);
  }
  destroy_event(resources_.compute.started);
  destroy_event(resources_.compute.done);
  result_.cleanup.events_destroyed = events_destroyed;

  bool streams_destroyed = true;
  const auto destroy_stream = [&](cuda::abi::Stream& stream) {
    if (stream == nullptr) {
      return;
    }
    const cuda::abi::Result code = api_.stream_destroy_(stream);
    if (code != cuda::abi::success) {
      streams_destroyed = false;
      append_diagnostic(result_, probe::DiagnosticLevel::error, "cuStreamDestroy",
                        cuda_message(api_, code), static_cast<std::int64_t>(code));
    }
    stream = nullptr;
  };
  destroy_stream(resources_.h2d_stream);
  destroy_stream(resources_.compute_stream);
  destroy_stream(resources_.d2h_stream);
  result_.cleanup.streams_destroyed = streams_destroyed;

  bool module_unloaded = true;
  if (resources_.module != nullptr) {
    const cuda::abi::Result code = api_.module_unload_(resources_.module);
    if (code != cuda::abi::success) {
      module_unloaded = false;
      append_diagnostic(result_, probe::DiagnosticLevel::error, "cuModuleUnload",
                        cuda_message(api_, code), static_cast<std::int64_t>(code));
    }
    resources_.module = nullptr;
  }
  result_.cleanup.module_unloaded = module_unloaded;

  bool pinned_released = streams_drained;
  if (streams_drained) {
    for (TransferSlot* slot : [&]() {
           std::vector<TransferSlot*> slots;
           slots.reserve(resources_.h2d_slots.size() + resources_.d2h_slots.size());
           for (TransferSlot& item : resources_.h2d_slots) {
             slots.push_back(&item);
           }
           for (TransferSlot& item : resources_.d2h_slots) {
             slots.push_back(&item);
           }
           return slots;
         }()) {
      if (slot->pinned == nullptr) {
        continue;
      }
      const cuda::abi::Result code = api_.mem_free_host_(slot->pinned);
      if (code != cuda::abi::success) {
        pinned_released = false;
        append_diagnostic(result_, probe::DiagnosticLevel::error, "cuMemFreeHost",
                          cuda_message(api_, code), static_cast<std::int64_t>(code));
      }
      slot->pinned = nullptr;
    }
  } else {
    append_diagnostic(result_, probe::DiagnosticLevel::warning, "cuMemFreeHost",
                      "pinned staging retained until worker exit after failed DMA drain");
  }
  result_.cleanup.pinned_staging_released = pinned_released;
}

[[nodiscard]] std::uint64_t counter_delta(const std::uint64_t after,
                                          const std::uint64_t before) noexcept {
  return after >= before ? after - before : 0;
}

[[nodiscard]] double hit_rate(const std::uint64_t hits, const std::uint64_t misses) noexcept {
  const std::uint64_t total = hits + misses;
  return total == 0 ? 0.0 : static_cast<double>(hits) / static_cast<double>(total);
}

[[nodiscard]] std::optional<Digest128Value>
digest_allocation(CacheManager& manager, const AllocationId allocation_id,
                  const std::uint64_t logical_bytes,
                  const std::function<void()>& heartbeat) {
  const auto words = manager.backing_words(allocation_id);
  if (!words.has_value()) {
    return std::nullopt;
  }
  Digest128Accumulator digest;
  const std::uint64_t word_count = logical_bytes / word_bytes;
  constexpr std::uint64_t digest_batch_words = 16ULL * 1024ULL * 1024ULL;
  for (std::uint64_t begin = 0; begin < word_count; begin += digest_batch_words) {
    const std::uint64_t count = std::min(digest_batch_words, word_count - begin);
    const auto span = std::span<const std::uint32_t>(
        *words + begin, static_cast<std::size_t>(count));
    if (!digest.update_words(span, begin)) {
      return std::nullopt;
    }
    heartbeat();
  }
  return digest.value();
}

void fill_workload_metrics(WorkloadResult& workload, const CacheMetrics& before,
                           const CacheMetrics& after, const std::uint64_t event_before,
                           const std::uint64_t event_after, const std::uint64_t unsafe_before,
                           const std::uint64_t unsafe_after, const std::uint64_t shrink_before,
                           const std::uint64_t shrink_after, const std::uint64_t grow_before,
                           const std::uint64_t grow_after) {
  workload.cache_hits = counter_delta(after.cache_hits, before.cache_hits);
  workload.cache_misses = counter_delta(after.cache_misses, before.cache_misses);
  workload.cache_hit_rate = hit_rate(*workload.cache_hits, *workload.cache_misses);
  workload.bytes_h2d = counter_delta(after.h2d_bytes, before.h2d_bytes);
  workload.bytes_d2h = counter_delta(after.d2h_bytes, before.d2h_bytes);
  workload.clean_evictions = counter_delta(after.clean_evictions, before.clean_evictions);
  workload.dirty_evictions = counter_delta(after.dirty_evictions, before.dirty_evictions);
  workload.writebacks_completed =
      counter_delta(after.writebacks_completed, before.writebacks_completed);
  workload.prefetch_issued = counter_delta(after.prefetch_issued, before.prefetch_issued);
  workload.prefetch_useful = counter_delta(after.prefetch_useful, before.prefetch_useful);
  workload.prefetch_wasted = counter_delta(after.prefetch_wasted, before.prefetch_wasted);
  workload.prefetch_cancelled = counter_delta(after.prefetch_cancelled, before.prefetch_cancelled);
  workload.prefetch_promoted =
      counter_delta(after.prefetch_promoted, before.prefetch_promoted);
  workload.sequential_bypasses = 0;
  workload.target_shrink_count = counter_delta(shrink_after, shrink_before);
  workload.target_grow_count = counter_delta(grow_after, grow_before);
  workload.mapping_count =
      counter_delta(after.mappings_completed, before.mappings_completed);
  workload.unmap_count = counter_delta(after.unmaps_completed, before.unmaps_completed);
  workload.set_access_count =
      counter_delta(after.set_access_completed, before.set_access_completed);
  workload.handle_reuse_count = counter_delta(after.handle_reuses, before.handle_reuses);
  workload.event_boundary_count = counter_delta(event_after, event_before);
  workload.unsafe_remap_count = counter_delta(after.unsafe_remaps, before.unsafe_remaps);
  workload.unsafe_transition_count = counter_delta(unsafe_after, unsafe_before);
}

[[nodiscard]] bool run_workload(CacheManager& manager, ExecutorResult& result,
                                const ExecutorOptions& options, const AllocationId allocation_id,
                                const std::uint64_t logical_bytes,
                                const std::uint64_t chunk_bytes,
                                const RequestedPolicy policy_kind, const ScenarioTrace& trace,
                                const ProgressCallback& progress) {
  WorkloadResult workload;
  workload.scenario = std::string(scenario_kind_name(trace.scenario));
  workload.policy = policy_kind == RequestedPolicy::lru ? "lru" : "clock";
  workload.status = "running";
  workload.logical_bytes = logical_bytes;
  workload.logical_chunk_count = ((logical_bytes - 1ULL) / chunk_bytes) + 1ULL;
  workload.maximum_working_set_bytes = chunk_bytes;
  workload.operations_retired = 0;
  workload.passes_completed = 0;
  workload.read_operations = 0;
  workload.read_write_operations = 0;
  workload.write_only_operations = 0;
  result.workloads.push_back(workload);
  WorkloadResult& current = result.workloads.back();
  manager.set_workload_context(current.policy, current.scenario);
  manager.set_policy(policy_kind);

  Clock::time_point last_heartbeat = Clock::now();
  const std::uint64_t total_accesses = static_cast<std::uint64_t>(std::count_if(
      trace.operations.begin(), trace.operations.end(), [](const ScenarioOperation& operation) {
        return operation.kind == ScenarioOperationKind::access;
      }));
  const auto send_progress = [&](const bool force) {
    if (!progress) {
      return;
    }
    const Clock::time_point now = Clock::now();
    if (force || now - last_heartbeat >= options.progress_heartbeat) {
      progress(result, current.operations_retired.value_or(0), total_accesses);
      last_heartbeat = now;
    }
  };
  if (!manager.reset_backing(allocation_id, [&]() { send_progress(false); }) ||
      !manager.prepare_tokens(static_cast<std::uint64_t>(trace.operations.size()))) {
    current.status = "failed";
    return false;
  }

  const CacheMetrics before = manager.metrics();
  const std::uint64_t event_before = manager.event_boundaries();
  const std::uint64_t unsafe_before = manager.unsafe_transitions();
  const std::uint64_t shrink_before = manager.target_shrinks();
  const std::uint64_t grow_before = manager.target_grows();
  const std::uint64_t mismatch_before = manager.mismatch_count();
  std::vector<std::uint32_t> expected_tokens(trace.operations.size(), 0);
  std::vector<std::uint32_t> actual_tokens(trace.operations.size(), 0);
  const Clock::time_point started = Clock::now();

  for (std::size_t index = 0; index < trace.operations.size(); ++index) {
    const ScenarioOperation& operation = trace.operations[index];
    if (!manager.poll()) {
      current.status = "failed";
      return false;
    }
    if (operation.kind == ScenarioOperationKind::pressure_acquire) {
      if (!manager.acquire_pressure(operation.pressure_bytes)) {
        current.status = "failed";
        return false;
      }
      send_progress(true);
      continue;
    }
    if (operation.kind == ScenarioOperationKind::pressure_release) {
      if (!manager.release_pressure()) {
        current.status = "failed";
        return false;
      }
      send_progress(true);
      continue;
    }
    if (!operation.access.has_value()) {
      current.status = "failed";
      record_failure(result, exit_failure, "execution", "scenario_trace",
                     "access operation has no range");
      return false;
    }
    std::uint64_t expected_token = 0;
    if (!manager.enqueue_transaction(operation, expected_tokens, expected_token)) {
      current.status = "failed";
      return false;
    }
    switch (operation.access->mode) {
    case AccessMode::read:
      ++*current.read_operations;
      break;
    case AccessMode::read_write:
      ++*current.read_write_operations;
      break;
    case AccessMode::write_only:
      ++*current.write_only_operations;
      break;
    }
    ++*current.operations_retired;
    current.passes_completed =
        std::max(*current.passes_completed, operation.pass_index + 1U);

    std::vector<ChunkKey> lookahead;
    lookahead.reserve(options.prefetch_distance);
    for (std::size_t future = index + 1U;
         future < trace.operations.size() && lookahead.size() < options.prefetch_distance;
         ++future) {
      const ScenarioOperation& candidate = trace.operations[future];
      if (candidate.kind != ScenarioOperationKind::access || !candidate.access.has_value()) {
        continue;
      }
      const ChunkKey key{candidate.access->allocation_id,
                         candidate.access->offset_bytes / chunk_bytes};
      if (std::find(lookahead.begin(), lookahead.end(), key) == lookahead.end()) {
        lookahead.push_back(key);
      }
    }
    if (!manager.prefetch(lookahead, operation.sequence)) {
      current.status = "failed";
      return false;
    }
    send_progress(true);
  }

  if (!manager.collect_tokens(actual_tokens)) {
    current.status = "failed";
    return false;
  }
  for (const ScenarioOperation& operation : trace.operations) {
    if (operation.kind != ScenarioOperationKind::access) {
      continue;
    }
    const std::size_t index = static_cast<std::size_t>(operation.sequence);
    if (expected_tokens[index] != actual_tokens[index]) {
      record_failure(result, exit_corruption, "verification", "verification_token",
                     "GPU verification token differs from the CPU reference", std::nullopt,
                     nullptr, std::nullopt,
                     operation.access.has_value()
                         ? std::optional<std::uint64_t>(operation.access->offset_bytes)
                         : std::nullopt,
                     current.policy, current.scenario);
      current.status = "failed";
      current.mismatch_count = 1;
      current.first_mismatch_byte_offset =
          operation.access.has_value()
              ? std::optional<std::uint64_t>(operation.access->offset_bytes)
              : std::nullopt;
      return false;
    }
  }
  const auto expected_digest = digest_allocation(
      manager, allocation_id, logical_bytes, [&]() { send_progress(false); });
  if (!expected_digest.has_value() || !manager.verify_and_flush()) {
    current.status = "failed";
    return false;
  }
  const auto output_digest = digest_allocation(
      manager, allocation_id, logical_bytes, [&]() { send_progress(false); });
  if (!output_digest.has_value()) {
    current.status = "failed";
    return false;
  }

  if (manager.metrics().handle_reuses == before.handle_reuses) {
    const std::array audit{ChunkKey{allocation_id, 0}};
    if (!manager.prefetch(audit, trace.operations.size()) || !manager.drain(true)) {
      current.status = "failed";
      return false;
    }
  }
  const CacheMetrics after = manager.metrics();
  fill_workload_metrics(current, before, after, event_before, manager.event_boundaries(),
                        unsafe_before, manager.unsafe_transitions(), shrink_before,
                        manager.target_shrinks(), grow_before, manager.target_grows());
  current.stable_addresses_verified = manager.stable_addresses();
  current.full_verification_completed = true;
  current.matches_cpu = *expected_digest == *output_digest &&
                        manager.mismatch_count() == mismatch_before;
  current.expected_digest128 = format_digest128(*expected_digest);
  current.output_digest128 = format_digest128(*output_digest);
  current.mismatch_count = counter_delta(manager.mismatch_count(), mismatch_before);
  current.first_mismatch_byte_offset = manager.first_mismatch_offset();
  current.elapsed_ms = milliseconds_between(started, Clock::now());
  const long double processed =
      static_cast<long double>(logical_bytes) *
      static_cast<long double>(current.operations_retired.value_or(0));
  if (*current.elapsed_ms > 0.0) {
    current.throughput_gib_per_second = static_cast<double>(
        processed / (1024.0L * 1024.0L * 1024.0L) /
        (static_cast<long double>(*current.elapsed_ms) / 1000.0L));
  } else {
    current.throughput_gib_per_second = 0.0;
  }
  if (!current.matches_cpu.value_or(false)) {
    record_failure(result, exit_corruption, "verification", "final_digest",
                   "final pageable backing digest differs from the CPU reference", std::nullopt,
                   nullptr, std::nullopt, current.first_mismatch_byte_offset, current.policy,
                   current.scenario);
    current.status = "failed";
    return false;
  }
  current.status = "completed";
  send_progress(true);
  return true;
}

void assign_timing_summary(TimingSummary& destination, const std::vector<double>& samples) {
  const auto summary = summarize_timings(samples);
  if (!summary.has_value()) {
    destination = {};
    return;
  }
  destination.sample_count = summary->count;
  destination.total_ms = summary->total;
  destination.minimum_ms = summary->minimum;
  destination.median_ms = summary->median;
  destination.p95_ms = summary->percentile_95;
  destination.maximum_ms = summary->maximum;
}

[[nodiscard]] std::string seed_string(const std::uint64_t seed) {
  std::ostringstream output;
  output << std::hex << std::setfill('0') << std::setw(16) << seed;
  return output.str();
}

[[nodiscard]] bool cleanup_complete(const Cleanup& cleanup) noexcept {
  return cleanup.transactions_drained.value_or(false) &&
         cleanup.prefetch_drained.value_or(false) &&
         cleanup.writebacks_completed.value_or(false) &&
         cleanup.events_drained.value_or(false) && cleanup.events_destroyed.value_or(false) &&
         cleanup.streams_destroyed.value_or(false) && cleanup.module_unloaded.value_or(false) &&
         cleanup.mappings_removed.value_or(false) &&
         cleanup.physical_handles_released.value_or(false) &&
         cleanup.virtual_reservations_released.value_or(false) &&
         cleanup.pinned_staging_released.value_or(false) &&
         cleanup.host_backing_released.value_or(false) &&
         cleanup.context_destroyed.value_or(false) && cleanup.trace_closed.value_or(false) &&
         cleanup.worker_terminated.value_or(false);
}

} // namespace

ExecutorResult run_executor(cuda::CudaApi& api, const ExecutorOptions& options,
                            const ProgressCallback& progress, const TraceCallback& trace,
                            const ExecutorEnvironment* environment) {
  static_assert(std::is_nothrow_move_constructible_v<ExecutorResult>);
  ExecutorResult result;
  result.exit_code = exit_failure;
  result.status = "failed";
  result.configuration.requested_device_ordinal = options.device_ordinal;
  result.configuration.requested_logical_bytes = options.logical_bytes;
  result.configuration.requested_chunk_bytes = options.chunk_bytes;
  result.configuration.requested_cache_target_bytes = options.cache_target_bytes;
  result.configuration.staging_slots = options.staging_slots;
  result.configuration.policy = options.policy == RequestedPolicy::clock
                                    ? "clock"
                                    : options.policy == RequestedPolicy::lru ? "lru" : "both";
  result.configuration.prefetch_distance = options.prefetch_distance;
  result.configuration.scenario = std::string(scenario_kind_name(options.scenario));
  result.configuration.passes = options.passes;
  result.configuration.pressure_bytes = options.pressure_bytes;
  result.configuration.device_headroom_bytes = options.device_headroom_bytes;
  result.configuration.budget_poll_ms =
      static_cast<std::uint64_t>(options.budget_poll_interval.count());
  result.configuration.stall_timeout_ms =
      static_cast<std::uint64_t>(options.stall_timeout.count());
  result.configuration.timeout_ms = 300'000;
  result.configuration.seed_hex = seed_string(options.seed);
  result.configuration.sizing_mode = options.logical_bytes.has_value() ? "explicit" : "auto";
  result.configuration.trace_enabled = options.trace_enabled;
  result.configuration.identifiers_included = options.include_identifiers;

  result.cleanup.transactions_drained = true;
  result.cleanup.prefetch_drained = true;
  result.cleanup.writebacks_completed = true;
  result.cleanup.events_drained = true;
  result.cleanup.events_destroyed = true;
  result.cleanup.streams_destroyed = true;
  result.cleanup.module_unloaded = true;
  result.cleanup.mappings_removed = true;
  result.cleanup.physical_handles_released = true;
  result.cleanup.virtual_reservations_released = true;
  result.cleanup.pinned_staging_released = true;
  result.cleanup.host_backing_released = true;
  result.cleanup.context_destroyed = true;
  result.cleanup.trace_closed = true;
  result.cleanup.worker_terminated = true;

  cuda::abi::Device device = 0;
  cuda::abi::Context previous_context = nullptr;
  cuda::abi::Context context = nullptr;
  std::unique_ptr<CacheManager> manager;
  const Clock::time_point execution_started = Clock::now();

  const auto populate_manager_report = [&]() {
    if (manager == nullptr) {
      return;
    }
    const CacheMetrics& metrics = manager->metrics();
    result.cache.target_bytes_initial = result.configuration.initial_cache_target_bytes;
    result.cache.target_bytes_minimum = manager->target_minimum_bytes();
    result.cache.target_bytes_maximum = manager->target_maximum_bytes();
    result.cache.target_bytes_end = manager->target_bytes();
    result.cache.resident_bytes_peak = metrics.resident_peak_bytes;
    result.cache.pinned_staging_bytes =
        result.configuration.effective_chunk_bytes.value_or(0) * options.staging_slots;
    result.cache.physical_frame_count_peak =
        result.configuration.effective_chunk_bytes.value_or(0) == 0
            ? 0
            : metrics.resident_peak_bytes /
                  result.configuration.effective_chunk_bytes.value_or(1);
    result.cache.physical_handle_create_count = metrics.handles_created;
    result.cache.physical_handle_release_count = metrics.handles_released;
    result.cache.handle_reuse_count = metrics.handle_reuses;
    result.cache.mapping_count = metrics.mappings_completed;
    result.cache.unmap_count = metrics.unmaps_completed;
    result.cache.set_access_count = metrics.set_access_completed;
    result.cache.event_boundary_count = manager->event_boundaries();
    result.cache.unsafe_remap_count = metrics.unsafe_remaps;
    result.cache.unsafe_transition_count = manager->unsafe_transitions();
    result.cache.cache_hits = metrics.cache_hits;
    result.cache.cache_misses = metrics.cache_misses;
    result.cache.cache_hit_rate = hit_rate(metrics.cache_hits, metrics.cache_misses);
    result.cache.clean_evictions = metrics.clean_evictions;
    result.cache.dirty_evictions = metrics.dirty_evictions;
    result.cache.writebacks_completed = metrics.writebacks_completed;
    result.cache.prefetch_issued = metrics.prefetch_issued;
    result.cache.prefetch_useful = metrics.prefetch_useful;
    result.cache.prefetch_wasted = metrics.prefetch_wasted;
    result.cache.prefetch_cancelled = metrics.prefetch_cancelled;
    result.cache.prefetch_promoted = metrics.prefetch_promoted;
    result.cache.sequential_bypasses = 0;
    result.cache.target_shrink_count = manager->target_shrinks();
    result.cache.target_grow_count = manager->target_grows();
    result.cache.target_oom_retry_count = manager->target_oom_retries();
    result.cache.maximum_working_set_bytes =
        result.configuration.effective_chunk_bytes.value_or(0);

    result.telemetry.total_elapsed_ms =
        milliseconds_between(execution_started, Clock::now());
    result.telemetry.bytes_h2d = metrics.h2d_bytes;
    result.telemetry.bytes_d2h = metrics.d2h_bytes;
    result.telemetry.budget_sample_count = manager->budget_samples();
    result.telemetry.cuda_free_bytes_minimum = manager->cuda_free_minimum();
    result.telemetry.cuda_free_bytes_end = manager->cuda_free_end();
    result.telemetry.wddm_available_bytes_minimum = manager->wddm_available_minimum();
    result.telemetry.wddm_available_bytes_end = manager->wddm_available_end();
    const std::uint64_t maximum_target = manager->target_maximum_bytes();
    result.telemetry.resident_occupancy_peak =
        maximum_target == 0 ? 0.0
                            : std::min(1.0, static_cast<double>(metrics.resident_peak_bytes) /
                                                static_cast<double>(maximum_target));
    result.telemetry.resident_occupancy_mean =
        manager->resident_samples() == 0
            ? 0.0
            : static_cast<double>(manager->resident_ratio_sum() /
                                  static_cast<long double>(manager->resident_samples()));
    assign_timing_summary(result.telemetry.remap_timing, manager->remap_samples());
    assign_timing_summary(result.telemetry.h2d_timing, manager->h2d_samples());
    assign_timing_summary(result.telemetry.kernel_timing, manager->kernel_samples());
    assign_timing_summary(result.telemetry.d2h_timing, manager->d2h_samples());
    assign_timing_summary(result.telemetry.writeback_timing, manager->writeback_samples());
    result.telemetry.trace_records_emitted = manager->trace_records();
    result.telemetry.trace_records_dropped = 0;
    result.telemetry.trace_complete = true;

    result.proof.logical_bytes = result.configuration.effective_logical_bytes;
    result.proof.chunk_bytes = result.configuration.effective_chunk_bytes;
    if (result.configuration.effective_logical_bytes.has_value() &&
        result.configuration.effective_chunk_bytes.has_value()) {
      result.proof.logical_chunk_count =
          ((*result.configuration.effective_logical_bytes - 1ULL) /
           *result.configuration.effective_chunk_bytes) +
          1ULL;
    }
    result.proof.maximum_cache_target_bytes = manager->target_maximum_bytes();
    result.proof.pinned_staging_bytes = result.cache.pinned_staging_bytes;
    result.proof.kernel_module_version = residency_workload_module_version;
    result.proof.kernel_module_sha256 = std::string(residency_workload_module_sha256);
    result.proof.pattern_version = "xvram_residency_workload_v1";
    result.proof.handles_reused = metrics.handle_reuses > 0;
    result.proof.stable_virtual_addresses_verified = manager->stable_addresses();
    result.proof.set_access_after_map_verified =
        manager->set_access_verified() && metrics.mappings_completed == metrics.set_access_completed;
    result.proof.event_boundaries_verified =
        metrics.unmaps_completed == manager->event_boundaries();
    result.proof.no_physical_aliases_verified = manager->no_aliases();
    result.proof.dirty_writeback_verified = manager->dirty_writeback_verified();
    result.proof.staging_pool_bounded =
        metrics.staging_slots_peak <= metrics.staging_slots_capacity;
    result.proof.cache_target_respected =
        metrics.resident_peak_bytes <= metrics.target_peak_bytes;
    result.proof.raw_virtual_addresses_omitted = true;
  };

  const auto finish = [&]() noexcept -> ExecutorResult {
    if (manager != nullptr) {
      manager->close();
      populate_manager_report();
    }
    if (context != nullptr) {
      std::size_t free_bytes = 0;
      std::size_t total_bytes = 0;
      if (api.mem_get_info_ != nullptr &&
          api.mem_get_info_(&free_bytes, &total_bytes) == cuda::abi::success &&
          result.device.has_value()) {
        result.device->free_memory_bytes_end = static_cast<std::uint64_t>(free_bytes);
      }
      (void)total_bytes;
      const cuda::abi::Result destroy = api.context_destroy_(context);
      if (destroy != cuda::abi::success) {
        result.cleanup.context_destroyed = false;
        append_diagnostic(result, probe::DiagnosticLevel::error, "cuCtxDestroy",
                          cuda_message(api, destroy), static_cast<std::int64_t>(destroy));
      } else {
        result.cleanup.context_destroyed = true;
      }
      context = nullptr;
      if (api.context_set_current_(previous_context) != cuda::abi::success) {
        result.cleanup.context_destroyed = false;
        append_diagnostic(result, probe::DiagnosticLevel::error, "cuCtxSetCurrent(previous)",
                          "failed to restore the previous CUDA context");
      }
    }
    if (result.device.has_value() && manager != nullptr) {
      result.device->wddm_available_bytes_minimum = manager->wddm_available_minimum();
      result.device->wddm_available_bytes_end = manager->wddm_available_end();
    }
    result.cleanup.complete = cleanup_complete(result.cleanup);
    if (result.status == "completed" && !result.cleanup.complete.value_or(false)) {
      try {
        record_failure(result, exit_failure, "cleanup", "resource_ledger",
                       "one or more cache cleanup stages failed");
      } catch (...) {
        result.exit_code = exit_failure;
        result.status = "failed";
      }
    }
    return std::move(result);
  };

  if (options.device_ordinal < 0 || options.chunk_bytes == 0 || options.staging_slots < 2U ||
      options.staging_slots > 8U || options.prefetch_distance > 8U || options.passes < 2U ||
      options.passes > 8U || options.budget_poll_interval.count() <= 0 ||
      options.stall_timeout.count() <= 0 || options.progress_heartbeat.count() <= 0 ||
      (options.logical_bytes.has_value() &&
       (*options.logical_bytes == 0 || *options.logical_bytes % word_bytes != 0)) ||
      (options.cache_target_bytes.has_value() && *options.cache_target_bytes == 0) ||
      (options.pressure_bytes.has_value() && *options.pressure_bytes == 0)) {
    result.reason = "invalid_configuration";
    record_failure(result, exit_prerequisite, "configuration", "validate",
                   "the requested Phase 2 configuration is unsafe or invalid");
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
    result.reason = "cuda_residency_symbols_unavailable";
    record_failure(result, exit_prerequisite, "preflight", "resolve_cuda_symbols",
                   "CUDA driver does not export every symbol required by Phase 2");
    return finish();
  }
  const cuda::abi::Result initialized = api.init_(0);
  if (initialized != cuda::abi::success) {
    result.reason = "cuda_initialization_failed";
    record_failure(result, exit_prerequisite, "preflight", "cuInit",
                   cuda_message(api, initialized), initialized, &api);
    return finish();
  }
  int device_count = 0;
  if (!require_cuda(result, api, api.device_get_count_(&device_count), "preflight",
                    "cuDeviceGetCount")) {
    return finish();
  }
  if (options.device_ordinal >= device_count) {
    result.reason = "device_ordinal_unavailable";
    record_failure(result, exit_prerequisite, "preflight", "select_device",
                   "requested CUDA device ordinal is unavailable");
    return finish();
  }
  if (!require_cuda(result, api, api.device_get_(&device, options.device_ordinal), "preflight",
                    "cuDeviceGet")) {
    return finish();
  }

  DeviceInfo device_info;
  device_info.ordinal = options.device_ordinal;
  std::array<char, 256> name{};
  if (!require_cuda(result, api,
                    api.device_get_name_(name.data(), static_cast<int>(name.size()), device),
                    "preflight", "cuDeviceGetName")) {
    return finish();
  }
  device_info.name = name.data();
  std::size_t total_memory = 0;
  if (!require_cuda(result, api, api.device_total_memory_(&total_memory, device), "preflight",
                    "cuDeviceTotalMem")) {
    return finish();
  }
  device_info.total_memory_bytes = static_cast<std::uint64_t>(total_memory);
  std::uint32_t vmm = 0;
  std::uint32_t uva = 0;
  std::uint32_t multiprocessors = 0;
  std::uint32_t warp_size = 0;
  std::uint32_t maximum_threads = 0;
  if (!get_attribute(api, result, device,
                     cuda::abi::attributes::virtual_memory_management_supported,
                     "CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED", vmm) ||
      !get_attribute(api, result, device, cuda::abi::attributes::unified_addressing,
                     "CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING", uva) ||
      !get_attribute(api, result, device, cuda::abi::attributes::multiprocessor_count,
                     "CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT", multiprocessors) ||
      !get_attribute(api, result, device, cuda::abi::attributes::warp_size,
                     "CU_DEVICE_ATTRIBUTE_WARP_SIZE", warp_size) ||
      !get_attribute(api, result, device, cuda::abi::attributes::max_threads_per_block,
                     "CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK", maximum_threads)) {
    return finish();
  }
  device_info.vmm_supported = vmm != 0;
  device_info.uva_supported = uva != 0;
  if (!device_info.vmm_supported || !device_info.uva_supported || multiprocessors == 0 ||
      warp_size == 0 || maximum_threads < warp_size) {
    result.device = device_info;
    result.reason = "vmm_uva_or_geometry_unavailable";
    record_failure(result, exit_prerequisite, "preflight", "capabilities",
                   "selected device lacks VMM, UVA, or safe kernel geometry");
    return finish();
  }
  if (options.include_identifiers && api.device_get_uuid_ != nullptr) {
    cuda::abi::Uuid uuid{};
    if (api.device_get_uuid_(&uuid, device) == cuda::abi::success) {
      device_info.uuid = bytes_as_uuid(reinterpret_cast<const unsigned char*>(uuid.bytes));
    }
  }
  if (options.include_identifiers && api.device_get_pci_bus_id_ != nullptr) {
    std::array<char, 32> pci{};
    if (api.device_get_pci_bus_id_(pci.data(), static_cast<int>(pci.size()), device) ==
        cuda::abi::success) {
      device_info.pci_bus_id = pci.data();
    }
  }
  int tcc = 0;
  if (api.device_get_attribute_(&tcc, cuda::abi::attributes::tcc_driver, device) ==
      cuda::abi::success) {
#ifdef _WIN32
    device_info.driver_model = tcc != 0 ? "tcc" : "wddm";
#else
    device_info.driver_model = tcc != 0 ? "tcc" : "unknown";
#endif
  }

  if (!require_cuda(result, api, api.context_get_current_(&previous_context), "preflight",
                    "cuCtxGetCurrent") ||
      !require_cuda(result, api,
                    api.context_create_(&context, CU_CTX_SCHED_BLOCKING_SYNC, device),
                    "preflight", "cuCtxCreate")) {
    return finish();
  }
  result.cleanup.context_destroyed = false;
  cuda::abi::Device context_device = -1;
  if (!require_cuda(result, api, api.context_get_device_(&context_device), "preflight",
                    "cuCtxGetDevice") ||
      context_device != device) {
    if (!result.failure.has_value()) {
      record_failure(result, exit_failure, "preflight", "cuCtxGetDevice",
                     "isolated context selected a different CUDA device");
    }
    return finish();
  }

  std::size_t free_memory = 0;
  std::size_t context_total = 0;
  if (!require_cuda(result, api, api.mem_get_info_(&free_memory, &context_total), "preflight",
                    "cuMemGetInfo")) {
    return finish();
  }
  (void)context_total;
  device_info.free_memory_bytes_start = static_cast<std::uint64_t>(free_memory);
  CUmemAllocationProp property{};
  property.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  property.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  property.location.id = device;
  property.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
  std::size_t minimum_granularity = 0;
  std::size_t recommended_granularity = 0;
  if (!require_cuda(result, api,
                    api.mem_get_allocation_granularity_(&minimum_granularity, &property,
                                                        CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                    "preflight", "cuMemGetAllocationGranularity(minimum)") ||
      !require_cuda(result, api,
                    api.mem_get_allocation_granularity_(&recommended_granularity, &property,
                                                        CU_MEM_ALLOC_GRANULARITY_RECOMMENDED),
                    "preflight", "cuMemGetAllocationGranularity(recommended)")) {
    return finish();
  }
  device_info.minimum_granularity_bytes = static_cast<std::uint64_t>(minimum_granularity);
  device_info.recommended_granularity_bytes = static_cast<std::uint64_t>(recommended_granularity);
  const auto vmm_alignment = checked_least_common_multiple(
      static_cast<std::uint64_t>(minimum_granularity),
      static_cast<std::uint64_t>(recommended_granularity));
  const auto alignment = vmm_alignment.has_value()
                             ? checked_least_common_multiple(*vmm_alignment, word_bytes)
                             : std::nullopt;
  const auto effective_chunk = alignment.has_value()
                                   ? checked_align_up(options.chunk_bytes, *alignment)
                                   : std::nullopt;
  if (!effective_chunk.has_value() || *effective_chunk == 0) {
    result.reason = "granularity_overflow";
    record_failure(result, exit_prerequisite, "planning", "align_chunk",
                   "effective VMM chunk alignment overflowed");
    return finish();
  }
  result.configuration.effective_chunk_bytes = *effective_chunk;

  std::optional<WddmIdentity> wddm_identity;
  std::optional<BudgetSnapshot> initial_budget;
#ifdef _WIN32
  if (environment == nullptr) {
    if (api.device_get_luid_ == nullptr) {
      result.device = device_info;
      result.reason = "cuda_luid_unavailable";
      record_failure(result, exit_prerequisite, "preflight", "cuDeviceGetLuid",
                     "Windows cache safety requires a CUDA adapter LUID");
      return finish();
    }
    WddmIdentity identity;
    std::array<char, 8> luid{};
    if (!require_cuda(result, api,
                      api.device_get_luid_(luid.data(), &identity.node_mask, device),
                      "preflight", "cuDeviceGetLuid")) {
      return finish();
    }
    std::memcpy(identity.luid.data(), luid.data(), identity.luid.size());
    if (options.include_identifiers) {
      device_info.luid = bytes_as_hex(identity.luid.data(), identity.luid.size());
    }
    wddm_identity = identity;
    initial_budget = query_wddm(identity, result);
    if (!initial_budget.has_value()) {
      result.device = device_info;
      result.reason = "wddm_budget_unavailable";
      record_failure(result, exit_prerequisite, "preflight", "QueryVideoMemoryInfo",
                     "Windows cache safety requires the DXGI local-memory budget");
      return finish();
    }
  }
#endif
  if (environment != nullptr && environment->initial_device_budget.has_value()) {
    initial_budget = environment->initial_device_budget;
  }
  if (initial_budget.has_value()) {
    device_info.wddm_budget_bytes_start = initial_budget->budget_bytes;
    device_info.wddm_usage_bytes_start = initial_budget->usage_bytes;
    device_info.wddm_available_bytes_start = initial_budget->available_bytes();
    device_info.wddm_available_bytes_minimum = initial_budget->available_bytes();
    device_info.wddm_available_bytes_end = initial_budget->available_bytes();
  }
  result.device = device_info;

  const probe::SystemInfo system = platform::collect_system_info();
  const std::uint64_t physical_host = environment != nullptr
                                          ? environment->physical_host_bytes
                                          : system.physical_memory_bytes.value_or(0);
  const std::uint64_t available_host = environment != nullptr
                                           ? environment->available_host_bytes
                                           : system.available_memory_bytes.value_or(0);
  const auto pinned_bytes = checked_multiply(*effective_chunk, options.staging_slots);
  constexpr std::uint64_t minimum_host_headroom = 4ULL * 1024ULL * 1024ULL * 1024ULL;
  const std::uint64_t host_headroom =
      std::max(minimum_host_headroom, physical_host / 4ULL);
  const auto reserved_one = pinned_bytes.has_value()
                                ? checked_add(host_headroom, *pinned_bytes)
                                : std::nullopt;
  const auto reserved = reserved_one.has_value()
                            ? checked_add(*reserved_one, service_host_bytes)
                            : std::nullopt;
  if (physical_host == 0 || available_host == 0 || !reserved.has_value() ||
      *reserved >= available_host) {
    result.reason = "insufficient_host_memory";
    record_failure(result, exit_oom, "planning", "host_memory",
                   "safe pageable backing limit is unavailable");
    return finish();
  }
  result.configuration.host_headroom_bytes = host_headroom;
  const std::uint64_t safe_logical = ((available_host - *reserved) / word_bytes) * word_bytes;
  std::uint64_t logical_bytes = 0;
  if (options.logical_bytes.has_value()) {
    logical_bytes = *options.logical_bytes;
    if (logical_bytes > safe_logical) {
      result.reason = "insufficient_host_memory";
      record_failure(result, exit_oom, "planning", "logical_size",
                     "requested logical heap exceeds the safe pageable RAM limit");
      return finish();
    }
  } else {
    const auto half = checked_add(device_info.total_memory_bytes / 2ULL,
                                  device_info.total_memory_bytes % 2ULL);
    const auto automatic = half.has_value()
                               ? checked_add(device_info.total_memory_bytes, *half)
                               : std::nullopt;
    if (!automatic.has_value()) {
      record_failure(result, exit_prerequisite, "planning", "logical_size",
                     "automatic logical size overflowed");
      return finish();
    }
    logical_bytes = (std::min(*automatic, safe_logical) / word_bytes) * word_bytes;
  }
  const auto minimum_logical = checked_multiply(*effective_chunk, 3ULL);
  if (!minimum_logical.has_value() || logical_bytes < *minimum_logical) {
    result.reason = "logical_heap_too_small";
    record_failure(result, exit_prerequisite, "planning", "logical_size",
                   "logical heap must contain at least three effective chunks");
    return finish();
  }
  result.configuration.effective_logical_bytes = logical_bytes;

  const std::uint64_t logical_cache_ceiling =
      ((logical_bytes - 1ULL) / *effective_chunk) * *effective_chunk;
  const std::uint64_t requested_cap =
      options.cache_target_bytes.value_or(device_info.total_memory_bytes);
  const std::uint64_t configured_cap =
      (std::min(requested_cap, logical_cache_ceiling) / *effective_chunk) * *effective_chunk;
  const auto minimum_target = checked_multiply(*effective_chunk, 2ULL);
  if (!minimum_target.has_value() || configured_cap < *minimum_target) {
    result.reason = "cache_target_too_small";
    record_failure(result, exit_prerequisite, "planning", "cache_target",
                   "cache target cannot satisfy the two-chunk minimum");
    return finish();
  }
  std::optional<WddmBudgetObservation> wddm;
  if (initial_budget.has_value()) {
    wddm = WddmBudgetObservation{initial_budget->budget_bytes, initial_budget->usage_bytes};
  }
  const BudgetTarget initial_target = calculate_budget_target(BudgetTargetInput{
      configured_cap, static_cast<std::uint64_t>(free_memory), 0, wddm,
      options.device_headroom_bytes, *effective_chunk, *effective_chunk});
  if (!initial_target) {
    result.reason = std::string(budget_target_error_name(initial_target.error));
    record_failure(result, exit_prerequisite, "planning", "initial_cache_target",
                   "initial working set cannot fit the live CUDA/WDDM budget");
    return finish();
  }
  result.configuration.initial_cache_target_bytes = initial_target.target_bytes;
  result.device->safe_device_budget_bytes = initial_target.target_bytes;

  const std::uint32_t threads =
      (std::min<std::uint32_t>(256U, maximum_threads) / warp_size) * warp_size;
  const std::uint32_t blocks = multiprocessors * 4U;
  std::function<std::optional<BudgetSnapshot>()> budget_provider;
  if (environment != nullptr && environment->query_device_budget) {
    budget_provider = environment->query_device_budget;
  }
#ifdef _WIN32
  else if (wddm_identity.has_value()) {
    budget_provider = [&result, identity = *wddm_identity]() mutable {
      return query_wddm(identity, result);
    };
  }
#endif
  manager = std::make_unique<CacheManager>(
      api, result, options, device, *effective_chunk, initial_target.target_bytes, configured_cap,
      blocks, threads, std::move(budget_provider), trace);
  if (!manager->setup()) {
    return finish();
  }
  const std::optional<AllocationId> allocation_id = manager->allocate(logical_bytes);
  if (!allocation_id.has_value()) {
    return finish();
  }

  ScenarioInput scenario_input;
  scenario_input.allocation_id = *allocation_id;
  scenario_input.logical_bytes = logical_bytes;
  scenario_input.chunk_bytes = *effective_chunk;
  scenario_input.cache_target_bytes = initial_target.target_bytes;
  scenario_input.pressure_bytes = options.pressure_bytes;
  scenario_input.passes = options.passes;
  scenario_input.seed = options.seed;
  std::vector<ScenarioTrace> traces;
  if (options.scenario == ScenarioKind::suite) {
    ScenarioSuiteResult generated = generate_scenario_suite(scenario_input);
    if (!generated) {
      result.reason = std::string(scenario_trace_error_name(generated.error));
      record_failure(result, exit_prerequisite, "planning", "generate_scenario_suite",
                     "scenario suite generation failed");
      return finish();
    }
    traces = std::move(generated.traces);
  } else {
    ScenarioTraceResult generated = generate_scenario_trace(options.scenario, scenario_input);
    if (!generated) {
      result.reason = std::string(scenario_trace_error_name(generated.error));
      record_failure(result, exit_prerequisite, "planning", "generate_scenario_trace",
                     "scenario generation failed");
      return finish();
    }
    traces.push_back(std::move(*generated.trace));
  }
  std::vector<RequestedPolicy> policies;
  if (options.policy == RequestedPolicy::both) {
    policies = {RequestedPolicy::clock, RequestedPolicy::lru};
  } else {
    policies = {options.policy};
  }

  bool workloads_ok = true;
  try {
    for (const RequestedPolicy policy : policies) {
      for (const ScenarioTrace& scenario_trace : traces) {
        if (!run_workload(*manager, result, options, *allocation_id, logical_bytes,
                          *effective_chunk, policy, scenario_trace, progress)) {
          workloads_ok = false;
          break;
        }
      }
      if (!workloads_ok) {
        break;
      }
    }
  } catch (const std::bad_alloc&) {
    record_failure(result, exit_oom, "execution", "host_allocation",
                   "host allocation failed while executing the cache workload");
    workloads_ok = false;
  } catch (const std::exception& exception) {
    record_failure(result, exit_failure, "execution", "internal_exception", exception.what());
    workloads_ok = false;
  } catch (...) {
    record_failure(result, exit_failure, "execution", "internal_exception",
                   "unknown exception while executing the cache workload");
    workloads_ok = false;
  }

  bool policies_match = true;
  if (workloads_ok && options.policy == RequestedPolicy::both) {
    for (const ScenarioTrace& scenario_trace : traces) {
      const std::string scenario_name = std::string(scenario_kind_name(scenario_trace.scenario));
      std::optional<std::string> reference;
      for (const WorkloadResult& workload : result.workloads) {
        if (workload.scenario != scenario_name) {
          continue;
        }
        if (!reference.has_value()) {
          reference = workload.output_digest128;
        } else if (reference != workload.output_digest128) {
          policies_match = false;
        }
      }
    }
    if (!policies_match) {
      record_failure(result, exit_corruption, "verification", "policy_digest",
                     "CLOCK and LRU produced different final digests");
      workloads_ok = false;
    }
  }

  if (workloads_ok) {
    Digest128Accumulator combined;
    std::uint64_t digest_index = 0;
    for (const WorkloadResult& workload : result.workloads) {
      if (!workload.output_digest128.has_value()) {
        workloads_ok = false;
        break;
      }
      for (const char character : *workload.output_digest128) {
        combined.update(static_cast<std::uint32_t>(static_cast<unsigned char>(character)),
                        digest_index++);
      }
    }
    result.proof.cpu_reference_digest128 = format_digest128(combined.value());
  }
  result.proof.logical_data_exceeds_vram = logical_bytes > device_info.total_memory_bytes;
  result.proof.cache_smaller_than_logical = initial_target.target_bytes < logical_bytes;
  result.proof.all_workloads_match_cpu =
      workloads_ok && std::all_of(result.workloads.begin(), result.workloads.end(),
                                  [](const WorkloadResult& workload) {
                                    return workload.status == "completed" &&
                                           workload.matches_cpu.value_or(false);
                                  });
  result.proof.policies_match = policies_match;

  if (workloads_ok) {
    const CacheMetrics& metrics = manager->metrics();
    const bool proof_ok = metrics.handle_reuses > 0 && manager->stable_addresses() &&
                          manager->set_access_verified() && manager->no_aliases() &&
                          manager->dirty_writeback_verified() && metrics.unsafe_remaps == 0 &&
                          manager->unsafe_transitions() == 0 &&
                          metrics.mappings_completed == metrics.set_access_completed &&
                          metrics.mappings_completed == metrics.unmaps_completed &&
                          metrics.unmaps_completed == manager->event_boundaries() &&
                          metrics.staging_slots_peak <= metrics.staging_slots_capacity &&
                          metrics.resident_peak_bytes <= metrics.target_peak_bytes &&
                          initial_target.target_bytes < logical_bytes && policies_match;
    if (!proof_ok) {
      record_failure(result, exit_failure, "proof", "invariants",
                     "one or more event-safe residency cache invariants were not satisfied");
    } else {
      result.exit_code = exit_completed;
      result.status = "completed";
      result.reason.reset();
    }
  }
  return finish();
}

} // namespace xvram::residency
