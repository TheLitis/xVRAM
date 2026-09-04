#define NOMINMAX

#include "platform/cuda/cuda_api.hpp"
#include "platform/nvcomp/nvcomp_api.hpp"
#include "residency/executor.hpp"
#include "residency/runtime.hpp"
#include "residency/workload.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using xvram::cuda::CudaApi;
using xvram::cuda::abi::Context;
using xvram::cuda::abi::DevicePointer;
using xvram::cuda::abi::Event;
using xvram::cuda::abi::Function;
using xvram::cuda::abi::GenericAllocationHandle;
using xvram::cuda::abi::Module;
using xvram::cuda::abi::Result;
using xvram::cuda::abi::Stream;

constexpr std::uint64_t kib = 1024ULL;
constexpr std::uint64_t mib = 1024ULL * kib;
constexpr std::uint64_t gib = 1024ULL * mib;
constexpr std::uint64_t chunk_bytes = 64ULL * kib;
constexpr std::uint64_t logical_bytes = 1536ULL * kib;

int failures = 0;
std::string_view check_context;

void check(const bool condition, const char* expression, const int line) {
  if (!condition) {
    if (!check_context.empty()) {
      std::cerr << '[' << check_context << "] ";
    }
    std::cerr << "line " << line << ": check failed: " << expression << '\n';
    ++failures;
  }
}

#define CHECK(expression) check(static_cast<bool>(expression), #expression, __LINE__)

struct CheckContext {
  explicit CheckContext(const std::string_view value) noexcept : previous(check_context) {
    check_context = value;
  }

  CheckContext(const CheckContext&) = delete;
  CheckContext& operator=(const CheckContext&) = delete;

  ~CheckContext() {
    check_context = previous;
  }

  std::string_view previous;
};

enum class FaultMode {
  none,
  invalid_event_query,
  fail_first_unmap,
};

enum class FaultSite {
  none,
  initialize,
  no_device,
  context_create,
  host_allocate,
  memory_create,
  map,
  set_access,
  h2d,
  kernel,
  event_record,
  event_query,
  d2h,
  unmap,
  release,
  event_destroy,
  context_destroy,
  corrupt_kernel_output,
};

struct FaultInjection {
  FaultSite site = FaultSite::none;
  Result result = CUDA_ERROR_UNKNOWN;
  std::uint64_t calls_to_skip = 0;
  std::uint64_t matching_calls = 0;
  bool persistent = false;
  bool triggered = false;

  [[nodiscard]] bool should_inject(const FaultSite candidate) noexcept {
    if (site != candidate) {
      return false;
    }
    ++matching_calls;
    if (matching_calls <= calls_to_skip || (triggered && !persistent)) {
      return false;
    }
    triggered = true;
    return true;
  }
};

struct FakeCuda {
  struct Token {
    std::uint64_t id = 0;
  };

  struct PhysicalAllocation {
    std::vector<std::byte> storage;
    std::uint64_t map_count = 0;
    bool released = false;
  };

  struct Mapping {
    GenericAllocationHandle handle = 0;
    std::size_t bytes = 0;
    bool access_set = false;
    bool work_started = false;
    bool completion_recorded = false;
    bool completion_observed = false;
  };

  struct LinearAllocation {
    std::vector<std::byte> storage;
  };

  struct StreamState {
    std::optional<DevicePointer> pending_mapping;
    Context context = nullptr;
    CUstreamCaptureStatus capture = CU_STREAM_CAPTURE_STATUS_NONE;
  };

  struct EventState {
    bool recorded = false;
    std::optional<DevicePointer> completion_mapping;
  };

  struct MappingView {
    DevicePointer base = 0;
    Mapping* mapping = nullptr;
    PhysicalAllocation* allocation = nullptr;
    std::size_t offset = 0;
  };

  struct LinearView {
    LinearAllocation* allocation = nullptr;
    std::size_t offset = 0;
  };

  FaultMode fault = FaultMode::none;
  FaultInjection injection;
  std::uint64_t free_memory_bytes = 16ULL * mib;
  std::uint64_t total_memory_bytes = 16ULL * mib;
  std::optional<std::uint64_t> free_memory_after_create_fault;
  bool event_fault_used = false;
  bool unmap_fault_used = false;
  bool quarantined_release_observed = false;
  std::vector<std::string> violations;
  std::vector<std::unique_ptr<Token>> tokens;
  std::map<DevicePointer, std::size_t> reservations;
  std::map<GenericAllocationHandle, PhysicalAllocation> physical;
  std::map<DevicePointer, Mapping> mappings;
  std::map<DevicePointer, LinearAllocation> linear;
  std::map<void*, std::unique_ptr<std::byte[]>, std::less<void*>> pinned;
  std::map<Stream, StreamState, std::less<Stream>> streams;
  std::vector<Stream> stream_creation_order;
  std::map<Event, EventState, std::less<Event>> events;
  std::map<Function, bool, std::less<Function>> codec_verification_functions;
  std::map<std::uint64_t, DevicePointer> stable_addresses;
  std::uint64_t next_token = 1;
  GenericAllocationHandle next_handle = 1;
  DevicePointer next_reservation = 0x100001000ULL;
  DevicePointer next_linear = 0x700000000ULL;
  Context current_context = nullptr;
  std::vector<Context> pushed_contexts;
  std::vector<Context> popped_contexts;
  std::vector<Context> destroyed_contexts;
  std::size_t maximum_active_mappings = 0;
  std::uint64_t map_calls = 0;
  std::uint64_t set_access_calls = 0;
  std::uint64_t unmap_calls = 0;
  std::uint64_t release_calls = 0;
  std::uint64_t address_free_calls = 0;
  std::uint64_t event_query_calls = 0;
  std::uint64_t kernel_calls = 0;
  std::uint64_t d2h_mapping_calls = 0;
  float event_elapsed_ms = 0.05F;

  void violation(std::string message) {
    violations.push_back(std::move(message));
  }

  template <typename Handle> [[nodiscard]] Handle make_token() {
    static_assert(std::is_pointer_v<Handle>);
    auto token = std::make_unique<Token>();
    token->id = next_token++;
    Token* raw = token.get();
    tokens.push_back(std::move(token));
    return reinterpret_cast<Handle>(raw);
  }

  [[nodiscard]] std::optional<MappingView> mapping_view(const DevicePointer address,
                                                        const std::size_t bytes) {
    auto found = mappings.upper_bound(address);
    if (found == mappings.begin()) {
      return std::nullopt;
    }
    --found;
    if (address < found->first) {
      return std::nullopt;
    }
    const std::uint64_t relative = address - found->first;
    if (relative > found->second.bytes || bytes > found->second.bytes - relative) {
      return std::nullopt;
    }
    auto allocation = physical.find(found->second.handle);
    if (allocation == physical.end() || allocation->second.released) {
      return std::nullopt;
    }
    return MappingView{found->first, &found->second, &allocation->second,
                       static_cast<std::size_t>(relative)};
  }

  [[nodiscard]] std::optional<LinearView> linear_view(const DevicePointer address,
                                                      const std::size_t bytes) {
    auto found = linear.upper_bound(address);
    if (found == linear.begin()) {
      return std::nullopt;
    }
    --found;
    const std::uint64_t relative = address - found->first;
    if (relative > found->second.storage.size() ||
        bytes > found->second.storage.size() - relative) {
      return std::nullopt;
    }
    return LinearView{&found->second, static_cast<std::size_t>(relative)};
  }

  void begin_mapping_work(const DevicePointer base, const Stream stream) {
    const auto mapping = mappings.find(base);
    const auto stream_state = streams.find(stream);
    if (mapping == mappings.end() || stream_state == streams.end()) {
      violation("mapping work referenced an unknown mapping or stream");
      return;
    }
    mapping->second.work_started = true;
    mapping->second.completion_recorded = false;
    mapping->second.completion_observed = false;
    stream_state->second.pending_mapping = base;
  }

  void observe_event(EventState& event) {
    if (!event.completion_mapping.has_value()) {
      return;
    }
    const auto mapping = mappings.find(*event.completion_mapping);
    if (mapping != mappings.end()) {
      mapping->second.completion_observed = true;
    }
  }

  void install(CudaApi& api);
};

FakeCuda* active_fake = nullptr;

[[nodiscard]] FakeCuda& fake() {
  return *active_fake;
}

struct ActiveFake {
  explicit ActiveFake(FakeCuda& value) {
    CHECK(active_fake == nullptr);
    active_fake = &value;
  }

  ActiveFake(const ActiveFake&) = delete;
  ActiveFake& operator=(const ActiveFake&) = delete;

  ~ActiveFake() {
    active_fake = nullptr;
  }
};

Result CUDAAPI fake_init(unsigned int) {
  if (fake().injection.should_inject(FaultSite::initialize)) {
    return fake().injection.result;
  }
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_driver_version(int* version) {
  *version = 13'000;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_error_name(const Result result, const char** name) {
  if (result == CUDA_ERROR_OUT_OF_MEMORY) {
    *name = "CUDA_ERROR_OUT_OF_MEMORY";
  } else if (result == CUDA_ERROR_NOT_READY) {
    *name = "CUDA_ERROR_NOT_READY";
  } else if (result == CUDA_ERROR_INVALID_VALUE) {
    *name = "CUDA_ERROR_INVALID_VALUE";
  } else {
    *name = "CUDA_ERROR_UNKNOWN";
  }
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_error_string(Result, const char** message) {
  *message = "injected fake CUDA failure";
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_device_count(int* count) {
  if (fake().injection.should_inject(FaultSite::no_device)) {
    *count = 0;
    return CUDA_SUCCESS;
  }
  *count = 1;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_device_get(CUdevice* device, const int ordinal) {
  if (ordinal != 0) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  *device = 0;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_device_name(char* name, const int length, CUdevice) {
  constexpr std::string_view value = "Fake Phase 2 CUDA VMM device";
  if (length <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const std::size_t capacity = static_cast<std::size_t>(length);
  const std::size_t copied = std::min(value.size(), capacity - 1U);
  std::memcpy(name, value.data(), copied);
  name[copied] = '\0';
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_device_total_memory(std::size_t* bytes, CUdevice) {
  *bytes = static_cast<std::size_t>(mib);
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_device_attribute(int* value, const CUdevice_attribute attribute, CUdevice) {
  using namespace xvram::cuda::abi::attributes;
  if (attribute == virtual_memory_management_supported || attribute == unified_addressing) {
    *value = 1;
  } else if (attribute == multiprocessor_count) {
    *value = 2;
  } else if (attribute == warp_size) {
    *value = 32;
  } else if (attribute == max_threads_per_block) {
    *value = 256;
  } else if (attribute == tcc_driver) {
    *value = 1;
  } else {
    *value = 0;
  }
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_granularity(std::size_t* granularity, const CUmemAllocationProp*,
                                const CUmemAllocationGranularity_flags option) {
  *granularity = option == CU_MEM_ALLOC_GRANULARITY_MINIMUM ? static_cast<std::size_t>(4ULL * kib)
                                                            : static_cast<std::size_t>(chunk_bytes);
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_context_get_current(Context* context) {
  *context = fake().current_context;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_context_get_device(CUdevice* device) {
  if (fake().current_context == nullptr) {
    return CUDA_ERROR_INVALID_CONTEXT;
  }
  *device = 0;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_context_set_current(const Context context) {
  fake().current_context = context;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_context_push_current(const Context context) {
  if (context == nullptr) {
    return CUDA_ERROR_INVALID_CONTEXT;
  }
  fake().pushed_contexts.push_back(context);
  fake().current_context = context;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_context_pop_current(Context* context) {
  if (context == nullptr || fake().current_context == nullptr) {
    return CUDA_ERROR_INVALID_CONTEXT;
  }
  *context = fake().current_context;
  fake().popped_contexts.push_back(*context);
  fake().current_context = nullptr;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_context_create(Context* context, unsigned int, CUdevice) {
  if (fake().injection.should_inject(FaultSite::context_create)) {
    return fake().injection.result;
  }
  *context = fake().make_token<Context>();
  fake().current_context = *context;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_context_destroy(const Context context) {
  fake().destroyed_contexts.push_back(context);
  if (fake().injection.should_inject(FaultSite::context_destroy)) {
    return fake().injection.result;
  }
  if (fake().current_context == context) {
    fake().current_context = nullptr;
  }
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_mem_get_info(std::size_t* free_bytes, std::size_t* total_bytes) {
  *free_bytes = static_cast<std::size_t>(fake().free_memory_bytes);
  *total_bytes = static_cast<std::size_t>(fake().total_memory_bytes);
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_address_reserve(DevicePointer* address, const std::size_t bytes, std::size_t,
                                    DevicePointer, unsigned long long) {
  const DevicePointer base = fake().next_reservation;
  fake().next_reservation += static_cast<DevicePointer>(bytes + 2ULL * chunk_bytes);
  fake().reservations.emplace(base, bytes);
  *address = base;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_address_free(const DevicePointer address, const std::size_t bytes) {
  ++fake().address_free_calls;
  const auto reservation = fake().reservations.find(address);
  if (reservation == fake().reservations.end() || reservation->second != bytes) {
    fake().violation("VA reservation was not freed with its exact original base and size");
    return CUDA_ERROR_INVALID_VALUE;
  }
  for (const auto& [mapped_address, mapping] : fake().mappings) {
    (void)mapping;
    if (mapped_address >= address && mapped_address - address < bytes) {
      fake().violation("VA reservation was freed while a mapping remained live");
      return CUDA_ERROR_INVALID_VALUE;
    }
  }
  fake().reservations.erase(reservation);
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_mem_create(GenericAllocationHandle* handle, const std::size_t bytes,
                               const CUmemAllocationProp*, unsigned long long) {
  if (fake().injection.should_inject(FaultSite::memory_create)) {
    if (fake().free_memory_after_create_fault.has_value()) {
      fake().free_memory_bytes = *fake().free_memory_after_create_fault;
    }
    return fake().injection.result;
  }
  const GenericAllocationHandle created = fake().next_handle++;
  FakeCuda::PhysicalAllocation allocation;
  allocation.storage.resize(bytes);
  fake().physical.emplace(created, std::move(allocation));
  *handle = created;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_mem_release(const GenericAllocationHandle handle) {
  ++fake().release_calls;
  const auto allocation = fake().physical.find(handle);
  if (allocation == fake().physical.end() || allocation->second.released) {
    fake().violation("physical handle was released more than once");
    return CUDA_ERROR_INVALID_HANDLE;
  }
  const bool still_mapped =
      std::any_of(fake().mappings.begin(), fake().mappings.end(),
                  [&](const auto& pair) { return pair.second.handle == handle; });
  if (still_mapped) {
    if (!fake().unmap_fault_used && !fake().event_fault_used) {
      fake().violation("a live mapped handle was released outside quarantine");
      return CUDA_ERROR_INVALID_VALUE;
    }
    fake().quarantined_release_observed = true;
  }
  if (fake().injection.should_inject(FaultSite::release)) {
    return fake().injection.result;
  }
  allocation->second.released = true;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_mem_map(const DevicePointer address, const std::size_t bytes, std::size_t,
                            const GenericAllocationHandle handle, unsigned long long) {
  ++fake().map_calls;
  if (fake().injection.should_inject(FaultSite::map)) {
    return fake().injection.result;
  }
  const auto allocation = fake().physical.find(handle);
  if (allocation == fake().physical.end() || allocation->second.released ||
      allocation->second.storage.size() != bytes) {
    fake().violation("mapping referenced an invalid physical allocation");
    return CUDA_ERROR_INVALID_VALUE;
  }
  const bool handle_alias =
      std::any_of(fake().mappings.begin(), fake().mappings.end(),
                  [&](const auto& pair) { return pair.second.handle == handle; });
  if (handle_alias) {
    fake().violation("one physical handle was mapped at two virtual addresses");
    return CUDA_ERROR_INVALID_VALUE;
  }
  bool inside_reservation = false;
  for (const auto& [base, reserved] : fake().reservations) {
    if (address >= base && address - base <= reserved && bytes <= reserved - (address - base)) {
      inside_reservation = true;
      break;
    }
  }
  if (!inside_reservation) {
    fake().violation("mapping fell outside the padded VA reservation");
    return CUDA_ERROR_INVALID_VALUE;
  }
  for (const auto& [base, mapping] : fake().mappings) {
    const bool disjoint = address + bytes <= base || base + mapping.bytes <= address;
    if (!disjoint) {
      fake().violation("physical mappings overlap in virtual address space");
      return CUDA_ERROR_INVALID_VALUE;
    }
  }
  fake().mappings.emplace(address, FakeCuda::Mapping{handle, bytes});
  ++allocation->second.map_count;
  fake().maximum_active_mappings = std::max(fake().maximum_active_mappings, fake().mappings.size());
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_mem_unmap(const DevicePointer address, const std::size_t bytes) {
  ++fake().unmap_calls;
  const auto mapping = fake().mappings.find(address);
  if (mapping == fake().mappings.end() || mapping->second.bytes != bytes) {
    fake().violation("unmap did not remove one exact full mapped chunk");
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (!mapping->second.access_set && mapping->second.work_started) {
    fake().violation("inaccessible mapping reached device work");
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (mapping->second.work_started &&
      (!mapping->second.completion_recorded || !mapping->second.completion_observed)) {
    fake().violation("mapping was unmapped before its completion event boundary");
    return CUDA_ERROR_NOT_READY;
  }
  if (fake().fault == FaultMode::fail_first_unmap && !fake().unmap_fault_used) {
    fake().unmap_fault_used = true;
    return CUDA_ERROR_UNKNOWN;
  }
  if (fake().injection.should_inject(FaultSite::unmap)) {
    fake().unmap_fault_used = true;
    return fake().injection.result;
  }
  fake().mappings.erase(mapping);
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_mem_set_access(const DevicePointer address, const std::size_t bytes,
                                   const CUmemAccessDesc* access, const std::size_t count) {
  ++fake().set_access_calls;
  if (fake().injection.should_inject(FaultSite::set_access)) {
    return fake().injection.result;
  }
  const auto mapping = fake().mappings.find(address);
  if (mapping == fake().mappings.end() || mapping->second.bytes != bytes || count != 1U ||
      access == nullptr || access->location.type != CU_MEM_LOCATION_TYPE_DEVICE ||
      access->location.id != 0 || access->flags != CU_MEM_ACCESS_FLAGS_PROT_READWRITE ||
      mapping->second.access_set) {
    fake().violation("SetAccess was missing, duplicated, or applied to the wrong mapping");
    return CUDA_ERROR_INVALID_VALUE;
  }
  mapping->second.access_set = true;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_mem_get_access(unsigned long long* flags, const CUmemLocation* location,
                                   const DevicePointer address) {
  const auto mapping = fake().mappings.find(address);
  if (flags == nullptr || location == nullptr || mapping == fake().mappings.end() ||
      !mapping->second.access_set || location->type != CU_MEM_LOCATION_TYPE_DEVICE ||
      location->id != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *flags = static_cast<unsigned long long>(CU_MEM_ACCESS_FLAGS_PROT_READWRITE);
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_mem_alloc(DevicePointer* address, const std::size_t bytes) {
  const DevicePointer base = fake().next_linear;
  fake().next_linear += static_cast<DevicePointer>(bytes + 4096U);
  FakeCuda::LinearAllocation allocation;
  allocation.storage.resize(bytes);
  fake().linear.emplace(base, std::move(allocation));
  *address = base;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_mem_free(const DevicePointer address) {
  if (fake().linear.erase(address) != 1U) {
    fake().violation("linear device allocation was freed more than once");
    return CUDA_ERROR_INVALID_VALUE;
  }
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_host_alloc(void** pointer, const std::size_t bytes, unsigned int) {
  if (fake().injection.should_inject(FaultSite::host_allocate)) {
    return fake().injection.result;
  }
  std::unique_ptr<std::byte[]> storage(new (std::nothrow) std::byte[bytes]);
  if (!storage) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  void* raw = storage.get();
  fake().pinned.emplace(raw, std::move(storage));
  *pointer = raw;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_host_free(void* pointer) {
  if (fake().pinned.erase(pointer) != 1U) {
    fake().violation("pinned staging allocation was freed more than once");
    return CUDA_ERROR_INVALID_VALUE;
  }
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_h2d(const DevicePointer destination, const void* source,
                        const std::size_t bytes, const Stream stream) {
  if (const auto view = fake().mapping_view(destination, bytes); view.has_value()) {
    if (fake().injection.should_inject(FaultSite::h2d)) {
      return fake().injection.result;
    }
    if (!view->mapping->access_set) {
      fake().violation("H2D targeted a mapping without SetAccess");
      return CUDA_ERROR_INVALID_VALUE;
    }
    std::memcpy(view->allocation->storage.data() + view->offset, source, bytes);
    fake().begin_mapping_work(view->base, stream);
    return CUDA_SUCCESS;
  }
  if (const auto view = fake().linear_view(destination, bytes); view.has_value()) {
    std::memcpy(view->allocation->storage.data() + view->offset, source, bytes);
    return CUDA_SUCCESS;
  }
  fake().violation("H2D referenced neither a mapping nor a linear allocation");
  return CUDA_ERROR_INVALID_VALUE;
}

Result CUDAAPI fake_d2h(void* destination, const DevicePointer source, const std::size_t bytes,
                        const Stream stream) {
  if (const auto view = fake().mapping_view(source, bytes); view.has_value()) {
    if (fake().injection.should_inject(FaultSite::d2h)) {
      return fake().injection.result;
    }
    if (!view->mapping->access_set) {
      fake().violation("D2H sourced a mapping without SetAccess");
      return CUDA_ERROR_INVALID_VALUE;
    }
    std::memcpy(destination, view->allocation->storage.data() + view->offset, bytes);
    fake().begin_mapping_work(view->base, stream);
    ++fake().d2h_mapping_calls;
    return CUDA_SUCCESS;
  }
  if (const auto view = fake().linear_view(source, bytes); view.has_value()) {
    std::memcpy(destination, view->allocation->storage.data() + view->offset, bytes);
    return CUDA_SUCCESS;
  }
  fake().violation("D2H referenced neither a mapping nor a linear allocation");
  return CUDA_ERROR_INVALID_VALUE;
}

Result CUDAAPI fake_stream_create(Stream* stream, unsigned int) {
  const Stream created = fake().make_token<Stream>();
  fake().streams.emplace(
      created, FakeCuda::StreamState{{}, fake().current_context, CU_STREAM_CAPTURE_STATUS_NONE});
  fake().stream_creation_order.push_back(created);
  *stream = created;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_stream_destroy(const Stream stream) {
  if (fake().streams.erase(stream) != 1U) {
    fake().violation("stream was destroyed more than once");
    return CUDA_ERROR_INVALID_HANDLE;
  }
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_stream_synchronize(const Stream stream) {
  return fake().streams.contains(stream) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_HANDLE;
}

Result CUDAAPI fake_stream_get_context(const Stream stream, Context* context) {
  const auto found = fake().streams.find(stream);
  if (context == nullptr || found == fake().streams.end()) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  *context = found->second.context;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_stream_wait_event(const Stream stream, const Event event, unsigned int) {
  return fake().streams.contains(stream) && fake().events.contains(event)
             ? CUDA_SUCCESS
             : CUDA_ERROR_INVALID_HANDLE;
}

Result CUDAAPI fake_stream_is_capturing(const Stream stream, CUstreamCaptureStatus* status) {
  const auto found = fake().streams.find(stream);
  if (status == nullptr || found == fake().streams.end()) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  *status = found->second.capture;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_event_create(Event* event, unsigned int) {
  const Event created = fake().make_token<Event>();
  fake().events.emplace(created, FakeCuda::EventState{});
  *event = created;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_event_destroy(const Event event) {
  if (fake().injection.should_inject(FaultSite::event_destroy)) {
    return fake().injection.result;
  }
  if (fake().events.erase(event) != 1U) {
    fake().violation("event was destroyed more than once");
    return CUDA_ERROR_INVALID_HANDLE;
  }
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_event_record(const Event event, const Stream stream) {
  if (fake().injection.should_inject(FaultSite::event_record)) {
    fake().event_fault_used = true;
    return fake().injection.result;
  }
  const auto event_state = fake().events.find(event);
  const auto stream_state = fake().streams.find(stream);
  if (event_state == fake().events.end() || stream_state == fake().streams.end()) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  event_state->second.recorded = true;
  event_state->second.completion_mapping = stream_state->second.pending_mapping;
  if (stream_state->second.pending_mapping.has_value()) {
    const auto mapping = fake().mappings.find(*stream_state->second.pending_mapping);
    if (mapping == fake().mappings.end()) {
      fake().violation("completion event referenced an unknown mapping");
      return CUDA_ERROR_INVALID_VALUE;
    }
    mapping->second.completion_recorded = true;
    stream_state->second.pending_mapping.reset();
  }
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_event_query(const Event event) {
  ++fake().event_query_calls;
  if (fake().fault == FaultMode::invalid_event_query && !fake().event_fault_used) {
    fake().event_fault_used = true;
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (fake().injection.should_inject(FaultSite::event_query)) {
    fake().event_fault_used = true;
    return fake().injection.result;
  }
  const auto state = fake().events.find(event);
  if (state == fake().events.end() || !state->second.recorded) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  fake().observe_event(state->second);
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_event_synchronize(const Event event) {
  const auto state = fake().events.find(event);
  if (state == fake().events.end() || !state->second.recorded) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  fake().observe_event(state->second);
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_event_elapsed(float* milliseconds, const Event start, const Event end) {
  if (!fake().events.contains(start) || !fake().events.contains(end)) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  *milliseconds = fake().event_elapsed_ms;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_module_load(Module* module, const void* image) {
  if (image == nullptr) {
    return CUDA_ERROR_INVALID_IMAGE;
  }
  *module = fake().make_token<Module>();
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_module_get_function(Function* function, const Module module, const char* name) {
  if (module == nullptr || name == nullptr ||
      (std::string_view(name) != "xvram_residency_workload_v1" &&
       std::string_view(name) != "xvram_lz4_verify_v1")) {
    return CUDA_ERROR_NOT_FOUND;
  }
  *function = fake().make_token<Function>();
  fake().codec_verification_functions.emplace(*function,
                                              std::string_view{name} == "xvram_lz4_verify_v1");
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_module_unload(Module) {
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_launch_kernel(Function function, const unsigned int blocks_x, unsigned int,
                                  unsigned int, const unsigned int threads_x, unsigned int,
                                  unsigned int, unsigned int, const Stream stream,
                                  void** parameters, void**) {
  ++fake().kernel_calls;
  if (fake().injection.should_inject(FaultSite::kernel)) {
    return fake().injection.result;
  }
  const auto function_kind = fake().codec_verification_functions.find(function);
  if (function_kind != fake().codec_verification_functions.end() && function_kind->second) {
    if (!fake().streams.contains(stream) || parameters == nullptr || blocks_x == 0U ||
        threads_x != 256U) {
      fake().violation("codec verification launch geometry or stream is invalid");
      return CUDA_ERROR_INVALID_VALUE;
    }
    const DevicePointer data = *static_cast<DevicePointer*>(parameters[0]);
    const std::uint64_t bytes = *static_cast<std::uint64_t*>(parameters[1]);
    const DevicePointer token_address = *static_cast<DevicePointer*>(parameters[2]);
    if (bytes > std::numeric_limits<std::size_t>::max()) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    const auto token_view =
        fake().linear_view(token_address, sizeof(xvram::residency::ContentToken));
    const auto mapping = fake().mapping_view(data, static_cast<std::size_t>(bytes));
    const auto linear = fake().linear_view(data, static_cast<std::size_t>(bytes));
    if (!token_view.has_value() || (!mapping.has_value() && !linear.has_value())) {
      fake().violation("codec verification referenced inaccessible memory");
      return CUDA_ERROR_INVALID_VALUE;
    }
    const std::byte* source = mapping.has_value()
                                  ? mapping->allocation->storage.data() + mapping->offset
                                  : linear->allocation->storage.data() + linear->offset;
    const xvram::residency::ContentToken token = xvram::residency::compute_content_token(
        std::span<const std::byte>{source, static_cast<std::size_t>(bytes)});
    std::memcpy(token_view->allocation->storage.data() + token_view->offset, &token, sizeof(token));
    if (mapping.has_value()) {
      fake().begin_mapping_work(mapping->base, stream);
    }
    return CUDA_SUCCESS;
  }
  if (!fake().streams.contains(stream) || parameters == nullptr || blocks_x != 8U ||
      threads_x == 0U || threads_x > 256U || threads_x % 32U != 0U) {
    fake().violation("kernel launch geometry or stream is invalid");
    return CUDA_ERROR_INVALID_VALUE;
  }
  const DevicePointer data = *static_cast<DevicePointer*>(parameters[0]);
  const std::uint64_t word_count = *static_cast<std::uint64_t*>(parameters[1]);
  const std::uint64_t global_start = *static_cast<std::uint64_t*>(parameters[2]);
  const std::uint32_t operation_index = *static_cast<std::uint32_t*>(parameters[3]);
  const std::uint64_t seed = *static_cast<std::uint64_t*>(parameters[4]);
  const std::uint32_t raw_mode = *static_cast<std::uint32_t*>(parameters[5]);
  const DevicePointer token_base = *static_cast<DevicePointer*>(parameters[6]);
  if (word_count > std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t) ||
      raw_mode > static_cast<std::uint32_t>(xvram::residency::AccessMode::write_only)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const std::size_t data_bytes = static_cast<std::size_t>(word_count * sizeof(std::uint32_t));
  const auto view = fake().mapping_view(data, data_bytes);
  const auto token = fake().linear_view(token_base + operation_index * sizeof(std::uint32_t),
                                        sizeof(std::uint32_t));
  if (!view.has_value() || !view->mapping->access_set || !token.has_value()) {
    fake().violation("kernel referenced inaccessible data or verification-token memory");
    return CUDA_ERROR_INVALID_VALUE;
  }
  const auto mode = static_cast<xvram::residency::AccessMode>(raw_mode);
  const auto stable = fake().stable_addresses.find(global_start);
  if (stable == fake().stable_addresses.end()) {
    fake().stable_addresses.emplace(global_start, data);
  } else if (stable->second != data) {
    fake().violation("a logical chunk changed its stable virtual address");
    return CUDA_ERROR_INVALID_ADDRESS_SPACE;
  }

  std::uint32_t local_token = 0;
  for (std::uint64_t index = 0; index < word_count; ++index) {
    const std::size_t byte_offset =
        view->offset + static_cast<std::size_t>(index * sizeof(std::uint32_t));
    std::uint32_t word = 0;
    std::memcpy(&word, view->allocation->storage.data() + byte_offset, sizeof(word));
    const std::uint64_t global_index = global_start + index;
    const std::uint32_t output =
        xvram::residency::transform_word(word, global_index, operation_index, seed, mode);
    if (mode != xvram::residency::AccessMode::read) {
      std::memcpy(view->allocation->storage.data() + byte_offset, &output, sizeof(output));
    }
    local_token ^= output ^ xvram::residency::operation_mix(global_index, operation_index, seed);
  }
  std::uint32_t accumulated_token = 0;
  std::memcpy(&accumulated_token, token->allocation->storage.data() + token->offset,
              sizeof(accumulated_token));
  accumulated_token ^= local_token;
  if (fake().injection.should_inject(FaultSite::corrupt_kernel_output)) {
    accumulated_token ^= 1U;
  }
  std::memcpy(token->allocation->storage.data() + token->offset, &accumulated_token,
              sizeof(accumulated_token));
  fake().begin_mapping_work(view->base, stream);
  return CUDA_SUCCESS;
}

nvcompStatus_t fake_nvcomp_properties(nvcompProperties_t* properties) {
  properties->version = 5300;
  properties->cudart_version = 13030;
  return nvcompSuccess;
}

const char* fake_nvcomp_status_string(nvcompStatus_t) {
  return "fake";
}

nvcompStatus_t fake_nvcomp_compress_alignment(nvcompBatchedLZ4CompressOpts_t,
                                              nvcompAlignmentRequirements_t* alignment) {
  *alignment = {1, 1, 1};
  return nvcompSuccess;
}

nvcompStatus_t fake_nvcomp_decompress_alignment(nvcompBatchedLZ4DecompressOpts_t,
                                                nvcompAlignmentRequirements_t* alignment) {
  *alignment = {1, 1, 1};
  return nvcompSuccess;
}

nvcompStatus_t fake_nvcomp_compress_temp(std::size_t, std::size_t, nvcompBatchedLZ4CompressOpts_t,
                                         std::size_t* bytes, std::size_t) {
  *bytes = 1024;
  return nvcompSuccess;
}

nvcompStatus_t fake_nvcomp_decompress_temp(std::size_t, std::size_t,
                                           nvcompBatchedLZ4DecompressOpts_t, std::size_t* bytes,
                                           std::size_t) {
  *bytes = 1024;
  return nvcompSuccess;
}

nvcompStatus_t fake_nvcomp_max_output(const std::size_t bytes, nvcompBatchedLZ4CompressOpts_t,
                                      std::size_t* output) {
  *output = bytes;
  return nvcompSuccess;
}

nvcompStatus_t fake_nvcomp_compress(const void* const*, const std::size_t*, std::size_t,
                                    std::size_t, void*, std::size_t, void* const*, std::size_t*,
                                    nvcompBatchedLZ4CompressOpts_t, nvcompStatus_t*, cudaStream_t) {
  return nvcompErrorInvalidValue;
}

nvcompStatus_t fake_nvcomp_compress_raw_success(const void* const* inputs,
                                                const std::size_t* input_sizes, std::size_t,
                                                const std::size_t count, void*, std::size_t,
                                                void* const* outputs, std::size_t* output_sizes,
                                                nvcompBatchedLZ4CompressOpts_t,
                                                nvcompStatus_t* statuses, cudaStream_t) {
  const auto input_pointers = fake().linear_view(
      static_cast<DevicePointer>(reinterpret_cast<std::uintptr_t>(inputs)), count * sizeof(void*));
  const auto sizes =
      fake().linear_view(static_cast<DevicePointer>(reinterpret_cast<std::uintptr_t>(input_sizes)),
                         count * sizeof(std::size_t));
  const auto output_pointers = fake().linear_view(
      static_cast<DevicePointer>(reinterpret_cast<std::uintptr_t>(outputs)), count * sizeof(void*));
  const auto actual_sizes =
      fake().linear_view(static_cast<DevicePointer>(reinterpret_cast<std::uintptr_t>(output_sizes)),
                         count * sizeof(std::size_t));
  const auto codec_statuses =
      fake().linear_view(static_cast<DevicePointer>(reinterpret_cast<std::uintptr_t>(statuses)),
                         count * sizeof(nvcompStatus_t));
  if (!input_pointers.has_value() || !sizes.has_value() || !output_pointers.has_value() ||
      !actual_sizes.has_value() || !codec_statuses.has_value()) {
    return nvcompErrorInvalidValue;
  }
  for (std::size_t index = 0; index < count; ++index) {
    const std::size_t pointer_offset = index * sizeof(void*);
    const std::size_t size_offset = index * sizeof(std::size_t);
    const std::size_t status_offset = index * sizeof(nvcompStatus_t);
    const void* input_pointer = nullptr;
    void* output_pointer = nullptr;
    std::size_t input_bytes = 0;
    std::memcpy(&input_pointer,
                input_pointers->allocation->storage.data() + input_pointers->offset +
                    pointer_offset,
                sizeof(input_pointer));
    std::memcpy(&output_pointer,
                output_pointers->allocation->storage.data() + output_pointers->offset +
                    pointer_offset,
                sizeof(output_pointer));
    std::memcpy(&input_bytes, sizes->allocation->storage.data() + sizes->offset + size_offset,
                sizeof(input_bytes));
    const DevicePointer input_address =
        static_cast<DevicePointer>(reinterpret_cast<std::uintptr_t>(input_pointer));
    const DevicePointer output_address =
        static_cast<DevicePointer>(reinterpret_cast<std::uintptr_t>(output_pointer));
    const auto input = fake().mapping_view(input_address, input_bytes);
    const auto output = fake().linear_view(output_address, input_bytes);
    if (!input.has_value() || !output.has_value()) {
      return nvcompErrorInvalidValue;
    }
    std::memcpy(output->allocation->storage.data() + output->offset,
                input->allocation->storage.data() + input->offset, input_bytes);
    std::memcpy(actual_sizes->allocation->storage.data() + actual_sizes->offset + size_offset,
                &input_bytes, sizeof(input_bytes));
    const nvcompStatus_t success = nvcompSuccess;
    std::memcpy(codec_statuses->allocation->storage.data() + codec_statuses->offset + status_offset,
                &success, sizeof(success));
  }
  return nvcompSuccess;
}

nvcompStatus_t fake_nvcomp_decompress(const void* const*, const std::size_t*, const std::size_t*,
                                      std::size_t*, std::size_t, void*, std::size_t, void* const*,
                                      nvcompBatchedLZ4DecompressOpts_t, nvcompStatus_t*,
                                      cudaStream_t) {
  return nvcompErrorInvalidValue;
}

[[nodiscard]] xvram::nvcomp::NvcompDispatch make_fake_nvcomp_dispatch() {
  return {
      fake_nvcomp_properties,           fake_nvcomp_status_string,   fake_nvcomp_compress_alignment,
      fake_nvcomp_compress_temp,        fake_nvcomp_max_output,      fake_nvcomp_compress,
      fake_nvcomp_decompress_alignment, fake_nvcomp_decompress_temp, fake_nvcomp_decompress};
}

[[nodiscard]] xvram::nvcomp::NvcompDispatch make_successful_fake_nvcomp_dispatch() {
  return {fake_nvcomp_properties,
          fake_nvcomp_status_string,
          fake_nvcomp_compress_alignment,
          fake_nvcomp_compress_temp,
          fake_nvcomp_max_output,
          fake_nvcomp_compress_raw_success,
          fake_nvcomp_decompress_alignment,
          fake_nvcomp_decompress_temp,
          fake_nvcomp_decompress};
}

void FakeCuda::install(CudaApi& api) {
  api.init_ = fake_init;
  api.driver_get_version_ = fake_driver_version;
  api.get_error_name_ = fake_error_name;
  api.get_error_string_ = fake_error_string;
  api.device_get_count_ = fake_device_count;
  api.device_get_ = fake_device_get;
  api.device_get_name_ = fake_device_name;
  api.device_total_memory_ = fake_device_total_memory;
  api.device_get_attribute_ = fake_device_attribute;
  api.mem_get_allocation_granularity_ = fake_granularity;
  api.context_get_current_ = fake_context_get_current;
  api.context_get_device_ = fake_context_get_device;
  api.context_set_current_ = fake_context_set_current;
  api.context_push_current_ = fake_context_push_current;
  api.context_pop_current_ = fake_context_pop_current;
  api.context_create_ = fake_context_create;
  api.context_destroy_ = fake_context_destroy;
  api.mem_get_info_ = fake_mem_get_info;
  api.mem_address_reserve_ = fake_address_reserve;
  api.mem_address_free_ = fake_address_free;
  api.mem_create_ = fake_mem_create;
  api.mem_release_ = fake_mem_release;
  api.mem_map_ = fake_mem_map;
  api.mem_unmap_ = fake_mem_unmap;
  api.mem_set_access_ = fake_mem_set_access;
  api.mem_get_access_ = fake_mem_get_access;
  api.mem_alloc_ = fake_mem_alloc;
  api.mem_free_ = fake_mem_free;
  api.mem_host_alloc_ = fake_host_alloc;
  api.mem_free_host_ = fake_host_free;
  api.memcpy_h2d_async_ = fake_h2d;
  api.memcpy_d2h_async_ = fake_d2h;
  api.stream_create_ = fake_stream_create;
  api.stream_destroy_ = fake_stream_destroy;
  api.stream_synchronize_ = fake_stream_synchronize;
  api.stream_get_context_ = fake_stream_get_context;
  api.stream_wait_event_ = fake_stream_wait_event;
  api.stream_is_capturing_ = fake_stream_is_capturing;
  api.event_create_ = fake_event_create;
  api.event_destroy_ = fake_event_destroy;
  api.event_record_ = fake_event_record;
  api.event_query_ = fake_event_query;
  api.event_synchronize_ = fake_event_synchronize;
  api.event_elapsed_time_ = fake_event_elapsed;
  api.module_load_data_ = fake_module_load;
  api.module_get_function_ = fake_module_get_function;
  api.module_unload_ = fake_module_unload;
  api.launch_kernel_ = fake_launch_kernel;
}

[[nodiscard]] xvram::residency::ExecutorOptions options_for_test() {
  xvram::residency::ExecutorOptions options;
  options.logical_bytes = logical_bytes;
  options.chunk_bytes = chunk_bytes;
  options.cache_target_bytes = 512ULL * kib;
  options.staging_slots = 4;
  options.policy = xvram::residency::RequestedPolicy::clock;
  options.prefetch_distance = 0;
  options.scenario = xvram::residency::ScenarioKind::sequential;
  options.passes = 2;
  options.device_headroom_bytes = chunk_bytes;
  options.budget_poll_interval = std::chrono::milliseconds(1);
  options.stall_timeout = std::chrono::milliseconds(100);
  options.progress_heartbeat = std::chrono::milliseconds(10);
  options.trace_enabled = true;
  return options;
}

[[nodiscard]] xvram::residency::ExecutorEnvironment
environment_for_test(const bool dynamic_pressure) {
  xvram::residency::ExecutorEnvironment environment;
  environment.physical_host_bytes = 16ULL * gib;
  environment.available_host_bytes = 12ULL * gib;
  environment.initial_device_budget = xvram::residency::BudgetSnapshot{16ULL * mib, 0};
  if (dynamic_pressure) {
    environment.query_device_budget = []() -> std::optional<xvram::residency::BudgetSnapshot> {
      return xvram::residency::BudgetSnapshot{chunk_bytes, chunk_bytes};
    };
  } else {
    environment.query_device_budget = []() -> std::optional<xvram::residency::BudgetSnapshot> {
      return xvram::residency::BudgetSnapshot{16ULL * mib, 0};
    };
  }
  return environment;
}

struct Run {
  FakeCuda cuda;
  std::vector<xvram::residency::TraceRecord> trace;
  xvram::residency::ExecutorResult result;
};

[[nodiscard]] Run execute(const FaultMode fault = FaultMode::none,
                          const bool dynamic_pressure = false, FaultInjection injection = {}) {
  Run run;
  run.cuda.fault = fault;
  run.cuda.injection = injection;
  CudaApi api(CudaApi::InjectedDispatch{});
  run.cuda.install(api);
  const ActiveFake active(run.cuda);
  const auto options = options_for_test();
  const auto environment = environment_for_test(dynamic_pressure);
  const auto trace = [&](const xvram::residency::TraceRecord& record) {
    run.trace.push_back(record);
  };
  run.result = xvram::residency::run_executor(api, options, {}, trace, &environment);
  return run;
}

[[nodiscard]] bool schema_trace_event(const std::string_view event) {
  static constexpr std::array<std::string_view, 20> events{
      "state_transition",    "cache_hit",          "cache_miss",       "prefetch_issued",
      "prefetch_retired",    "prefetch_cancelled", "target_changed",   "operation_retired",
      "transaction_retired", "budget_sample",      "diagnostic",       "allocation",
      "kernel_submit",       "kernel_retire",      "prefetch_useful",  "victim_select",
      "victim_selected",     "writeback_prepare",  "pressure_acquire", "pressure_release"};
  return std::find(events.begin(), events.end(), event) != events.end();
}

[[nodiscard]] bool schema_chunk_state(const std::string_view state) {
  static constexpr std::array<std::string_view, 10> states{
      "host_clean",     "prefetch_queued",  "mapping",       "h2d_in_flight", "resident_clean",
      "resident_dirty", "writeback_queued", "d2h_in_flight", "evicting",      "poisoned"};
  return std::find(states.begin(), states.end(), state) != states.end();
}

void check_trace_contract(const Run& run) {
  CHECK(!run.trace.empty());
  CHECK(run.result.telemetry.trace_records_emitted.value_or(0) == run.trace.size());
  CHECK(run.result.telemetry.trace_records_dropped.value_or(1) == 0);
  CHECK(run.result.telemetry.trace_complete.value_or(false));
  for (std::size_t index = 0; index < run.trace.size(); ++index) {
    const auto& record = run.trace[index];
    CHECK(record.schema_version == 1U);
    CHECK(record.report_type == "xvram.residency_trace");
    CHECK(schema_trace_event(record.event));
    if (index != 0U) {
      CHECK(record.sequence == run.trace[index - 1U].sequence + 1U);
      CHECK(record.monotonic_time_ns >= run.trace[index - 1U].monotonic_time_ns);
    }
    const auto lacks_raw_address = [](const std::optional<std::string>& value) {
      return !value.has_value() || value->find("0x") == std::string::npos;
    };
    CHECK(record.event.find("0x") == std::string::npos);
    CHECK(lacks_raw_address(record.reason));
    CHECK(lacks_raw_address(record.policy));
    CHECK(lacks_raw_address(record.from_state));
    CHECK(lacks_raw_address(record.to_state));
    if (record.event == "state_transition") {
      CHECK(record.allocation_id.has_value());
      CHECK(record.chunk_index.has_value());
      CHECK(record.from_state.has_value() && schema_chunk_state(*record.from_state));
      CHECK(record.to_state.has_value() && schema_chunk_state(*record.to_state));
      CHECK(record.generation.has_value());
    }
  }
}

void completed_sequential_test() {
  Run run = execute();
  CHECK(run.result.exit_code == 0);
  CHECK(run.result.status == "completed");
  CHECK(run.result.cleanup.complete.value_or(false));
  CHECK(run.result.workloads.size() == 1);
  CHECK(run.result.workloads.front().status == "completed");
  CHECK(run.result.workloads.front().matches_cpu.value_or(false));
  CHECK(run.result.cache.mapping_count.has_value());
  CHECK(run.result.cache.mapping_count == run.result.cache.set_access_count);
  CHECK(run.result.cache.mapping_count == run.result.cache.unmap_count);
  CHECK(run.result.cache.unmap_count == run.result.cache.event_boundary_count);
  CHECK(run.result.cache.handle_reuse_count.value_or(0) > 0);
  CHECK(run.result.cache.dirty_evictions.value_or(0) > 0);
  CHECK(run.result.cache.writebacks_completed == run.result.cache.dirty_evictions);
  CHECK(run.result.cache.unsafe_remap_count == 0);
  CHECK(run.result.cache.unsafe_transition_count == 0);
  CHECK(run.result.proof.logical_data_exceeds_vram.value_or(false));
  CHECK(run.result.proof.cache_smaller_than_logical.value_or(false));
  CHECK(run.result.proof.handles_reused.value_or(false));
  CHECK(run.result.proof.stable_virtual_addresses_verified.value_or(false));
  CHECK(run.result.proof.set_access_after_map_verified.value_or(false));
  CHECK(run.result.proof.event_boundaries_verified.value_or(false));
  CHECK(run.result.proof.no_physical_aliases_verified.value_or(false));
  CHECK(run.result.proof.dirty_writeback_verified.value_or(false));
  CHECK(run.result.proof.all_workloads_match_cpu.value_or(false));
  CHECK(run.cuda.violations.empty());
  CHECK(run.cuda.map_calls > 0);
  CHECK(run.cuda.map_calls == run.cuda.set_access_calls);
  CHECK(run.cuda.map_calls == run.cuda.unmap_calls);
  CHECK(run.cuda.d2h_mapping_calls > 0);
  CHECK(run.cuda.maximum_active_mappings <= 7U);
  CHECK(run.cuda.stable_addresses.size() == logical_bytes / chunk_bytes);
  CHECK(run.cuda.reservations.empty());
  CHECK(run.cuda.mappings.empty());
  CHECK(run.cuda.linear.empty());
  CHECK(run.cuda.pinned.empty());
  CHECK(run.cuda.streams.empty());
  CHECK(run.cuda.events.empty());
  CHECK(std::all_of(run.cuda.physical.begin(), run.cuda.physical.end(), [](const auto& pair) {
    return pair.second.released && pair.second.map_count > 1;
  }));
  check_trace_contract(run);
}

void runtime_allocation_release_unmaps_test() {
  const CheckContext context("runtime allocation release unmaps");
  FakeCuda cuda;
  CudaApi api(CudaApi::InjectedDispatch{});
  cuda.install(api);
  const ActiveFake active(cuda);

  xvram::residency::RuntimeConfig config;
  config.chunk_bytes = chunk_bytes;
  config.cache_target_bytes = 8ULL * chunk_bytes;
  config.device_headroom_bytes = chunk_bytes;
  config.staging_slots = 2;
  config.stall_timeout = std::chrono::milliseconds(100);
  config.budget_poll_interval = std::chrono::milliseconds(1);
  std::uint64_t lifecycle_progress = 0;
  config.lifecycle_progress = [&]() { ++lifecycle_progress; };
  xvram::residency::Runtime runtime(api, config);
  CHECK(runtime.setup() == xvram::residency::RuntimeStatus::success);

  xvram::residency::RuntimeAllocation allocation;
  CHECK(runtime.allocate(chunk_bytes, xvram::residency::ResidencyHint::normal, allocation) ==
        xvram::residency::RuntimeStatus::success);
  CHECK(lifecycle_progress == 1U);
  const std::array access{xvram::residency::AccessRange{allocation.id, 0, chunk_bytes,
                                                        xvram::residency::AccessMode::read}};
  CHECK(runtime.execute(access, 0, [](const xvram::residency::TransactionContext& transaction) {
    return transaction.ranges.size() == 1U && transaction.ranges.front().device_address != 0
               ? xvram::residency::RuntimeStatus::success
               : xvram::residency::RuntimeStatus::internal_failure;
  }) == xvram::residency::RuntimeStatus::success);
  CHECK(runtime.telemetry().maps == 1U);
  CHECK(runtime.telemetry().unmaps == 0U);
  CHECK(runtime.telemetry().resident_bytes == chunk_bytes);

  CHECK(runtime.release(allocation.id) == xvram::residency::RuntimeStatus::success);
  CHECK(runtime.telemetry().allocations_released == 1U);
  CHECK(runtime.telemetry().maps == runtime.telemetry().unmaps);
  CHECK(runtime.telemetry().event_boundaries == runtime.telemetry().unmaps);
  CHECK(runtime.telemetry().unsafe_remaps == 0U);
  CHECK(runtime.telemetry().resident_bytes == 0U);
  CHECK(lifecycle_progress == 2U);
  CHECK(cuda.mappings.empty());
  CHECK(cuda.reservations.empty());

  CHECK(runtime.close() == xvram::residency::RuntimeStatus::success);
  CHECK(cuda.violations.empty());
  CHECK(cuda.linear.empty());
  CHECK(cuda.pinned.empty());
  CHECK(cuda.streams.empty());
  CHECK(cuda.events.empty());
  CHECK(std::all_of(cuda.physical.begin(), cuda.physical.end(),
                    [](const auto& pair) { return pair.second.released; }));
}

void runtime_disabled_compression_never_loads_nvcomp_test() {
  const CheckContext context("disabled compression never loads nvCOMP");
  FakeCuda cuda;
  CudaApi api(CudaApi::InjectedDispatch{});
  cuda.install(api);
  const ActiveFake active(cuda);
  xvram::nvcomp::NvcompApi untouched_codec;

  xvram::residency::RuntimeConfig config;
  config.chunk_bytes = chunk_bytes;
  config.cache_target_bytes = 4ULL * chunk_bytes;
  config.device_headroom_bytes = chunk_bytes;
  config.staging_slots = 2;
  config.stall_timeout = std::chrono::milliseconds(100);
  config.compression_mode = xvram::residency::CompressionMode::disabled;
  config.nvcomp_api = &untouched_codec;
  xvram::residency::Runtime runtime(api, config);
  CHECK(runtime.setup() == xvram::residency::RuntimeStatus::success);
  CHECK(!runtime.telemetry().nvcomp_available);
  CHECK(runtime.telemetry().gpu_codec_fallbacks == 0U);
  CHECK(runtime.telemetry().codec_slots_created == 0U);
  CHECK(runtime.telemetry().codec_workspace_bytes == 0U);
  CHECK(runtime.close() == xvram::residency::RuntimeStatus::success);

  // A default NvcompApi only sets load_attempted_ inside load().  The deliberately invalid
  // relative path makes this a side-effect-free probe: attempted_now proves the disabled/v1
  // runtime never touched the injected codec object during setup or cleanup.
  xvram::nvcomp::NvcompLoadOptions probe;
  probe.library = "relative-nvcomp-must-not-be-opened";
  const xvram::nvcomp::NvcompLoadResult result = untouched_codec.load(probe);
  CHECK(result.attempted_now);
  CHECK(result.status == xvram::nvcomp::NvcompLoadStatus::invalid_path);
  CHECK(cuda.linear.empty());
  CHECK(cuda.mappings.empty());
  CHECK(cuda.reservations.empty());
  CHECK(cuda.violations.empty());
}

void runtime_raw_transfer_trace_generations_test() {
  const CheckContext context("runtime raw transfer trace generations");
  FakeCuda cuda;
  CudaApi api(CudaApi::InjectedDispatch{});
  cuda.install(api);
  const ActiveFake active(cuda);

  std::vector<xvram::residency::CompressionTraceEvent> trace;
  xvram::residency::RuntimeConfig config;
  config.chunk_bytes = chunk_bytes;
  config.cache_target_bytes = 8ULL * chunk_bytes;
  config.device_headroom_bytes = chunk_bytes;
  config.staging_slots = 2;
  config.stall_timeout = std::chrono::milliseconds(100);
  config.compression_trace = [&](const xvram::residency::CompressionTraceEvent& event) {
    trace.push_back(event);
    if (trace.size() == 1U) {
      throw 1;
    }
  };
  xvram::residency::Runtime runtime(api, config);
  CHECK(runtime.setup() == xvram::residency::RuntimeStatus::success);

  xvram::residency::RuntimeAllocation allocation;
  CHECK(runtime.allocate(chunk_bytes, xvram::residency::ResidencyHint::normal, allocation) ==
        xvram::residency::RuntimeStatus::success);
  const std::array read{xvram::residency::AccessRange{allocation.id, 0, chunk_bytes,
                                                      xvram::residency::AccessMode::read}};
  for (std::uint32_t cycle = 0; cycle < 3U; ++cycle) {
    CHECK(runtime.execute(read, 0, [](const auto&) {
      return xvram::residency::RuntimeStatus::success;
    }) == xvram::residency::RuntimeStatus::success);
    CHECK(runtime.drain(true) == xvram::residency::RuntimeStatus::success);
  }
  const std::array write_only{xvram::residency::AccessRange{
      allocation.id, 0, chunk_bytes, xvram::residency::AccessMode::write_only}};
  CHECK(runtime.execute(write_only, 0, [](const auto&) {
    return xvram::residency::RuntimeStatus::success;
  }) == xvram::residency::RuntimeStatus::success);
  CHECK(runtime.release(allocation.id) == xvram::residency::RuntimeStatus::success);

  CHECK(trace.size() == 8U);
  const std::array<std::uint64_t, 3> expected_h2d_slot_generations{1U, 2U, 3U};
  std::uint64_t previous_operation = 0;
  for (std::size_t cycle = 0; cycle < expected_h2d_slot_generations.size(); ++cycle) {
    const auto& submitted = trace[cycle * 2U];
    const auto& retired = trace[cycle * 2U + 1U];
    CHECK(submitted.kind == xvram::residency::CompressionTraceEventKind::h2d_submit);
    CHECK(retired.kind == xvram::residency::CompressionTraceEventKind::h2d_retire);
    CHECK((submitted.key == xvram::residency::ChunkKey{allocation.id, 0}));
    CHECK(submitted.key == retired.key);
    CHECK(submitted.operation_id != 0U);
    CHECK(submitted.operation_id > previous_operation);
    CHECK(submitted.operation_id == retired.operation_id);
    CHECK(submitted.source_generation == retired.source_generation);
    CHECK(submitted.slot_generation == expected_h2d_slot_generations[cycle]);
    CHECK(submitted.slot_generation == retired.slot_generation);
    CHECK(submitted.path == xvram::residency::CompressionPath::raw);
    CHECK(submitted.from_representation == xvram::residency::BackingRepresentation::raw);
    CHECK(!submitted.target_generation.has_value());
    CHECK(submitted.logical_bytes == chunk_bytes);
    CHECK(submitted.physical_bytes == chunk_bytes);
    CHECK(!submitted.speculative);
    previous_operation = submitted.operation_id;
  }
  const auto& d2h_submitted = trace[6];
  const auto& d2h_retired = trace[7];
  CHECK(d2h_submitted.kind == xvram::residency::CompressionTraceEventKind::d2h_submit);
  CHECK(d2h_retired.kind == xvram::residency::CompressionTraceEventKind::d2h_retire);
  CHECK((d2h_submitted.key == xvram::residency::ChunkKey{allocation.id, 0}));
  CHECK(d2h_submitted.key == d2h_retired.key);
  CHECK(d2h_submitted.operation_id == d2h_retired.operation_id);
  CHECK(d2h_submitted.source_generation == d2h_retired.source_generation);
  CHECK(d2h_submitted.target_generation == d2h_retired.target_generation);
  CHECK(d2h_submitted.slot_generation == 1U);
  CHECK(d2h_submitted.slot_generation == d2h_retired.slot_generation);
  CHECK(d2h_submitted.path == xvram::residency::CompressionPath::raw);
  CHECK(d2h_submitted.to_representation == xvram::residency::BackingRepresentation::raw);
  CHECK(d2h_submitted.logical_bytes == chunk_bytes);
  CHECK(d2h_submitted.physical_bytes == chunk_bytes);
  CHECK(d2h_submitted.target_generation == d2h_submitted.source_generation + 1U);
  CHECK(runtime.close() == xvram::residency::RuntimeStatus::success);
  CHECK(cuda.violations.empty());
}

void runtime_safe_callback_skip_keeps_session_reusable_test() {
  const CheckContext context("runtime safe callback skip keeps session reusable");
  FakeCuda cuda;
  CudaApi api(CudaApi::InjectedDispatch{});
  cuda.install(api);
  const ActiveFake active(cuda);

  xvram::residency::RuntimeConfig config;
  config.chunk_bytes = chunk_bytes;
  config.cache_target_bytes = 4ULL * chunk_bytes;
  config.device_headroom_bytes = chunk_bytes;
  config.staging_slots = 2;
  config.stall_timeout = std::chrono::milliseconds(100);
  xvram::residency::Runtime runtime(api, config);
  CHECK(runtime.setup() == xvram::residency::RuntimeStatus::success);
  xvram::residency::RuntimeAllocation allocation;
  CHECK(runtime.allocate(chunk_bytes, xvram::residency::ResidencyHint::normal, allocation) ==
        xvram::residency::RuntimeStatus::success);
  const std::array access{xvram::residency::AccessRange{allocation.id, 0, chunk_bytes,
                                                        xvram::residency::AccessMode::read}};

  CHECK(runtime.execute(access, 0, [](const auto&) {
    return xvram::residency::RuntimeStatus::callback_skipped;
  }) == xvram::residency::RuntimeStatus::callback_skipped);
  CHECK(!runtime.poisoned());
  CHECK(!runtime.telemetry().quarantined);
  CHECK(runtime.telemetry().transactions_completed == 0U);

  CHECK(runtime.execute(access, 0, [](const auto&) {
    return xvram::residency::RuntimeStatus::success;
  }) == xvram::residency::RuntimeStatus::success);
  CHECK(runtime.telemetry().transactions_completed == 1U);
  CHECK(runtime.close() == xvram::residency::RuntimeStatus::success);
  CHECK(cuda.violations.empty());
}

void runtime_external_lease_and_discard_dead_test() {
  const CheckContext context("runtime external lease and dead allocation discard");
  FakeCuda cuda;
  CudaApi api(CudaApi::InjectedDispatch{});
  cuda.install(api);
  const ActiveFake active(cuda);

  xvram::residency::RuntimeConfig config;
  config.chunk_bytes = chunk_bytes;
  config.cache_target_bytes = 4ULL * chunk_bytes;
  config.device_headroom_bytes = chunk_bytes;
  config.staging_slots = 2;
  config.stall_timeout = std::chrono::milliseconds(100);
  xvram::residency::Runtime runtime(api, config);
  CHECK(runtime.setup() == xvram::residency::RuntimeStatus::success);

  CHECK(cuda.stream_creation_order.size() >= 3U);
  const Stream runtime_compute_stream = cuda.stream_creation_order[1];
  xvram::residency::RuntimeAllocation allocation;
  CHECK(runtime.allocate(chunk_bytes, xvram::residency::ResidencyHint::normal, allocation) ==
        xvram::residency::RuntimeStatus::success);
  const std::array read_write{xvram::residency::AccessRange{
      allocation.id, 0, chunk_bytes, xvram::residency::AccessMode::read_write}};

  const Context original_stream_context = cuda.streams.at(runtime_compute_stream).context;
  cuda.streams.at(runtime_compute_stream).context = nullptr;
  xvram::residency::ExternalLease rejected;
  CHECK(runtime.acquire_external({read_write, 0}, rejected) ==
        xvram::residency::RuntimeStatus::invalid_argument);
  cuda.streams.at(runtime_compute_stream).context = original_stream_context;
  cuda.streams.at(runtime_compute_stream).capture = CU_STREAM_CAPTURE_STATUS_ACTIVE;
  CHECK(runtime.acquire_external({read_write, 0}, rejected) ==
        xvram::residency::RuntimeStatus::unsupported);
  cuda.streams.at(runtime_compute_stream).capture = CU_STREAM_CAPTURE_STATUS_NONE;

  xvram::residency::ExternalLease lease;
  CHECK(runtime.acquire_external({read_write, 0}, lease) ==
        xvram::residency::RuntimeStatus::success);
  CHECK(lease.id);
  CHECK(lease.stream == runtime_compute_stream);
  CHECK(lease.ranges.size() == 1U);
  CHECK(lease.ranges.front().device_address != 0U);
  const DevicePointer stable_address = lease.ranges.front().device_address;
  xvram::residency::ExternalLeasePoll poll;
  CHECK(runtime.poll_external(lease.id, poll) == xvram::residency::RuntimeStatus::success);
  CHECK(poll.state == xvram::residency::ExternalLeaseState::armed);
  CHECK(runtime.release(allocation.id) == xvram::residency::RuntimeStatus::invalid_argument);
  CHECK(runtime.discard_dead(allocation.id, 0, chunk_bytes) ==
        xvram::residency::RuntimeStatus::invalid_argument);
  xvram::residency::ExternalLease overlapping;
  CHECK(runtime.acquire_external({read_write, 0}, overlapping) ==
        xvram::residency::RuntimeStatus::invalid_argument);

  CHECK(runtime.seal_external(lease.id, xvram::residency::ExternalSealMode::success) ==
        xvram::residency::RuntimeStatus::success);
  CHECK(runtime.wait_external(lease.id, std::chrono::milliseconds(100), poll) ==
        xvram::residency::RuntimeStatus::success);
  CHECK(poll.state == xvram::residency::ExternalLeaseState::completed);
  CHECK(runtime.telemetry().transactions_completed == 1U);
  const std::uint64_t d2h_before_discard = cuda.d2h_mapping_calls;
  CHECK(runtime.discard_dead(allocation.id, 1, chunk_bytes - 1U) ==
        xvram::residency::RuntimeStatus::invalid_argument);
  CHECK(runtime.discard_dead(allocation.id) == xvram::residency::RuntimeStatus::success);
  CHECK(cuda.d2h_mapping_calls == d2h_before_discard);
  CHECK(cuda.mappings.empty());
  CHECK(cuda.reservations.size() == 1U);
  CHECK(runtime.allocation_size(allocation.id).has_value());
  std::byte byte{};
  CHECK(runtime.read(allocation.id, 0, &byte, 1) ==
        xvram::residency::RuntimeStatus::invalid_argument);
  CHECK(runtime.acquire_external({read_write, 0}, rejected) ==
        xvram::residency::RuntimeStatus::invalid_argument);

  const std::array rewrite{xvram::residency::AccessRange{allocation.id, 0, chunk_bytes,
                                                         xvram::residency::AccessMode::write_only}};
  xvram::residency::ExternalLease rewritten;
  CHECK(runtime.acquire_external({rewrite, 0}, rewritten) ==
        xvram::residency::RuntimeStatus::success);
  CHECK(rewritten.ranges.front().device_address == stable_address);
  CHECK(runtime.seal_external(rewritten.id, xvram::residency::ExternalSealMode::success) ==
        xvram::residency::RuntimeStatus::success);
  CHECK(runtime.wait_external(rewritten.id, std::chrono::milliseconds(100), poll) ==
        xvram::residency::RuntimeStatus::success);
  const std::array resident_read{
      xvram::residency::AccessRange{allocation.id, 0, 1, xvram::residency::AccessMode::read}};
  xvram::residency::ExternalLease read_after_rewrite;
  CHECK(runtime.acquire_external({resident_read, 0}, read_after_rewrite) ==
        xvram::residency::RuntimeStatus::success);
  CHECK(runtime.seal_external(read_after_rewrite.id, xvram::residency::ExternalSealMode::success) ==
        xvram::residency::RuntimeStatus::success);
  CHECK(runtime.wait_external(read_after_rewrite.id, std::chrono::milliseconds(100), poll) ==
        xvram::residency::RuntimeStatus::success);
  CHECK(runtime.discard_dead(allocation.id, 0, chunk_bytes) ==
        xvram::residency::RuntimeStatus::success);
  CHECK(cuda.d2h_mapping_calls == d2h_before_discard);
  CHECK(runtime.release(allocation.id) == xvram::residency::RuntimeStatus::success);
  CHECK(cuda.reservations.empty());

  xvram::residency::RuntimeAllocation cancelled_allocation;
  CHECK(runtime.allocate(chunk_bytes, xvram::residency::ResidencyHint::normal,
                         cancelled_allocation) == xvram::residency::RuntimeStatus::success);
  const std::array write_only{xvram::residency::AccessRange{
      cancelled_allocation.id, 0, chunk_bytes, xvram::residency::AccessMode::write_only}};
  xvram::residency::ExternalLease cancelled;
  CHECK(runtime.acquire_external({write_only, 0}, cancelled) ==
        xvram::residency::RuntimeStatus::success);
  CHECK(cancelled.id.value > lease.id.value);
  CHECK(runtime.seal_external(cancelled.id,
                              xvram::residency::ExternalSealMode::cancelled_before_submission) ==
        xvram::residency::RuntimeStatus::success);
  CHECK(runtime.wait_external(cancelled.id, std::chrono::milliseconds(100), poll) ==
        xvram::residency::RuntimeStatus::callback_skipped);
  CHECK(poll.state == xvram::residency::ExternalLeaseState::cancelled);
  CHECK(cuda.mappings.empty());
  CHECK(runtime.release(cancelled_allocation.id) == xvram::residency::RuntimeStatus::success);

  CHECK(runtime.close() == xvram::residency::RuntimeStatus::success);
  CHECK(cuda.violations.empty());
}

void compressed_discarded_full_write_reuse_test() {
  const CheckContext context("compressed discarded full-write reuse");
  FakeCuda cuda;
  CudaApi api(CudaApi::InjectedDispatch{});
  cuda.install(api);
  const ActiveFake active(cuda);
  xvram::nvcomp::NvcompApi unavailable_codec(xvram::nvcomp::NvcompDispatch{},
                                             "injected-unavailable");

  xvram::residency::RuntimeConfig config;
  config.chunk_bytes = chunk_bytes;
  config.cache_target_bytes = 8ULL * chunk_bytes;
  config.device_headroom_bytes = chunk_bytes;
  config.staging_slots = 2;
  config.stall_timeout = std::chrono::milliseconds(100);
  config.compression_mode = xvram::residency::CompressionMode::adaptive;
  config.host_store_cap_bytes = 8ULL * chunk_bytes;
  config.compression_workspace_cap_bytes = chunk_bytes;
  config.codec_slots = 2;
  config.codec_workers = 2;
  config.nvcomp_api = &unavailable_codec;
  xvram::residency::Runtime runtime(api, config);
  CHECK(runtime.setup() == xvram::residency::RuntimeStatus::success);

  xvram::residency::RuntimeAllocation allocation;
  CHECK(runtime.allocate(chunk_bytes, xvram::residency::ResidencyHint::normal, allocation) ==
        xvram::residency::RuntimeStatus::success);
  CHECK(runtime.discard_dead(allocation.id) == xvram::residency::RuntimeStatus::success);
  const std::array write_only{xvram::residency::AccessRange{
      allocation.id, 0, chunk_bytes, xvram::residency::AccessMode::write_only}};

  xvram::residency::ExternalLease cancelled;
  CHECK(runtime.acquire_external({write_only, 0}, cancelled) ==
        xvram::residency::RuntimeStatus::success);
  CHECK(runtime.telemetry().spill_reserved_bytes >= chunk_bytes);
  CHECK(runtime.seal_external(cancelled.id,
                              xvram::residency::ExternalSealMode::cancelled_before_submission) ==
        xvram::residency::RuntimeStatus::success);
  xvram::residency::ExternalLeasePoll poll;
  CHECK(runtime.wait_external(cancelled.id, std::chrono::milliseconds(100), poll) ==
        xvram::residency::RuntimeStatus::callback_skipped);
  CHECK(runtime.telemetry().spill_reserved_bytes == 0U);

  xvram::residency::ExternalLease rewritten;
  CHECK(runtime.acquire_external({write_only, 0}, rewritten) ==
        xvram::residency::RuntimeStatus::success);
  CHECK(runtime.seal_external(rewritten.id, xvram::residency::ExternalSealMode::success) ==
        xvram::residency::RuntimeStatus::success);
  CHECK(runtime.wait_external(rewritten.id, std::chrono::milliseconds(100), poll) ==
        xvram::residency::RuntimeStatus::success);
  CHECK(runtime.discard_dead(allocation.id) == xvram::residency::RuntimeStatus::success);
  CHECK(runtime.telemetry().spill_reserved_bytes == 0U);
  CHECK(runtime.release(allocation.id) == xvram::residency::RuntimeStatus::success);
  CHECK(runtime.close() == xvram::residency::RuntimeStatus::success);
  CHECK(cuda.violations.empty());
}

void adaptive_cpu_rejection_preserves_raw_generation_test() {
  const CheckContext context("adaptive CPU rejection preserves raw generation");
  FakeCuda cuda;
  CudaApi api(CudaApi::InjectedDispatch{});
  cuda.install(api);
  const ActiveFake active(cuda);
  xvram::nvcomp::NvcompApi unavailable_codec(xvram::nvcomp::NvcompDispatch{},
                                             "injected-unavailable");

  std::vector<xvram::residency::CompressionTraceEvent> trace;
  xvram::residency::RuntimeConfig config;
  config.chunk_bytes = chunk_bytes;
  config.cache_target_bytes = 8ULL * chunk_bytes;
  config.device_headroom_bytes = chunk_bytes;
  config.staging_slots = 2;
  config.stall_timeout = std::chrono::milliseconds(100);
  config.compression_mode = xvram::residency::CompressionMode::adaptive;
  config.host_store_cap_bytes = 8ULL * mib;
  config.compression_workspace_cap_bytes = chunk_bytes;
  config.codec_slots = 2;
  config.codec_workers = 2;
  config.nvcomp_api = &unavailable_codec;
  config.compression_trace = [&](const xvram::residency::CompressionTraceEvent& event) {
    trace.push_back(event);
  };
  xvram::residency::Runtime runtime(api, config);
  CHECK(runtime.setup() == xvram::residency::RuntimeStatus::success);

  xvram::residency::RuntimeAllocation allocation;
  CHECK(runtime.allocate(chunk_bytes, xvram::residency::ResidencyHint::normal, allocation) ==
        xvram::residency::RuntimeStatus::success);
  const auto before = runtime.telemetry();
  std::vector<std::byte> incompressible(static_cast<std::size_t>(chunk_bytes));
  std::uint64_t state = 0x585652414D503035ULL;
  for (std::size_t offset = 0; offset < incompressible.size(); offset += sizeof(state)) {
    state += 0x9E3779B97F4A7C15ULL;
    state = (state ^ (state >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    state = (state ^ (state >> 27U)) * 0x94D049BB133111EBULL;
    state ^= state >> 31U;
    const std::size_t count = std::min(sizeof(state), incompressible.size() - offset);
    std::memcpy(incompressible.data() + offset, &state, count);
  }
  CHECK(runtime.write(allocation.id, 0, incompressible.data(), chunk_bytes) ==
        xvram::residency::RuntimeStatus::success);
  const auto info = runtime.backing_info(allocation.id, 0);
  CHECK(info.has_value());
  if (info.has_value()) {
    CHECK(info->representation == xvram::residency::BackingRepresentation::raw);
    CHECK(info->generation == 2U);
  }
  const auto after = runtime.telemetry();
  CHECK(after.compression_attempts - before.compression_attempts == 3U);
  CHECK(after.compression_commits == before.compression_commits);
  CHECK(after.expansion_rejections - before.expansion_rejections == 3U);
  CHECK(after.never_compress_decisions - before.never_compress_decisions == 1U);
  CHECK(after.raw_path_decisions - before.raw_path_decisions == 1U);
  CHECK(after.generations_created - before.generations_created == 4U);
  CHECK(after.generations_committed - before.generations_committed == 1U);
  CHECK(after.generations_discarded - before.generations_discarded == 3U);
  CHECK(after.generations_created == after.generations_committed + after.generations_discarded);
  CHECK(trace.size() == 3U);
  std::uint64_t previous_operation = 0;
  for (const auto& event : trace) {
    CHECK(event.kind == xvram::residency::CompressionTraceEventKind::generation_discard);
    CHECK((event.key == xvram::residency::ChunkKey{allocation.id, 0}));
    CHECK(event.operation_id > previous_operation);
    CHECK(event.source_generation == 2U);
    CHECK(event.target_generation == 3U);
    CHECK(!event.slot_generation.has_value());
    CHECK(event.path == xvram::residency::CompressionPath::cpu_lz4_gpu_decode);
    CHECK(event.from_representation == xvram::residency::BackingRepresentation::raw);
    CHECK(event.to_representation == xvram::residency::BackingRepresentation::lz4_blocks);
    CHECK(event.logical_bytes == chunk_bytes);
    CHECK(event.physical_bytes == chunk_bytes);
    CHECK(event.reason == "cpu_candidate_expansion_rejected");
    previous_operation = event.operation_id;
  }

  CHECK(runtime.release(allocation.id) == xvram::residency::RuntimeStatus::success);
  CHECK(runtime.close() == xvram::residency::RuntimeStatus::success);
  CHECK(cuda.violations.empty());
}

void cpu_compression_reuses_only_adjacent_identical_blocks_test() {
  const CheckContext context("CPU compression adjacent block reuse");
  constexpr std::uint64_t block_count = 80U;
  constexpr std::uint64_t test_chunk_bytes =
      block_count * xvram::residency::compression_block_bytes;
  FakeCuda cuda;
  cuda.free_memory_bytes = 256ULL * mib;
  cuda.total_memory_bytes = 256ULL * mib;
  CudaApi api(CudaApi::InjectedDispatch{});
  cuda.install(api);
  const ActiveFake active(cuda);
  xvram::nvcomp::NvcompApi unavailable_codec(xvram::nvcomp::NvcompDispatch{},
                                             "injected-unavailable");

  xvram::residency::RuntimeConfig config;
  config.chunk_bytes = test_chunk_bytes;
  config.cache_target_bytes = 8ULL * test_chunk_bytes;
  config.device_headroom_bytes = test_chunk_bytes;
  config.staging_slots = 2;
  config.stall_timeout = std::chrono::milliseconds(1000);
  config.budget_poll_interval = std::chrono::hours(1);
  config.compression_mode = xvram::residency::CompressionMode::capacity;
  config.host_store_cap_bytes = 128ULL * mib;
  config.compression_workspace_cap_bytes = test_chunk_bytes;
  config.codec_slots = 2;
  config.codec_workers = 2;
  config.nvcomp_api = &unavailable_codec;
  xvram::residency::Runtime runtime(api, config);
  CHECK(runtime.setup() == xvram::residency::RuntimeStatus::success);

  std::vector<std::byte> repeated(static_cast<std::size_t>(test_chunk_bytes));
  for (std::size_t offset = 0;
       offset < static_cast<std::size_t>(xvram::residency::compression_block_bytes); ++offset) {
    repeated[offset] = static_cast<std::byte>((offset % 251U) + 1U);
  }
  for (std::uint64_t block = 1; block < block_count; ++block) {
    std::memcpy(repeated.data() +
                    static_cast<std::size_t>(block * xvram::residency::compression_block_bytes),
                repeated.data(),
                static_cast<std::size_t>(xvram::residency::compression_block_bytes));
  }

  xvram::residency::RuntimeAllocation repeated_allocation;
  CHECK(runtime.allocate(test_chunk_bytes, xvram::residency::ResidencyHint::normal,
                         repeated_allocation) == xvram::residency::RuntimeStatus::success);
  const auto before_repeated = runtime.telemetry();
  CHECK(runtime.write(repeated_allocation.id, 0, repeated.data(), test_chunk_bytes) ==
        xvram::residency::RuntimeStatus::success);
  const auto after_repeated = runtime.telemetry();
  CHECK(after_repeated.cpu_codec_blocks_submitted - before_repeated.cpu_codec_blocks_submitted ==
        1U);
  const auto repeated_backing = runtime.backing_info(repeated_allocation.id, 0);
  CHECK(repeated_backing.has_value());
  if (repeated_backing.has_value()) {
    CHECK(repeated_backing->representation == xvram::residency::BackingRepresentation::lz4_blocks);
  }
  std::vector<std::byte> round_trip(repeated.size());
  CHECK(runtime.read(repeated_allocation.id, 0, round_trip.data(), test_chunk_bytes) ==
        xvram::residency::RuntimeStatus::success);
  CHECK(round_trip == repeated);

  std::vector<std::byte> distinct(static_cast<std::size_t>(test_chunk_bytes));
  for (std::uint64_t block = 0; block < block_count; ++block) {
    const std::byte value = static_cast<std::byte>(block + 1U);
    const auto begin = distinct.begin() + static_cast<std::ptrdiff_t>(
                                              block * xvram::residency::compression_block_bytes);
    std::fill_n(begin, static_cast<std::size_t>(xvram::residency::compression_block_bytes), value);
  }
  xvram::residency::RuntimeAllocation distinct_allocation;
  CHECK(runtime.allocate(test_chunk_bytes, xvram::residency::ResidencyHint::normal,
                         distinct_allocation) == xvram::residency::RuntimeStatus::success);
  const auto before_distinct = runtime.telemetry();
  CHECK(runtime.write(distinct_allocation.id, 0, distinct.data(), test_chunk_bytes) ==
        xvram::residency::RuntimeStatus::success);
  const auto after_distinct = runtime.telemetry();
  CHECK(after_distinct.cpu_codec_blocks_submitted - before_distinct.cpu_codec_blocks_submitted ==
        block_count);
  const auto distinct_backing = runtime.backing_info(distinct_allocation.id, 0);
  CHECK(distinct_backing.has_value());
  if (distinct_backing.has_value()) {
    CHECK(distinct_backing->representation == xvram::residency::BackingRepresentation::lz4_blocks);
  }
  CHECK(runtime.read(distinct_allocation.id, 0, round_trip.data(), test_chunk_bytes) ==
        xvram::residency::RuntimeStatus::success);
  CHECK(round_trip == distinct);

  CHECK(runtime.release(distinct_allocation.id) == xvram::residency::RuntimeStatus::success);
  CHECK(runtime.release(repeated_allocation.id) == xvram::residency::RuntimeStatus::success);
  CHECK(runtime.close() == xvram::residency::RuntimeStatus::success);
  CHECK(cuda.violations.empty());
}

void adaptive_cpu_compression_commits_after_reuse_history_test() {
  const CheckContext context("adaptive CPU compression after reuse history");
  constexpr std::uint64_t block_count = 80U;
  constexpr std::uint64_t test_chunk_bytes =
      block_count * xvram::residency::compression_block_bytes;
  FakeCuda cuda;
  cuda.free_memory_bytes = 256ULL * mib;
  cuda.total_memory_bytes = 256ULL * mib;
  CudaApi api(CudaApi::InjectedDispatch{});
  cuda.install(api);
  const ActiveFake active(cuda);
  xvram::nvcomp::NvcompApi unavailable_codec(xvram::nvcomp::NvcompDispatch{},
                                             "injected-unavailable");

  xvram::residency::RuntimeConfig config;
  config.chunk_bytes = test_chunk_bytes;
  config.cache_target_bytes = 8ULL * test_chunk_bytes;
  config.device_headroom_bytes = test_chunk_bytes;
  config.staging_slots = 2;
  config.stall_timeout = std::chrono::milliseconds(1000);
  config.budget_poll_interval = std::chrono::hours(1);
  config.compression_mode = xvram::residency::CompressionMode::adaptive;
  config.host_store_cap_bytes = 128ULL * mib;
  config.compression_workspace_cap_bytes = test_chunk_bytes;
  config.codec_slots = 2;
  config.codec_workers = 2;
  config.nvcomp_api = &unavailable_codec;
  xvram::residency::Runtime runtime(api, config);
  CHECK(runtime.setup() == xvram::residency::RuntimeStatus::success);

  // Prime the pool's historical high-water mark with full, distinct batches. A live-queue
  // availability estimate must recover after these jobs retire.
  xvram::residency::RuntimeAllocation priming_allocation;
  CHECK(runtime.allocate(test_chunk_bytes, xvram::residency::ResidencyHint::normal,
                         priming_allocation) == xvram::residency::RuntimeStatus::success);
  std::vector<std::byte> incompressible(static_cast<std::size_t>(test_chunk_bytes));
  std::uint64_t state = 0xA0761D6478BD642FULL;
  for (std::size_t offset = 0; offset < incompressible.size(); offset += sizeof(state)) {
    state += 0x9E3779B97F4A7C15ULL;
    state = (state ^ (state >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    state = (state ^ (state >> 27U)) * 0x94D049BB133111EBULL;
    state ^= state >> 31U;
    const std::size_t count = std::min(sizeof(state), incompressible.size() - offset);
    std::memcpy(incompressible.data() + offset, &state, count);
  }
  const auto before_priming = runtime.telemetry();
  CHECK(runtime.write(priming_allocation.id, 0, incompressible.data(), test_chunk_bytes) ==
        xvram::residency::RuntimeStatus::success);
  const auto after_priming = runtime.telemetry();
  CHECK(after_priming.cpu_codec_blocks_submitted - before_priming.cpu_codec_blocks_submitted ==
        3U * block_count);
  CHECK(runtime.release(priming_allocation.id) == xvram::residency::RuntimeStatus::success);

  xvram::residency::RuntimeAllocation reused_allocation;
  CHECK(runtime.allocate(test_chunk_bytes, xvram::residency::ResidencyHint::hot,
                         reused_allocation) == xvram::residency::RuntimeStatus::success);
  const std::array read_range{xvram::residency::AccessRange{
      reused_allocation.id, 0, test_chunk_bytes, xvram::residency::AccessMode::read}};
  for (std::uint32_t cycle = 0; cycle < 65U; ++cycle) {
    CHECK(runtime.execute(read_range, 0, [](const auto&) {
      return xvram::residency::RuntimeStatus::success;
    }) == xvram::residency::RuntimeStatus::success);
  }

  std::vector<std::byte> repeated(static_cast<std::size_t>(test_chunk_bytes));
  for (std::size_t offset = 0;
       offset < static_cast<std::size_t>(xvram::residency::compression_block_bytes); ++offset) {
    repeated[offset] = static_cast<std::byte>((offset % 239U) + 1U);
  }
  for (std::uint64_t block = 1; block < block_count; ++block) {
    std::memcpy(repeated.data() +
                    static_cast<std::size_t>(block * xvram::residency::compression_block_bytes),
                repeated.data(),
                static_cast<std::size_t>(xvram::residency::compression_block_bytes));
  }
  const auto before_reused_write = runtime.telemetry();
  const xvram::residency::RuntimeStatus reused_write_status =
      runtime.write(reused_allocation.id, 0, repeated.data(), test_chunk_bytes);
  if (reused_write_status != xvram::residency::RuntimeStatus::success) {
    std::cerr << "adaptive reuse write failed: "
              << xvram::residency::runtime_status_name(reused_write_status) << " at "
              << runtime.error().stage << '/' << runtime.error().operation << ": "
              << runtime.error().message << '\n';
  }
  CHECK(reused_write_status == xvram::residency::RuntimeStatus::success);
  const auto after_reused_write = runtime.telemetry();
  CHECK(after_reused_write.cpu_lz4_gpu_decode_decisions >
        before_reused_write.cpu_lz4_gpu_decode_decisions);
  CHECK(after_reused_write.compression_commits > before_reused_write.compression_commits);
  CHECK(after_reused_write.cpu_codec_blocks_submitted -
            before_reused_write.cpu_codec_blocks_submitted >=
        2U);
  CHECK(after_reused_write.cpu_codec_blocks_submitted -
            before_reused_write.cpu_codec_blocks_submitted <=
        3U);
  const auto backing = runtime.backing_info(reused_allocation.id, 0);
  CHECK(backing.has_value());
  if (backing.has_value()) {
    CHECK(backing->representation == xvram::residency::BackingRepresentation::lz4_blocks);
  }

  CHECK(runtime.release(reused_allocation.id) == xvram::residency::RuntimeStatus::success);
  CHECK(runtime.close() == xvram::residency::RuntimeStatus::success);
  CHECK(cuda.violations.empty());
}

void runtime_compressed_backing_path_provenance_test() {
  const CheckContext context("runtime compressed backing path provenance");
  FakeCuda cuda;
  CudaApi api(CudaApi::InjectedDispatch{});
  cuda.install(api);
  const ActiveFake active(cuda);
  xvram::nvcomp::NvcompApi codec(make_successful_fake_nvcomp_dispatch(), "injected-successful");

  std::vector<xvram::residency::CompressionTraceEvent> trace;
  xvram::residency::RuntimeConfig config;
  config.chunk_bytes = chunk_bytes;
  config.cache_target_bytes = 8ULL * chunk_bytes;
  config.device_headroom_bytes = chunk_bytes;
  config.staging_slots = 2;
  config.stall_timeout = std::chrono::milliseconds(100);
  config.budget_poll_interval = std::chrono::hours(1);
  config.compression_mode = xvram::residency::CompressionMode::capacity;
  config.host_store_cap_bytes = 8ULL * mib;
  config.compression_workspace_cap_bytes = chunk_bytes;
  config.codec_slots = 2;
  config.codec_workers = 2;
  config.nvcomp_api = &codec;
  config.compression_trace = [&](const xvram::residency::CompressionTraceEvent& event) {
    trace.push_back(event);
  };
  xvram::residency::Runtime runtime(api, config);
  CHECK(runtime.setup() == xvram::residency::RuntimeStatus::success);

  xvram::residency::RuntimeAllocation allocation;
  CHECK(runtime.allocate(chunk_bytes, xvram::residency::ResidencyHint::normal, allocation) ==
        xvram::residency::RuntimeStatus::success);
  const std::vector<std::byte> input(static_cast<std::size_t>(chunk_bytes), std::byte{0x5a});
  CHECK(runtime.write(allocation.id, 0, input.data(), chunk_bytes) ==
        xvram::residency::RuntimeStatus::success);
  const auto backing = runtime.backing_info(allocation.id, 0);
  CHECK(backing.has_value());
  if (backing.has_value()) {
    CHECK(backing->representation == xvram::residency::BackingRepresentation::lz4_blocks);
  }
  CHECK(trace.empty());

  const auto before = runtime.telemetry();
  const std::array read{xvram::residency::AccessRange{allocation.id, 0, chunk_bytes,
                                                      xvram::residency::AccessMode::read}};
  CHECK(runtime.execute(read, 0, [](const auto&) {
    return xvram::residency::RuntimeStatus::success;
  }) == xvram::residency::RuntimeStatus::success);
  const auto after = runtime.telemetry();
  CHECK(trace.size() == 4U);
  if (trace.size() == 4U) {
    CHECK(trace[0].kind == xvram::residency::CompressionTraceEventKind::h2d_submit);
    CHECK(trace[1].kind == xvram::residency::CompressionTraceEventKind::h2d_retire);
    CHECK(trace[2].kind == xvram::residency::CompressionTraceEventKind::h2d_submit);
    CHECK(trace[3].kind == xvram::residency::CompressionTraceEventKind::h2d_retire);
    for (const auto& event : trace) {
      CHECK((event.key == xvram::residency::ChunkKey{allocation.id, 0}));
      CHECK(event.path == xvram::residency::CompressionPath::cpu_lz4_gpu_decode);
    }
    CHECK(trace[0].operation_id == trace[1].operation_id);
    CHECK(trace[0].slot_generation == trace[1].slot_generation);
    CHECK(trace[2].operation_id == trace[3].operation_id);
    CHECK(trace[2].slot_generation == trace[3].slot_generation);
    const std::uint64_t traced_physical_bytes = trace[0].physical_bytes + trace[2].physical_bytes;
    CHECK(traced_physical_bytes == after.pcie_h2d_bytes - before.pcie_h2d_bytes);
    CHECK(traced_physical_bytes == after.pcie_h2d_payload_bytes - before.pcie_h2d_payload_bytes +
                                       after.pcie_h2d_metadata_bytes -
                                       before.pcie_h2d_metadata_bytes);
  }

  CHECK(runtime.release(allocation.id) == xvram::residency::RuntimeStatus::success);
  CHECK(runtime.close() == xvram::residency::RuntimeStatus::success);
  CHECK(cuda.violations.empty());
}

void automatic_host_budget_uses_one_post_context_sample_test() {
  const CheckContext context("automatic host budget uses one post-context sample");
  using Status = xvram::residency::RuntimeStatus;
  struct Case {
    std::optional<std::uint64_t> physical;
    std::optional<std::uint64_t> available;
    std::uint64_t cap;
    std::uint64_t headroom;
    Status status;
    std::uint64_t expected_cap;
    std::uint64_t expected_headroom;
  };
  const std::array cases{Case{64 * gib, 48 * gib, 0, 0, Status::success, 32 * gib, 16 * gib},
                         Case{8 * gib, 6 * gib, 0, 0, Status::success, 2 * gib, 4 * gib},
                         Case{8 * gib, 8 * gib, 0, 2 * gib, Status::success, 6 * gib, 2 * gib},
                         Case{8 * gib, 10 * gib, 4 * gib, 0, Status::success, 4 * gib, 4 * gib},
                         Case{8 * gib, 3 * gib, 0, 0, Status::host_oom, 0, 0},
                         Case{8 * gib, 8 * gib, 8 * gib, 2 * gib, Status::unavailable, 0, 0},
                         Case{8 * gib, std::nullopt, 0, 0, Status::unavailable, 0, 0},
                         Case{std::nullopt, 8 * gib, 0, 0, Status::unavailable, 0, 0},
                         Case{8 * gib, std::numeric_limits<std::uint64_t>::max(),
                              std::numeric_limits<std::uint64_t>::max(), 1, Status::unavailable, 0,
                              0}};
  for (const Case& item : cases) {
    FakeCuda cuda;
    CudaApi api(CudaApi::InjectedDispatch{});
    cuda.install(api);
    const ActiveFake active(cuda);
    xvram::nvcomp::NvcompApi codec(make_successful_fake_nvcomp_dispatch(), "injected-successful");
    std::uint64_t samples = 0;
    xvram::residency::RuntimeConfig config;
    config.chunk_bytes = chunk_bytes;
    config.cache_target_bytes = 8ULL * chunk_bytes;
    config.device_headroom_bytes = chunk_bytes;
    config.staging_slots = 2;
    config.compression_mode = xvram::residency::CompressionMode::adaptive;
    config.compression_workspace_cap_bytes = chunk_bytes;
    config.nvcomp_api = &codec;
    config.resolve_host_budget = true;
    config.host_store_cap_bytes = item.cap;
    config.host_headroom_bytes = item.headroom;
    config.host_memory_sample = [&] {
      ++samples;
      // A second snapshot would observe pressure and reject an already-resolved auto cap.
      return xvram::residency::HostMemorySample{
          item.physical, samples == 1U ? item.available : std::optional<std::uint64_t>{0U}};
    };
    xvram::residency::Runtime runtime(api, config);
    CHECK(runtime.setup() == item.status);
    CHECK(samples == 1U);
    if (item.status == Status::success) {
      CHECK(runtime.telemetry().host_store_cap_bytes == item.expected_cap);
      CHECK(runtime.telemetry().host_headroom_bytes == item.expected_headroom);
    }
    CHECK(runtime.close() == Status::success);
    CHECK(cuda.violations.empty());
  }
}

void adaptive_gpu_encode_rejection_uses_admitted_raw_spill_test() {
  const CheckContext context("adaptive GPU encode rejection uses admitted raw spill");
  FakeCuda cuda;
  CudaApi api(CudaApi::InjectedDispatch{});
  cuda.install(api);
  const ActiveFake active(cuda);
  xvram::nvcomp::NvcompApi codec(make_successful_fake_nvcomp_dispatch(), "injected-successful");

  std::vector<xvram::residency::CompressionTraceEvent> trace;
  xvram::residency::RuntimeConfig config;
  config.chunk_bytes = chunk_bytes;
  config.cache_target_bytes = 8ULL * chunk_bytes;
  config.device_headroom_bytes = chunk_bytes;
  config.staging_slots = 2;
  config.stall_timeout = std::chrono::milliseconds(100);
  config.budget_poll_interval = std::chrono::hours(1);
  config.compression_mode = xvram::residency::CompressionMode::adaptive;
  config.host_store_cap_bytes = 8ULL * mib;
  config.compression_workspace_cap_bytes = chunk_bytes;
  config.codec_slots = 2;
  config.codec_workers = 2;
  config.nvcomp_api = &codec;
  config.compression_trace = [&](const xvram::residency::CompressionTraceEvent& event) {
    trace.push_back(event);
  };
  xvram::residency::Runtime runtime(api, config);
  CHECK(runtime.setup() == xvram::residency::RuntimeStatus::success);
  CHECK(runtime.telemetry().nvcomp_available);

  xvram::residency::RuntimeAllocation allocation;
  CHECK(runtime.allocate(chunk_bytes, xvram::residency::ResidencyHint::normal, allocation) ==
        xvram::residency::RuntimeStatus::success);
  const auto generation_before = runtime.telemetry();
  const std::array write_only{xvram::residency::AccessRange{
      allocation.id, 0, chunk_bytes, xvram::residency::AccessMode::write_only}};
  CHECK(runtime.execute(write_only, 0, [](const auto&) {
    // A successful external submission is allowed to leave the deterministic zero-filled fake
    // frame unchanged. The first cost-model sample must still reject the raw-sized GPU candidate.
    return xvram::residency::RuntimeStatus::success;
  }) == xvram::residency::RuntimeStatus::success);
  CHECK(runtime.telemetry().spill_reserved_bytes != 0U);

  const std::uint64_t mapping_d2h_before = cuda.d2h_mapping_calls;
  CHECK(runtime.drain(true) == xvram::residency::RuntimeStatus::success);
  const auto backing = runtime.backing_info(allocation.id, 0);
  CHECK(backing.has_value());
  if (backing.has_value()) {
    CHECK(backing->representation == xvram::residency::BackingRepresentation::raw);
    CHECK(backing->stored_payload_bytes == chunk_bytes);
  }
  CHECK(runtime.telemetry().compression_attempts == 1U);
  CHECK(runtime.telemetry().compression_commits == 0U);
  CHECK(runtime.telemetry().gpu_encode_operations == 1U);
  CHECK(runtime.telemetry().raw_fallbacks == 1U);
  CHECK(runtime.telemetry().raw_path_decisions == 1U);
  CHECK(runtime.telemetry().gpu_lz4_decisions == 0U);
  CHECK(runtime.telemetry().spill_reserved_bytes == 0U);
  CHECK(runtime.telemetry().d2h_bytes == chunk_bytes);
  CHECK(runtime.telemetry().logical_d2h_bytes == 2ULL * chunk_bytes);
  CHECK(runtime.telemetry().pcie_d2h_bytes > 2ULL * chunk_bytes);
  // The rejected raw-sized codec candidate and the authoritative raw spill each cross PCIe from
  // the mapped frame, proving that this is the direct-D2H fallback rather than CPU materialization.
  CHECK(cuda.d2h_mapping_calls == mapping_d2h_before + 2U);
  CHECK(runtime.telemetry().codec_events_recorded == runtime.telemetry().codec_events_retired);
  const auto generation_after = runtime.telemetry();
  CHECK(generation_after.generations_created - generation_before.generations_created == 2U);
  CHECK(generation_after.generations_committed - generation_before.generations_committed == 1U);
  CHECK(generation_after.generations_discarded - generation_before.generations_discarded == 1U);
  CHECK(generation_after.generations_created ==
        generation_after.generations_committed + generation_after.generations_discarded);
  const auto discard = std::find_if(trace.begin(), trace.end(), [](const auto& event) {
    return event.kind == xvram::residency::CompressionTraceEventKind::generation_discard;
  });
  CHECK(discard != trace.end());
  CHECK(std::count_if(trace.begin(), trace.end(), [](const auto& event) {
          return event.kind == xvram::residency::CompressionTraceEventKind::generation_discard;
        }) == 1);
  if (discard != trace.end()) {
    CHECK((discard->key == xvram::residency::ChunkKey{allocation.id, 0}));
    CHECK(discard->operation_id != 0U);
    CHECK(discard->source_generation == 1U);
    CHECK(discard->target_generation == 2U);
    CHECK(discard->slot_generation.has_value());
    CHECK(discard->slot_generation.value_or(0U) != 0U);
    CHECK(discard->path == xvram::residency::CompressionPath::nvcomp_gpu_codec);
    CHECK(discard->from_representation == xvram::residency::BackingRepresentation::implicit_zero);
    CHECK(discard->to_representation == xvram::residency::BackingRepresentation::lz4_blocks);
    CHECK(discard->logical_bytes == chunk_bytes);
    CHECK(discard->physical_bytes == chunk_bytes);
    CHECK(discard->reason == "gpu_candidate_raw_fallback_committed");
    const auto codec_submit = std::find_if(trace.begin(), trace.end(), [](const auto& event) {
      return event.kind == xvram::residency::CompressionTraceEventKind::encode_submit;
    });
    CHECK(codec_submit != trace.end());
    if (codec_submit != trace.end()) {
      CHECK(discard->operation_id == codec_submit->operation_id);
      CHECK(discard->slot_generation == codec_submit->slot_generation);
      CHECK(discard->key == codec_submit->key);
    }
  }

  std::vector<std::byte> roundtrip(static_cast<std::size_t>(chunk_bytes));
  CHECK(runtime.read(allocation.id, 0, roundtrip.data(), chunk_bytes) ==
        xvram::residency::RuntimeStatus::success);
  CHECK(std::all_of(roundtrip.begin(), roundtrip.end(),
                    [](const std::byte value) { return value == std::byte{0}; }));
  CHECK(runtime.release(allocation.id) == xvram::residency::RuntimeStatus::success);
  CHECK(runtime.close() == xvram::residency::RuntimeStatus::success);
  CHECK(runtime.telemetry().maps == runtime.telemetry().set_access);
  CHECK(runtime.telemetry().maps == runtime.telemetry().unmaps);
  CHECK(cuda.mappings.empty());
  CHECK(cuda.reservations.empty());
  CHECK(cuda.violations.empty());
}

void adaptive_learned_raw_skips_later_gpu_encode_test() {
  const CheckContext context("adaptive learned-raw decision skips later GPU encode");
  FakeCuda cuda;
  CudaApi api(CudaApi::InjectedDispatch{});
  cuda.install(api);
  const ActiveFake active(cuda);
  xvram::nvcomp::NvcompApi codec(make_successful_fake_nvcomp_dispatch(), "injected-successful");

  xvram::residency::RuntimeConfig config;
  config.chunk_bytes = chunk_bytes;
  config.cache_target_bytes = 8ULL * chunk_bytes;
  config.device_headroom_bytes = chunk_bytes;
  config.staging_slots = 2;
  config.stall_timeout = std::chrono::milliseconds(100);
  config.budget_poll_interval = std::chrono::hours(1);
  config.compression_mode = xvram::residency::CompressionMode::adaptive;
  config.host_store_cap_bytes = 8ULL * mib;
  config.compression_workspace_cap_bytes = chunk_bytes;
  config.codec_slots = 2;
  config.codec_workers = 2;
  config.nvcomp_api = &codec;
  xvram::residency::Runtime runtime(api, config);
  CHECK(runtime.setup() == xvram::residency::RuntimeStatus::success);

  xvram::residency::RuntimeAllocation allocation;
  CHECK(runtime.allocate(chunk_bytes, xvram::residency::ResidencyHint::normal, allocation) ==
        xvram::residency::RuntimeStatus::success);
  const std::array write_only{xvram::residency::AccessRange{
      allocation.id, 0, chunk_bytes, xvram::residency::AccessMode::write_only}};

  // The fake GPU codec emits a verified raw-sized candidate. Two real samples calibrate both
  // paths for the unchanged zero generation; the third write-back must go directly to the
  // pre-admitted raw spill instead of paying a third encode and candidate D2H.
  for (std::uint32_t pass = 0; pass < 3U; ++pass) {
    CHECK(runtime.execute(write_only, 0, [](const auto&) {
      return xvram::residency::RuntimeStatus::success;
    }) == xvram::residency::RuntimeStatus::success);
    CHECK(runtime.drain(true) == xvram::residency::RuntimeStatus::success);
    if (pass == 0U) {
      CHECK(runtime.telemetry().gpu_encode_operations == 1U);
      CHECK(runtime.telemetry().logical_d2h_bytes == 2ULL * chunk_bytes);
    } else if (pass == 1U) {
      CHECK(runtime.telemetry().gpu_encode_operations == 2U);
      CHECK(runtime.telemetry().logical_d2h_bytes == 4ULL * chunk_bytes);
    }
  }

  CHECK(runtime.telemetry().gpu_encode_operations == 2U);
  CHECK(runtime.telemetry().compression_attempts == 2U);
  CHECK(runtime.telemetry().logical_d2h_bytes == 5ULL * chunk_bytes);
  CHECK(runtime.telemetry().d2h_bytes == 3ULL * chunk_bytes);
  CHECK(cuda.d2h_mapping_calls == 5U);
  CHECK(runtime.telemetry().raw_path_decisions >= 1U);
  CHECK(runtime.telemetry().generations_created ==
        runtime.telemetry().generations_committed + runtime.telemetry().generations_discarded);
  CHECK(runtime.telemetry().codec_events_recorded == runtime.telemetry().codec_events_retired);

  CHECK(runtime.release(allocation.id) == xvram::residency::RuntimeStatus::success);
  CHECK(runtime.close() == xvram::residency::RuntimeStatus::success);
  CHECK(runtime.telemetry().maps == runtime.telemetry().set_access);
  CHECK(runtime.telemetry().maps == runtime.telemetry().unmaps);
  CHECK(cuda.mappings.empty());
  CHECK(cuda.reservations.empty());
  CHECK(cuda.violations.empty());
}

void adaptive_gpu_candidate_quarantine_remains_unresolved_test() {
  const CheckContext context("quarantined GPU candidate remains unresolved");
  FakeCuda cuda;
  CudaApi api(CudaApi::InjectedDispatch{});
  cuda.install(api);
  const ActiveFake active(cuda);
  xvram::nvcomp::NvcompApi codec(make_successful_fake_nvcomp_dispatch(), "injected-successful");

  xvram::residency::RuntimeConfig config;
  config.chunk_bytes = chunk_bytes;
  config.cache_target_bytes = 8ULL * chunk_bytes;
  config.device_headroom_bytes = chunk_bytes;
  config.staging_slots = 2;
  config.stall_timeout = std::chrono::milliseconds(100);
  config.budget_poll_interval = std::chrono::hours(1);
  config.compression_mode = xvram::residency::CompressionMode::adaptive;
  config.host_store_cap_bytes = 8ULL * mib;
  config.compression_workspace_cap_bytes = chunk_bytes;
  config.codec_slots = 2;
  config.codec_workers = 2;
  config.nvcomp_api = &codec;
  xvram::residency::Runtime runtime(api, config);
  CHECK(runtime.setup() == xvram::residency::RuntimeStatus::success);

  xvram::residency::RuntimeAllocation allocation;
  CHECK(runtime.allocate(chunk_bytes, xvram::residency::ResidencyHint::normal, allocation) ==
        xvram::residency::RuntimeStatus::success);
  const auto generation_before = runtime.telemetry();
  const std::array write_only{xvram::residency::AccessRange{
      allocation.id, 0, chunk_bytes, xvram::residency::AccessMode::write_only}};
  CHECK(runtime.execute(write_only, 0, [](const auto&) {
    return xvram::residency::RuntimeStatus::success;
  }) == xvram::residency::RuntimeStatus::success);

  // The first mapped D2H completes the verified GPU candidate. Fail the following raw spill copy,
  // after submission is already known to have occurred, so the candidate cannot be called safely
  // discarded and its codec slot/mapping must remain quarantined with the worker.
  cuda.injection.site = FaultSite::d2h;
  cuda.injection.result = CUDA_ERROR_UNKNOWN;
  cuda.injection.calls_to_skip = 1U;
  CHECK(runtime.drain(true) != xvram::residency::RuntimeStatus::success);
  CHECK(cuda.injection.triggered);
  CHECK(runtime.async_completion_unknown());
  CHECK(runtime.telemetry().quarantined);
  const auto generation_after = runtime.telemetry();
  CHECK(generation_after.generations_created - generation_before.generations_created == 1U);
  CHECK(generation_after.generations_committed == generation_before.generations_committed);
  CHECK(generation_after.generations_discarded == generation_before.generations_discarded);
  CHECK(generation_after.generations_created >
        generation_after.generations_committed + generation_after.generations_discarded);

  CHECK(runtime.close() == xvram::residency::RuntimeStatus::cleanup_failure);
  CHECK(std::any_of(cuda.physical.begin(), cuda.physical.end(),
                    [](const auto& pair) { return !pair.second.released; }));
  CHECK(cuda.violations.empty());
}

void runtime_codec_pinned_budget_fallback_preserves_device_peak_test() {
  const CheckContext context("runtime codec pinned-budget fallback preserves device peak");
  FakeCuda cuda;
  CudaApi api(CudaApi::InjectedDispatch{});
  cuda.install(api);
  const ActiveFake active(cuda);
  xvram::nvcomp::NvcompApi codec(make_fake_nvcomp_dispatch(), "injected-available");

  xvram::residency::RuntimeConfig config;
  config.chunk_bytes = chunk_bytes;
  config.cache_target_bytes = 8ULL * chunk_bytes;
  config.device_headroom_bytes = chunk_bytes;
  config.staging_slots = 2;
  config.stall_timeout = std::chrono::milliseconds(100);
  config.compression_mode = xvram::residency::CompressionMode::adaptive;
  // The ordinary H2D/D2H staging pool consumes two chunks.  Leave only one more chunk so the
  // fully created nvCOMP pinned slots cannot be admitted and runtime must safely retain the CPU
  // codec fallback.
  config.host_store_cap_bytes = 3ULL * chunk_bytes;
  config.compression_workspace_cap_bytes = chunk_bytes;
  config.codec_slots = 2;
  config.codec_workers = 2;
  config.nvcomp_api = &codec;

  xvram::residency::Runtime runtime(api, config);
  CHECK(runtime.setup() == xvram::residency::RuntimeStatus::success);
  const auto& telemetry = runtime.telemetry();
  CHECK(!telemetry.nvcomp_available);
  CHECK(telemetry.codec_workspace_bytes == 0U);
  CHECK(telemetry.codec_slot_bytes == 0U);
  CHECK(telemetry.codec_workspace_peak_bytes != 0U);
  CHECK(telemetry.codec_slot_capacity_bytes != 0U);
  CHECK(telemetry.codec_slot_peak_bytes == telemetry.codec_slot_capacity_bytes);
  CHECK(telemetry.codec_slot_capacity_bytes <=
        std::numeric_limits<std::uint64_t>::max() - telemetry.codec_workspace_peak_bytes);
  CHECK(telemetry.device_reserve_bytes_peak >=
        telemetry.codec_slot_capacity_bytes + telemetry.codec_workspace_peak_bytes);
  CHECK(telemetry.device_budget_violation_count == 0U);

  CHECK(runtime.close() == xvram::residency::RuntimeStatus::success);
  CHECK(cuda.linear.empty());
  CHECK(cuda.mappings.empty());
  CHECK(cuda.reservations.empty());
  CHECK(cuda.violations.empty());
}

void runtime_post_map_backing_budget_failure_test(const bool nvcomp_available) {
  const CheckContext context(nvcomp_available ? "runtime post-map codec export budget failure"
                                              : "runtime post-map CPU decode budget failure");
  FakeCuda cuda;
  CudaApi api(CudaApi::InjectedDispatch{});
  cuda.install(api);
  const ActiveFake active(cuda);
  xvram::nvcomp::NvcompApi codec(nvcomp_available ? make_fake_nvcomp_dispatch()
                                                  : xvram::nvcomp::NvcompDispatch{},
                                 nvcomp_available ? "injected-available" : "injected-unavailable");

  xvram::residency::RuntimeConfig config;
  config.chunk_bytes = chunk_bytes;
  config.cache_target_bytes = 32ULL * chunk_bytes;
  config.device_headroom_bytes = chunk_bytes;
  config.staging_slots = 2;
  config.stall_timeout = std::chrono::milliseconds(100);
  config.budget_poll_interval = std::chrono::hours(1);
  config.compression_mode = xvram::residency::CompressionMode::capacity;
  config.host_store_cap_bytes = mib;
  config.compression_workspace_cap_bytes = chunk_bytes;
  config.codec_slots = 2;
  config.codec_workers = 2;
  config.nvcomp_api = &codec;
  xvram::residency::Runtime runtime(api, config);
  CHECK(runtime.setup() == xvram::residency::RuntimeStatus::success);
  CHECK(runtime.telemetry().nvcomp_available == nvcomp_available);

  xvram::residency::RuntimeAllocation compressed;
  CHECK(runtime.allocate(chunk_bytes, xvram::residency::ResidencyHint::normal, compressed) ==
        xvram::residency::RuntimeStatus::success);
  std::vector<std::byte> compressible(static_cast<std::size_t>(chunk_bytes), std::byte{0x5a});
  CHECK(runtime.write(compressed.id, 0, compressible.data(), chunk_bytes) ==
        xvram::residency::RuntimeStatus::success);
  const auto compressed_info = runtime.backing_info(compressed.id, 0);
  CHECK(compressed_info.has_value());
  if (compressed_info.has_value()) {
    CHECK(compressed_info->representation == xvram::residency::BackingRepresentation::lz4_blocks);
    CHECK(compressed_info->stored_payload_bytes < compressed_info->valid_bytes);
  }
  const auto decode_before = runtime.telemetry();
  std::vector<std::byte> host_readback(static_cast<std::size_t>(chunk_bytes));
  CHECK(runtime.read(compressed.id, 0, host_readback.data(), chunk_bytes) ==
        xvram::residency::RuntimeStatus::success);
  CHECK(host_readback == compressible);
  const auto decode_after = runtime.telemetry();
  CHECK(decode_after.cpu_decode_operations - decode_before.cpu_decode_operations == 1U);
  CHECK(decode_after.cpu_decode_nanoseconds >= decode_before.cpu_decode_nanoseconds);
  CHECK(decode_after.generations_created ==
        decode_after.generations_committed + decode_after.generations_discarded);

  std::vector<xvram::residency::RuntimeAllocation> blockers;
  std::size_t dirty_blockers = 0;
  bool exhausted_spill_budget = false;
  for (std::size_t index = 0; index < 24U; ++index) {
    xvram::residency::RuntimeAllocation blocker;
    CHECK(runtime.allocate(chunk_bytes, xvram::residency::ResidencyHint::normal, blocker) ==
          xvram::residency::RuntimeStatus::success);
    blockers.push_back(blocker);
    const std::array write_only{xvram::residency::AccessRange{
        blocker.id, 0, chunk_bytes, xvram::residency::AccessMode::write_only}};
    xvram::residency::ExternalLease lease;
    const xvram::residency::RuntimeStatus acquired =
        runtime.acquire_external({write_only, 0}, lease);
    if (acquired == xvram::residency::RuntimeStatus::host_oom) {
      exhausted_spill_budget = true;
      CHECK(runtime.error().operation == "reserve_spill");
      break;
    }
    CHECK(acquired == xvram::residency::RuntimeStatus::success);
    if (acquired != xvram::residency::RuntimeStatus::success) {
      break;
    }
    CHECK(runtime.seal_external(lease.id, xvram::residency::ExternalSealMode::success) ==
          xvram::residency::RuntimeStatus::success);
    xvram::residency::ExternalLeasePoll poll;
    CHECK(runtime.wait_external(lease.id, std::chrono::milliseconds(100), poll) ==
          xvram::residency::RuntimeStatus::success);
    ++dirty_blockers;
  }
  CHECK(exhausted_spill_budget);
  CHECK(dirty_blockers != 0U);

  const std::uint64_t maps_before = runtime.telemetry().maps;
  const std::uint64_t set_access_before = runtime.telemetry().set_access;
  const std::uint64_t unmaps_before = runtime.telemetry().unmaps;
  const std::uint64_t event_boundaries_before = runtime.telemetry().event_boundaries;
  const std::uint64_t queries_before = cuda.event_query_calls;
  const std::size_t live_mappings_before = cuda.mappings.size();
  const std::array read{xvram::residency::AccessRange{compressed.id, 0, chunk_bytes,
                                                      xvram::residency::AccessMode::read}};
  xvram::residency::ExternalLease failed_lease;
  CHECK(runtime.acquire_external({read, 0}, failed_lease) ==
        xvram::residency::RuntimeStatus::host_oom);
  CHECK(!failed_lease.id);
  CHECK(runtime.error().operation ==
        (nvcomp_available ? "reserve_codec_export" : "reserve_cpu_decode_scratch"));
  CHECK(runtime.telemetry().maps == maps_before + 1U);
  CHECK(runtime.telemetry().set_access == set_access_before + 1U);
  CHECK(runtime.telemetry().unmaps == unmaps_before + 1U);
  CHECK(runtime.telemetry().event_boundaries == event_boundaries_before + 1U);
  CHECK(cuda.event_query_calls > queries_before);
  CHECK(cuda.mappings.size() == live_mappings_before);
  CHECK(runtime.telemetry().unsafe_remaps == 0U);
  CHECK(runtime.telemetry().unsafe_transitions == 0U);
  CHECK(!runtime.poisoned());
  CHECK(!runtime.async_completion_unknown());
  CHECK(!runtime.telemetry().quarantined);
  CHECK(cuda.violations.empty());

  for (std::size_t index = 0; index < blockers.size(); ++index) {
    if (index < dirty_blockers) {
      CHECK(runtime.discard_dead(blockers[index].id) == xvram::residency::RuntimeStatus::success);
    }
    CHECK(runtime.release(blockers[index].id) == xvram::residency::RuntimeStatus::success);
  }
  CHECK(runtime.release(compressed.id) == xvram::residency::RuntimeStatus::success);
  CHECK(runtime.close() == xvram::residency::RuntimeStatus::success);
  CHECK(runtime.telemetry().maps == runtime.telemetry().set_access);
  CHECK(runtime.telemetry().maps == runtime.telemetry().unmaps);
  CHECK(cuda.mappings.empty());
  CHECK(cuda.reservations.empty());
  CHECK(cuda.violations.empty());
}

struct SpillPressureFixture {
  FakeCuda cuda;
  CudaApi api{CudaApi::InjectedDispatch{}};
  ActiveFake active{cuda};
  xvram::nvcomp::NvcompApi codec{xvram::nvcomp::NvcompDispatch{}, "injected-unavailable"};
  xvram::residency::RuntimeConfig config;
  std::unique_ptr<xvram::residency::Runtime> runtime;
  xvram::residency::RuntimeAllocation allocation;
  std::vector<std::byte> expected;
  std::uint64_t callbacks = 0;

  [[nodiscard]] bool setup(const bool no_spill_room = false, const std::size_t logical_chunks = 3U,
                           const std::uint32_t staging_slots = 2U) {
    cuda.install(api);
    const auto raw_charge = xvram::residency::maximum_raw_backing_charge(chunk_bytes);
    CHECK(raw_charge.has_value());
    if (!raw_charge.has_value()) {
      return false;
    }
    config.chunk_bytes = chunk_bytes;
    config.cache_target_bytes = 8ULL * chunk_bytes;
    config.device_headroom_bytes = chunk_bytes;
    config.staging_slots = staging_slots;
    config.stall_timeout = std::chrono::milliseconds(100);
    config.budget_poll_interval = std::chrono::hours(1);
    config.compression_mode = xvram::residency::CompressionMode::capacity;
    config.forced_compression_path = xvram::residency::CompressionPath::raw;
    // All raw authorities plus one fewer raw spill than logical chunks fit.
    // Retiring a dirty spill replaces its old authority, freeing one raw charge.
    config.host_store_cap_bytes =
        staging_slots * chunk_bytes +
        (logical_chunks + (no_spill_room ? 0U : logical_chunks - 1U)) * *raw_charge +
        (no_spill_room ? chunk_bytes / 2ULL : 0ULL);
    config.compression_workspace_cap_bytes = chunk_bytes;
    config.codec_slots = 2;
    config.codec_workers = 2;
    config.nvcomp_api = &codec;
    runtime = std::make_unique<xvram::residency::Runtime>(api, config);
    auto status = runtime->setup();
    CHECK(status == xvram::residency::RuntimeStatus::success);
    if (status != xvram::residency::RuntimeStatus::success) {
      return false;
    }
    status = runtime->allocate(logical_chunks * chunk_bytes,
                               xvram::residency::ResidencyHint::normal, allocation);
    CHECK(status == xvram::residency::RuntimeStatus::success);
    if (status != xvram::residency::RuntimeStatus::success) {
      return false;
    }
    expected.resize(logical_chunks);
    std::vector<std::byte> initial(static_cast<std::size_t>(chunk_bytes));
    for (std::size_t index = 0; index < expected.size(); ++index) {
      expected[index] = static_cast<std::byte>(0x10U + index);
      std::fill(initial.begin(), initial.end(), expected[index]);
      status = runtime->write(allocation.id, index * chunk_bytes, initial.data(), chunk_bytes);
      CHECK(status == xvram::residency::RuntimeStatus::success);
      if (status != xvram::residency::RuntimeStatus::success) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] xvram::residency::RuntimeStatus
  write(const std::size_t index, const std::byte value, const bool protect_chunk_zero = false) {
    std::vector<xvram::residency::AccessRange> ranges;
    if (protect_chunk_zero) {
      ranges.push_back({allocation.id, 0, chunk_bytes, xvram::residency::AccessMode::read});
    }
    ranges.push_back({allocation.id, index * chunk_bytes, chunk_bytes,
                      xvram::residency::AccessMode::read_write});
    const auto status = runtime->execute(ranges, 0, [&](const auto& context) {
      ++callbacks;
      const auto& output = context.ranges.back();
      const auto view =
          cuda.mapping_view(output.device_address, static_cast<std::size_t>(chunk_bytes));
      CHECK(view.has_value());
      if (!view.has_value()) {
        return xvram::residency::RuntimeStatus::callback_failed;
      }
      std::fill_n(view->allocation->storage.begin() + static_cast<std::ptrdiff_t>(view->offset),
                  static_cast<std::size_t>(chunk_bytes), value);
      cuda.begin_mapping_work(view->base, context.stream);
      return xvram::residency::RuntimeStatus::success;
    });
    if (status == xvram::residency::RuntimeStatus::success) {
      expected[index] = value;
    }
    return status;
  }

  void verify_and_close() {
    std::vector<std::byte> output(static_cast<std::size_t>(chunk_bytes));
    for (std::size_t index = 0; index < expected.size(); ++index) {
      CHECK(runtime->read(allocation.id, index * chunk_bytes, output.data(), chunk_bytes) ==
            xvram::residency::RuntimeStatus::success);
      CHECK(std::all_of(output.begin(), output.end(),
                        [&](const std::byte value) { return value == expected[index]; }));
    }
    CHECK(runtime->release(allocation.id) == xvram::residency::RuntimeStatus::success);
    CHECK(runtime->close() == xvram::residency::RuntimeStatus::success);
    CHECK(runtime->telemetry().cleanup_operations_drained);
    CHECK(runtime->telemetry().cleanup_events_drained);
    CHECK(runtime->telemetry().host_budget_peak_bytes <= config.host_store_cap_bytes);
    CHECK(runtime->telemetry().maps == runtime->telemetry().set_access);
    CHECK(runtime->telemetry().maps == runtime->telemetry().unmaps);
    CHECK(runtime->telemetry().unsafe_remaps == 0U);
    CHECK(runtime->telemetry().unsafe_transitions == 0U);
    CHECK(cuda.mappings.empty());
    CHECK(cuda.reservations.empty());
    CHECK(cuda.violations.empty());
  }
};

void runtime_spill_pressure_recovery_test(const bool protect_first_dirty) {
  const CheckContext context(protect_first_dirty ? "spill pressure skips pinned working-set chunk"
                                                 : "spill pressure retires lowest eligible key");
  SpillPressureFixture fixture;
  if (!fixture.setup()) {
    return;
  }
  CHECK(fixture.write(0, std::byte{0xa0}) == xvram::residency::RuntimeStatus::success);
  CHECK(fixture.write(1, std::byte{0xa1}) == xvram::residency::RuntimeStatus::success);
  const auto before = fixture.runtime->telemetry();
  const auto first_before = fixture.runtime->backing_info(fixture.allocation.id, 0);
  const auto second_before = fixture.runtime->backing_info(fixture.allocation.id, 1);
  CHECK(first_before.has_value() && second_before.has_value());
  const auto callbacks_before = fixture.callbacks;
  CHECK(fixture.write(2, std::byte{0xa2}, protect_first_dirty) ==
        xvram::residency::RuntimeStatus::success);
  const auto after = fixture.runtime->telemetry();
  const auto first_after = fixture.runtime->backing_info(fixture.allocation.id, 0);
  const auto second_after = fixture.runtime->backing_info(fixture.allocation.id, 1);
  CHECK(first_after.has_value() && second_after.has_value());
  if (first_before && second_before && first_after && second_after) {
    CHECK(first_after->generation == first_before->generation + (protect_first_dirty ? 0U : 1U));
    CHECK(second_after->generation == second_before->generation + (protect_first_dirty ? 1U : 0U));
  }
  CHECK(after.dirty_writebacks == before.dirty_writebacks + 1U);
  CHECK(after.d2h_bytes == before.d2h_bytes + chunk_bytes);
  CHECK(after.unmaps == before.unmaps);
  CHECK(fixture.callbacks == callbacks_before + 1U);
  CHECK(after.spill_reserved_bytes == before.spill_reserved_bytes);
  CHECK(!after.quarantined);
  fixture.verify_and_close();
}

void runtime_spill_pressure_no_safe_victim_test(const bool protect_later_dirty) {
  const CheckContext context(protect_later_dirty ? "spill pressure protects all declared chunks"
                                                 : "spill pressure genuine host OOM before launch");
  SpillPressureFixture fixture;
  if (!fixture.setup(!protect_later_dirty)) {
    return;
  }
  if (protect_later_dirty) {
    CHECK(fixture.write(1, std::byte{0xb1}) == xvram::residency::RuntimeStatus::success);
    CHECK(fixture.write(2, std::byte{0xb2}) == xvram::residency::RuntimeStatus::success);
  }
  const auto before = fixture.runtime->telemetry();
  const auto callbacks_before = fixture.callbacks;
  const std::array ranges{xvram::residency::AccessRange{
      fixture.allocation.id, 0, protect_later_dirty ? 3ULL * chunk_bytes : chunk_bytes,
      xvram::residency::AccessMode::read_write}};
  bool invoked = false;
  CHECK(fixture.runtime->execute(ranges, 0, [&](const auto&) {
    invoked = true;
    return xvram::residency::RuntimeStatus::success;
  }) == xvram::residency::RuntimeStatus::host_oom);
  CHECK(!invoked);
  CHECK(fixture.callbacks == callbacks_before);
  CHECK(fixture.runtime->error().operation == "reserve_spill");
  CHECK(fixture.runtime->telemetry().dirty_writebacks == before.dirty_writebacks);
  CHECK(fixture.runtime->telemetry().d2h_bytes == before.d2h_bytes);
  CHECK(fixture.runtime->telemetry().maps == before.maps);
  CHECK(fixture.runtime->telemetry().spill_reserved_bytes == before.spill_reserved_bytes);
  CHECK(!fixture.runtime->poisoned());
  fixture.verify_and_close();
}

void runtime_spill_pressure_active_lease_is_protected_test() {
  const CheckContext context("spill pressure cannot recover through an active lease");
  SpillPressureFixture fixture;
  if (!fixture.setup()) {
    return;
  }
  CHECK(fixture.write(0, std::byte{0xc0}) == xvram::residency::RuntimeStatus::success);
  CHECK(fixture.write(1, std::byte{0xc1}) == xvram::residency::RuntimeStatus::success);
  const std::array read{xvram::residency::AccessRange{fixture.allocation.id, 0, chunk_bytes,
                                                      xvram::residency::AccessMode::read}};
  xvram::residency::ExternalLease lease;
  CHECK(fixture.runtime->acquire_external({read, 0}, lease) ==
        xvram::residency::RuntimeStatus::success);
  const auto before = fixture.runtime->telemetry();
  const std::array next{xvram::residency::AccessRange{fixture.allocation.id, 2ULL * chunk_bytes,
                                                      chunk_bytes,
                                                      xvram::residency::AccessMode::read_write}};
  xvram::residency::ExternalLease rejected;
  CHECK(fixture.runtime->acquire_external({next, 0}, rejected) ==
        xvram::residency::RuntimeStatus::invalid_argument);
  CHECK(!rejected.id);
  CHECK(fixture.runtime->telemetry().d2h_bytes == before.d2h_bytes);
  CHECK(fixture.runtime->telemetry().spill_reserved_bytes == before.spill_reserved_bytes);
  CHECK(fixture.runtime->seal_external(
            lease.id, xvram::residency::ExternalSealMode::cancelled_before_submission) ==
        xvram::residency::RuntimeStatus::success);
  xvram::residency::ExternalLeasePoll poll;
  CHECK(fixture.runtime->wait_external(lease.id, std::chrono::milliseconds(100), poll) ==
        xvram::residency::RuntimeStatus::callback_skipped);
  fixture.verify_and_close();
}

void runtime_spill_pressure_event_failure_quarantines_test(const FaultSite fault) {
  const CheckContext context(fault == FaultSite::event_record ? "spill recovery event record fault"
                                                              : "spill recovery event query fault");
  SpillPressureFixture fixture;
  if (!fixture.setup(false, 4U, 4U)) {
    return;
  }
  CHECK(fixture.write(0, std::byte{0xd0}) == xvram::residency::RuntimeStatus::success);
  CHECK(fixture.write(1, std::byte{0xd1}) == xvram::residency::RuntimeStatus::success);
  CHECK(fixture.write(2, std::byte{0xd2}) == xvram::residency::RuntimeStatus::success);
  const auto callbacks_before = fixture.callbacks;
  const auto unmaps_before = fixture.cuda.unmap_calls;
  const auto releases_before = fixture.cuda.release_calls;
  const auto d2h_before = fixture.cuda.d2h_mapping_calls;
  fixture.cuda.injection.site = fault;
  fixture.cuda.injection.result = CUDA_ERROR_UNKNOWN;
  CHECK(fixture.write(3, std::byte{0xd3}) == xvram::residency::RuntimeStatus::poisoned);
  CHECK(fixture.cuda.injection.triggered);
  CHECK(fixture.callbacks == callbacks_before);
  CHECK(fixture.runtime->poisoned());
  CHECK(fixture.runtime->async_completion_unknown());
  CHECK(fixture.runtime->telemetry().quarantined);
  CHECK(fixture.cuda.d2h_mapping_calls == d2h_before + 1U);
  const auto pinned_before_close = fixture.cuda.pinned.size();
  // The first D2H slot is unknown. Continuing cleanup on either remaining dirty
  // frame could cycle the two-slot pool and reuse its still-live generation.
  CHECK(fixture.runtime->close() == xvram::residency::RuntimeStatus::cleanup_failure);
  CHECK(!fixture.runtime->telemetry().cleanup_operations_drained);
  CHECK(!fixture.runtime->telemetry().cleanup_events_drained);
  CHECK(fixture.cuda.unmap_calls == unmaps_before);
  CHECK(fixture.cuda.release_calls == releases_before);
  CHECK(fixture.cuda.d2h_mapping_calls == d2h_before + 1U);
  CHECK(fixture.cuda.pinned.size() == pinned_before_close);
  CHECK(!fixture.cuda.mappings.empty());
  CHECK(!fixture.cuda.reservations.empty());
  CHECK(fixture.cuda.violations.empty());
}

void runtime_codec_budget_suspension_test() {
  const CheckContext context("runtime codec budget suspension restores after safe hysteresis");
  FakeCuda cuda;
  CudaApi api(CudaApi::InjectedDispatch{});
  cuda.install(api);
  const ActiveFake active(cuda);
  xvram::nvcomp::NvcompApi codec(make_successful_fake_nvcomp_dispatch(), "injected-successful");

  xvram::residency::RuntimeConfig config;
  config.chunk_bytes = chunk_bytes;
  config.cache_target_bytes = 16ULL * chunk_bytes;
  config.device_headroom_bytes = chunk_bytes;
  config.staging_slots = 2;
  config.stall_timeout = std::chrono::milliseconds(100);
  config.budget_poll_interval = std::chrono::milliseconds(0);
  config.compression_mode = xvram::residency::CompressionMode::adaptive;
  config.host_store_cap_bytes = 4ULL * mib;
  config.compression_workspace_cap_bytes = chunk_bytes;
  config.codec_slots = 2;
  config.codec_workers = 2;
  config.nvcomp_api = &codec;
  xvram::residency::Runtime runtime(api, config);
  CHECK(runtime.setup() == xvram::residency::RuntimeStatus::success);
  CHECK(runtime.telemetry().nvcomp_available);
  CHECK(runtime.telemetry().codec_workspace_bytes != 0U);
  CHECK(runtime.telemetry().codec_slot_bytes != 0U);
  const std::uint64_t initial_codec_workspace = runtime.telemetry().codec_workspace_bytes;
  const std::uint64_t initial_codec_slots = runtime.telemetry().codec_slot_bytes;
  const std::uint64_t initial_slots_created = runtime.telemetry().codec_slots_created;
  CHECK(runtime.telemetry().codec_slots_peak == initial_slots_created);

  xvram::residency::RuntimeAllocation allocation;
  CHECK(runtime.allocate(chunk_bytes, xvram::residency::ResidencyHint::normal, allocation) ==
        xvram::residency::RuntimeStatus::success);
  cuda.free_memory_bytes = chunk_bytes;
  const std::array read{xvram::residency::AccessRange{allocation.id, 0, chunk_bytes,
                                                      xvram::residency::AccessMode::read}};
  CHECK(runtime.execute(read, 0, [](const auto&) {
    return xvram::residency::RuntimeStatus::success;
  }) == xvram::residency::RuntimeStatus::success);
  CHECK(runtime.telemetry().budget_shrinks >= 1U);
  CHECK(runtime.telemetry().codec_workspace_bytes == 0U);
  CHECK(runtime.telemetry().codec_slot_bytes == 0U);
  CHECK(runtime.telemetry().codec_events_recorded == runtime.telemetry().codec_events_retired);
  CHECK(runtime.telemetry().maps == 1U);
  CHECK(runtime.telemetry().set_access == 1U);
  CHECK(runtime.telemetry().unmaps == 0U);
  CHECK(runtime.telemetry().unsafe_remaps == 0U);
  CHECK(runtime.telemetry().unsafe_transitions == 0U);
  CHECK(runtime.telemetry().device_budget_violation_count == 0U);
  CHECK(!runtime.poisoned());
  CHECK(!runtime.async_completion_unknown());

  cuda.free_memory_bytes = 16ULL * mib;
  for (std::uint32_t sample = 0; sample < 9U; ++sample) {
    CHECK(runtime.execute(read, 0, [](const auto&) {
      return xvram::residency::RuntimeStatus::success;
    }) == xvram::residency::RuntimeStatus::success);
    CHECK(runtime.telemetry().codec_workspace_bytes == 0U);
    CHECK(runtime.telemetry().codec_slot_bytes == 0U);
  }
  // One unsafe sample resets the codec-specific hysteresis without forcing an invalid working
  // set; nine subsequent safe samples still must not restore the resources.
  cuda.free_memory_bytes = 2ULL * chunk_bytes;
  CHECK(runtime.execute(read, 0, [](const auto&) {
    return xvram::residency::RuntimeStatus::success;
  }) == xvram::residency::RuntimeStatus::success);
  CHECK(runtime.telemetry().codec_workspace_bytes == 0U);
  cuda.free_memory_bytes = 16ULL * mib;
  for (std::uint32_t sample = 0; sample < 9U; ++sample) {
    CHECK(runtime.execute(read, 0, [](const auto&) {
      return xvram::residency::RuntimeStatus::success;
    }) == xvram::residency::RuntimeStatus::success);
    CHECK(runtime.telemetry().codec_workspace_bytes == 0U);
  }
  CHECK(runtime.execute(read, 0, [](const auto&) {
    return xvram::residency::RuntimeStatus::success;
  }) == xvram::residency::RuntimeStatus::success);
  CHECK(runtime.telemetry().codec_workspace_bytes == initial_codec_workspace);
  CHECK(runtime.telemetry().codec_slot_bytes == initial_codec_slots);
  CHECK(runtime.telemetry().codec_slots_created == initial_slots_created * 2U);
  CHECK(runtime.telemetry().codec_slots_peak == initial_slots_created);

  // A dirty retirement after restoration must use the new codec epoch while lifetime counters
  // remain monotonic across both pipeline instances.
  const std::array write_only{xvram::residency::AccessRange{
      allocation.id, 0, chunk_bytes, xvram::residency::AccessMode::write_only}};
  CHECK(runtime.execute(write_only, 0, [](const auto&) {
    return xvram::residency::RuntimeStatus::success;
  }) == xvram::residency::RuntimeStatus::success);
  CHECK(runtime.drain(true) == xvram::residency::RuntimeStatus::success);
  CHECK(runtime.telemetry().gpu_encode_operations == 1U);
  CHECK(runtime.telemetry().codec_events_recorded == runtime.telemetry().codec_events_retired);
  CHECK(runtime.telemetry().codec_slots_created == initial_slots_created * 2U);
  CHECK(runtime.telemetry().codec_slots_peak == initial_slots_created);

  CHECK(runtime.release(allocation.id) == xvram::residency::RuntimeStatus::success);
  CHECK(runtime.close() == xvram::residency::RuntimeStatus::success);
  CHECK(runtime.telemetry().maps == runtime.telemetry().set_access);
  CHECK(runtime.telemetry().maps == runtime.telemetry().unmaps);
  CHECK(runtime.telemetry().unmaps == runtime.telemetry().event_boundaries);
  CHECK(cuda.linear.empty());
  CHECK(cuda.mappings.empty());
  CHECK(cuda.reservations.empty());
  CHECK(cuda.violations.empty());
}

void runtime_codec_restore_setup_failure_falls_back_then_retries_test() {
  const CheckContext context("runtime codec restore setup failure safely falls back and retries");
  FakeCuda cuda;
  CudaApi api(CudaApi::InjectedDispatch{});
  cuda.install(api);
  const ActiveFake active(cuda);
  xvram::nvcomp::NvcompApi codec(make_successful_fake_nvcomp_dispatch(), "injected-successful");

  xvram::residency::RuntimeConfig config;
  config.chunk_bytes = chunk_bytes;
  config.cache_target_bytes = 16ULL * chunk_bytes;
  config.device_headroom_bytes = chunk_bytes;
  config.staging_slots = 2;
  config.stall_timeout = std::chrono::milliseconds(100);
  config.budget_poll_interval = std::chrono::milliseconds(0);
  config.compression_mode = xvram::residency::CompressionMode::adaptive;
  config.host_store_cap_bytes = 4ULL * mib;
  config.compression_workspace_cap_bytes = chunk_bytes;
  config.codec_slots = 2;
  config.codec_workers = 2;
  config.nvcomp_api = &codec;
  xvram::residency::Runtime runtime(api, config);
  CHECK(runtime.setup() == xvram::residency::RuntimeStatus::success);

  xvram::residency::RuntimeAllocation allocation;
  CHECK(runtime.allocate(chunk_bytes, xvram::residency::ResidencyHint::normal, allocation) ==
        xvram::residency::RuntimeStatus::success);
  const std::array read{xvram::residency::AccessRange{allocation.id, 0, chunk_bytes,
                                                      xvram::residency::AccessMode::read}};
  cuda.free_memory_bytes = chunk_bytes;
  CHECK(runtime.execute(read, 0, [](const auto&) {
    return xvram::residency::RuntimeStatus::success;
  }) == xvram::residency::RuntimeStatus::success);
  CHECK(runtime.telemetry().codec_workspace_bytes == 0U);
  const std::size_t suspended_pinned_allocations = cuda.pinned.size();
  const std::size_t suspended_linear_allocations = cuda.linear.size();

  cuda.free_memory_bytes = 16ULL * mib;
  for (std::uint32_t sample = 0; sample < 9U; ++sample) {
    CHECK(runtime.execute(read, 0, [](const auto&) {
      return xvram::residency::RuntimeStatus::success;
    }) == xvram::residency::RuntimeStatus::success);
  }
  cuda.injection.site = FaultSite::host_allocate;
  cuda.injection.result = CUDA_ERROR_OUT_OF_MEMORY;
  CHECK(runtime.execute(read, 0, [](const auto&) {
    return xvram::residency::RuntimeStatus::success;
  }) == xvram::residency::RuntimeStatus::success);
  CHECK(cuda.injection.triggered);
  CHECK(runtime.telemetry().codec_workspace_bytes == 0U);
  CHECK(runtime.telemetry().codec_slot_bytes == 0U);
  CHECK(cuda.pinned.size() == suspended_pinned_allocations);
  CHECK(cuda.linear.size() == suspended_linear_allocations);
  CHECK(!runtime.poisoned());
  CHECK(!runtime.async_completion_unknown());

  cuda.injection = {};
  for (std::uint32_t sample = 0; sample < 10U; ++sample) {
    CHECK(runtime.execute(read, 0, [](const auto&) {
      return xvram::residency::RuntimeStatus::success;
    }) == xvram::residency::RuntimeStatus::success);
  }
  CHECK(runtime.telemetry().codec_workspace_bytes != 0U);
  CHECK(runtime.telemetry().codec_slot_bytes != 0U);
  CHECK(runtime.telemetry().gpu_codec_fallbacks >= 1U);

  CHECK(runtime.release(allocation.id) == xvram::residency::RuntimeStatus::success);
  CHECK(runtime.close() == xvram::residency::RuntimeStatus::success);
  CHECK(runtime.telemetry().maps == runtime.telemetry().set_access);
  CHECK(runtime.telemetry().maps == runtime.telemetry().unmaps);
  CHECK(cuda.linear.empty());
  CHECK(cuda.mappings.empty());
  CHECK(cuda.reservations.empty());
  CHECK(cuda.violations.empty());
}

void runtime_restored_codec_unknown_event_quarantines_test() {
  const CheckContext context("restored codec unknown event remains a quarantine boundary");
  FakeCuda cuda;
  CudaApi api(CudaApi::InjectedDispatch{});
  cuda.install(api);
  const ActiveFake active(cuda);
  xvram::nvcomp::NvcompApi codec(make_successful_fake_nvcomp_dispatch(), "injected-successful");

  xvram::residency::RuntimeConfig config;
  config.chunk_bytes = chunk_bytes;
  config.cache_target_bytes = 16ULL * chunk_bytes;
  config.device_headroom_bytes = chunk_bytes;
  config.staging_slots = 2;
  config.stall_timeout = std::chrono::milliseconds(100);
  config.budget_poll_interval = std::chrono::milliseconds(0);
  config.compression_mode = xvram::residency::CompressionMode::adaptive;
  config.host_store_cap_bytes = 4ULL * mib;
  config.compression_workspace_cap_bytes = chunk_bytes;
  config.codec_slots = 2;
  config.codec_workers = 2;
  config.nvcomp_api = &codec;
  xvram::residency::Runtime runtime(api, config);
  CHECK(runtime.setup() == xvram::residency::RuntimeStatus::success);

  xvram::residency::RuntimeAllocation allocation;
  CHECK(runtime.allocate(chunk_bytes, xvram::residency::ResidencyHint::normal, allocation) ==
        xvram::residency::RuntimeStatus::success);
  const std::array read{xvram::residency::AccessRange{allocation.id, 0, chunk_bytes,
                                                      xvram::residency::AccessMode::read}};
  cuda.free_memory_bytes = chunk_bytes;
  CHECK(runtime.execute(read, 0, [](const auto&) {
    return xvram::residency::RuntimeStatus::success;
  }) == xvram::residency::RuntimeStatus::success);
  cuda.free_memory_bytes = 16ULL * mib;
  for (std::uint32_t sample = 0; sample < 10U; ++sample) {
    CHECK(runtime.execute(read, 0, [](const auto&) {
      return xvram::residency::RuntimeStatus::success;
    }) == xvram::residency::RuntimeStatus::success);
  }
  CHECK(runtime.telemetry().codec_workspace_bytes != 0U);

  const std::array write_only{xvram::residency::AccessRange{
      allocation.id, 0, chunk_bytes, xvram::residency::AccessMode::write_only}};
  CHECK(runtime.execute(write_only, 0, [](const auto&) {
    return xvram::residency::RuntimeStatus::success;
  }) == xvram::residency::RuntimeStatus::success);
  cuda.injection.site = FaultSite::event_query;
  cuda.injection.result = CUDA_ERROR_UNKNOWN;
  CHECK(runtime.drain(true) == xvram::residency::RuntimeStatus::poisoned);
  CHECK(cuda.injection.triggered);
  CHECK(runtime.poisoned());
  CHECK(runtime.async_completion_unknown());
  CHECK(runtime.telemetry().quarantined);
  CHECK(runtime.telemetry().codec_events_recorded > runtime.telemetry().codec_events_retired);

  CHECK(runtime.close() == xvram::residency::RuntimeStatus::cleanup_failure);
  CHECK(std::any_of(cuda.physical.begin(), cuda.physical.end(),
                    [](const auto& pair) { return !pair.second.released; }));
  CHECK(cuda.violations.empty());
}

void runtime_d2h_allocation_class_accounting_test() {
  const CheckContext context("runtime D2H allocation-class accounting");
  FakeCuda cuda;
  CudaApi api(CudaApi::InjectedDispatch{});
  cuda.install(api);
  const ActiveFake active(cuda);

  xvram::residency::RuntimeConfig config;
  config.chunk_bytes = chunk_bytes;
  config.cache_target_bytes = 8ULL * chunk_bytes;
  config.device_headroom_bytes = chunk_bytes;
  config.staging_slots = 2;
  config.stall_timeout = std::chrono::milliseconds(100);
  xvram::residency::Runtime runtime(api, config);
  CHECK(runtime.setup() == xvram::residency::RuntimeStatus::success);

  const auto dirty_and_release = [&](const xvram::residency::ResidencyHint hint) {
    xvram::residency::RuntimeAllocation allocation;
    CHECK(runtime.allocate(chunk_bytes, hint, allocation) ==
          xvram::residency::RuntimeStatus::success);
    const std::array write_only{xvram::residency::AccessRange{
        allocation.id, 0, chunk_bytes, xvram::residency::AccessMode::write_only}};
    xvram::residency::ExternalLease lease;
    CHECK(runtime.acquire_external({write_only, 0}, lease) ==
          xvram::residency::RuntimeStatus::success);
    CHECK(runtime.seal_external(lease.id, xvram::residency::ExternalSealMode::success) ==
          xvram::residency::RuntimeStatus::success);
    xvram::residency::ExternalLeasePoll poll;
    CHECK(runtime.wait_external(lease.id, std::chrono::milliseconds(100), poll) ==
          xvram::residency::RuntimeStatus::success);
    CHECK(runtime.release(allocation.id) == xvram::residency::RuntimeStatus::success);
  };

  dirty_and_release(xvram::residency::ResidencyHint::hot);
  dirty_and_release(xvram::residency::ResidencyHint::normal);
  CHECK(runtime.telemetry().d2h_bytes == 2ULL * chunk_bytes);
  CHECK(runtime.telemetry().hot_allocation_d2h_bytes == chunk_bytes);
  CHECK(runtime.telemetry().non_hot_allocation_d2h_bytes == chunk_bytes);
  CHECK(runtime.telemetry().d2h_bytes == runtime.telemetry().hot_allocation_d2h_bytes +
                                             runtime.telemetry().non_hot_allocation_d2h_bytes);
  CHECK(runtime.close() == xvram::residency::RuntimeStatus::success);
  CHECK(cuda.violations.empty());
}

void runtime_external_poll_not_ready_test() {
  const CheckContext context("runtime external lease nonblocking poll");
  FakeCuda cuda;
  CudaApi api(CudaApi::InjectedDispatch{});
  cuda.install(api);
  const ActiveFake active(cuda);

  xvram::residency::RuntimeConfig config;
  config.chunk_bytes = chunk_bytes;
  config.cache_target_bytes = 4ULL * chunk_bytes;
  config.device_headroom_bytes = chunk_bytes;
  config.staging_slots = 2;
  config.stall_timeout = std::chrono::milliseconds(100);
  xvram::residency::Runtime runtime(api, config);
  CHECK(runtime.setup() == xvram::residency::RuntimeStatus::success);
  xvram::residency::RuntimeAllocation allocation;
  CHECK(runtime.allocate(chunk_bytes, xvram::residency::ResidencyHint::normal, allocation) ==
        xvram::residency::RuntimeStatus::success);
  const std::array access{xvram::residency::AccessRange{allocation.id, 0, chunk_bytes,
                                                        xvram::residency::AccessMode::read}};
  xvram::residency::ExternalLease lease;
  CHECK(runtime.acquire_external({access, 0}, lease) == xvram::residency::RuntimeStatus::success);
  CHECK(runtime.seal_external(lease.id, xvram::residency::ExternalSealMode::success) ==
        xvram::residency::RuntimeStatus::success);

  cuda.injection.site = FaultSite::event_query;
  cuda.injection.result = CUDA_ERROR_NOT_READY;
  cuda.injection.persistent = true;
  xvram::residency::ExternalLeasePoll poll;
  CHECK(runtime.poll_external(lease.id, poll) == xvram::residency::RuntimeStatus::success);
  CHECK(poll.state == xvram::residency::ExternalLeaseState::submitted);
  CHECK(cuda.injection.triggered);
  CHECK(runtime.release(allocation.id) == xvram::residency::RuntimeStatus::invalid_argument);
  cuda.injection.site = FaultSite::none;
  CHECK(runtime.wait_external(lease.id, std::chrono::milliseconds(100), poll) ==
        xvram::residency::RuntimeStatus::success);
  CHECK(poll.state == xvram::residency::ExternalLeaseState::completed);
  CHECK(runtime.release(allocation.id) == xvram::residency::RuntimeStatus::success);
  CHECK(runtime.close() == xvram::residency::RuntimeStatus::success);
  CHECK(cuda.violations.empty());
}

void runtime_external_duration_guard_test() {
  const CheckContext context("runtime external lease duration guard");
  FakeCuda cuda;
  cuda.event_elapsed_ms = 300.0F;
  CudaApi api(CudaApi::InjectedDispatch{});
  cuda.install(api);
  const ActiveFake active(cuda);

  xvram::residency::RuntimeConfig config;
  config.chunk_bytes = chunk_bytes;
  config.cache_target_bytes = 4ULL * chunk_bytes;
  config.device_headroom_bytes = chunk_bytes;
  config.staging_slots = 2;
  config.stall_timeout = std::chrono::milliseconds(100);
  config.maximum_transaction_duration = std::chrono::milliseconds(250);
  xvram::residency::Runtime runtime(api, config);
  CHECK(runtime.setup() == xvram::residency::RuntimeStatus::success);
  xvram::residency::RuntimeAllocation allocation;
  CHECK(runtime.allocate(chunk_bytes, xvram::residency::ResidencyHint::normal, allocation) ==
        xvram::residency::RuntimeStatus::success);
  const std::array access{xvram::residency::AccessRange{allocation.id, 0, chunk_bytes,
                                                        xvram::residency::AccessMode::read}};
  CHECK(runtime.execute(access, 0, [](const auto&) {
    return xvram::residency::RuntimeStatus::success;
  }) == xvram::residency::RuntimeStatus::timeout);
  CHECK(runtime.telemetry().transactions_completed == 1U);
  CHECK(runtime.telemetry().watchdog_rejections == 1U);
  CHECK(!runtime.async_completion_unknown());
  CHECK(runtime.execute(access, 0, [](const auto&) {
    return xvram::residency::RuntimeStatus::success;
  }) == xvram::residency::RuntimeStatus::invalid_argument);
  CHECK(runtime.close() == xvram::residency::RuntimeStatus::success);
  CHECK(cuda.violations.empty());
}

void runtime_failed_callback_retires_before_poison_test() {
  const CheckContext context("runtime failed callback retires before logical poison");
  FakeCuda cuda;
  CudaApi api(CudaApi::InjectedDispatch{});
  cuda.install(api);
  const ActiveFake active(cuda);

  xvram::residency::RuntimeConfig config;
  config.chunk_bytes = chunk_bytes;
  config.cache_target_bytes = 4ULL * chunk_bytes;
  config.device_headroom_bytes = chunk_bytes;
  config.staging_slots = 2;
  config.stall_timeout = std::chrono::milliseconds(100);
  xvram::residency::Runtime runtime(api, config);
  CHECK(runtime.setup() == xvram::residency::RuntimeStatus::success);
  xvram::residency::RuntimeAllocation allocation;
  CHECK(runtime.allocate(chunk_bytes, xvram::residency::ResidencyHint::normal, allocation) ==
        xvram::residency::RuntimeStatus::success);
  const std::array access{xvram::residency::AccessRange{allocation.id, 0, chunk_bytes,
                                                        xvram::residency::AccessMode::read}};

  CHECK(runtime.execute(access, 0, [](const auto&) {
    return xvram::residency::RuntimeStatus::callback_failed;
  }) == xvram::residency::RuntimeStatus::callback_failed);
  CHECK(runtime.poisoned());
  CHECK(!runtime.async_completion_unknown());
  CHECK(runtime.close() == xvram::residency::RuntimeStatus::success);
  CHECK(cuda.mappings.empty());
  CHECK(cuda.reservations.empty());
  CHECK(cuda.pinned.empty());
  CHECK(cuda.events.empty());
  CHECK(cuda.streams.empty());
  CHECK(cuda.violations.empty());
}

void runtime_set_access_rollback_test() {
  const CheckContext context("runtime SetAccess rollback");
  FakeCuda cuda;
  cuda.injection.site = FaultSite::set_access;
  CudaApi api(CudaApi::InjectedDispatch{});
  cuda.install(api);
  const ActiveFake active(cuda);

  xvram::residency::RuntimeConfig config;
  config.chunk_bytes = chunk_bytes;
  config.cache_target_bytes = 8ULL * chunk_bytes;
  config.device_headroom_bytes = chunk_bytes;
  config.staging_slots = 2;
  config.stall_timeout = std::chrono::milliseconds(100);
  config.budget_poll_interval = std::chrono::milliseconds(1);
  xvram::residency::Runtime runtime(api, config);
  CHECK(runtime.setup() == xvram::residency::RuntimeStatus::success);

  xvram::residency::RuntimeAllocation allocation;
  CHECK(runtime.allocate(chunk_bytes, xvram::residency::ResidencyHint::normal, allocation) ==
        xvram::residency::RuntimeStatus::success);
  const std::array access{xvram::residency::AccessRange{allocation.id, 0, chunk_bytes,
                                                        xvram::residency::AccessMode::read}};
  CHECK(runtime.execute(access, 0, [](const auto&) {
    return xvram::residency::RuntimeStatus::success;
  }) == xvram::residency::RuntimeStatus::cuda_failure);
  CHECK(cuda.injection.triggered);
  CHECK(!runtime.poisoned());
  CHECK(runtime.telemetry().maps == 0U);
  CHECK(runtime.telemetry().unmaps == 0U);
  CHECK(cuda.unmap_calls == 1U);
  CHECK(cuda.mappings.empty());
  CHECK(cuda.release_calls == 0U);

  CHECK(runtime.release(allocation.id) == xvram::residency::RuntimeStatus::success);
  CHECK(runtime.close() == xvram::residency::RuntimeStatus::success);
  CHECK(cuda.reservations.empty());
  CHECK(cuda.mappings.empty());
  CHECK(cuda.violations.empty());
  CHECK(std::all_of(cuda.physical.begin(), cuda.physical.end(),
                    [](const auto& pair) { return pair.second.released; }));
}

void runtime_set_access_rollback_quarantine_test() {
  const CheckContext context("runtime SetAccess rollback quarantine");
  FakeCuda cuda;
  cuda.fault = FaultMode::fail_first_unmap;
  cuda.injection.site = FaultSite::set_access;
  CudaApi api(CudaApi::InjectedDispatch{});
  cuda.install(api);
  const ActiveFake active(cuda);

  xvram::residency::RuntimeConfig config;
  config.chunk_bytes = chunk_bytes;
  config.cache_target_bytes = 8ULL * chunk_bytes;
  config.device_headroom_bytes = chunk_bytes;
  config.staging_slots = 2;
  config.stall_timeout = std::chrono::milliseconds(100);
  config.budget_poll_interval = std::chrono::milliseconds(1);
  xvram::residency::Runtime runtime(api, config);
  CHECK(runtime.setup() == xvram::residency::RuntimeStatus::success);

  xvram::residency::RuntimeAllocation allocation;
  CHECK(runtime.allocate(chunk_bytes, xvram::residency::ResidencyHint::normal, allocation) ==
        xvram::residency::RuntimeStatus::success);
  const std::array access{xvram::residency::AccessRange{allocation.id, 0, chunk_bytes,
                                                        xvram::residency::AccessMode::read}};
  CHECK(runtime.execute(access, 0, [](const auto&) {
    return xvram::residency::RuntimeStatus::success;
  }) == xvram::residency::RuntimeStatus::poisoned);
  CHECK(cuda.injection.triggered);
  CHECK(cuda.unmap_fault_used);
  CHECK(runtime.poisoned());
  CHECK(runtime.telemetry().quarantined);
  CHECK(cuda.mappings.size() == 1U);
  CHECK(cuda.reservations.size() == 1U);
  CHECK(cuda.release_calls == 1U);
  CHECK(std::all_of(cuda.physical.begin(), cuda.physical.end(),
                    [](const auto& pair) { return pair.second.released; }));

  // A failed unmap is the worker quarantine boundary. The handle is released, while the mapping
  // and its exact reservation remain represented until process exit and are never retried.
  CHECK(runtime.close() == xvram::residency::RuntimeStatus::cleanup_failure);
  CHECK(cuda.unmap_calls == 1U);
  CHECK(cuda.mappings.size() == 1U);
  CHECK(cuda.reservations.size() == 1U);
  CHECK(cuda.quarantined_release_observed);
  CHECK(cuda.violations.empty());
}

void runtime_close_attempts_every_mapping_test() {
  const CheckContext context("runtime close attempts every mapping");
  FakeCuda cuda;
  CudaApi api(CudaApi::InjectedDispatch{});
  cuda.install(api);
  const ActiveFake active(cuda);

  xvram::residency::RuntimeConfig config;
  config.chunk_bytes = chunk_bytes;
  config.cache_target_bytes = 8ULL * chunk_bytes;
  config.device_headroom_bytes = chunk_bytes;
  config.staging_slots = 2;
  config.stall_timeout = std::chrono::milliseconds(100);
  config.budget_poll_interval = std::chrono::milliseconds(1);
  xvram::residency::Runtime runtime(api, config);
  CHECK(runtime.setup() == xvram::residency::RuntimeStatus::success);

  xvram::residency::RuntimeAllocation allocation;
  CHECK(runtime.allocate(2ULL * chunk_bytes, xvram::residency::ResidencyHint::normal, allocation) ==
        xvram::residency::RuntimeStatus::success);
  const std::array access{xvram::residency::AccessRange{allocation.id, 0, 2ULL * chunk_bytes,
                                                        xvram::residency::AccessMode::read}};
  CHECK(runtime.execute(access, 0, [](const auto&) {
    return xvram::residency::RuntimeStatus::success;
  }) == xvram::residency::RuntimeStatus::success);
  CHECK(cuda.mappings.size() == 2U);
  CHECK(cuda.physical.size() == 2U);

  cuda.injection.site = FaultSite::unmap;
  cuda.injection.persistent = true;
  const std::uint64_t unmap_calls_before = cuda.unmap_calls;
  CHECK(runtime.close() == xvram::residency::RuntimeStatus::cleanup_failure);
  CHECK(cuda.injection.triggered);
  CHECK(cuda.unmap_calls - unmap_calls_before == 2U);
  CHECK(cuda.mappings.size() == 2U);
  CHECK(cuda.reservations.size() == 1U);
  CHECK(cuda.address_free_calls == 0U);
  CHECK(cuda.release_calls == 2U);
  CHECK(cuda.quarantined_release_observed);
  CHECK(runtime.telemetry().quarantined);
  CHECK(runtime.telemetry().resident_bytes == 2ULL * chunk_bytes);
  CHECK(cuda.violations.empty());
  CHECK(std::all_of(cuda.physical.begin(), cuda.physical.end(),
                    [](const auto& pair) { return pair.second.released; }));
}

void runtime_attach_current_context_ownership_test() {
  const CheckContext context("runtime attach-current context ownership");
  const auto make_config = [](const Context attached) {
    xvram::residency::RuntimeConfig config;
    config.context_mode = xvram::residency::RuntimeContextMode::attach_current;
    config.attached_context = attached;
    config.chunk_bytes = chunk_bytes;
    config.cache_target_bytes = 8ULL * chunk_bytes;
    config.device_headroom_bytes = chunk_bytes;
    config.staging_slots = 2;
    config.stall_timeout = std::chrono::milliseconds(100);
    config.budget_poll_interval = std::chrono::milliseconds(1);
    return config;
  };

  {
    FakeCuda cuda;
    CudaApi api(CudaApi::InjectedDispatch{});
    cuda.install(api);
    const ActiveFake active(cuda);
    const Context captured = cuda.make_token<Context>();
    xvram::residency::Runtime runtime(api, make_config(captured));
    CHECK(runtime.setup() == xvram::residency::RuntimeStatus::success);
    CHECK(runtime.context() == captured);
    CHECK(cuda.pushed_contexts == std::vector<Context>{captured});
    CHECK(runtime.close() == xvram::residency::RuntimeStatus::success);
    CHECK(cuda.popped_contexts == std::vector<Context>{captured});
    CHECK(cuda.destroyed_contexts.empty());
    CHECK(cuda.current_context == nullptr);
  }

  {
    FakeCuda cuda;
    CudaApi api(CudaApi::InjectedDispatch{});
    cuda.install(api);
    const ActiveFake active(cuda);
    const Context captured = cuda.make_token<Context>();
    cuda.injection.site = FaultSite::host_allocate;
    cuda.injection.result = CUDA_ERROR_OUT_OF_MEMORY;
    xvram::residency::Runtime runtime(api, make_config(captured));
    CHECK(runtime.setup() != xvram::residency::RuntimeStatus::success);
    CHECK(cuda.injection.triggered);
    CHECK(cuda.pushed_contexts == std::vector<Context>{captured});
    CHECK(runtime.close() == xvram::residency::RuntimeStatus::success);
    CHECK(cuda.popped_contexts == std::vector<Context>{captured});
    CHECK(cuda.destroyed_contexts.empty());
    CHECK(cuda.current_context == nullptr);
    CHECK(cuda.streams.empty());
    CHECK(cuda.events.empty());
    CHECK(cuda.pinned.empty());
  }
}

void runtime_staging_pool_configuration_test() {
  const CheckContext context("runtime staging pool honors the configured bound");
  FakeCuda cuda;
  CudaApi api(CudaApi::InjectedDispatch{});
  cuda.install(api);
  const ActiveFake active(cuda);

  xvram::residency::RuntimeConfig config;
  config.chunk_bytes = chunk_bytes;
  config.cache_target_bytes = 8ULL * chunk_bytes;
  config.device_headroom_bytes = chunk_bytes;
  config.staging_slots = 5;
  xvram::residency::Runtime runtime(api, config);
  CHECK(runtime.setup() == xvram::residency::RuntimeStatus::success);
  CHECK(cuda.pinned.size() == config.staging_slots);
  CHECK(runtime.telemetry().pinned_staging_bytes == config.staging_slots * chunk_bytes);
  CHECK(runtime.close() == xvram::residency::RuntimeStatus::success);
  CHECK(cuda.pinned.empty());
  CHECK(cuda.events.empty());
}

void runtime_retries_one_frame_oom_test() {
  const CheckContext context("runtime retries one frame allocation OOM after budget refresh");
  FakeCuda cuda;
  CudaApi api(CudaApi::InjectedDispatch{});
  cuda.install(api);
  const ActiveFake active(cuda);

  xvram::residency::RuntimeConfig config;
  config.chunk_bytes = chunk_bytes;
  config.cache_target_bytes = 8ULL * chunk_bytes;
  config.device_headroom_bytes = chunk_bytes;
  config.staging_slots = 2;
  xvram::residency::Runtime runtime(api, config);
  CHECK(runtime.setup() == xvram::residency::RuntimeStatus::success);
  xvram::residency::RuntimeAllocation allocation;
  CHECK(runtime.allocate(chunk_bytes, xvram::residency::ResidencyHint::normal, allocation) ==
        xvram::residency::RuntimeStatus::success);

  cuda.injection.site = FaultSite::memory_create;
  cuda.injection.result = CUDA_ERROR_OUT_OF_MEMORY;
  const std::array access{xvram::residency::AccessRange{allocation.id, 0, chunk_bytes,
                                                        xvram::residency::AccessMode::read}};
  CHECK(runtime.execute(access, 0, [](const auto&) {
    return xvram::residency::RuntimeStatus::success;
  }) == xvram::residency::RuntimeStatus::success);
  CHECK(cuda.injection.triggered);
  CHECK(runtime.telemetry().target_oom_retries == 1U);
  CHECK(runtime.close() == xvram::residency::RuntimeStatus::success);
  CHECK(cuda.violations.empty());
}

void runtime_oom_shrink_defers_pinned_working_set_test() {
  const CheckContext context("runtime OOM shrink defers a pinned working set");
  FakeCuda cuda;
  CudaApi api(CudaApi::InjectedDispatch{});
  cuda.install(api);
  const ActiveFake active(cuda);

  xvram::residency::RuntimeConfig config;
  config.chunk_bytes = chunk_bytes;
  config.cache_target_bytes = 4ULL * chunk_bytes;
  config.device_headroom_bytes = chunk_bytes;
  config.staging_slots = 2;
  config.stall_timeout = std::chrono::milliseconds(100);
  xvram::residency::Runtime runtime(api, config);
  CHECK(runtime.setup() == xvram::residency::RuntimeStatus::success);
  xvram::residency::RuntimeAllocation allocation;
  CHECK(runtime.allocate(2ULL * chunk_bytes, xvram::residency::ResidencyHint::normal, allocation) ==
        xvram::residency::RuntimeStatus::success);

  cuda.injection.site = FaultSite::memory_create;
  cuda.injection.result = CUDA_ERROR_OUT_OF_MEMORY;
  cuda.injection.calls_to_skip = 1;
  cuda.free_memory_after_create_fault = 0;
  const std::array access{xvram::residency::AccessRange{allocation.id, 0, 2ULL * chunk_bytes,
                                                        xvram::residency::AccessMode::read}};
  CHECK(runtime.execute(access, 0, [](const auto&) {
    return xvram::residency::RuntimeStatus::success;
  }) == xvram::residency::RuntimeStatus::budget_pressure);
  CHECK(cuda.injection.triggered);
  CHECK(!runtime.poisoned());
  CHECK(!runtime.telemetry().quarantined);
  CHECK(runtime.telemetry().unsafe_remaps == 0U);
  CHECK(runtime.telemetry().target_oom_retries == 1U);
  CHECK(runtime.telemetry().budget_shrinks == 1U);
  CHECK(cuda.unmap_calls == 1U);
  CHECK(cuda.release_calls == 1U);
  CHECK(cuda.mappings.empty());
  CHECK(runtime.close() == xvram::residency::RuntimeStatus::success);
  CHECK(cuda.violations.empty());
}

void runtime_stall_timeout_quarantines_generation_test() {
  const CheckContext context("runtime stall timeout quarantines the in-flight generation");
  FakeCuda cuda;
  CudaApi api(CudaApi::InjectedDispatch{});
  cuda.install(api);
  const ActiveFake active(cuda);

  xvram::residency::RuntimeConfig config;
  config.chunk_bytes = chunk_bytes;
  config.cache_target_bytes = 8ULL * chunk_bytes;
  config.device_headroom_bytes = chunk_bytes;
  config.staging_slots = 2;
  config.stall_timeout = std::chrono::milliseconds(1);
  xvram::residency::Runtime runtime(api, config);
  CHECK(runtime.setup() == xvram::residency::RuntimeStatus::success);
  xvram::residency::RuntimeAllocation allocation;
  CHECK(runtime.allocate(chunk_bytes, xvram::residency::ResidencyHint::normal, allocation) ==
        xvram::residency::RuntimeStatus::success);

  cuda.injection.site = FaultSite::event_query;
  cuda.injection.result = CUDA_ERROR_NOT_READY;
  cuda.injection.persistent = true;
  const std::array access{xvram::residency::AccessRange{allocation.id, 0, chunk_bytes,
                                                        xvram::residency::AccessMode::read}};
  CHECK(runtime.execute(access, 0, [](const auto&) {
    return xvram::residency::RuntimeStatus::success;
  }) == xvram::residency::RuntimeStatus::timeout);
  CHECK(runtime.poisoned());
  CHECK(runtime.telemetry().quarantined);
  const std::uint64_t mappings_before_retry = cuda.map_calls;
  CHECK(runtime.execute(access, 0, [](const auto&) {
    return xvram::residency::RuntimeStatus::success;
  }) == xvram::residency::RuntimeStatus::invalid_argument);
  CHECK(cuda.map_calls == mappings_before_retry);
  CHECK(runtime.close() == xvram::residency::RuntimeStatus::cleanup_failure);
  CHECK(cuda.release_calls == 0U);
  CHECK(std::any_of(cuda.physical.begin(), cuda.physical.end(),
                    [](const auto& pair) { return !pair.second.released; }));
  CHECK(cuda.violations.empty());
}

void runtime_unsealed_external_lease_retains_physical_handle_test() {
  const CheckContext context("unsealed external lease retains its physical handle");
  FakeCuda cuda;
  CudaApi api(CudaApi::InjectedDispatch{});
  cuda.install(api);
  const ActiveFake active(cuda);

  xvram::residency::RuntimeConfig config;
  config.chunk_bytes = chunk_bytes;
  config.cache_target_bytes = 8ULL * chunk_bytes;
  config.device_headroom_bytes = chunk_bytes;
  config.staging_slots = 2;
  config.stall_timeout = std::chrono::milliseconds(100);
  xvram::residency::Runtime runtime(api, config);
  CHECK(runtime.setup() == xvram::residency::RuntimeStatus::success);
  xvram::residency::RuntimeAllocation allocation;
  CHECK(runtime.allocate(chunk_bytes, xvram::residency::ResidencyHint::normal, allocation) ==
        xvram::residency::RuntimeStatus::success);

  const std::array access{xvram::residency::AccessRange{allocation.id, 0, chunk_bytes,
                                                        xvram::residency::AccessMode::read}};
  xvram::residency::ExternalLease lease;
  CHECK(runtime.acquire_external({access, 0}, lease) == xvram::residency::RuntimeStatus::success);
  CHECK(lease.id);
  CHECK(lease.stream != nullptr);
  CHECK(cuda.mappings.size() == 1U);

  // The client owns the runtime stream after acquire. Without seal there is no observable event
  // boundary proving that the mapped allocation is idle, so close must retain every asynchronous
  // resource until the isolated worker exits.
  const std::size_t pinned_before_close = cuda.pinned.size();
  const std::size_t events_before_close = cuda.events.size();
  const std::size_t streams_before_close = cuda.streams.size();
  CHECK(runtime.close() == xvram::residency::RuntimeStatus::cleanup_failure);
  CHECK(runtime.poisoned());
  CHECK(runtime.telemetry().quarantined);
  CHECK(cuda.release_calls == 0U);
  CHECK(std::any_of(cuda.physical.begin(), cuda.physical.end(),
                    [](const auto& pair) { return !pair.second.released; }));
  CHECK(cuda.pinned.size() == pinned_before_close);
  CHECK(cuda.events.size() == events_before_close);
  CHECK(cuda.streams.size() == streams_before_close);
  CHECK(cuda.destroyed_contexts.empty());
  CHECK(cuda.reservations.size() == 1U);
  CHECK(cuda.mappings.size() == 1U);
  CHECK(cuda.violations.empty());
}

void runtime_event_record_failure_preserves_async_resources_test() {
  const CheckContext context("runtime event-record failure preserves asynchronous resources");
  FakeCuda cuda;
  CudaApi api(CudaApi::InjectedDispatch{});
  cuda.install(api);
  const ActiveFake active(cuda);

  xvram::residency::RuntimeConfig config;
  config.chunk_bytes = chunk_bytes;
  config.cache_target_bytes = 8ULL * chunk_bytes;
  config.device_headroom_bytes = chunk_bytes;
  config.staging_slots = 2;
  config.stall_timeout = std::chrono::milliseconds(100);
  xvram::residency::Runtime runtime(api, config);
  CHECK(runtime.setup() == xvram::residency::RuntimeStatus::success);
  xvram::residency::RuntimeAllocation allocation;
  CHECK(runtime.allocate(chunk_bytes, xvram::residency::ResidencyHint::normal, allocation) ==
        xvram::residency::RuntimeStatus::success);

  cuda.injection.site = FaultSite::event_record;
  cuda.injection.result = CUDA_ERROR_UNKNOWN;
  const std::array access{xvram::residency::AccessRange{allocation.id, 0, chunk_bytes,
                                                        xvram::residency::AccessMode::read}};
  CHECK(runtime.execute(access, 0, [](const auto&) {
    return xvram::residency::RuntimeStatus::success;
  }) == xvram::residency::RuntimeStatus::poisoned);
  CHECK(runtime.poisoned());
  CHECK(runtime.telemetry().quarantined);
  const std::size_t pinned_before_close = cuda.pinned.size();
  const std::size_t events_before_close = cuda.events.size();
  const std::size_t streams_before_close = cuda.streams.size();

  CHECK(runtime.close() == xvram::residency::RuntimeStatus::cleanup_failure);
  CHECK(cuda.release_calls == 0U);
  CHECK(!cuda.quarantined_release_observed);
  CHECK(std::any_of(cuda.physical.begin(), cuda.physical.end(),
                    [](const auto& pair) { return !pair.second.released; }));
  CHECK(cuda.pinned.size() == pinned_before_close);
  CHECK(cuda.events.size() == events_before_close);
  CHECK(cuda.streams.size() == streams_before_close);
  CHECK(cuda.destroyed_contexts.empty());
  CHECK(cuda.reservations.size() == 1U);
  CHECK(cuda.mappings.size() == 1U);
  CHECK(cuda.violations.empty());
}

void runtime_budget_pressure_shrinks_before_failure_test() {
  const CheckContext context("runtime budget pressure drains before failure");
  FakeCuda cuda;
  CudaApi api(CudaApi::InjectedDispatch{});
  cuda.install(api);
  const ActiveFake active(cuda);

  xvram::residency::RuntimeConfig config;
  config.chunk_bytes = chunk_bytes;
  config.cache_target_bytes = 8ULL * chunk_bytes;
  config.device_headroom_bytes = chunk_bytes;
  config.staging_slots = 2;
  config.stall_timeout = std::chrono::milliseconds(100);
  config.budget_poll_interval = std::chrono::milliseconds(1);
  xvram::residency::Runtime runtime(api, config);
  CHECK(runtime.setup() == xvram::residency::RuntimeStatus::success);

  xvram::residency::RuntimeAllocation allocation;
  CHECK(runtime.allocate(6ULL * chunk_bytes, xvram::residency::ResidencyHint::normal, allocation) ==
        xvram::residency::RuntimeStatus::success);
  const std::array access{xvram::residency::AccessRange{allocation.id, 0, 6ULL * chunk_bytes,
                                                        xvram::residency::AccessMode::read}};
  CHECK(runtime.execute(access, 0, [](const auto&) {
    return xvram::residency::RuntimeStatus::success;
  }) == xvram::residency::RuntimeStatus::success);
  CHECK(runtime.telemetry().resident_bytes == 6ULL * chunk_bytes);

  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  cuda.free_memory_bytes = 0;
  CHECK(runtime.execute(access, 0, [](const auto&) {
    return xvram::residency::RuntimeStatus::success;
  }) == xvram::residency::RuntimeStatus::budget_pressure);
  CHECK(runtime.telemetry().budget_shrinks == 1U);
  CHECK(runtime.telemetry().resident_bytes == 5ULL * chunk_bytes);
  CHECK(runtime.telemetry().handles_released == 1U);
  CHECK(cuda.mappings.size() == 5U);

  CHECK(runtime.close() == xvram::residency::RuntimeStatus::success);
  CHECK(cuda.mappings.empty());
  CHECK(cuda.reservations.empty());
  CHECK(cuda.violations.empty());
}

void runtime_explicit_target_is_post_headroom_cap_test() {
  const CheckContext context("runtime explicit target is post-headroom cap");
  FakeCuda cuda;
  CudaApi api(CudaApi::InjectedDispatch{});
  cuda.install(api);
  const ActiveFake active(cuda);

  xvram::residency::RuntimeConfig config;
  config.chunk_bytes = chunk_bytes;
  config.cache_target_bytes = 4ULL * chunk_bytes;
  config.device_headroom_bytes = 8ULL * chunk_bytes;
  config.staging_slots = 2;
  config.budget_poll_interval = std::chrono::milliseconds(1);
  xvram::residency::Runtime runtime(api, config);
  CHECK(runtime.setup() == xvram::residency::RuntimeStatus::success);
  CHECK(runtime.target_bytes() == 4ULL * chunk_bytes);

  xvram::residency::RuntimeAllocation allocation;
  CHECK(runtime.allocate(chunk_bytes, xvram::residency::ResidencyHint::normal, allocation) ==
        xvram::residency::RuntimeStatus::success);
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  const std::array access{xvram::residency::AccessRange{allocation.id, 0, chunk_bytes,
                                                        xvram::residency::AccessMode::read}};
  CHECK(runtime.execute(access, 0, [](const auto&) {
    return xvram::residency::RuntimeStatus::success;
  }) == xvram::residency::RuntimeStatus::success);
  CHECK(runtime.target_bytes() == 4ULL * chunk_bytes);
  CHECK(runtime.close() == xvram::residency::RuntimeStatus::success);
  CHECK(cuda.violations.empty());
}

void invalid_event_query_test() {
  Run run = execute(FaultMode::invalid_event_query);
  CHECK(run.result.exit_code == 27);
  CHECK(run.result.status == "failed");
  CHECK(run.result.failure.has_value());
  CHECK(run.result.failure->operation == "cuEventQuery(h2d_done)");
  CHECK(run.cuda.event_fault_used);
  CHECK(run.cuda.event_query_calls == 1);
  CHECK(!run.result.cleanup.complete.value_or(true));
  CHECK(run.cuda.violations.empty());
  CHECK(run.cuda.address_free_calls == 0);
  CHECK(run.cuda.reservations.size() == 1);
  CHECK(!run.cuda.mappings.empty());
  CHECK(!run.result.cleanup.events_drained.value_or(true));
  CHECK(!run.result.cleanup.mappings_removed.value_or(true));
  CHECK(!run.result.cleanup.virtual_reservations_released.value_or(true));
  CHECK(run.result.cleanup.pinned_staging_released.value_or(false));
  CHECK(run.result.cleanup.context_destroyed.value_or(false));
}

void failed_unmap_quarantine_test() {
  Run run = execute(FaultMode::fail_first_unmap);
  CHECK(run.result.exit_code == 27);
  CHECK(run.result.status == "failed");
  CHECK(run.result.failure.has_value());
  CHECK(run.result.failure->operation == "cuMemUnmap");
  CHECK(run.cuda.unmap_fault_used);
  CHECK(run.cuda.quarantined_release_observed);
  CHECK(run.cuda.release_calls > 0);
  CHECK(run.cuda.address_free_calls == 0);
  CHECK(run.cuda.reservations.size() == 1);
  CHECK(!run.cuda.mappings.empty());
  CHECK(std::all_of(run.cuda.mappings.begin(), run.cuda.mappings.end(), [&](const auto& pair) {
    const auto allocation = run.cuda.physical.find(pair.second.handle);
    return allocation != run.cuda.physical.end() && allocation->second.released;
  }));
  CHECK(!run.result.cleanup.complete.value_or(true));
  CHECK(!run.result.cleanup.mappings_removed.value_or(true));
  CHECK(!run.result.cleanup.virtual_reservations_released.value_or(true));
  CHECK(run.result.cleanup.physical_handles_released.value_or(false));
  CHECK(run.result.cleanup.pinned_staging_released.value_or(false));
  CHECK(run.result.cleanup.context_destroyed.value_or(false));
  CHECK(run.result.cache.unsafe_remap_count.value_or(1) == 0);
  CHECK(run.cuda.violations.empty());
}

void dynamic_budget_pressure_test() {
  Run run = execute(FaultMode::none, true);
  CHECK(run.result.exit_code == 25);
  CHECK(run.result.status == "failed");
  CHECK(run.result.failure.has_value());
  CHECK(run.result.failure->stage == "budget");
  CHECK(run.result.failure->operation == "calculate_target" ||
        run.result.failure->operation == "target_hysteresis");
  CHECK(run.result.cleanup.complete.value_or(false));
  CHECK(run.cuda.kernel_calls == 0);
  CHECK(run.cuda.violations.empty());
  CHECK(run.cuda.reservations.empty());
  CHECK(run.cuda.mappings.empty());
  CHECK(run.cuda.linear.empty());
  CHECK(run.cuda.pinned.empty());
}

enum class LedgerExpectation {
  complete,
  transactions_retained,
  mappings_quarantined,
  handles_retained,
  events_retained,
  context_failed,
};

struct FaultCase {
  std::string_view name;
  FaultSite site = FaultSite::none;
  Result injected_result = CUDA_ERROR_UNKNOWN;
  bool persistent = false;
  int expected_exit = 27;
  std::string_view expected_stage;
  std::string_view expected_operation;
  LedgerExpectation ledger = LedgerExpectation::complete;
  std::string_view expected_diagnostic;
};

void check_complete_fake_cleanup(const Run& run) {
  CHECK(run.cuda.current_context == nullptr);
  CHECK(run.cuda.reservations.empty());
  CHECK(run.cuda.mappings.empty());
  CHECK(run.cuda.linear.empty());
  CHECK(run.cuda.pinned.empty());
  CHECK(run.cuda.streams.empty());
  CHECK(run.cuda.events.empty());
  CHECK(std::all_of(run.cuda.physical.begin(), run.cuda.physical.end(),
                    [](const auto& pair) { return pair.second.released; }));
}

void check_fault_ledger(const Run& run, const LedgerExpectation expectation) {
  switch (expectation) {
  case LedgerExpectation::complete:
    CHECK(run.result.cleanup.complete.value_or(false));
    check_complete_fake_cleanup(run);
    break;
  case LedgerExpectation::transactions_retained:
    CHECK(!run.result.cleanup.complete.value_or(true));
    CHECK(!run.result.cleanup.transactions_drained.value_or(true));
    break;
  case LedgerExpectation::mappings_quarantined:
    CHECK(!run.result.cleanup.complete.value_or(true));
    CHECK(!run.result.cleanup.mappings_removed.value_or(true));
    CHECK(!run.result.cleanup.virtual_reservations_released.value_or(true));
    CHECK(!run.cuda.mappings.empty());
    CHECK(!run.cuda.reservations.empty());
    break;
  case LedgerExpectation::handles_retained:
    CHECK(!run.result.cleanup.complete.value_or(true));
    CHECK(!run.result.cleanup.physical_handles_released.value_or(true));
    CHECK(std::any_of(run.cuda.physical.begin(), run.cuda.physical.end(),
                      [](const auto& pair) { return !pair.second.released; }));
    break;
  case LedgerExpectation::events_retained:
    CHECK(!run.result.cleanup.complete.value_or(true));
    CHECK(!run.result.cleanup.events_destroyed.value_or(true));
    CHECK(!run.cuda.events.empty());
    break;
  case LedgerExpectation::context_failed:
    CHECK(!run.result.cleanup.complete.value_or(true));
    CHECK(!run.result.cleanup.context_destroyed.value_or(true));
    break;
  }
}

void fault_injection_matrix_test() {
  static constexpr FaultCase cases[] = {
      {"initialization prerequisite",
       FaultSite::initialize,
       CUDA_ERROR_INVALID_VALUE,
       false,
       23,
       "preflight",
       "cuInit",
       LedgerExpectation::complete,
       {}},
      {"no CUDA device",
       FaultSite::no_device,
       CUDA_SUCCESS,
       false,
       23,
       "preflight",
       "select_device",
       LedgerExpectation::complete,
       {}},
      {"context setup",
       FaultSite::context_create,
       CUDA_ERROR_UNKNOWN,
       false,
       27,
       "preflight",
       "cuCtxCreate",
       LedgerExpectation::complete,
       {}},
      {"pinned setup OOM",
       FaultSite::host_allocate,
       CUDA_ERROR_OUT_OF_MEMORY,
       false,
       25,
       "setup",
       "cuMemHostAlloc",
       LedgerExpectation::complete,
       {}},
      {"physical frame OOM",
       FaultSite::memory_create,
       CUDA_ERROR_OUT_OF_MEMORY,
       true,
       25,
       "execution",
       "cuMemCreate",
       LedgerExpectation::complete,
       {}},
      {"VMM map",
       FaultSite::map,
       CUDA_ERROR_UNKNOWN,
       false,
       27,
       "execution",
       "cuMemMap",
       LedgerExpectation::complete,
       {}},
      {"VMM SetAccess",
       FaultSite::set_access,
       CUDA_ERROR_UNKNOWN,
       false,
       27,
       "execution",
       "cuMemSetAccess",
       LedgerExpectation::mappings_quarantined,
       {}},
      {"mapping H2D",
       FaultSite::h2d,
       CUDA_ERROR_UNKNOWN,
       false,
       27,
       "execution",
       "cuMemcpyHtoDAsync",
       LedgerExpectation::mappings_quarantined,
       {}},
      {"kernel launch",
       FaultSite::kernel,
       CUDA_ERROR_UNKNOWN,
       false,
       27,
       "execution",
       "cuLaunchKernel",
       LedgerExpectation::mappings_quarantined,
       {}},
      {"event record",
       FaultSite::event_record,
       CUDA_ERROR_UNKNOWN,
       false,
       27,
       "execution",
       "cuEventRecord(h2d_start)",
       LedgerExpectation::mappings_quarantined,
       {}},
      {"event query",
       FaultSite::event_query,
       CUDA_ERROR_INVALID_VALUE,
       false,
       27,
       "execution",
       "cuEventQuery(h2d_done)",
       LedgerExpectation::mappings_quarantined,
       {}},
      {"mapping D2H",
       FaultSite::d2h,
       CUDA_ERROR_UNKNOWN,
       false,
       27,
       "execution",
       "cuMemcpyDtoHAsync",
       LedgerExpectation::mappings_quarantined,
       {}},
      {"VMM unmap",
       FaultSite::unmap,
       CUDA_ERROR_UNKNOWN,
       false,
       27,
       "cleanup",
       "cuMemUnmap",
       LedgerExpectation::mappings_quarantined,
       {}},
      {"handle release", FaultSite::release, CUDA_ERROR_UNKNOWN, false, 27, "cleanup",
       "resource_ledger", LedgerExpectation::handles_retained, "cuMemRelease"},
      {"event cleanup", FaultSite::event_destroy, CUDA_ERROR_UNKNOWN, false, 27, "cleanup",
       "resource_ledger", LedgerExpectation::events_retained, "cuEventDestroy"},
      {"context cleanup", FaultSite::context_destroy, CUDA_ERROR_UNKNOWN, false, 27, "cleanup",
       "resource_ledger", LedgerExpectation::context_failed, "cuCtxDestroy"},
      {"verification corruption",
       FaultSite::corrupt_kernel_output,
       CUDA_SUCCESS,
       false,
       24,
       "verification",
       "verification_token",
       LedgerExpectation::complete,
       {}},
  };

  for (const FaultCase& fault_case : cases) {
    const CheckContext context(fault_case.name);
    FaultInjection injection;
    injection.site = fault_case.site;
    injection.result = fault_case.injected_result;
    injection.persistent = fault_case.persistent;
    const Run run = execute(FaultMode::none, false, injection);
    CHECK(run.cuda.injection.triggered);
    CHECK(run.result.exit_code == fault_case.expected_exit);
    CHECK(run.result.status == (fault_case.expected_exit == 23 ? "skipped" : "failed"));
    CHECK(run.result.failure.has_value());
    if (run.result.failure.has_value()) {
      CHECK(run.result.failure->stage == fault_case.expected_stage);
      CHECK(run.result.failure->operation == fault_case.expected_operation);
    }
    CHECK(run.cuda.violations.empty());
    check_fault_ledger(run, fault_case.ledger);
    if (!fault_case.expected_diagnostic.empty()) {
      CHECK(std::any_of(run.result.diagnostics.begin(), run.result.diagnostics.end(),
                        [&](const auto& diagnostic) {
                          return diagnostic.operation == fault_case.expected_diagnostic;
                        }));
    }
  }
}

} // namespace

int main() {
  completed_sequential_test();
  runtime_allocation_release_unmaps_test();
  runtime_disabled_compression_never_loads_nvcomp_test();
  runtime_raw_transfer_trace_generations_test();
  runtime_safe_callback_skip_keeps_session_reusable_test();
  runtime_external_lease_and_discard_dead_test();
  compressed_discarded_full_write_reuse_test();
  adaptive_cpu_rejection_preserves_raw_generation_test();
  cpu_compression_reuses_only_adjacent_identical_blocks_test();
  adaptive_cpu_compression_commits_after_reuse_history_test();
  runtime_compressed_backing_path_provenance_test();
  automatic_host_budget_uses_one_post_context_sample_test();
  adaptive_gpu_encode_rejection_uses_admitted_raw_spill_test();
  adaptive_learned_raw_skips_later_gpu_encode_test();
  adaptive_gpu_candidate_quarantine_remains_unresolved_test();
  runtime_codec_pinned_budget_fallback_preserves_device_peak_test();
  runtime_post_map_backing_budget_failure_test(false);
  runtime_post_map_backing_budget_failure_test(true);
  runtime_spill_pressure_recovery_test(false);
  runtime_spill_pressure_recovery_test(true);
  runtime_spill_pressure_no_safe_victim_test(false);
  runtime_spill_pressure_no_safe_victim_test(true);
  runtime_spill_pressure_active_lease_is_protected_test();
  runtime_spill_pressure_event_failure_quarantines_test(FaultSite::event_record);
  runtime_spill_pressure_event_failure_quarantines_test(FaultSite::event_query);
  runtime_codec_budget_suspension_test();
  runtime_codec_restore_setup_failure_falls_back_then_retries_test();
  runtime_restored_codec_unknown_event_quarantines_test();
  runtime_d2h_allocation_class_accounting_test();
  runtime_external_poll_not_ready_test();
  runtime_external_duration_guard_test();
  runtime_failed_callback_retires_before_poison_test();
  runtime_set_access_rollback_test();
  runtime_set_access_rollback_quarantine_test();
  runtime_close_attempts_every_mapping_test();
  runtime_attach_current_context_ownership_test();
  runtime_staging_pool_configuration_test();
  runtime_retries_one_frame_oom_test();
  runtime_oom_shrink_defers_pinned_working_set_test();
  runtime_stall_timeout_quarantines_generation_test();
  runtime_unsealed_external_lease_retains_physical_handle_test();
  runtime_event_record_failure_preserves_async_resources_test();
  runtime_budget_pressure_shrinks_before_failure_test();
  runtime_explicit_target_is_post_headroom_cap_test();
  invalid_event_query_test();
  failed_unmap_quarantine_test();
  dynamic_budget_pressure_test();
  fault_injection_matrix_test();
  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
    return 1;
  }
  std::cout << "all residency executor fake tests passed\n";
  return 0;
}
