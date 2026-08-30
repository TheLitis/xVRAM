#define NOMINMAX

#include "platform/cuda/cuda_api.hpp"
#include "residency/executor.hpp"
#include "residency/workload.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <string_view>
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
  std::map<Event, EventState, std::less<Event>> events;
  std::map<std::uint64_t, DevicePointer> stable_addresses;
  std::uint64_t next_token = 1;
  GenericAllocationHandle next_handle = 1;
  DevicePointer next_reservation = 0x100001000ULL;
  DevicePointer next_linear = 0x700000000ULL;
  Context current_context = nullptr;
  std::size_t maximum_active_mappings = 0;
  std::uint64_t map_calls = 0;
  std::uint64_t set_access_calls = 0;
  std::uint64_t unmap_calls = 0;
  std::uint64_t release_calls = 0;
  std::uint64_t address_free_calls = 0;
  std::uint64_t event_query_calls = 0;
  std::uint64_t kernel_calls = 0;
  std::uint64_t d2h_mapping_calls = 0;

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

Result CUDAAPI fake_context_create(Context* context, unsigned int, CUdevice) {
  if (fake().injection.should_inject(FaultSite::context_create)) {
    return fake().injection.result;
  }
  *context = fake().make_token<Context>();
  fake().current_context = *context;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_context_destroy(const Context context) {
  if (fake().injection.should_inject(FaultSite::context_destroy)) {
    return fake().injection.result;
  }
  if (fake().current_context == context) {
    fake().current_context = nullptr;
  }
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_mem_get_info(std::size_t* free_bytes, std::size_t* total_bytes) {
  *free_bytes = static_cast<std::size_t>(16ULL * mib);
  *total_bytes = static_cast<std::size_t>(mib);
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
    if (!fake().unmap_fault_used) {
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
  if (mapping == fake().mappings.end() || mapping->second.bytes != bytes ||
      bytes != static_cast<std::size_t>(chunk_bytes)) {
    fake().violation("unmap did not remove one exact full mapped chunk");
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (!mapping->second.access_set) {
    fake().violation("mapping reached unmap without SetAccess");
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
  fake().streams.emplace(created, FakeCuda::StreamState{});
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

Result CUDAAPI fake_stream_wait_event(const Stream stream, const Event event, unsigned int) {
  return fake().streams.contains(stream) && fake().events.contains(event)
             ? CUDA_SUCCESS
             : CUDA_ERROR_INVALID_HANDLE;
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
  *milliseconds = 0.05F;
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
      std::string_view(name) != "xvram_residency_workload_v1") {
    return CUDA_ERROR_NOT_FOUND;
  }
  *function = fake().make_token<Function>();
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_module_unload(Module) {
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_launch_kernel(Function, const unsigned int blocks_x, unsigned int, unsigned int,
                                  const unsigned int threads_x, unsigned int, unsigned int,
                                  unsigned int, const Stream stream, void** parameters, void**) {
  ++fake().kernel_calls;
  if (fake().injection.should_inject(FaultSite::kernel)) {
    return fake().injection.result;
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
  api.stream_wait_event_ = fake_stream_wait_event;
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
