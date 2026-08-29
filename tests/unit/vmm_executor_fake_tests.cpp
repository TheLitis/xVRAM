#define NOMINMAX

#include "platform/cuda/cuda_api.hpp"
#include "vmm_poc/executor.hpp"
#include "vmm_poc/workload.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
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
constexpr std::uint64_t fake_chunk_bytes = 64ULL * kib;
constexpr std::uint64_t fake_logical_bytes = 2ULL * mib + xvram::vmm_poc::word_bytes;

int failures = 0;

void check(const bool condition, const char* expression, const int line) {
  if (!condition) {
    std::cerr << "line " << line << ": check failed: " << expression << '\n';
    ++failures;
  }
}

#define CHECK(expression) check(static_cast<bool>(expression), #expression, __LINE__)

enum class Call {
  none,
  module_load,
  address_reserve,
  mem_create,
  host_alloc,
  event_create,
  stream_create,
  map,
  set_access,
  h2d,
  event_record,
  stream_wait,
  kernel,
  d2h,
  event_query,
  event_elapsed,
  unmap,
  event_synchronize,
  stream_synchronize,
  mem_release,
  address_free,
  event_destroy,
  stream_destroy,
  module_unload,
  host_free,
  context_destroy,
  context_set_current,
};

struct Fault {
  Call call = Call::none;
  std::size_t occurrence = 1;
  Result result = CUDA_ERROR_UNKNOWN;
};

struct FakeCuda {
  struct Token {
    std::uint64_t id = 0;
  };

  struct Allocation {
    std::vector<std::byte> storage;
    std::uint64_t map_count = 0;
    bool released = false;
  };

  struct Mapping {
    GenericAllocationHandle handle = 0;
    std::size_t bytes = 0;
    bool access_attempted = false;
    bool access_set = false;
    bool work_started = false;
    bool completion_recorded = false;
    bool completion_observed = false;
    std::optional<std::uint64_t> global_word_start;
  };

  struct StreamState {
    std::optional<DevicePointer> pending_completion_address;
  };

  struct EventState {
    bool recorded = false;
    std::size_t query_count = 0;
    std::optional<DevicePointer> completion_address;
  };

  Fault fault;
  std::optional<Fault> secondary_fault;
  bool return_not_ready_once = false;
  std::map<Call, std::size_t> call_counts;
  std::vector<std::string> violations;
  std::vector<std::unique_ptr<Token>> tokens;
  std::map<GenericAllocationHandle, Allocation> allocations;
  std::map<DevicePointer, Mapping> mappings;
  std::map<Stream, StreamState, std::less<Stream>> streams;
  std::map<Event, EventState, std::less<Event>> events;
  std::map<void*, std::unique_ptr<std::byte[]>, std::less<void*>> pinned;
  std::map<std::uint64_t, DevicePointer> stable_addresses;
  std::vector<std::uint64_t> launch_global_starts;
  std::vector<std::uint64_t> unmap_global_starts;
  std::uint64_t next_token = 1;
  GenericAllocationHandle next_handle = 1;
  DevicePointer reservation = 0;
  std::size_t reservation_bytes = 0;
  Context current_context = nullptr;
  std::size_t maximum_active_mappings = 0;
  std::size_t stream_wait_count = 0;

  [[nodiscard]] bool should_fail(const Call call) {
    const std::size_t occurrence = ++call_counts[call];
    const auto matches = [&](const Fault& candidate) {
      return candidate.call == call && candidate.occurrence == occurrence;
    };
    return matches(fault) || (secondary_fault.has_value() && matches(*secondary_fault));
  }

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

  [[nodiscard]] Mapping* find_mapping(const DevicePointer address) {
    const auto found = mappings.find(address);
    return found == mappings.end() ? nullptr : &found->second;
  }

  [[nodiscard]] Allocation* allocation_for(const DevicePointer address) {
    Mapping* mapping = find_mapping(address);
    if (mapping == nullptr) {
      return nullptr;
    }
    const auto found = allocations.find(mapping->handle);
    return found == allocations.end() ? nullptr : &found->second;
  }

  void observe_completion(EventState& event) {
    if (!event.completion_address.has_value()) {
      return;
    }
    Mapping* mapping = find_mapping(*event.completion_address);
    if (mapping != nullptr) {
      mapping->completion_observed = true;
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
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_driver_version(int* version) {
  *version = 12'000;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_error_name(const Result result, const char** name) {
  *name = result == CUDA_ERROR_OUT_OF_MEMORY ? "CUDA_ERROR_OUT_OF_MEMORY" : "CUDA_ERROR_UNKNOWN";
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_error_string(Result, const char** message) {
  *message = "injected fake CUDA failure";
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_device_count(int* count) {
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
  constexpr std::string_view value = "Fake CUDA VMM device";
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
    *value = 4;
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
  *granularity = option == CU_MEM_ALLOC_GRANULARITY_MINIMUM
                     ? static_cast<std::size_t>(4ULL * kib)
                     : static_cast<std::size_t>(fake_chunk_bytes);
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_context_get_current(Context* context) {
  *context = fake().current_context;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_context_set_current(const Context context) {
  if (fake().should_fail(Call::context_set_current)) {
    return fake().fault.result;
  }
  fake().current_context = context;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_context_create(Context* context, unsigned int, CUdevice) {
  *context = fake().make_token<Context>();
  fake().current_context = *context;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_context_destroy(Context context) {
  if (fake().should_fail(Call::context_destroy)) {
    return fake().fault.result;
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
  if (fake().should_fail(Call::address_reserve)) {
    return fake().fault.result;
  }
  fake().reservation = static_cast<DevicePointer>(0x100001000ULL);
  fake().reservation_bytes = bytes;
  *address = fake().reservation;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_address_free(const DevicePointer address, const std::size_t bytes) {
  if (fake().should_fail(Call::address_free)) {
    return fake().fault.result;
  }
  if (address != fake().reservation || bytes != fake().reservation_bytes) {
    fake().violation("virtual reservation was not freed with its original base and size");
    return CUDA_ERROR_INVALID_VALUE;
  }
  fake().reservation = 0;
  fake().reservation_bytes = 0;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_mem_create(GenericAllocationHandle* handle, const std::size_t bytes,
                               const CUmemAllocationProp*, unsigned long long) {
  if (fake().should_fail(Call::mem_create)) {
    return fake().fault.result;
  }
  const GenericAllocationHandle created = fake().next_handle++;
  FakeCuda::Allocation allocation;
  allocation.storage.resize(bytes);
  fake().allocations.emplace(created, std::move(allocation));
  *handle = created;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_mem_release(const GenericAllocationHandle handle) {
  if (fake().should_fail(Call::mem_release)) {
    return fake().fault.result;
  }
  const auto found = fake().allocations.find(handle);
  if (found == fake().allocations.end() || found->second.released) {
    fake().violation("physical handle was released more than once");
    return CUDA_ERROR_INVALID_HANDLE;
  }
  found->second.released = true;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_mem_map(const DevicePointer address, const std::size_t bytes, std::size_t,
                            const GenericAllocationHandle handle, unsigned long long) {
  if (fake().should_fail(Call::map)) {
    return fake().fault.result;
  }
  const auto allocation = fake().allocations.find(handle);
  if (allocation == fake().allocations.end() || allocation->second.released ||
      allocation->second.storage.size() != bytes || fake().mappings.contains(address)) {
    fake().violation("invalid or overlapping physical mapping");
    return CUDA_ERROR_INVALID_VALUE;
  }
  FakeCuda::Mapping mapping;
  mapping.handle = handle;
  mapping.bytes = bytes;
  fake().mappings.emplace(address, mapping);
  ++allocation->second.map_count;
  fake().maximum_active_mappings = std::max(fake().maximum_active_mappings, fake().mappings.size());
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_mem_unmap(const DevicePointer address, const std::size_t bytes) {
  if (fake().should_fail(Call::unmap)) {
    return fake().fault.result;
  }
  const auto found = fake().mappings.find(address);
  if (found == fake().mappings.end()) {
    fake().violation("unmap did not reference a live mapping");
    return CUDA_ERROR_INVALID_VALUE;
  }
  const FakeCuda::Mapping& mapping = found->second;
  if (bytes != mapping.bytes || bytes != static_cast<std::size_t>(fake_chunk_bytes)) {
    fake().violation("unmap did not remove the exact full mapped chunk");
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (!mapping.access_attempted) {
    fake().violation("mapping was unmapped without a SetAccess attempt");
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (mapping.work_started && (!mapping.completion_recorded || !mapping.completion_observed)) {
    fake().violation("mapping was unmapped before a completion event boundary");
    return CUDA_ERROR_NOT_READY;
  }
  if (mapping.global_word_start.has_value()) {
    fake().unmap_global_starts.push_back(*mapping.global_word_start);
  }
  fake().mappings.erase(found);
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_mem_set_access(const DevicePointer address, const std::size_t bytes,
                                   const CUmemAccessDesc*, const std::size_t count) {
  FakeCuda::Mapping* mapping = fake().find_mapping(address);
  if (mapping != nullptr) {
    mapping->access_attempted = true;
  }
  if (fake().should_fail(Call::set_access)) {
    return fake().fault.result;
  }
  if (mapping == nullptr || mapping->bytes != bytes || count != 1U || mapping->access_set) {
    fake().violation("SetAccess was missing, duplicated, or applied to the wrong range");
    return CUDA_ERROR_INVALID_VALUE;
  }
  mapping->access_set = true;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_host_alloc(void** pointer, const std::size_t bytes, unsigned int) {
  if (fake().should_fail(Call::host_alloc)) {
    return fake().fault.result;
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
  if (fake().should_fail(Call::host_free)) {
    return fake().fault.result;
  }
  if (fake().pinned.erase(pointer) != 1U) {
    fake().violation("pinned staging was freed more than once");
    return CUDA_ERROR_INVALID_VALUE;
  }
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_h2d(const DevicePointer destination, const void* source,
                        const std::size_t bytes, Stream) {
  if (fake().should_fail(Call::h2d)) {
    return fake().fault.result;
  }
  FakeCuda::Mapping* mapping = fake().find_mapping(destination);
  FakeCuda::Allocation* allocation = fake().allocation_for(destination);
  if (mapping == nullptr || allocation == nullptr || !mapping->access_set ||
      bytes > mapping->bytes) {
    fake().violation("H2D used an inaccessible or invalid mapping");
    return CUDA_ERROR_INVALID_VALUE;
  }
  std::memcpy(allocation->storage.data(), source, bytes);
  mapping->work_started = true;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_d2h(void* destination, const DevicePointer source, const std::size_t bytes,
                        const Stream stream) {
  if (fake().should_fail(Call::d2h)) {
    return fake().fault.result;
  }
  FakeCuda::Mapping* mapping = fake().find_mapping(source);
  FakeCuda::Allocation* allocation = fake().allocation_for(source);
  const auto stream_found = fake().streams.find(stream);
  if (mapping == nullptr || allocation == nullptr || !mapping->access_set ||
      bytes > mapping->bytes || stream_found == fake().streams.end()) {
    fake().violation("D2H used an inaccessible mapping or invalid stream");
    return CUDA_ERROR_INVALID_VALUE;
  }
  std::memcpy(destination, allocation->storage.data(), bytes);
  stream_found->second.pending_completion_address = source;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_stream_create(Stream* stream, unsigned int) {
  if (fake().should_fail(Call::stream_create)) {
    return fake().fault.result;
  }
  const Stream created = fake().make_token<Stream>();
  fake().streams.emplace(created, FakeCuda::StreamState{});
  *stream = created;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_stream_destroy(const Stream stream) {
  if (fake().should_fail(Call::stream_destroy)) {
    return fake().fault.result;
  }
  if (fake().streams.erase(stream) != 1U) {
    fake().violation("stream was destroyed more than once");
    return CUDA_ERROR_INVALID_HANDLE;
  }
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_stream_synchronize(const Stream stream) {
  if (fake().should_fail(Call::stream_synchronize)) {
    return fake().fault.result;
  }
  return fake().streams.contains(stream) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_HANDLE;
}

Result CUDAAPI fake_stream_wait_event(const Stream stream, const Event event, unsigned int) {
  if (fake().should_fail(Call::stream_wait)) {
    return fake().fault.result;
  }
  const auto stream_found = fake().streams.find(stream);
  const auto event_found = fake().events.find(event);
  if (stream_found == fake().streams.end() || event_found == fake().events.end() ||
      !event_found->second.recorded) {
    fake().violation("stream waited on an unrecorded event");
    return CUDA_ERROR_INVALID_HANDLE;
  }
  ++fake().stream_wait_count;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_event_create(Event* event, unsigned int) {
  if (fake().should_fail(Call::event_create)) {
    return fake().fault.result;
  }
  const Event created = fake().make_token<Event>();
  fake().events.emplace(created, FakeCuda::EventState{});
  *event = created;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_event_destroy(const Event event) {
  if (fake().should_fail(Call::event_destroy)) {
    return fake().fault.result;
  }
  if (fake().events.erase(event) != 1U) {
    fake().violation("event was destroyed more than once");
    return CUDA_ERROR_INVALID_HANDLE;
  }
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_event_record(const Event event, const Stream stream) {
  if (fake().should_fail(Call::event_record)) {
    return fake().fault.result;
  }
  const auto event_found = fake().events.find(event);
  const auto stream_found = fake().streams.find(stream);
  if (event_found == fake().events.end() || stream_found == fake().streams.end()) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  FakeCuda::EventState& state = event_found->second;
  state.recorded = true;
  state.query_count = 0;
  state.completion_address.reset();
  if (stream_found->second.pending_completion_address.has_value()) {
    state.completion_address = stream_found->second.pending_completion_address;
    FakeCuda::Mapping* mapping = fake().find_mapping(*state.completion_address);
    if (mapping != nullptr) {
      mapping->completion_recorded = true;
    }
    stream_found->second.pending_completion_address.reset();
  }
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_event_query(const Event event) {
  if (fake().should_fail(Call::event_query)) {
    return fake().fault.result;
  }
  const auto found = fake().events.find(event);
  if (found == fake().events.end() || !found->second.recorded) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  FakeCuda::EventState& state = found->second;
  if (fake().return_not_ready_once && state.query_count++ == 0U) {
    return CUDA_ERROR_NOT_READY;
  }
  fake().observe_completion(state);
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_event_synchronize(const Event event) {
  if (fake().should_fail(Call::event_synchronize)) {
    return fake().fault.result;
  }
  const auto found = fake().events.find(event);
  if (found == fake().events.end() || !found->second.recorded) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  fake().observe_completion(found->second);
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_event_elapsed(float* milliseconds, const Event start, const Event end) {
  if (fake().should_fail(Call::event_elapsed)) {
    return fake().fault.result;
  }
  if (!fake().events.contains(start) || !fake().events.contains(end)) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  *milliseconds = 0.125F;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_module_load(Module* module, const void* image) {
  if (fake().should_fail(Call::module_load)) {
    return fake().fault.result;
  }
  if (image == nullptr) {
    return CUDA_ERROR_INVALID_IMAGE;
  }
  *module = fake().make_token<Module>();
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_module_get_function(Function* function, const Module module, const char* name) {
  if (module == nullptr || std::string_view(name) != "xvram_vmm_transform_v1") {
    return CUDA_ERROR_NOT_FOUND;
  }
  *function = fake().make_token<Function>();
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_module_unload(Module) {
  if (fake().should_fail(Call::module_unload)) {
    return fake().fault.result;
  }
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_launch_kernel(Function, const unsigned int blocks_x, unsigned int, unsigned int,
                                  const unsigned int threads_x, unsigned int, unsigned int,
                                  unsigned int, const Stream stream, void** parameters, void**) {
  if (fake().should_fail(Call::kernel)) {
    return fake().fault.result;
  }
  if (!fake().streams.contains(stream) || parameters == nullptr || blocks_x != 16U ||
      threads_x == 0U || threads_x > 256U || threads_x % 32U != 0U) {
    fake().violation("kernel launch geometry or stream was invalid");
    return CUDA_ERROR_INVALID_VALUE;
  }

  const DevicePointer address = *static_cast<DevicePointer*>(parameters[0]);
  const std::uint64_t word_count = *static_cast<std::uint64_t*>(parameters[1]);
  const std::uint64_t global_start = *static_cast<std::uint64_t*>(parameters[2]);
  const std::uint32_t pass = *static_cast<std::uint32_t*>(parameters[3]);
  const std::uint64_t seed = *static_cast<std::uint64_t*>(parameters[4]);
  FakeCuda::Mapping* mapping = fake().find_mapping(address);
  FakeCuda::Allocation* allocation = fake().allocation_for(address);
  if (mapping == nullptr || allocation == nullptr || !mapping->access_set ||
      word_count > mapping->bytes / xvram::vmm_poc::word_bytes) {
    fake().violation("kernel accessed an invalid or inaccessible mapping");
    return CUDA_ERROR_INVALID_VALUE;
  }

  const auto stable = fake().stable_addresses.find(global_start);
  if (stable == fake().stable_addresses.end()) {
    fake().stable_addresses.emplace(global_start, address);
  } else if (stable->second != address) {
    fake().violation("logical tile changed virtual address");
    return CUDA_ERROR_INVALID_ADDRESS_SPACE;
  }
  mapping->global_word_start = global_start;
  fake().launch_global_starts.push_back(global_start);

  for (std::uint64_t index = 0; index < word_count; ++index) {
    std::uint32_t word = 0;
    const std::size_t byte_offset = static_cast<std::size_t>(index * xvram::vmm_poc::word_bytes);
    std::memcpy(&word, allocation->storage.data() + byte_offset, sizeof(word));
    word = xvram::vmm_poc::transform_word(word, global_start + index, pass, seed);
    std::memcpy(allocation->storage.data() + byte_offset, &word, sizeof(word));
  }
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

[[nodiscard]] xvram::vmm_poc::ExecutorOptions
options_for(const xvram::vmm_poc::RequestedMode mode) {
  xvram::vmm_poc::ExecutorOptions options;
  options.logical_bytes = fake_logical_bytes;
  options.chunk_bytes = fake_chunk_bytes;
  options.window_slots = 2;
  options.passes = 2;
  options.mode = mode;
  options.stall_timeout = std::chrono::milliseconds(250);
  options.device_headroom_bytes = fake_chunk_bytes;
  return options;
}

[[nodiscard]] xvram::vmm_poc::ExecutorEnvironment environment_for() {
  xvram::vmm_poc::ExecutorEnvironment environment;
  environment.physical_host_bytes = 16ULL * gib;
  environment.available_host_bytes = 12ULL * gib;
  environment.initial_device_budget = xvram::vmm_poc::BudgetSnapshot{16ULL * mib, 0};
  environment.query_device_budget = []() -> std::optional<xvram::vmm_poc::BudgetSnapshot> {
    return xvram::vmm_poc::BudgetSnapshot{16ULL * mib, 0};
  };
  return environment;
}

struct Run {
  FakeCuda cuda;
  xvram::vmm_poc::ExecutorResult result;
};

[[nodiscard]] Run execute(const xvram::vmm_poc::RequestedMode mode, const Fault fault = {},
                          const bool one_not_ready = false) {
  Run run;
  run.cuda.fault = fault;
  run.cuda.return_not_ready_once = one_not_ready;
  CudaApi api(CudaApi::InjectedDispatch{});
  run.cuda.install(api);
  const ActiveFake active(run.cuda);
  const auto options = options_for(mode);
  const auto environment = environment_for();
  run.result = xvram::vmm_poc::run_executor(api, options, {}, &environment);
  return run;
}

void happy_both_mode_test() {
  Run run = execute(xvram::vmm_poc::RequestedMode::both, {}, true);
  const std::uint64_t chunk_count = ((fake_logical_bytes - 1ULL) / fake_chunk_bytes) + 1ULL;
  const std::uint64_t visits = chunk_count * 2ULL;

  CHECK(run.result.exit_code == 0);
  CHECK(run.result.status == "completed");
  CHECK(run.result.cleanup.complete.value_or(false));
  CHECK(run.result.reference.status == "completed");
  CHECK(run.result.pipeline.status == "completed");
  CHECK(run.result.reference.mappings == visits);
  CHECK(run.result.reference.set_access_calls == visits);
  CHECK(run.result.reference.unmappings == visits);
  CHECK(run.result.reference.event_boundaries == visits);
  CHECK(run.result.reference.handle_reuses == visits - 1ULL);
  CHECK(run.result.pipeline.mappings == visits);
  CHECK(run.result.pipeline.set_access_calls == visits);
  CHECK(run.result.pipeline.unmappings == visits);
  CHECK(run.result.pipeline.event_boundaries == visits);
  CHECK(run.result.pipeline.handle_reuses == visits - 2ULL);
  CHECK(run.result.reference.h2d_bytes == fake_logical_bytes * 2ULL);
  CHECK(run.result.reference.d2h_bytes == fake_logical_bytes * 2ULL);
  CHECK(run.result.pipeline.h2d_bytes == fake_logical_bytes * 2ULL);
  CHECK(run.result.pipeline.d2h_bytes == fake_logical_bytes * 2ULL);
  CHECK(run.result.reference.digest128 == run.result.pipeline.digest128);
  CHECK(run.result.reference.matches_cpu);
  CHECK(run.result.pipeline.matches_cpu);
  CHECK(run.result.reference.unsafe_remaps == 0);
  CHECK(run.result.pipeline.unsafe_remaps == 0);
  CHECK(run.cuda.violations.empty());
  CHECK(run.cuda.maximum_active_mappings == 2U);
  CHECK(run.cuda.stream_wait_count == static_cast<std::size_t>(visits * 2ULL));
  CHECK(run.cuda.stable_addresses.size() == static_cast<std::size_t>(chunk_count));
  CHECK(run.cuda.launch_global_starts.size() == static_cast<std::size_t>(visits * 2ULL));
  CHECK(run.cuda.unmap_global_starts == run.cuda.launch_global_starts);
  CHECK(run.cuda.reservation == 0);
  CHECK(run.cuda.pinned.empty());
  CHECK(run.cuda.mappings.empty());

  const std::size_t pipeline_offset = static_cast<std::size_t>(visits);
  const std::uint64_t words_per_chunk = fake_chunk_bytes / xvram::vmm_poc::word_bytes;
  for (std::uint64_t ordinal = 0; ordinal < visits; ++ordinal) {
    const auto visit = xvram::vmm_poc::tile_visit_at(chunk_count, 2, ordinal);
    CHECK(visit.has_value());
    if (visit.has_value()) {
      CHECK(run.cuda.launch_global_starts[pipeline_offset + static_cast<std::size_t>(ordinal)] ==
            visit->tile_index * words_per_chunk);
    }
  }
}

void completed_cleanup_test(const Call call) {
  Run run = execute(xvram::vmm_poc::RequestedMode::reference, Fault{call, 1});
  CHECK(run.result.exit_code == 27);
  CHECK(run.result.status == "failed");
  CHECK(run.result.failure.has_value());
  CHECK(run.result.cleanup.complete.value_or(false));
  CHECK(run.cuda.violations.empty());
  CHECK(run.cuda.reservation == 0);
  CHECK(run.cuda.pinned.empty());
}

void setup_map_and_prework_fault_tests() {
  completed_cleanup_test(Call::module_load);
  completed_cleanup_test(Call::address_reserve);
  completed_cleanup_test(Call::mem_create);
  completed_cleanup_test(Call::host_alloc);
  completed_cleanup_test(Call::event_create);
  completed_cleanup_test(Call::stream_create);
  completed_cleanup_test(Call::map);
  completed_cleanup_test(Call::set_access);
  completed_cleanup_test(Call::h2d);
  completed_cleanup_test(Call::event_record);
}

void partial_generation_fault_test(
    const Call call, const std::size_t occurrence = 1U,
    const xvram::vmm_poc::RequestedMode mode = xvram::vmm_poc::RequestedMode::reference) {
  Run run = execute(mode, Fault{call, occurrence});
  CHECK(run.result.exit_code == 27);
  CHECK(run.result.status == "failed");
  CHECK(run.result.failure.has_value());
  CHECK(!run.result.cleanup.complete.value_or(true));
  CHECK(!run.result.cleanup.events_drained.value_or(true));
  CHECK(!run.result.cleanup.mappings_removed.value_or(true));
  CHECK(!run.result.cleanup.reservation_released.value_or(true));
  CHECK(run.result.cleanup.handles_released.value_or(false));
  CHECK(run.result.cleanup.pinned_buffers_released.value_or(false));
  CHECK(run.result.cleanup.context_destroyed.value_or(false));
  CHECK(run.cuda.call_counts[Call::unmap] == 0U);
  CHECK(run.cuda.call_counts[Call::mem_release] >= 1U);
  CHECK(run.cuda.call_counts[Call::address_free] == 0U);
  CHECK(run.cuda.violations.empty());
}

void copy_kernel_and_completion_fault_tests() {
  partial_generation_fault_test(Call::kernel);
  partial_generation_fault_test(Call::d2h);
  partial_generation_fault_test(Call::stream_wait, 1U, xvram::vmm_poc::RequestedMode::pipeline);
  partial_generation_fault_test(Call::event_record, 6U);

  Run query = execute(xvram::vmm_poc::RequestedMode::reference,
                      Fault{Call::event_query, 1U, CUDA_ERROR_UNKNOWN});
  CHECK(query.result.exit_code == 27);
  CHECK(query.result.failure.has_value());
  CHECK(query.result.failure->operation == "cuEventQuery(slot_done)");
  CHECK(query.result.cleanup.complete.value_or(false));
  CHECK(query.cuda.violations.empty());
  CHECK(query.cuda.call_counts[Call::event_synchronize] >= 1U);
  CHECK(query.cuda.call_counts[Call::unmap] == 1U);
}

void unmap_and_cleanup_fault_tests() {
  Run unmap = execute(xvram::vmm_poc::RequestedMode::reference, Fault{Call::unmap, 1U});
  CHECK(unmap.result.exit_code == 27);
  CHECK(unmap.result.failure.has_value());
  CHECK(unmap.result.failure->operation == "cuMemUnmap");
  CHECK(!unmap.result.cleanup.complete.value_or(true));
  CHECK(unmap.result.cleanup.mappings_removed.value_or(false));
  CHECK(!unmap.result.cleanup.reservation_released.value_or(true));
  CHECK(unmap.result.cleanup.handles_released.value_or(false));
  CHECK(unmap.cuda.call_counts[Call::unmap] == 2U);
  CHECK(unmap.cuda.call_counts[Call::address_free] == 0U);
  CHECK(unmap.cuda.violations.empty());

  Run release = execute(xvram::vmm_poc::RequestedMode::reference, Fault{Call::mem_release, 1U});
  CHECK(release.result.exit_code == 27);
  CHECK(release.result.status == "failed");
  CHECK(!release.result.cleanup.complete.value_or(true));
  CHECK(!release.result.cleanup.handles_released.value_or(true));
  CHECK(release.result.cleanup.mappings_removed.value_or(false));
  CHECK(release.result.cleanup.reservation_released.value_or(false));
  CHECK(release.cuda.violations.empty());
}

void cleanup_stage_fault_tests() {
  constexpr std::array<Call, 7> cleanup_calls{
      Call::address_free, Call::event_destroy,   Call::stream_destroy,     Call::module_unload,
      Call::host_free,    Call::context_destroy, Call::context_set_current};
  for (const Call call : cleanup_calls) {
    Run run = execute(xvram::vmm_poc::RequestedMode::reference, Fault{call, 1U});
    CHECK(run.result.exit_code == 27);
    CHECK(run.result.status == "failed");
    CHECK(run.result.failure.has_value());
    CHECK(run.result.failure->stage == "cleanup");
    CHECK(!run.result.cleanup.complete.value_or(true));
    CHECK(run.cuda.violations.empty());
    switch (call) {
    case Call::address_free:
      CHECK(!run.result.cleanup.reservation_released.value_or(true));
      break;
    case Call::event_destroy:
      CHECK(!run.result.cleanup.events_destroyed.value_or(true));
      break;
    case Call::stream_destroy:
      CHECK(!run.result.cleanup.streams_destroyed.value_or(true));
      break;
    case Call::module_unload:
      CHECK(!run.result.cleanup.module_unloaded.value_or(true));
      break;
    case Call::host_free:
      CHECK(!run.result.cleanup.pinned_buffers_released.value_or(true));
      break;
    case Call::context_destroy:
    case Call::context_set_current:
      CHECK(!run.result.cleanup.context_destroyed.value_or(true));
      break;
    default:
      CHECK(false);
      break;
    }
  }
}

void dma_quarantine_lifetime_test() {
  FakeCuda cuda;
  cuda.fault = Fault{Call::kernel, 1U};
  cuda.secondary_fault = Fault{Call::stream_synchronize, 1U};
  CudaApi api(CudaApi::InjectedDispatch{});
  cuda.install(api);
  const ActiveFake active(cuda);
  const auto options = options_for(xvram::vmm_poc::RequestedMode::reference);
  const auto environment = environment_for();
  const auto result = xvram::vmm_poc::run_executor(api, options, {}, &environment);

  CHECK(result.exit_code == 27);
  CHECK(result.status == "failed");
  CHECK(result.failure.has_value());
  CHECK(!result.cleanup.complete.value_or(true));
  CHECK(!result.cleanup.events_drained.value_or(true));
  CHECK(!result.cleanup.mappings_removed.value_or(true));
  CHECK(!result.cleanup.reservation_released.value_or(true));
  CHECK(!result.cleanup.pinned_buffers_released.value_or(true));
  CHECK(result.cleanup.handles_released.value_or(false));
  CHECK(result.cleanup.context_destroyed.value_or(false));
  CHECK(cuda.call_counts[Call::stream_synchronize] == 4U);
  CHECK(cuda.call_counts[Call::unmap] == 0U);
  CHECK(cuda.call_counts[Call::mem_release] == 1U);
  CHECK(cuda.call_counts[Call::address_free] == 0U);
  CHECK(cuda.call_counts[Call::host_free] == 0U);
  CHECK(cuda.call_counts[Call::context_destroy] == 1U);
  CHECK(cuda.pinned.size() == 1U);
  CHECK(cuda.violations.empty());
}

void progress_exception_cleanup_test() {
  FakeCuda cuda;
  CudaApi api(CudaApi::InjectedDispatch{});
  cuda.install(api);
  const ActiveFake active(cuda);
  auto options = options_for(xvram::vmm_poc::RequestedMode::reference);
  options.logical_bytes = 257ULL * mib;
  options.progress_heartbeat = std::chrono::milliseconds(1);
  const auto environment = environment_for();
  std::uint32_t progress_calls = 0;
  const auto result = xvram::vmm_poc::run_executor(
      api, options,
      [&](const std::string_view, const xvram::vmm_poc::ModeStatistics&, const std::uint64_t) {
        ++progress_calls;
        throw std::runtime_error("injected progress callback failure");
      },
      &environment);

  CHECK(progress_calls >= 2U);
  CHECK(result.exit_code == 27);
  CHECK(result.status == "failed");
  CHECK(result.failure.has_value());
  CHECK(result.failure->operation == "internal_exception");
  CHECK(result.cleanup.complete.value_or(false));
  CHECK(cuda.reservation == 0);
  CHECK(cuda.pinned.empty());
  CHECK(cuda.mappings.empty());
  CHECK(cuda.violations.empty());
}

void finish_progress_exception_test() {
  FakeCuda cuda;
  CudaApi api(CudaApi::InjectedDispatch{});
  cuda.install(api);
  const ActiveFake active(cuda);
  const auto options = options_for(xvram::vmm_poc::RequestedMode::reference);
  const auto environment = environment_for();
  std::uint32_t completed_progress_calls = 0;
  const auto result = xvram::vmm_poc::run_executor(
      api, options,
      [&](const std::string_view, const xvram::vmm_poc::ModeStatistics& statistics,
          const std::uint64_t) {
        if (statistics.status == "completed" && ++completed_progress_calls == 2U) {
          throw std::runtime_error("injected final progress callback failure");
        }
      },
      &environment);

  CHECK(completed_progress_calls == 2U);
  CHECK(result.exit_code == 27);
  CHECK(result.status == "failed");
  CHECK(result.failure.has_value());
  CHECK(result.failure->operation == "progress_callback");
  CHECK(result.cleanup.complete.value_or(false));
  CHECK(cuda.violations.empty());
}

void progress_exception_preserves_cuda_failure_test() {
  FakeCuda cuda;
  cuda.fault = Fault{Call::map, 1U};
  CudaApi api(CudaApi::InjectedDispatch{});
  cuda.install(api);
  const ActiveFake active(cuda);
  const auto options = options_for(xvram::vmm_poc::RequestedMode::reference);
  const auto environment = environment_for();
  const auto result = xvram::vmm_poc::run_executor(
      api, options,
      [](const std::string_view, const xvram::vmm_poc::ModeStatistics&, const std::uint64_t) {
        throw std::runtime_error("injected progress failure");
      },
      &environment);

  CHECK(result.exit_code == 27);
  CHECK(result.status == "failed");
  CHECK(result.failure.has_value());
  CHECK(result.failure->operation == "cuMemMap");
  CHECK(result.cleanup.complete.value_or(false));
  CHECK(cuda.violations.empty());
}

} // namespace

int main() {
  happy_both_mode_test();
  setup_map_and_prework_fault_tests();
  copy_kernel_and_completion_fault_tests();
  unmap_and_cleanup_fault_tests();
  cleanup_stage_fault_tests();
  dma_quarantine_lifetime_test();
  progress_exception_cleanup_test();
  finish_progress_exception_test();
  progress_exception_preserves_cuda_failure_test();

  if (failures != 0) {
    std::cerr << failures << " fake CUDA executor test(s) failed\n";
    return 1;
  }
  std::cout << "all fake CUDA executor tests passed\n";
  return 0;
}
