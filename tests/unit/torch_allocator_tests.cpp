#define NOMINMAX

#include "platform/cuda/cuda_api.hpp"
#include "torch/allocator.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using xvram::cuda::CudaApi;
using xvram::cuda::abi::Context;
using xvram::cuda::abi::DevicePointer;
using xvram::cuda::abi::Event;
using xvram::cuda::abi::GenericAllocationHandle;
using xvram::cuda::abi::Result;
using xvram::cuda::abi::Stream;
using xvram::torch_allocator::SegmentAllocator;

static_assert(std::is_standard_layout_v<xvram_torch_allocator_stats_v1>);
static_assert(sizeof(xvram_torch_allocator_stats_v1) == 312U);
static_assert(offsetof(xvram_torch_allocator_stats_v1, handles_created) == 112U);
static_assert(offsetof(xvram_torch_allocator_stats_v1, last_status) == 240U);

constexpr std::size_t kib = 1024U;
constexpr std::size_t minimum_granularity = 4U * kib;
constexpr std::size_t recommended_granularity = 64U * kib;

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
  context_get_current,
  context_get_device,
  context_push,
  context_pop,
  stream_is_capturing,
  granularity_minimum,
  granularity_recommended,
  address_reserve,
  address_free,
  mem_create,
  mem_release,
  map,
  set_access,
  event_create,
  event_record,
  event_synchronize,
  event_destroy,
  unmap,
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

  struct HandleState {
    std::size_t bytes = 0;
    bool released = false;
  };

  struct MappingState {
    std::size_t bytes = 0;
    GenericAllocationHandle handle = 0;
    Context context = nullptr;
    bool access_attempted = false;
    bool access_set = false;
    bool completion_observed = false;
  };

  struct EventState {
    Context context = nullptr;
    Stream stream = nullptr;
    bool recorded = false;
    bool synchronized = false;
    std::optional<DevicePointer> completion_address;
  };

  Fault fault;
  std::map<Call, std::size_t> call_counts;
  std::vector<Call> calls;
  std::vector<std::string> violations;
  std::vector<std::unique_ptr<Token>> tokens;
  std::map<Context, CUdevice, std::less<Context>> context_devices;
  std::vector<Context> context_stack;
  std::map<Stream, CUstreamCaptureStatus, std::less<Stream>> capture_status;
  std::map<DevicePointer, std::size_t> reservations;
  std::map<GenericAllocationHandle, HandleState> handles;
  std::map<DevicePointer, MappingState> mappings;
  std::map<Event, EventState, std::less<Event>> events;
  Context current_context = nullptr;
  DevicePointer next_address = static_cast<DevicePointer>(0x1'0000'0000ULL);
  GenericAllocationHandle next_handle = 1;
  std::uint64_t next_token = 1;
  std::uint64_t sequence = 0;
  std::optional<DevicePointer> expected_fence_address;

  [[nodiscard]] bool should_fail(const Call call) {
    calls.push_back(call);
    ++sequence;
    const std::size_t occurrence = ++call_counts[call];
    return fault.call == call && fault.occurrence == occurrence;
  }

  template <typename Handle> [[nodiscard]] Handle make_token() {
    static_assert(std::is_pointer_v<Handle>);
    auto token = std::make_unique<Token>();
    token->id = next_token++;
    Token* raw = token.get();
    tokens.push_back(std::move(token));
    return reinterpret_cast<Handle>(raw);
  }

  [[nodiscard]] Context make_context(const CUdevice device) {
    const Context context = make_token<Context>();
    context_devices.emplace(context, device);
    return context;
  }

  [[nodiscard]] Stream make_stream() {
    const Stream stream = make_token<Stream>();
    capture_status.emplace(stream, CU_STREAM_CAPTURE_STATUS_NONE);
    return stream;
  }

  void violation(std::string message) {
    violations.push_back(std::move(message));
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

Result CUDAAPI fake_error_name(const Result result, const char** name) {
  *name = result == CUDA_ERROR_OUT_OF_MEMORY ? "CUDA_ERROR_OUT_OF_MEMORY" : "CUDA_ERROR_UNKNOWN";
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_error_string(Result, const char** message) {
  *message = "injected fake CUDA failure";
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_context_get_current(Context* context) {
  if (fake().should_fail(Call::context_get_current)) {
    return fake().fault.result;
  }
  *context = fake().current_context;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_context_get_device(CUdevice* device) {
  if (fake().should_fail(Call::context_get_device)) {
    return fake().fault.result;
  }
  const auto found = fake().context_devices.find(fake().current_context);
  if (found == fake().context_devices.end()) {
    return CUDA_ERROR_INVALID_CONTEXT;
  }
  *device = found->second;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_context_push_current(const Context context) {
  if (fake().should_fail(Call::context_push)) {
    return fake().fault.result;
  }
  if (!fake().context_devices.contains(context)) {
    return CUDA_ERROR_INVALID_CONTEXT;
  }
  fake().context_stack.push_back(fake().current_context);
  fake().current_context = context;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_context_pop_current(Context* context) {
  if (fake().should_fail(Call::context_pop)) {
    return fake().fault.result;
  }
  if (fake().context_stack.empty()) {
    return CUDA_ERROR_INVALID_CONTEXT;
  }
  *context = fake().current_context;
  fake().current_context = fake().context_stack.back();
  fake().context_stack.pop_back();
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_stream_is_capturing(const Stream stream, CUstreamCaptureStatus* status) {
  if (fake().should_fail(Call::stream_is_capturing)) {
    return fake().fault.result;
  }
  const auto found = fake().capture_status.find(stream);
  if (found == fake().capture_status.end()) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  *status = found->second;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_granularity(std::size_t* granularity, const CUmemAllocationProp* property,
                                const CUmemAllocationGranularity_flags option) {
  const Call call = option == CU_MEM_ALLOC_GRANULARITY_MINIMUM ? Call::granularity_minimum
                                                               : Call::granularity_recommended;
  if (fake().should_fail(call)) {
    return fake().fault.result;
  }
  if (property == nullptr || property->type != CU_MEM_ALLOCATION_TYPE_PINNED ||
      property->location.type != CU_MEM_LOCATION_TYPE_DEVICE) {
    fake().violation("granularity query used an invalid allocation property");
    return CUDA_ERROR_INVALID_VALUE;
  }
  *granularity =
      option == CU_MEM_ALLOC_GRANULARITY_MINIMUM ? minimum_granularity : recommended_granularity;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_address_reserve(DevicePointer* address, const std::size_t bytes,
                                    const std::size_t alignment, DevicePointer,
                                    unsigned long long) {
  if (fake().should_fail(Call::address_reserve)) {
    return fake().fault.result;
  }
  if (bytes == 0 || bytes % minimum_granularity != 0 ||
      (alignment != 0 && alignment % minimum_granularity != 0)) {
    fake().violation("reservation size or alignment was not VMM-granularity aligned");
    return CUDA_ERROR_INVALID_VALUE;
  }
  const DevicePointer created = fake().next_address;
  fake().next_address += static_cast<DevicePointer>(bytes + minimum_granularity);
  fake().reservations.emplace(created, bytes);
  *address = created;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_address_free(const DevicePointer address, const std::size_t bytes) {
  if (fake().should_fail(Call::address_free)) {
    return fake().fault.result;
  }
  const auto found = fake().reservations.find(address);
  if (found == fake().reservations.end() || found->second != bytes) {
    fake().violation("reservation was not freed with its exact original address and size");
    return CUDA_ERROR_INVALID_VALUE;
  }
  fake().reservations.erase(found);
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_mem_create(GenericAllocationHandle* handle, const std::size_t bytes,
                               const CUmemAllocationProp* property, unsigned long long) {
  if (fake().should_fail(Call::mem_create)) {
    return fake().fault.result;
  }
  if (bytes == 0 || bytes % minimum_granularity != 0 || property == nullptr ||
      property->location.type != CU_MEM_LOCATION_TYPE_DEVICE) {
    fake().violation("physical allocation did not match VMM requirements");
    return CUDA_ERROR_INVALID_VALUE;
  }
  const GenericAllocationHandle created = fake().next_handle++;
  fake().handles.emplace(created, FakeCuda::HandleState{bytes, false});
  *handle = created;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_mem_release(const GenericAllocationHandle handle) {
  if (fake().should_fail(Call::mem_release)) {
    return fake().fault.result;
  }
  const auto found = fake().handles.find(handle);
  if (found == fake().handles.end() || found->second.released) {
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
  const auto reservation = fake().reservations.find(address);
  const auto physical = fake().handles.find(handle);
  if (reservation == fake().reservations.end() || reservation->second != bytes ||
      physical == fake().handles.end() || physical->second.released ||
      physical->second.bytes != bytes || fake().mappings.contains(address)) {
    fake().violation("map did not cover one exact live reservation and handle");
    return CUDA_ERROR_INVALID_VALUE;
  }
  fake().mappings.emplace(
      address, FakeCuda::MappingState{bytes, handle, fake().current_context, false, false, false});
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_mem_set_access(const DevicePointer address, const std::size_t bytes,
                                   const CUmemAccessDesc* descriptor, const std::size_t count) {
  const auto found = fake().mappings.find(address);
  if (found != fake().mappings.end()) {
    found->second.access_attempted = true;
  }
  if (fake().should_fail(Call::set_access)) {
    return fake().fault.result;
  }
  if (found == fake().mappings.end() || found->second.bytes != bytes || descriptor == nullptr ||
      count != 1U || descriptor->location.type != CU_MEM_LOCATION_TYPE_DEVICE ||
      descriptor->flags != CU_MEM_ACCESS_FLAGS_PROT_READWRITE || found->second.access_set) {
    fake().violation("SetAccess was missing, duplicated, or applied to the wrong range");
    return CUDA_ERROR_INVALID_VALUE;
  }
  found->second.access_set = true;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_event_create(Event* event, unsigned int) {
  if (fake().should_fail(Call::event_create)) {
    return fake().fault.result;
  }
  const Event created = fake().make_token<Event>();
  fake().events.emplace(
      created, FakeCuda::EventState{fake().current_context, nullptr, false, false, std::nullopt});
  *event = created;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_event_record(const Event event, const Stream stream) {
  if (fake().should_fail(Call::event_record)) {
    return fake().fault.result;
  }
  const auto found = fake().events.find(event);
  if (found == fake().events.end() || found->second.context != fake().current_context ||
      !fake().capture_status.contains(stream) || !fake().expected_fence_address.has_value()) {
    fake().violation("completion event was recorded in the wrong context or stream");
    return CUDA_ERROR_INVALID_CONTEXT;
  }
  found->second.stream = stream;
  found->second.recorded = true;
  found->second.completion_address = fake().expected_fence_address;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_event_synchronize(const Event event) {
  if (fake().should_fail(Call::event_synchronize)) {
    return fake().fault.result;
  }
  const auto found = fake().events.find(event);
  if (found == fake().events.end() || !found->second.recorded ||
      !found->second.completion_address.has_value()) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  found->second.synchronized = true;
  const auto mapping = fake().mappings.find(*found->second.completion_address);
  if (mapping == fake().mappings.end()) {
    fake().violation("completion event did not identify a live mapping generation");
    return CUDA_ERROR_INVALID_VALUE;
  }
  mapping->second.completion_observed = true;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_event_destroy(const Event event) {
  if (fake().should_fail(Call::event_destroy)) {
    return fake().fault.result;
  }
  if (fake().events.erase(event) != 1U) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_mem_unmap(const DevicePointer address, const std::size_t bytes) {
  if (fake().should_fail(Call::unmap)) {
    return fake().fault.result;
  }
  const auto found = fake().mappings.find(address);
  if (found == fake().mappings.end() || found->second.bytes != bytes) {
    fake().violation("unmap did not remove the exact full mapped range");
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (!found->second.access_attempted) {
    fake().violation("mapping was unmapped without a SetAccess attempt");
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (found->second.access_set && !found->second.completion_observed) {
    fake().violation("published mapping was unmapped before a completed event boundary");
    return CUDA_ERROR_NOT_READY;
  }
  fake().mappings.erase(found);
  return CUDA_SUCCESS;
}

void FakeCuda::install(CudaApi& api) {
  api.get_error_name_ = fake_error_name;
  api.get_error_string_ = fake_error_string;
  api.context_get_current_ = fake_context_get_current;
  api.context_get_device_ = fake_context_get_device;
  api.context_push_current_ = fake_context_push_current;
  api.context_pop_current_ = fake_context_pop_current;
  api.stream_is_capturing_ = fake_stream_is_capturing;
  api.mem_get_allocation_granularity_ = fake_granularity;
  api.mem_address_reserve_ = fake_address_reserve;
  api.mem_address_free_ = fake_address_free;
  api.mem_create_ = fake_mem_create;
  api.mem_release_ = fake_mem_release;
  api.mem_map_ = fake_mem_map;
  api.mem_unmap_ = fake_mem_unmap;
  api.mem_set_access_ = fake_mem_set_access;
  api.event_create_ = fake_event_create;
  api.event_record_ = fake_event_record;
  api.event_synchronize_ = fake_event_synchronize;
  api.event_destroy_ = fake_event_destroy;
}

[[nodiscard]] std::size_t call_index(const FakeCuda& cuda, const Call call) {
  const auto found = std::find(cuda.calls.begin(), cuda.calls.end(), call);
  return found == cuda.calls.end() ? std::numeric_limits<std::size_t>::max()
                                   : static_cast<std::size_t>(found - cuda.calls.begin());
}

struct Fixture {
  FakeCuda cuda;
  CudaApi api{CudaApi::InjectedDispatch{}};
  Context primary = nullptr;
  Stream stream = nullptr;

  Fixture() {
    primary = cuda.make_context(0);
    stream = cuda.make_stream();
    cuda.current_context = primary;
    cuda.install(api);
  }
};

void deallocate_segment(SegmentAllocator& allocator, FakeCuda& cuda, void* pointer,
                        const std::size_t bytes, const int device, const Stream stream) {
  cuda.expected_fence_address = static_cast<DevicePointer>(
      reinterpret_cast<std::uintptr_t>(pointer));
  allocator.deallocate(pointer, bytes, device, stream);
  cuda.expected_fence_address.reset();
}

void alignment_tests() {
  using xvram::torch_allocator::checked_align_up;
  CHECK(!checked_align_up(1, 0).has_value());
  CHECK(checked_align_up(0, minimum_granularity) == 0U);
  CHECK(checked_align_up(1, minimum_granularity) == minimum_granularity);
  CHECK(checked_align_up(minimum_granularity, minimum_granularity) == minimum_granularity);
  CHECK(checked_align_up(minimum_granularity + 1U, minimum_granularity) ==
        2U * minimum_granularity);
  CHECK(!checked_align_up(std::numeric_limits<std::size_t>::max() - 3U, 8U).has_value());
}

void successful_lifecycle_tests() {
  Fixture fixture;
  ActiveFake active(fixture.cuda);
  SegmentAllocator allocator(fixture.api);

  void* first = allocator.allocate(1, 0, fixture.stream);
  CHECK(first != nullptr);
  CHECK(reinterpret_cast<std::uintptr_t>(first) == 0x1'0000'0000ULL);
  CHECK(allocator.live_allocation_count() == 1U);
  CHECK(fixture.cuda.reservations.size() == 1U);
  CHECK(fixture.cuda.mappings.size() == 1U);
  CHECK(call_index(fixture.cuda, Call::address_reserve) <
        call_index(fixture.cuda, Call::mem_create));
  CHECK(call_index(fixture.cuda, Call::mem_create) < call_index(fixture.cuda, Call::map));
  CHECK(call_index(fixture.cuda, Call::map) < call_index(fixture.cuda, Call::set_access));

  const auto allocation = fixture.cuda.mappings.begin();
  CHECK(allocation->first == static_cast<DevicePointer>(reinterpret_cast<std::uintptr_t>(first)));
  CHECK(allocation->second.bytes == minimum_granularity);
  CHECK(allocation->second.access_set);

  const Stream second_stream = fixture.cuda.make_stream();
  void* second = allocator.allocate(minimum_granularity + 1U, 0, second_stream);
  CHECK(second != nullptr);
  CHECK(second != first);
  CHECK(allocator.live_allocation_count() == 2U);
  CHECK(
      fixture.cuda.mappings.at(static_cast<DevicePointer>(reinterpret_cast<std::uintptr_t>(second)))
          .bytes == 2U * minimum_granularity);

  deallocate_segment(allocator, fixture.cuda, first, 1, 0, fixture.stream);
  CHECK(allocator.live_allocation_count() == 1U);
  deallocate_segment(allocator, fixture.cuda, second, minimum_granularity + 1U, 0,
                     second_stream);
  CHECK(allocator.live_allocation_count() == 0U);
  CHECK(fixture.cuda.reservations.empty());
  CHECK(fixture.cuda.mappings.empty());
  CHECK(fixture.cuda.violations.empty());
  CHECK(fixture.cuda.call_counts[Call::event_record] == 2U);
  CHECK(fixture.cuda.call_counts[Call::event_synchronize] == 2U);
  CHECK(fixture.cuda.call_counts[Call::unmap] == 2U);
  CHECK(fixture.cuda.call_counts[Call::mem_release] == 2U);
  CHECK(fixture.cuda.call_counts[Call::address_free] == 2U);

  const auto telemetry = allocator.stats();
  CHECK(telemetry.allocation_calls == 2U);
  CHECK(telemetry.allocation_failures == 0U);
  CHECK(telemetry.free_calls == 2U);
  CHECK(telemetry.free_failures == 0U);
  CHECK(telemetry.requested_bytes_total == minimum_granularity + 2U);
  CHECK(telemetry.mapped_bytes_current == 0U);
  CHECK(telemetry.mapped_bytes_peak == 3U * minimum_granularity);
  CHECK(telemetry.handle_releases == 2U);
  CHECK(telemetry.maps == 2U);
  CHECK(telemetry.set_access_calls == 2U);
  CHECK(telemetry.event_boundaries == 2U);
  CHECK(telemetry.unmaps == 2U);
  CHECK(telemetry.reservation_frees == 2U);
  CHECK(telemetry.unsafe_unmaps == 0U);
  CHECK(telemetry.quarantined_segments == 0U);
}

void allocation_rollback_tests() {
  constexpr Call sites[]{Call::mem_create, Call::map, Call::set_access};
  for (const Call site : sites) {
    Fixture fixture;
    ActiveFake active(fixture.cuda);
    fixture.cuda.fault = Fault{site, 1, CUDA_ERROR_OUT_OF_MEMORY};
    SegmentAllocator allocator(fixture.api);

    CHECK(allocator.allocate(17, 0, fixture.stream) == nullptr);
    CHECK(allocator.live_allocation_count() == 0U);
    CHECK(fixture.cuda.reservations.empty());
    CHECK(fixture.cuda.mappings.empty());
    CHECK(fixture.cuda.violations.empty());

    const auto telemetry = allocator.stats();
    CHECK(telemetry.allocation_calls == 1U);
    CHECK(telemetry.allocation_failures == 1U);
    CHECK(telemetry.mapped_bytes_current == 0U);
    CHECK(telemetry.last_native_error == CUDA_ERROR_OUT_OF_MEMORY);
    if (site == Call::mem_create) {
      CHECK(fixture.cuda.call_counts[Call::address_free] == 1U);
      CHECK(fixture.cuda.call_counts[Call::mem_release] == 0U);
      CHECK(fixture.cuda.call_counts[Call::unmap] == 0U);
    } else if (site == Call::map) {
      CHECK(fixture.cuda.call_counts[Call::mem_release] == 1U);
      CHECK(fixture.cuda.call_counts[Call::address_free] == 1U);
      CHECK(fixture.cuda.call_counts[Call::unmap] == 0U);
    } else {
      CHECK(fixture.cuda.call_counts[Call::unmap] == 1U);
      CHECK(fixture.cuda.call_counts[Call::mem_release] == 1U);
      CHECK(fixture.cuda.call_counts[Call::address_free] == 1U);
    }
  }
}

void registry_oom_quarantine_tests() {
  {
    Fixture fixture;
    ActiveFake active(fixture.cuda);
    SegmentAllocator allocator(fixture.api, 0U);

    CHECK(allocator.allocate(17, 0, fixture.stream) == nullptr);
    CHECK(allocator.live_allocation_count() == 0U);
    CHECK(fixture.cuda.reservations.empty());
    const auto telemetry = allocator.stats();
    CHECK(telemetry.allocation_failures == 1U);
    CHECK(telemetry.oom_failures == 1U);
    CHECK(telemetry.reservations == 1U);
    CHECK(telemetry.reservation_frees == 1U);
    CHECK(telemetry.quarantined_segments == 0U);
  }

  {
    Fixture fixture;
    ActiveFake active(fixture.cuda);
    fixture.cuda.fault = Fault{Call::address_free, 1, CUDA_ERROR_UNKNOWN};
    SegmentAllocator allocator(fixture.api, 0U);

    CHECK(allocator.allocate(17, 0, fixture.stream) == nullptr);
    CHECK(allocator.live_allocation_count() == 1U);
    CHECK(fixture.cuda.reservations.size() == 1U);
    auto telemetry = allocator.stats();
    CHECK(telemetry.quarantined_segments == 1U);
    CHECK(telemetry.last_status == XVRAM_TORCH_ALLOCATOR_QUARANTINED);

    fixture.cuda.fault = {};
    CHECK(allocator.allocate(17, 0, fixture.stream) == nullptr);
    CHECK(fixture.cuda.call_counts[Call::address_reserve] == 1U);
    allocator.reset_stats();
    telemetry = allocator.stats();
    CHECK(telemetry.quarantined_segments == 1U);
    CHECK(telemetry.last_status == XVRAM_TORCH_ALLOCATOR_QUARANTINED);
  }
}

void event_fence_failure_quarantines_tests() {
  Fixture fixture;
  ActiveFake active(fixture.cuda);
  SegmentAllocator allocator(fixture.api);
  void* pointer = allocator.allocate(1, 0, fixture.stream);
  CHECK(pointer != nullptr);

  fixture.cuda.fault = Fault{Call::event_synchronize, 1, CUDA_ERROR_UNKNOWN};
  deallocate_segment(allocator, fixture.cuda, pointer, 1, 0, fixture.stream);

  CHECK(allocator.live_allocation_count() == 1U);
  CHECK(fixture.cuda.mappings.size() == 1U);
  CHECK(fixture.cuda.call_counts[Call::unmap] == 0U);
  CHECK(fixture.cuda.call_counts[Call::mem_release] == 0U);
  CHECK(fixture.cuda.call_counts[Call::address_free] == 0U);
  CHECK(fixture.cuda.violations.empty());
  const auto telemetry = allocator.stats();
  CHECK(telemetry.free_calls == 1U);
  CHECK(telemetry.free_failures == 1U);
  CHECK(telemetry.event_boundaries == 0U);
  CHECK(telemetry.quarantined_segments == 1U);
  CHECK(telemetry.unsafe_unmaps == 0U);
  CHECK(telemetry.last_status == XVRAM_TORCH_ALLOCATOR_QUARANTINED);
}

void event_setup_failure_quarantines_tests() {
  constexpr Call sites[]{Call::event_create, Call::event_record};
  for (const Call site : sites) {
    Fixture fixture;
    ActiveFake active(fixture.cuda);
    SegmentAllocator allocator(fixture.api);
    void* pointer = allocator.allocate(1, 0, fixture.stream);
    CHECK(pointer != nullptr);

    fixture.cuda.fault = Fault{site, 1, CUDA_ERROR_UNKNOWN};
    deallocate_segment(allocator, fixture.cuda, pointer, 1, 0, fixture.stream);

    CHECK(allocator.live_allocation_count() == 1U);
    CHECK(fixture.cuda.mappings.size() == 1U);
    CHECK(fixture.cuda.call_counts[Call::unmap] == 0U);
    CHECK(fixture.cuda.violations.empty());
    const auto telemetry = allocator.stats();
    CHECK(telemetry.free_failures == 1U);
    CHECK(telemetry.event_boundaries == 0U);
    CHECK(telemetry.quarantined_segments == 1U);
    CHECK(telemetry.last_status == XVRAM_TORCH_ALLOCATOR_QUARANTINED);
  }
}

void post_boundary_cleanup_failure_tests() {
  {
    Fixture fixture;
    ActiveFake active(fixture.cuda);
    SegmentAllocator allocator(fixture.api);
    void* pointer = allocator.allocate(1, 0, fixture.stream);
    CHECK(pointer != nullptr);

    fixture.cuda.fault = Fault{Call::event_destroy, 1, CUDA_ERROR_UNKNOWN};
    deallocate_segment(allocator, fixture.cuda, pointer, 1, 0, fixture.stream);

    CHECK(allocator.live_allocation_count() == 0U);
    CHECK(fixture.cuda.mappings.empty());
    CHECK(fixture.cuda.reservations.empty());
    const auto telemetry = allocator.stats();
    CHECK(telemetry.free_failures == 1U);
    CHECK(telemetry.event_boundaries == 1U);
    CHECK(telemetry.unmaps == 1U);
    CHECK(telemetry.quarantined_segments == 0U);
  }

  {
    Fixture fixture;
    ActiveFake active(fixture.cuda);
    SegmentAllocator allocator(fixture.api);
    void* pointer = allocator.allocate(1, 0, fixture.stream);
    CHECK(pointer != nullptr);

    fixture.cuda.fault = Fault{Call::address_free, 1, CUDA_ERROR_UNKNOWN};
    deallocate_segment(allocator, fixture.cuda, pointer, 1, 0, fixture.stream);

    CHECK(allocator.live_allocation_count() == 1U);
    CHECK(fixture.cuda.mappings.empty());
    CHECK(fixture.cuda.reservations.size() == 1U);
    const auto telemetry = allocator.stats();
    CHECK(telemetry.free_failures == 1U);
    CHECK(telemetry.handle_releases == 1U);
    CHECK(telemetry.reservation_frees == 0U);
    CHECK(telemetry.quarantined_segments == 1U);
  }

  {
    Fixture fixture;
    ActiveFake active(fixture.cuda);
    SegmentAllocator allocator(fixture.api);
    void* pointer = allocator.allocate(1, 0, fixture.stream);
    CHECK(pointer != nullptr);
    const Context foreign = fixture.cuda.make_context(0);
    fixture.cuda.current_context = foreign;

    fixture.cuda.fault = Fault{Call::context_pop, 1, CUDA_ERROR_UNKNOWN};
    deallocate_segment(allocator, fixture.cuda, pointer, 1, 0, fixture.stream);

    CHECK(allocator.live_allocation_count() == 0U);
    CHECK(fixture.cuda.mappings.empty());
    CHECK(fixture.cuda.reservations.empty());
    const auto telemetry = allocator.stats();
    CHECK(telemetry.free_failures == 1U);
    CHECK(telemetry.quarantined_segments == 0U);
    CHECK(telemetry.last_status == XVRAM_TORCH_ALLOCATOR_QUARANTINED);
    CHECK(telemetry.last_native_error == CUDA_ERROR_UNKNOWN);
    const auto error = allocator.last_error();
    CHECK(error.status == XVRAM_TORCH_ALLOCATOR_QUARANTINED);
    CHECK(error.native_code == CUDA_ERROR_UNKNOWN);
    CHECK(std::string(error.operation.data()) == "cuCtxPopCurrent");

    const auto address_reserves = fixture.cuda.call_counts[Call::address_reserve];
    fixture.cuda.fault = {};
    CHECK(allocator.allocate(1, 0, fixture.stream) == nullptr);
    CHECK(fixture.cuda.call_counts[Call::address_reserve] == address_reserves);
  }
}

void sticky_poison_diagnostic_tests() {
  Fixture fixture;
  ActiveFake active(fixture.cuda);
  SegmentAllocator allocator(fixture.api);
  const Stream second_stream = fixture.cuda.make_stream();
  void* first = allocator.allocate(1, 0, fixture.stream);
  void* second = allocator.allocate(1, 0, second_stream);
  CHECK(first != nullptr);
  CHECK(second != nullptr);

  fixture.cuda.fault = Fault{Call::unmap, 1, CUDA_ERROR_UNKNOWN};
  deallocate_segment(allocator, fixture.cuda, first, 1, 0, fixture.stream);
  const auto poison = allocator.last_error();
  CHECK(poison.status == XVRAM_TORCH_ALLOCATOR_QUARANTINED);
  CHECK(poison.native_code == CUDA_ERROR_UNKNOWN);
  CHECK(std::string(poison.operation.data()) == "cuMemUnmap");

  fixture.cuda.fault = {};
  deallocate_segment(allocator, fixture.cuda, second, 1, 0, second_stream);
  allocator.deallocate(nullptr, 0, 0, fixture.stream);
  auto telemetry = allocator.stats();
  auto error = allocator.last_error();
  CHECK(allocator.live_allocation_count() == 1U);
  CHECK(telemetry.last_status == XVRAM_TORCH_ALLOCATOR_QUARANTINED);
  CHECK(telemetry.last_native_error == poison.native_code);
  CHECK(std::string(error.stage.data()) == std::string(poison.stage.data()));
  CHECK(std::string(error.operation.data()) == std::string(poison.operation.data()));
  CHECK(std::string(error.message.data()) == std::string(poison.message.data()));

  allocator.reset_stats();
  telemetry = allocator.stats();
  error = allocator.last_error();
  CHECK(telemetry.last_status == XVRAM_TORCH_ALLOCATOR_QUARANTINED);
  CHECK(telemetry.last_native_error == poison.native_code);
  CHECK(std::string(error.operation.data()) == std::string(poison.operation.data()));

  const auto address_reserves = fixture.cuda.call_counts[Call::address_reserve];
  CHECK(allocator.allocate(1, 0, fixture.stream) == nullptr);
  CHECK(fixture.cuda.call_counts[Call::address_reserve] == address_reserves);
  error = allocator.last_error();
  CHECK(std::string(error.operation.data()) == std::string(poison.operation.data()));
}

void unmap_failure_preserves_reservation_tests() {
  Fixture fixture;
  ActiveFake active(fixture.cuda);
  SegmentAllocator allocator(fixture.api);
  void* pointer = allocator.allocate(1, 0, fixture.stream);
  CHECK(pointer != nullptr);

  fixture.cuda.fault = Fault{Call::unmap, 1, CUDA_ERROR_UNKNOWN};
  deallocate_segment(allocator, fixture.cuda, pointer, 1, 0, fixture.stream);

  CHECK(fixture.cuda.call_counts[Call::event_synchronize] == 1U);
  CHECK(fixture.cuda.call_counts[Call::mem_release] == 1U);
  CHECK(fixture.cuda.call_counts[Call::address_free] == 0U);
  CHECK(fixture.cuda.reservations.size() == 1U);
  CHECK(fixture.cuda.mappings.size() == 1U);
  CHECK(fixture.cuda.violations.empty());
  const auto telemetry = allocator.stats();
  CHECK(telemetry.event_boundaries == 1U);
  CHECK(telemetry.unmaps == 0U);
  CHECK(telemetry.handle_releases == 1U);
  CHECK(telemetry.reservation_frees == 0U);
  CHECK(telemetry.quarantined_segments == 1U);
  CHECK(telemetry.unsafe_unmaps == 0U);
  CHECK(telemetry.last_status == XVRAM_TORCH_ALLOCATOR_QUARANTINED);
}

void release_failure_preserves_reservation_tests() {
  Fixture fixture;
  ActiveFake active(fixture.cuda);
  SegmentAllocator allocator(fixture.api);
  void* pointer = allocator.allocate(1, 0, fixture.stream);
  CHECK(pointer != nullptr);

  fixture.cuda.fault = Fault{Call::mem_release, 1, CUDA_ERROR_UNKNOWN};
  deallocate_segment(allocator, fixture.cuda, pointer, 1, 0, fixture.stream);

  CHECK(fixture.cuda.mappings.empty());
  CHECK(fixture.cuda.reservations.size() == 1U);
  CHECK(fixture.cuda.call_counts[Call::address_free] == 0U);
  CHECK(allocator.live_allocation_count() == 1U);
  const auto address_reserves = fixture.cuda.call_counts[Call::address_reserve];

  fixture.cuda.fault = {};
  CHECK(allocator.allocate(1, 0, fixture.stream) == nullptr);
  CHECK(fixture.cuda.call_counts[Call::address_reserve] == address_reserves);
  const auto telemetry = allocator.stats();
  CHECK(telemetry.free_failures == 1U);
  CHECK(telemetry.handle_releases == 0U);
  CHECK(telemetry.reservation_frees == 0U);
  CHECK(telemetry.quarantined_segments == 1U);
  CHECK(telemetry.last_status == XVRAM_TORCH_ALLOCATOR_QUARANTINED);
}

void context_switch_and_quarantine_tests() {
  {
    Fixture fixture;
    ActiveFake active(fixture.cuda);
    SegmentAllocator allocator(fixture.api);
    void* pointer = allocator.allocate(1, 0, fixture.stream);
    CHECK(pointer != nullptr);

    const Context foreign = fixture.cuda.make_context(0);
    fixture.cuda.current_context = foreign;
    deallocate_segment(allocator, fixture.cuda, pointer, 1, 0, fixture.stream);

    CHECK(fixture.cuda.current_context == foreign);
    CHECK(fixture.cuda.call_counts[Call::context_push] == 1U);
    CHECK(fixture.cuda.call_counts[Call::context_pop] == 1U);
    CHECK(fixture.cuda.mappings.empty());
    CHECK(fixture.cuda.violations.empty());
    const auto telemetry = allocator.stats();
    CHECK(telemetry.context_mismatches == 1U);
    CHECK(telemetry.quarantined_segments == 0U);
  }

  {
    Fixture fixture;
    ActiveFake active(fixture.cuda);
    SegmentAllocator allocator(fixture.api);
    void* pointer = allocator.allocate(1, 0, fixture.stream);
    CHECK(pointer != nullptr);

    fixture.cuda.current_context = fixture.cuda.make_context(0);
    fixture.cuda.fault = Fault{Call::context_push, 1, CUDA_ERROR_INVALID_CONTEXT};
    deallocate_segment(allocator, fixture.cuda, pointer, 1, 0, fixture.stream);

    CHECK(fixture.cuda.call_counts[Call::event_create] == 0U);
    CHECK(fixture.cuda.call_counts[Call::unmap] == 0U);
    CHECK(fixture.cuda.mappings.size() == 1U);
    CHECK(fixture.cuda.violations.empty());
    const auto telemetry = allocator.stats();
    CHECK(telemetry.free_failures == 1U);
    CHECK(telemetry.quarantined_segments == 1U);
    CHECK(telemetry.last_status == XVRAM_TORCH_ALLOCATOR_QUARANTINED);
  }
}

void callback_metadata_mismatch_uses_authoritative_registry_tests() {
  Fixture fixture;
  ActiveFake active(fixture.cuda);
  SegmentAllocator allocator(fixture.api);
  void* pointer = allocator.allocate(123, 0, fixture.stream);
  CHECK(pointer != nullptr);
  const Stream foreign_stream = fixture.cuda.make_stream();

  deallocate_segment(allocator, fixture.cuda, pointer, 999, 7, foreign_stream);

  CHECK(allocator.live_allocation_count() == 0U);
  CHECK(fixture.cuda.mappings.empty());
  CHECK(fixture.cuda.reservations.empty());
  CHECK(fixture.cuda.violations.empty());
  const auto telemetry = allocator.stats();
  CHECK(telemetry.size_mismatches == 1U);
  CHECK(telemetry.stream_mismatches == 1U);
  CHECK(telemetry.context_mismatches == 1U);
  CHECK(telemetry.free_failures == 0U);
  CHECK(telemetry.event_boundaries == 1U);
}

void capture_rejection_tests() {
  {
    Fixture fixture;
    ActiveFake active(fixture.cuda);
    fixture.cuda.capture_status[fixture.stream] = CU_STREAM_CAPTURE_STATUS_ACTIVE;
    SegmentAllocator allocator(fixture.api);

    CHECK(allocator.allocate(1, 0, fixture.stream) == nullptr);
    CHECK(fixture.cuda.call_counts[Call::address_reserve] == 0U);
    CHECK(fixture.cuda.reservations.empty());
    const auto telemetry = allocator.stats();
    CHECK(telemetry.capture_rejections == 1U);
    CHECK(telemetry.allocation_failures == 1U);
    CHECK(telemetry.quarantined_segments == 0U);
  }

  {
    Fixture fixture;
    ActiveFake active(fixture.cuda);
    SegmentAllocator allocator(fixture.api);
    void* pointer = allocator.allocate(1, 0, fixture.stream);
    CHECK(pointer != nullptr);

    fixture.cuda.capture_status[fixture.stream] = CU_STREAM_CAPTURE_STATUS_ACTIVE;
    deallocate_segment(allocator, fixture.cuda, pointer, 1, 0, fixture.stream);

    CHECK(fixture.cuda.call_counts[Call::event_create] == 0U);
    CHECK(fixture.cuda.call_counts[Call::unmap] == 0U);
    CHECK(fixture.cuda.mappings.size() == 1U);
    const auto telemetry = allocator.stats();
    CHECK(telemetry.capture_rejections == 1U);
    CHECK(telemetry.free_failures == 1U);
    CHECK(telemetry.quarantined_segments == 1U);
    CHECK(telemetry.last_status == XVRAM_TORCH_ALLOCATOR_QUARANTINED);
  }
}

void invalid_request_tests() {
  Fixture fixture;
  ActiveFake active(fixture.cuda);
  SegmentAllocator allocator(fixture.api);

  CHECK(allocator.allocate(0, 0, fixture.stream) == nullptr);
  CHECK(allocator.allocate(1, 1, fixture.stream) == nullptr);
  CHECK(allocator.allocate(std::numeric_limits<std::size_t>::max(), 0, fixture.stream) == nullptr);
  CHECK(fixture.cuda.call_counts[Call::address_reserve] == 0U);
  CHECK(allocator.live_allocation_count() == 0U);
  const auto telemetry = allocator.stats();
  CHECK(telemetry.allocation_calls == 3U);
  CHECK(telemetry.allocation_failures == 3U);
  CHECK(telemetry.mapped_bytes_current == 0U);
}

void unknown_pointer_and_reset_tests() {
  Fixture fixture;
  ActiveFake active(fixture.cuda);
  SegmentAllocator allocator(fixture.api);

  deallocate_segment(
      allocator, fixture.cuda,
      reinterpret_cast<void*>(static_cast<std::uintptr_t>(0xDEAD'0000ULL)), 64, 0,
      fixture.stream);
  CHECK(fixture.cuda.call_counts[Call::event_create] == 0U);
  CHECK(fixture.cuda.call_counts[Call::unmap] == 0U);
  CHECK(allocator.live_allocation_count() == 0U);
  auto telemetry = allocator.stats();
  CHECK(telemetry.free_calls == 1U);
  CHECK(telemetry.free_failures == 1U);
  CHECK(telemetry.last_status == XVRAM_TORCH_ALLOCATOR_UNKNOWN_POINTER);

  allocator.reset_stats();
  telemetry = allocator.stats();
  CHECK(telemetry.struct_size == sizeof(xvram_torch_allocator_stats_v1));
  CHECK(telemetry.abi_version == XVRAM_TORCH_ALLOCATOR_ABI_VERSION_1);
  CHECK(telemetry.allocation_calls == 0U);
  CHECK(telemetry.free_calls == 0U);
  CHECK(telemetry.free_failures == 0U);
  CHECK(telemetry.last_status == XVRAM_TORCH_ALLOCATOR_SUCCESS);
}

} // namespace

int main() {
  alignment_tests();
  successful_lifecycle_tests();
  allocation_rollback_tests();
  registry_oom_quarantine_tests();
  event_fence_failure_quarantines_tests();
  event_setup_failure_quarantines_tests();
  post_boundary_cleanup_failure_tests();
  sticky_poison_diagnostic_tests();
  unmap_failure_preserves_reservation_tests();
  release_failure_preserves_reservation_tests();
  context_switch_and_quarantine_tests();
  callback_metadata_mismatch_uses_authoritative_registry_tests();
  capture_rejection_tests();
  invalid_request_tests();
  unknown_pointer_and_reset_tests();

  if (failures != 0) {
    std::cerr << failures << " torch allocator test(s) failed\n";
    return 1;
  }
  return 0;
}
