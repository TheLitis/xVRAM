#include "residency/nvcomp_pipeline.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <new>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

using xvram::cuda::CudaApi;
using xvram::cuda::abi::DevicePointer;
using xvram::cuda::abi::Event;
using xvram::cuda::abi::Function;
using xvram::cuda::abi::Module;
using xvram::cuda::abi::Result;
using xvram::cuda::abi::Stream;
using namespace xvram::residency;

int failures = 0;

void check(const bool condition, const char* expression, const int line) {
  if (!condition) {
    std::cerr << "line " << line << ": check failed: " << expression << '\n';
    ++failures;
  }
}

#define CHECK(expression) check(static_cast<bool>(expression), #expression, __LINE__)

class FakeBlockCodec final : public BlockCodec {
public:
  [[nodiscard]] CompressionCodec codec() const noexcept override {
    return CompressionCodec::lz4;
  }
  [[nodiscard]] std::string_view implementation_name() const noexcept override {
    return "fake-rle";
  }
  [[nodiscard]] std::optional<std::size_t>
  maximum_compressed_bytes(const std::size_t input_bytes) const noexcept override {
    return input_bytes;
  }
  [[nodiscard]] CodecStatus compress(const std::span<const std::byte> input,
                                     const std::span<std::byte> output,
                                     std::size_t& written_bytes) const noexcept override {
    written_bytes = 0;
    if (input.empty() || output.empty()) {
      return CodecStatus::invalid_argument;
    }
    if (std::all_of(input.begin(), input.end(),
                    [&](const std::byte value) { return value == input.front(); })) {
      output.front() = input.front();
      written_bytes = 1;
      return CodecStatus::success;
    }
    if (output.size() < input.size()) {
      return CodecStatus::output_too_small;
    }
    std::memcpy(output.data(), input.data(), input.size());
    written_bytes = input.size();
    return CodecStatus::success;
  }
  [[nodiscard]] CodecStatus decompress(const std::span<const std::byte> input,
                                       const std::span<std::byte> output) const noexcept override {
    if (input.empty() || output.empty()) {
      return CodecStatus::corrupt_input;
    }
    if (input.size() == 1) {
      std::fill(output.begin(), output.end(), input.front());
      return CodecStatus::success;
    }
    if (input.size() != output.size()) {
      return CodecStatus::corrupt_input;
    }
    std::memcpy(output.data(), input.data(), input.size());
    return CodecStatus::success;
  }
};

enum class CudaFailureSite {
  none,
  device_alloc,
  host_alloc,
  h2d,
  d2h,
  stream_create,
  stream_synchronize,
  event_create,
  event_record,
  event_query,
  event_synchronize,
  module_load,
  module_get_function,
  launch_kernel,
};

enum class NvcompFailureSite {
  none,
  compress_alignment,
  decompress_alignment,
  compress_temp_size,
  decompress_temp_size,
  max_output_size,
  compress_launch,
  decompress_launch,
};

template <typename Site> struct InjectedFailure {
  Site site = Site::none;
  std::uint32_t calls_to_skip = 0;
  std::uint32_t matching_calls = 0;
  bool triggered = false;

  [[nodiscard]] bool should_fail(const Site candidate) noexcept {
    if (site != candidate || triggered) {
      return false;
    }
    ++matching_calls;
    if (matching_calls <= calls_to_skip) {
      return false;
    }
    triggered = true;
    return true;
  }
};

struct FakeCuda {
  struct EventState {
    bool recorded = false;
    std::uint32_t queries = 0;
  };

  std::unordered_map<void*, std::size_t> device_allocations;
  std::unordered_map<void*, std::size_t> host_allocations;
  std::unordered_set<Stream> streams;
  std::unordered_map<Event, EventState> events;
  std::vector<std::size_t> h2d_copy_sizes;
  InjectedFailure<CudaFailureSite> injected_cuda;
  InjectedFailure<NvcompFailureSite> injected_nvcomp;
  Result injected_cuda_result = CUDA_ERROR_UNKNOWN;
  Result forced_stream_synchronize = CUDA_SUCCESS;
  nvcompStatus_t injected_nvcomp_status = nvcompErrorInternal;
  bool not_ready_once = true;
  bool corrupt_compress = false;
  bool corrupt_compress_status = false;
  bool corrupt_decompress_status = false;
  std::uint32_t fail_device_frees = 0;
  std::uint32_t fail_host_frees = 0;
  std::uint32_t fail_stream_destroys = 0;
  std::uint32_t fail_event_destroys = 0;
  std::uint32_t fail_module_unloads = 0;
  bool fail_module_get_function = false;
  bool module_loaded = false;
  Result forced_query = CUDA_SUCCESS;

  ~FakeCuda() {
    for (const auto& [pointer, unused] : device_allocations) {
      (void)unused;
      ::operator delete(pointer, std::align_val_t{256});
    }
    for (const auto& [pointer, unused] : host_allocations) {
      (void)unused;
      ::operator delete(pointer, std::align_val_t{256});
    }
    for (const Stream stream : streams) {
      delete reinterpret_cast<std::uint64_t*>(stream);
    }
    for (const auto& [event, unused] : events) {
      (void)unused;
      delete reinterpret_cast<std::uint64_t*>(event);
    }
  }
};

thread_local FakeCuda* active_cuda = nullptr;

class ActiveCuda {
public:
  explicit ActiveCuda(FakeCuda& cuda) noexcept {
    active_cuda = &cuda;
  }
  ~ActiveCuda() {
    active_cuda = nullptr;
  }
};

FakeCuda& fake() {
  return *active_cuda;
}

bool consume_failure(std::uint32_t& remaining) {
  if (remaining == 0) {
    return false;
  }
  --remaining;
  return true;
}

Result CUDAAPI fake_mem_alloc(DevicePointer* output, const std::size_t bytes) {
  if (fake().injected_cuda.should_fail(CudaFailureSite::device_alloc)) {
    return fake().injected_cuda_result;
  }
  void* pointer = ::operator new(bytes, std::align_val_t{256}, std::nothrow);
  if (pointer == nullptr) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  fake().device_allocations.emplace(pointer, bytes);
  *output = static_cast<DevicePointer>(reinterpret_cast<std::uintptr_t>(pointer));
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_mem_free(const DevicePointer address) {
  void* pointer = reinterpret_cast<void*>(static_cast<std::uintptr_t>(address));
  if (consume_failure(fake().fail_device_frees)) {
    return CUDA_ERROR_UNKNOWN;
  }
  if (fake().device_allocations.erase(pointer) != 1) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  ::operator delete(pointer, std::align_val_t{256});
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_host_alloc(void** output, const std::size_t bytes, unsigned int) {
  if (fake().injected_cuda.should_fail(CudaFailureSite::host_alloc)) {
    return fake().injected_cuda_result;
  }
  void* pointer = ::operator new(bytes, std::align_val_t{256}, std::nothrow);
  if (pointer == nullptr) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  fake().host_allocations.emplace(pointer, bytes);
  *output = pointer;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_host_free(void* pointer) {
  if (consume_failure(fake().fail_host_frees)) {
    return CUDA_ERROR_UNKNOWN;
  }
  if (fake().host_allocations.erase(pointer) != 1) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  ::operator delete(pointer, std::align_val_t{256});
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_h2d(const DevicePointer destination, const void* source,
                        const std::size_t bytes, Stream) {
  if (fake().injected_cuda.should_fail(CudaFailureSite::h2d)) {
    return fake().injected_cuda_result;
  }
  fake().h2d_copy_sizes.push_back(bytes);
  std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(destination)), source, bytes);
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_d2h(void* destination, const DevicePointer source, const std::size_t bytes,
                        Stream) {
  if (fake().injected_cuda.should_fail(CudaFailureSite::d2h)) {
    return fake().injected_cuda_result;
  }
  std::memcpy(destination, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(source)),
              bytes);
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_stream_create(Stream* output, unsigned int) {
  if (fake().injected_cuda.should_fail(CudaFailureSite::stream_create)) {
    return fake().injected_cuda_result;
  }
  auto* token = new (std::nothrow) std::uint64_t{1};
  if (token == nullptr) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  *output = reinterpret_cast<Stream>(token);
  fake().streams.insert(*output);
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_stream_destroy(const Stream stream) {
  if (consume_failure(fake().fail_stream_destroys)) {
    return CUDA_ERROR_UNKNOWN;
  }
  if (fake().streams.erase(stream) != 1) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  delete reinterpret_cast<std::uint64_t*>(stream);
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_stream_synchronize(Stream) {
  if (fake().injected_cuda.should_fail(CudaFailureSite::stream_synchronize)) {
    return fake().injected_cuda_result;
  }
  return fake().forced_stream_synchronize;
}

Result CUDAAPI fake_event_create(Event* output, unsigned int) {
  if (fake().injected_cuda.should_fail(CudaFailureSite::event_create)) {
    return fake().injected_cuda_result;
  }
  auto* token = new (std::nothrow) std::uint64_t{1};
  if (token == nullptr) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  *output = reinterpret_cast<Event>(token);
  fake().events.emplace(*output, FakeCuda::EventState{});
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_event_destroy(const Event event) {
  if (consume_failure(fake().fail_event_destroys)) {
    return CUDA_ERROR_UNKNOWN;
  }
  if (fake().events.erase(event) != 1) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  delete reinterpret_cast<std::uint64_t*>(event);
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_event_record(const Event event, Stream) {
  if (fake().injected_cuda.should_fail(CudaFailureSite::event_record)) {
    return fake().injected_cuda_result;
  }
  auto found = fake().events.find(event);
  if (found == fake().events.end()) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  found->second = {true, 0};
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_event_query(const Event event) {
  if (fake().injected_cuda.should_fail(CudaFailureSite::event_query)) {
    return fake().injected_cuda_result;
  }
  if (fake().forced_query != CUDA_SUCCESS) {
    return fake().forced_query;
  }
  auto found = fake().events.find(event);
  if (found == fake().events.end() || !found->second.recorded) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  if (fake().not_ready_once && found->second.queries++ == 0) {
    return CUDA_ERROR_NOT_READY;
  }
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_event_synchronize(const Event event) {
  if (fake().injected_cuda.should_fail(CudaFailureSite::event_synchronize)) {
    return fake().injected_cuda_result;
  }
  const auto found = fake().events.find(event);
  return found != fake().events.end() && found->second.recorded ? CUDA_SUCCESS
                                                                : CUDA_ERROR_INVALID_HANDLE;
}

Result CUDAAPI fake_module_load(Module* module, const void* image) {
  if (fake().injected_cuda.should_fail(CudaFailureSite::module_load)) {
    return fake().injected_cuda_result;
  }
  if (module == nullptr || image == nullptr) {
    return CUDA_ERROR_INVALID_IMAGE;
  }
  *module = reinterpret_cast<Module>(static_cast<std::uintptr_t>(1));
  fake().module_loaded = true;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_module_get_function(Function* function, const Module module, const char* name) {
  if (fake().injected_cuda.should_fail(CudaFailureSite::module_get_function) ||
      fake().fail_module_get_function || function == nullptr || module == nullptr ||
      !fake().module_loaded || name == nullptr || std::string_view{name} != "xvram_lz4_verify_v1") {
    return CUDA_ERROR_NOT_FOUND;
  }
  *function = reinterpret_cast<Function>(static_cast<std::uintptr_t>(2));
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_module_unload(Module module) {
  if (consume_failure(fake().fail_module_unloads)) {
    return CUDA_ERROR_UNKNOWN;
  }
  if (module == nullptr || !fake().module_loaded) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  fake().module_loaded = false;
  return CUDA_SUCCESS;
}

Result CUDAAPI fake_launch_kernel(Function function, unsigned int, unsigned int, unsigned int,
                                  unsigned int, unsigned int, unsigned int, unsigned int, Stream,
                                  void** parameters, void**) {
  if (fake().injected_cuda.should_fail(CudaFailureSite::launch_kernel)) {
    return fake().injected_cuda_result;
  }
  if (function == nullptr || parameters == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const DevicePointer input = *static_cast<DevicePointer*>(parameters[0]);
  const std::uint64_t bytes = *static_cast<std::uint64_t*>(parameters[1]);
  const DevicePointer token = *static_cast<DevicePointer*>(parameters[2]);
  *reinterpret_cast<ContentToken*>(static_cast<std::uintptr_t>(token)) =
      compute_content_token(std::span<const std::byte>{
          reinterpret_cast<const std::byte*>(static_cast<std::uintptr_t>(input)),
          static_cast<std::size_t>(bytes)});
  return CUDA_SUCCESS;
}

void install_cuda(CudaApi& api) {
  api.mem_alloc_ = fake_mem_alloc;
  api.mem_free_ = fake_mem_free;
  api.mem_host_alloc_ = fake_host_alloc;
  api.mem_free_host_ = fake_host_free;
  api.memcpy_h2d_async_ = fake_h2d;
  api.memcpy_d2h_async_ = fake_d2h;
  api.stream_create_ = fake_stream_create;
  api.stream_destroy_ = fake_stream_destroy;
  api.stream_synchronize_ = fake_stream_synchronize;
  api.event_create_ = fake_event_create;
  api.event_destroy_ = fake_event_destroy;
  api.event_record_ = fake_event_record;
  api.event_query_ = fake_event_query;
  api.event_synchronize_ = fake_event_synchronize;
  api.module_load_data_ = fake_module_load;
  api.module_get_function_ = fake_module_get_function;
  api.module_unload_ = fake_module_unload;
  api.launch_kernel_ = fake_launch_kernel;
}

nvcompStatus_t fake_properties(nvcompProperties_t* properties) {
  properties->version = 5300;
  properties->cudart_version = 13030;
  return nvcompSuccess;
}

const char* fake_status_string(nvcompStatus_t) {
  return "fake";
}

nvcompStatus_t fake_alignment(nvcompBatchedLZ4CompressOpts_t,
                              nvcompAlignmentRequirements_t* alignment) {
  if (fake().injected_nvcomp.should_fail(NvcompFailureSite::compress_alignment)) {
    return fake().injected_nvcomp_status;
  }
  *alignment = {1, 1, 1};
  return nvcompSuccess;
}

nvcompStatus_t fake_decompression_alignment(nvcompBatchedLZ4DecompressOpts_t,
                                            nvcompAlignmentRequirements_t* alignment) {
  if (fake().injected_nvcomp.should_fail(NvcompFailureSite::decompress_alignment)) {
    return fake().injected_nvcomp_status;
  }
  *alignment = {1, 1, 1};
  return nvcompSuccess;
}

nvcompStatus_t fake_compress_temp(std::size_t, std::size_t, nvcompBatchedLZ4CompressOpts_t,
                                  std::size_t* bytes, std::size_t) {
  if (fake().injected_nvcomp.should_fail(NvcompFailureSite::compress_temp_size)) {
    return fake().injected_nvcomp_status;
  }
  *bytes = 4096;
  return nvcompSuccess;
}

nvcompStatus_t fake_decompress_temp(std::size_t, std::size_t, nvcompBatchedLZ4DecompressOpts_t,
                                    std::size_t* bytes, std::size_t) {
  if (fake().injected_nvcomp.should_fail(NvcompFailureSite::decompress_temp_size)) {
    return fake().injected_nvcomp_status;
  }
  *bytes = 4096;
  return nvcompSuccess;
}

nvcompStatus_t fake_max_output(const std::size_t bytes, nvcompBatchedLZ4CompressOpts_t,
                               std::size_t* output) {
  if (fake().injected_nvcomp.should_fail(NvcompFailureSite::max_output_size)) {
    return fake().injected_nvcomp_status;
  }
  *output = bytes;
  return nvcompSuccess;
}

nvcompStatus_t fake_compress(const void* const* inputs, const std::size_t* input_sizes, std::size_t,
                             const std::size_t count, void*, std::size_t, void* const* outputs,
                             std::size_t* output_sizes, nvcompBatchedLZ4CompressOpts_t,
                             nvcompStatus_t* statuses, cudaStream_t) {
  if (fake().injected_nvcomp.should_fail(NvcompFailureSite::compress_launch)) {
    return fake().injected_nvcomp_status;
  }
  for (std::size_t index = 0; index < count; ++index) {
    const auto* input = static_cast<const std::byte*>(inputs[index]);
    auto* output = static_cast<std::byte*>(outputs[index]);
    const bool uniform = std::all_of(input, input + input_sizes[index],
                                     [&](const std::byte value) { return value == input[0]; });
    if (uniform) {
      output[0] = input[0];
      if (fake().corrupt_compress) {
        output[0] ^= std::byte{1};
      }
      output_sizes[index] = 1;
    } else {
      std::memcpy(output, input, input_sizes[index]);
      output_sizes[index] = input_sizes[index];
    }
    statuses[index] = fake().corrupt_compress_status ? nvcompErrorInternal : nvcompSuccess;
  }
  return nvcompSuccess;
}

nvcompStatus_t fake_decompress(const void* const* inputs, const std::size_t* input_sizes,
                               const std::size_t* output_capacities, std::size_t* actual_sizes,
                               const std::size_t count, void*, std::size_t, void* const* outputs,
                               nvcompBatchedLZ4DecompressOpts_t options, nvcompStatus_t* statuses,
                               cudaStream_t) {
  if (fake().injected_nvcomp.should_fail(NvcompFailureSite::decompress_launch)) {
    return fake().injected_nvcomp_status;
  }
  if (options.backend != NVCOMP_DECOMPRESS_BACKEND_CUDA) {
    return nvcompErrorInvalidValue;
  }
  for (std::size_t index = 0; index < count; ++index) {
    if (input_sizes[index] != 1) {
      statuses[index] = nvcompErrorCannotDecompress;
      actual_sizes[index] = 0;
      continue;
    }
    std::memset(outputs[index], std::to_integer<int>(*static_cast<const std::byte*>(inputs[index])),
                output_capacities[index]);
    actual_sizes[index] = output_capacities[index];
    statuses[index] = fake().corrupt_decompress_status ? nvcompErrorInternal : nvcompSuccess;
  }
  return nvcompSuccess;
}

xvram::nvcomp::NvcompDispatch make_nvcomp_dispatch() {
  return {fake_properties,
          fake_status_string,
          fake_alignment,
          fake_compress_temp,
          fake_max_output,
          fake_compress,
          fake_decompression_alignment,
          fake_decompress_temp,
          fake_decompress};
}

Result fake_verify(void*, const DevicePointer address, const std::uint64_t bytes,
                   const ContentToken expected, const DevicePointer match, Stream) noexcept {
  const auto actual = compute_content_token(std::span<const std::byte>{
      reinterpret_cast<const std::byte*>(static_cast<std::uintptr_t>(address)),
      static_cast<std::size_t>(bytes)});
  *reinterpret_cast<std::uint32_t*>(static_cast<std::uintptr_t>(match)) =
      actual == expected ? 1U : 0U;
  return CUDA_SUCCESS;
}

struct Fixture {
  FakeCuda cuda;
  ActiveCuda active{cuda};
  CudaApi cuda_api{CudaApi::InjectedDispatch{}};
  xvram::nvcomp::NvcompApi nvcomp_api{make_nvcomp_dispatch(), "fake-nvcomp"};
  FakeBlockCodec codec;

  Fixture() {
    install_cuda(cuda_api);
  }

  [[nodiscard]] NvcompPipelineConfig config() const {
    NvcompPipelineConfig value;
    value.chunk_bytes = 3 * nvcomp_lz4_block_bytes;
    value.slot_count = 2;
    value.workspace_cap_bytes = 16 * 1024;
    value.verification_submit = fake_verify;
    return value;
  }
};

struct TraceCapture {
  std::array<CompressionTraceEvent, 32> events{};
  std::size_t size = 0;
};

void capture_trace(void* const user_data, const CompressionTraceEvent& event) noexcept {
  auto& capture = *static_cast<TraceCapture*>(user_data);
  if (capture.size < capture.events.size()) {
    capture.events[capture.size++] = event;
  }
}

[[nodiscard]] bool same_trace_identity(const CompressionTraceEvent& left,
                                       const CompressionTraceEvent& right) noexcept {
  return left.key == right.key && left.operation_id == right.operation_id &&
         left.source_generation == right.source_generation &&
         left.slot_generation == right.slot_generation;
}

std::vector<std::byte> source_bytes() {
  std::vector<std::byte> bytes(2 * nvcomp_lz4_block_bytes);
  std::fill_n(bytes.begin(), static_cast<std::size_t>(nvcomp_lz4_block_bytes), std::byte{0x41});
  for (std::size_t index = nvcomp_lz4_block_bytes; index < bytes.size(); ++index) {
    bytes[index] = static_cast<std::byte>(index & 0xffU);
  }
  return bytes;
}

Lz4BlocksV1 source_container(const std::vector<std::byte>& bytes) {
  Lz4BlocksV1 source;
  source.valid_bytes = bytes.size();
  source.generation = 7;
  source.content_token = compute_content_token(bytes);
  Lz4BlockV1 compressed;
  compressed.storage = BlockStorage::lz4;
  compressed.uncompressed_bytes = static_cast<std::uint32_t>(nvcomp_lz4_block_bytes);
  compressed.payload = {std::byte{0x41}};
  source.blocks.push_back(std::move(compressed));
  Lz4BlockV1 raw;
  raw.storage = BlockStorage::raw;
  raw.uncompressed_bytes = static_cast<std::uint32_t>(nvcomp_lz4_block_bytes);
  raw.payload.assign(bytes.begin() + static_cast<std::ptrdiff_t>(nvcomp_lz4_block_bytes),
                     bytes.end());
  source.blocks.push_back(std::move(raw));
  return source;
}

void decode_and_event_boundary_test() {
  Fixture fixture;
  TraceCapture trace;
  auto config = fixture.config();
  config.trace_observer = capture_trace;
  config.trace_user_data = &trace;
  NvcompLz4Pipeline pipeline(fixture.cuda_api, fixture.nvcomp_api, fixture.codec, config);
  CHECK(pipeline.setup() == NvcompPipelineStatus::success);
  constexpr std::uint64_t expected_metadata_bytes = 192;
  constexpr std::uint64_t expected_slot_capacity =
      2 * (3 * nvcomp_lz4_block_bytes + expected_metadata_bytes);
  CHECK(pipeline.telemetry().device_slot_capacity_bytes == expected_slot_capacity);
  CHECK(pipeline.telemetry().device_slot_peak_bytes == expected_slot_capacity + 2 * 4096);
  const auto initial = source_bytes();
  auto expected = initial;
  expected.resize(expected.size() + static_cast<std::size_t>(nvcomp_lz4_block_bytes), std::byte{0});
  auto source = source_container(initial);
  source.valid_bytes = expected.size();
  source.content_token = compute_content_token(expected);
  Lz4BlockV1 zero;
  zero.storage = BlockStorage::implicit_zero;
  zero.uncompressed_bytes = static_cast<std::uint32_t>(nvcomp_lz4_block_bytes);
  source.blocks.push_back(std::move(zero));
  std::vector<std::byte> output(expected.size());
  NvcompPipelineTicket ticket;
  NvcompDecodeRequest request;
  request.source = &source;
  request.stable_output =
      static_cast<DevicePointer>(reinterpret_cast<std::uintptr_t>(output.data()));
  request.key = ChunkKey{AllocationId{41}, 9};
  request.path = CompressionPath::cpu_lz4_gpu_decode;
  request.speculative = true;
  CHECK(pipeline.decode(request, ticket) == NvcompPipelineStatus::success);
  CHECK(ticket);
  CHECK(trace.size == 2);
  CHECK(trace.events[0].kind == CompressionTraceEventKind::h2d_submit);
  CHECK(trace.events[1].kind == CompressionTraceEventKind::decode_submit);
  CHECK(same_trace_identity(trace.events[0], trace.events[1]));
  CHECK(trace.events[0].key == request.key);
  CHECK(trace.events[0].operation_id == ticket.operation_id);
  CHECK(trace.events[0].source_generation == source.generation);
  CHECK(trace.events[0].slot_generation == ticket.slot_generation);
  CHECK(trace.events[0].path == request.path);
  CHECK(trace.events[0].logical_bytes == expected.size());
  CHECK(pipeline.telemetry().pcie_h2d_metadata_bytes != 0);
  CHECK(trace.events[0].physical_bytes ==
        pipeline.telemetry().pcie_h2d_payload_bytes + pipeline.telemetry().pcie_h2d_metadata_bytes);
  CHECK(trace.events[0].from_representation == BackingRepresentation::lz4_blocks);
  CHECK(!trace.events[0].target_generation.has_value());
  CHECK(trace.events[0].speculative);
  CHECK(pipeline.poll(ticket) == NvcompPipelineStatus::not_ready);
  CHECK(trace.size == 2);
  CHECK(pipeline.poll(ticket) == NvcompPipelineStatus::success);
  CHECK(trace.size == 4);
  CHECK(trace.events[2].kind == CompressionTraceEventKind::h2d_retire);
  CHECK(trace.events[3].kind == CompressionTraceEventKind::decode_retire);
  CHECK(same_trace_identity(trace.events[0], trace.events[2]));
  CHECK(same_trace_identity(trace.events[1], trace.events[3]));
  CHECK(output == expected);
  CHECK(pipeline.poll(ticket) == NvcompPipelineStatus::stale_ticket);
  CHECK(pipeline.telemetry().logical_decode_bytes == expected.size());
  CHECK(pipeline.telemetry().pcie_h2d_payload_bytes == 2 * nvcomp_lz4_block_bytes + 1);
  CHECK(pipeline.telemetry().zero_h2d_blocks == 1);
  CHECK(pipeline.telemetry().events_recorded == 1);
  CHECK(pipeline.telemetry().events_retired == 1);
  CHECK(pipeline.telemetry().verification_submissions == 1);
  CHECK(pipeline.close() == NvcompPipelineStatus::success);
  CHECK(pipeline.telemetry().workspace_bytes == 0);
  CHECK(pipeline.telemetry().device_slot_bytes == 0);
  CHECK(pipeline.telemetry().device_slot_capacity_bytes == expected_slot_capacity);
  CHECK(pipeline.telemetry().pinned_slot_bytes == 0);
}

void compressed_decode_payload_is_submitted_as_one_batch_test() {
  Fixture fixture;
  NvcompLz4Pipeline pipeline(fixture.cuda_api, fixture.nvcomp_api, fixture.codec, fixture.config());
  CHECK(pipeline.setup() == NvcompPipelineStatus::success);

  Lz4BlocksV1 source;
  source.valid_bytes = 3 * nvcomp_lz4_block_bytes;
  source.generation = 3;
  std::vector<std::byte> expected(static_cast<std::size_t>(source.valid_bytes));
  for (std::uint32_t index = 0; index < 3; ++index) {
    const std::byte value = static_cast<std::byte>(0x31U + index);
    std::fill_n(expected.begin() + static_cast<std::ptrdiff_t>(index * nvcomp_lz4_block_bytes),
                static_cast<std::size_t>(nvcomp_lz4_block_bytes), value);
    Lz4BlockV1 block;
    block.storage = BlockStorage::lz4;
    block.uncompressed_bytes = static_cast<std::uint32_t>(nvcomp_lz4_block_bytes);
    block.payload = {value};
    source.blocks.push_back(std::move(block));
  }
  source.content_token = compute_content_token(expected);

  std::vector<std::byte> output(expected.size());
  NvcompPipelineTicket ticket;
  CHECK(pipeline.decode(
            {&source, static_cast<DevicePointer>(reinterpret_cast<std::uintptr_t>(output.data()))},
            ticket) == NvcompPipelineStatus::success);
  CHECK(pipeline.wait(ticket) == NvcompPipelineStatus::success);
  CHECK(output == expected);
  CHECK(std::count(fixture.cuda.h2d_copy_sizes.begin(), fixture.cuda.h2d_copy_sizes.end(), 1U) ==
        0);
  CHECK(std::count(fixture.cuda.h2d_copy_sizes.begin(), fixture.cuda.h2d_copy_sizes.end(), 3U) ==
        1);
  CHECK(pipeline.telemetry().pcie_h2d_payload_bytes == 3);
  CHECK(pipeline.close() == NvcompPipelineStatus::success);
}

void encode_fallback_and_generation_test() {
  Fixture fixture;
  TraceCapture trace;
  auto config = fixture.config();
  config.trace_observer = capture_trace;
  config.trace_user_data = &trace;
  NvcompLz4Pipeline pipeline(fixture.cuda_api, fixture.nvcomp_api, fixture.codec, config);
  CHECK(pipeline.setup() == NvcompPipelineStatus::success);
  auto input = source_bytes();
  NvcompEncodeRequest request;
  request.stable_input = static_cast<DevicePointer>(reinterpret_cast<std::uintptr_t>(input.data()));
  request.valid_bytes = input.size();
  request.source_generation = 11;
  request.expected_token = compute_content_token(input);
  request.key = ChunkKey{AllocationId{52}, 7};
  request.path = CompressionPath::nvcomp_gpu_codec;
  NvcompPipelineTicket ticket;
  CHECK(pipeline.encode(request, ticket) == NvcompPipelineStatus::success);
  CHECK(trace.size == 1);
  CHECK(trace.events[0].kind == CompressionTraceEventKind::encode_submit);
  CHECK(trace.events[0].key == request.key);
  CHECK(trace.events[0].operation_id == ticket.operation_id);
  CHECK(trace.events[0].source_generation == request.source_generation);
  CHECK(trace.events[0].target_generation == request.source_generation + 1);
  CHECK(trace.events[0].slot_generation == ticket.slot_generation);
  CHECK(pipeline.poll(ticket) == NvcompPipelineStatus::not_ready);
  CHECK(trace.size == 1);
  CHECK(pipeline.poll(ticket) == NvcompPipelineStatus::not_ready);
  CHECK(trace.size == 3);
  CHECK(trace.events[1].kind == CompressionTraceEventKind::encode_retire);
  CHECK(trace.events[2].kind == CompressionTraceEventKind::d2h_submit);
  CHECK(same_trace_identity(trace.events[0], trace.events[1]));
  CHECK(same_trace_identity(trace.events[0], trace.events[2]));
  CHECK(trace.events[1].physical_bytes == nvcomp_lz4_block_bytes + 1);
  CHECK(pipeline.telemetry().pcie_d2h_metadata_bytes != 0);
  CHECK(trace.events[2].physical_bytes ==
        pipeline.telemetry().pcie_d2h_payload_bytes + pipeline.telemetry().pcie_d2h_metadata_bytes);
  CHECK(pipeline.poll(ticket) == NvcompPipelineStatus::not_ready);
  CHECK(trace.size == 3);
  Lz4BlocksV1 candidate;
  CHECK(pipeline.poll(ticket, &candidate) == NvcompPipelineStatus::success);
  CHECK(trace.size == 3);
  CHECK(candidate.generation == 12);
  CHECK(candidate.valid_bytes == input.size());
  CHECK(candidate.content_token == *request.expected_token);
  CHECK(candidate.blocks.size() == 2);
  CHECK(candidate.blocks[0].storage == BlockStorage::lz4);
  CHECK(candidate.blocks[0].payload.size() == 1);
  CHECK(candidate.blocks[1].storage == BlockStorage::raw);
  CHECK(candidate.blocks[1].payload.size() == nvcomp_lz4_block_bytes);
  CHECK(pipeline.telemetry().pcie_d2h_payload_bytes == nvcomp_lz4_block_bytes + 1);
  CHECK(pipeline.telemetry().raw_fallback_blocks == 1);
  CHECK(pipeline.telemetry().raw_fallback_bytes == nvcomp_lz4_block_bytes);
  CHECK(pipeline.telemetry().events_recorded == 2);
  CHECK(pipeline.telemetry().events_retired == 2);
  CHECK(pipeline.telemetry().operations_retired == 0);
  CHECK(pipeline.acknowledge_host_commit(ticket) == NvcompPipelineStatus::success);
  CHECK(trace.size == 4);
  CHECK(trace.events[3].kind == CompressionTraceEventKind::d2h_retire);
  CHECK(same_trace_identity(trace.events[2], trace.events[3]));
  CHECK(trace.events[3].physical_bytes ==
        pipeline.telemetry().pcie_d2h_payload_bytes + pipeline.telemetry().pcie_d2h_metadata_bytes);
  CHECK(pipeline.telemetry().operations_retired == 1);
  CHECK(pipeline.poll(ticket, &candidate) == NvcompPipelineStatus::stale_ticket);
  CHECK(pipeline.close() == NvcompPipelineStatus::success);
}

void trace_slot_generation_reuse_test() {
  Fixture fixture;
  TraceCapture trace;
  auto config = fixture.config();
  config.trace_observer = capture_trace;
  config.trace_user_data = &trace;
  NvcompLz4Pipeline pipeline(fixture.cuda_api, fixture.nvcomp_api, fixture.codec, config);
  CHECK(pipeline.setup() == NvcompPipelineStatus::success);
  const auto bytes = source_bytes();
  const auto source = source_container(bytes);
  std::vector<std::byte> output(bytes.size());
  std::array<NvcompPipelineTicket, 3> tickets{};
  for (std::size_t index = 0; index < tickets.size(); ++index) {
    NvcompDecodeRequest request;
    request.source = &source;
    request.stable_output =
        static_cast<DevicePointer>(reinterpret_cast<std::uintptr_t>(output.data()));
    request.key = ChunkKey{AllocationId{73}, index};
    CHECK(pipeline.decode(request, tickets[index]) == NvcompPipelineStatus::success);
    CHECK(pipeline.wait(tickets[index]) == NvcompPipelineStatus::success);
  }
  CHECK(tickets[0].slot_index == tickets[2].slot_index);
  CHECK(tickets[2].slot_generation == tickets[0].slot_generation + 1);
  CHECK(tickets[0].operation_id != tickets[2].operation_id);
  CHECK(trace.size == 12);
  CHECK(trace.events[8].slot_generation == tickets[2].slot_generation);
  CHECK(trace.events[8].operation_id == tickets[2].operation_id);
  CHECK(pipeline.close() == NvcompPipelineStatus::success);
}

void slot_aba_and_quarantine_test() {
  Fixture fixture;
  TraceCapture trace;
  auto config = fixture.config();
  config.trace_observer = capture_trace;
  config.trace_user_data = &trace;
  NvcompLz4Pipeline pipeline(fixture.cuda_api, fixture.nvcomp_api, fixture.codec, config);
  CHECK(pipeline.setup() == NvcompPipelineStatus::success);
  const auto bytes = source_bytes();
  const auto source = source_container(bytes);
  std::vector<std::byte> output(bytes.size());
  NvcompPipelineTicket first;
  CHECK(pipeline.decode(
            {&source, static_cast<DevicePointer>(reinterpret_cast<std::uintptr_t>(output.data()))},
            first) == NvcompPipelineStatus::success);
  CHECK(pipeline.wait(first) == NvcompPipelineStatus::success);
  CHECK(trace.size == 4);

  NvcompPipelineTicket second;
  CHECK(pipeline.decode(
            {&source, static_cast<DevicePointer>(reinterpret_cast<std::uintptr_t>(output.data()))},
            second) == NvcompPipelineStatus::success);
  CHECK(trace.size == 6);
  CHECK(second.operation_id != first.operation_id);
  CHECK(pipeline.poll(first) == NvcompPipelineStatus::stale_ticket);

  fixture.cuda.forced_query = CUDA_ERROR_INVALID_CONTEXT;
  CHECK(pipeline.poll(second) == NvcompPipelineStatus::quarantined);
  CHECK(trace.size == 6);
  CHECK(trace.events[4].kind == CompressionTraceEventKind::h2d_submit);
  CHECK(trace.events[5].kind == CompressionTraceEventKind::decode_submit);
  CHECK(pipeline.quarantined());
  CHECK(pipeline.telemetry().quarantines == 1);
  CHECK(pipeline.close() == NvcompPipelineStatus::quarantined);
}

void polling_timeout_quarantines_test() {
  Fixture fixture;
  auto config = fixture.config();
  config.wait_timeout = std::chrono::milliseconds{1};
  NvcompLz4Pipeline pipeline(fixture.cuda_api, fixture.nvcomp_api, fixture.codec, config);
  CHECK(pipeline.setup() == NvcompPipelineStatus::success);
  const auto bytes = source_bytes();
  const auto source = source_container(bytes);
  std::vector<std::byte> output(bytes.size());
  fixture.cuda.forced_query = CUDA_ERROR_NOT_READY;
  NvcompPipelineTicket ticket;
  CHECK(pipeline.decode(
            {&source, static_cast<DevicePointer>(reinterpret_cast<std::uintptr_t>(output.data()))},
            ticket) == NvcompPipelineStatus::success);
  CHECK(pipeline.wait(ticket) == NvcompPipelineStatus::quarantined);
  CHECK(pipeline.quarantined());
  CHECK(pipeline.telemetry().quarantines == 1);
  CHECK(pipeline.close() == NvcompPipelineStatus::quarantined);
}

void encode_source_token_rejects_corrupt_candidate_test() {
  Fixture fixture;
  auto config = fixture.config();
  config.verification_submit = nullptr;
  fixture.cuda.corrupt_compress = true;
  NvcompLz4Pipeline pipeline(fixture.cuda_api, fixture.nvcomp_api, fixture.codec, config);
  CHECK(pipeline.setup() == NvcompPipelineStatus::success);
  auto input = source_bytes();
  NvcompPipelineTicket ticket;
  CHECK(pipeline.encode({static_cast<DevicePointer>(reinterpret_cast<std::uintptr_t>(input.data())),
                         input.size(), 20, std::nullopt},
                        ticket) == NvcompPipelineStatus::success);
  Lz4BlocksV1 candidate;
  CHECK(pipeline.wait(ticket, &candidate) == NvcompPipelineStatus::corrupt_data);
  CHECK(pipeline.telemetry().verification_failures == 1);
  CHECK(pipeline.close() == NvcompPipelineStatus::success);
}

void uncommitted_encode_quarantines_test() {
  Fixture fixture;
  TraceCapture trace;
  auto config = fixture.config();
  config.trace_observer = capture_trace;
  config.trace_user_data = &trace;
  NvcompLz4Pipeline pipeline(fixture.cuda_api, fixture.nvcomp_api, fixture.codec, config);
  CHECK(pipeline.setup() == NvcompPipelineStatus::success);
  auto input = source_bytes();
  NvcompPipelineTicket ticket;
  const ContentToken expected = compute_content_token(input);
  CHECK(pipeline.encode({static_cast<DevicePointer>(reinterpret_cast<std::uintptr_t>(input.data())),
                         input.size(), 30, expected},
                        ticket) == NvcompPipelineStatus::success);
  Lz4BlocksV1 candidate;
  CHECK(pipeline.wait(ticket, &candidate) == NvcompPipelineStatus::success);
  CHECK(trace.size == 3);
  CHECK(trace.events[2].kind == CompressionTraceEventKind::d2h_submit);
  CHECK(pipeline.telemetry().operations_retired == 0);
  const std::size_t device_allocations = fixture.cuda.device_allocations.size();
  const std::size_t host_allocations = fixture.cuda.host_allocations.size();
  const std::size_t streams = fixture.cuda.streams.size();
  const std::size_t events = fixture.cuda.events.size();
  CHECK(pipeline.close() == NvcompPipelineStatus::quarantined);
  CHECK(trace.size == 3);
  CHECK(pipeline.quarantined());
  CHECK(pipeline.close() == NvcompPipelineStatus::quarantined);
  CHECK(fixture.cuda.device_allocations.size() == device_allocations);
  CHECK(fixture.cuda.host_allocations.size() == host_allocations);
  CHECK(fixture.cuda.streams.size() == streams);
  CHECK(fixture.cuda.events.size() == events);
}

void setup_cleanup_failure_retains_resources_for_retry_test() {
  Fixture fixture;
  auto config = fixture.config();
  config.verification_submit = nullptr;
  fixture.cuda.fail_module_get_function = true;
  fixture.cuda.fail_device_frees = 1;
  NvcompLz4Pipeline pipeline(fixture.cuda_api, fixture.nvcomp_api, fixture.codec, config);

  CHECK(pipeline.setup() == NvcompPipelineStatus::cleanup_failure);
  CHECK(!pipeline.is_setup());
  CHECK(!pipeline.quarantined());
  CHECK(fixture.cuda.device_allocations.size() == 1);
  CHECK(fixture.cuda.host_allocations.empty());
  CHECK(fixture.cuda.streams.empty());
  CHECK(fixture.cuda.events.empty());
  CHECK(!fixture.cuda.module_loaded);
  CHECK(pipeline.telemetry().device_slot_bytes != 0);
  CHECK(pipeline.setup() == NvcompPipelineStatus::invalid_configuration);

  fixture.cuda.fail_module_get_function = false;
  CHECK(pipeline.close() == NvcompPipelineStatus::success);
  CHECK(fixture.cuda.device_allocations.empty());
  CHECK(pipeline.telemetry().workspace_bytes == 0);
  CHECK(pipeline.telemetry().device_slot_bytes == 0);
  CHECK(pipeline.telemetry().pinned_slot_bytes == 0);
  CHECK(pipeline.close() == NvcompPipelineStatus::success);
}

void setup_module_cleanup_failure_is_retryable_test() {
  Fixture fixture;
  auto config = fixture.config();
  config.verification_submit = nullptr;
  fixture.cuda.fail_module_get_function = true;
  fixture.cuda.fail_module_unloads = 1;
  NvcompLz4Pipeline pipeline(fixture.cuda_api, fixture.nvcomp_api, fixture.codec, config);

  CHECK(pipeline.setup() == NvcompPipelineStatus::cleanup_failure);
  CHECK(!pipeline.is_setup());
  CHECK(!pipeline.quarantined());
  CHECK(fixture.cuda.module_loaded);
  CHECK(fixture.cuda.device_allocations.empty());
  CHECK(fixture.cuda.host_allocations.empty());
  CHECK(fixture.cuda.streams.empty());
  CHECK(fixture.cuda.events.empty());

  fixture.cuda.fail_module_get_function = false;
  CHECK(pipeline.close() == NvcompPipelineStatus::success);
  CHECK(!fixture.cuda.module_loaded);
  CHECK(pipeline.close() == NvcompPipelineStatus::success);
}

void close_cleanup_failure_preserves_exact_owners_for_retry_test() {
  Fixture fixture;
  NvcompLz4Pipeline pipeline(fixture.cuda_api, fixture.nvcomp_api, fixture.codec, fixture.config());
  CHECK(pipeline.setup() == NvcompPipelineStatus::success);
  fixture.cuda.fail_device_frees = 1;
  fixture.cuda.fail_host_frees = 1;
  fixture.cuda.fail_stream_destroys = 1;
  fixture.cuda.fail_event_destroys = 1;

  CHECK(pipeline.close() == NvcompPipelineStatus::cleanup_failure);
  CHECK(!pipeline.is_setup());
  CHECK(!pipeline.quarantined());
  CHECK(fixture.cuda.device_allocations.size() == 1);
  CHECK(fixture.cuda.host_allocations.size() == 1);
  CHECK(fixture.cuda.streams.size() == 1);
  CHECK(fixture.cuda.events.size() == 1);
  CHECK(pipeline.telemetry().device_slot_bytes != 0);
  CHECK(pipeline.telemetry().pinned_slot_bytes != 0);
  CHECK(pipeline.setup() == NvcompPipelineStatus::invalid_configuration);

  CHECK(pipeline.close() == NvcompPipelineStatus::success);
  CHECK(fixture.cuda.device_allocations.empty());
  CHECK(fixture.cuda.host_allocations.empty());
  CHECK(fixture.cuda.streams.empty());
  CHECK(fixture.cuda.events.empty());
  CHECK(pipeline.telemetry().workspace_bytes == 0);
  CHECK(pipeline.telemetry().device_slot_bytes == 0);
  CHECK(pipeline.telemetry().pinned_slot_bytes == 0);
  CHECK(pipeline.close() == NvcompPipelineStatus::success);
}

void setup_fault_injection_matrix_test() {
  struct CudaCase {
    CudaFailureSite site;
    std::uint32_t calls_to_skip;
    bool needs_embedded_verifier;
  };
  constexpr std::array cuda_cases{
      CudaCase{CudaFailureSite::stream_create, 0, false},
      CudaCase{CudaFailureSite::stream_create, 1, false},
      CudaCase{CudaFailureSite::event_create, 0, false},
      CudaCase{CudaFailureSite::event_create, 1, false},
      CudaCase{CudaFailureSite::device_alloc, 0, false},
      CudaCase{CudaFailureSite::device_alloc, 1, false},
      CudaCase{CudaFailureSite::device_alloc, 2, false},
      CudaCase{CudaFailureSite::device_alloc, 3, false},
      CudaCase{CudaFailureSite::host_alloc, 0, false},
      CudaCase{CudaFailureSite::host_alloc, 1, false},
      CudaCase{CudaFailureSite::host_alloc, 2, false},
      CudaCase{CudaFailureSite::module_load, 0, true},
      CudaCase{CudaFailureSite::module_get_function, 0, true},
  };

  for (const CudaCase& fault : cuda_cases) {
    Fixture fixture;
    fixture.cuda.injected_cuda.site = fault.site;
    fixture.cuda.injected_cuda.calls_to_skip = fault.calls_to_skip;
    auto config = fixture.config();
    if (fault.needs_embedded_verifier) {
      config.verification_submit = nullptr;
    }
    NvcompLz4Pipeline pipeline(fixture.cuda_api, fixture.nvcomp_api, fixture.codec, config);
    CHECK(pipeline.setup() == NvcompPipelineStatus::cuda_failure);
    CHECK(fixture.cuda.injected_cuda.triggered);
    CHECK(!pipeline.is_setup());
    CHECK(!pipeline.quarantined());
    CHECK(pipeline.telemetry().cuda_failures == 1);
    CHECK(fixture.cuda.device_allocations.empty());
    CHECK(fixture.cuda.host_allocations.empty());
    CHECK(fixture.cuda.streams.empty());
    CHECK(fixture.cuda.events.empty());
    CHECK(!fixture.cuda.module_loaded);
    CHECK(pipeline.close() == NvcompPipelineStatus::success);
  }

  constexpr std::array codec_cases{
      NvcompFailureSite::compress_alignment, NvcompFailureSite::decompress_alignment,
      NvcompFailureSite::compress_temp_size, NvcompFailureSite::decompress_temp_size,
      NvcompFailureSite::max_output_size,
  };
  for (const NvcompFailureSite site : codec_cases) {
    Fixture fixture;
    fixture.cuda.injected_nvcomp.site = site;
    NvcompLz4Pipeline pipeline(fixture.cuda_api, fixture.nvcomp_api, fixture.codec,
                               fixture.config());
    CHECK(pipeline.setup() == NvcompPipelineStatus::codec_failure);
    CHECK(fixture.cuda.injected_nvcomp.triggered);
    CHECK(!pipeline.is_setup());
    CHECK(!pipeline.quarantined());
    CHECK(fixture.cuda.device_allocations.empty());
    CHECK(fixture.cuda.host_allocations.empty());
    CHECK(fixture.cuda.streams.empty());
    CHECK(fixture.cuda.events.empty());
    CHECK(pipeline.close() == NvcompPipelineStatus::success);
  }
}

void decode_submission_fault_boundaries_test() {
  struct CudaCase {
    CudaFailureSite site;
    std::uint32_t calls_to_skip;
    bool embedded_verifier;
    NvcompPipelineStatus expected;
    bool quarantined;
  };
  constexpr std::array cases{
      CudaCase{CudaFailureSite::h2d, 0, false, NvcompPipelineStatus::cuda_failure, false},
      CudaCase{CudaFailureSite::h2d, 1, false, NvcompPipelineStatus::cuda_failure, false},
      CudaCase{CudaFailureSite::d2h, 0, false, NvcompPipelineStatus::quarantined, true},
      CudaCase{CudaFailureSite::event_record, 0, false, NvcompPipelineStatus::quarantined, true},
      CudaCase{CudaFailureSite::launch_kernel, 0, true, NvcompPipelineStatus::quarantined, true},
  };

  for (const CudaCase& fault : cases) {
    Fixture fixture;
    auto config = fixture.config();
    if (fault.embedded_verifier) {
      config.verification_submit = nullptr;
    }
    NvcompLz4Pipeline pipeline(fixture.cuda_api, fixture.nvcomp_api, fixture.codec, config);
    CHECK(pipeline.setup() == NvcompPipelineStatus::success);
    fixture.cuda.injected_cuda.site = fault.site;
    fixture.cuda.injected_cuda.calls_to_skip = fault.calls_to_skip;
    const auto bytes = source_bytes();
    const auto source = source_container(bytes);
    std::vector<std::byte> output(bytes.size());
    NvcompPipelineTicket ticket;
    const NvcompPipelineStatus status = pipeline.decode(
        {&source, static_cast<DevicePointer>(reinterpret_cast<std::uintptr_t>(output.data()))},
        ticket);
    CHECK(status == fault.expected);
    CHECK(fixture.cuda.injected_cuda.triggered);
    CHECK(pipeline.quarantined() == fault.quarantined);
    CHECK(pipeline.telemetry().cuda_failures >= 1);
    CHECK(pipeline.close() ==
          (fault.quarantined ? NvcompPipelineStatus::quarantined : NvcompPipelineStatus::success));
  }

  {
    Fixture fixture;
    NvcompLz4Pipeline pipeline(fixture.cuda_api, fixture.nvcomp_api, fixture.codec,
                               fixture.config());
    CHECK(pipeline.setup() == NvcompPipelineStatus::success);
    fixture.cuda.injected_nvcomp.site = NvcompFailureSite::decompress_launch;
    const auto bytes = source_bytes();
    const auto source = source_container(bytes);
    std::vector<std::byte> output(bytes.size());
    NvcompPipelineTicket ticket;
    CHECK(pipeline.decode({&source, static_cast<DevicePointer>(
                                        reinterpret_cast<std::uintptr_t>(output.data()))},
                          ticket) == NvcompPipelineStatus::codec_failure);
    CHECK(fixture.cuda.injected_nvcomp.triggered);
    CHECK(!pipeline.quarantined());
    CHECK(pipeline.telemetry().codec_failures == 1);
    CHECK(pipeline.close() == NvcompPipelineStatus::success);
  }

  {
    Fixture fixture;
    NvcompLz4Pipeline pipeline(fixture.cuda_api, fixture.nvcomp_api, fixture.codec,
                               fixture.config());
    CHECK(pipeline.setup() == NvcompPipelineStatus::success);
    fixture.cuda.injected_nvcomp.site = NvcompFailureSite::decompress_launch;
    fixture.cuda.forced_stream_synchronize = CUDA_ERROR_INVALID_CONTEXT;
    const auto bytes = source_bytes();
    const auto source = source_container(bytes);
    std::vector<std::byte> output(bytes.size());
    NvcompPipelineTicket ticket;
    CHECK(pipeline.decode({&source, static_cast<DevicePointer>(
                                        reinterpret_cast<std::uintptr_t>(output.data()))},
                          ticket) == NvcompPipelineStatus::quarantined);
    CHECK(fixture.cuda.injected_nvcomp.triggered);
    CHECK(pipeline.quarantined());
    CHECK(pipeline.close() == NvcompPipelineStatus::quarantined);
  }

  {
    Fixture fixture;
    fixture.cuda.corrupt_decompress_status = true;
    NvcompLz4Pipeline pipeline(fixture.cuda_api, fixture.nvcomp_api, fixture.codec,
                               fixture.config());
    CHECK(pipeline.setup() == NvcompPipelineStatus::success);
    const auto bytes = source_bytes();
    const auto source = source_container(bytes);
    std::vector<std::byte> output(bytes.size());
    NvcompPipelineTicket ticket;
    CHECK(pipeline.decode({&source, static_cast<DevicePointer>(
                                        reinterpret_cast<std::uintptr_t>(output.data()))},
                          ticket) == NvcompPipelineStatus::success);
    CHECK(pipeline.wait(ticket) == NvcompPipelineStatus::corrupt_data);
    CHECK(!pipeline.quarantined());
    CHECK(pipeline.telemetry().codec_failures == 1);
    CHECK(pipeline.close() == NvcompPipelineStatus::success);
  }
}

void encode_submission_fault_boundaries_test() {
  struct CudaCase {
    CudaFailureSite site;
    std::uint32_t calls_to_skip;
    bool embedded_verifier;
    NvcompPipelineStatus expected;
    bool quarantined;
  };
  constexpr std::array cases{
      CudaCase{CudaFailureSite::h2d, 0, false, NvcompPipelineStatus::cuda_failure, false},
      CudaCase{CudaFailureSite::d2h, 0, false, NvcompPipelineStatus::quarantined, true},
      CudaCase{CudaFailureSite::event_record, 0, false, NvcompPipelineStatus::quarantined, true},
      CudaCase{CudaFailureSite::launch_kernel, 0, true, NvcompPipelineStatus::quarantined, true},
  };

  for (const CudaCase& fault : cases) {
    Fixture fixture;
    auto config = fixture.config();
    if (fault.embedded_verifier) {
      config.verification_submit = nullptr;
    }
    NvcompLz4Pipeline pipeline(fixture.cuda_api, fixture.nvcomp_api, fixture.codec, config);
    CHECK(pipeline.setup() == NvcompPipelineStatus::success);
    fixture.cuda.injected_cuda.site = fault.site;
    fixture.cuda.injected_cuda.calls_to_skip = fault.calls_to_skip;
    auto input = source_bytes();
    NvcompPipelineTicket ticket;
    const NvcompPipelineStatus status =
        pipeline.encode({static_cast<DevicePointer>(reinterpret_cast<std::uintptr_t>(input.data())),
                         input.size(), 9, compute_content_token(input)},
                        ticket);
    CHECK(status == fault.expected);
    CHECK(fixture.cuda.injected_cuda.triggered);
    CHECK(pipeline.quarantined() == fault.quarantined);
    CHECK(pipeline.telemetry().cuda_failures >= 1);
    CHECK(pipeline.close() ==
          (fault.quarantined ? NvcompPipelineStatus::quarantined : NvcompPipelineStatus::success));
  }

  {
    Fixture fixture;
    NvcompLz4Pipeline pipeline(fixture.cuda_api, fixture.nvcomp_api, fixture.codec,
                               fixture.config());
    CHECK(pipeline.setup() == NvcompPipelineStatus::success);
    fixture.cuda.injected_nvcomp.site = NvcompFailureSite::compress_launch;
    auto input = source_bytes();
    NvcompPipelineTicket ticket;
    CHECK(
        pipeline.encode({static_cast<DevicePointer>(reinterpret_cast<std::uintptr_t>(input.data())),
                         input.size(), 10, compute_content_token(input)},
                        ticket) == NvcompPipelineStatus::codec_failure);
    CHECK(fixture.cuda.injected_nvcomp.triggered);
    CHECK(!pipeline.quarantined());
    CHECK(pipeline.telemetry().codec_failures == 1);
    CHECK(pipeline.close() == NvcompPipelineStatus::success);
  }

  {
    Fixture fixture;
    NvcompLz4Pipeline pipeline(fixture.cuda_api, fixture.nvcomp_api, fixture.codec,
                               fixture.config());
    CHECK(pipeline.setup() == NvcompPipelineStatus::success);
    fixture.cuda.injected_cuda.site = CudaFailureSite::d2h;
    fixture.cuda.injected_cuda.calls_to_skip = 3;
    auto input = source_bytes();
    NvcompPipelineTicket ticket;
    CHECK(
        pipeline.encode({static_cast<DevicePointer>(reinterpret_cast<std::uintptr_t>(input.data())),
                         input.size(), 11, compute_content_token(input)},
                        ticket) == NvcompPipelineStatus::success);
    CHECK(pipeline.wait(ticket) == NvcompPipelineStatus::cuda_failure);
    CHECK(fixture.cuda.injected_cuda.triggered);
    CHECK(!pipeline.quarantined());
    CHECK(pipeline.close() == NvcompPipelineStatus::success);
  }

  {
    Fixture fixture;
    NvcompLz4Pipeline pipeline(fixture.cuda_api, fixture.nvcomp_api, fixture.codec,
                               fixture.config());
    CHECK(pipeline.setup() == NvcompPipelineStatus::success);
    fixture.cuda.injected_cuda.site = CudaFailureSite::event_record;
    fixture.cuda.injected_cuda.calls_to_skip = 1;
    auto input = source_bytes();
    NvcompPipelineTicket ticket;
    CHECK(
        pipeline.encode({static_cast<DevicePointer>(reinterpret_cast<std::uintptr_t>(input.data())),
                         input.size(), 12, compute_content_token(input)},
                        ticket) == NvcompPipelineStatus::success);
    CHECK(pipeline.wait(ticket) == NvcompPipelineStatus::quarantined);
    CHECK(fixture.cuda.injected_cuda.triggered);
    CHECK(pipeline.quarantined());
    CHECK(pipeline.close() == NvcompPipelineStatus::quarantined);
  }

  {
    Fixture fixture;
    fixture.cuda.corrupt_compress_status = true;
    NvcompLz4Pipeline pipeline(fixture.cuda_api, fixture.nvcomp_api, fixture.codec,
                               fixture.config());
    CHECK(pipeline.setup() == NvcompPipelineStatus::success);
    auto input = source_bytes();
    NvcompPipelineTicket ticket;
    CHECK(
        pipeline.encode({static_cast<DevicePointer>(reinterpret_cast<std::uintptr_t>(input.data())),
                         input.size(), 13, compute_content_token(input)},
                        ticket) == NvcompPipelineStatus::success);
    Lz4BlocksV1 candidate;
    CHECK(pipeline.wait(ticket, &candidate) == NvcompPipelineStatus::codec_failure);
    CHECK(!pipeline.quarantined());
    CHECK(pipeline.telemetry().codec_failures == 1);
    CHECK(pipeline.close() == NvcompPipelineStatus::success);
  }
}

void close_active_generation_sync_failure_quarantines_test() {
  Fixture fixture;
  NvcompLz4Pipeline pipeline(fixture.cuda_api, fixture.nvcomp_api, fixture.codec, fixture.config());
  CHECK(pipeline.setup() == NvcompPipelineStatus::success);
  const auto bytes = source_bytes();
  const auto source = source_container(bytes);
  std::vector<std::byte> output(bytes.size());
  NvcompPipelineTicket ticket;
  CHECK(pipeline.decode(
            {&source, static_cast<DevicePointer>(reinterpret_cast<std::uintptr_t>(output.data()))},
            ticket) == NvcompPipelineStatus::success);
  fixture.cuda.injected_cuda.site = CudaFailureSite::event_synchronize;
  CHECK(pipeline.close() == NvcompPipelineStatus::quarantined);
  CHECK(fixture.cuda.injected_cuda.triggered);
  CHECK(pipeline.quarantined());
  CHECK(pipeline.telemetry().cuda_failures >= 1);
}

void configuration_and_status_name_test() {
  Fixture fixture;
  auto config = fixture.config();
  config.slot_count = 1;
  NvcompLz4Pipeline pipeline(fixture.cuda_api, fixture.nvcomp_api, fixture.codec, config);
  CHECK(pipeline.setup() == NvcompPipelineStatus::invalid_configuration);
  CHECK(std::string_view{nvcomp_pipeline_status_name(NvcompPipelineStatus::corrupt_data)} ==
        "corrupt_data");
}

} // namespace

int main() {
  decode_and_event_boundary_test();
  compressed_decode_payload_is_submitted_as_one_batch_test();
  encode_fallback_and_generation_test();
  trace_slot_generation_reuse_test();
  slot_aba_and_quarantine_test();
  polling_timeout_quarantines_test();
  encode_source_token_rejects_corrupt_candidate_test();
  uncommitted_encode_quarantines_test();
  setup_cleanup_failure_retains_resources_for_retry_test();
  setup_module_cleanup_failure_is_retryable_test();
  close_cleanup_failure_preserves_exact_owners_for_retry_test();
  setup_fault_injection_matrix_test();
  decode_submission_fault_boundaries_test();
  encode_submission_fault_boundaries_test();
  close_active_generation_sync_failure_quarantines_test();
  configuration_and_status_name_test();
  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
    return 1;
  }
  std::cout << "nvCOMP pipeline tests passed\n";
  return 0;
}
