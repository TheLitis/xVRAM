#include "gemm_bench/executor.hpp"

#include "gemm_bench/pattern.hpp"
#include "gemm_bench/sizing.hpp"

#include "platform/cuda/cuda_api.hpp"
#include "platform/system_info.hpp"
#include "xvram/version.hpp"
#include "xvram/xvram.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace xvram::gemm_bench {
namespace {

constexpr int exit_completed = 0;
constexpr int exit_prerequisite = 23;
constexpr int exit_corruption = 24;
constexpr int exit_oom = 25;
constexpr int exit_timeout = 26;
constexpr int exit_failure = 27;
constexpr std::uint64_t io_batch_bytes = 16ULL * 1024ULL * 1024ULL;
using Clock = std::chrono::steady_clock;

struct Digest128 {
  std::uint64_t low = 1469598103934665603ULL;
  std::uint64_t high = 1099511628211ULL ^ 0x9E3779B97F4A7C15ULL;

  void update(const std::span<const std::byte> bytes,
              const std::uint64_t absolute_offset) noexcept {
    std::uint64_t index = absolute_offset;
    for (const std::byte byte : bytes) {
      const std::uint64_t value = static_cast<std::uint64_t>(std::to_integer<unsigned char>(byte));
      low ^= value + (index & 0xFFULL);
      low *= 1099511628211ULL;
      high ^= value + 0x9EULL + (index >> 8U);
      high *= 0x100000001B3ULL;
      high ^= high >> 29U;
      ++index;
    }
  }

  [[nodiscard]] std::string format() const {
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(16) << high << std::setw(16) << low;
    return output.str();
  }
};

struct TimingSamples {
  std::vector<double> values;

  void add(const double milliseconds) {
    if (std::isfinite(milliseconds) && milliseconds >= 0.0) {
      values.push_back(milliseconds);
    }
  }

  [[nodiscard]] TimingSummary summarize() const {
    TimingSummary output;
    output.sample_count = values.size();
    if (values.empty()) {
      return output;
    }
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const auto percentile = [&](const double fraction) {
      const std::size_t index =
          static_cast<std::size_t>(std::ceil(fraction * static_cast<double>(sorted.size())) - 1.0);
      return sorted[std::min(index, sorted.size() - 1U)];
    };
    double total = 0.0;
    for (const double value : sorted) {
      total += value;
    }
    output.total_ms = total;
    output.minimum_ms = sorted.front();
    output.median_ms = percentile(0.50);
    output.p95_ms = percentile(0.95);
    output.maximum_ms = sorted.back();
    return output;
  }
};

struct Shape {
  std::uint64_t m = 0;
  std::uint64_t n = 0;
  std::uint64_t k = 0;
};

struct MatrixStorage {
  std::uint64_t rows = 0;
  std::uint64_t columns = 0;
  std::uint64_t leading_dimension = 0;
  std::uint64_t bytes = 0;
};

[[nodiscard]] std::string utc_timestamp() {
  const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::tm utc{};
#ifdef _WIN32
  gmtime_s(&utc, &now);
#else
  gmtime_r(&now, &utc);
#endif
  std::ostringstream output;
  output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
  return output.str();
}

[[nodiscard]] std::string compiler_description() {
#if defined(_MSC_VER)
  return "MSVC " + std::to_string(_MSC_VER);
#elif defined(__clang__)
  return "Clang " + std::to_string(__clang_major__) + "." + std::to_string(__clang_minor__) + "." +
         std::to_string(__clang_patchlevel__);
#elif defined(__GNUC__)
  return "GCC " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__) + "." +
         std::to_string(__GNUC_PATCHLEVEL__);
#else
  return "unknown";
#endif
}

[[nodiscard]] std::string build_type() {
#ifdef NDEBUG
  return "Release";
#else
  return "Debug";
#endif
}

[[nodiscard]] xvram_data_type public_type(const RequestedDataType type) noexcept {
  switch (type) {
  case RequestedDataType::fp16:
    return XVRAM_DATA_FP16;
  case RequestedDataType::bf16:
    return XVRAM_DATA_BF16;
  case RequestedDataType::fp64:
    return XVRAM_DATA_FP64;
  case RequestedDataType::fp32:
  case RequestedDataType::suite:
  default:
    return XVRAM_DATA_FP32;
  }
}

[[nodiscard]] xvram_compute_mode public_compute(const ExecutorOptions& options,
                                                const RequestedDataType type) noexcept {
  if (type == RequestedDataType::fp64 || options.compute_mode == RequestedComputeMode::fp64) {
    return XVRAM_COMPUTE_FP64;
  }
  if (options.compute_mode == RequestedComputeMode::strict_fp32) {
    return XVRAM_COMPUTE_FP32_STRICT;
  }
  if (options.compute_mode == RequestedComputeMode::tf32) {
    return XVRAM_COMPUTE_FP32_TF32;
  }
  return XVRAM_COMPUTE_AUTO;
}

[[nodiscard]] std::optional<std::uint64_t> checked_multiply(const std::uint64_t left,
                                                            const std::uint64_t right) noexcept {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
    return std::nullopt;
  }
  return left * right;
}

[[nodiscard]] std::optional<std::uint64_t> checked_add(const std::uint64_t left,
                                                       const std::uint64_t right) noexcept {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    return std::nullopt;
  }
  return left + right;
}

[[nodiscard]] std::optional<std::uint64_t>
checked_least_common_multiple(const std::uint64_t left, const std::uint64_t right) noexcept {
  if (left == 0 || right == 0) {
    return std::nullopt;
  }
  return checked_multiply(left / std::gcd(left, right), right);
}

[[nodiscard]] std::optional<std::uint64_t>
checked_align_up(const std::uint64_t value, const std::uint64_t alignment) noexcept {
  if (alignment == 0) {
    return std::nullopt;
  }
  const std::uint64_t remainder = value % alignment;
  return remainder == 0 ? std::optional<std::uint64_t>{value}
                        : checked_add(value, alignment - remainder);
}

[[nodiscard]] std::optional<MatrixStorage> matrix_storage(const std::uint64_t rows,
                                                          const std::uint64_t columns,
                                                          const RequestedLayout layout,
                                                          const std::uint64_t item_bytes) noexcept {
  const std::uint64_t leading = layout == RequestedLayout::row_major ? columns : rows;
  const auto elements = checked_multiply(rows, columns);
  const auto bytes = elements.has_value() ? checked_multiply(*elements, item_bytes) : std::nullopt;
  if (!bytes.has_value()) {
    return std::nullopt;
  }
  return MatrixStorage{rows, columns, leading, *bytes};
}

[[nodiscard]] std::optional<Shape> choose_shape(const ExecutorOptions& options,
                                                const RequestedDataType type,
                                                const std::uint64_t total_vram,
                                                const std::uint64_t safe_host_limit) noexcept {
  if (options.m.has_value() || options.n.has_value() || options.k.has_value()) {
    if (!options.m.has_value() || !options.n.has_value() || !options.k.has_value() ||
        *options.m == 0 || *options.n == 0 || *options.k == 0) {
      return std::nullopt;
    }
    return Shape{*options.m, *options.n, *options.k};
  }
  const auto automatic =
      choose_auto_gemm_shape(total_vram, pattern_element_bytes(type), safe_host_limit);
  return automatic.has_value()
             ? std::optional<Shape>{Shape{automatic->m, automatic->n, automatic->k}}
             : std::nullopt;
}

[[nodiscard]] std::string bytes_as_uuid(const unsigned char* bytes) {
  static constexpr char hex[] = "0123456789abcdef";
  std::string output;
  output.reserve(36);
  for (std::size_t index = 0; index < 16U; ++index) {
    if (index == 4U || index == 6U || index == 8U || index == 10U) {
      output.push_back('-');
    }
    output.push_back(hex[(bytes[index] >> 4U) & 0x0FU]);
    output.push_back(hex[bytes[index] & 0x0FU]);
  }
  return output;
}

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

void fail_report(Report& report, const int exit_code, std::string reason, std::string stage,
                 std::string operation, std::string message) {
  report.outcome.status = exit_code == exit_prerequisite ? "skipped" : "failed";
  report.outcome.exit_code = exit_code;
  report.outcome.reason = std::move(reason);
  report.outcome.stage = std::move(stage);
  report.outcome.operation = std::move(operation);
  report.outcome.message = std::move(message);
}

[[nodiscard]] int exit_code_for_status(const xvram_status status) noexcept {
  switch (status) {
  case XVRAM_STATUS_SUCCESS:
    return exit_completed;
  case XVRAM_STATUS_INVALID_ARGUMENT:
  case XVRAM_STATUS_INCOMPATIBLE_ABI:
  case XVRAM_STATUS_UNSUPPORTED:
  case XVRAM_STATUS_UNAVAILABLE:
    return exit_prerequisite;
  case XVRAM_STATUS_HOST_OUT_OF_MEMORY:
  case XVRAM_STATUS_DEVICE_OUT_OF_MEMORY:
  case XVRAM_STATUS_BUDGET_PRESSURE:
    return exit_oom;
  case XVRAM_STATUS_TIMEOUT:
    return exit_timeout;
  case XVRAM_STATUS_CORRUPTION:
    return exit_corruption;
  default:
    return exit_failure;
  }
}

[[nodiscard]] std::string error_message(const xvram_api_v1& api, const xvram_session session,
                                        const xvram_status status) {
  xvram_error_info_v1 info = XVRAM_ERROR_INFO_V1_INIT;
  if (session != nullptr && api.session_get_error(session, &info) == XVRAM_STATUS_SUCCESS &&
      info.message[0] != '\0') {
    return info.message;
  }
  return api.status_name(status);
}

[[nodiscard]] bool write_pattern_allocation(
    const xvram_api_v1& api, const xvram_allocation allocation, const std::uint64_t bytes,
    const PatternMatrix& matrix, const PatternOperand operand, const PatternProblem& problem,
    std::vector<std::byte>& buffer, const std::function<void(std::uint64_t)>& progress) {
  std::uint64_t offset = 0;
  while (offset < bytes) {
    const std::uint64_t batch = std::min<std::uint64_t>(buffer.size(), bytes - offset);
    const std::span<std::byte> batch_buffer(buffer.data(), static_cast<std::size_t>(batch));
    if (!fill_pattern_bytes(batch_buffer, offset, matrix, operand, problem)) {
      return false;
    }
    if (api.allocation_write(allocation, offset, buffer.data(), batch) != XVRAM_STATUS_SUCCESS) {
      return false;
    }
    offset += batch;
    if (progress) {
      progress(offset);
    }
  }
  return true;
}

[[nodiscard]] xvram_matrix_v1 matrix_desc(const xvram_allocation allocation,
                                          const MatrixStorage& storage,
                                          const RequestedDataType type,
                                          const RequestedLayout layout) {
  xvram_matrix_v1 matrix = XVRAM_MATRIX_V1_INIT;
  matrix.data_type = public_type(type);
  matrix.allocation = allocation;
  matrix.rows = storage.rows;
  matrix.columns = storage.columns;
  matrix.leading_dimension = storage.leading_dimension;
  matrix.layout =
      layout == RequestedLayout::row_major ? XVRAM_MATRIX_ROW_MAJOR : XVRAM_MATRIX_COLUMN_MAJOR;
  return matrix;
}

} // namespace

const char* requested_data_type_name(const RequestedDataType value) noexcept {
  switch (value) {
  case RequestedDataType::suite:
    return "suite";
  case RequestedDataType::fp16:
    return "fp16";
  case RequestedDataType::bf16:
    return "bf16";
  case RequestedDataType::fp32:
    return "fp32";
  case RequestedDataType::fp64:
    return "fp64";
  }
  return "invalid";
}

const char* requested_compute_mode_name(const RequestedComputeMode value) noexcept {
  switch (value) {
  case RequestedComputeMode::automatic:
    return "auto";
  case RequestedComputeMode::strict_fp32:
    return "fp32_strict";
  case RequestedComputeMode::tf32:
    return "tf32";
  case RequestedComputeMode::fp64:
    return "fp64";
  }
  return "invalid";
}

const char* requested_layout_name(const RequestedLayout value) noexcept {
  return value == RequestedLayout::column_major ? "column_major" : "row_major";
}

const char* requested_operation_name(const RequestedOperation value) noexcept {
  return value == RequestedOperation::transpose ? "t" : "n";
}

Report run_executor(const ExecutorOptions& options, const ProgressCallback& progress) {
  const Clock::time_point executor_started = Clock::now();
  Report report;
  report.generated_at_utc = utc_timestamp();
  report.build = {XVRAM_VERSION, XVRAM_GIT_COMMIT, compiler_description(), build_type(),
                  XVRAM_CUDA_HEADERS_VERSION};
  report.system = platform::collect_system_info();
  report.configuration.requested_device_ordinal = options.device_ordinal;
  report.configuration.requested_m = options.m;
  report.configuration.requested_n = options.n;
  report.configuration.requested_k = options.k;
  report.configuration.scenario = options.data_type == RequestedDataType::suite ? "suite" : "gemm";
  const RequestedDataType configuration_type =
      options.data_type == RequestedDataType::suite ? RequestedDataType::fp32 : options.data_type;
  report.configuration.a_data_type = requested_data_type_name(configuration_type);
  report.configuration.b_data_type = requested_data_type_name(configuration_type);
  report.configuration.c_data_type = requested_data_type_name(configuration_type);
  report.configuration.compute_mode = requested_compute_mode_name(options.compute_mode);
  report.configuration.a_layout = requested_layout_name(options.a_layout);
  report.configuration.b_layout = requested_layout_name(options.b_layout);
  report.configuration.c_layout = requested_layout_name(options.c_layout);
  report.configuration.a_operation = requested_operation_name(options.a_operation);
  report.configuration.b_operation = requested_operation_name(options.b_operation);
  report.configuration.requested_chunk_bytes = options.chunk_bytes;
  report.configuration.requested_cache_target_bytes = options.cache_target_bytes;
  report.configuration.requested_workspace_cap_bytes = options.workspace_bytes;
  report.configuration.staging_slots = options.staging_slots;
  report.configuration.passes = options.passes;
  report.configuration.timeout_ms = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(options.timeout).count());
  report.configuration.identifiers_included = options.include_identifiers;
  fail_report(report, exit_prerequisite, "device_unavailable", "preflight", "cuInit",
              "CUDA device preflight did not complete");

  cuda::CudaApi cuda;
  if (cuda.load().status != cuda::CudaApi::LoadStatus::loaded || cuda.init_ == nullptr ||
      cuda.init_(0) != cuda::abi::success) {
    return report;
  }
  cuda::abi::Device device = 0;
  if (cuda.device_get_ == nullptr || cuda.device_total_memory_ == nullptr ||
      cuda.device_get_(&device, options.device_ordinal) != cuda::abi::success) {
    return report;
  }
  std::size_t total_memory = 0;
  if (cuda.device_total_memory_(&total_memory, device) != cuda::abi::success) {
    return report;
  }
  DeviceInfo device_info;
  device_info.ordinal = options.device_ordinal;
  device_info.total_memory_bytes = static_cast<std::uint64_t>(total_memory);
  if (cuda.device_get_name_ != nullptr) {
    std::array<char, 256> name{};
    if (cuda.device_get_name_(name.data(), static_cast<int>(name.size()), device) ==
        cuda::abi::success) {
      device_info.name = name.data();
    }
  }
  int vmm = 0;
  int uva = 0;
  int compute_major = 0;
  int compute_minor = 0;
  if (cuda.device_get_attribute_ != nullptr) {
    (void)cuda.device_get_attribute_(
        &vmm, cuda::abi::attributes::virtual_memory_management_supported, device);
    (void)cuda.device_get_attribute_(&uva, cuda::abi::attributes::unified_addressing, device);
    if (cuda.device_get_attribute_(&compute_major, cuda::abi::attributes::compute_capability_major,
                                   device) == cuda::abi::success) {
      device_info.compute_capability_major = static_cast<std::uint32_t>(compute_major);
    }
    if (cuda.device_get_attribute_(&compute_minor, cuda::abi::attributes::compute_capability_minor,
                                   device) == cuda::abi::success) {
      device_info.compute_capability_minor = static_cast<std::uint32_t>(compute_minor);
    }
  }
  device_info.vmm_supported = vmm != 0;
  device_info.uva_supported = uva != 0;
  if (options.include_identifiers && cuda.device_get_uuid_ != nullptr) {
    cuda::abi::Uuid uuid{};
    if (cuda.device_get_uuid_(&uuid, device) == cuda::abi::success) {
      device_info.uuid = bytes_as_uuid(reinterpret_cast<const unsigned char*>(&uuid));
    }
  }
  if (options.include_identifiers && cuda.device_get_luid_ != nullptr) {
    std::array<char, 8> luid{};
    unsigned int node_mask = 0;
    if (cuda.device_get_luid_(luid.data(), &node_mask, device) == cuda::abi::success) {
      device_info.luid =
          bytes_as_hex(reinterpret_cast<const unsigned char*>(luid.data()), luid.size());
    }
  }
  if (options.include_identifiers && cuda.device_get_pci_bus_id_ != nullptr) {
    std::array<char, 64> pci{};
    if (cuda.device_get_pci_bus_id_(pci.data(), static_cast<int>(pci.size()), device) ==
        cuda::abi::success) {
      device_info.pci_bus_id = pci.data();
    }
  }
  if (cuda.mem_get_allocation_granularity_ != nullptr) {
    cuda::abi::MemAllocationProp property{};
    property.type = cuda::abi::allocation_type_pinned;
    property.location = cuda::abi::MemLocation{cuda::abi::location_device, device};
    std::size_t minimum = 0;
    std::size_t recommended = 0;
    if (cuda.mem_get_allocation_granularity_(&minimum, &property, cuda::abi::granularity_minimum) ==
        cuda::abi::success) {
      device_info.minimum_granularity_bytes = static_cast<std::uint64_t>(minimum);
    }
    if (cuda.mem_get_allocation_granularity_(
            &recommended, &property, cuda::abi::granularity_recommended) == cuda::abi::success) {
      device_info.recommended_granularity_bytes = static_cast<std::uint64_t>(recommended);
    }
  }
  report.device = device_info;
  if (!device_info.vmm_supported || !device_info.uva_supported) {
    fail_report(report, exit_prerequisite,
                !device_info.vmm_supported ? "vmm_unsupported" : "uva_unsupported", "preflight",
                "device_attributes", "selected device lacks VMM or unified addressing");
    return report;
  }

  const std::uint64_t minimum_granularity = device_info.minimum_granularity_bytes.value_or(0);
  const std::uint64_t recommended_granularity =
      device_info.recommended_granularity_bytes.value_or(minimum_granularity);
  const auto common_granularity =
      checked_least_common_multiple(minimum_granularity, recommended_granularity);
  const auto effective_chunk_bytes =
      common_granularity.has_value() ? checked_align_up(options.chunk_bytes, *common_granularity)
                                     : std::nullopt;
  if (!effective_chunk_bytes.has_value() || *effective_chunk_bytes == 0) {
    fail_report(report, exit_prerequisite, "invalid_granularity", "preflight",
                "allocation_granularity",
                "CUDA VMM granularities cannot represent the requested chunk size");
    return report;
  }

  const auto safe_host_limit =
      safe_host_logical_limit(HostSizingInput{report.system.physical_memory_bytes.value_or(0),
                                              report.system.available_memory_bytes.value_or(0),
                                              *effective_chunk_bytes, options.staging_slots});
  if (!safe_host_limit.has_value()) {
    fail_report(report, exit_oom, "insufficient_host_memory", "planning", "host_memory",
                "safe pageable GEMM backing limit is unavailable");
    return report;
  }

  xvram_api_v1 api{};
  if (xvram_get_api(XVRAM_ABI_VERSION_1, sizeof(api), &api) != XVRAM_STATUS_SUCCESS) {
    fail_report(report, exit_failure, "internal_error", "sdk", "xvram_get_api",
                "public SDK ABI v1 is unavailable");
    return report;
  }
  xvram_session_config_v1 session_config = XVRAM_SESSION_CONFIG_V1_INIT;
  session_config.device_ordinal = options.device_ordinal;
  session_config.chunk_size_bytes = *effective_chunk_bytes;
  session_config.cache_target_bytes = options.cache_target_bytes.value_or(0);
  session_config.device_headroom_bytes = options.device_headroom_bytes;
  session_config.workspace_cap_bytes = options.workspace_bytes;
  session_config.staging_slots = options.staging_slots;
  xvram_session session = nullptr;
  const xvram_status create_status = api.session_create(&session_config, &session);
  if (create_status != XVRAM_STATUS_SUCCESS) {
    fail_report(report, exit_code_for_status(create_status),
                create_status == XVRAM_STATUS_UNAVAILABLE ? "cuda_error" : "invalid_configuration",
                "sdk", "session_create", error_message(api, session, create_status));
    return report;
  }
  xvram_session_telemetry_v1 initial_telemetry{};
  initial_telemetry.struct_size = sizeof(initial_telemetry);
  if (api.session_get_telemetry(session, &initial_telemetry) == XVRAM_STATUS_SUCCESS) {
    report.device->free_memory_bytes_start = initial_telemetry.cuda_free_bytes_end;
    report.device->safe_device_budget_bytes = initial_telemetry.cache_target_bytes;
    if (initial_telemetry.wddm_budget_observed != 0U) {
      report.device->wddm_available_bytes_start = initial_telemetry.wddm_available_bytes_end;
    }
  }

  std::vector<RequestedDataType> data_types;
  if (options.data_type == RequestedDataType::suite) {
    data_types = {RequestedDataType::fp16, RequestedDataType::bf16, RequestedDataType::fp32,
                  RequestedDataType::fp64};
  } else {
    data_types.push_back(options.data_type);
  }
  const std::uint64_t total_cases = static_cast<std::uint64_t>(data_types.size());
  std::uint64_t completed_cases = 0;
  bool all_match = true;
  std::uint64_t aggregate_logical_bytes = 0;
  std::uint64_t verified_elements = 0;
  std::uint64_t mismatches = 0;
  double maximum_absolute_error = 0.0;
  double maximum_relative_error = 0.0;
  double maximum_absolute_tolerance = 0.0;
  std::optional<std::uint64_t> first_mismatch;
  Digest128 actual_digest;
  Digest128 expected_digest;
  TimingSamples plan_timings;
  TimingSamples gemm_timings;

  for (const RequestedDataType type : data_types) {
    bool case_match = true;
    Digest128 case_actual_digest;
    Digest128 case_expected_digest;
    const std::optional<Shape> selected =
        choose_shape(options, type, device_info.total_memory_bytes, *safe_host_limit);
    if (!selected.has_value()) {
      fail_report(report, exit_prerequisite, "invalid_configuration", "planning", "shape",
                  "M/N/K dimensions are incomplete or overflowed");
      all_match = false;
      break;
    }
    const Shape shape = *selected;
    const std::uint64_t item_bytes = pattern_element_bytes(type);
    const std::uint64_t a_rows =
        options.a_operation == RequestedOperation::none ? shape.m : shape.k;
    const std::uint64_t a_columns =
        options.a_operation == RequestedOperation::none ? shape.k : shape.m;
    const std::uint64_t b_rows =
        options.b_operation == RequestedOperation::none ? shape.k : shape.n;
    const std::uint64_t b_columns =
        options.b_operation == RequestedOperation::none ? shape.n : shape.k;
    const auto a_storage = matrix_storage(a_rows, a_columns, options.a_layout, item_bytes);
    const auto b_storage = matrix_storage(b_rows, b_columns, options.b_layout, item_bytes);
    const auto c_storage = matrix_storage(shape.m, shape.n, options.c_layout, item_bytes);
    const auto ab_bytes = a_storage.has_value() && b_storage.has_value()
                              ? checked_add(a_storage->bytes, b_storage->bytes)
                              : std::nullopt;
    const auto logical_bytes = ab_bytes.has_value() && c_storage.has_value()
                                   ? checked_add(*ab_bytes, c_storage->bytes)
                                   : std::nullopt;
    const auto verification_stream_bytes =
        c_storage.has_value()
            ? checked_multiply(c_storage->bytes, static_cast<std::uint64_t>(options.passes))
            : std::nullopt;
    if (!a_storage.has_value() || !b_storage.has_value() || !c_storage.has_value() ||
        !logical_bytes.has_value() || !verification_stream_bytes.has_value()) {
      fail_report(report, exit_prerequisite, "invalid_configuration", "planning", "matrix_storage",
                  "matrix storage size overflowed");
      all_match = false;
      break;
    }
    if (*logical_bytes > *safe_host_limit) {
      fail_report(report, exit_oom, "insufficient_host_memory", "planning", "matrix_storage",
                  "GEMM matrices exceed the safe pageable RAM limit");
      all_match = false;
      break;
    }

    xvram_allocation_desc_v1 allocation_desc = XVRAM_ALLOCATION_DESC_V1_INIT;
    allocation_desc.flags = XVRAM_ALLOCATION_FLAG_ZERO_INITIALIZE;
    xvram_allocation a = nullptr;
    xvram_allocation b = nullptr;
    xvram_allocation c = nullptr;
    allocation_desc.size_bytes = a_storage->bytes;
    xvram_status status = api.allocation_create(session, &allocation_desc, &a);
    if (status == XVRAM_STATUS_SUCCESS && progress) {
      progress(report, 1, 3);
    }
    if (status == XVRAM_STATUS_SUCCESS) {
      allocation_desc.size_bytes = b_storage->bytes;
      status = api.allocation_create(session, &allocation_desc, &b);
      if (status == XVRAM_STATUS_SUCCESS && progress) {
        progress(report, 2, 3);
      }
    }
    if (status == XVRAM_STATUS_SUCCESS) {
      allocation_desc.size_bytes = c_storage->bytes;
      status = api.allocation_create(session, &allocation_desc, &c);
      if (status == XVRAM_STATUS_SUCCESS && progress) {
        progress(report, 3, 3);
      }
    }
    if (status != XVRAM_STATUS_SUCCESS) {
      fail_report(report, exit_code_for_status(status), "host_oom", "allocation",
                  "allocation_create", error_message(api, session, status));
      if (a != nullptr) {
        (void)api.allocation_release(a);
      }
      if (b != nullptr) {
        (void)api.allocation_release(b);
      }
      if (c != nullptr) {
        (void)api.allocation_release(c);
      }
      all_match = false;
      break;
    }

    std::vector<std::byte> io_buffer(static_cast<std::size_t>(
        std::min<std::uint64_t>(io_batch_bytes, std::max(a_storage->bytes, b_storage->bytes))));
    const PatternProblem pattern_problem{shape.m, shape.n, shape.k, options.a_operation,
                                         options.b_operation};
    const PatternMatrix a_pattern{a_storage->rows, a_storage->columns, a_storage->leading_dimension,
                                  options.a_layout, type};
    const PatternMatrix b_pattern{b_storage->rows, b_storage->columns, b_storage->leading_dimension,
                                  options.b_layout, type};
    const PatternMatrix c_pattern{c_storage->rows, c_storage->columns, c_storage->leading_dimension,
                                  options.c_layout, type};
    const auto a_initialization_progress = [&](const std::uint64_t bytes_completed) {
      if (progress) {
        progress(report, bytes_completed, *ab_bytes);
      }
    };
    const auto b_initialization_progress = [&](const std::uint64_t bytes_completed) {
      if (progress) {
        progress(report, a_storage->bytes + bytes_completed, *ab_bytes);
      }
    };
    if (!write_pattern_allocation(api, a, a_storage->bytes, a_pattern, PatternOperand::a,
                                  pattern_problem, io_buffer, a_initialization_progress) ||
        !write_pattern_allocation(api, b, b_storage->bytes, b_pattern, PatternOperand::b,
                                  pattern_problem, io_buffer, b_initialization_progress)) {
      fail_report(report, exit_failure, "platform_error", "initialization", "allocation_write",
                  "failed to initialize structured GEMM operands");
      (void)api.allocation_release(a);
      (void)api.allocation_release(b);
      (void)api.allocation_release(c);
      all_match = false;
      break;
    }

    xvram_gemm_desc_v1 gemm = XVRAM_GEMM_DESC_V1_INIT;
    gemm.a = matrix_desc(a, *a_storage, type, options.a_layout);
    gemm.b = matrix_desc(b, *b_storage, type, options.b_layout);
    gemm.c = matrix_desc(c, *c_storage, type, options.c_layout);
    gemm.operation_a =
        options.a_operation == RequestedOperation::none ? XVRAM_MATRIX_OP_N : XVRAM_MATRIX_OP_T;
    gemm.operation_b =
        options.b_operation == RequestedOperation::none ? XVRAM_MATRIX_OP_N : XVRAM_MATRIX_OP_T;
    gemm.compute_mode = public_compute(options, type);
    gemm.workspace_cap_bytes = options.workspace_bytes;
    xvram_gemm_plan plan = nullptr;
    const Clock::time_point plan_started = Clock::now();
    status = api.gemm_plan_create(session, &gemm, &plan);
    plan_timings.add(
        std::chrono::duration<double, std::milli>(Clock::now() - plan_started).count());
    if (status != XVRAM_STATUS_SUCCESS) {
      fail_report(report, exit_code_for_status(status), "working_set_too_large", "planning",
                  "gemm_plan_create", error_message(api, session, status));
      (void)api.allocation_release(a);
      (void)api.allocation_release(b);
      (void)api.allocation_release(c);
      all_match = false;
      break;
    }
    xvram_gemm_plan_info_v1 plan_info = XVRAM_GEMM_PLAN_INFO_V1_INIT;
    status = api.gemm_plan_get_info(plan, &plan_info);
    if (status != XVRAM_STATUS_SUCCESS) {
      fail_report(report, exit_failure, "internal_error", "planning", "gemm_plan_get_info",
                  error_message(api, session, status));
      api.gemm_plan_release(plan);
      (void)api.allocation_release(a);
      (void)api.allocation_release(b);
      (void)api.allocation_release(c);
      all_match = false;
      break;
    }
    PlanResult plan_result;
    plan_result.plan_id = report.plans.size() + 1ULL;
    plan_result.status = "completed";
    plan_result.m = shape.m;
    plan_result.n = shape.n;
    plan_result.k = shape.k;
    plan_result.a_data_type = requested_data_type_name(type);
    plan_result.b_data_type = requested_data_type_name(type);
    plan_result.c_data_type = requested_data_type_name(type);
    plan_result.effective_compute_mode =
        plan_info.effective_compute_mode == XVRAM_COMPUTE_FP64
            ? "fp64"
            : (plan_info.effective_compute_mode == XVRAM_COMPUTE_FP32_STRICT ? "fp32_strict"
                                                                             : "tf32");
    plan_result.tile_m = plan_info.tile_m;
    plan_result.tile_n = plan_info.tile_n;
    plan_result.tile_k = plan_info.tile_k;
    plan_result.tile_count = plan_info.tile_count;
    plan_result.maximum_working_set_bytes = plan_info.maximum_working_set_bytes;
    plan_result.workspace_bytes = plan_info.workspace_bytes;
    plan_result.uses_cublas_lt = plan_info.uses_cublas_lt != 0;
    report.plans.push_back(plan_result);

    WorkloadResult workload;
    workload.name = std::string("structured_") + requested_data_type_name(type);
    workload.plan_id = plan_result.plan_id;
    workload.status = "running";
    workload.tiles_total = plan_info.tile_count * options.passes;
    workload.logical_allocation_bytes = *logical_bytes;
    workload.oversubscribed = *logical_bytes > device_info.total_memory_bytes;
    const std::uint64_t verify_batch = std::min<std::uint64_t>(io_batch_bytes, c_storage->bytes);
    io_buffer.resize(static_cast<std::size_t>(verify_batch));
    std::vector<std::byte> expected_buffer(io_buffer.size());
    const double tolerance = type == RequestedDataType::fp16
                                 ? 0.5
                                 : (type == RequestedDataType::bf16
                                        ? 2.0
                                        : (type == RequestedDataType::fp32 ? 1.0e-4 : 1.0e-10));
    maximum_absolute_tolerance = std::max(maximum_absolute_tolerance, tolerance);
    bool reference_ok = true;
    xvram_status verification_status = XVRAM_STATUS_SUCCESS;
    const auto verify_output = [&](const std::uint32_t pass) {
      std::uint64_t offset = 0;
      const std::uint64_t pass_offset = static_cast<std::uint64_t>(pass) * c_storage->bytes;
      while (offset < c_storage->bytes) {
        const std::uint64_t batch =
            std::min<std::uint64_t>(io_buffer.size(), c_storage->bytes - offset);
        verification_status = api.allocation_read(c, offset, io_buffer.data(), batch);
        if (verification_status != XVRAM_STATUS_SUCCESS) {
          return false;
        }
        expected_buffer.resize(static_cast<std::size_t>(batch));
        if (!fill_pattern_bytes(expected_buffer, offset, c_pattern, PatternOperand::expected_c,
                                pattern_problem)) {
          reference_ok = false;
          return false;
        }
        const std::uint64_t stride = item_bytes;
        for (std::uint64_t byte = 0; byte < batch; byte += stride) {
          const auto actual = decode_pattern_value(
              std::span<const std::byte>(io_buffer.data() + byte, static_cast<std::size_t>(stride)),
              type);
          const auto expected =
              decode_pattern_value(std::span<const std::byte>(expected_buffer.data() + byte,
                                                              static_cast<std::size_t>(stride)),
                                   type);
          if (!actual.has_value() || !expected.has_value()) {
            reference_ok = false;
            return false;
          }
          const double absolute_error = std::abs(*actual - *expected);
          const double relative_error =
              *expected == 0.0 ? absolute_error : absolute_error / std::abs(*expected);
          maximum_absolute_error = std::max(maximum_absolute_error, absolute_error);
          maximum_relative_error = std::max(maximum_relative_error, relative_error);
          if (!std::isfinite(*actual) || absolute_error > tolerance) {
            ++mismatches;
            all_match = false;
            case_match = false;
            if (!first_mismatch.has_value()) {
              first_mismatch = offset + byte;
            }
          }
          ++verified_elements;
        }
        const std::span<const std::byte> actual_bytes(io_buffer.data(),
                                                      static_cast<std::size_t>(batch));
        actual_digest.update(actual_bytes, pass_offset + offset);
        case_actual_digest.update(actual_bytes, pass_offset + offset);
        expected_digest.update(expected_buffer, pass_offset + offset);
        case_expected_digest.update(expected_buffer, pass_offset + offset);
        offset += batch;
        if (progress) {
          progress(report, pass_offset + offset, *verification_stream_bytes);
        }
      }
      return true;
    };
    xvram_session_telemetry_v1 workload_before{};
    workload_before.struct_size = sizeof(workload_before);
    (void)api.session_get_telemetry(session, &workload_before);
    const Clock::time_point workload_started = Clock::now();
    double case_gemm_elapsed_ms = 0.0;
    std::optional<xvram_error_info_v1> operation_failure;
    for (std::uint32_t pass = 0; pass < options.passes; ++pass) {
      const Clock::time_point gemm_started = Clock::now();
      xvram_operation operation = nullptr;
      status = api.gemm_submit(plan, &operation);
      if (status != XVRAM_STATUS_SUCCESS) {
        break;
      }
      xvram_operation_info_v1 operation_info = XVRAM_OPERATION_INFO_V1_INIT;
      for (;;) {
        status = api.operation_poll(operation, &operation_info);
        if (status != XVRAM_STATUS_SUCCESS || operation_info.state == XVRAM_OPERATION_COMPLETED ||
            operation_info.state == XVRAM_OPERATION_FAILED ||
            operation_info.state == XVRAM_OPERATION_CANCELLED) {
          break;
        }
        if (progress) {
          workload.tiles_retired = static_cast<std::uint64_t>(pass) * plan_info.tile_count +
                                   std::min(operation_info.units_completed, plan_info.tile_count);
          report.workloads.push_back(workload);
          progress(report, completed_cases, total_cases);
          report.workloads.pop_back();
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
      if (status == XVRAM_STATUS_SUCCESS && operation_info.result != XVRAM_STATUS_SUCCESS) {
        status = operation_info.result;
      }
      if (status != XVRAM_STATUS_SUCCESS && operation != nullptr) {
        xvram_error_info_v1 native = XVRAM_ERROR_INFO_V1_INIT;
        if (api.operation_get_error(operation, &native) == XVRAM_STATUS_SUCCESS &&
            native.status != XVRAM_STATUS_SUCCESS) {
          operation_failure = native;
        }
      }
      api.operation_release(operation);
      const double gemm_elapsed_ms =
          std::chrono::duration<double, std::milli>(Clock::now() - gemm_started).count();
      gemm_timings.add(gemm_elapsed_ms);
      case_gemm_elapsed_ms += gemm_elapsed_ms;
      if (status != XVRAM_STATUS_SUCCESS) {
        break;
      }
      if (!verify_output(pass)) {
        break;
      }
      ++workload.passes_completed;
      workload.tiles_retired = (static_cast<std::uint64_t>(pass) + 1ULL) * plan_info.tile_count;
    }
    workload.elapsed_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - workload_started).count();
    const long double operation_count =
        2.0L * static_cast<long double>(shape.m) * static_cast<long double>(shape.n) *
        static_cast<long double>(shape.k) * static_cast<long double>(workload.passes_completed);
    workload.achieved_tflops =
        case_gemm_elapsed_ms > 0.0
            ? static_cast<double>(operation_count /
                                  (static_cast<long double>(case_gemm_elapsed_ms) * 1.0e9L))
            : 0.0;
    if (status != XVRAM_STATUS_SUCCESS) {
      const std::string message =
          operation_failure.has_value() && operation_failure->message[0] != '\0'
              ? std::string(operation_failure->message)
              : error_message(api, session, status);
      fail_report(report, exit_code_for_status(status), "cuda_error", "execution", "gemm_execute",
                  message);
      if (operation_failure.has_value()) {
        report.outcome.stage = operation_failure->stage;
        report.outcome.operation = operation_failure->operation;
        report.outcome.native_code = operation_failure->native_code;
        if (operation_failure->native_name[0] != '\0') {
          report.outcome.native_name = operation_failure->native_name;
        }
      }
      workload.status = "failed";
      report.workloads.push_back(workload);
      api.gemm_plan_release(plan);
      (void)api.allocation_release(a);
      (void)api.allocation_release(b);
      (void)api.allocation_release(c);
      all_match = false;
      break;
    }

    if (!reference_ok) {
      fail_report(report, exit_failure, "internal_error", "verification", "generate_reference",
                  "failed to generate the deterministic physical C reference");
      all_match = false;
    } else if (verification_status != XVRAM_STATUS_SUCCESS) {
      fail_report(report, exit_code_for_status(verification_status), "cuda_error", "verification",
                  "allocation_read", error_message(api, session, verification_status));
      all_match = false;
    }
    case_match = case_match && reference_ok && verification_status == XVRAM_STATUS_SUCCESS &&
                 case_actual_digest.format() == case_expected_digest.format();
    all_match = all_match && case_match;
    workload.status = case_match ? "completed" : "failed";
    workload.output_digest128 = case_actual_digest.format();
    xvram_session_telemetry_v1 workload_after{};
    workload_after.struct_size = sizeof(workload_after);
    if (api.session_get_telemetry(session, &workload_after) == XVRAM_STATUS_SUCCESS) {
      const auto delta = [](const std::uint64_t after, const std::uint64_t before) {
        return after >= before ? after - before : 0;
      };
      workload.cache_hits = delta(workload_after.cache_hits, workload_before.cache_hits);
      workload.cache_misses = delta(workload_after.cache_misses, workload_before.cache_misses);
      const std::uint64_t requests = *workload.cache_hits + *workload.cache_misses;
      workload.cache_hit_rate =
          requests == 0 ? 0.0
                        : static_cast<double>(*workload.cache_hits) / static_cast<double>(requests);
      workload.bytes_h2d = delta(workload_after.bytes_h2d, workload_before.bytes_h2d);
      workload.bytes_d2h = delta(workload_after.bytes_d2h, workload_before.bytes_d2h);
      workload.clean_evictions =
          delta(workload_after.clean_evictions, workload_before.clean_evictions);
      workload.dirty_evictions =
          delta(workload_after.dirty_evictions, workload_before.dirty_evictions);
      workload.writebacks_completed =
          delta(workload_after.writebacks_completed, workload_before.writebacks_completed);
      workload.mapping_count = delta(workload_after.mappings, workload_before.mappings);
      workload.unmap_count = delta(workload_after.unmaps, workload_before.unmaps);
      workload.set_access_count =
          delta(workload_after.set_access_calls, workload_before.set_access_calls);
      workload.handle_reuse_count =
          delta(workload_after.handle_reuses, workload_before.handle_reuses);
      workload.event_boundary_count =
          delta(workload_after.event_boundaries, workload_before.event_boundaries);
      workload.unsafe_remap_count =
          delta(workload_after.unsafe_remaps, workload_before.unsafe_remaps);
      workload.unsafe_transition_count =
          delta(workload_after.unsafe_transitions, workload_before.unsafe_transitions);
    }
    report.workloads.push_back(workload);
    aggregate_logical_bytes = std::max(aggregate_logical_bytes, *logical_bytes);
    ++completed_cases;

    // A backend GEMM plan retains the three allocation backends used to create it. Drop that
    // ownership before asking the public allocation handles to release their runtime storage.
    api.gemm_plan_release(plan);
    const xvram_status release_a = api.allocation_release(a);
    const xvram_status release_b = api.allocation_release(b);
    const xvram_status release_c = api.allocation_release(c);
    if (release_a != XVRAM_STATUS_SUCCESS || release_b != XVRAM_STATUS_SUCCESS ||
        release_c != XVRAM_STATUS_SUCCESS) {
      fail_report(report, exit_failure, "cleanup_error", "cleanup", "allocation_release",
                  "one or more GEMM allocations could not be released");
      break;
    }
    if (progress) {
      progress(report, completed_cases, total_cases);
    }
  }

  const bool all_cases_completed = completed_cases == total_cases;
  const xvram_status close_status = api.session_close(session, XVRAM_TIMEOUT_INFINITE);
  xvram_session_telemetry_v1 telemetry{};
  telemetry.struct_size = sizeof(telemetry);
  (void)api.session_get_telemetry(session, &telemetry);
  const std::string close_message = close_status == XVRAM_STATUS_SUCCESS
                                        ? std::string{}
                                        : error_message(api, session, close_status);
  api.session_release(session);
  report.configuration.effective_chunk_bytes = *effective_chunk_bytes;
  report.configuration.initial_cache_target_bytes = initial_telemetry.cache_target_bytes;
  report.configuration.effective_workspace_cap_bytes = options.workspace_bytes;
  report.cache.target_bytes_initial = initial_telemetry.cache_target_bytes;
  report.cache.target_bytes_minimum = telemetry.cache_target_minimum_bytes;
  report.cache.target_bytes_maximum = telemetry.cache_target_maximum_bytes;
  report.cache.target_bytes_end = telemetry.cache_target_bytes;
  report.cache.resident_bytes_peak = telemetry.resident_bytes_peak;
  report.cache.pinned_staging_bytes = telemetry.pinned_staging_bytes;
  report.cache.workspace_bytes_peak = telemetry.workspace_bytes;
  report.cache.physical_frame_count_peak = telemetry.resident_bytes_peak / *effective_chunk_bytes;
  report.cache.physical_handle_create_count = telemetry.physical_handles_created;
  report.cache.physical_handle_release_count = telemetry.physical_handles_released;
  report.cache.handle_reuse_count = telemetry.handle_reuses;
  report.cache.mapping_count = telemetry.mappings;
  report.cache.unmap_count = telemetry.unmaps;
  report.cache.set_access_count = telemetry.set_access_calls;
  report.cache.event_boundary_count = telemetry.event_boundaries;
  report.cache.unsafe_remap_count = telemetry.unsafe_remaps;
  report.cache.unsafe_transition_count = telemetry.unsafe_transitions;
  report.cache.cache_hits = telemetry.cache_hits;
  report.cache.cache_misses = telemetry.cache_misses;
  const std::uint64_t global_requests = telemetry.cache_hits + telemetry.cache_misses;
  report.cache.cache_hit_rate = global_requests == 0 ? 0.0
                                                     : static_cast<double>(telemetry.cache_hits) /
                                                           static_cast<double>(global_requests);
  report.cache.clean_evictions = telemetry.clean_evictions;
  report.cache.dirty_evictions = telemetry.dirty_evictions;
  report.cache.writebacks_completed = telemetry.writebacks_completed;
  report.cache.target_shrink_count = telemetry.budget_shrinks;
  report.cache.target_grow_count = telemetry.budget_grows;
  report.cache.target_oom_retry_count = telemetry.target_oom_retries;
  report.telemetry.cuda_driver_version = telemetry.cuda_driver_version;
  report.telemetry.cublas_version = telemetry.cublas_version;
  report.telemetry.cublas_lt_version = telemetry.cublas_lt_version;
  report.telemetry.cublas_lt_available = telemetry.cublas_lt_available != 0;
  if (telemetry.cublas_library_source == XVRAM_CUBLAS_SOURCE_SYSTEM) {
    report.telemetry.cublas_library_source = "system";
  } else if (telemetry.cublas_library_source == XVRAM_CUBLAS_SOURCE_APP_LOCAL) {
    report.telemetry.cublas_library_source = "app_local";
  } else if (telemetry.cublas_library_source == XVRAM_CUBLAS_SOURCE_EXPLICIT) {
    report.telemetry.cublas_library_source = "explicit";
  }
  report.telemetry.algorithms_selected = telemetry.algorithm_selections;
  report.telemetry.algorithm_cache_hits = telemetry.algorithm_cache_hits;
  report.telemetry.budget_sample_count = telemetry.budget_sample_count;
  report.telemetry.cuda_free_bytes_minimum = telemetry.cuda_free_bytes_minimum;
  report.telemetry.cuda_free_bytes_end = telemetry.cuda_free_bytes_end;
  if (report.device.has_value()) {
    report.device->free_memory_bytes_end = telemetry.cuda_free_bytes_end;
  }
  if (telemetry.wddm_budget_observed != 0U) {
    report.telemetry.wddm_available_bytes_minimum = telemetry.wddm_available_bytes_minimum;
    report.telemetry.wddm_available_bytes_end = telemetry.wddm_available_bytes_end;
    if (report.device.has_value()) {
      report.device->wddm_available_bytes_minimum = telemetry.wddm_available_bytes_minimum;
      report.device->wddm_available_bytes_end = telemetry.wddm_available_bytes_end;
    }
  }
  report.telemetry.replans = 0;
  report.telemetry.watchdog_rejections = telemetry.watchdog_rejections;
  report.telemetry.total_elapsed_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - executor_started).count();
  report.telemetry.plan_timing = plan_timings.summarize();
  report.telemetry.gemm_timing = gemm_timings.summarize();
  report.numerics.validation_mode = "structured";
  report.numerics.reference_precision = "analytic";
  report.numerics.elements_verified = verified_elements;
  report.numerics.mismatch_count = mismatches;
  report.numerics.absolute_tolerance = maximum_absolute_tolerance;
  report.numerics.relative_tolerance = 0.0;
  report.numerics.maximum_absolute_error = maximum_absolute_error;
  report.numerics.maximum_relative_error = maximum_relative_error;
  report.numerics.first_mismatch_byte_offset = first_mismatch;
  report.numerics.expected_digest128 = expected_digest.format();
  report.numerics.output_digest128 = actual_digest.format();
  report.numerics.all_results_match =
      all_match && expected_digest.format() == actual_digest.format();
  report.proof.logical_allocation_bytes = aggregate_logical_bytes;
  report.proof.total_vram_bytes = device_info.total_memory_bytes;
  report.proof.maximum_cache_target_bytes = telemetry.cache_target_maximum_bytes;
  report.proof.maximum_working_set_bytes =
      report.plans.empty() ? 0
                           : std::max_element(report.plans.begin(), report.plans.end(),
                                              [](const PlanResult& left, const PlanResult& right) {
                                                return left.maximum_working_set_bytes <
                                                       right.maximum_working_set_bytes;
                                              })
                                 ->maximum_working_set_bytes;
  report.proof.workspace_cap_bytes = options.workspace_bytes;
  report.proof.logical_data_exceeds_vram = aggregate_logical_bytes > device_info.total_memory_bytes;
  report.proof.cache_smaller_than_logical =
      telemetry.cache_target_maximum_bytes < aggregate_logical_bytes;
  report.proof.tiled_execution_verified =
      all_cases_completed && !report.plans.empty() &&
      std::all_of(report.workloads.begin(), report.workloads.end(), [](const WorkloadResult& item) {
        return item.tiles_total != 0 && item.tiles_retired == item.tiles_total;
      });
  report.proof.handles_reused = telemetry.handle_reuses > 0;
  report.proof.stable_virtual_addresses_verified =
      (telemetry.flags & XVRAM_TELEMETRY_STABLE_VIRTUAL_ADDRESSES) != 0U;
  report.proof.set_access_after_map_verified = telemetry.mappings == telemetry.set_access_calls;
  report.proof.event_boundaries_verified = telemetry.mappings == telemetry.unmaps &&
                                           telemetry.unmaps == telemetry.event_boundaries &&
                                           telemetry.unsafe_remaps == 0;
  report.proof.no_physical_aliases_verified =
      (telemetry.flags & XVRAM_TELEMETRY_NO_PHYSICAL_ALIASES) != 0U;
  report.proof.dirty_writeback_verified = telemetry.writebacks_completed > 0;
  report.proof.cache_target_respected =
      telemetry.resident_bytes_peak <= telemetry.cache_target_maximum_bytes &&
      telemetry.workspace_bytes <=
          telemetry.cache_target_maximum_bytes - telemetry.resident_bytes_peak;
  report.proof.workspace_bounded = telemetry.workspace_bytes <= options.workspace_bytes;
  report.proof.all_workloads_match_reference = report.numerics.all_results_match;
  report.proof.raw_virtual_addresses_omitted = true;
  const bool session_close_ok = close_status == XVRAM_STATUS_SUCCESS;
  const bool mapping_ledger_complete = telemetry.resident_bytes == 0 &&
                                       telemetry.mappings == telemetry.unmaps &&
                                       telemetry.unmaps == telemetry.event_boundaries;
  const bool handle_ledger_complete =
      telemetry.physical_handles_created == telemetry.physical_handles_released;
  const bool cleanup_ok = session_close_ok && mapping_ledger_complete && handle_ledger_complete;
  report.cleanup.complete = cleanup_ok;
  report.cleanup.operations_drained = session_close_ok;
  report.cleanup.events_drained = session_close_ok;
  report.cleanup.events_destroyed = session_close_ok;
  report.cleanup.streams_destroyed = session_close_ok;
  report.cleanup.cublas_handles_destroyed = session_close_ok;
  report.cleanup.mappings_removed = session_close_ok && mapping_ledger_complete;
  report.cleanup.physical_handles_released = session_close_ok && handle_ledger_complete;
  report.cleanup.workspace_released = session_close_ok;
  report.cleanup.virtual_reservations_released = session_close_ok;
  report.cleanup.pinned_staging_released = session_close_ok;
  report.cleanup.host_backing_released = session_close_ok;
  report.cleanup.context_released = session_close_ok;
  report.cleanup.worker_terminated = true;

  if (report.outcome.exit_code != exit_prerequisite && report.outcome.exit_code != exit_completed &&
      report.outcome.status == "failed") {
    return report;
  }
  if (!cleanup_ok) {
    std::string cleanup_message = close_message;
    if (cleanup_message.empty() && !mapping_ledger_complete) {
      cleanup_message =
          "cleanup telemetry did not balance mappings, unmaps, event boundaries, and residency";
    } else if (cleanup_message.empty() && !handle_ledger_complete) {
      cleanup_message = "cleanup telemetry did not balance physical handle creation and release";
    }
    fail_report(report, exit_failure, "cleanup_error", "cleanup", "session_close",
                std::move(cleanup_message));
  } else if (!all_cases_completed) {
    if (report.outcome.message.value_or("").empty()) {
      fail_report(report, exit_failure, "internal_error", "execution", "incomplete_suite",
                  "not every requested GEMM case completed");
    }
  } else if (!all_match || mismatches != 0 || expected_digest.format() != actual_digest.format()) {
    fail_report(report, exit_corruption, "data_mismatch", "verification", "full_reference",
                "GEMM output differs from the analytical CPU reference");
  } else {
    report.outcome.status = "completed";
    report.outcome.reason.reset();
    report.outcome.exit_code = exit_completed;
    report.outcome.stage.reset();
    report.outcome.operation.reset();
    report.outcome.message.reset();
  }
  return report;
}

} // namespace xvram::gemm_bench
