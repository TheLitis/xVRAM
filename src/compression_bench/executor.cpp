#include "compression_bench/executor.hpp"

#include "platform/cuda/cuda_api.hpp"
#include "platform/system_info.hpp"
#include "residency/core.hpp"
#include "residency/runtime.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <iomanip>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace xvram::compression {
namespace {

constexpr int exit_completed = 0;
constexpr int exit_prerequisite = 23;
constexpr int exit_corruption = 24;
constexpr int exit_oom = 25;
constexpr int exit_timeout = 26;
constexpr int exit_failure = 27;
constexpr std::uint64_t gibibyte = 1024ULL * 1024ULL * 1024ULL;
constexpr std::uint64_t default_service_reserve = 256ULL * 1024ULL * 1024ULL;

using Clock = std::chrono::steady_clock;

struct Digest128 {
  std::uint64_t high = 0x6a09e667f3bcc909ULL;
  std::uint64_t low = 0xbb67ae8584caa73bULL;

  void update(const std::span<const std::byte> bytes) noexcept {
    std::size_t cursor = 0;
    while (cursor + sizeof(std::uint64_t) <= bytes.size()) {
      std::uint64_t value = 0;
      std::memcpy(&value, bytes.data() + cursor, sizeof(value));
      low ^= value + 0x9e3779b97f4a7c15ULL + (high << 6U) + (high >> 2U);
      low *= 0x100000001b3ULL;
      high ^= low + value * 0x517cc1b727220a95ULL;
      high = (high << 17U) | (high >> 47U);
      high *= 0x94d049bb133111ebULL;
      cursor += sizeof(value);
    }
    while (cursor < bytes.size()) {
      const std::uint64_t value =
          static_cast<std::uint64_t>(std::to_integer<unsigned char>(bytes[cursor++]));
      low = (low ^ value) * 0x100000001b3ULL;
      high ^= low + value * 0x517cc1b727220a95ULL;
    }
  }

  [[nodiscard]] std::string hex() const {
    std::ostringstream output;
    output << std::hex << std::nouppercase << std::setfill('0') << std::setw(16) << high
           << std::setw(16) << low;
    return output.str();
  }
};

struct CaseSpec {
  residency::CompressionMode compression_mode = residency::CompressionMode::adaptive;
  RequestedPath path = RequestedPath::automatic;
  residency::RuntimePolicy replacement_policy = residency::RuntimePolicy::clock;
  ScenarioKind scenario = ScenarioKind::compressible_read;
};

struct CaseResult {
  WorkloadResult workload;
  residency::RuntimeTelemetry telemetry;
  residency::RuntimeTelemetry storage_telemetry;
  Cleanup cleanup;
  std::vector<double> measurement_samples_ms;
  std::uint64_t effective_chunk_bytes = 0;
  bool close_complete = false;
  bool write_admission_verified = false;
  bool gpu_encode_before_d2h_verified = false;
  std::optional<ExecutionFailure> failure;
  std::optional<std::string> reason;
  int exit_code = exit_failure;
};

[[nodiscard]] std::uint64_t saturating_add(const std::uint64_t left,
                                           const std::uint64_t right) noexcept {
  return right > std::numeric_limits<std::uint64_t>::max() - left
             ? std::numeric_limits<std::uint64_t>::max()
             : left + right;
}

[[nodiscard]] std::uint64_t saturating_multiply(const std::uint64_t left,
                                                const std::uint64_t right) noexcept {
  if (left == 0U || right == 0U) {
    return 0U;
  }
  return right > std::numeric_limits<std::uint64_t>::max() / left
             ? std::numeric_limits<std::uint64_t>::max()
             : left * right;
}

[[nodiscard]] std::uint64_t splitmix64(std::uint64_t value) noexcept {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

void fill_pattern(const ScenarioKind scenario, const std::uint64_t seed,
                  const std::uint64_t absolute_offset, const std::uint64_t chunk_bytes,
                  const std::span<std::byte> output, const bool phase_changed = false) noexcept {
  const std::uint64_t chunk_index = chunk_bytes == 0U ? 0U : absolute_offset / chunk_bytes;
  const bool compressible = scenario == ScenarioKind::compressible_read ||
                            scenario == ScenarioKind::reuse ||
                            scenario == ScenarioKind::dirty_writeback ||
                            (scenario == ScenarioKind::mixed &&
                             detail::mixed_chunk_is_compressible(chunk_index, phase_changed));
  if (compressible) {
    static constexpr std::array<unsigned char, 32> motif{
        0x58, 0x56, 0x52, 0x41, 0x4d, 0x2d, 0x4c, 0x5a, 0x34, 0x2d, 0x56,
        0x31, 0x00, 0x00, 0x00, 0x00, 0x58, 0x56, 0x52, 0x41, 0x4d, 0x2d,
        0x4c, 0x5a, 0x34, 0x2d, 0x56, 0x31, 0x00, 0x00, 0x00, 0x00};
    std::array<std::byte, motif.size()> seeded_motif{};
    for (std::size_t index = 0; index < motif.size(); ++index) {
      const std::uint64_t generation_seed = phase_changed ? seed ^ 0xd1b54a32d192ed03ULL : seed;
      seeded_motif[index] = static_cast<std::byte>(
          motif[index] ^ static_cast<unsigned char>(generation_seed & 0x03U));
    }
    std::size_t cursor = 0;
    std::size_t motif_offset = static_cast<std::size_t>(absolute_offset % motif.size());
    if (motif_offset != 0U && !output.empty()) {
      const std::size_t initial = std::min(output.size(), motif.size() - motif_offset);
      std::memcpy(output.data(), seeded_motif.data() + motif_offset, initial);
      cursor += initial;
    }
    while (cursor + seeded_motif.size() <= output.size()) {
      std::memcpy(output.data() + cursor, seeded_motif.data(), seeded_motif.size());
      cursor += seeded_motif.size();
    }
    if (cursor < output.size()) {
      std::memcpy(output.data() + cursor, seeded_motif.data(), output.size() - cursor);
    }
    return;
  }
  std::size_t cursor = 0;
  while (cursor < output.size()) {
    const std::uint64_t word_index = (absolute_offset + cursor) / sizeof(std::uint64_t);
    const std::uint64_t phase_seed = phase_changed ? seed ^ 0xa5a5a5a55a5a5a5aULL : seed;
    const std::uint64_t word = splitmix64(phase_seed ^ word_index);
    const std::size_t available = std::min(sizeof(word), output.size() - cursor);
    std::memcpy(output.data() + cursor, &word, available);
    cursor += available;
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

[[nodiscard]] int classify_runtime(const residency::RuntimeStatus status) noexcept {
  switch (status) {
  case residency::RuntimeStatus::success:
    return exit_completed;
  case residency::RuntimeStatus::invalid_argument:
  case residency::RuntimeStatus::unavailable:
  case residency::RuntimeStatus::unsupported:
    return exit_prerequisite;
  case residency::RuntimeStatus::host_oom:
  case residency::RuntimeStatus::device_oom:
  case residency::RuntimeStatus::budget_pressure:
    return exit_oom;
  case residency::RuntimeStatus::timeout:
    return exit_timeout;
  case residency::RuntimeStatus::callback_skipped:
  case residency::RuntimeStatus::callback_failed:
  case residency::RuntimeStatus::cuda_failure:
  case residency::RuntimeStatus::poisoned:
  case residency::RuntimeStatus::cleanup_failure:
  case residency::RuntimeStatus::internal_failure:
    return exit_failure;
  }
  return exit_failure;
}

[[nodiscard]] std::string classify_reason(const residency::RuntimeStatus status) {
  switch (status) {
  case residency::RuntimeStatus::invalid_argument:
    return "invalid_configuration";
  case residency::RuntimeStatus::unavailable:
    return "device_unavailable";
  case residency::RuntimeStatus::unsupported:
    return "codec_or_vmm_unsupported";
  case residency::RuntimeStatus::host_oom:
    return "host_oom";
  case residency::RuntimeStatus::device_oom:
    return "device_oom";
  case residency::RuntimeStatus::budget_pressure:
    return "budget_pressure";
  case residency::RuntimeStatus::timeout:
    return "timeout";
  case residency::RuntimeStatus::success:
    return "completed";
  default:
    return "runtime_failure";
  }
}

[[nodiscard]] std::string mode_name(const residency::CompressionMode mode) {
  return std::string(residency::compression_mode_name(mode));
}

[[nodiscard]] std::string policy_name(const residency::RuntimePolicy value) {
  return value == residency::RuntimePolicy::lru ? "lru" : "clock";
}

[[nodiscard]] std::string workload_path_name(const RequestedPath value) {
  switch (value) {
  case RequestedPath::cpu_lz4_gpu:
    return "cpu_lz4_gpu_decode";
  case RequestedPath::gpu_lz4:
    return "nvcomp_gpu_codec";
  case RequestedPath::automatic:
  case RequestedPath::raw:
  case RequestedPath::all:
    return "raw";
  }
  return "raw";
}

[[nodiscard]] std::uint64_t monotonic_nanoseconds() noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
          .count());
}

[[nodiscard]] const char*
trace_event_name(const residency::CompressionTraceEventKind value) noexcept {
  switch (value) {
  case residency::CompressionTraceEventKind::encode_submit:
    return "encode_submit";
  case residency::CompressionTraceEventKind::encode_retire:
    return "encode_retire";
  case residency::CompressionTraceEventKind::decode_submit:
    return "decode_submit";
  case residency::CompressionTraceEventKind::decode_retire:
    return "decode_retire";
  case residency::CompressionTraceEventKind::h2d_submit:
    return "h2d_submit";
  case residency::CompressionTraceEventKind::h2d_retire:
    return "h2d_retire";
  case residency::CompressionTraceEventKind::d2h_submit:
    return "d2h_submit";
  case residency::CompressionTraceEventKind::d2h_retire:
    return "d2h_retire";
  case residency::CompressionTraceEventKind::generation_discard:
    return "generation_discard";
  }
  return "diagnostic";
}

void emit_runtime_trace(const TraceCallback& trace, std::uint64_t& sequence,
                        const residency::CompressionTraceEvent& source) {
  if (!trace) {
    return;
  }
  TraceRecord record;
  record.sequence = ++sequence;
  record.monotonic_time_ns = monotonic_nanoseconds();
  record.event = trace_event_name(source.kind);
  record.allocation_id = source.key.allocation_id.value;
  record.chunk_index = source.key.chunk_index;
  record.operation_id = source.operation_id;
  record.source_generation = source.source_generation;
  record.target_generation = source.target_generation;
  record.slot_generation = source.slot_generation;
  if (source.from_representation.has_value()) {
    record.from_representation =
        std::string(residency::backing_representation_name(*source.from_representation));
  }
  if (source.to_representation.has_value()) {
    record.to_representation =
        std::string(residency::backing_representation_name(*source.to_representation));
  }
  record.path = std::string(residency::compression_path_name(source.path));
  record.logical_bytes = source.logical_bytes;
  record.physical_bytes = source.physical_bytes;
  record.reason = std::string(source.reason);
  record.speculative = source.speculative;
  trace(record);
}

[[nodiscard]] std::string bytes_as_uuid(const unsigned char* bytes) {
  static constexpr std::array<std::size_t, 4> hyphens{4U, 6U, 8U, 10U};
  static constexpr char hex[] = "0123456789abcdef";
  std::string output;
  output.reserve(36U);
  for (std::size_t index = 0; index < 16U; ++index) {
    if (std::find(hyphens.begin(), hyphens.end(), index) != hyphens.end()) {
      output.push_back('-');
    }
    output.push_back(hex[(bytes[index] >> 4U) & 0x0FU]);
    output.push_back(hex[bytes[index] & 0x0FU]);
  }
  return output;
}

[[nodiscard]] bool get_attribute(cuda::CudaApi& api, const cuda::abi::Device device,
                                 const cuda::abi::NativeDeviceAttribute attribute,
                                 std::uint32_t& output) noexcept {
  int value = 0;
  if (api.device_get_attribute_ == nullptr ||
      api.device_get_attribute_(&value, attribute, device) != cuda::abi::success || value < 0) {
    return false;
  }
  output = static_cast<std::uint32_t>(value);
  return true;
}

void set_failure(CaseResult& result, const residency::Runtime& runtime,
                 const residency::RuntimeStatus status, const ScenarioKind scenario,
                 const std::optional<std::uint64_t> chunk_index = std::nullopt) {
  result.exit_code = classify_runtime(status);
  result.reason = classify_reason(status);
  const residency::RuntimeError& error = runtime.error();
  ExecutionFailure failure;
  failure.stage = error.stage.empty() ? "execution" : error.stage;
  failure.operation = error.operation.empty() ? "runtime" : error.operation;
  failure.message = error.message.empty() ? residency::runtime_status_name(status) : error.message;
  failure.native_code = error.native_code;
  failure.scenario = std::string(scenario_name(scenario));
  failure.chunk_index = chunk_index;
  result.failure = std::move(failure);
  result.workload.status = result.exit_code == exit_prerequisite ? "not_run" : "failed";
}

[[nodiscard]] std::vector<ScenarioKind> expand_scenarios(const ScenarioKind requested) {
  if (requested != ScenarioKind::suite) {
    return {requested};
  }
  return {ScenarioKind::compressible_read, ScenarioKind::incompressible_read, ScenarioKind::reuse,
          ScenarioKind::dirty_writeback, ScenarioKind::mixed};
}

[[nodiscard]] std::vector<residency::CompressionMode>
expand_compression_modes(const RequestedCompressionPolicy requested) {
  if (requested == RequestedCompressionPolicy::both) {
    return {residency::CompressionMode::adaptive, residency::CompressionMode::capacity};
  }
  return {requested == RequestedCompressionPolicy::capacity ? residency::CompressionMode::capacity
                                                            : residency::CompressionMode::adaptive};
}

[[nodiscard]] std::vector<RequestedPath> expand_paths(const RequestedPath requested) {
  if (requested == RequestedPath::all) {
    return {RequestedPath::raw, RequestedPath::cpu_lz4_gpu, RequestedPath::gpu_lz4};
  }
  return {requested};
}

[[nodiscard]] std::vector<residency::RuntimePolicy>
expand_policies(const RequestedReplacementPolicy requested) {
  if (requested == RequestedReplacementPolicy::both) {
    return {residency::RuntimePolicy::clock, residency::RuntimePolicy::lru};
  }
  return {requested == RequestedReplacementPolicy::lru ? residency::RuntimePolicy::lru
                                                       : residency::RuntimePolicy::clock};
}

[[nodiscard]] std::uint64_t planned_operation_count(const ScenarioKind scenario,
                                                    const ExecutorOptions& options,
                                                    const std::uint64_t chunks,
                                                    const std::uint64_t hot_chunks,
                                                    const bool gpu_initialization) noexcept {
  std::uint64_t transactions = 0;
  switch (scenario) {
  case ScenarioKind::compressible_read:
    transactions = saturating_multiply(chunks, options.warmup_passes + options.measurement_passes);
    break;
  case ScenarioKind::incompressible_read:
  case ScenarioKind::dirty_writeback:
    transactions = saturating_multiply(chunks, options.passes);
    break;
  case ScenarioKind::mixed:
    transactions = saturating_multiply(chunks, options.passes);
    if (options.passes > 1U) {
      transactions = saturating_add(transactions, chunks);
      std::uint64_t history_chunks = 0;
      for (std::uint64_t chunk = 0; chunk < chunks; ++chunk) {
        history_chunks =
            saturating_add(history_chunks, detail::mixed_chunk_needs_history(chunk) ? 1U : 0U);
      }
      transactions = saturating_add(
          transactions, saturating_multiply(history_chunks, detail::mixed_history_cycles));
    }
    break;
  case ScenarioKind::reuse:
    transactions = saturating_add(chunks, saturating_multiply(hot_chunks, 7U));
    break;
  case ScenarioKind::budget_pressure:
    transactions = saturating_add(saturating_multiply(chunks, 2U), 10U);
    break;
  case ScenarioKind::suite:
    break;
  }
  // Host initialization and final verification both report bounded progress.
  const std::uint64_t initialization = saturating_multiply(chunks, gpu_initialization ? 3U : 2U);
  return saturating_add(initialization, transactions);
}

void emit_trace(const TraceCallback& trace, std::uint64_t& sequence, std::string event,
                const residency::RuntimeAllocation& allocation, const std::uint64_t chunk_index,
                const std::uint64_t operation_id, const RequestedPath path,
                const std::uint64_t logical_bytes,
                const std::optional<std::uint64_t> physical_bytes, std::string reason,
                const std::optional<residency::HostChunkInfo>& source = std::nullopt,
                const std::optional<residency::HostChunkInfo>& target = std::nullopt) {
  if (!trace) {
    return;
  }
  TraceRecord record;
  record.sequence = ++sequence;
  record.monotonic_time_ns = monotonic_nanoseconds();
  record.event = std::move(event);
  record.allocation_id = allocation.id.value;
  record.chunk_index = chunk_index;
  record.operation_id = operation_id;
  record.path = workload_path_name(path);
  record.logical_bytes = logical_bytes;
  record.physical_bytes = physical_bytes;
  record.reason = std::move(reason);
  if (source.has_value()) {
    record.source_generation = source->generation;
    record.from_representation =
        std::string(residency::backing_representation_name(source->representation));
  }
  if (target.has_value()) {
    record.target_generation = target->generation;
    record.to_representation =
        std::string(residency::backing_representation_name(target->representation));
  }
  trace(record);
}

void add_timing_total(TimingSummary& timing, const std::uint64_t samples,
                      const std::uint64_t nanoseconds) {
  timing.sample_count = saturating_add(timing.sample_count, samples);
  const double milliseconds = static_cast<double>(nanoseconds) / 1'000'000.0;
  timing.total_ms = timing.total_ms.value_or(0.0) + milliseconds;
}

void finish_timing_summary(TimingSummary& timing) {
  if (timing.sample_count == 0U || !timing.total_ms.has_value()) {
    return;
  }
  // Runtime aggregation preserves exact counts and totals. Distribution fields stay absent unless
  // the benchmark retained real per-sample observations; an average must not masquerade as p50 or
  // p95.
  timing.minimum_ms.reset();
  timing.median_ms.reset();
  timing.p95_ms.reset();
  timing.maximum_ms.reset();
}

void finish_timing_samples(TimingSummary& timing, std::vector<double> samples) {
  if (samples.empty()) {
    return;
  }
  std::sort(samples.begin(), samples.end());
  const auto percentile = [&](const double fraction) {
    if (samples.size() == 1U) {
      return samples.front();
    }
    const double position = fraction * static_cast<double>(samples.size() - 1U);
    const std::size_t lower = static_cast<std::size_t>(position);
    const std::size_t upper = std::min(samples.size() - 1U, lower + 1U);
    const double weight = position - static_cast<double>(lower);
    return samples[lower] + (samples[upper] - samples[lower]) * weight;
  };
  timing.sample_count = static_cast<std::uint64_t>(samples.size());
  timing.total_ms = 0.0;
  for (const double sample : samples) {
    *timing.total_ms += sample;
  }
  timing.minimum_ms = samples.front();
  timing.median_ms = percentile(0.50);
  timing.p95_ms = percentile(0.95);
  timing.maximum_ms = samples.back();
}

using CaseProgress =
    std::function<void(const WorkloadResult&, std::uint64_t completed, std::uint64_t total)>;

void capture_cleanup_evidence(CaseResult& result, residency::Runtime& runtime,
                              const RequestedPath requested_path,
                              const bool close_complete) noexcept {
  result.close_complete = close_complete;
  result.telemetry = runtime.telemetry();
  result.cleanup = detail::derive_runtime_cleanup(result.telemetry, close_complete);
  result.write_admission_verified =
      detail::derive_write_admission_proof(result.telemetry, requested_path);

  const residency::RuntimeTelemetry& telemetry = result.telemetry;
  const bool gpu_encode_not_applicable = telemetry.gpu_encode_operations == 0U;
  result.gpu_encode_before_d2h_verified =
      gpu_encode_not_applicable ||
      (telemetry.logical_d2h_bytes != 0U && telemetry.pcie_d2h_bytes != 0U &&
       telemetry.logical_d2h_bytes >= telemetry.pcie_d2h_bytes &&
       telemetry.codec_events_recorded == telemetry.codec_events_retired &&
       telemetry.codec_verification_failures == 0U);
}

void merge_cleanup_evidence(Cleanup& aggregate, const Cleanup& current) noexcept {
  const auto merge = [](std::optional<bool>& destination, const std::optional<bool>& source) {
    destination = destination.value_or(true) && source.value_or(false);
  };
  merge(aggregate.complete, current.complete);
  merge(aggregate.operations_drained, current.operations_drained);
  merge(aggregate.codec_slots_drained, current.codec_slots_drained);
  merge(aggregate.events_drained, current.events_drained);
  merge(aggregate.events_destroyed, current.events_destroyed);
  merge(aggregate.streams_destroyed, current.streams_destroyed);
  merge(aggregate.mappings_removed, current.mappings_removed);
  merge(aggregate.physical_handles_released, current.physical_handles_released);
  merge(aggregate.codec_workspace_released, current.codec_workspace_released);
  merge(aggregate.virtual_reservations_released, current.virtual_reservations_released);
  merge(aggregate.pinned_staging_released, current.pinned_staging_released);
  merge(aggregate.spill_reservations_released, current.spill_reservations_released);
  merge(aggregate.host_backing_released, current.host_backing_released);
  merge(aggregate.context_released, current.context_released);
}

[[nodiscard]] CaseResult run_case(cuda::CudaApi& api, const ExecutorOptions& options,
                                  const CaseSpec& spec, const std::uint64_t logical_bytes,
                                  const std::uint64_t host_store_cap_bytes,
                                  const CaseProgress& progress, const TraceCallback& trace,
                                  std::uint64_t& trace_sequence) {
  CaseResult result;
  WorkloadResult& workload = result.workload;
  workload.scenario = std::string(scenario_name(spec.scenario));
  workload.compression_policy = mode_name(spec.compression_mode);
  workload.path = workload_path_name(spec.path);
  workload.replacement_policy = policy_name(spec.replacement_policy);
  workload.status = "not_run";
  workload.logical_bytes = logical_bytes;
  workload.raw_input_bytes = logical_bytes;

  std::uint64_t operations_completed = 0;
  std::uint64_t operations_total = 0;
  const auto report_progress = [&]() {
    workload.operations_retired = operations_completed;
    if (progress) {
      progress(workload, operations_completed, operations_total);
    }
  };

  residency::RuntimeConfig config;
  config.device_ordinal = options.device_ordinal;
  config.chunk_bytes = options.chunk_bytes;
  config.cache_target_bytes = options.cache_target_bytes.value_or(0U);
  config.device_headroom_bytes = options.device_headroom_bytes;
  config.workspace_reserve_bytes = 0;
  config.staging_slots = options.staging_slots;
  config.policy = spec.replacement_policy;
  config.stall_timeout = options.stall_timeout;
  config.budget_poll_interval = options.budget_poll_interval;
  config.compression_mode = spec.path == RequestedPath::raw ? residency::CompressionMode::disabled
                                                            : spec.compression_mode;
  config.compression_codec = options.codec;
  switch (spec.path) {
  case RequestedPath::raw:
    config.forced_compression_path = residency::CompressionPath::raw;
    break;
  case RequestedPath::cpu_lz4_gpu:
    config.forced_compression_path = residency::CompressionPath::cpu_lz4_gpu_decode;
    break;
  case RequestedPath::gpu_lz4:
    config.forced_compression_path = residency::CompressionPath::nvcomp_gpu_codec;
    break;
  case RequestedPath::automatic:
  case RequestedPath::all:
    config.forced_compression_path.reset();
    break;
  }
  config.host_store_cap_bytes = host_store_cap_bytes;
  config.host_headroom_bytes = options.host_headroom_bytes.value_or(0U);
  config.compression_workspace_cap_bytes = options.compression_scratch_cap_bytes;
  config.codec_slots = options.codec_slots;
  config.codec_workers = options.codec_workers;
  config.lifecycle_progress = report_progress;
  config.compression_trace = [&](const residency::CompressionTraceEvent& event) {
    emit_runtime_trace(trace, trace_sequence, event);
  };

  residency::Runtime runtime(api, config);
  residency::RuntimeStatus status = runtime.setup();
  if (status != residency::RuntimeStatus::success) {
    set_failure(result, runtime, status, spec.scenario);
    const residency::RuntimeStatus close_status = runtime.close();
    capture_cleanup_evidence(result, runtime, spec.path,
                             close_status == residency::RuntimeStatus::success);
    return result;
  }

  const std::uint64_t chunk_bytes = runtime.chunk_bytes();
  result.effective_chunk_bytes = chunk_bytes;
  const std::uint64_t chunk_count = ((logical_bytes - 1U) / chunk_bytes) + 1U;
  const std::uint64_t target_chunks =
      std::max<std::uint64_t>(1U, runtime.target_bytes() / chunk_bytes);
  const std::uint64_t hot_chunks =
      std::max<std::uint64_t>(1U, std::min(chunk_count, target_chunks / 2U));
  operations_total = planned_operation_count(spec.scenario, options, chunk_count, hot_chunks,
                                             spec.path == RequestedPath::gpu_lz4);
  workload.logical_chunk_count = chunk_count;
  std::uint64_t operation_id = 0;

  residency::RuntimeAllocation allocation;
  status = runtime.allocate(logical_bytes, residency::ResidencyHint::streaming, allocation);
  if (status != residency::RuntimeStatus::success) {
    set_failure(result, runtime, status, spec.scenario);
    const residency::RuntimeStatus close_status = runtime.close();
    capture_cleanup_evidence(result, runtime, spec.path,
                             close_status == residency::RuntimeStatus::success);
    return result;
  }

  std::vector<std::byte> buffer;
  std::vector<std::byte> expected;
  try {
    buffer.resize(static_cast<std::size_t>(chunk_bytes));
    expected.resize(static_cast<std::size_t>(chunk_bytes));
  } catch (const std::bad_alloc&) {
    result.exit_code = exit_oom;
    result.reason = "host_oom";
    result.failure = ExecutionFailure{"host", "allocate_verification_buffers",
                                      "host verification buffer allocation failed"};
    workload.status = "failed";
    const residency::RuntimeStatus release_status = runtime.release(allocation.id);
    const residency::RuntimeStatus close_status = runtime.close();
    capture_cleanup_evidence(result, runtime, spec.path,
                             release_status == residency::RuntimeStatus::success &&
                                 close_status == residency::RuntimeStatus::success);
    return result;
  }

  Digest128 expected_digest;
  bool mixed_phase_changed = false;
  for (std::uint64_t chunk = 0; chunk < chunk_count; ++chunk) {
    const std::uint64_t offset = chunk * chunk_bytes;
    const std::uint64_t valid = std::min(chunk_bytes, logical_bytes - offset);
    const auto valid_buffer = std::span<std::byte>{buffer.data(), static_cast<std::size_t>(valid)};
    fill_pattern(spec.scenario, options.seed, offset, chunk_bytes, valid_buffer);
    expected_digest.update(valid_buffer);
    const std::optional<residency::HostChunkInfo> backing_before =
        runtime.backing_info(allocation.id, chunk);
    status = runtime.write(allocation.id, offset, buffer.data(), valid);
    if (status != residency::RuntimeStatus::success) {
      set_failure(result, runtime, status, spec.scenario, chunk);
      break;
    }
    ++operations_completed;
    const std::optional<residency::HostChunkInfo> backing_after =
        runtime.backing_info(allocation.id, chunk);
    emit_trace(trace, trace_sequence, "generation_commit", allocation, chunk, ++operation_id,
               spec.path, valid,
               backing_after.has_value()
                   ? std::optional<std::uint64_t>{backing_after->stored_payload_bytes}
                   : std::nullopt,
               "initial_backing", backing_before, backing_after);
    report_progress();
    if (spec.path == RequestedPath::gpu_lz4) {
      const std::uint64_t pcie_before = runtime.telemetry().pcie_d2h_bytes;
      const std::array ranges{
          residency::AccessRange{allocation.id, offset, valid, residency::AccessMode::read_write}};
      status = runtime.execute(ranges, 0U, [](const residency::TransactionContext&) {
        return residency::RuntimeStatus::success;
      });
      if (status == residency::RuntimeStatus::success) {
        status = runtime.drain(true);
      }
      if (status != residency::RuntimeStatus::success) {
        set_failure(result, runtime, status, spec.scenario, chunk);
        break;
      }
      ++operations_completed;
      const std::uint64_t pcie_after = runtime.telemetry().pcie_d2h_bytes;
      emit_trace(trace, trace_sequence, "operation_retired", allocation, chunk, ++operation_id,
                 spec.path, valid, pcie_after >= pcie_before ? pcie_after - pcie_before : 0U,
                 "gpu_encode_initial_backing");
      report_progress();
    }
  }

  cuda::abi::DevicePointer pressure_allocation = 0;
  const auto release_pressure = [&]() {
    if (pressure_allocation != 0 && api.mem_free_ != nullptr) {
      (void)api.mem_free_(pressure_allocation);
      pressure_allocation = 0;
    }
  };

  const auto run_transaction = [&](const std::uint64_t chunk, const residency::AccessMode mode,
                                   const bool allow_prefetch) -> bool {
    const std::uint64_t offset = chunk * chunk_bytes;
    const std::uint64_t valid = std::min(chunk_bytes, logical_bytes - offset);
    if (allow_prefetch && options.prefetch_distance != 0U) {
      std::vector<residency::AccessRange> prefetch_ranges;
      prefetch_ranges.reserve(options.prefetch_distance);
      for (std::uint32_t distance = 1; distance <= options.prefetch_distance; ++distance) {
        const std::uint64_t next = chunk + distance;
        if (next >= chunk_count) {
          break;
        }
        prefetch_ranges.push_back(
            residency::AccessRange{allocation.id, next * chunk_bytes,
                                   std::min(chunk_bytes, logical_bytes - next * chunk_bytes),
                                   residency::AccessMode::read});
      }
      if (!prefetch_ranges.empty()) {
        const residency::RuntimeStatus prefetch_status = runtime.prefetch(prefetch_ranges);
        if (prefetch_status != residency::RuntimeStatus::success &&
            prefetch_status != residency::RuntimeStatus::budget_pressure) {
          status = prefetch_status;
          return false;
        }
      }
    }
    const residency::RuntimeTelemetry before = runtime.telemetry();
    const std::array ranges{residency::AccessRange{allocation.id, offset, valid, mode}};
    status = runtime.execute(ranges, 0U, [](const residency::TransactionContext&) {
      // The benchmark intentionally measures backing/residency paths. The event-safe lease still
      // records and retires a real CUDA event even when the resident data is not transformed.
      return residency::RuntimeStatus::success;
    });
    if (status != residency::RuntimeStatus::success) {
      return false;
    }
    const residency::RuntimeTelemetry after = runtime.telemetry();
    const std::uint64_t physical = after.pcie_h2d_bytes >= before.pcie_h2d_bytes
                                       ? after.pcie_h2d_bytes - before.pcie_h2d_bytes
                                       : 0U;
    ++operations_completed;
    const std::uint64_t retired_operation_id = ++operation_id;
    const std::optional<residency::HostChunkInfo> backing =
        runtime.backing_info(allocation.id, chunk);
    emit_trace(trace, trace_sequence, "operation_retired", allocation, chunk, retired_operation_id,
               spec.path, valid, physical,
               mode == residency::AccessMode::read ? "read" : "read_write", backing, backing);
    if (after.gpu_decode_operations > before.gpu_decode_operations) {
      emit_trace(trace, trace_sequence, "codec_decision", allocation, chunk, retired_operation_id,
                 RequestedPath::gpu_lz4, valid, physical, "nvcomp_decode_verified", backing,
                 backing);
    } else if (after.cpu_decode_operations > before.cpu_decode_operations) {
      emit_trace(trace, trace_sequence, "codec_decision", allocation, chunk, retired_operation_id,
                 RequestedPath::cpu_lz4_gpu, valid, physical, "cpu_decode_fallback", backing,
                 backing);
    }
    if (after.gpu_encode_operations > before.gpu_encode_operations) {
      const std::uint64_t encoded_bytes = after.pcie_d2h_bytes >= before.pcie_d2h_bytes
                                              ? after.pcie_d2h_bytes - before.pcie_d2h_bytes
                                              : 0U;
      emit_trace(trace, trace_sequence, "codec_decision", allocation, chunk, retired_operation_id,
                 RequestedPath::gpu_lz4, valid, encoded_bytes,
                 "nvcomp_encode_verified_before_host_commit", backing, backing);
    }
    if (after.spill_reserved_bytes > before.spill_reserved_bytes) {
      emit_trace(trace, trace_sequence, "spill_reserve", allocation, chunk, retired_operation_id,
                 spec.path, valid, after.spill_reserved_bytes - before.spill_reserved_bytes,
                 "write_admission_before_dirty_submission", backing, backing);
    } else if (after.spill_reserved_bytes < before.spill_reserved_bytes) {
      emit_trace(trace, trace_sequence, "spill_release", allocation, chunk, retired_operation_id,
                 spec.path, valid, before.spill_reserved_bytes - after.spill_reserved_bytes,
                 "authoritative_host_commit", backing, backing);
    }
    if (after.budget_shrinks > before.budget_shrinks) {
      emit_trace(trace, trace_sequence, "target_change", allocation, chunk, retired_operation_id,
                 spec.path, valid, after.target_bytes, "live_device_budget", backing, backing);
    }
    if (after.budget_grows > before.budget_grows) {
      emit_trace(trace, trace_sequence, "target_change", allocation, chunk, retired_operation_id,
                 spec.path, valid, after.target_bytes, "safe_sample_hysteresis", backing, backing);
    }
    report_progress();
    return true;
  };

  const auto scan = [&](const std::uint32_t pass, const residency::AccessMode mode,
                        const std::uint64_t count = std::numeric_limits<std::uint64_t>::max()) {
    const std::uint64_t limit = std::min(chunk_count, count);
    for (std::uint64_t position = 0; position < limit; ++position) {
      const std::uint64_t chunk = (pass & 1U) == 0U ? position : limit - position - 1U;
      if (!run_transaction(chunk, mode, true)) {
        return false;
      }
    }
    return true;
  };

  const auto execute_scenario = [&]() -> bool {
    switch (spec.scenario) {
    case ScenarioKind::compressible_read: {
      const std::uint32_t total_passes = options.warmup_passes + options.measurement_passes;
      for (std::uint32_t pass = 0; pass < total_passes; ++pass) {
        const Clock::time_point pass_started = Clock::now();
        if (!scan(pass, residency::AccessMode::read)) {
          return false;
        }
        const double pass_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - pass_started).count();
        if (pass < options.warmup_passes) {
          workload.warmup_passes_completed = pass + 1U;
        } else {
          result.measurement_samples_ms.push_back(pass_ms);
          workload.measurement_passes_completed = pass - options.warmup_passes + 1U;
        }
      }
      return true;
    }
    case ScenarioKind::incompressible_read:
      for (std::uint32_t pass = 0; pass < options.passes; ++pass) {
        if (!scan(pass, residency::AccessMode::read)) {
          return false;
        }
      }
      return true;
    case ScenarioKind::reuse:
      if (!scan(0U, residency::AccessMode::read)) {
        return false;
      }
      for (std::uint32_t cycle = 0; cycle < 7U; ++cycle) {
        if (!scan(cycle, residency::AccessMode::read, hot_chunks)) {
          return false;
        }
      }
      return true;
    case ScenarioKind::dirty_writeback:
      for (std::uint32_t pass = 0; pass < options.passes; ++pass) {
        if (!scan(pass, residency::AccessMode::read_write)) {
          return false;
        }
      }
      return true;
    case ScenarioKind::mixed:
      for (std::uint32_t pass = 0; pass < options.passes; ++pass) {
        if (pass == 1U) {
          // Accumulate real cache-hit history before changing the authoritative content. The
          // cost model keeps its EWMA across generations, so the following host rewrite can make
          // a time-based adaptive decision over the same reuse horizon instead of behaving like
          // a first-touch compression probe.
          for (std::uint64_t chunk = 0; chunk < chunk_count; ++chunk) {
            if (!detail::mixed_chunk_needs_history(chunk)) {
              continue;
            }
            for (std::uint32_t cycle = 0; cycle < detail::mixed_history_cycles; ++cycle) {
              if (!run_transaction(chunk, residency::AccessMode::read, false)) {
                return false;
              }
            }
          }
          status = runtime.drain(true);
          if (status != residency::RuntimeStatus::success) {
            return false;
          }
          for (std::uint64_t chunk = 0; chunk < chunk_count; ++chunk) {
            const std::uint64_t offset = chunk * chunk_bytes;
            const std::uint64_t valid = std::min(chunk_bytes, logical_bytes - offset);
            const auto valid_buffer =
                std::span<std::byte>{buffer.data(), static_cast<std::size_t>(valid)};
            fill_pattern(spec.scenario, options.seed, offset, chunk_bytes, valid_buffer, true);
            const std::optional<residency::HostChunkInfo> backing_before =
                runtime.backing_info(allocation.id, chunk);
            status = runtime.write(allocation.id, offset, buffer.data(), valid);
            if (status != residency::RuntimeStatus::success) {
              return false;
            }
            ++operations_completed;
            const std::optional<residency::HostChunkInfo> backing_after =
                runtime.backing_info(allocation.id, chunk);
            emit_trace(trace, trace_sequence, "generation_commit", allocation, chunk,
                       ++operation_id, spec.path, valid,
                       backing_after.has_value()
                           ? std::optional<std::uint64_t>{backing_after->stored_payload_bytes}
                           : std::nullopt,
                       "mixed_content_phase_change", backing_before, backing_after);
            report_progress();
          }
          mixed_phase_changed = true;
        }
        for (std::uint64_t position = 0; position < chunk_count; ++position) {
          const std::uint64_t chunk = (pass & 1U) == 0U ? position : chunk_count - position - 1U;
          // Keep the chunks whose next generation will use the reuse forecast clean during the
          // first pass. Their phase rewrite can then exercise the host-side adaptive decision
          // directly; dirty traffic remains on the other two classes and still covers event-safe
          // write-back without conflating first-use nvCOMP setup with the history decision.
          const residency::AccessMode mode = detail::mixed_chunk_is_write_capable(chunk, pass)
                                                 ? residency::AccessMode::read_write
                                                 : residency::AccessMode::read;
          if (!run_transaction(chunk, mode, true)) {
            return false;
          }
        }
      }
      return true;
    case ScenarioKind::budget_pressure: {
      if (!scan(0U, residency::AccessMode::read)) {
        return false;
      }
      const std::uint64_t pressure_bytes = saturating_multiply(chunk_bytes, 4U);
      if (pressure_bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
          api.mem_alloc_ == nullptr ||
          api.mem_alloc_(&pressure_allocation, static_cast<std::size_t>(pressure_bytes)) !=
              cuda::abi::success) {
        status = residency::RuntimeStatus::device_oom;
        return false;
      }
      std::this_thread::sleep_for(options.budget_poll_interval);
      if (!scan(1U, residency::AccessMode::read)) {
        return false;
      }
      release_pressure();
      for (std::uint32_t safe_sample = 0; safe_sample < 10U; ++safe_sample) {
        std::this_thread::sleep_for(options.budget_poll_interval);
        if (!run_transaction(static_cast<std::uint64_t>(safe_sample) % chunk_count,
                             residency::AccessMode::read, false)) {
          return false;
        }
      }
      return true;
    }
    case ScenarioKind::suite:
      return false;
    }
    return false;
  };

  const Clock::time_point execution_begin = Clock::now();
  if (status == residency::RuntimeStatus::success && !execute_scenario()) {
    set_failure(result, runtime, status, spec.scenario);
  }
  release_pressure();
  if (status == residency::RuntimeStatus::success) {
    status = runtime.drain(true);
    if (status != residency::RuntimeStatus::success) {
      set_failure(result, runtime, status, spec.scenario);
    }
  }

  Digest128 actual_digest;
  std::uint64_t mismatch_count = 0;
  std::optional<std::uint64_t> first_mismatch;
  if (status == residency::RuntimeStatus::success) {
    if (mixed_phase_changed) {
      expected_digest = Digest128{};
      for (std::uint64_t chunk = 0; chunk < chunk_count; ++chunk) {
        const std::uint64_t offset = chunk * chunk_bytes;
        const std::uint64_t valid = std::min(chunk_bytes, logical_bytes - offset);
        auto expected_span = std::span<std::byte>{expected.data(), static_cast<std::size_t>(valid)};
        fill_pattern(spec.scenario, options.seed, offset, chunk_bytes, expected_span, true);
        expected_digest.update(expected_span);
      }
    }
    for (std::uint64_t chunk = 0; chunk < chunk_count; ++chunk) {
      const std::uint64_t offset = chunk * chunk_bytes;
      const std::uint64_t valid = std::min(chunk_bytes, logical_bytes - offset);
      status = runtime.read(allocation.id, offset, buffer.data(), valid);
      if (status != residency::RuntimeStatus::success) {
        set_failure(result, runtime, status, spec.scenario, chunk);
        break;
      }
      const auto valid_buffer =
          std::span<const std::byte>{buffer.data(), static_cast<std::size_t>(valid)};
      actual_digest.update(valid_buffer);
      auto expected_span = std::span<std::byte>{expected.data(), static_cast<std::size_t>(valid)};
      fill_pattern(spec.scenario, options.seed, offset, chunk_bytes, expected_span,
                   mixed_phase_changed);
      if (std::memcmp(buffer.data(), expected.data(), static_cast<std::size_t>(valid)) != 0) {
        for (std::uint64_t index = 0; index < valid; ++index) {
          if (buffer[static_cast<std::size_t>(index)] !=
              expected[static_cast<std::size_t>(index)]) {
            ++mismatch_count;
            if (!first_mismatch.has_value()) {
              first_mismatch = offset + index;
            }
          }
        }
      }
      ++operations_completed;
      emit_trace(trace, trace_sequence, "operation_retired", allocation, chunk, ++operation_id,
                 spec.path, valid, std::nullopt, "cpu_reference");
      report_progress();
    }
  }
  const Clock::time_point execution_end = Clock::now();

  workload.expected_digest128 = expected_digest.hex();
  workload.output_digest128 = actual_digest.hex();
  workload.mismatch_count = mismatch_count;
  workload.first_mismatch_byte_offset = first_mismatch;
  workload.elapsed_ms =
      std::chrono::duration<double, std::milli>(execution_end - execution_begin).count();

  // Capture storage before release removes the authoritative generations. Peak values remain
  // available after close and are used for aggregate budget proof.
  const residency::RuntimeTelemetry before_release = runtime.telemetry();
  result.storage_telemetry = before_release;
  workload.stored_bytes = before_release.host_stored_bytes;
  workload.stored_ratio =
      logical_bytes == 0U
          ? std::optional<double>{}
          : std::optional<double>{static_cast<double>(before_release.host_stored_bytes) /
                                  static_cast<double>(logical_bytes)};

  const residency::RuntimeStatus release_status = runtime.release(allocation.id);
  const residency::RuntimeStatus close_status = runtime.close();
  capture_cleanup_evidence(result, runtime, spec.path,
                           release_status == residency::RuntimeStatus::success &&
                               close_status == residency::RuntimeStatus::success);
  const residency::RuntimeTelemetry& telemetry = result.telemetry;
  workload.logical_h2d_bytes = telemetry.logical_h2d_bytes;
  workload.pcie_h2d_bytes = telemetry.pcie_h2d_bytes;
  workload.logical_d2h_bytes = telemetry.logical_d2h_bytes;
  workload.pcie_d2h_bytes = telemetry.pcie_d2h_bytes;
  workload.raw_path_decisions = telemetry.raw_path_decisions;
  workload.cpu_lz4_gpu_decisions = telemetry.cpu_lz4_gpu_decode_decisions;
  workload.gpu_lz4_decisions = telemetry.gpu_lz4_decisions;
  workload.never_compress_decisions = telemetry.never_compress_decisions;
  workload.fallback_count =
      saturating_add(telemetry.raw_fallbacks,
                     saturating_add(telemetry.cpu_codec_fallbacks, telemetry.gpu_codec_fallbacks));
  workload.mappings = telemetry.maps;
  workload.set_access = telemetry.set_access;
  workload.unmaps = telemetry.unmaps;
  workload.events_recorded =
      saturating_add(telemetry.event_boundaries, telemetry.codec_events_recorded);
  workload.events_retired =
      saturating_add(telemetry.event_boundaries, telemetry.codec_events_retired);
  workload.unsafe_remaps = telemetry.unsafe_remaps;
  workload.unsafe_transitions = telemetry.unsafe_transitions;
  workload.operations_retired = operations_completed;
  if (spec.path == RequestedPath::automatic) {
    workload.path = telemetry.gpu_lz4_decisions != 0U              ? "nvcomp_gpu_codec"
                    : telemetry.cpu_lz4_gpu_decode_decisions != 0U ? "cpu_lz4_gpu_decode"
                                                                   : "raw";
  }

  if (status == residency::RuntimeStatus::success && mismatch_count != 0U) {
    result.exit_code = exit_corruption;
    result.reason = "data_mismatch";
    result.failure = ExecutionFailure{"verification",
                                      "compare_reference",
                                      "resident result differs from deterministic host reference",
                                      std::nullopt,
                                      std::nullopt,
                                      workload.scenario,
                                      allocation.id.value,
                                      std::nullopt,
                                      std::nullopt,
                                      first_mismatch};
    workload.status = "failed";
  } else if (status == residency::RuntimeStatus::success && !result.close_complete) {
    result.exit_code = exit_failure;
    result.reason = "cleanup_failure";
    result.failure =
        ExecutionFailure{"cleanup", "runtime_close", "runtime cleanup did not complete"};
    workload.status = "failed";
  } else if (status == residency::RuntimeStatus::success && spec.path == RequestedPath::gpu_lz4 &&
             telemetry.gpu_lz4_decisions == 0U) {
    result.exit_code = exit_prerequisite;
    result.reason = "gpu_codec_unavailable";
    result.failure =
        ExecutionFailure{"codec", "force_gpu_lz4",
                         "the forced nvCOMP GPU path did not retire a GPU codec operation"};
    workload.status = "not_run";
  } else if (status == residency::RuntimeStatus::success &&
             spec.scenario == ScenarioKind::budget_pressure &&
             (telemetry.budget_shrinks == 0U || telemetry.budget_grows == 0U)) {
    result.exit_code = exit_failure;
    result.reason = "budget_hysteresis_not_observed";
    result.failure = ExecutionFailure{
        "budget", "shrink_grow",
        "controlled pressure did not produce both an event-safe shrink and hysteretic grow"};
    workload.status = "failed";
  } else if (status == residency::RuntimeStatus::success) {
    result.exit_code = exit_completed;
    result.reason.reset();
    workload.status = "completed";
  } else if (!result.failure.has_value()) {
    set_failure(result, runtime, status, spec.scenario);
  }
  return result;
}

} // namespace

namespace detail {

bool deliver_trace_record(const TraceCallback& callback, const TraceRecord& record) noexcept {
  try {
    if (!callback) {
      return false;
    }
    callback(record);
    return true;
  } catch (...) {
    return false;
  }
}

void apply_trace_delivery_result(ExecutorResult& result, const std::uint64_t produced,
                                 const std::uint64_t dropped) {
  result.telemetry.trace_records_emitted = produced >= dropped ? produced - dropped : 0U;
  result.telemetry.trace_records_dropped = dropped;
  result.telemetry.trace_complete = dropped == 0U;
  if (result.status == "completed" && dropped != 0U) {
    result.exit_code = exit_failure;
    result.status = "failed";
    result.reason = "trace_delivery_failure";
    result.failure = ExecutionFailure{
        "trace", "trace_callback", "one or more compression trace records could not be delivered"};
  }
}

bool mixed_chunk_is_compressible(const std::uint64_t chunk_index,
                                 const bool phase_changed) noexcept {
  switch (chunk_index & 3U) {
  case 0U:
    return true;
  case 1U:
    return false;
  case 2U:
    return !phase_changed;
  case 3U:
    return phase_changed;
  }
  return false;
}

bool mixed_chunk_needs_history(const std::uint64_t chunk_index) noexcept {
  // Build reuse history for chunks that are compression candidates after the phase rewrite. The
  // stable-compressible class proves history reuse, while class 3 proves raw-to-compressed change.
  return mixed_chunk_is_compressible(chunk_index, true);
}

bool mixed_chunk_is_write_capable(const std::uint64_t chunk_index,
                                  const std::uint32_t pass) noexcept {
  const bool history_candidate = pass == 0U && mixed_chunk_needs_history(chunk_index);
  return !history_candidate && ((chunk_index + pass) % 3U) == 0U;
}

Cleanup derive_runtime_cleanup(const residency::RuntimeTelemetry& telemetry,
                               const bool close_complete) noexcept {
  Cleanup cleanup;
  cleanup.operations_drained = telemetry.transactions_submitted == telemetry.transactions_completed;
  cleanup.codec_slots_drained = telemetry.codec_events_recorded == telemetry.codec_events_retired &&
                                telemetry.codec_slot_bytes == 0U;
  cleanup.events_drained = cleanup.operations_drained.value_or(false) &&
                           telemetry.codec_events_recorded == telemetry.codec_events_retired;
  cleanup.mappings_removed = telemetry.maps == telemetry.unmaps && telemetry.resident_bytes == 0U;
  cleanup.physical_handles_released = telemetry.handles_created == telemetry.handles_released;
  cleanup.codec_workspace_released = telemetry.codec_workspace_bytes == 0U;
  cleanup.virtual_reservations_released =
      telemetry.allocations_created == telemetry.allocations_released &&
      cleanup.mappings_removed.value_or(false);
  cleanup.pinned_staging_released = telemetry.host_budget_bytes == 0U;
  cleanup.spill_reservations_released = telemetry.spill_reserved_bytes == 0U;
  cleanup.host_backing_released =
      telemetry.allocations_created == telemetry.allocations_released &&
      telemetry.logical_bytes == 0U && telemetry.host_stored_bytes == 0U &&
      telemetry.host_raw_bytes == 0U && telemetry.host_compressed_bytes == 0U &&
      telemetry.host_authoritative_bytes == 0U && telemetry.host_invalid_chunks == 0U &&
      telemetry.host_implicit_zero_chunks == 0U && telemetry.host_raw_chunks == 0U &&
      telemetry.host_lz4_chunks == 0U;

  // RuntimeTelemetry does not yet expose individual destroy counters for these resources. A
  // successful close is nevertheless direct evidence here: Runtime::close reports success only
  // after all runtime events/streams and the owned context have been released. Keep these
  // assignments explicit so they cannot be confused with the counter/current-byte-derived stages
  // above.
  cleanup.events_destroyed = close_complete;
  cleanup.streams_destroyed = close_complete;
  cleanup.context_released = close_complete;
  cleanup.complete = close_complete && cleanup.operations_drained.value_or(false) &&
                     cleanup.codec_slots_drained.value_or(false) &&
                     cleanup.events_drained.value_or(false) &&
                     cleanup.mappings_removed.value_or(false) &&
                     cleanup.physical_handles_released.value_or(false) &&
                     cleanup.codec_workspace_released.value_or(false) &&
                     cleanup.virtual_reservations_released.value_or(false) &&
                     cleanup.pinned_staging_released.value_or(false) &&
                     cleanup.spill_reservations_released.value_or(false) &&
                     cleanup.host_backing_released.value_or(false);
  return cleanup;
}

bool derive_write_admission_proof(const residency::RuntimeTelemetry& telemetry,
                                  const RequestedPath requested_path) noexcept {
  const bool writeback_observed =
      telemetry.dirty_writebacks != 0U || telemetry.logical_d2h_bytes != 0U;
  if (!writeback_observed || requested_path == RequestedPath::raw) {
    return true;
  }
  return telemetry.spill_reserved_peak_bytes != 0U && telemetry.spill_reserved_bytes == 0U;
}

void fill_workload_pattern(const ScenarioKind scenario, const std::uint64_t seed,
                           const std::uint64_t absolute_offset, const std::uint64_t chunk_bytes,
                           const std::span<std::byte> output, const bool phase_changed) noexcept {
  fill_pattern(scenario, seed, absolute_offset, chunk_bytes, output, phase_changed);
}

} // namespace detail

ExecutorResult run_executor(cuda::CudaApi& api, const ExecutorOptions& options,
                            const ProgressCallback& progress, const TraceCallback& trace) {
  ExecutorResult result;
  result.configuration.requested_device_ordinal = options.device_ordinal;
  result.configuration.requested_logical_bytes = options.logical_bytes;
  result.configuration.requested_chunk_bytes = options.chunk_bytes;
  result.configuration.requested_cache_target_bytes = options.cache_target_bytes;
  result.configuration.compression_policy =
      std::string(compression_policy_name(options.compression_policy));
  result.configuration.path = std::string(path_name(options.path));
  result.configuration.codec = std::string(residency::compression_codec_name(options.codec));
  result.configuration.replacement_policy =
      std::string(replacement_policy_name(options.replacement_policy));
  result.configuration.scenario = std::string(scenario_name(options.scenario));
  result.configuration.passes = options.passes;
  result.configuration.warmup_passes = options.warmup_passes;
  result.configuration.measurement_passes = options.measurement_passes;
  result.configuration.host_headroom_bytes = options.host_headroom_bytes.value_or(0U);
  result.configuration.device_headroom_bytes = options.device_headroom_bytes;
  result.configuration.compression_scratch_cap_bytes = options.compression_scratch_cap_bytes;
  result.configuration.codec_slots = options.codec_slots;
  result.configuration.codec_workers = options.codec_workers;
  result.configuration.staging_slots = options.staging_slots;
  result.configuration.prefetch_distance = options.prefetch_distance;
  result.configuration.budget_poll_ms =
      static_cast<std::uint64_t>(options.budget_poll_interval.count());
  result.configuration.stall_timeout_ms = static_cast<std::uint64_t>(options.stall_timeout.count());
  result.configuration.seed_hex = [&]() {
    std::ostringstream output;
    output << std::hex << std::nouppercase << std::setfill('0') << std::setw(16) << options.seed;
    return output.str();
  }();
  result.configuration.sizing_mode = options.logical_bytes.has_value() ? "explicit" : "auto";
  result.configuration.trace_enabled = options.trace_enabled;
  result.configuration.identifiers_included = options.include_identifiers;
  result.codec.cpu_codec_version = "1.10.0";

  const auto finish_early = [&](const int code, std::string reason, std::string stage,
                                std::string operation, std::string message,
                                const std::optional<std::int64_t> native = std::nullopt,
                                const std::optional<std::string> native_name = std::nullopt) {
    result.exit_code = code;
    result.status = code == exit_prerequisite ? "skipped" : "failed";
    result.reason = std::move(reason);
    result.failure = ExecutionFailure{std::move(stage), std::move(operation), std::move(message),
                                      native, std::move(native_name)};
    result.cleanup.worker_terminated = true;
    result.cleanup.trace_closed = true;
    return result;
  };

  const auto load = api.load();
  if (load.status != cuda::CudaApi::LoadStatus::loaded) {
    return finish_early(exit_prerequisite, "device_unavailable", "preflight", "load_cuda_driver",
                        api.error().empty() ? "CUDA Driver API is unavailable" : api.error());
  }
  if (api.init_ == nullptr || api.device_get_count_ == nullptr || api.device_get_ == nullptr ||
      api.device_get_name_ == nullptr || api.device_total_memory_ == nullptr ||
      api.device_get_attribute_ == nullptr) {
    return finish_early(exit_prerequisite, "cuda_symbols_unavailable", "preflight",
                        "resolve_cuda_symbols", "CUDA driver lacks required VMM symbols");
  }
  cuda::abi::Result cuda_status = api.init_(0U);
  if (cuda_status != cuda::abi::success) {
    return finish_early(exit_prerequisite, "cuda_initialization_failed", "preflight", "cuInit",
                        cuda_message(api, cuda_status), static_cast<std::int64_t>(cuda_status),
                        cuda_name(api, cuda_status));
  }
  int device_count = 0;
  cuda_status = api.device_get_count_(&device_count);
  if (cuda_status != cuda::abi::success || options.device_ordinal >= device_count) {
    return finish_early(exit_prerequisite, "device_unavailable", "preflight", "select_device",
                        "requested CUDA device ordinal is unavailable",
                        cuda_status == cuda::abi::success
                            ? std::optional<std::int64_t>{}
                            : std::optional<std::int64_t>{static_cast<std::int64_t>(cuda_status)},
                        cuda_status == cuda::abi::success
                            ? std::optional<std::string>{}
                            : std::optional<std::string>{cuda_name(api, cuda_status)});
  }
  cuda::abi::Device device = 0;
  cuda_status = api.device_get_(&device, options.device_ordinal);
  if (cuda_status != cuda::abi::success) {
    return finish_early(exit_prerequisite, "device_unavailable", "preflight", "cuDeviceGet",
                        cuda_message(api, cuda_status), static_cast<std::int64_t>(cuda_status),
                        cuda_name(api, cuda_status));
  }

  DeviceInfo device_info;
  device_info.ordinal = options.device_ordinal;
  std::array<char, 256> device_name{};
  std::size_t total_memory = 0;
  if (api.device_get_name_(device_name.data(), static_cast<int>(device_name.size()), device) !=
          cuda::abi::success ||
      api.device_total_memory_(&total_memory, device) != cuda::abi::success) {
    return finish_early(exit_failure, "cuda_error", "preflight", "query_device",
                        "CUDA device identity query failed");
  }
  device_info.name = device_name.data();
  device_info.total_memory_bytes = static_cast<std::uint64_t>(total_memory);
  std::uint32_t vmm = 0;
  std::uint32_t uva = 0;
  std::uint32_t major = 0;
  std::uint32_t minor = 0;
  if (!get_attribute(api, device, cuda::abi::attributes::virtual_memory_management_supported,
                     vmm) ||
      !get_attribute(api, device, cuda::abi::attributes::unified_addressing, uva) ||
      !get_attribute(api, device, cuda::abi::attributes::compute_capability_major, major) ||
      !get_attribute(api, device, cuda::abi::attributes::compute_capability_minor, minor)) {
    return finish_early(exit_failure, "cuda_error", "preflight", "query_capabilities",
                        "CUDA capability query failed");
  }
  device_info.vmm_supported = vmm != 0U;
  device_info.uva_supported = uva != 0U;
  device_info.compute_capability_major = major;
  device_info.compute_capability_minor = minor;
  if (!device_info.vmm_supported || !device_info.uva_supported) {
    result.device = device_info;
    return finish_early(exit_prerequisite, "vmm_or_uva_unsupported", "preflight", "capabilities",
                        "selected CUDA device lacks VMM or UVA");
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
  result.device = device_info;

  const probe::SystemInfo system = platform::collect_system_info();
  const std::uint64_t physical_host = system.physical_memory_bytes.value_or(0U);
  const std::uint64_t available_host = system.available_memory_bytes.value_or(0U);
  const std::uint64_t automatic_headroom =
      std::max(4U * gibibyte, physical_host == 0U ? 4U * gibibyte : physical_host / 4U);
  const std::uint64_t host_headroom = options.host_headroom_bytes.value_or(automatic_headroom);
  result.configuration.host_headroom_bytes = host_headroom;
  const std::uint64_t pinned_reserve =
      saturating_multiply(options.chunk_bytes, options.staging_slots);
  const std::uint64_t fixed_host_reserve = saturating_add(
      host_headroom,
      saturating_add(pinned_reserve, saturating_add(options.compression_scratch_cap_bytes,
                                                    default_service_reserve)));
  const std::uint64_t safe_host_limit =
      available_host > fixed_host_reserve ? available_host - fixed_host_reserve : 0U;
  const std::uint64_t default_logical =
      std::min(saturating_multiply(device_info.total_memory_bytes, 3U) / 2U, safe_host_limit);
  const std::uint64_t logical_bytes = options.logical_bytes.value_or(default_logical);
  if (logical_bytes == 0U || logical_bytes <= device_info.total_memory_bytes) {
    return finish_early(exit_prerequisite, "insufficient_host_memory", "preflight", "logical_size",
                        options.logical_bytes.has_value()
                            ? "compression proof requires logical data larger than physical VRAM"
                            : "safe host budget cannot hold the default oversubscribed workload");
  }
  const std::uint64_t automatic_host_cap =
      available_host > host_headroom ? available_host - host_headroom : 0U;
  const std::uint64_t host_store_cap = options.host_store_cap_bytes.value_or(automatic_host_cap);
  if (host_store_cap == 0U) {
    return finish_early(exit_prerequisite, "insufficient_host_memory", "preflight",
                        "host_store_cap", "host storage budget is zero after headroom");
  }
  result.configuration.effective_logical_bytes = logical_bytes;
  result.configuration.effective_chunk_bytes.reset();
  result.configuration.host_store_cap_bytes = host_store_cap;
  result.configuration.initial_cache_target_bytes = options.cache_target_bytes;
  result.backing.logical_bytes = logical_bytes;
  result.backing.host_store_cap_bytes = host_store_cap;

  std::vector<CaseSpec> cases;
  try {
    for (const residency::CompressionMode mode :
         expand_compression_modes(options.compression_policy)) {
      for (const RequestedPath path : expand_paths(options.path)) {
        for (const residency::RuntimePolicy policy : expand_policies(options.replacement_policy)) {
          for (const ScenarioKind scenario : expand_scenarios(options.scenario)) {
            cases.push_back(CaseSpec{mode, path, policy, scenario});
          }
        }
      }
    }
  } catch (const std::bad_alloc&) {
    return finish_early(exit_oom, "host_oom", "planning", "expand_cases",
                        "benchmark case planning allocation failed");
  }

  std::uint64_t trace_sequence = 0;
  std::uint64_t trace_records_dropped = 0;
  TraceCallback safe_trace;
  if (options.trace_enabled) {
    safe_trace = [&](const TraceRecord& record) noexcept {
      if (!detail::deliver_trace_record(trace, record)) {
        ++trace_records_dropped;
      }
    };
  }
  std::uint64_t total_completed = 0;
  std::uint64_t total_planned = 0;
  // The exact aligned chunk count is known inside each Runtime. This conservative count is used
  // only for human progress; the watchdog is driven by every retired operation.
  const std::uint64_t nominal_chunks = ((logical_bytes - 1U) / options.chunk_bytes) + 1U;
  for (const CaseSpec& spec : cases) {
    total_planned = saturating_add(
        total_planned, planned_operation_count(spec.scenario, options, nominal_chunks,
                                               std::max<std::uint64_t>(1U, nominal_chunks / 4U),
                                               spec.path == RequestedPath::gpu_lz4));
  }

  bool all_complete = true;
  bool all_cleanup = true;
  bool all_write_admission = true;
  bool all_gpu_encode_order = true;
  Cleanup cleanup_evidence;
  bool all_stable_addresses = true;
  bool any_nvcomp_available = false;
  bool all_nvcomp_app_local = true;
  bool all_nvcomp_identity_verified = true;
  std::optional<std::string> verified_nvcomp_version;
  std::optional<std::string> verified_nvcomp_sha256;
  bool have_backing_snapshot = false;
  std::uint64_t backing_snapshot_compressed_bytes = 0;
  std::uint64_t backing_snapshot_stored_bytes = 0;
  std::vector<double> raw_transfer_samples;
  std::vector<double> compressed_transfer_samples;
  for (const CaseSpec& spec : cases) {
    const std::uint64_t case_base = total_completed;
    std::uint64_t case_last = 0;
    const CaseProgress case_progress = [&](const WorkloadResult& partial,
                                           const std::uint64_t completed,
                                           const std::uint64_t total) {
      case_last = completed;
      ExecutorResult snapshot = result;
      snapshot.status = "running";
      snapshot.exit_code = exit_failure;
      snapshot.workloads.push_back(partial);
      if (progress) {
        progress(snapshot, saturating_add(case_base, completed),
                 std::max(total_planned, saturating_add(case_base, total)));
      }
    };
    CaseResult case_result = run_case(api, options, spec, logical_bytes, host_store_cap,
                                      case_progress, safe_trace, trace_sequence);
    if (case_result.effective_chunk_bytes != 0U) {
      if (!result.configuration.effective_chunk_bytes.has_value()) {
        result.configuration.effective_chunk_bytes = case_result.effective_chunk_bytes;
      } else if (*result.configuration.effective_chunk_bytes != case_result.effective_chunk_bytes) {
        all_complete = false;
        result.exit_code = exit_failure;
        result.status = "failed";
        result.reason = "inconsistent_chunk_granularity";
        result.failure = ExecutionFailure{
            "planning", "effective_chunk_bytes",
            "independent benchmark cases resolved different VMM chunk granularities"};
      }
    }
    std::vector<double>& transfer_samples =
        spec.path == RequestedPath::raw ? raw_transfer_samples : compressed_transfer_samples;
    transfer_samples.insert(transfer_samples.end(), case_result.measurement_samples_ms.begin(),
                            case_result.measurement_samples_ms.end());
    total_completed = saturating_add(total_completed, case_last);
    result.workloads.push_back(case_result.workload);
    merge_cleanup_evidence(cleanup_evidence, case_result.cleanup);
    all_cleanup = all_cleanup && case_result.cleanup.complete.value_or(false);
    all_write_admission = all_write_admission && case_result.write_admission_verified;
    all_gpu_encode_order = all_gpu_encode_order && case_result.gpu_encode_before_d2h_verified;
    if (case_result.exit_code != exit_completed) {
      all_complete = false;
      result.exit_code = case_result.exit_code;
      result.status = case_result.exit_code == exit_prerequisite ? "skipped" : "failed";
      result.reason = case_result.reason;
      result.failure = case_result.failure;
    } else if (!case_result.cleanup.complete.value_or(false)) {
      all_complete = false;
      result.exit_code = exit_failure;
      result.status = "failed";
      result.reason = "cleanup_telemetry_mismatch";
      result.failure = ExecutionFailure{
          "cleanup", "reconcile_runtime_telemetry",
          "post-close runtime counters or current resource bytes did not reconcile"};
    }

    const residency::RuntimeTelemetry& telemetry = case_result.telemetry;
    any_nvcomp_available = any_nvcomp_available || telemetry.nvcomp_available;
    all_nvcomp_app_local =
        all_nvcomp_app_local && (!telemetry.nvcomp_available || telemetry.nvcomp_app_local);
    if (telemetry.nvcomp_available) {
      const bool identity_present =
          !telemetry.nvcomp_version.empty() && telemetry.nvcomp_library_sha256.size() == 64U;
      all_nvcomp_identity_verified = all_nvcomp_identity_verified && identity_present;
      if (identity_present) {
        if (!verified_nvcomp_version.has_value()) {
          verified_nvcomp_version = telemetry.nvcomp_version;
          verified_nvcomp_sha256 = telemetry.nvcomp_library_sha256;
        } else {
          all_nvcomp_identity_verified = all_nvcomp_identity_verified &&
                                         *verified_nvcomp_version == telemetry.nvcomp_version &&
                                         *verified_nvcomp_sha256 == telemetry.nvcomp_library_sha256;
        }
      }
    }
    all_stable_addresses = all_stable_addresses && telemetry.stable_addresses &&
                           telemetry.no_physical_aliases && !telemetry.quarantined;
    result.telemetry.logical_h2d_bytes = saturating_add(
        result.telemetry.logical_h2d_bytes.value_or(0U), telemetry.logical_h2d_bytes);
    result.telemetry.pcie_h2d_bytes =
        saturating_add(result.telemetry.pcie_h2d_bytes.value_or(0U), telemetry.pcie_h2d_bytes);
    result.telemetry.pcie_h2d_payload_bytes = saturating_add(
        result.telemetry.pcie_h2d_payload_bytes.value_or(0U), telemetry.pcie_h2d_payload_bytes);
    result.telemetry.pcie_h2d_metadata_bytes = saturating_add(
        result.telemetry.pcie_h2d_metadata_bytes.value_or(0U), telemetry.pcie_h2d_metadata_bytes);
    result.telemetry.logical_d2h_bytes = saturating_add(
        result.telemetry.logical_d2h_bytes.value_or(0U), telemetry.logical_d2h_bytes);
    result.telemetry.pcie_d2h_bytes =
        saturating_add(result.telemetry.pcie_d2h_bytes.value_or(0U), telemetry.pcie_d2h_bytes);
    result.telemetry.pcie_d2h_payload_bytes = saturating_add(
        result.telemetry.pcie_d2h_payload_bytes.value_or(0U), telemetry.pcie_d2h_payload_bytes);
    result.telemetry.pcie_d2h_metadata_bytes = saturating_add(
        result.telemetry.pcie_d2h_metadata_bytes.value_or(0U), telemetry.pcie_d2h_metadata_bytes);
    result.telemetry.rejected_candidate_logical_d2h_bytes =
        saturating_add(result.telemetry.rejected_candidate_logical_d2h_bytes.value_or(0U),
                       telemetry.rejected_candidate_logical_d2h_bytes);
    result.telemetry.mapping_count =
        saturating_add(result.telemetry.mapping_count.value_or(0U), telemetry.maps);
    result.telemetry.set_access_count =
        saturating_add(result.telemetry.set_access_count.value_or(0U), telemetry.set_access);
    result.telemetry.unmap_count =
        saturating_add(result.telemetry.unmap_count.value_or(0U), telemetry.unmaps);
    result.telemetry.event_record_count =
        saturating_add(result.telemetry.event_record_count.value_or(0U),
                       saturating_add(telemetry.event_boundaries, telemetry.codec_events_recorded));
    result.telemetry.event_retire_count =
        saturating_add(result.telemetry.event_retire_count.value_or(0U),
                       saturating_add(telemetry.event_boundaries, telemetry.codec_events_retired));
    result.telemetry.handle_reuse_count =
        saturating_add(result.telemetry.handle_reuse_count.value_or(0U), telemetry.handles_reused);
    result.telemetry.unsafe_remap_count =
        saturating_add(result.telemetry.unsafe_remap_count.value_or(0U), telemetry.unsafe_remaps);
    result.telemetry.unsafe_transition_count = saturating_add(
        result.telemetry.unsafe_transition_count.value_or(0U), telemetry.unsafe_transitions);
    result.telemetry.writeback_count =
        saturating_add(result.telemetry.writeback_count.value_or(0U), telemetry.dirty_writebacks);
    result.telemetry.target_shrink_count =
        saturating_add(result.telemetry.target_shrink_count.value_or(0U), telemetry.budget_shrinks);
    result.telemetry.target_grow_count =
        saturating_add(result.telemetry.target_grow_count.value_or(0U), telemetry.budget_grows);
    result.telemetry.budget_sample_count =
        saturating_add(result.telemetry.budget_sample_count.value_or(0U), telemetry.budget_samples);
    result.telemetry.cache_target_bytes_minimum =
        result.telemetry.cache_target_bytes_minimum.has_value()
            ? std::min(*result.telemetry.cache_target_bytes_minimum, telemetry.target_minimum_bytes)
            : std::optional<std::uint64_t>{telemetry.target_minimum_bytes};
    result.telemetry.cache_target_bytes_maximum = std::max(
        result.telemetry.cache_target_bytes_maximum.value_or(0U), telemetry.target_maximum_bytes);
    result.telemetry.safe_device_budget_bytes_minimum =
        result.telemetry.safe_device_budget_bytes_minimum.has_value()
            ? std::min(*result.telemetry.safe_device_budget_bytes_minimum,
                       telemetry.safe_device_budget_minimum_bytes)
            : std::optional<std::uint64_t>{telemetry.safe_device_budget_minimum_bytes};
    result.telemetry.managed_device_bytes_peak =
        std::max(result.telemetry.managed_device_bytes_peak.value_or(0U),
                 telemetry.managed_device_bytes_peak);
    result.telemetry.device_reserve_bytes_peak =
        std::max(result.telemetry.device_reserve_bytes_peak.value_or(0U),
                 telemetry.device_reserve_bytes_peak);
    result.telemetry.device_budget_violation_count =
        saturating_add(result.telemetry.device_budget_violation_count.value_or(0U),
                       telemetry.device_budget_violation_count);

    result.backing.host_bytes_peak =
        std::max(result.backing.host_bytes_peak.value_or(0U), telemetry.host_stored_peak_bytes);
    const std::uint64_t candidate_compressed_bytes =
        case_result.storage_telemetry.host_compressed_bytes;
    const std::uint64_t candidate_stored_bytes = case_result.storage_telemetry.host_stored_bytes;
    // "current" representation counters describe one coherent authoritative snapshot.  Cases
    // execute in separate runtimes, so independently taking the maximum of every representation
    // can manufacture a state where raw_bytes + compressed_bytes exceeds host_bytes.  Prefer a
    // compressed snapshot when one was observed (needed for the authority proof), then the largest
    // snapshot of the same kind.  Peak and cumulative counters remain aggregated below.
    const bool select_backing_snapshot =
        !have_backing_snapshot ||
        (candidate_compressed_bytes != 0U && backing_snapshot_compressed_bytes == 0U) ||
        ((candidate_compressed_bytes != 0U) == (backing_snapshot_compressed_bytes != 0U) &&
         (candidate_compressed_bytes > backing_snapshot_compressed_bytes ||
          (candidate_compressed_bytes == backing_snapshot_compressed_bytes &&
           candidate_stored_bytes > backing_snapshot_stored_bytes)));
    if (select_backing_snapshot) {
      have_backing_snapshot = true;
      backing_snapshot_compressed_bytes = candidate_compressed_bytes;
      backing_snapshot_stored_bytes = candidate_stored_bytes;
      result.backing.host_bytes_current = candidate_stored_bytes;
      result.backing.host_budget_bytes_current = case_result.storage_telemetry.host_budget_bytes;
      result.backing.raw_bytes_current = case_result.storage_telemetry.host_raw_bytes;
      result.backing.compressed_bytes_current = candidate_compressed_bytes;
      result.backing.invalid_chunks = case_result.storage_telemetry.host_invalid_chunks;
      result.backing.implicit_zero_chunks = case_result.storage_telemetry.host_implicit_zero_chunks;
      result.backing.raw_chunks = case_result.storage_telemetry.host_raw_chunks;
      result.backing.lz4_chunks = case_result.storage_telemetry.host_lz4_chunks;
    }
    result.backing.host_budget_bytes_peak = std::max(
        result.backing.host_budget_bytes_peak.value_or(0U), telemetry.host_budget_peak_bytes);
    result.backing.raw_bytes_peak =
        std::max(result.backing.raw_bytes_peak.value_or(0U), telemetry.host_raw_peak_bytes);
    result.backing.compressed_bytes_peak = std::max(
        result.backing.compressed_bytes_peak.value_or(0U), telemetry.host_compressed_peak_bytes);
    result.backing.spill_reserved_bytes_peak = std::max(
        result.backing.spill_reserved_bytes_peak.value_or(0U), telemetry.spill_reserved_peak_bytes);
    result.backing.conversion_scratch_bytes_peak =
        std::max(result.backing.conversion_scratch_bytes_peak.value_or(0U),
                 telemetry.conversion_scratch_peak_bytes);
    result.backing.generations_created = saturating_add(
        result.backing.generations_created.value_or(0U), telemetry.generations_created);
    result.backing.generations_committed = saturating_add(
        result.backing.generations_committed.value_or(0U), telemetry.generations_committed);
    result.backing.generations_discarded = saturating_add(
        result.backing.generations_discarded.value_or(0U), telemetry.generations_discarded);
    result.backing.atomic_commit_failures = saturating_add(
        result.backing.atomic_commit_failures.value_or(0U), telemetry.atomic_commit_failures);
    result.backing.expansion_rejections = saturating_add(
        result.backing.expansion_rejections.value_or(0U), telemetry.expansion_rejections);

    result.codec.cpu_encode_operations = saturating_add(
        result.codec.cpu_encode_operations.value_or(0U), telemetry.cpu_encode_operations);
    result.codec.cpu_decode_operations = saturating_add(
        result.codec.cpu_decode_operations.value_or(0U), telemetry.cpu_decode_operations);
    result.codec.gpu_encode_operations = saturating_add(
        result.codec.gpu_encode_operations.value_or(0U), telemetry.gpu_encode_operations);
    result.codec.gpu_decode_operations = saturating_add(
        result.codec.gpu_decode_operations.value_or(0U), telemetry.gpu_decode_operations);
    result.codec.raw_path_decisions =
        saturating_add(result.codec.raw_path_decisions.value_or(0U), telemetry.raw_path_decisions);
    result.codec.cpu_lz4_gpu_decisions = saturating_add(
        result.codec.cpu_lz4_gpu_decisions.value_or(0U), telemetry.cpu_lz4_gpu_decode_decisions);
    result.codec.gpu_lz4_decisions =
        saturating_add(result.codec.gpu_lz4_decisions.value_or(0U), telemetry.gpu_lz4_decisions);
    result.codec.never_compress_decisions = saturating_add(
        result.codec.never_compress_decisions.value_or(0U), telemetry.never_compress_decisions);
    result.codec.calibration_samples = saturating_add(result.codec.calibration_samples.value_or(0U),
                                                      telemetry.codec_calibration_samples);
    result.codec.fallback_count = saturating_add(
        result.codec.fallback_count.value_or(0U),
        saturating_add(telemetry.raw_fallbacks, saturating_add(telemetry.cpu_codec_fallbacks,
                                                               telemetry.gpu_codec_fallbacks)));
    result.codec.codec_slots_peak =
        std::max(result.codec.codec_slots_peak.value_or(0U), telemetry.codec_slots_peak);
    result.codec.workspace_bytes_peak = std::max(result.codec.workspace_bytes_peak.value_or(0U),
                                                 telemetry.codec_workspace_peak_bytes);
    result.codec.device_slot_bytes_peak =
        std::max(result.codec.device_slot_bytes_peak.value_or(0U), telemetry.codec_slot_peak_bytes);
    result.codec.device_slot_capacity_bytes = std::max(
        result.codec.device_slot_capacity_bytes.value_or(0U), telemetry.codec_slot_capacity_bytes);
    result.codec.verification_failures = saturating_add(
        result.codec.verification_failures.value_or(0U), telemetry.codec_verification_failures);
    add_timing_total(result.codec.cpu_encode_timing, telemetry.cpu_encode_operations,
                     telemetry.cpu_encode_nanoseconds);
    add_timing_total(result.codec.cpu_decode_timing, telemetry.cpu_decode_operations,
                     telemetry.cpu_decode_nanoseconds);
    add_timing_total(result.codec.gpu_encode_timing, telemetry.gpu_encode_operations,
                     telemetry.gpu_encode_nanoseconds);
    add_timing_total(result.codec.gpu_decode_timing, telemetry.gpu_decode_operations,
                     telemetry.gpu_decode_nanoseconds);
    add_timing_total(
        result.codec.verification_timing,
        saturating_add(telemetry.gpu_encode_operations, telemetry.gpu_decode_operations),
        telemetry.verification_nanoseconds);
    if (result.device.has_value()) {
      result.device->free_memory_bytes_start =
          result.device->free_memory_bytes_start.has_value()
              ? std::max(*result.device->free_memory_bytes_start, telemetry.cuda_free_minimum_bytes)
              : std::optional<std::uint64_t>{telemetry.cuda_free_minimum_bytes};
      result.device->free_memory_bytes_end = telemetry.cuda_free_end_bytes;
      result.device->safe_device_budget_bytes = std::max(
          result.device->safe_device_budget_bytes.value_or(0U), telemetry.target_maximum_bytes);
      if (telemetry.wddm_available_minimum_bytes.has_value()) {
        result.device->wddm_available_bytes_minimum =
            result.device->wddm_available_bytes_minimum.has_value()
                ? std::min(*result.device->wddm_available_bytes_minimum,
                           *telemetry.wddm_available_minimum_bytes)
                : telemetry.wddm_available_minimum_bytes;
      }
    }
  }

  result.backing.invalid_chunks = result.backing.invalid_chunks.value_or(0U);
  result.backing.implicit_zero_chunks = result.backing.implicit_zero_chunks.value_or(0U);
  result.backing.raw_chunks = result.backing.raw_chunks.value_or(0U);
  result.backing.lz4_chunks = result.backing.lz4_chunks.value_or(0U);
  result.backing.effective_stored_ratio =
      logical_bytes == 0U
          ? std::optional<double>{}
          : std::optional<double>{static_cast<double>(result.backing.host_bytes_peak.value_or(0U)) /
                                  static_cast<double>(logical_bytes)};
  result.codec.nvcomp_available = any_nvcomp_available;
  if (any_nvcomp_available) {
    result.codec.nvcomp_library_source = all_nvcomp_app_local ? "app_local" : "explicit";
    result.codec.nvcomp_version = verified_nvcomp_version;
    result.codec.nvcomp_library_sha256 = verified_nvcomp_sha256;
  } else {
    result.codec.nvcomp_version.reset();
    result.codec.nvcomp_library_source.reset();
    result.codec.nvcomp_library_sha256.reset();
  }
  if (any_nvcomp_available && !all_nvcomp_identity_verified && all_complete) {
    all_complete = false;
    result.exit_code = exit_failure;
    result.status = "failed";
    result.reason = "nvcomp_identity_mismatch";
    result.failure = ExecutionFailure{
        "codec", "reconcile_nvcomp_identity",
        "loaded nvCOMP instances did not expose one consistent verified version and SHA-256"};
  }
  if (!result.configuration.initial_cache_target_bytes.has_value()) {
    result.configuration.initial_cache_target_bytes = result.telemetry.cache_target_bytes_maximum;
  }
  double elapsed_total_ms = 0.0;
  for (const WorkloadResult& workload : result.workloads) {
    const double elapsed = workload.elapsed_ms.value_or(0.0);
    elapsed_total_ms += elapsed;
    if (workload.logical_d2h_bytes.value_or(0U) != 0U) {
      ++result.telemetry.writeback_timing.sample_count;
      result.telemetry.writeback_timing.total_ms =
          result.telemetry.writeback_timing.total_ms.value_or(0.0) + elapsed;
    }
  }
  result.telemetry.total_elapsed_ms = elapsed_total_ms;
  finish_timing_summary(result.codec.cpu_encode_timing);
  finish_timing_summary(result.codec.cpu_decode_timing);
  finish_timing_summary(result.codec.gpu_encode_timing);
  finish_timing_summary(result.codec.gpu_decode_timing);
  finish_timing_summary(result.codec.verification_timing);
  finish_timing_samples(result.telemetry.raw_transfer_timing, std::move(raw_transfer_samples));
  finish_timing_samples(result.telemetry.compressed_transfer_timing,
                        std::move(compressed_transfer_samples));
  finish_timing_summary(result.telemetry.writeback_timing);
  result.proof.stable_virtual_addresses_verified =
      all_stable_addresses && result.telemetry.unsafe_remap_count.value_or(1U) == 0U;
  result.proof.write_admission_verified = all_write_admission;
  result.proof.gpu_encode_before_d2h_verified = all_gpu_encode_order;

  result.cleanup = cleanup_evidence;
  result.cleanup.complete = all_cleanup;
  result.cleanup.trace_closed = true;
  result.cleanup.worker_terminated = true;

  if (all_complete) {
    result.exit_code = exit_completed;
    result.status = "completed";
    result.reason.reset();
    result.failure.reset();
  }
  detail::apply_trace_delivery_result(result, trace_sequence, trace_records_dropped);
  return result;
}

ExecutorResult run_executor(const ExecutorOptions& options, const ProgressCallback& progress,
                            const TraceCallback& trace) {
  cuda::CudaApi api;
  return run_executor(api, options, progress, trace);
}

} // namespace xvram::compression
