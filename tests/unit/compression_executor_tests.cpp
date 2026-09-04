#include "compression_bench/executor.hpp"
#include "residency/runtime.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;

void expect(const bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

[[nodiscard]] bool repeats_every(const std::span<const std::byte> bytes, const std::size_t period) {
  if (period == 0U || bytes.size() <= period) {
    return false;
  }
  for (std::size_t index = period; index < bytes.size(); ++index) {
    if (bytes[index] != bytes[index % period]) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] std::uint64_t digest(const std::span<const std::byte> bytes) noexcept {
  std::uint64_t value = 1469598103934665603ULL;
  for (const std::byte item : bytes) {
    value ^= std::to_integer<std::uint8_t>(item);
    value *= 1099511628211ULL;
  }
  return value;
}

void mixed_whole_chunk_classes_and_phase_switch_test() {
  using xvram::compression::ScenarioKind;
  using xvram::compression::detail::fill_workload_pattern;
  using xvram::compression::detail::mixed_chunk_is_compressible;
  using xvram::compression::detail::mixed_chunk_is_write_capable;
  using xvram::compression::detail::mixed_chunk_needs_history;

  constexpr std::uint64_t seed = 0x585652414D503035ULL;
  constexpr std::uint64_t chunk_bytes = 256U * 1024U;
  constexpr std::size_t sample_bytes = 128U * 1024U;
  std::vector<std::byte> initial(sample_bytes);
  std::vector<std::byte> repeated(sample_bytes);
  std::vector<std::byte> changed(sample_bytes);

  std::uint32_t initial_compressible = 0;
  std::uint32_t changed_compressible = 0;
  std::uint32_t switched_classes = 0;
  std::uint32_t history_classes = 0;
  for (std::uint64_t chunk = 0; chunk < 4U; ++chunk) {
    const bool before = mixed_chunk_is_compressible(chunk, false);
    const bool after = mixed_chunk_is_compressible(chunk, true);
    const bool expected_switch = (chunk & 2U) != 0U;
    expect((before != after) == expected_switch,
           "mixed stable and phase-swapped classes must follow the four-chunk schedule");
    initial_compressible += before ? 1U : 0U;
    changed_compressible += after ? 1U : 0U;
    switched_classes += before != after ? 1U : 0U;
    history_classes += mixed_chunk_needs_history(chunk) ? 1U : 0U;
    expect(!mixed_chunk_needs_history(chunk) || !mixed_chunk_is_write_capable(chunk, 0U),
           "next-generation compression candidates must remain clean before history capture");

    const std::uint64_t offset = chunk * chunk_bytes;
    fill_workload_pattern(ScenarioKind::mixed, seed, offset, chunk_bytes, initial, false);
    fill_workload_pattern(ScenarioKind::mixed, seed, offset, chunk_bytes, repeated, false);
    fill_workload_pattern(ScenarioKind::mixed, seed, offset, chunk_bytes, changed, true);

    expect(initial == repeated, "mixed fixture must be deterministic for one generation");
    expect(digest(initial) != digest(changed),
           "mixed phase change must create a distinct deterministic generation");
    expect(repeats_every(initial, 32U) == before,
           "the initial class must apply to the entire sampled chunk");
    expect(repeats_every(changed, 32U) == after,
           "the changed class must apply to the entire sampled chunk");
  }
  expect(initial_compressible == 2U,
         "the initial mixed generation must contain both whole-chunk classes");
  expect(changed_compressible == 2U,
         "the changed mixed generation must contain both whole-chunk classes");
  expect(switched_classes == 2U,
         "the mixed fixture must contain both stable and phase-swapped histories");
  expect(history_classes == 2U,
         "history must be established only for next-generation compression candidates");
  expect(xvram::compression::detail::mixed_history_cycles >= 2U,
         "mixed history must contain multiple cache-hit observations");
  expect(mixed_chunk_is_write_capable(2U, 1U),
         "the changed generation must still exercise write-capable traffic");
}

void other_scenario_patterns_remain_stable_test() {
  using xvram::compression::ScenarioKind;
  using xvram::compression::detail::fill_workload_pattern;

  constexpr std::uint64_t seed = 0x585652414D503035ULL;
  constexpr std::uint64_t chunk_bytes = 256U * 1024U;
  std::vector<std::byte> compressible(128U * 1024U);
  std::vector<std::byte> incompressible(128U * 1024U);

  fill_workload_pattern(ScenarioKind::compressible_read, seed, chunk_bytes, chunk_bytes,
                        compressible);
  fill_workload_pattern(ScenarioKind::incompressible_read, seed, chunk_bytes, chunk_bytes,
                        incompressible);
  expect(repeats_every(compressible, 32U),
         "compressible-read must retain its periodic whole-buffer pattern");
  expect(!repeats_every(incompressible, 32U),
         "incompressible-read must retain its deterministic pseudo-random pattern");
}

void runtime_cleanup_evidence_test() {
  xvram::residency::RuntimeTelemetry telemetry;
  telemetry.allocations_created = 1U;
  telemetry.allocations_released = 1U;
  telemetry.handles_created = 3U;
  telemetry.handles_released = 3U;
  telemetry.maps = 8U;
  telemetry.unmaps = 8U;
  telemetry.transactions_submitted = 6U;
  telemetry.transactions_completed = 6U;
  telemetry.codec_events_recorded = 4U;
  telemetry.codec_events_retired = 4U;

  auto cleanup = xvram::compression::detail::derive_runtime_cleanup(telemetry, true);
  expect(cleanup.complete.value_or(false),
         "reconciled post-close counters must prove complete runtime cleanup");
  expect(cleanup.operations_drained.value_or(false),
         "completed transaction counters must prove operation drain");
  expect(cleanup.mappings_removed.value_or(false),
         "balanced maps with zero residency must prove mapping removal");
  expect(cleanup.physical_handles_released.value_or(false),
         "balanced handle counters must prove physical release");

  telemetry.unmaps = 7U;
  cleanup = xvram::compression::detail::derive_runtime_cleanup(telemetry, true);
  expect(!cleanup.mappings_removed.value_or(true),
         "a post-close mapping imbalance must fail mapping cleanup evidence");
  expect(!cleanup.complete.value_or(true),
         "a stage-specific cleanup mismatch must fail aggregate cleanup");

  telemetry.unmaps = telemetry.maps;
  telemetry.codec_workspace_bytes = 1U;
  cleanup = xvram::compression::detail::derive_runtime_cleanup(telemetry, true);
  expect(!cleanup.codec_workspace_released.value_or(true),
         "live post-close codec workspace bytes must fail cleanup evidence");

  telemetry.codec_workspace_bytes = 0U;
  telemetry.host_budget_bytes = 1U;
  cleanup = xvram::compression::detail::derive_runtime_cleanup(telemetry, true);
  expect(!cleanup.pinned_staging_released.value_or(true),
         "a non-empty post-close host budget ledger must fail pinned staging cleanup evidence");
}

void write_admission_evidence_test() {
  xvram::residency::RuntimeTelemetry telemetry;
  telemetry.dirty_writebacks = 1U;
  telemetry.logical_d2h_bytes = 64U;
  expect(!xvram::compression::detail::derive_write_admission_proof(
             telemetry, xvram::compression::RequestedPath::cpu_lz4_gpu),
         "compressed dirty traffic without a spill reservation must not prove admission");
  telemetry.spill_reserved_peak_bytes = 64U;
  expect(xvram::compression::detail::derive_write_admission_proof(
             telemetry, xvram::compression::RequestedPath::cpu_lz4_gpu),
         "retired compressed dirty traffic with released spill credit must prove admission");
  telemetry.spill_reserved_bytes = 64U;
  expect(!xvram::compression::detail::derive_write_admission_proof(
             telemetry, xvram::compression::RequestedPath::cpu_lz4_gpu),
         "an outstanding spill reservation must fail write-admission cleanup evidence");
  expect(xvram::compression::detail::derive_write_admission_proof(
             telemetry, xvram::compression::RequestedPath::raw),
         "the raw path must not require a compression spill credit");
}

void trace_callback_failure_report_test() {
  using xvram::compression::detail::apply_trace_delivery_result;
  using xvram::compression::detail::deliver_trace_record;
  xvram::compression::TraceRecord record;
  std::uint64_t accepted = 0;
  const xvram::compression::TraceCallback callback = [&](const auto& item) {
    if (item.sequence == 2U) {
      throw std::runtime_error("injected trace output exception");
    }
    ++accepted;
  };
  std::uint64_t dropped = 0;
  for (std::uint64_t sequence = 1U; sequence <= 3U; ++sequence) {
    record.sequence = sequence;
    if (!deliver_trace_record(callback, record)) {
      ++dropped;
    }
  }
  expect(accepted == 2U && dropped == 1U,
         "trace callback exceptions must be counted while cleanup can keep emitting records");
  xvram::compression::ExecutorResult result;
  result.status = "completed";
  result.exit_code = 0;
  apply_trace_delivery_result(result, 3U, dropped);
  expect(result.status == "failed" && result.exit_code == 27,
         "a completed workload with a dropped trace record must become an explicit failure");
  expect(result.reason == "trace_delivery_failure" && result.failure.has_value(),
         "trace failure must identify its cause in the final result");
  expect(result.telemetry.trace_complete == false && result.telemetry.trace_records_dropped == 1U &&
             result.telemetry.trace_records_emitted == 2U,
         "trace telemetry must reconcile accepted and rejected callback deliveries");

  expect(!deliver_trace_record({}, record),
         "an enabled trace with no callback must fail delivery rather than silently succeed");
  result.status = "corruption";
  result.exit_code = 24;
  result.reason = "reference_mismatch";
  apply_trace_delivery_result(result, 3U, dropped);
  expect(result.status == "corruption" && result.exit_code == 24 &&
             result.reason == "reference_mismatch" && result.telemetry.trace_complete == false,
         "trace failure must not hide an existing data-integrity failure");

  result = {};
  result.status = "completed";
  result.exit_code = 0;
  apply_trace_delivery_result(result, 3U, 0U);
  expect(result.status == "completed" && result.exit_code == 0 &&
             result.telemetry.trace_complete == true &&
             result.telemetry.trace_records_emitted == 3U,
         "successful trace delivery must preserve successful workload output");
}

} // namespace

int main() {
  mixed_whole_chunk_classes_and_phase_switch_test();
  other_scenario_patterns_remain_stable_test();
  runtime_cleanup_evidence_test();
  write_admission_evidence_test();
  trace_callback_failure_report_test();
  if (failures != 0) {
    std::cerr << failures << " compression executor test(s) failed\n";
    return 1;
  }
  return 0;
}
