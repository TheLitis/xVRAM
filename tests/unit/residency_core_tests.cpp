#include "residency/budget.hpp"
#include "residency/core.hpp"
#include "residency/metrics.hpp"
#include "residency/policy.hpp"
#include "residency/scenario.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

namespace {

int failures = 0;

void check(const bool condition, const char* expression, const int line) {
  if (!condition) {
    std::cerr << "line " << line << ": check failed: " << expression << '\n';
    ++failures;
  }
}

#define CHECK(expression) check(static_cast<bool>(expression), #expression, __LINE__)

using xvram::residency::AccessMode;
using xvram::residency::AllocationId;
using xvram::residency::ChunkKey;

[[nodiscard]] bool same_range(const xvram::residency::AccessRange& range,
                              const AllocationId allocation_id, const std::uint64_t offset,
                              const std::uint64_t length, const AccessMode mode) {
  return range.allocation_id == allocation_id && range.offset_bytes == offset &&
         range.length_bytes == length && range.mode == mode;
}

void access_plan_tests() {
  using namespace xvram::residency;
  constexpr AllocationId first{1};
  constexpr AllocationId second{2};
  constexpr std::array layouts{AllocationLayout{second, 8}, AllocationLayout{first, 20}};
  constexpr std::array ranges{
      AccessRange{first, 0, 16, AccessMode::write_only},
      AccessRange{first, 4, 4, AccessMode::read},
      AccessRange{first, 16, 4, AccessMode::write_only},
      AccessRange{second, 0, 4, AccessMode::write_only},
      AccessRange{second, 4, 4, AccessMode::write_only},
  };

  const AccessPlanResult result = normalize_and_split_accesses(layouts, ranges, 8);
  CHECK(result);
  CHECK(result.normalized_ranges.size() == 4);
  CHECK(same_range(result.normalized_ranges[0], first, 0, 4, AccessMode::write_only));
  CHECK(same_range(result.normalized_ranges[1], first, 4, 4, AccessMode::read_write));
  CHECK(same_range(result.normalized_ranges[2], first, 8, 12, AccessMode::write_only));
  CHECK(same_range(result.normalized_ranges[3], second, 0, 8, AccessMode::write_only));
  CHECK(result.chunks.size() == 4);

  const ChunkAccessPlan& mixed = result.chunks[0];
  CHECK((mixed.key == ChunkKey{first, 0}));
  CHECK(mixed.spans.size() == 2);
  CHECK(mixed.reads_existing_data);
  CHECK(mixed.marks_dirty);
  CHECK(!mixed.full_write_only);
  CHECK(mixed.requires_h2d);

  const ChunkAccessPlan& full = result.chunks[1];
  CHECK((full.key == ChunkKey{first, 1}));
  CHECK(full.full_write_only);
  CHECK(!full.requires_h2d);
  CHECK(!full.reads_existing_data);
  CHECK(full.marks_dirty);

  const ChunkAccessPlan& tail = result.chunks[2];
  CHECK((tail.key == ChunkKey{first, 2}));
  CHECK(tail.valid_bytes == 4);
  CHECK(tail.full_write_only);
  CHECK(!tail.requires_h2d);

  const ChunkAccessPlan& adjacent = result.chunks[3];
  CHECK((adjacent.key == ChunkKey{second, 0}));
  CHECK(adjacent.spans.size() == 1);
  CHECK(adjacent.full_write_only);

  constexpr std::array partial_range{
      AccessRange{first, 2, 4, AccessMode::write_only},
  };
  const AccessPlanResult partial = normalize_and_split_accesses(layouts, partial_range, 8);
  CHECK(partial);
  CHECK(partial.chunks.size() == 1);
  CHECK(!partial.chunks[0].full_write_only);
  CHECK(partial.chunks[0].requires_h2d);

  const AccessPlanResult empty =
      normalize_and_split_accesses(layouts, std::span<const AccessRange>{}, 8);
  CHECK(empty);
  CHECK(empty.chunks.empty());

  CHECK(normalize_and_split_accesses(layouts, ranges, 0).error ==
        AccessPlanError::invalid_chunk_size);
  constexpr std::array duplicate_layouts{AllocationLayout{first, 8}, AllocationLayout{first, 16}};
  CHECK(normalize_and_split_accesses(duplicate_layouts, partial_range, 8).error ==
        AccessPlanError::duplicate_allocation);
  constexpr std::array unknown_range{AccessRange{AllocationId{9}, 0, 1, AccessMode::read}};
  CHECK(normalize_and_split_accesses(layouts, unknown_range, 8).error ==
        AccessPlanError::unknown_allocation);
  constexpr std::array zero_range{AccessRange{first, 0, 0, AccessMode::read}};
  CHECK(normalize_and_split_accesses(layouts, zero_range, 8).error == AccessPlanError::zero_length);
  constexpr std::array overflow_range{
      AccessRange{first, std::numeric_limits<std::uint64_t>::max() - 1U, 4, AccessMode::read}};
  CHECK(normalize_and_split_accesses(layouts, overflow_range, 8).error ==
        AccessPlanError::range_overflow);
  constexpr std::array out_of_bounds{AccessRange{first, 19, 2, AccessMode::read}};
  CHECK(normalize_and_split_accesses(layouts, out_of_bounds, 8).error ==
        AccessPlanError::range_out_of_bounds);
  const std::array invalid_mode{
      AccessRange{first, 0, 1, static_cast<AccessMode>(77)},
  };
  CHECK(normalize_and_split_accesses(layouts, invalid_mode, 8).error ==
        AccessPlanError::invalid_access_mode);
}

void state_machine_tests() {
  using namespace xvram::residency;
  ChunkState state = ChunkState::host_clean;
  CHECK(transition_chunk_state(state, ChunkState::mapping) == StateTransitionResult::success);
  CHECK(transition_chunk_state(state, ChunkState::resident_clean) ==
        StateTransitionResult::success);
  CHECK(transition_chunk_state(state, ChunkState::resident_dirty) ==
        StateTransitionResult::success);
  CHECK(transition_chunk_state(state, ChunkState::writeback_queued) ==
        StateTransitionResult::success);
  CHECK(transition_chunk_state(state, ChunkState::d2h_in_flight) == StateTransitionResult::success);
  CHECK(transition_chunk_state(state, ChunkState::resident_clean) ==
        StateTransitionResult::success);
  CHECK(transition_chunk_state(state, ChunkState::host_clean) ==
        StateTransitionResult::illegal_transition);
  CHECK(state == ChunkState::resident_clean);
  CHECK(transition_chunk_state(state, ChunkState::poisoned) == StateTransitionResult::success);
  CHECK(transition_chunk_state(state, ChunkState::host_clean) ==
        StateTransitionResult::illegal_transition);

  ChunkRecord host{};
  host.key = ChunkKey{AllocationId{1}, 0};
  CHECK(validate_chunk_record(host) == ChunkInvariantError::none);
  host.frame_index = 1;
  CHECK(validate_chunk_record(host) == ChunkInvariantError::unexpected_frame);

  ChunkRecord transfer{};
  transfer.key = ChunkKey{AllocationId{1}, 1};
  transfer.state = ChunkState::h2d_in_flight;
  transfer.frame_index = 0;
  CHECK(validate_chunk_record(transfer) == ChunkInvariantError::missing_staging);
  transfer.staging_slot = 1;
  CHECK(validate_chunk_record(transfer) == ChunkInvariantError::none);
  CHECK(chunk_has_in_flight_work(transfer));

  ChunkRecord resident{};
  resident.key = ChunkKey{AllocationId{1}, 2};
  resident.state = ChunkState::resident_clean;
  resident.frame_index = 2;
  CHECK(is_victim_eligible(resident));
  resident.pin_count = 1;
  CHECK(!is_victim_eligible(resident));
  resident.pin_count = 0;
  resident.in_current_working_set = true;
  CHECK(!is_victim_eligible(resident));
  resident.in_current_working_set = false;
  CHECK(begin_event_generation(resident) == EventGenerationResult::success);
  CHECK(resident.event_generation == 1);
  CHECK(chunk_has_in_flight_work(resident));
  CHECK(!is_victim_eligible(resident));
  CHECK(complete_event_generation(resident, 2) == EventGenerationResult::future_generation);
  CHECK(complete_event_generation(resident, 1) == EventGenerationResult::success);
  CHECK(complete_event_generation(resident, 1) == EventGenerationResult::already_completed);
  CHECK(is_victim_eligible(resident));
  CHECK(begin_event_generation(resident) == EventGenerationResult::success);
  CHECK(complete_event_generation(resident, 1) == EventGenerationResult::already_completed);
  CHECK(complete_event_generation(resident, 2) == EventGenerationResult::success);

  resident.state = ChunkState::evicting;
  CHECK(validate_chunk_record(resident) == ChunkInvariantError::none);
  CHECK(is_safe_to_unmap(resident));
  resident.completed_generation = 1;
  CHECK(validate_chunk_record(resident) == ChunkInvariantError::incomplete_evicting_generation);
  CHECK(!is_safe_to_unmap(resident));
}

[[nodiscard]] xvram::residency::VictimCandidate
candidate(const std::uint64_t index, const xvram::residency::ChunkState state) {
  return {ChunkKey{AllocationId{1}, index}, state};
}

void policy_tests() {
  using namespace xvram::residency;
  constexpr ChunkKey first{AllocationId{1}, 1};
  constexpr ChunkKey second{AllocationId{1}, 2};
  constexpr ChunkKey third{AllocationId{1}, 3};

  ClockPolicy clock;
  clock.insert(first, 1, false, false);
  clock.insert(second, 2, true, false);
  clock.insert(third, 3, false, true);
  CHECK(clock.name() == std::string_view{"clock"});
  CHECK(clock.size() == 3);
  std::array candidates{candidate(1, ChunkState::resident_clean),
                        candidate(2, ChunkState::resident_clean),
                        candidate(3, ChunkState::resident_dirty)};
  CHECK(clock.select_victim(candidates) == second);
  candidates[1].pin_count = 1;
  CHECK(clock.select_victim(candidates) == first);
  candidates[0].in_current_working_set = true;
  CHECK(clock.select_victim(candidates) == third);
  candidates[2].in_flight = true;
  CHECK(!clock.select_victim(candidates).has_value());
  CHECK(clock.erase(second));
  CHECK(!clock.erase(second));
  CHECK(clock.size() == 2);

  LruPolicy lru;
  lru.insert(second, 10, false, false);
  lru.insert(first, 10, false, false);
  lru.insert(third, 20, false, false);
  candidates = {candidate(1, ChunkState::resident_clean), candidate(2, ChunkState::resident_dirty),
                candidate(3, ChunkState::resident_clean)};
  CHECK(lru.select_victim(candidates) == first);
  lru.touch(first, 30, false);
  CHECK(lru.select_victim(candidates) == second);
  candidates[1].pin_count = 1;
  CHECK(lru.select_victim(candidates) == third);
}

void budget_tests() {
  using namespace xvram::residency;
  constexpr std::uint64_t mib = 1024ULL * 1024ULL;
  constexpr std::uint64_t gib = 1024ULL * mib;
  constexpr std::uint64_t chunk = 64ULL * mib;

  BudgetTargetInput input;
  input.configured_cap_bytes = 8ULL * gib;
  input.cuda_free_bytes = 7ULL * gib;
  input.managed_frame_bytes = 1ULL * gib;
  input.wddm = WddmBudgetObservation{9ULL * gib, 2ULL * gib};
  input.device_headroom_bytes = 512ULL * mib;
  input.chunk_bytes = chunk;
  input.maximum_working_set_bytes = 96ULL * mib;
  const BudgetTarget target = calculate_budget_target(input);
  CHECK(target);
  CHECK(target.cuda_reclaimable_bytes == 8ULL * gib);
  CHECK(target.wddm_available_bytes == 7ULL * gib);
  CHECK(target.wddm_reclaimable_bytes == 8ULL * gib);
  CHECK(target.target_bytes == 7ULL * gib + 512ULL * mib);
  CHECK(target.required_minimum_bytes == 128ULL * mib);
  CHECK(target.limited_by_configured_cap);
  CHECK(target.limited_by_cuda);
  CHECK(target.limited_by_wddm);

  input.wddm = WddmBudgetObservation{1ULL * gib, 2ULL * gib};
  const BudgetTarget exhausted_wddm = calculate_budget_target(input);
  CHECK(exhausted_wddm);
  CHECK(exhausted_wddm.wddm_available_bytes == 0);
  CHECK(exhausted_wddm.target_bytes == 512ULL * mib);

  input.maximum_working_set_bytes = 768ULL * mib;
  CHECK(calculate_budget_target(input).error == BudgetTargetError::below_minimum);
  input.chunk_bytes = 0;
  CHECK(calculate_budget_target(input).error == BudgetTargetError::invalid_configuration);

  TargetHysteresisConfig config;
  config.chunk_bytes = chunk;
  TargetHysteresisState state{4ULL * chunk, 0};
  TargetDecision decision = observe_safe_target(state, 3ULL * chunk, 2ULL * chunk, config);
  CHECK(decision.action == TargetAction::shrink);
  CHECK(state.current_target_bytes == 3ULL * chunk);
  decision = observe_safe_target(state, chunk, 2ULL * chunk, config);
  CHECK(decision.action == TargetAction::budget_pressure);
  CHECK(state.current_target_bytes == chunk);

  state = TargetHysteresisState{2ULL * chunk, 0};
  for (std::uint32_t sample = 0; sample < 9; ++sample) {
    decision = observe_safe_target(state, 8ULL * chunk, 2ULL * chunk, config);
    CHECK(decision.action == TargetAction::unchanged);
  }
  CHECK(state.consecutive_safe_samples == 9);
  decision = observe_safe_target(state, 8ULL * chunk, 2ULL * chunk, config);
  CHECK(decision.action == TargetAction::grow);
  CHECK(state.current_target_bytes == 4ULL * chunk);
  CHECK(state.consecutive_safe_samples == 0);
  decision = observe_safe_target(state, 5ULL * chunk, 2ULL * chunk, config);
  CHECK(decision.action == TargetAction::unchanged);
  CHECK(state.consecutive_safe_samples == 0);
}

[[nodiscard]] bool same_operation(const xvram::residency::ScenarioOperation& left,
                                  const xvram::residency::ScenarioOperation& right) {
  if (left.sequence != right.sequence || left.kind != right.kind ||
      left.pressure_bytes != right.pressure_bytes || left.pass_index != right.pass_index ||
      left.warmup != right.warmup || left.sequential_one_touch != right.sequential_one_touch ||
      left.access.has_value() != right.access.has_value()) {
    return false;
  }
  if (!left.access.has_value()) {
    return true;
  }
  return same_range(*left.access, right.access->allocation_id, right.access->offset_bytes,
                    right.access->length_bytes, right.access->mode);
}

void scenario_tests() {
  using namespace xvram::residency;
  ScenarioInput input;
  input.logical_bytes = 20;
  input.chunk_bytes = 8;
  input.cache_target_bytes = 16;
  input.passes = 2;
  input.seed = 1234;

  const ScenarioTraceResult sequential = generate_scenario_trace(ScenarioKind::sequential, input);
  CHECK(sequential);
  CHECK(sequential.trace->operations.size() == 6);
  constexpr std::array<std::uint64_t, 6> expected_offsets{0, 8, 16, 16, 8, 0};
  for (std::size_t index = 0; index < expected_offsets.size(); ++index) {
    const ScenarioOperation& operation = sequential.trace->operations[index];
    CHECK(operation.sequence == index);
    CHECK(operation.access.has_value());
    CHECK(operation.access->offset_bytes == expected_offsets[index]);
    CHECK(operation.access->mode == AccessMode::read_write);
    CHECK(operation.sequential_one_touch);
  }
  CHECK(sequential.trace->operations[2].access->length_bytes == 4);

  const ScenarioTraceResult reuse = generate_scenario_trace(ScenarioKind::reuse, input);
  CHECK(reuse);
  CHECK(reuse.trace->operations.size() == 8);
  CHECK(reuse.trace->operations.front().warmup);
  CHECK(!reuse.trace->operations[1].warmup);

  const ScenarioTraceResult random_first = generate_scenario_trace(ScenarioKind::random, input);
  const ScenarioTraceResult random_second = generate_scenario_trace(ScenarioKind::random, input);
  CHECK(random_first && random_second);
  CHECK(random_first.trace->operations.size() == 12);
  CHECK(random_first.trace->operations.size() == random_second.trace->operations.size());
  for (std::size_t index = 0; index < random_first.trace->operations.size(); ++index) {
    CHECK(same_operation(random_first.trace->operations[index],
                         random_second.trace->operations[index]));
    CHECK(random_first.trace->operations[index].access->mode != AccessMode::write_only);
  }

  const ScenarioTraceResult read_only = generate_scenario_trace(ScenarioKind::read_only, input);
  CHECK(read_only);
  for (const ScenarioOperation& operation : read_only.trace->operations) {
    CHECK(operation.access->mode == AccessMode::read);
  }

  const ScenarioTraceResult write_heavy = generate_scenario_trace(ScenarioKind::write_heavy, input);
  CHECK(write_heavy);
  CHECK(write_heavy.trace->operations.size() == 6);
  std::uint64_t write_heavy_reads = 0;
  std::uint64_t write_heavy_read_writes = 0;
  std::uint64_t write_heavy_write_only = 0;
  for (const ScenarioOperation& operation : write_heavy.trace->operations) {
    write_heavy_reads += operation.access->mode == AccessMode::read ? 1U : 0U;
    write_heavy_read_writes += operation.access->mode == AccessMode::read_write ? 1U : 0U;
    write_heavy_write_only += operation.access->mode == AccessMode::write_only ? 1U : 0U;
  }
  CHECK(write_heavy_reads == 1);
  CHECK(write_heavy_read_writes == 4);
  CHECK(write_heavy_write_only == 1);

  const ScenarioTraceResult pressure =
      generate_scenario_trace(ScenarioKind::budget_pressure, input);
  CHECK(pressure);
  CHECK(pressure.trace->operations.size() == 8);
  CHECK(pressure.trace->operations[2].kind == ScenarioOperationKind::pressure_acquire);
  CHECK(pressure.trace->operations[5].kind == ScenarioOperationKind::pressure_release);
  CHECK(pressure.trace->operations[2].pressure_bytes == 32);

  const ScenarioSuiteResult suite = generate_scenario_suite(input);
  CHECK(suite);
  CHECK(suite.traces.size() == 5);
  CHECK(suite.traces.front().scenario == ScenarioKind::sequential);
  CHECK(suite.traces.back().scenario == ScenarioKind::write_heavy);
  CHECK(generate_scenario_trace(ScenarioKind::suite, input).error ==
        ScenarioTraceError::suite_requires_suite_generator);

  input.cache_target_bytes = 8;
  CHECK(generate_scenario_trace(ScenarioKind::sequential, input).error ==
        ScenarioTraceError::invalid_configuration);
}

[[nodiscard]] xvram::residency::CacheMetrics valid_metrics() {
  xvram::residency::CacheMetrics metrics;
  metrics.demand_accesses = 10;
  metrics.cache_hits = 7;
  metrics.cache_misses = 3;
  metrics.prefetch_issued = 5;
  metrics.prefetch_useful = 2;
  metrics.prefetch_wasted = 1;
  metrics.prefetch_cancelled = 1;
  metrics.prefetch_in_flight = 1;
  metrics.dirty_evictions = 2;
  metrics.eviction_writebacks_completed = 2;
  metrics.drain_writebacks_completed = 1;
  metrics.writebacks_completed = 3;
  metrics.mappings_completed = 6;
  metrics.set_access_completed = 6;
  metrics.unmaps_completed = 4;
  metrics.active_mappings = 2;
  metrics.handles_created = 3;
  metrics.handles_released = 1;
  metrics.live_handles = 2;
  metrics.handle_reuses = 4;
  metrics.resident_peak_bytes = 128;
  metrics.target_peak_bytes = 256;
  metrics.staging_slots_peak = 3;
  metrics.staging_slots_capacity = 4;
  return metrics;
}

void metrics_tests() {
  using namespace xvram::residency;
  CacheMetrics metrics = valid_metrics();
  CHECK(reconcile_metrics(metrics));

  metrics.cache_hits = 8;
  MetricsReconciliation reconciliation = reconcile_metrics(metrics);
  CHECK(!reconciliation);
  CHECK(reconciliation.contains(MetricIssue::demand_accounting));
  metrics = valid_metrics();
  metrics.prefetch_wasted = 2;
  CHECK(reconcile_metrics(metrics).contains(MetricIssue::prefetch_accounting));
  metrics = valid_metrics();
  metrics.unsafe_remaps = 1;
  CHECK(reconcile_metrics(metrics).contains(MetricIssue::unsafe_remap));
  metrics = valid_metrics();
  metrics.resident_peak_bytes = 257;
  CHECK(reconcile_metrics(metrics).contains(MetricIssue::residency_exceeds_target));

  CacheMetrics total;
  CacheMetrics delta = valid_metrics();
  CHECK(accumulate_metrics(total, delta));
  CHECK(total.demand_accesses == delta.demand_accesses);
  CHECK(total.resident_peak_bytes == delta.resident_peak_bytes);
  CacheMetrics overflow;
  overflow.demand_accesses = std::numeric_limits<std::uint64_t>::max();
  const CacheMetrics before = total;
  CHECK(!accumulate_metrics(total, overflow));
  CHECK(total.demand_accesses == before.demand_accesses);
  CHECK(total.cache_hits == before.cache_hits);
}

} // namespace

int main() {
  access_plan_tests();
  state_machine_tests();
  policy_tests();
  budget_tests();
  scenario_tests();
  metrics_tests();
  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
    return 1;
  }
  std::cout << "all residency core tests passed\n";
  return 0;
}
