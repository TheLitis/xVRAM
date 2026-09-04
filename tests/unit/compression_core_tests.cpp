#include "residency/compression.hpp"
#include "residency/lz4_codec.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
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

void backing_store_tests() {
  using namespace xvram::residency;
  Lz4BlockCodec codec;
  HostBackingStore store({2 * compression_block_bytes, 8 * compression_block_bytes}, codec);
  const ChunkKey key{AllocationId{7}, 3};

  const BackingResult created =
      store.register_chunk(key, compression_block_bytes + 17, BackingRepresentation::implicit_zero);
  CHECK(created);
  CHECK(created.chunk.generation == 1);
  CHECK(created.chunk.representation == BackingRepresentation::implicit_zero);
  CHECK(created.chunk.stored_payload_bytes == 0);
  CHECK(created.chunk.budget_charge_bytes == 512);
  const HostBudgetSnapshot zero_budget = store.budget().snapshot();
  CHECK(zero_budget.authoritative_bytes == created.chunk.budget_charge_bytes);
  CHECK(zero_budget.conversion_scratch_bytes == 0);
  CHECK(zero_budget.total_bytes == created.chunk.budget_charge_bytes);

  std::vector<std::byte> zeros(static_cast<std::size_t>(created.chunk.valid_bytes));
  std::vector<std::byte> readback(zeros.size(), std::byte{0x7F});
  CHECK(store.read(key, 0, readback));
  CHECK(readback == zeros);
  CHECK(created.chunk.content_token == compute_content_token(zeros));

  const std::vector patch{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
  const BackingResult patched = store.write(key, 1, compression_block_bytes - 2, patch);
  CHECK(patched);
  CHECK(patched.chunk.generation == 2);
  CHECK(patched.chunk.representation == BackingRepresentation::lz4_blocks);
  CHECK(store.write(key, 1, 0, patch).status == BackingStatus::stale_generation);
  std::copy(patch.begin(), patch.end(),
            zeros.begin() + static_cast<std::ptrdiff_t>(compression_block_bytes - 2));
  CHECK(store.read(key, 0, readback));
  CHECK(readback == zeros);
  CHECK(patched.chunk.content_token == compute_content_token(zeros));

  const BackingResult raw = store.materialize_raw(key, 2);
  CHECK(raw);
  CHECK(raw.chunk.generation == 3);
  CHECK(raw.chunk.representation == BackingRepresentation::raw);
  const BackingResult compressed = store.compress(key, 3);
  CHECK(compressed);
  CHECK(compressed.chunk.generation == 4);
  CHECK(compressed.chunk.representation == BackingRepresentation::lz4_blocks);
  CHECK(compressed.chunk.stored_payload_bytes < compressed.chunk.valid_bytes);

  Lz4BlocksV1 exported;
  CHECK(store.export_lz4_blocks(key, exported) == BackingStatus::success);
  CHECK(exported.generation == 4);
  CHECK(exported.content_token == compute_content_token(zeros));
  CHECK(store.read(key, 0, readback));
  CHECK(readback == zeros);

  exported.generation = 5;
  exported.content_token.low ^= 1U;
  CHECK(store.commit_lz4_blocks(key, 4, std::move(exported)).status == BackingStatus::corrupt_data);
  CHECK(store.inspect(key).chunk.generation == 4);

  const HostBudgetSnapshot budget = store.budget().snapshot();
  CHECK(budget.authoritative_bytes == compressed.chunk.budget_charge_bytes);
  CHECK(budget.conversion_scratch_bytes == 0);
  CHECK(budget.total_bytes == budget.authoritative_bytes);
  CHECK(store.erase_chunk(key, 4));
  CHECK(store.budget().snapshot().total_bytes == 0);
}

void budget_tests() {
  using namespace xvram::residency;
  HostBudgetLedger ledger(100);
  HostBudgetReservation spill;
  HostBudgetReservation staging;
  CHECK(ledger.reserve(HostBudgetCategory::spill, 60, spill) == HostBudgetStatus::success);
  CHECK(ledger.reserve(HostBudgetCategory::pinned_staging, 41, staging) ==
        HostBudgetStatus::limit_exceeded);
  CHECK(ledger.snapshot().rejected_bytes == 41);
  CHECK(ledger.release(spill) == HostBudgetStatus::success);
  CHECK(ledger.snapshot().total_bytes == 0);

  Lz4BlockCodec codec;
  HostBackingStore constrained({compression_block_bytes, 512}, codec);
  const ChunkKey key{AllocationId{2}, 0};
  const BackingResult constrained_created =
      constrained.register_chunk(key, compression_block_bytes);
  CHECK(constrained_created);
  const std::vector<std::byte> bytes(static_cast<std::size_t>(compression_block_bytes),
                                     std::byte{1});
  CHECK(constrained.replace_raw(key, 1, bytes).status == BackingStatus::host_budget_exceeded);
  CHECK(constrained.inspect(key).chunk.generation == 1);
  const HostBudgetSnapshot constrained_budget = constrained.budget().snapshot();
  CHECK(constrained_budget.authoritative_bytes == constrained_created.chunk.budget_charge_bytes);
  CHECK(constrained_budget.conversion_scratch_bytes == 0);
}

void eager_raw_registration_test() {
  using namespace xvram::residency;
  Lz4BlockCodec codec;
  HostBackingStore store({compression_block_bytes, 2 * compression_block_bytes}, codec);
  const ChunkKey key{AllocationId{11}, 0};
  const BackingResult created =
      store.register_chunk(key, compression_block_bytes, BackingRepresentation::raw);
  CHECK(created);
  CHECK(created.chunk.representation == BackingRepresentation::raw);
  CHECK(created.chunk.stored_payload_bytes == compression_block_bytes);
  CHECK(created.chunk.budget_charge_bytes >= compression_block_bytes);

  std::vector<std::byte> readback(static_cast<std::size_t>(compression_block_bytes),
                                  std::byte{0x7F});
  CHECK(store.read(key, 0, readback));
  CHECK(std::all_of(readback.begin(), readback.end(),
                    [](const std::byte value) { return value == std::byte{0}; }));
  CHECK(created.chunk.content_token == compute_content_token(readback));
}

void spill_to_authority_commit_test() {
  using namespace xvram::residency;
  Lz4BlockCodec codec;
  HostBackingStore store({compression_block_bytes, 3 * compression_block_bytes}, codec);
  const ChunkKey key{AllocationId{12}, 0};
  const BackingResult created = store.register_chunk(key, compression_block_bytes);
  CHECK(created);
  const auto charge = maximum_raw_backing_charge(compression_block_bytes);
  CHECK(charge.has_value());
  HostBudgetReservation spill;
  CHECK(store.budget().reserve(HostBudgetCategory::spill, *charge, spill) ==
        HostBudgetStatus::success);
  PreparedRawBacking prepared;
  CHECK(store.prepare_raw_replacement(key, created.chunk.generation, prepared));
  CHECK(prepared.valid());
  std::vector<std::byte> replacement(static_cast<std::size_t>(compression_block_bytes),
                                     std::byte{0x4D});
  CHECK(store.fill_prepared_raw(prepared, replacement) == BackingStatus::success);
  const BackingResult committed =
      store.commit_prepared_raw(key, created.chunk.generation, prepared, &spill);
  CHECK(committed);
  CHECK(!prepared.valid());
  CHECK(spill.id == 0U);
  CHECK(spill.bytes == 0U);
  CHECK(committed.chunk.representation == BackingRepresentation::raw);
  const HostBudgetSnapshot budget = store.budget().snapshot();
  CHECK(budget.spill_bytes == 0U);
  CHECK(budget.conversion_scratch_bytes == 0U);
  CHECK(budget.authoritative_bytes == committed.chunk.budget_charge_bytes);
  std::vector<std::byte> readback(replacement.size());
  CHECK(store.read(key, 0, readback));
  CHECK(readback == replacement);
}

void raw_block_ownership_and_tail_test() {
  using namespace xvram::residency;
  Lz4BlockCodec codec;
  const std::uint64_t valid_bytes = 3U * compression_block_bytes + 17U;
  const auto raw_charge = maximum_raw_backing_charge(valid_bytes);
  CHECK(raw_charge.has_value());
  if (!raw_charge.has_value()) {
    return;
  }
  HostBackingStore store({valid_bytes, 4U * *raw_charge}, codec);
  const ChunkKey key{AllocationId{13}, 0};
  BackingResult current = store.register_chunk(key, valid_bytes, BackingRepresentation::raw);
  CHECK(current);
  CHECK(current.chunk.budget_charge_bytes == *raw_charge);
  CHECK(current.chunk.stored_payload_bytes == valid_bytes);

  std::vector<std::byte> expected(static_cast<std::size_t>(valid_bytes));
  std::uint64_t random = 0x585652414d503035ULL;
  for (std::byte& value : expected) {
    random ^= random >> 12U;
    random ^= random << 25U;
    random ^= random >> 27U;
    value = static_cast<std::byte>(random & 0xffU);
  }
  current = store.replace_raw(key, current.chunk.generation, expected);
  CHECK(current);
  CHECK(current.chunk.budget_charge_bytes == *raw_charge);
  std::vector<std::byte> readback(expected.size());

  // Replacing one block must retain every untouched block, including the short vector-backed
  // tail. On Windows the complete raw blocks have independent pageable reservations; no old
  // chunk-sized allocation can remain hidden behind the surviving tail or its shared generation.
  const std::vector<std::byte> patch(31U, std::byte{0x6d});
  const std::uint64_t patch_offset = compression_block_bytes + 7U;
  current = store.write(key, current.chunk.generation, patch_offset, patch);
  CHECK(current);
  std::copy(patch.begin(), patch.end(),
            expected.begin() + static_cast<std::ptrdiff_t>(patch_offset));
  CHECK(store.read(key, 0, readback));
  CHECK(readback == expected);
  CHECK(current.chunk.content_token == compute_content_token(expected));
  CHECK(store.budget().snapshot().authoritative_bytes == current.chunk.budget_charge_bytes);
  CHECK(store.budget().snapshot().conversion_scratch_bytes == 0U);

  const std::vector<std::byte> tail(17U, std::byte{0x2a});
  current = store.write(key, current.chunk.generation, 3U * compression_block_bytes, tail);
  CHECK(current);
  std::copy(tail.begin(), tail.end(), expected.end() - static_cast<std::ptrdiff_t>(tail.size()));
  CHECK(store.read(key, 0, readback));
  CHECK(readback == expected);

  HostBudgetReservation spill;
  CHECK(store.budget().reserve(HostBudgetCategory::spill, *raw_charge, spill) ==
        HostBudgetStatus::success);
  PreparedRawBacking prepared;
  CHECK(store.prepare_raw_replacement(key, current.chunk.generation, prepared));
  CHECK(store.fill_prepared_raw(prepared, expected) == BackingStatus::success);
  current = store.commit_prepared_raw(key, current.chunk.generation, prepared, &spill);
  CHECK(current);
  CHECK(!prepared.valid());
  CHECK(current.chunk.budget_charge_bytes == *raw_charge);
  CHECK(store.budget().snapshot().total_bytes == *raw_charge);
  CHECK(store.read(key, 0, readback));
  CHECK(readback == expected);

  const BackingResult zero = store.set_implicit_zero(key, current.chunk.generation);
  CHECK(zero);
  CHECK(zero.chunk.budget_charge_bytes == 256U + 4U * 128U);
  CHECK(store.budget().snapshot().total_bytes == zero.chunk.budget_charge_bytes);
  CHECK(store.read(key, 0, readback));
  CHECK(std::all_of(readback.begin(), readback.end(),
                    [](const std::byte value) { return value == std::byte{0}; }));
  CHECK(store.erase_chunk(key, zero.chunk.generation));
  CHECK(store.budget().snapshot().total_bytes == 0U);
}

xvram::residency::CostObservation observation(const xvram::residency::CompressionPath path,
                                              const bool fast) {
  using namespace xvram::residency;
  CostObservation result;
  result.path = path;
  result.raw_bytes = 1000;
  result.stored_bytes = path == CompressionPath::raw ? 1000 : 400;
  result.encode_us = path == CompressionPath::raw ? 0.0 : 10.0;
  result.decode_us = path == CompressionPath::raw ? 0.0 : 10.0;
  result.transfer_us = path == CompressionPath::raw ? 100.0 : (fast ? 20.0 : 80.0);
  result.staging_us = 0.0;
  return result;
}

void cost_model_tests() {
  using namespace xvram::residency;
  CompressionCostModel model;
  CHECK(model.valid());
  for (int sample = 0; sample < 2; ++sample) {
    CHECK(model.observe(9, observation(CompressionPath::raw, true)));
    CHECK(model.observe(9, observation(CompressionPath::cpu_lz4_gpu_decode, true)));
  }
  CostDecision decision = model.decide(CompressionMode::adaptive, 1000, true, false);
  CHECK(!decision.calibration_required);
  CHECK(decision.path == CompressionPath::cpu_lz4_gpu_decode);
  CHECK(decision.margin_satisfied);

  CHECK(!model.record_probe(9, CompressionPath::cpu_lz4_gpu_decode, 100.0, 90.0));
  CHECK(!model.record_probe(9, CompressionPath::cpu_lz4_gpu_decode, 100.0, 90.0));
  CHECK(!model.record_probe(9, CompressionPath::cpu_lz4_gpu_decode, 100.0, 90.0));
  CHECK(model.never_compress());
  CHECK(model.unfavorable_probe_count() == 3);
  CHECK(model.decide(CompressionMode::adaptive, 1000, true, false).path == CompressionPath::raw);
  CHECK(model.record_probe(9, CompressionPath::cpu_lz4_gpu_decode, 1000.0, 100.0));
  CHECK(model.never_compress());
  CHECK(model.unfavorable_probe_count() == 3);
  CHECK(model.decide(CompressionMode::adaptive, 1000, true, false).path == CompressionPath::raw);
  const CostDecision capacity = model.decide(CompressionMode::capacity, 1000, true, false);
  CHECK(capacity.path == CompressionPath::cpu_lz4_gpu_decode);

  CompressionCostModel slow_model;
  for (int sample = 0; sample < 2; ++sample) {
    CHECK(slow_model.observe(1, observation(CompressionPath::raw, true)));
    CHECK(slow_model.observe(1, observation(CompressionPath::cpu_lz4_gpu_decode, false)));
  }
  const CostDecision slow_capacity =
      slow_model.decide(CompressionMode::capacity, 1000, true, false);
  CHECK(slow_capacity.path == CompressionPath::cpu_lz4_gpu_decode);
  CHECK(slow_capacity.capacity_override);

  CHECK(model.observe(10, observation(CompressionPath::raw, true)));
  CHECK(model.generation() == 10);
  CHECK(!model.never_compress());
  CHECK(model.unfavorable_probe_count() == 0);
  CHECK(model.metrics(CompressionPath::raw).samples == 3);
  CHECK(model.metrics(CompressionPath::cpu_lz4_gpu_decode).samples == 2);
  const CostDecision inherited = model.decide(CompressionMode::adaptive, 1000, true, false);
  CHECK(!inherited.calibration_required);
  CHECK(inherited.path == CompressionPath::cpu_lz4_gpu_decode);
}

void cost_model_context_and_phase_switch_tests() {
  using namespace xvram::residency;

  CostObservation raw = observation(CompressionPath::raw, true);
  raw.transfer_us = 200.0;
  CostObservation cpu = observation(CompressionPath::cpu_lz4_gpu_decode, true);
  cpu.cpu_queue_us = 5.0;

  CompressionCostModel cpu_pressure(1.0);
  for (int sample = 0; sample < 2; ++sample) {
    CHECK(cpu_pressure.observe(30, raw));
    CHECK(cpu_pressure.observe(30, cpu));
  }
  const CostEstimate available_cpu =
      cpu_pressure.estimate(CompressionPath::cpu_lz4_gpu_decode, 1000);
  CHECK(cpu_pressure.decide(CompressionMode::adaptive, 1000, true, false).path ==
        CompressionPath::cpu_lz4_gpu_decode);
  cpu.cpu_availability = 0.05;
  for (int sample = 0; sample < 2; ++sample) {
    CHECK(cpu_pressure.observe(30, raw));
    CHECK(cpu_pressure.observe(30, cpu));
  }
  const CostEstimate starved_cpu = cpu_pressure.estimate(CompressionPath::cpu_lz4_gpu_decode, 1000);
  CHECK(starved_cpu.predicted_total_us > available_cpu.predicted_total_us);
  CHECK(cpu_pressure.decide(CompressionMode::adaptive, 1000, true, false).path ==
        CompressionPath::raw);

  raw.transfer_us = 130.0;
  CostObservation gpu = observation(CompressionPath::nvcomp_gpu_codec, true);
  gpu.sm_opportunity_us = 20.0;
  CompressionCostModel gpu_pressure(1.0);
  for (int sample = 0; sample < 2; ++sample) {
    CHECK(gpu_pressure.observe(31, raw));
    CHECK(gpu_pressure.observe(31, gpu));
  }
  const CostEstimate idle_gpu = gpu_pressure.estimate(CompressionPath::nvcomp_gpu_codec, 1000);
  CHECK(gpu_pressure.decide(CompressionMode::adaptive, 1000, false, true).path ==
        CompressionPath::nvcomp_gpu_codec);
  gpu.gpu_occupancy = 1.0;
  for (int sample = 0; sample < 2; ++sample) {
    CHECK(gpu_pressure.observe(31, raw));
    CHECK(gpu_pressure.observe(31, gpu));
  }
  const CostEstimate saturated_gpu = gpu_pressure.estimate(CompressionPath::nvcomp_gpu_codec, 1000);
  CHECK(saturated_gpu.predicted_total_us > idle_gpu.predicted_total_us);
  CHECK(gpu_pressure.decide(CompressionMode::adaptive, 1000, false, true).path ==
        CompressionPath::raw);

  raw = observation(CompressionPath::raw, true);
  raw.transfer_us = 100.0;
  cpu = observation(CompressionPath::cpu_lz4_gpu_decode, true);
  cpu.encode_us = 200.0;
  cpu.decode_us = 10.0;
  cpu.transfer_us = 20.0;
  CompressionCostModel phase_model(1.0);
  for (int sample = 0; sample < 2; ++sample) {
    CHECK(phase_model.observe(40, raw));
    CHECK(phase_model.observe(40, cpu));
  }
  CHECK(phase_model.decide(CompressionMode::adaptive, 1000, true, false).path ==
        CompressionPath::raw);

  phase_model.reset(41);
  raw.reuse_count = 9;
  cpu.reuse_count = 9;
  for (int sample = 0; sample < 2; ++sample) {
    CHECK(phase_model.observe(41, raw));
    CHECK(phase_model.observe(41, cpu));
  }
  const CostEstimate reused_clean = phase_model.estimate(CompressionPath::cpu_lz4_gpu_decode, 1000);
  CHECK(phase_model.decide(CompressionMode::adaptive, 1000, true, false).path ==
        CompressionPath::cpu_lz4_gpu_decode);

  phase_model.reset(42);
  raw.dirty = true;
  cpu.dirty = true;
  for (int sample = 0; sample < 2; ++sample) {
    CHECK(phase_model.observe(42, raw));
    CHECK(phase_model.observe(42, cpu));
  }
  const CostEstimate reused_dirty = phase_model.estimate(CompressionPath::cpu_lz4_gpu_decode, 1000);
  CHECK(reused_dirty.predicted_total_us > reused_clean.predicted_total_us);
  CHECK(phase_model.decide(CompressionMode::adaptive, 1000, true, false).path ==
        CompressionPath::raw);
  const CostDecision capacity = phase_model.decide(CompressionMode::capacity, 1000, true, false);
  CHECK(capacity.path == CompressionPath::cpu_lz4_gpu_decode);
  CHECK(capacity.capacity_override);
}

void cost_model_calibration_and_generation_tests() {
  using namespace xvram::residency;

  CompressionCostModel model(0.5);
  CHECK(model.observe(21, observation(CompressionPath::raw, true)));
  CHECK(model.observe(21, observation(CompressionPath::cpu_lz4_gpu_decode, true)));
  CostDecision decision = model.decide(CompressionMode::adaptive, 1000, true, false);
  CHECK(decision.calibration_required);
  CHECK(decision.path == CompressionPath::raw);

  CHECK(model.observe(21, observation(CompressionPath::raw, true)));
  decision = model.decide(CompressionMode::adaptive, 1000, true, false);
  CHECK(decision.calibration_required);
  CHECK(decision.path == CompressionPath::raw);

  CHECK(model.observe(21, observation(CompressionPath::cpu_lz4_gpu_decode, true)));
  decision = model.decide(CompressionMode::adaptive, 1000, true, false);
  CHECK(!decision.calibration_required);
  CHECK(decision.path == CompressionPath::cpu_lz4_gpu_decode);

  const CostPathMetrics raw_before = model.metrics(CompressionPath::raw);
  const CostPathMetrics candidate_before = model.metrics(CompressionPath::cpu_lz4_gpu_decode);
  model.reset(22);
  CHECK(model.generation() == 22);
  CHECK(!model.never_compress());
  CHECK(model.unfavorable_probe_count() == 0);
  CHECK(model.metrics(CompressionPath::raw).samples == raw_before.samples);
  CHECK(model.metrics(CompressionPath::raw).transfer_us_per_stored_byte ==
        raw_before.transfer_us_per_stored_byte);
  CHECK(model.metrics(CompressionPath::cpu_lz4_gpu_decode).samples == candidate_before.samples);
  CHECK(model.metrics(CompressionPath::cpu_lz4_gpu_decode).compression_ratio ==
        candidate_before.compression_ratio);
  decision = model.decide(CompressionMode::adaptive, 1000, true, false);
  CHECK(!decision.calibration_required);
  CHECK(decision.path == CompressionPath::cpu_lz4_gpu_decode);

  // The time-margin rule is inclusive and uses the larger of 10% and 50 microseconds.
  CHECK(model.record_probe(22, CompressionPath::cpu_lz4_gpu_decode, 100.0, 50.0));
  CHECK(!model.record_probe(22, CompressionPath::cpu_lz4_gpu_decode, 100.0, 50.001));
  CHECK(model.record_probe(22, CompressionPath::cpu_lz4_gpu_decode, 1000.0, 900.0));
  CHECK(!model.record_probe(22, CompressionPath::cpu_lz4_gpu_decode, 1000.0, 900.001));
  CHECK(model.unfavorable_probe_count() == 1);

  CHECK(!model.record_probe(22, CompressionPath::cpu_lz4_gpu_decode, 1000.0, 901.0));
  CHECK(!model.record_probe(22, CompressionPath::cpu_lz4_gpu_decode, 1000.0, 950.0));
  CHECK(model.never_compress());
  CHECK(model.unfavorable_probe_count() == 3);
  CHECK(model.decide(CompressionMode::adaptive, 1000, true, false).path == CompressionPath::raw);

  model.reset(23);
  CHECK(model.generation() == 23);
  CHECK(!model.never_compress());
  CHECK(model.unfavorable_probe_count() == 0);
  CHECK(model.metrics(CompressionPath::raw).samples == raw_before.samples);
  CHECK(model.metrics(CompressionPath::cpu_lz4_gpu_decode).samples == candidate_before.samples);
  decision = model.decide(CompressionMode::adaptive, 1000, true, false);
  CHECK(!decision.calibration_required);
  CHECK(decision.path == CompressionPath::cpu_lz4_gpu_decode);
}

void reuse_probe_forecast_tests() {
  using namespace xvram::residency;

  const auto one_touch = forecast_clean_reuse_probe(100.0, 200.0, 10.0, 20.0, 0U);
  CHECK(one_touch.has_value());
  if (one_touch.has_value()) {
    CHECK(one_touch->raw_total_us == 100.0);
    CHECK(one_touch->candidate_total_us == 230.0);
  }

  const auto reused = forecast_clean_reuse_probe(100.0, 200.0, 10.0, 20.0, 9U);
  CHECK(reused.has_value());
  if (reused.has_value()) {
    CHECK(reused->raw_total_us == 1000.0);
    CHECK(reused->candidate_total_us == 500.0);
  }

  CHECK(!forecast_clean_reuse_probe(std::numeric_limits<double>::infinity(), 0.0, 0.0, 0.0, 0U)
             .has_value());
}

} // namespace

int main() {
  backing_store_tests();
  budget_tests();
  eager_raw_registration_test();
  spill_to_authority_commit_test();
  raw_block_ownership_and_tail_test();
  cost_model_tests();
  cost_model_context_and_phase_switch_tests();
  cost_model_calibration_and_generation_tests();
  reuse_probe_forecast_tests();
  if (failures != 0) {
    std::cerr << failures << " compression-core test(s) failed\n";
    return 1;
  }
  std::cout << "compression core tests passed\n";
  return 0;
}
