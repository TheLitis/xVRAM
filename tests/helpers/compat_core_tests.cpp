#include "compat_bench/final_validation.hpp"
#include "compat_bench/options.hpp"
#include "compat_bench/pattern.hpp"
#include "compat_bench/report.hpp"
#include "compat_bench/worker_protocol.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {
int failures = 0;
void check(const bool condition, const char* message) {
  if (!condition) {
    ++failures;
    std::cerr << message << '\n';
  }
}
void pattern_tests() {
  using namespace xvram::compat_bench;
  for (std::uint64_t seed : {0ULL, 1ULL, 0x585652414d503661ULL}) {
    for (std::uint64_t row = 0; row < 9; ++row)
      for (std::uint64_t column = 0; column < 7; ++column)
        for (std::uint64_t k = 1; k < 259; ++k)
          check(reference_dot(row, column, k, seed, true) ==
                    reference_dot(row, column, k, seed, false),
                "analytic reference disagrees with full CPU dot");
  }
  for (bool a : {false, true})
    for (bool b : {false, true}) {
      Shape shape{7, 5, 11, 3, 13, a, b};
      for (auto op : {Operand::a, Operand::b, Operand::c}) {
        const auto matrix = matrix_storage(shape, op);
        check(matrix.has_value(), "valid padded matrix rejected");
        std::vector<float> whole(static_cast<std::size_t>(matrix->elements)), parts(whole.size());
        fill_pattern(whole, 0, shape, op, 71);
        for (std::size_t offset = 0; offset < parts.size();) {
          const auto count = std::min<std::size_t>(17, parts.size() - offset);
          fill_pattern(std::span<float>(parts.data() + offset, count), offset, shape, op, 71);
          offset += count;
        }
        check(whole == parts, "streaming pattern tails are inconsistent");
        check(whole.front() == padding_value && whole.back() == padding_value,
              "prefix/suffix padding was overwritten");
        for (std::uint64_t column = 0; column < matrix->columns; ++column)
          for (std::uint64_t row = 0; row < matrix->rows; ++row) {
            const auto actual =
                whole[static_cast<std::size_t>(matrix->offset + column * matrix->ld + row)];
            const auto expected =
                op == Operand::a   ? pattern_a(a ? column : row, a ? row : column, 71)
                : op == Operand::b ? pattern_b(b ? column : row, b ? row : column, 71)
                                   : pattern_c(row, column);
            check(actual == expected, "transpose physical layout mismatch");
          }
      }
      check(operand_bytes(shape) == static_cast<std::uint64_t>((7 * 11 + 11 * 5 + 7 * 5) * 4),
            "logical bytes must exclude padding");
    }
  check(!matrix_storage(Shape{0, 1, 1, 0, 0, false, false}, Operand::a), "zero shape accepted");
  check(!matrix_storage(Shape{INT32_MAX, 1, 1, 1, 0, false, false}, Operand::a),
        "leading dimension overflow accepted");
  check(!matrix_storage(Shape{1, 1, 1, 0, UINT64_MAX, false, false}, Operand::a),
        "prefix/tail overflow accepted");
  check(!operand_bytes(Shape{UINT64_MAX, UINT64_MAX, UINT64_MAX, 0, 0, false, false}),
        "logical multiplication overflow accepted");
  Digest x, y;
  for (float v : {1.0F, -1.0F, 0.5F, 7.0F}) {
    x.add(v);
    y.add(v);
  }
  check(x.string() == y.string() && x.string().size() == 32, "digest nondeterministic");
  y.add(1.0F);
  check(x.string() != y.string(), "digest failed to change");
}
void protocol_tests() {
  using namespace xvram::compat_bench;
  std::ostringstream wire;
  check(write_worker_frame(wire, WorkerFrameType::plan, 1, "{}"), "plan encode failed");
  check(write_worker_frame(wire, WorkerFrameType::heartbeat, 2, "{}"), "heartbeat encode failed");
  check(write_worker_frame(wire, WorkerFrameType::progress, 3, "1"), "progress encode failed");
  check(write_worker_frame(wire, WorkerFrameType::trace, 4, "{}\n"), "trace encode failed");
  const FinalWorkerPayload payload{23, "{}", "skipped"};
  check(write_worker_frame(wire, WorkerFrameType::final, 5, encode_final_worker_payload(payload)),
        "final encode failed");
  const auto bytes = wire.str();
  check(bytes.starts_with("XVI1"), "wrong wire magic");
  for (std::size_t fragment = 1; fragment < 24; ++fragment) {
    WorkerFrameDecoder decoder;
    std::vector<WorkerFrame> frames;
    std::string error;
    for (std::size_t offset = 0; offset < bytes.size();) {
      const auto size = std::min(fragment, bytes.size() - offset);
      check(decoder.append(bytes.data() + offset, size, frames, error),
            "fragmented decoding failed");
      offset += size;
    }
    check(decoder.finish(error) && frames.size() == 5, "frame termination failed");
  }
  std::ostringstream malformed;
  check(write_worker_frame(malformed, WorkerFrameType::plan, 1, "{}"),
        "negative plan setup failed");
  check(write_worker_frame(malformed, WorkerFrameType::progress, 1, "1"),
        "negative progress setup failed");
  WorkerFrameDecoder decoder;
  std::vector<WorkerFrame> frames;
  std::string error;
  const auto repeated = malformed.str();
  check(!decoder.append(repeated.data(), repeated.size(), frames, error),
        "repeated frame sequence accepted");
  check(!write_worker_frame(malformed, WorkerFrameType::trace, 2,
                            std::string(maximum_trace_batch_bytes + 1, 'x')),
        "oversized trace accepted");
  check(encode_final_worker_payload({63, "{}", ""}).empty(), "invalid final code accepted");
}
void option_tests() {
  using namespace xvram::compat_bench;
  Options options;
  options.m = 37;
  options.n = 19;
  options.k = 67;
  options.transpose_a = true;
  options.padding = 7;
  options.offset_elements = 11;
  options.alpha = 1.25F;
  options.beta = -0.5F;
  options.seed = UINT64_MAX;
  options.policy = "lru";
  const auto args = worker_arguments(options);
  std::vector<std::string> owned{"bench"};
  owned.insert(owned.end(), args.begin(), args.end());
  std::vector<char*> pointers;
  for (auto& v : owned)
    pointers.push_back(v.data());
  Options copy;
  std::string error;
  check(parse_options(static_cast<int>(pointers.size()), pointers.data(), copy, error),
        "options roundtrip parse failed");
  check(copy.seed == options.seed && copy.alpha == options.alpha && copy.beta == options.beta &&
            copy.m == options.m && copy.policy == options.policy,
        "options roundtrip changed values");
}
void final_json_tests() {
  using namespace xvram::compat_bench;
  const auto report = base_report({});
  const auto json = report_json(report, false);
  std::string output, error;
  check(finalize_json(json, 23, true, output, error), "valid final report rejected");
  check(output.find("\"worker_terminated\": true") != std::string::npos,
        "reaped final report missing typed cleanup amendment");
  check(output.find("\n  ") != std::string::npos, "pretty final serialization ignored");
  check(!finalize_json(json, 27, false, output, error),
        "contradictory report/wire outcome accepted");
  auto duplicate = json;
  duplicate.insert(1, "\"schema_version\":1,");
  check(!finalize_json(duplicate, 23, false, output, error), "duplicate root JSON key accepted");
  duplicate = json;
  duplicate.insert(1, "\"schema_ver\\u0073ion\":1,");
  check(!finalize_json(duplicate, 23, false, output, error), "escaped duplicate JSON key accepted");
  for (const char* malformed :
       {"{broken", "[]", "{\"x\":01}", "{\"x\":NaN}", "{\"x\":1e999}", "{\"x\":\"\\uD800\"}",
        "{\"x\":true,}", "{\"x\":[1,]}", "{\"x\":true} trailing"})
    check(!finalize_json(malformed, 23, false, output, error), "malformed JSON accepted");
  check(!finalize_json(std::string(40, '[') + "0" + std::string(40, ']'), 23, false, output, error),
        "excessively nested final report accepted");
}
void fixtures() {
  using namespace xvram::compat_bench;
  for (int code : {0, 23, 24, 25, 26, 27, 70, 74}) {
    auto r = base_report({});
    r.exit_code = code;
    r.reason = "contract_fixture";
    r.message = "synthetic no-driver contract fixture";
    r.cleanup["worker_terminated"] = true;
    r.cleanup["trace_closed"] = true;
    if (code == 0) {
      for (auto& [key, value] : r.proof) {
        static_cast<void>(key);
        value = true;
      }
      for (auto& [key, value] : r.cleanup) {
        static_cast<void>(key);
        value = true;
      }
      r.telemetry_observed = true;
      r.telemetry.total_vram_bytes = 8ULL << 30U;
      r.telemetry.gemm_calls = 2;
      // Many compute transactions can reuse three resident mappings. Unmap boundaries
      // must reconcile with unmaps, independently of transaction retirement counts.
      r.telemetry.tiles_retired = 1000;
      r.telemetry.tiles_submitted = 1000;
      r.telemetry.runtime.transactions_completed = 1000;
      r.telemetry.runtime.event_boundaries = 3;
      r.telemetry.runtime.mappings = 3;
      r.telemetry.runtime.unmaps = 3;
      r.telemetry.runtime.set_access_calls = 3;
      r.telemetry.runtime.cache_hits = 2997;
      r.telemetry.runtime.cache_misses = 3;
      r.telemetry.calls_attempted = 20;
      r.telemetry.calls_submitted = 19;
      r.telemetry.calls_completed = 18;
      r.telemetry.calls_rejected = 2;
      r.telemetry.effective_chunk_bytes = 64ULL << 20U;
      r.telemetry.host_store_cap_bytes = 512ULL << 20U;
      r.telemetry.runtime.pinned_staging_bytes = 256ULL << 20U;
      r.telemetry.runtime.workspace_bytes = 4ULL << 20U;
      r.telemetry.runtime.cache_target_bytes = 192ULL << 20U;
      r.telemetry.runtime.cache_target_minimum_bytes = 192ULL << 20U;
      r.telemetry.runtime.cache_target_maximum_bytes = 192ULL << 20U;
      r.telemetry.runtime.resident_bytes_peak = 192ULL << 20U;
      Workload w;
      w.name = "fixture";
      w.status = "completed";
      w.shape = {100, 100, 50, 0, 0, false, false};
      w.logical_bytes = *operand_bytes(w.shape);
      w.storage_bytes = w.logical_bytes;
      w.passes_completed = 2;
      w.output_elements_checked = 10000;
      w.digest = std::string(32, 'a');
      w.reference_digest = w.digest;
      w.reference_kind = "cpu_fp64_full";
      w.native_baseline_equal = true;
      w.native_baseline_ms = 1.0;
      w.tiles_retired = 1000;
      w.mappings = 3;
      w.unmaps = 3;
      w.pass_timings_ms = {1, 2};
      r.workloads.push_back(w);
      r.trace_complete = true;
    }
    std::cout << report_json(r, false) << '\n';
  }
}
void event_proof_tests() {
  using namespace xvram::compat_bench;
  auto report = base_report({});
  report.telemetry_observed = true;
  report.telemetry.tiles_submitted = 1000;
  report.telemetry.tiles_retired = 1000;
  report.telemetry.runtime.transactions_completed = 1000;
  report.telemetry.runtime.unmaps = 3;
  report.telemetry.runtime.event_boundaries = 3;
  report.telemetry.cleanup_events_drained = 1;
  complete_proof(report);
  check(report.proof.at("event_safe") == true,
        "resident hits must not require another unmap boundary per transaction");
  --report.telemetry.runtime.transactions_completed;
  complete_proof(report);
  check(report.proof.at("event_safe") == false,
        "unretired transaction incorrectly proved event-safe");
  ++report.telemetry.runtime.transactions_completed;
  --report.telemetry.runtime.event_boundaries;
  complete_proof(report);
  check(report.proof.at("event_safe") == false,
        "unmap missing event boundary incorrectly proved event-safe");
}
} // namespace
int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--emit-fixtures") {
    fixtures();
    return 0;
  }
  pattern_tests();
  protocol_tests();
  option_tests();
  final_json_tests();
  event_proof_tests();
  return failures == 0 ? 0 : 1;
}
