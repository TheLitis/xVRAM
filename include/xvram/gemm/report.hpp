#pragma once

#include "xvram/probe/report.hpp"

#include <cstdint>
#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

namespace xvram::gemm_bench {

struct DeviceInfo {
  std::int32_t ordinal = 0;
  std::string name;
  std::optional<std::string> uuid;
  std::optional<std::string> luid;
  std::optional<std::string> pci_bus_id;
  std::optional<std::string> driver_model;
  std::uint64_t total_memory_bytes = 0;
  std::optional<std::uint64_t> free_memory_bytes_start;
  std::optional<std::uint64_t> free_memory_bytes_end;
  std::optional<std::uint64_t> safe_device_budget_bytes;
  std::optional<std::uint64_t> wddm_budget_bytes_start;
  std::optional<std::uint64_t> wddm_usage_bytes_start;
  std::optional<std::uint64_t> wddm_available_bytes_start;
  std::optional<std::uint64_t> wddm_available_bytes_minimum;
  std::optional<std::uint64_t> wddm_available_bytes_end;
  bool vmm_supported = false;
  bool uva_supported = false;
  std::optional<std::uint64_t> minimum_granularity_bytes;
  std::optional<std::uint64_t> recommended_granularity_bytes;
  std::optional<std::uint32_t> compute_capability_major;
  std::optional<std::uint32_t> compute_capability_minor;
};

struct Configuration {
  std::optional<std::int32_t> requested_device_ordinal;
  std::optional<std::uint64_t> requested_m;
  std::optional<std::uint64_t> requested_n;
  std::optional<std::uint64_t> requested_k;
  std::string scenario = "suite";
  std::string a_data_type = "fp32";
  std::string b_data_type = "fp32";
  std::string c_data_type = "fp32";
  std::string compute_mode = "tf32";
  std::string a_layout = "row_major";
  std::string b_layout = "row_major";
  std::string c_layout = "row_major";
  std::string a_operation = "n";
  std::string b_operation = "n";
  double alpha = 1.0;
  double beta = 0.0;
  std::optional<std::uint64_t> requested_chunk_bytes;
  std::optional<std::uint64_t> effective_chunk_bytes;
  std::optional<std::uint64_t> requested_cache_target_bytes;
  std::optional<std::uint64_t> initial_cache_target_bytes;
  std::optional<std::uint64_t> requested_workspace_cap_bytes;
  std::optional<std::uint64_t> effective_workspace_cap_bytes;
  std::string context_mode = "isolated";
  std::string policy = "clock";
  std::uint32_t staging_slots = 4;
  std::uint32_t prefetch_distance = 2;
  std::uint32_t passes = 1;
  std::uint64_t budget_poll_ms = 100;
  std::uint64_t stall_timeout_ms = 5'000;
  std::uint64_t timeout_ms = 300'000;
  std::uint64_t max_tile_ms = 250;
  bool identifiers_included = false;
};

struct TimingSummary {
  std::uint64_t sample_count = 0;
  std::optional<double> total_ms;
  std::optional<double> minimum_ms;
  std::optional<double> median_ms;
  std::optional<double> p95_ms;
  std::optional<double> maximum_ms;
};

struct PlanResult {
  std::uint64_t plan_id = 0;
  std::string status = "not_run";
  std::uint64_t m = 0;
  std::uint64_t n = 0;
  std::uint64_t k = 0;
  std::string a_data_type = "fp32";
  std::string b_data_type = "fp32";
  std::string c_data_type = "fp32";
  std::string effective_compute_mode = "tf32";
  std::uint64_t tile_m = 0;
  std::uint64_t tile_n = 0;
  std::uint64_t tile_k = 0;
  std::uint64_t tile_count = 0;
  std::uint64_t maximum_working_set_bytes = 0;
  std::uint64_t workspace_bytes = 0;
  bool uses_cublas_lt = false;
  std::optional<std::uint64_t> heuristic_algorithm_count;
  std::optional<std::uint64_t> replan_count;
};

struct WorkloadResult {
  std::string name;
  std::uint64_t plan_id = 0;
  std::string status = "not_run";
  std::uint32_t passes_completed = 0;
  std::uint64_t tiles_total = 0;
  std::uint64_t tiles_retired = 0;
  std::uint64_t logical_allocation_bytes = 0;
  bool oversubscribed = false;
  std::optional<std::uint64_t> cache_hits;
  std::optional<std::uint64_t> cache_misses;
  std::optional<double> cache_hit_rate;
  std::optional<std::uint64_t> bytes_h2d;
  std::optional<std::uint64_t> bytes_d2h;
  std::optional<std::uint64_t> clean_evictions;
  std::optional<std::uint64_t> dirty_evictions;
  std::optional<std::uint64_t> writebacks_completed;
  std::optional<std::uint64_t> mapping_count;
  std::optional<std::uint64_t> unmap_count;
  std::optional<std::uint64_t> set_access_count;
  std::optional<std::uint64_t> handle_reuse_count;
  std::optional<std::uint64_t> event_boundary_count;
  std::optional<std::uint64_t> unsafe_remap_count;
  std::optional<std::uint64_t> unsafe_transition_count;
  std::optional<double> elapsed_ms;
  std::optional<double> achieved_tflops;
  std::optional<std::string> output_digest128;
};

struct Numerics {
  std::string validation_mode = "none";
  std::string reference_precision = "none";
  std::optional<double> absolute_tolerance;
  std::optional<double> relative_tolerance;
  std::optional<double> maximum_absolute_error;
  std::optional<double> maximum_relative_error;
  std::optional<std::uint64_t> elements_verified;
  std::optional<std::uint64_t> mismatch_count;
  std::optional<std::uint64_t> first_mismatch_byte_offset;
  std::optional<std::string> expected_digest128;
  std::optional<std::string> output_digest128;
  std::optional<bool> all_results_match;
};

struct CacheStatistics {
  std::optional<std::uint64_t> target_bytes_initial;
  std::optional<std::uint64_t> target_bytes_minimum;
  std::optional<std::uint64_t> target_bytes_maximum;
  std::optional<std::uint64_t> target_bytes_end;
  std::optional<std::uint64_t> resident_bytes_peak;
  std::optional<std::uint64_t> pinned_staging_bytes;
  std::optional<std::uint64_t> workspace_bytes_peak;
  std::optional<std::uint64_t> physical_frame_count_peak;
  std::optional<std::uint64_t> physical_handle_create_count;
  std::optional<std::uint64_t> physical_handle_release_count;
  std::optional<std::uint64_t> handle_reuse_count;
  std::optional<std::uint64_t> mapping_count;
  std::optional<std::uint64_t> unmap_count;
  std::optional<std::uint64_t> set_access_count;
  std::optional<std::uint64_t> event_boundary_count;
  std::optional<std::uint64_t> unsafe_remap_count;
  std::optional<std::uint64_t> unsafe_transition_count;
  std::optional<std::uint64_t> cache_hits;
  std::optional<std::uint64_t> cache_misses;
  std::optional<double> cache_hit_rate;
  std::optional<std::uint64_t> clean_evictions;
  std::optional<std::uint64_t> dirty_evictions;
  std::optional<std::uint64_t> writebacks_completed;
  std::optional<std::uint64_t> target_shrink_count;
  std::optional<std::uint64_t> target_grow_count;
  std::optional<std::uint64_t> target_oom_retry_count;
};

struct Telemetry {
  std::optional<std::uint32_t> cuda_driver_version;
  std::optional<std::uint32_t> cublas_version;
  std::optional<std::uint32_t> cublas_lt_version;
  std::optional<bool> cublas_lt_available;
  std::optional<std::string> cublas_library_source;
  std::optional<std::uint64_t> algorithms_selected;
  std::optional<std::uint64_t> algorithm_cache_hits;
  std::optional<std::uint64_t> budget_sample_count;
  std::optional<std::uint64_t> cuda_free_bytes_minimum;
  std::optional<std::uint64_t> cuda_free_bytes_end;
  std::optional<std::uint64_t> wddm_available_bytes_minimum;
  std::optional<std::uint64_t> wddm_available_bytes_end;
  std::optional<std::uint64_t> replans;
  std::optional<std::uint64_t> watchdog_rejections;
  std::optional<double> total_elapsed_ms;
  TimingSummary plan_timing;
  TimingSummary tile_timing;
  TimingSummary gemm_timing;
  TimingSummary h2d_timing;
  TimingSummary d2h_timing;
  TimingSummary writeback_timing;
};

struct Proof {
  std::optional<std::uint64_t> logical_allocation_bytes;
  std::optional<std::uint64_t> total_vram_bytes;
  std::optional<std::uint64_t> maximum_cache_target_bytes;
  std::optional<std::uint64_t> maximum_working_set_bytes;
  std::optional<std::uint64_t> workspace_cap_bytes;
  std::optional<bool> logical_data_exceeds_vram;
  std::optional<bool> cache_smaller_than_logical;
  std::optional<bool> tiled_execution_verified;
  std::optional<bool> handles_reused;
  std::optional<bool> stable_virtual_addresses_verified;
  std::optional<bool> set_access_after_map_verified;
  std::optional<bool> event_boundaries_verified;
  std::optional<bool> no_physical_aliases_verified;
  std::optional<bool> dirty_writeback_verified;
  std::optional<bool> cache_target_respected;
  std::optional<bool> workspace_bounded;
  std::optional<bool> all_workloads_match_reference;
  std::optional<bool> raw_virtual_addresses_omitted;
};

struct Outcome {
  std::string status = "skipped";
  std::optional<std::string> reason;
  std::int32_t exit_code = 23;
  std::optional<std::string> stage;
  std::optional<std::string> operation;
  std::optional<std::int64_t> native_code;
  std::optional<std::string> native_name;
  std::optional<std::string> message;
  std::optional<std::uint64_t> plan_id;
  std::optional<std::uint64_t> tile_sequence;
  std::optional<std::uint64_t> logical_byte_offset;
};

struct Cleanup {
  std::optional<bool> complete;
  std::optional<bool> operations_drained;
  std::optional<bool> events_drained;
  std::optional<bool> events_destroyed;
  std::optional<bool> streams_destroyed;
  std::optional<bool> cublas_handles_destroyed;
  std::optional<bool> mappings_removed;
  std::optional<bool> physical_handles_released;
  std::optional<bool> workspace_released;
  std::optional<bool> virtual_reservations_released;
  std::optional<bool> pinned_staging_released;
  std::optional<bool> host_backing_released;
  std::optional<bool> context_released;
  std::optional<bool> worker_terminated;
};

struct Report {
  std::uint32_t schema_version = 1;
  std::string report_type = "xvram.gemm_bench";
  std::string generated_at_utc;
  probe::BuildInfo build;
  probe::SystemInfo system;
  std::optional<DeviceInfo> device;
  Configuration configuration;
  std::vector<PlanResult> plans;
  std::vector<WorkloadResult> workloads;
  Numerics numerics;
  CacheStatistics cache;
  Telemetry telemetry;
  Proof proof;
  Outcome outcome;
  Cleanup cleanup;
  std::vector<probe::Diagnostic> diagnostics;
};

void write_json(const Report& report, std::ostream& output, bool pretty);
void write_text(const Report& report, std::ostream& output);

} // namespace xvram::gemm_bench
