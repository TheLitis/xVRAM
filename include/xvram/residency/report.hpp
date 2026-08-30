#pragma once

#include "xvram/probe/report.hpp"

#include <cstdint>
#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

namespace xvram::residency {

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
};

struct Configuration {
  std::optional<std::int32_t> requested_device_ordinal;
  std::optional<std::uint64_t> requested_logical_bytes;
  std::optional<std::uint64_t> effective_logical_bytes;
  std::optional<std::uint64_t> requested_chunk_bytes;
  std::optional<std::uint64_t> effective_chunk_bytes;
  std::optional<std::uint64_t> requested_cache_target_bytes;
  std::optional<std::uint64_t> initial_cache_target_bytes;
  std::uint32_t staging_slots = 4;
  std::string policy = "clock";
  std::uint32_t prefetch_distance = 2;
  std::string scenario = "suite";
  std::uint32_t passes = 2;
  std::optional<std::uint64_t> pressure_bytes;
  std::uint64_t host_headroom_bytes = 0;
  std::uint64_t device_headroom_bytes = 0;
  std::uint64_t budget_poll_ms = 100;
  std::uint64_t stall_timeout_ms = 5'000;
  std::uint64_t timeout_ms = 300'000;
  std::string seed_hex = "585652414d503032";
  std::string sizing_mode = "auto";
  bool trace_enabled = false;
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

struct WorkloadResult {
  std::string scenario;
  std::string policy;
  std::string status = "not_run";
  std::optional<std::uint64_t> operations_retired;
  std::optional<std::uint32_t> passes_completed;
  std::optional<std::uint64_t> logical_bytes;
  std::optional<std::uint64_t> logical_chunk_count;
  std::optional<std::uint64_t> maximum_working_set_bytes;
  std::optional<std::uint64_t> read_operations;
  std::optional<std::uint64_t> read_write_operations;
  std::optional<std::uint64_t> write_only_operations;
  std::optional<std::uint64_t> cache_hits;
  std::optional<std::uint64_t> cache_misses;
  std::optional<double> cache_hit_rate;
  std::optional<std::uint64_t> bytes_h2d;
  std::optional<std::uint64_t> bytes_d2h;
  std::optional<std::uint64_t> clean_evictions;
  std::optional<std::uint64_t> dirty_evictions;
  std::optional<std::uint64_t> writebacks_completed;
  std::optional<std::uint64_t> prefetch_issued;
  std::optional<std::uint64_t> prefetch_useful;
  std::optional<std::uint64_t> prefetch_wasted;
  std::optional<std::uint64_t> prefetch_cancelled;
  std::optional<std::uint64_t> prefetch_promoted;
  std::optional<std::uint64_t> sequential_bypasses;
  std::optional<std::uint64_t> target_shrink_count;
  std::optional<std::uint64_t> target_grow_count;
  std::optional<std::uint64_t> mapping_count;
  std::optional<std::uint64_t> unmap_count;
  std::optional<std::uint64_t> set_access_count;
  std::optional<std::uint64_t> handle_reuse_count;
  std::optional<std::uint64_t> event_boundary_count;
  std::optional<std::uint64_t> unsafe_remap_count;
  std::optional<std::uint64_t> unsafe_transition_count;
  std::optional<bool> stable_addresses_verified;
  std::optional<bool> full_verification_completed;
  std::optional<bool> matches_cpu;
  std::optional<double> elapsed_ms;
  std::optional<double> throughput_gib_per_second;
  std::optional<std::string> expected_digest128;
  std::optional<std::string> output_digest128;
  std::optional<std::uint64_t> mismatch_count;
  std::optional<std::uint64_t> first_mismatch_byte_offset;
};

struct CacheStatistics {
  std::optional<std::uint64_t> target_bytes_initial;
  std::optional<std::uint64_t> target_bytes_minimum;
  std::optional<std::uint64_t> target_bytes_maximum;
  std::optional<std::uint64_t> target_bytes_end;
  std::optional<std::uint64_t> resident_bytes_peak;
  std::optional<std::uint64_t> pinned_staging_bytes;
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
  std::optional<std::uint64_t> prefetch_issued;
  std::optional<std::uint64_t> prefetch_useful;
  std::optional<std::uint64_t> prefetch_wasted;
  std::optional<std::uint64_t> prefetch_cancelled;
  std::optional<std::uint64_t> prefetch_promoted;
  std::optional<std::uint64_t> sequential_bypasses;
  std::optional<std::uint64_t> target_shrink_count;
  std::optional<std::uint64_t> target_grow_count;
  std::optional<std::uint64_t> target_oom_retry_count;
  std::optional<std::uint64_t> maximum_working_set_bytes;
};

struct Telemetry {
  std::optional<double> total_elapsed_ms;
  std::optional<std::uint64_t> bytes_h2d;
  std::optional<std::uint64_t> bytes_d2h;
  std::optional<std::uint64_t> budget_sample_count;
  std::optional<std::uint64_t> cuda_free_bytes_minimum;
  std::optional<std::uint64_t> cuda_free_bytes_end;
  std::optional<std::uint64_t> wddm_available_bytes_minimum;
  std::optional<std::uint64_t> wddm_available_bytes_end;
  std::optional<double> resident_occupancy_peak;
  std::optional<double> resident_occupancy_mean;
  TimingSummary remap_timing;
  TimingSummary h2d_timing;
  TimingSummary kernel_timing;
  TimingSummary d2h_timing;
  TimingSummary writeback_timing;
  std::optional<std::uint64_t> trace_records_emitted;
  std::optional<std::uint64_t> trace_records_dropped;
  std::optional<bool> trace_complete;
};

struct Proof {
  std::optional<std::uint64_t> logical_bytes;
  std::optional<std::uint64_t> chunk_bytes;
  std::optional<std::uint64_t> logical_chunk_count;
  std::optional<std::uint64_t> maximum_cache_target_bytes;
  std::optional<std::uint64_t> pinned_staging_bytes;
  std::uint32_t kernel_module_version = 0;
  std::string kernel_module_sha256;
  std::string pattern_version;
  std::optional<std::string> cpu_reference_digest128;
  std::optional<bool> logical_data_exceeds_vram;
  std::optional<bool> cache_smaller_than_logical;
  std::optional<bool> handles_reused;
  std::optional<bool> stable_virtual_addresses_verified;
  std::optional<bool> set_access_after_map_verified;
  std::optional<bool> event_boundaries_verified;
  std::optional<bool> no_physical_aliases_verified;
  std::optional<bool> dirty_writeback_verified;
  std::optional<bool> staging_pool_bounded;
  std::optional<bool> cache_target_respected;
  std::optional<bool> all_workloads_match_cpu;
  std::optional<bool> policies_match;
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
  std::optional<std::string> policy;
  std::optional<std::string> scenario;
  std::optional<std::uint64_t> allocation_id;
  std::optional<std::uint64_t> chunk_index;
  std::optional<std::uint64_t> logical_byte_offset;
};

struct Cleanup {
  std::optional<bool> complete;
  std::optional<bool> transactions_drained;
  std::optional<bool> prefetch_drained;
  std::optional<bool> writebacks_completed;
  std::optional<bool> events_drained;
  std::optional<bool> events_destroyed;
  std::optional<bool> streams_destroyed;
  std::optional<bool> module_unloaded;
  std::optional<bool> mappings_removed;
  std::optional<bool> physical_handles_released;
  std::optional<bool> device_allocations_released;
  std::optional<bool> virtual_reservations_released;
  std::optional<bool> pinned_staging_released;
  std::optional<bool> host_backing_released;
  std::optional<bool> context_destroyed;
  std::optional<bool> trace_closed;
  std::optional<bool> worker_terminated;
};

struct Report {
  std::uint32_t schema_version = 1;
  std::string report_type = "xvram.residency_cache";
  std::string generated_at_utc;
  probe::BuildInfo build;
  probe::SystemInfo system;
  std::optional<DeviceInfo> device;
  Configuration configuration;
  std::vector<WorkloadResult> workloads;
  CacheStatistics cache;
  Telemetry telemetry;
  Proof proof;
  Outcome outcome;
  Cleanup cleanup;
  std::vector<probe::Diagnostic> diagnostics;
};

struct TraceRecord {
  std::uint32_t schema_version = 1;
  std::string report_type = "xvram.residency_trace";
  std::uint64_t sequence = 0;
  std::uint64_t monotonic_time_ns = 0;
  std::string event;
  std::optional<std::uint64_t> allocation_id;
  std::optional<std::uint64_t> chunk_index;
  std::optional<std::uint64_t> operation_id;
  std::optional<std::uint64_t> transaction_id;
  std::optional<std::string> from_state;
  std::optional<std::string> to_state;
  std::optional<std::uint64_t> bytes;
  std::optional<std::string> reason;
  std::optional<std::string> policy;
  std::optional<bool> speculative;
  std::optional<std::uint64_t> generation;
};

void write_json(const Report& report, std::ostream& output, bool pretty);
void write_text(const Report& report, std::ostream& output);
void write_trace_json(const TraceRecord& record, std::ostream& output);

} // namespace xvram::residency
