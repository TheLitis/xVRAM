#pragma once

#include "xvram/probe/report.hpp"

#include <cstdint>
#include <iosfwd>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace xvram::compression {

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
  std::optional<std::uint64_t> wddm_available_bytes_minimum;
  std::optional<std::uint32_t> compute_capability_major;
  std::optional<std::uint32_t> compute_capability_minor;
  bool vmm_supported = false;
  bool uva_supported = false;
};

struct Configuration {
  std::optional<std::int32_t> requested_device_ordinal;
  std::optional<std::uint64_t> requested_logical_bytes;
  std::optional<std::uint64_t> effective_logical_bytes;
  std::optional<std::uint64_t> requested_chunk_bytes;
  std::optional<std::uint64_t> effective_chunk_bytes;
  std::optional<std::uint64_t> requested_cache_target_bytes;
  std::optional<std::uint64_t> initial_cache_target_bytes;
  std::string compression_policy = "adaptive";
  std::string path = "auto";
  std::string codec = "auto";
  std::string replacement_policy = "clock";
  std::string scenario = "suite";
  std::uint32_t passes = 2;
  std::uint32_t warmup_passes = 2;
  std::uint32_t measurement_passes = 5;
  std::optional<std::uint64_t> host_store_cap_bytes;
  std::uint64_t host_headroom_bytes = 0;
  std::uint64_t device_headroom_bytes = 0;
  std::uint64_t compression_scratch_cap_bytes = 256ULL * 1024ULL * 1024ULL;
  std::uint32_t codec_slots = 2;
  std::uint32_t codec_workers = 2;
  std::uint32_t staging_slots = 4;
  std::uint32_t prefetch_distance = 2;
  std::uint64_t budget_poll_ms = 100;
  std::uint64_t stall_timeout_ms = 5'000;
  std::uint64_t timeout_ms = 900'000;
  std::string seed_hex = "585652414d503035";
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
  std::string compression_policy;
  std::string path;
  std::string replacement_policy;
  std::string status = "not_run";
  std::optional<std::uint64_t> logical_bytes;
  std::optional<std::uint64_t> logical_chunk_count;
  std::optional<std::uint64_t> operations_retired;
  std::optional<std::uint32_t> warmup_passes_completed;
  std::optional<std::uint32_t> measurement_passes_completed;
  std::optional<std::uint64_t> raw_input_bytes;
  std::optional<std::uint64_t> stored_bytes;
  std::optional<double> stored_ratio;
  std::optional<std::uint64_t> logical_h2d_bytes;
  std::optional<std::uint64_t> pcie_h2d_bytes;
  std::optional<std::uint64_t> logical_d2h_bytes;
  std::optional<std::uint64_t> pcie_d2h_bytes;
  std::optional<std::uint64_t> raw_path_decisions;
  std::optional<std::uint64_t> cpu_lz4_gpu_decisions;
  std::optional<std::uint64_t> gpu_lz4_decisions;
  std::optional<std::uint64_t> never_compress_decisions;
  std::optional<std::uint64_t> fallback_count;
  std::optional<std::uint64_t> mappings;
  std::optional<std::uint64_t> set_access;
  std::optional<std::uint64_t> unmaps;
  std::optional<std::uint64_t> events_recorded;
  std::optional<std::uint64_t> events_retired;
  std::optional<std::uint64_t> unsafe_remaps;
  std::optional<std::uint64_t> unsafe_transitions;
  std::optional<double> elapsed_ms;
  std::optional<std::string> expected_digest128;
  std::optional<std::string> output_digest128;
  std::optional<std::uint64_t> mismatch_count;
  std::optional<std::uint64_t> first_mismatch_byte_offset;
};

struct BackingStatistics {
  std::optional<std::uint64_t> logical_bytes;
  std::optional<std::uint64_t> host_store_cap_bytes;
  std::optional<std::uint64_t> host_bytes_current;
  std::optional<std::uint64_t> host_bytes_peak;
  std::optional<std::uint64_t> host_budget_bytes_current;
  std::optional<std::uint64_t> host_budget_bytes_peak;
  std::optional<std::uint64_t> raw_bytes_current;
  std::optional<std::uint64_t> raw_bytes_peak;
  std::optional<std::uint64_t> compressed_bytes_current;
  std::optional<std::uint64_t> compressed_bytes_peak;
  std::optional<std::uint64_t> spill_reserved_bytes_peak;
  std::optional<std::uint64_t> conversion_scratch_bytes_peak;
  std::optional<std::uint64_t> invalid_chunks;
  std::optional<std::uint64_t> implicit_zero_chunks;
  std::optional<std::uint64_t> raw_chunks;
  std::optional<std::uint64_t> lz4_chunks;
  std::optional<std::uint64_t> generations_created;
  std::optional<std::uint64_t> generations_committed;
  std::optional<std::uint64_t> generations_discarded;
  std::optional<std::uint64_t> atomic_commit_failures;
  std::optional<std::uint64_t> expansion_rejections;
  std::optional<double> effective_stored_ratio;
};

struct CodecStatistics {
  std::string container_version = "xvram_lz4_blocks_v1";
  std::uint64_t block_bytes = 64ULL * 1024ULL;
  std::string cpu_codec = "lz4";
  std::optional<std::string> cpu_codec_version;
  std::optional<bool> nvcomp_available;
  std::optional<std::string> nvcomp_version;
  std::optional<std::string> nvcomp_library_source;
  std::optional<std::string> nvcomp_library_sha256;
  std::optional<std::uint64_t> cpu_encode_operations;
  std::optional<std::uint64_t> cpu_decode_operations;
  std::optional<std::uint64_t> gpu_encode_operations;
  std::optional<std::uint64_t> gpu_decode_operations;
  std::optional<std::uint64_t> raw_path_decisions;
  std::optional<std::uint64_t> cpu_lz4_gpu_decisions;
  std::optional<std::uint64_t> gpu_lz4_decisions;
  std::optional<std::uint64_t> never_compress_decisions;
  std::optional<std::uint64_t> calibration_samples;
  std::optional<std::uint64_t> fallback_count;
  std::optional<std::uint64_t> verification_failures;
  std::optional<std::uint64_t> codec_slots_peak;
  std::optional<std::uint64_t> workspace_bytes_peak;
  std::optional<std::uint64_t> device_slot_bytes_peak;
  std::optional<std::uint64_t> device_slot_capacity_bytes;
  TimingSummary cpu_encode_timing;
  TimingSummary cpu_decode_timing;
  TimingSummary gpu_encode_timing;
  TimingSummary gpu_decode_timing;
  TimingSummary verification_timing;
};

struct Telemetry {
  std::optional<double> total_elapsed_ms;
  std::optional<std::uint64_t> logical_h2d_bytes;
  std::optional<std::uint64_t> pcie_h2d_bytes;
  std::optional<std::uint64_t> pcie_h2d_payload_bytes;
  std::optional<std::uint64_t> pcie_h2d_metadata_bytes;
  std::optional<std::uint64_t> logical_d2h_bytes;
  std::optional<std::uint64_t> pcie_d2h_bytes;
  std::optional<std::uint64_t> pcie_d2h_payload_bytes;
  std::optional<std::uint64_t> pcie_d2h_metadata_bytes;
  std::optional<std::uint64_t> rejected_candidate_logical_d2h_bytes;
  std::optional<std::uint64_t> mapping_count;
  std::optional<std::uint64_t> set_access_count;
  std::optional<std::uint64_t> unmap_count;
  std::optional<std::uint64_t> event_record_count;
  std::optional<std::uint64_t> event_retire_count;
  std::optional<std::uint64_t> handle_reuse_count;
  std::optional<std::uint64_t> unsafe_remap_count;
  std::optional<std::uint64_t> unsafe_transition_count;
  std::optional<std::uint64_t> writeback_count;
  std::optional<std::uint64_t> target_shrink_count;
  std::optional<std::uint64_t> target_grow_count;
  std::optional<std::uint64_t> budget_sample_count;
  std::optional<std::uint64_t> cache_target_bytes_minimum;
  std::optional<std::uint64_t> cache_target_bytes_maximum;
  std::optional<std::uint64_t> safe_device_budget_bytes_minimum;
  std::optional<std::uint64_t> managed_device_bytes_peak;
  std::optional<std::uint64_t> device_reserve_bytes_peak;
  std::optional<std::uint64_t> device_budget_violation_count;
  std::optional<std::uint64_t> trace_records_emitted;
  std::optional<std::uint64_t> trace_records_dropped;
  std::optional<bool> trace_complete;
  TimingSummary raw_transfer_timing;
  TimingSummary compressed_transfer_timing;
  TimingSummary writeback_timing;
};

struct Proof {
  std::optional<bool> logical_data_exceeds_vram;
  std::optional<bool> cache_smaller_than_logical;
  std::optional<bool> authoritative_compressed_backing_verified;
  std::optional<bool> no_expansion_stored;
  std::optional<bool> generation_atomicity_verified;
  std::optional<bool> write_admission_verified;
  std::optional<bool> gpu_encode_before_d2h_verified;
  std::optional<bool> compressed_h2d_reduction_verified;
  std::optional<bool> stable_virtual_addresses_verified;
  std::optional<bool> maps_match_set_access;
  std::optional<bool> event_boundaries_verified;
  std::optional<bool> host_budget_respected;
  std::optional<bool> device_budget_respected;
  std::optional<bool> all_workloads_match_reference;
  std::optional<bool> path_digests_match;
  std::optional<bool> policy_digests_match;
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
  std::optional<std::string> scenario;
  std::optional<std::uint64_t> allocation_id;
  std::optional<std::uint64_t> chunk_index;
  std::optional<std::uint64_t> generation;
  std::optional<std::uint64_t> logical_byte_offset;
};

struct Cleanup {
  std::optional<bool> complete;
  std::optional<bool> operations_drained;
  std::optional<bool> codec_slots_drained;
  std::optional<bool> events_drained;
  std::optional<bool> events_destroyed;
  std::optional<bool> streams_destroyed;
  std::optional<bool> mappings_removed;
  std::optional<bool> physical_handles_released;
  std::optional<bool> codec_workspace_released;
  std::optional<bool> virtual_reservations_released;
  std::optional<bool> pinned_staging_released;
  std::optional<bool> spill_reservations_released;
  std::optional<bool> host_backing_released;
  std::optional<bool> context_released;
  std::optional<bool> trace_closed;
  std::optional<bool> worker_terminated;
};

struct Report {
  std::uint32_t schema_version = 1;
  std::string report_type = "xvram.adaptive_compression";
  std::string generated_at_utc;
  probe::BuildInfo build;
  probe::SystemInfo system;
  std::optional<DeviceInfo> device;
  Configuration configuration;
  std::vector<WorkloadResult> workloads;
  BackingStatistics backing;
  CodecStatistics codec;
  Telemetry telemetry;
  Proof proof;
  Outcome outcome;
  Cleanup cleanup;
  std::vector<probe::Diagnostic> diagnostics;
};

struct TraceRecord {
  std::uint32_t schema_version = 1;
  std::string report_type = "xvram.compression_trace";
  std::uint64_t sequence = 0;
  std::uint64_t monotonic_time_ns = 0;
  std::string event;
  std::optional<std::uint64_t> allocation_id;
  std::optional<std::uint64_t> chunk_index;
  std::optional<std::uint64_t> operation_id;
  std::optional<std::uint64_t> source_generation;
  std::optional<std::uint64_t> target_generation;
  std::optional<std::uint64_t> slot_generation;
  std::optional<std::string> from_representation;
  std::optional<std::string> to_representation;
  std::optional<std::string> path;
  std::optional<std::uint64_t> logical_bytes;
  std::optional<std::uint64_t> physical_bytes;
  std::optional<std::string> reason;
  std::optional<bool> speculative;
};

void finalize_proof(Report& report);
[[nodiscard]] bool contains_forbidden_runtime_identity(std::string_view text) noexcept;
[[nodiscard]] std::vector<std::string> validate_success_semantics(const Report& report);
void write_json(const Report& report, std::ostream& output, bool pretty);
void write_text(const Report& report, std::ostream& output);
void write_trace_json(const TraceRecord& record, std::ostream& output);

} // namespace xvram::compression
