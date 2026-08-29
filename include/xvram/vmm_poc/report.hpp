#pragma once

#include "xvram/probe/report.hpp"

#include <cstdint>
#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

namespace xvram::vmm_poc {

struct DeviceInfo {
  std::int32_t ordinal = 0;
  std::string name;
  std::optional<std::string> uuid;
  std::optional<std::string> luid;
  std::optional<std::string> pci_bus_id;
  std::optional<std::string> driver_model;
  std::uint64_t total_memory_bytes = 0;
  std::uint64_t free_memory_bytes_start = 0;
  std::optional<std::uint64_t> safe_device_budget_bytes;
  std::optional<std::uint64_t> wddm_budget_bytes_start;
  std::optional<std::uint64_t> wddm_usage_bytes_start;
  std::optional<std::uint64_t> wddm_available_bytes_start;
  std::optional<std::uint64_t> wddm_available_bytes_minimum;
  std::optional<std::uint64_t> wddm_available_bytes_end;
  bool vmm_supported = false;
  std::optional<std::uint64_t> minimum_granularity_bytes;
  std::optional<std::uint64_t> recommended_granularity_bytes;
};

struct Configuration {
  std::optional<std::int32_t> requested_device_ordinal;
  std::optional<std::uint64_t> requested_logical_bytes;
  std::optional<std::uint64_t> requested_chunk_bytes;
  std::optional<std::uint32_t> requested_window_slots;
  std::optional<std::uint32_t> effective_window_slots;
  std::uint32_t passes = 2;
  std::uint64_t timeout_ms = 120'000;
  std::uint64_t stall_timeout_ms = 5'000;
  std::string seed_hex = "0000000000000000";
  std::uint64_t host_headroom_bytes = 0;
  std::uint64_t device_headroom_bytes = 0;
  std::string sizing_mode = "auto";
  std::string mode = "compare";
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

struct ModeResult {
  std::string status = "not_run";
  std::optional<std::uint64_t> elements_processed;
  std::optional<std::uint64_t> tiles_processed;
  std::optional<std::uint32_t> passes_completed;
  std::optional<std::uint64_t> bytes_h2d;
  std::optional<std::uint64_t> bytes_d2h;
  std::optional<std::uint64_t> mapping_count;
  std::optional<std::uint64_t> unmap_count;
  std::optional<std::uint64_t> remap_count;
  std::optional<std::uint64_t> event_boundary_count;
  std::optional<std::uint64_t> unsafe_remap_count;
  std::optional<std::uint64_t> physical_handle_count;
  std::optional<std::uint64_t> max_concurrent_mappings;
  std::optional<std::uint32_t> slot_count;
  std::optional<std::uint64_t> set_access_count;
  std::optional<std::uint64_t> handle_reuse_count;
  std::optional<bool> stable_addresses_verified;
  std::optional<bool> full_verification_completed;
  std::optional<bool> matches_cpu;
  std::optional<double> elapsed_ms;
  std::optional<double> setup_ms;
  std::optional<double> h2d_ms;
  std::optional<double> kernel_ms;
  std::optional<double> d2h_ms;
  std::optional<double> verification_ms;
  TimingSummary remap_timing;
  std::optional<std::string> expected_digest128;
  std::optional<std::string> output_digest128;
  std::optional<std::uint64_t> mismatch_count;
  std::optional<std::uint64_t> first_mismatch_byte_offset;
};

struct Modes {
  std::string backing = "pageable_host";
  std::string traversal = "alternating_sequential";
  std::string verification = "full_uint32_cpu_reference";
  std::string timeout_enforcement = "isolated_worker";
  ModeResult reference;
  ModeResult pipeline;
  std::optional<double> pipeline_speedup;
};

struct Proof {
  std::optional<std::uint64_t> effective_logical_bytes;
  std::optional<std::uint64_t> effective_chunk_bytes;
  std::optional<std::uint64_t> logical_element_count;
  std::optional<std::uint64_t> logical_chunk_count;
  std::optional<std::uint64_t> tile_visit_count;
  std::optional<std::uint64_t> address_revisit_count;
  std::optional<std::uint64_t> resident_physical_bytes;
  std::optional<std::uint64_t> pinned_staging_bytes;
  std::uint32_t kernel_module_version = 0;
  std::string kernel_module_sha256;
  std::string pattern_version;
  std::optional<std::string> cpu_reference_digest128;
  std::optional<bool> handles_reused;
  std::optional<bool> physical_window_smaller;
  std::optional<bool> stable_virtual_addresses_verified;
  std::optional<bool> event_boundaries_verified;
  std::optional<bool> reference_matches_cpu;
  std::optional<bool> pipeline_matches_cpu;
  std::optional<bool> modes_match;
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
  std::optional<std::string> mode;
  std::optional<std::uint32_t> pass_index;
  std::optional<std::uint64_t> tile_index;
  std::optional<std::uint64_t> logical_byte_offset;
};

struct Cleanup {
  std::optional<bool> complete;
  std::optional<bool> events_drained;
  std::optional<bool> events_destroyed;
  std::optional<bool> streams_destroyed;
  std::optional<bool> module_unloaded;
  std::optional<bool> device_allocations_released;
  std::optional<bool> mappings_removed;
  std::optional<bool> physical_handle_released;
  std::optional<bool> virtual_reservation_released;
  std::optional<bool> pinned_staging_released;
  std::optional<bool> host_backing_released;
  std::optional<bool> context_destroyed;
  std::optional<bool> worker_terminated;
};

struct Report {
  std::uint32_t schema_version = 1;
  std::string report_type = "xvram.vmm_poc";
  std::string generated_at_utc;
  probe::BuildInfo build;
  probe::SystemInfo system;
  std::optional<DeviceInfo> device;
  Configuration configuration;
  Modes modes;
  Proof proof;
  Outcome outcome;
  Cleanup cleanup;
  std::vector<probe::Diagnostic> diagnostics;
};

void write_json(const Report& report, std::ostream& output, bool pretty);
void write_text(const Report& report, std::ostream& output);

} // namespace xvram::vmm_poc
