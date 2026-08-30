#pragma once

#include <compare>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace xvram::residency {

struct AllocationId {
  std::uint64_t value = 0;

  [[nodiscard]] explicit operator bool() const noexcept {
    return value != 0;
  }

  [[nodiscard]] auto operator<=>(const AllocationId&) const noexcept = default;
};

struct ChunkKey {
  AllocationId allocation_id;
  std::uint64_t chunk_index = 0;

  [[nodiscard]] auto operator<=>(const ChunkKey&) const noexcept = default;
};

struct ChunkKeyHash {
  [[nodiscard]] std::size_t operator()(const ChunkKey& key) const noexcept;
};

enum class AccessMode {
  read,
  read_write,
  write_only,
};

[[nodiscard]] std::string_view access_mode_name(AccessMode mode) noexcept;

struct AllocationLayout {
  AllocationId id;
  std::uint64_t size_bytes = 0;
};

struct AccessRange {
  AllocationId allocation_id;
  std::uint64_t offset_bytes = 0;
  std::uint64_t length_bytes = 0;
  AccessMode mode = AccessMode::read;
};

struct ChunkAccessSpan {
  std::uint64_t offset_bytes = 0;
  std::uint64_t length_bytes = 0;
  AccessMode mode = AccessMode::read;
};

struct ChunkAccessPlan {
  ChunkKey key;
  std::uint64_t allocation_offset_bytes = 0;
  std::uint64_t valid_bytes = 0;
  std::vector<ChunkAccessSpan> spans;
  bool reads_existing_data = false;
  bool marks_dirty = false;
  bool full_write_only = false;
  bool requires_h2d = true;
};

enum class AccessPlanError {
  none,
  invalid_chunk_size,
  invalid_allocation,
  duplicate_allocation,
  unknown_allocation,
  zero_length,
  range_overflow,
  range_out_of_bounds,
  invalid_access_mode,
};

[[nodiscard]] std::string_view access_plan_error_name(AccessPlanError error) noexcept;

struct AccessPlanResult {
  AccessPlanError error = AccessPlanError::none;
  std::optional<std::size_t> input_range_index;
  std::vector<AccessRange> normalized_ranges;
  std::vector<ChunkAccessPlan> chunks;

  [[nodiscard]] explicit operator bool() const noexcept {
    return error == AccessPlanError::none;
  }
};

// Treats the supplied ranges as one transaction. Overlaps are split into disjoint ranges and
// their read/write requirements are combined. The chunk plans are ordered by ChunkKey.
[[nodiscard]] AccessPlanResult
normalize_and_split_accesses(std::span<const AllocationLayout> allocations,
                             std::span<const AccessRange> ranges, std::uint64_t chunk_bytes);

enum class ChunkState {
  host_clean,
  prefetch_queued,
  mapping,
  h2d_in_flight,
  resident_clean,
  resident_dirty,
  writeback_queued,
  d2h_in_flight,
  evicting,
  poisoned,
};

[[nodiscard]] std::string_view chunk_state_name(ChunkState state) noexcept;
[[nodiscard]] bool is_legal_transition(ChunkState from, ChunkState to) noexcept;

enum class StateTransitionResult {
  success,
  illegal_transition,
};

[[nodiscard]] StateTransitionResult transition_chunk_state(ChunkState& state,
                                                           ChunkState target) noexcept;

struct ChunkRecord {
  ChunkKey key;
  ChunkState state = ChunkState::host_clean;
  std::uint32_t pin_count = 0;
  std::optional<std::uint64_t> frame_index;
  std::optional<std::uint32_t> staging_slot;
  std::uint64_t event_generation = 0;
  std::uint64_t completed_generation = 0;
  bool in_current_working_set = false;
  bool speculative = false;
  bool sequential_one_touch = false;
};

enum class ChunkInvariantError {
  none,
  invalid_key,
  completed_generation_ahead,
  unexpected_frame,
  missing_frame,
  unexpected_staging,
  missing_staging,
  pinned_host_chunk,
  pinned_evicting_chunk,
  working_set_host_chunk,
  working_set_evicting_chunk,
  incomplete_evicting_generation,
};

[[nodiscard]] std::string_view chunk_invariant_error_name(ChunkInvariantError error) noexcept;
[[nodiscard]] ChunkInvariantError validate_chunk_record(const ChunkRecord& record) noexcept;
[[nodiscard]] bool chunk_has_in_flight_work(const ChunkRecord& record) noexcept;
[[nodiscard]] bool is_victim_eligible(const ChunkRecord& record) noexcept;
[[nodiscard]] bool is_safe_to_unmap(const ChunkRecord& record) noexcept;

enum class EventGenerationResult {
  success,
  overflow,
  invalid_generation,
  stale_generation,
  future_generation,
  already_completed,
};

[[nodiscard]] EventGenerationResult begin_event_generation(ChunkRecord& record) noexcept;
[[nodiscard]] EventGenerationResult complete_event_generation(ChunkRecord& record,
                                                              std::uint64_t generation) noexcept;

} // namespace xvram::residency
