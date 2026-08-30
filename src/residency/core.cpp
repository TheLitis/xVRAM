#include "residency/core.hpp"

#include <algorithm>
#include <limits>

namespace xvram::residency {
namespace {

struct IndexedRange {
  AccessRange range;
};

[[nodiscard]] std::optional<unsigned int> mode_bits(const AccessMode mode) noexcept {
  switch (mode) {
  case AccessMode::read:
    return 1U;
  case AccessMode::write_only:
    return 2U;
  case AccessMode::read_write:
    return 3U;
  }
  return std::nullopt;
}

[[nodiscard]] AccessMode mode_from_bits(const unsigned int bits) noexcept {
  if (bits == 1U) {
    return AccessMode::read;
  }
  if (bits == 2U) {
    return AccessMode::write_only;
  }
  return AccessMode::read_write;
}

[[nodiscard]] bool same_key(const ChunkKey& left, const ChunkKey& right) noexcept {
  return left == right;
}

[[nodiscard]] bool has_frame(const ChunkState state) noexcept {
  switch (state) {
  case ChunkState::mapping:
  case ChunkState::h2d_in_flight:
  case ChunkState::resident_clean:
  case ChunkState::resident_dirty:
  case ChunkState::writeback_queued:
  case ChunkState::d2h_in_flight:
  case ChunkState::evicting:
    return true;
  case ChunkState::host_clean:
  case ChunkState::prefetch_queued:
  case ChunkState::poisoned:
    return false;
  }
  return false;
}

[[nodiscard]] bool requires_staging(const ChunkState state) noexcept {
  return state == ChunkState::h2d_in_flight || state == ChunkState::d2h_in_flight;
}

[[nodiscard]] bool forbids_staging(const ChunkState state) noexcept {
  switch (state) {
  case ChunkState::host_clean:
  case ChunkState::prefetch_queued:
  case ChunkState::resident_clean:
  case ChunkState::resident_dirty:
  case ChunkState::writeback_queued:
  case ChunkState::evicting:
    return true;
  case ChunkState::mapping:
  case ChunkState::h2d_in_flight:
  case ChunkState::d2h_in_flight:
  case ChunkState::poisoned:
    return false;
  }
  return false;
}

} // namespace

std::size_t ChunkKeyHash::operator()(const ChunkKey& key) const noexcept {
  std::uint64_t value = key.allocation_id.value;
  value ^= key.chunk_index + 0x9E3779B97F4A7C15ULL + (value << 6U) + (value >> 2U);
  value ^= value >> 30U;
  value *= 0xBF58476D1CE4E5B9ULL;
  value ^= value >> 27U;
  value *= 0x94D049BB133111EBULL;
  value ^= value >> 31U;
  return static_cast<std::size_t>(value);
}

std::string_view access_mode_name(const AccessMode mode) noexcept {
  switch (mode) {
  case AccessMode::read:
    return "read";
  case AccessMode::read_write:
    return "read_write";
  case AccessMode::write_only:
    return "write_only";
  }
  return "invalid";
}

std::string_view access_plan_error_name(const AccessPlanError error) noexcept {
  switch (error) {
  case AccessPlanError::none:
    return "none";
  case AccessPlanError::invalid_chunk_size:
    return "invalid_chunk_size";
  case AccessPlanError::invalid_allocation:
    return "invalid_allocation";
  case AccessPlanError::duplicate_allocation:
    return "duplicate_allocation";
  case AccessPlanError::unknown_allocation:
    return "unknown_allocation";
  case AccessPlanError::zero_length:
    return "zero_length";
  case AccessPlanError::range_overflow:
    return "range_overflow";
  case AccessPlanError::range_out_of_bounds:
    return "range_out_of_bounds";
  case AccessPlanError::invalid_access_mode:
    return "invalid_access_mode";
  }
  return "invalid";
}

AccessPlanResult normalize_and_split_accesses(const std::span<const AllocationLayout> allocations,
                                              const std::span<const AccessRange> ranges,
                                              const std::uint64_t chunk_bytes) {
  AccessPlanResult result;
  const auto fail = [&](const AccessPlanError error,
                        const std::optional<std::size_t> input_index = std::nullopt) {
    result.error = error;
    result.input_range_index = input_index;
    result.normalized_ranges.clear();
    result.chunks.clear();
    return result;
  };

  if (chunk_bytes == 0) {
    return fail(AccessPlanError::invalid_chunk_size);
  }

  std::vector<AllocationLayout> layouts(allocations.begin(), allocations.end());
  std::sort(layouts.begin(), layouts.end(),
            [](const AllocationLayout& left, const AllocationLayout& right) {
              return left.id < right.id;
            });
  for (std::size_t index = 0; index < layouts.size(); ++index) {
    if (!layouts[index].id || layouts[index].size_bytes == 0) {
      return fail(AccessPlanError::invalid_allocation);
    }
    if (index != 0 && layouts[index - 1U].id == layouts[index].id) {
      return fail(AccessPlanError::duplicate_allocation);
    }
  }

  std::vector<IndexedRange> indexed_ranges;
  indexed_ranges.reserve(ranges.size());
  for (std::size_t index = 0; index < ranges.size(); ++index) {
    const AccessRange& range = ranges[index];
    if (!mode_bits(range.mode).has_value()) {
      return fail(AccessPlanError::invalid_access_mode, index);
    }
    if (range.length_bytes == 0) {
      return fail(AccessPlanError::zero_length, index);
    }
    const auto layout_it = std::lower_bound(
        layouts.begin(), layouts.end(), range.allocation_id,
        [](const AllocationLayout& layout, const AllocationId id) { return layout.id < id; });
    if (layout_it == layouts.end() || layout_it->id != range.allocation_id) {
      return fail(AccessPlanError::unknown_allocation, index);
    }
    if (range.length_bytes > std::numeric_limits<std::uint64_t>::max() - range.offset_bytes) {
      return fail(AccessPlanError::range_overflow, index);
    }
    const std::uint64_t end = range.offset_bytes + range.length_bytes;
    if (range.offset_bytes >= layout_it->size_bytes || end > layout_it->size_bytes) {
      return fail(AccessPlanError::range_out_of_bounds, index);
    }
    indexed_ranges.push_back(IndexedRange{range});
  }

  for (const AllocationLayout& layout : layouts) {
    std::vector<const IndexedRange*> allocation_ranges;
    std::vector<std::uint64_t> boundaries;
    for (const IndexedRange& indexed : indexed_ranges) {
      if (indexed.range.allocation_id == layout.id) {
        allocation_ranges.push_back(&indexed);
        boundaries.push_back(indexed.range.offset_bytes);
        boundaries.push_back(indexed.range.offset_bytes + indexed.range.length_bytes);
      }
    }
    if (allocation_ranges.empty()) {
      continue;
    }

    std::sort(boundaries.begin(), boundaries.end());
    boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());
    for (std::size_t boundary_index = 0; boundary_index + 1U < boundaries.size();
         ++boundary_index) {
      const std::uint64_t begin = boundaries[boundary_index];
      const std::uint64_t end = boundaries[boundary_index + 1U];
      unsigned int combined_bits = 0;
      for (const IndexedRange* indexed : allocation_ranges) {
        const std::uint64_t range_end = indexed->range.offset_bytes + indexed->range.length_bytes;
        if (indexed->range.offset_bytes <= begin && range_end >= end) {
          combined_bits |= *mode_bits(indexed->range.mode);
        }
      }
      if (combined_bits == 0U) {
        continue;
      }

      const AccessMode combined_mode = mode_from_bits(combined_bits);
      if (!result.normalized_ranges.empty()) {
        AccessRange& previous = result.normalized_ranges.back();
        if (previous.allocation_id == layout.id && previous.mode == combined_mode &&
            previous.offset_bytes + previous.length_bytes == begin) {
          previous.length_bytes += end - begin;
          continue;
        }
      }
      result.normalized_ranges.push_back(AccessRange{layout.id, begin, end - begin, combined_mode});
    }
  }

  for (const AccessRange& range : result.normalized_ranges) {
    const auto layout_it = std::lower_bound(
        layouts.begin(), layouts.end(), range.allocation_id,
        [](const AllocationLayout& layout, const AllocationId id) { return layout.id < id; });
    std::uint64_t cursor = range.offset_bytes;
    std::uint64_t remaining = range.length_bytes;
    while (remaining != 0) {
      const std::uint64_t chunk_index = cursor / chunk_bytes;
      const std::uint64_t chunk_origin = chunk_index * chunk_bytes;
      const std::uint64_t valid_bytes = std::min(chunk_bytes, layout_it->size_bytes - chunk_origin);
      const std::uint64_t chunk_offset = cursor - chunk_origin;
      const std::uint64_t span_bytes = std::min(remaining, valid_bytes - chunk_offset);
      const ChunkKey key{range.allocation_id, chunk_index};

      if (result.chunks.empty() || !same_key(result.chunks.back().key, key)) {
        result.chunks.push_back(ChunkAccessPlan{key, chunk_origin, valid_bytes});
      }
      ChunkAccessPlan& chunk = result.chunks.back();
      if (!chunk.spans.empty()) {
        ChunkAccessSpan& previous = chunk.spans.back();
        if (previous.mode == range.mode &&
            previous.offset_bytes + previous.length_bytes == chunk_offset) {
          previous.length_bytes += span_bytes;
        } else {
          chunk.spans.push_back(ChunkAccessSpan{chunk_offset, span_bytes, range.mode});
        }
      } else {
        chunk.spans.push_back(ChunkAccessSpan{chunk_offset, span_bytes, range.mode});
      }

      cursor += span_bytes;
      remaining -= span_bytes;
    }
  }

  for (ChunkAccessPlan& chunk : result.chunks) {
    for (const ChunkAccessSpan& span : chunk.spans) {
      chunk.reads_existing_data = chunk.reads_existing_data || span.mode != AccessMode::write_only;
      chunk.marks_dirty = chunk.marks_dirty || span.mode != AccessMode::read;
    }
    chunk.full_write_only = chunk.spans.size() == 1U && chunk.spans.front().offset_bytes == 0 &&
                            chunk.spans.front().length_bytes == chunk.valid_bytes &&
                            chunk.spans.front().mode == AccessMode::write_only;
    chunk.requires_h2d = !chunk.full_write_only;
  }

  return result;
}

std::string_view chunk_state_name(const ChunkState state) noexcept {
  switch (state) {
  case ChunkState::host_clean:
    return "host_clean";
  case ChunkState::prefetch_queued:
    return "prefetch_queued";
  case ChunkState::mapping:
    return "mapping";
  case ChunkState::h2d_in_flight:
    return "h2d_in_flight";
  case ChunkState::resident_clean:
    return "resident_clean";
  case ChunkState::resident_dirty:
    return "resident_dirty";
  case ChunkState::writeback_queued:
    return "writeback_queued";
  case ChunkState::d2h_in_flight:
    return "d2h_in_flight";
  case ChunkState::evicting:
    return "evicting";
  case ChunkState::poisoned:
    return "poisoned";
  }
  return "invalid";
}

bool is_legal_transition(const ChunkState from, const ChunkState to) noexcept {
  if (to == ChunkState::poisoned) {
    return true;
  }
  switch (from) {
  case ChunkState::host_clean:
    return to == ChunkState::prefetch_queued || to == ChunkState::mapping;
  case ChunkState::prefetch_queued:
    return to == ChunkState::host_clean || to == ChunkState::mapping;
  case ChunkState::mapping:
    return to == ChunkState::h2d_in_flight || to == ChunkState::resident_clean;
  case ChunkState::h2d_in_flight:
    return to == ChunkState::resident_clean;
  case ChunkState::resident_clean:
    return to == ChunkState::resident_dirty || to == ChunkState::evicting;
  case ChunkState::resident_dirty:
    return to == ChunkState::writeback_queued;
  case ChunkState::writeback_queued:
    return to == ChunkState::d2h_in_flight;
  case ChunkState::d2h_in_flight:
    return to == ChunkState::resident_clean;
  case ChunkState::evicting:
    return to == ChunkState::host_clean;
  case ChunkState::poisoned:
    return to == ChunkState::poisoned;
  }
  return false;
}

StateTransitionResult transition_chunk_state(ChunkState& state, const ChunkState target) noexcept {
  if (!is_legal_transition(state, target)) {
    return StateTransitionResult::illegal_transition;
  }
  state = target;
  return StateTransitionResult::success;
}

std::string_view chunk_invariant_error_name(const ChunkInvariantError error) noexcept {
  switch (error) {
  case ChunkInvariantError::none:
    return "none";
  case ChunkInvariantError::invalid_key:
    return "invalid_key";
  case ChunkInvariantError::completed_generation_ahead:
    return "completed_generation_ahead";
  case ChunkInvariantError::unexpected_frame:
    return "unexpected_frame";
  case ChunkInvariantError::missing_frame:
    return "missing_frame";
  case ChunkInvariantError::unexpected_staging:
    return "unexpected_staging";
  case ChunkInvariantError::missing_staging:
    return "missing_staging";
  case ChunkInvariantError::pinned_host_chunk:
    return "pinned_host_chunk";
  case ChunkInvariantError::pinned_evicting_chunk:
    return "pinned_evicting_chunk";
  case ChunkInvariantError::working_set_host_chunk:
    return "working_set_host_chunk";
  case ChunkInvariantError::working_set_evicting_chunk:
    return "working_set_evicting_chunk";
  case ChunkInvariantError::incomplete_evicting_generation:
    return "incomplete_evicting_generation";
  }
  return "invalid";
}

ChunkInvariantError validate_chunk_record(const ChunkRecord& record) noexcept {
  if (!record.key.allocation_id) {
    return ChunkInvariantError::invalid_key;
  }
  if (record.completed_generation > record.event_generation) {
    return ChunkInvariantError::completed_generation_ahead;
  }
  if (record.state == ChunkState::poisoned) {
    return ChunkInvariantError::none;
  }
  if (has_frame(record.state) && !record.frame_index.has_value()) {
    return ChunkInvariantError::missing_frame;
  }
  if (!has_frame(record.state) && record.frame_index.has_value()) {
    return ChunkInvariantError::unexpected_frame;
  }
  if (requires_staging(record.state) && !record.staging_slot.has_value()) {
    return ChunkInvariantError::missing_staging;
  }
  if (forbids_staging(record.state) && record.staging_slot.has_value()) {
    return ChunkInvariantError::unexpected_staging;
  }
  if ((record.state == ChunkState::host_clean || record.state == ChunkState::prefetch_queued) &&
      record.pin_count != 0) {
    return ChunkInvariantError::pinned_host_chunk;
  }
  if (record.state == ChunkState::evicting && record.pin_count != 0) {
    return ChunkInvariantError::pinned_evicting_chunk;
  }
  if ((record.state == ChunkState::host_clean || record.state == ChunkState::prefetch_queued) &&
      record.in_current_working_set) {
    return ChunkInvariantError::working_set_host_chunk;
  }
  if (record.state == ChunkState::evicting && record.in_current_working_set) {
    return ChunkInvariantError::working_set_evicting_chunk;
  }
  if (record.state == ChunkState::evicting &&
      record.completed_generation != record.event_generation) {
    return ChunkInvariantError::incomplete_evicting_generation;
  }
  return ChunkInvariantError::none;
}

bool chunk_has_in_flight_work(const ChunkRecord& record) noexcept {
  return record.state == ChunkState::mapping || record.state == ChunkState::h2d_in_flight ||
         record.state == ChunkState::d2h_in_flight ||
         record.completed_generation != record.event_generation;
}

bool is_victim_eligible(const ChunkRecord& record) noexcept {
  const bool resident =
      record.state == ChunkState::resident_clean || record.state == ChunkState::resident_dirty;
  return resident && record.pin_count == 0 && !record.in_current_working_set &&
         !record.staging_slot.has_value() &&
         record.completed_generation == record.event_generation &&
         validate_chunk_record(record) == ChunkInvariantError::none;
}

bool is_safe_to_unmap(const ChunkRecord& record) noexcept {
  return record.state == ChunkState::evicting && record.pin_count == 0 &&
         !record.in_current_working_set && record.frame_index.has_value() &&
         !record.staging_slot.has_value() &&
         record.completed_generation == record.event_generation &&
         validate_chunk_record(record) == ChunkInvariantError::none;
}

EventGenerationResult begin_event_generation(ChunkRecord& record) noexcept {
  if (record.event_generation == std::numeric_limits<std::uint64_t>::max()) {
    return EventGenerationResult::overflow;
  }
  ++record.event_generation;
  return EventGenerationResult::success;
}

EventGenerationResult complete_event_generation(ChunkRecord& record,
                                                const std::uint64_t generation) noexcept {
  if (generation == 0) {
    return EventGenerationResult::invalid_generation;
  }
  if (generation < record.event_generation) {
    return generation <= record.completed_generation ? EventGenerationResult::already_completed
                                                     : EventGenerationResult::stale_generation;
  }
  if (generation > record.event_generation) {
    return EventGenerationResult::future_generation;
  }
  if (generation <= record.completed_generation) {
    return EventGenerationResult::already_completed;
  }
  record.completed_generation = generation;
  return EventGenerationResult::success;
}

} // namespace xvram::residency
