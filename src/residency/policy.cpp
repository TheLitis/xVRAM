#include "residency/policy.hpp"

#include <algorithm>
#include <limits>

namespace xvram::residency {
namespace {

[[nodiscard]] const VictimCandidate*
find_candidate(const std::span<const VictimCandidate> candidates, const ChunkKey key) noexcept {
  const auto found =
      std::find_if(candidates.begin(), candidates.end(),
                   [&](const VictimCandidate& candidate) { return candidate.key == key; });
  return found == candidates.end() ? nullptr : &*found;
}

[[nodiscard]] unsigned int clock_cost(const bool speculative, const bool sequential_one_touch,
                                      const ChunkState state, const bool hot) noexcept {
  const bool dirty = state == ChunkState::resident_dirty;
  unsigned int cost = (dirty ? 3U : 0U) + (hot ? 4U : 0U);
  if (sequential_one_touch) {
    return cost;
  }
  if (speculative) {
    return cost + 1U;
  }
  return cost + 2U;
}

} // namespace

bool is_policy_candidate_eligible(const VictimCandidate& candidate) noexcept {
  const bool resident = candidate.state == ChunkState::resident_clean ||
                        candidate.state == ChunkState::resident_dirty;
  return resident && candidate.pin_count == 0 && !candidate.in_flight &&
         !candidate.in_current_working_set;
}

std::string_view ClockPolicy::name() const noexcept {
  return "clock";
}

void ClockPolicy::insert(const ChunkKey key, const std::uint64_t sequence, const bool speculative,
                         const bool sequential_one_touch) {
  (void)sequence;
  const auto found = std::find_if(entries_.begin(), entries_.end(),
                                  [&](const Entry& entry) { return entry.key == key; });
  if (found != entries_.end()) {
    found->referenced = found->referenced || !speculative;
    found->speculative = speculative;
    found->sequential_one_touch = sequential_one_touch;
    return;
  }
  entries_.push_back(Entry{key, !speculative, speculative, sequential_one_touch});
}

void ClockPolicy::touch(const ChunkKey key, const std::uint64_t sequence,
                        const bool sequential_one_touch) {
  (void)sequence;
  const auto found = std::find_if(entries_.begin(), entries_.end(),
                                  [&](const Entry& entry) { return entry.key == key; });
  if (found == entries_.end()) {
    insert(key, sequence, false, sequential_one_touch);
    return;
  }
  found->referenced = true;
  found->speculative = false;
  found->sequential_one_touch = sequential_one_touch;
}

bool ClockPolicy::erase(const ChunkKey key) noexcept {
  const auto found = std::find_if(entries_.begin(), entries_.end(),
                                  [&](const Entry& entry) { return entry.key == key; });
  if (found == entries_.end()) {
    return false;
  }
  const std::size_t erased_index = static_cast<std::size_t>(found - entries_.begin());
  entries_.erase(found);
  if (!entries_.empty() && erased_index < hand_) {
    --hand_;
  }
  if (entries_.empty() || hand_ >= entries_.size()) {
    hand_ = 0;
  }
  return true;
}

std::optional<ChunkKey>
ClockPolicy::select_victim(const std::span<const VictimCandidate> candidates) {
  if (entries_.empty()) {
    return std::nullopt;
  }

  unsigned int cheapest_cost = std::numeric_limits<unsigned int>::max();
  for (const Entry& entry : entries_) {
    const VictimCandidate* candidate = find_candidate(candidates, entry.key);
    if (candidate != nullptr && is_policy_candidate_eligible(*candidate)) {
      cheapest_cost =
          std::min(cheapest_cost, clock_cost(entry.speculative, entry.sequential_one_touch,
                                             candidate->state, candidate->hot));
    }
  }
  if (cheapest_cost == std::numeric_limits<unsigned int>::max()) {
    return std::nullopt;
  }

  // One pass clears reference bits for the cheapest class; the second selects one if all entries
  // in that class had a second chance.
  for (unsigned int pass = 0; pass < 2U; ++pass) {
    for (std::size_t scanned = 0; scanned < entries_.size(); ++scanned) {
      const std::size_t index = (hand_ + scanned) % entries_.size();
      Entry& entry = entries_[index];
      const VictimCandidate* candidate = find_candidate(candidates, entry.key);
      if (candidate == nullptr || !is_policy_candidate_eligible(*candidate) ||
          clock_cost(entry.speculative, entry.sequential_one_touch, candidate->state,
                     candidate->hot) != cheapest_cost) {
        continue;
      }
      if (entry.referenced) {
        entry.referenced = false;
        continue;
      }
      hand_ = (index + 1U) % entries_.size();
      return entry.key;
    }
  }
  return std::nullopt;
}

std::size_t ClockPolicy::size() const noexcept {
  return entries_.size();
}

std::string_view LruPolicy::name() const noexcept {
  return "lru";
}

void LruPolicy::insert(const ChunkKey key, const std::uint64_t sequence, const bool speculative,
                       const bool sequential_one_touch) {
  (void)speculative;
  (void)sequential_one_touch;
  const auto found = std::find_if(entries_.begin(), entries_.end(),
                                  [&](const Entry& entry) { return entry.key == key; });
  if (found != entries_.end()) {
    found->last_touch_sequence = sequence;
    return;
  }
  entries_.push_back(Entry{key, sequence});
}

void LruPolicy::touch(const ChunkKey key, const std::uint64_t sequence,
                      const bool sequential_one_touch) {
  (void)sequential_one_touch;
  insert(key, sequence, false, false);
}

bool LruPolicy::erase(const ChunkKey key) noexcept {
  const auto found = std::find_if(entries_.begin(), entries_.end(),
                                  [&](const Entry& entry) { return entry.key == key; });
  if (found == entries_.end()) {
    return false;
  }
  entries_.erase(found);
  return true;
}

std::optional<ChunkKey>
LruPolicy::select_victim(const std::span<const VictimCandidate> candidates) {
  const Entry* selected = nullptr;
  for (const Entry& entry : entries_) {
    const VictimCandidate* candidate = find_candidate(candidates, entry.key);
    if (candidate == nullptr || !is_policy_candidate_eligible(*candidate)) {
      continue;
    }
    const VictimCandidate* selected_candidate =
        selected == nullptr ? nullptr : find_candidate(candidates, selected->key);
    if (selected == nullptr ||
        (selected_candidate != nullptr && selected_candidate->hot && !candidate->hot) ||
        (selected_candidate != nullptr && selected_candidate->hot == candidate->hot &&
         (entry.last_touch_sequence < selected->last_touch_sequence ||
          (entry.last_touch_sequence == selected->last_touch_sequence &&
           entry.key < selected->key)))) {
      selected = &entry;
    }
  }
  return selected == nullptr ? std::nullopt : std::optional<ChunkKey>{selected->key};
}

std::size_t LruPolicy::size() const noexcept {
  return entries_.size();
}

} // namespace xvram::residency
