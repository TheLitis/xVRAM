#pragma once

#include "residency/core.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace xvram::residency {

struct VictimCandidate {
  ChunkKey key;
  ChunkState state = ChunkState::host_clean;
  std::uint32_t pin_count = 0;
  bool in_flight = false;
  bool in_current_working_set = false;
  bool hot = false;
};

[[nodiscard]] bool is_policy_candidate_eligible(const VictimCandidate& candidate) noexcept;

class VictimPolicy {
public:
  virtual ~VictimPolicy() = default;

  [[nodiscard]] virtual std::string_view name() const noexcept = 0;
  virtual void insert(ChunkKey key, std::uint64_t sequence, bool speculative,
                      bool sequential_one_touch) = 0;
  virtual void touch(ChunkKey key, std::uint64_t sequence, bool sequential_one_touch) = 0;
  virtual bool erase(ChunkKey key) noexcept = 0;
  [[nodiscard]] virtual std::optional<ChunkKey>
  select_victim(std::span<const VictimCandidate> candidates) = 0;
  [[nodiscard]] virtual std::size_t size() const noexcept = 0;
};

// A deterministic, cost-aware CLOCK. Safety eligibility remains a manager responsibility and is
// checked again here. Within CLOCK's second-chance ordering, clean, speculative and sequential
// one-touch entries are cheaper than reusable or dirty entries.
class ClockPolicy final : public VictimPolicy {
public:
  [[nodiscard]] std::string_view name() const noexcept override;
  void insert(ChunkKey key, std::uint64_t sequence, bool speculative,
              bool sequential_one_touch) override;
  void touch(ChunkKey key, std::uint64_t sequence, bool sequential_one_touch) override;
  bool erase(ChunkKey key) noexcept override;
  [[nodiscard]] std::optional<ChunkKey>
  select_victim(std::span<const VictimCandidate> candidates) override;
  [[nodiscard]] std::size_t size() const noexcept override;

private:
  struct Entry {
    ChunkKey key;
    bool referenced = false;
    bool speculative = false;
    bool sequential_one_touch = false;
  };

  std::vector<Entry> entries_;
  std::size_t hand_ = 0;
};

// A deterministic true-LRU baseline. Equal sequence numbers are resolved by ChunkKey.
class LruPolicy final : public VictimPolicy {
public:
  [[nodiscard]] std::string_view name() const noexcept override;
  void insert(ChunkKey key, std::uint64_t sequence, bool speculative,
              bool sequential_one_touch) override;
  void touch(ChunkKey key, std::uint64_t sequence, bool sequential_one_touch) override;
  bool erase(ChunkKey key) noexcept override;
  [[nodiscard]] std::optional<ChunkKey>
  select_victim(std::span<const VictimCandidate> candidates) override;
  [[nodiscard]] std::size_t size() const noexcept override;

private:
  struct Entry {
    ChunkKey key;
    std::uint64_t last_touch_sequence = 0;
  };

  std::vector<Entry> entries_;
};

} // namespace xvram::residency
