#pragma once
#include "trace_core.hpp"
#include <cstddef>
#include <cstring>
#include <deque>
#include <vector>

namespace xvram::audit {
// Caller serializes access. Active bytes include a popped item until retirement.
// Admission never waits for space, and accounting commits only after owned copy.
struct ModuleCopy {
  std::uint64_t context = 0, module = 0, generation = 0, source_sequence = 0;
  std::vector<std::byte> bytes;
};
class ModuleCopies {
public:
  using Copier = void (*)(std::vector<std::byte> &, const void *, std::size_t);
  static void owned_copy(std::vector<std::byte> & output, const void * data, std::size_t bytes) {
    output.resize(bytes); std::memcpy(output.data(),data,bytes);
  }
  explicit ModuleCopies(std::size_t single = 64U*1024U*1024U,
                        std::size_t active = 128U*1024U*1024U,
                        std::size_t total = 512U*1024U*1024U, Copier copier = owned_copy)
      : single_(single), active_limit_(active), total_limit_(total), copier_(copier) {}
  bool push(const void * data, std::size_t bytes, ModuleCopy item) {
    if (!data || bytes == 0 || bytes > single_ || bytes > active_limit_-active_bytes ||
        bytes > total_limit_-total_bytes || queued_.size() >= max_identities) return false;
    copier_(item.bytes,data,bytes);
    if (item.bytes.size()!=bytes) throw std::logic_error("invalid module copy");
    queued_.push_back(std::move(item));
    active_bytes += bytes; total_bytes += bytes; ++submitted;
    if (active_bytes > peak_bytes) peak_bytes = active_bytes;
    return true;
  }
  bool pop(ModuleCopy & item) {
    if (queued_.empty()) return false;
    item = std::move(queued_.front()); queued_.pop_front(); return true;
  }
  void retire(std::size_t bytes) {
    if (bytes > active_bytes || retired >= submitted) throw std::logic_error("module copy retirement");
    active_bytes -= bytes; ++retired;
  }
  std::size_t active_bytes=0, total_bytes=0, peak_bytes=0;
  std::uint64_t submitted=0, retired=0;
private:
  std::size_t single_, active_limit_, total_limit_;
  Copier copier_;
  std::deque<ModuleCopy> queued_;
};

// Generation changes on each load, never inferred from buffered activity order.
class ModuleLifetimes {
public:
  struct Life { std::uint64_t generation=0; bool live=false; };
  Life load(std::uint64_t context, std::uint64_t module) {
    const auto key=std::pair{context,module};
    if (!items_.contains(key) && items_.size() >= max_identities) throw std::length_error("module lifetime limit");
    auto & life=items_[key];
    if (life.live) throw std::logic_error("duplicate module load");
    if (life.generation == std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("module generation");
    ++life.generation; life.live=true; return life;
  }
  Life lookup(std::uint64_t context, std::uint64_t module) const {
    const auto found=items_.find({context,module});
    return found==items_.end() ? Life{} : found->second;
  }
  Life unload(std::uint64_t context, std::uint64_t module) {
    auto found=items_.find({context,module});
    if (found==items_.end() || !found->second.live) return {};
    const auto prior=found->second; found->second.live=false; return prior;
  }
private:
  std::map<std::pair<std::uint64_t,std::uint64_t>,Life> items_;
};
} // namespace xvram::audit
