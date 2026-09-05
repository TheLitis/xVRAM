#pragma once

#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace xvram::audit {
inline constexpr std::size_t max_record_bytes = 1024U * 1024U;
inline constexpr std::size_t max_identities = 65536;

// Output is ASCII JSON. No raw address or handle is accepted by this interface.
class JsonLine {
public:
  JsonLine() : text_("{") {}
  void number(std::string_view key, std::uint64_t value) { field(key, std::to_string(value)); }
  void integer(std::string_view key, std::int64_t value) { field(key, std::to_string(value)); }
  void boolean(std::string_view key, bool value) { field(key, value ? "true" : "false"); }
  void string(std::string_view key, std::string_view value) { field(key, quoted(value)); }
  [[nodiscard]] std::string finish() const {
    if (text_.size() + 2 > max_record_bytes) throw std::length_error("record limit");
    return text_ + "}\n";
  }
private:
  static std::string quoted(std::string_view value) {
    if (value.size() > 4096) throw std::length_error("string limit");
    constexpr char hex[] = "0123456789abcdef";
    std::string result = "\"";
    for (const unsigned char ch : value) {
      if (ch == '"' || ch == '\\') { result += '\\'; result += static_cast<char>(ch); }
      else if (ch < 0x20 || ch >= 0x7f) {
        result += "\\u00"; result += hex[ch >> 4]; result += hex[ch & 15];
      } else result += static_cast<char>(ch);
    }
    return result + '"';
  }
  void field(std::string_view key, std::string value) {
    auto addition = (first_ ? "" : ",") + quoted(key) + ':' + value;
    if (text_.size() + addition.size() + 2 > max_record_bytes) throw std::length_error("record limit");
    text_ += addition; first_ = false;
  }
  std::string text_;
  bool first_ = true;
};

struct Range {
  std::uint64_t allocation_id = 0;
  std::uint64_t generation = 0;
  std::uint64_t offset_bytes = 0;
  bool known = false;
};

// Addresses exist only in this transient process-local registry, never in JSON.
class Registry {
public:
  explicit Registry(std::size_t cap = max_identities) : cap_(cap) {}
  std::uint64_t identity(std::string_view category, std::uint64_t native) {
    const auto key = std::pair{std::string(category), native};
    if (auto found = identities_.find(key); found != identities_.end()) return found->second;
    require_capacity();
    return identities_.emplace(key, next_id_++).first->second;
  }
  void retire_identity(std::string_view category, std::uint64_t native) {
    // Keep the slot bounded, but a reused native identity receives a fresh ID.
    const auto key = std::pair{std::string(category), native};
    if (auto found = identities_.find(key); found != identities_.end()) found->second = next_id_++;
  }
  Range allocate(std::uint64_t address, std::uint64_t bytes) {
    if (address == 0 || bytes == 0 || bytes > std::numeric_limits<std::uint64_t>::max() - address)
      return {};
    auto found = allocations_.find(address);
    if (found != allocations_.end() && found->second.live) {
      // Runtime/driver callbacks may describe the same allocation twice.
      if (found->second.bytes != bytes) throw std::runtime_error("conflicting allocation descriptions");
      return {found->second.id, found->second.generation, 0, true};
    }
    for (const auto & [base, other] : allocations_) {
      if (other.live && address < base + other.bytes && base < address + bytes)
        throw std::runtime_error("overlapping allocation descriptions");
    }
    if (found == allocations_.end()) {
      require_capacity();
      found = allocations_.emplace(address, Allocation{}).first;
    }
    auto & allocation = found->second;
    allocation.id = next_id_++;
    ++allocation.generation;
    allocation.bytes = bytes;
    allocation.live = true;
    return {allocation.id, allocation.generation, 0, true};
  }
  Range lookup(std::uint64_t address, std::uint64_t bytes) const {
    auto found = allocations_.upper_bound(address);
    while (found != allocations_.begin()) {
      --found;
      const auto & allocation = found->second;
      if (!allocation.live) continue;
      const auto offset = address - found->first;
      if (offset > allocation.bytes || bytes > allocation.bytes - offset) return {};
      return {allocation.id, allocation.generation, offset, true};
    }
    return {};
  }
  Range release(std::uint64_t address) {
    auto found = allocations_.find(address);
    if (found == allocations_.end()) return {};
    auto & allocation = found->second;
    allocation.live = false;
    return {allocation.id, allocation.generation, 0, true};
  }
private:
  struct Allocation { std::uint64_t id = 0, generation = 0, bytes = 0; bool live = false; };
  void require_capacity() const {
    if (identities_.size() + allocations_.size() >= cap_) throw std::length_error("identity limit");
  }
  std::size_t cap_;
  std::uint64_t next_id_ = 1;
  std::map<std::pair<std::string, std::uint64_t>, std::uint64_t> identities_;
  std::map<std::uint64_t, Allocation> allocations_;
};
inline void add_range(JsonLine & line, const Range & range, std::string_view prefix = {}) {
  const std::string p(prefix);
  line.number(p + "allocation_id", range.allocation_id);
  line.number(p + "generation", range.generation);
  line.number(p + "offset_bytes", range.offset_bytes);
  line.boolean(p + "range_known", range.known);
}
} // namespace xvram::audit
