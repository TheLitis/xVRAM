#pragma once
// Diagnostic allocation/mapping ledger. Its ranges are NOT tensor boundaries,
// GPU retirement evidence, or permission to issue a CUDA operation.
#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

namespace xvram::launch_probe::memory_witness {
using Word = std::uint64_t;
inline Word sum(Word a, Word b) {
  if (b > std::numeric_limits<Word>::max() - a) throw std::runtime_error("memory_overflow");
  return a + b;
}
inline Word end(Word address, Word bytes) {
  if (!address || !bytes) throw std::runtime_error("memory_empty_range");
  return sum(address, bytes);
}
struct MappingSpan {
  Word mapping_id=0, physical_id=0, physical_generation=0;
  Word offset_bytes=0, bytes=0, physical_offset_bytes=0;
};
struct AccessSpan { Word offset_bytes=0, bytes=0; int device=0; unsigned flags=0; };
struct Resolution {
  Word allocation_id=0, generation=0, offset_bytes=0, bytes=0, allocation_bytes=0;
  Word revision=0, observed_sequence=0, alignment=0, base_mod_alignment=0;
  bool known=false, mapped=false, readable=false, writable=false;
  std::vector<MappingSpan> mappings;
  std::vector<AccessSpan> access;
};
struct Identity { Word id=0, generation=0, bytes=0; };
struct LedgerCounts {
  Word allocations=0, reservations=0, handles=0, mappings=0;
  Word allocation_bytes=0, reservation_bytes=0, physical_bytes=0, mapped_bytes=0;
};

class MemoryRegistry {
  struct Allocation {
    Word id=0, generation=0, bytes=0, context=0;
    bool live=false, reservation=false;
  };
  struct Handle { Word id=0, generation=0; bool live=false; };
  struct Physical { Word generation=0, bytes=0, maps=0; bool retained=true; };
  struct Mapping {
    Word id=0, allocation=0, physical=0, physical_generation=0, bytes=0, physical_offset=0;
    std::vector<AccessSpan> access;
  };
  std::map<Word, Allocation> allocations_;
  std::map<Word, Handle> handles_;
  std::map<Word, Physical> physical_;
  std::map<Word, Mapping> mappings_;
  std::size_t object_cap_, access_cap_, access_count_=0;
  Word byte_cap_, next_=1, revision_=0;
  LedgerCounts counts_;

  Word id() {
    if (next_ == std::numeric_limits<Word>::max()) throw std::runtime_error("memory_id_exhausted");
    return next_++;
  }
  void capacity(std::size_t extra) const {
    const auto size=allocations_.size()+handles_.size()+physical_.size()+mappings_.size();
    if (size > object_cap_ || extra > object_cap_-size) throw std::runtime_error("memory_object_capacity");
  }
  Word budget(Word current, Word bytes) const {
    const auto value=sum(current,bytes);
    if (value > byte_cap_) throw std::runtime_error("memory_byte_capacity");
    return value;
  }
  auto allocation(Word address, Word bytes) const {
    const auto limit=end(address,bytes);
    auto it=allocations_.upper_bound(address);
    while (it!=allocations_.begin()) {
      --it;
      if (!it->second.live) continue;
      if (address >= it->first && limit <= sum(it->first,it->second.bytes)) return it;
      break;
    }
    return allocations_.end();
  }
  std::vector<Word> covering_maps(Word address, Word bytes) const {
    const auto limit=end(address,bytes);
    auto it=mappings_.upper_bound(address);
    if (it!=mappings_.begin()) --it;
    Word cursor=address;
    std::vector<Word> found;
    for (;it!=mappings_.end() && cursor<limit;++it) {
      const auto finish=sum(it->first,it->second.bytes);
      if (finish<=cursor) continue;
      if (it->first>cursor) break;
      found.push_back(it->first);
      if (found.size()>4096) throw std::runtime_error("memory_resolution_capacity");
      cursor=std::min(limit,finish);
    }
    if (cursor!=limit) throw std::runtime_error("memory_unmapped_range");
    return found;
  }
  void retire_physical(Word key) {
    auto found=physical_.find(key);
    if (found==physical_.end()) throw std::runtime_error("memory_unknown_physical");
    if (!found->second.retained && !found->second.maps) {
      counts_.physical_bytes-=found->second.bytes;
      physical_.erase(found);
    }
  }
public:
  explicit MemoryRegistry(std::size_t objects=65536, std::size_t access_ranges=262144,
                          Word bytes=std::numeric_limits<Word>::max())
      : object_cap_(objects), access_cap_(access_ranges), byte_cap_(bytes) {}
  [[nodiscard]] Word revision() const noexcept { return revision_; }
  [[nodiscard]] LedgerCounts counts() const noexcept { return counts_; }
  Identity allocate(Word address, Word bytes, bool reservation, Word context=0) {
    const auto limit=end(address,bytes);
    for (const auto& [base, old]:allocations_)
      if (old.live && address<sum(base,old.bytes) && base<limit)
        throw std::runtime_error("memory_allocation_overlap");
    auto old=allocations_.find(address);
    capacity(old==allocations_.end() ? 1 : 0);
    const auto total=budget(reservation ? counts_.reservation_bytes : counts_.allocation_bytes,bytes);
    Allocation value{id(),old==allocations_.end() ? 1 : sum(old->second.generation,1),bytes,context,true,reservation};
    allocations_.insert_or_assign(address,value);
    if (reservation) { ++counts_.reservations; counts_.reservation_bytes=total; }
    else { ++counts_.allocations; counts_.allocation_bytes=total; }
    ++revision_;
    return {value.id,value.generation,value.bytes};
  }
  Identity release(Word address, Word bytes, bool reservation) {
    auto found=allocations_.find(address);
    if (found==allocations_.end() || !found->second.live || found->second.reservation!=reservation
        || (reservation && bytes!=found->second.bytes)) throw std::runtime_error("memory_unknown_or_partial_free");
    auto& value=found->second;
    for (const auto& [base,map]:mappings_) {
      (void)base;
      if (map.allocation==value.id) throw std::runtime_error("memory_mapped_reservation_free");
    }
    value.live=false;
    if (reservation) { --counts_.reservations; counts_.reservation_bytes-=value.bytes; }
    else { --counts_.allocations; counts_.allocation_bytes-=value.bytes; }
    ++revision_;
    return {value.id,value.generation,value.bytes};
  }
  Identity create(Word handle, Word bytes) {
    if (!handle || !bytes) throw std::runtime_error("memory_empty_handle");
    auto found=handles_.find(handle);
    if (found!=handles_.end() && found->second.live) throw std::runtime_error("memory_live_handle_reuse");
    capacity(found==handles_.end() ? 2 : 1);
    const auto total=budget(counts_.physical_bytes,bytes);
    Handle value{id(),found==handles_.end() ? 1 : sum(found->second.generation,1),true};
    physical_.emplace(value.id,Physical{value.generation,bytes,0,true});
    handles_.insert_or_assign(handle,value);
    ++counts_.handles; counts_.physical_bytes=total; ++revision_;
    return {value.id,value.generation,bytes};
  }
  Identity release_handle(Word handle) {
    auto found=handles_.find(handle);
    if (found==handles_.end() || !found->second.live) throw std::runtime_error("memory_unknown_handle_release");
    auto& physical=physical_.at(found->second.id);
    Identity result{found->second.id,found->second.generation,physical.bytes};
    physical.retained=false; found->second.live=false; --counts_.handles;
    retire_physical(result.id); ++revision_;
    return result;
  }
  MappingSpan map(Word address, Word bytes, Word handle, Word offset=0) {
    const auto owner=allocation(address,bytes);
    if (owner==allocations_.end() || !owner->second.reservation) throw std::runtime_error("memory_unknown_reservation");
    const auto source=handles_.find(handle);
    if (source==handles_.end() || !source->second.live) throw std::runtime_error("memory_unknown_map_handle");
    auto& physical=physical_.at(source->second.id);
    // CUDA 13.3 requires offset zero; do not invent support for another profile.
    if (offset || bytes>physical.bytes) throw std::runtime_error("memory_map_physical_bounds");
    const auto limit=sum(address,bytes);
    for (const auto& [base,old]:mappings_)
      if (address<sum(base,old.bytes) && base<limit) throw std::runtime_error("memory_mapping_overlap");
    capacity(1);
    const auto total=budget(counts_.mapped_bytes,bytes);
    Mapping value{id(),owner->second.id,source->second.id,source->second.generation,bytes,offset,{}};
    mappings_.emplace(address,value); ++physical.maps; ++counts_.mappings;
    counts_.mapped_bytes=total; ++revision_;
    return {value.id,value.physical,value.physical_generation,address-owner->first,bytes,offset};
  }
  MappingSpan unmap(Word address, Word bytes) {
    auto found=mappings_.find(address);
    if (found==mappings_.end() || found->second.bytes!=bytes) throw std::runtime_error("memory_partial_or_unknown_unmap");
    const auto owner=allocation(address,bytes);
    const auto value=found->second;
    access_count_-=value.access.size();
    mappings_.erase(found); --physical_.at(value.physical).maps; --counts_.mappings;
    counts_.mapped_bytes-=bytes; retire_physical(value.physical); ++revision_;
    return {value.id,value.physical,value.physical_generation,address-owner->first,bytes,value.physical_offset};
  }
  std::vector<MappingSpan> unmap_range(Word address, Word bytes) {
    // Observe a successful native unmap of a union of WHOLE original mappings.
    // Never split a mapping, bridge a hole, or cross a reservation boundary.
    // The exact single-mapping unmap() contract above is intentionally unchanged.
    const auto finish=end(address,bytes);
    const auto owner=allocation(address,bytes);
    if (owner==allocations_.end() || !owner->second.reservation)
      throw std::runtime_error("memory_unmap_reservation_bounds");
    const auto keys=covering_maps(address,bytes);
    if (keys.empty() || keys.size()>4096) throw std::runtime_error("memory_unmap_segment_capacity");
    std::vector<MappingSpan> result;
    result.reserve(keys.size()); // all retirement storage before any mutation
    std::vector<std::pair<Word,Word>> physical_uses;
    physical_uses.reserve(keys.size());
    Word cursor=address;
    std::size_t retired_access=0;
    for (const auto key:keys) {
      const auto& value=mappings_.at(key);
      const auto physical=physical_.find(value.physical);
      if (key!=cursor || value.allocation!=owner->second.id || value.bytes>finish-cursor)
        throw std::runtime_error("memory_partial_or_unknown_unmap");
      if (physical==physical_.end() || physical->second.generation!=value.physical_generation)
        throw std::runtime_error("memory_unmap_physical_lifetime");
      cursor=sum(cursor,value.bytes);
      if (value.access.size()>access_count_-retired_access)
        throw std::runtime_error("memory_unmap_access_accounting");
      retired_access+=value.access.size();
      auto uses=std::find_if(physical_uses.begin(),physical_uses.end(),[&](const auto& entry) { return entry.first==value.physical; });
      if (uses==physical_uses.end()) physical_uses.emplace_back(value.physical,1);
      else ++uses->second;
      result.push_back({value.id,value.physical,value.physical_generation,key-owner->first,value.bytes,value.physical_offset});
    }
    if (cursor!=finish || counts_.mappings<keys.size() || counts_.mapped_bytes<bytes)
      throw std::runtime_error("memory_unmap_range_accounting");
    for (const auto& [key,uses]:physical_uses)
      if (physical_.at(key).maps<uses) throw std::runtime_error("memory_unmap_physical_lifetime");
    const auto next_revision=sum(revision_,static_cast<Word>(keys.size()));
    // Everything below is non-allocating. Physical references are prevalidated,
    // including a handle released while several mappings still retain it.
    for (std::size_t i=0;i<keys.size();++i) {
      mappings_.erase(keys[i]);
      auto& physical=physical_.at(result[i].physical_id);
      --physical.maps;
      if (!physical.retained && !physical.maps) {
        counts_.physical_bytes-=physical.bytes;
        physical_.erase(result[i].physical_id);
      }
    }
    access_count_-=retired_access;
    counts_.mappings-=keys.size(); counts_.mapped_bytes-=bytes; revision_=next_revision;
    return result;
  }
  void set_access(Word address, Word bytes, int device, unsigned flags) {
    if (device<0 || (flags!=0 && flags!=1 && flags!=3)) throw std::runtime_error("memory_access_descriptor");
    const auto owner=allocation(address,bytes);
    if (owner==allocations_.end() || !owner->second.reservation)
      throw std::runtime_error("memory_access_reservation_bounds");
    const auto finish=end(address,bytes);
    const auto keys=covering_maps(address,bytes);
    std::vector<std::pair<Word,std::vector<AccessSpan>>> updates;
    auto total=access_count_;
    for (auto key:keys) {
      const auto& map=mappings_.at(key);
      const auto lo=std::max(address,key)-key, hi=std::min(finish,sum(key,map.bytes))-key;
      std::vector<AccessSpan> next;
      for (const auto& access:map.access) {
        const auto old_end=sum(access.offset_bytes,access.bytes);
        if (access.device!=device || old_end<=lo || access.offset_bytes>=hi) next.push_back(access);
        else {
          if (access.offset_bytes<lo) next.push_back({access.offset_bytes,lo-access.offset_bytes,device,access.flags});
          if (old_end>hi) next.push_back({hi,old_end-hi,device,access.flags});
        }
      }
      next.push_back({lo,hi-lo,device,flags});
      total-=map.access.size();
      if (total>access_cap_ || next.size()>access_cap_-total) throw std::runtime_error("memory_access_capacity");
      total+=next.size();
      updates.emplace_back(key,std::move(next));
    }
    for (auto& [key,access]:updates) mappings_.at(key).access=std::move(access);
    access_count_=total; ++revision_;
  }
  [[nodiscard]] Resolution resolve(Word address, Word bytes, int device, Word context=0, Word alignment=16) const {
    if (device<0) throw std::runtime_error("memory_device");
    if (!alignment || (alignment & (alignment-1)) || alignment>1024ULL*1024*1024)
      throw std::runtime_error("memory_alignment");
    const auto owner=allocation(address,bytes);
    Resolution out; out.revision=revision_; out.bytes=bytes; out.alignment=alignment;
    if (owner==allocations_.end()) return out;
    const auto& allocation_value=owner->second;
    out.allocation_id=allocation_value.id; out.generation=allocation_value.generation;
    out.offset_bytes=address-owner->first; out.allocation_bytes=allocation_value.bytes;
    out.base_mod_alignment=owner->first%alignment; out.known=true;
    if (!allocation_value.reservation) {
      out.mapped=true;
      out.readable=out.writable=context && context==allocation_value.context;
      return out;
    }
    const auto finish=sum(address,bytes);
    Word map_bytes=0, read_bytes=0, write_bytes=0;
    for (const auto& [base,map]:mappings_) {
      const auto lo=std::max(base,address), hi=std::min(sum(base,map.bytes),finish);
      if (lo>=hi || map.allocation!=out.allocation_id) continue;
      if (out.mappings.size()>=4096) throw std::runtime_error("memory_resolution_capacity");
      out.mappings.push_back({map.id,map.physical,map.physical_generation,lo-owner->first,hi-lo,
                             sum(map.physical_offset,lo-base)});
      map_bytes=sum(map_bytes,hi-lo);
      for (const auto& access:map.access) {
        const auto start=std::max(lo,sum(base,access.offset_bytes));
        const auto stop=std::min(hi,sum(sum(base,access.offset_bytes),access.bytes));
        if (start>=stop || access.device!=device) continue;
        if (out.access.size()>=4096) throw std::runtime_error("memory_resolution_capacity");
        out.access.push_back({start-owner->first,stop-start,device,access.flags});
        if (access.flags&1) read_bytes=sum(read_bytes,stop-start);
        if (access.flags&2) write_bytes=sum(write_bytes,stop-start);
      }
    }
    out.mapped=map_bytes==bytes; out.readable=out.mapped && read_bytes==bytes;
    out.writable=out.mapped && write_bytes==bytes;
    return out;
  }
};
}
