#include "residency/compression.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <unordered_map>
#include <utility>

namespace xvram::residency {
namespace {

constexpr std::uint64_t backing_base_charge = 256;
constexpr std::uint64_t backing_block_charge = 128;
constexpr std::uint64_t token_low_seed = 0x585652414D4C4F57ULL;
constexpr std::uint64_t token_high_seed = 0x585652414D484947ULL;
constexpr std::uint64_t token_low_prime = 0x100000001B3ULL;
constexpr std::uint64_t token_high_prime = 0x9E3779B185EBCA87ULL;
constexpr std::uint64_t token_high_word_xor = 0xA5A5A5A5A5A5A5A5ULL;

[[nodiscard]] bool checked_add(const std::uint64_t left, const std::uint64_t right,
                               std::uint64_t& output) noexcept {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    return false;
  }
  output = left + right;
  return true;
}

[[nodiscard]] bool checked_multiply(const std::uint64_t left, const std::uint64_t right,
                                    std::uint64_t& output) noexcept {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
    return false;
  }
  output = left * right;
  return true;
}

[[nodiscard]] bool range_fits(const std::uint64_t offset, const std::uint64_t length,
                              const std::uint64_t limit) noexcept {
  return offset <= limit && length <= limit - offset;
}

[[nodiscard]] std::uint64_t mix64(std::uint64_t value) noexcept {
  value ^= value >> 30U;
  value *= 0xBF58476D1CE4E5B9ULL;
  value ^= value >> 27U;
  value *= 0x94D049BB133111EBULL;
  return value ^ (value >> 31U);
}

[[nodiscard]] ContentToken token_for_block(const std::span<const std::byte> bytes,
                                           const std::uint64_t block_index) noexcept {
  std::uint64_t low = token_low_seed;
  std::uint64_t high = token_high_seed;
  std::size_t offset = 0;
  while (bytes.size() - offset >= sizeof(std::uint64_t)) {
    std::uint64_t word = 0;
    std::memcpy(&word, bytes.data() + static_cast<std::ptrdiff_t>(offset), sizeof(word));
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    word = ((word & 0x00000000000000FFULL) << 56U) | ((word & 0x000000000000FF00ULL) << 40U) |
           ((word & 0x0000000000FF0000ULL) << 24U) | ((word & 0x00000000FF000000ULL) << 8U) |
           ((word & 0x000000FF00000000ULL) >> 8U) | ((word & 0x0000FF0000000000ULL) >> 24U) |
           ((word & 0x00FF000000000000ULL) >> 40U) | ((word & 0xFF00000000000000ULL) >> 56U);
#endif
    low = low * token_low_prime + word + 1U;
    high = high * token_high_prime + (word ^ token_high_word_xor) + 1U;
    offset += sizeof(word);
  }
  if (offset != bytes.size()) {
    std::uint64_t word = 0;
    std::uint32_t shift = 0;
    while (offset < bytes.size()) {
      word |= static_cast<std::uint64_t>(std::to_integer<unsigned char>(bytes[offset])) << shift;
      shift += 8U;
      ++offset;
    }
    low = low * token_low_prime + word + 1U;
    high = high * token_high_prime + (word ^ token_high_word_xor) + 1U;
  }
  const std::uint64_t position =
      mix64(block_index ^ (static_cast<std::uint64_t>(bytes.size()) << 32U));
  return {mix64(high ^ position), mix64(low + position)};
}

[[nodiscard]] ContentToken token_for_zero_block(const std::size_t size,
                                                const std::uint64_t block_index) noexcept {
  const auto advance = [](const std::uint64_t initial, std::uint64_t count,
                          std::uint64_t multiplier, std::uint64_t addend) noexcept {
    std::uint64_t result_multiplier = 1;
    std::uint64_t result_addend = 0;
    while (count != 0) {
      if ((count & 1U) != 0) {
        result_addend = multiplier * result_addend + addend;
        result_multiplier = multiplier * result_multiplier;
      }
      addend = multiplier * addend + addend;
      multiplier *= multiplier;
      count >>= 1U;
    }
    return result_multiplier * initial + result_addend;
  };
  const std::uint64_t words = (static_cast<std::uint64_t>(size) + 7U) / 8U;
  const std::uint64_t low = advance(token_low_seed, words, token_low_prime, 1);
  const std::uint64_t high =
      advance(token_high_seed, words, token_high_prime, token_high_word_xor + 1U);
  const std::uint64_t position = mix64(block_index ^ (static_cast<std::uint64_t>(size) << 32U));
  return {mix64(high ^ position), mix64(low + position)};
}

[[nodiscard]] ContentToken combine_token(ContentToken token, const ContentToken block) noexcept {
  token.high ^= block.high;
  token.low ^= block.low;
  return token;
}

[[nodiscard]] ContentToken token_seed(const std::uint64_t valid_bytes) noexcept {
  return {mix64(token_high_seed ^ valid_bytes), mix64(token_low_seed + valid_bytes)};
}

[[nodiscard]] std::uint64_t block_count_for(const std::uint64_t valid_bytes) noexcept {
  return valid_bytes / compression_block_bytes +
         (valid_bytes % compression_block_bytes == 0 ? 0U : 1U);
}

[[nodiscard]] std::uint32_t block_valid_bytes(const std::uint64_t valid_bytes,
                                              const std::uint64_t block_index) noexcept {
  const std::uint64_t offset = block_index * compression_block_bytes;
  return static_cast<std::uint32_t>(std::min(compression_block_bytes, valid_bytes - offset));
}

[[nodiscard]] bool finite_nonnegative(const double value) noexcept {
  return std::isfinite(value) && value >= 0.0;
}

[[nodiscard]] bool valid_path(const CompressionPath path) noexcept {
  return path == CompressionPath::raw || path == CompressionPath::cpu_lz4_gpu_decode ||
         path == CompressionPath::nvcomp_gpu_codec;
}

} // namespace

std::string_view compression_mode_name(const CompressionMode mode) noexcept {
  switch (mode) {
  case CompressionMode::disabled:
    return "disabled";
  case CompressionMode::adaptive:
    return "adaptive";
  case CompressionMode::capacity:
    return "capacity";
  }
  return "invalid";
}

std::string_view compression_codec_name(const CompressionCodec codec) noexcept {
  switch (codec) {
  case CompressionCodec::automatic:
    return "auto";
  case CompressionCodec::lz4:
    return "lz4";
  }
  return "invalid";
}

std::string_view compression_path_name(const CompressionPath path) noexcept {
  switch (path) {
  case CompressionPath::raw:
    return "raw";
  case CompressionPath::cpu_lz4_gpu_decode:
    return "cpu_lz4_gpu_decode";
  case CompressionPath::nvcomp_gpu_codec:
    return "nvcomp_gpu_codec";
  }
  return "invalid";
}

std::string_view backing_representation_name(const BackingRepresentation value) noexcept {
  switch (value) {
  case BackingRepresentation::invalid:
    return "invalid";
  case BackingRepresentation::implicit_zero:
    return "implicit_zero";
  case BackingRepresentation::raw:
    return "raw";
  case BackingRepresentation::lz4_blocks:
    return "lz4_blocks";
  }
  return "invalid";
}

std::string_view codec_status_name(const CodecStatus status) noexcept {
  switch (status) {
  case CodecStatus::success:
    return "success";
  case CodecStatus::invalid_argument:
    return "invalid_argument";
  case CodecStatus::output_too_small:
    return "output_too_small";
  case CodecStatus::corrupt_input:
    return "corrupt_input";
  case CodecStatus::internal_failure:
    return "internal_failure";
  }
  return "invalid";
}

ContentToken compute_content_token(const std::span<const std::byte> bytes) noexcept {
  ContentToken result = token_seed(static_cast<std::uint64_t>(bytes.size()));
  std::size_t offset = 0;
  std::uint64_t block_index = 0;
  while (offset < bytes.size()) {
    const std::size_t count = std::min<std::size_t>(
        static_cast<std::size_t>(compression_block_bytes), bytes.size() - offset);
    result = combine_token(result, token_for_block(bytes.subspan(offset, count), block_index));
    offset += count;
    ++block_index;
  }
  return result;
}

CodecStatus compute_lz4_blocks_content_token(const Lz4BlocksV1& container, const BlockCodec& codec,
                                             ContentToken& token) noexcept {
  token = {};
  if (container.format_version != lz4_blocks_format_version || container.valid_bytes == 0 ||
      container.blocks.size() != block_count_for(container.valid_bytes)) {
    return CodecStatus::invalid_argument;
  }
  try {
    std::vector<std::byte> decoded(static_cast<std::size_t>(compression_block_bytes));
    ContentToken combined = token_seed(container.valid_bytes);
    for (std::size_t index = 0; index < container.blocks.size(); ++index) {
      const Lz4BlockV1& block = container.blocks[index];
      const std::uint32_t expected =
          block_valid_bytes(container.valid_bytes, static_cast<std::uint64_t>(index));
      if (block.uncompressed_bytes != expected) {
        return CodecStatus::corrupt_input;
      }
      const std::span<std::byte> output{decoded.data(), expected};
      switch (block.storage) {
      case BlockStorage::implicit_zero:
        if (!block.payload.empty()) {
          return CodecStatus::corrupt_input;
        }
        std::fill(output.begin(), output.end(), std::byte{0});
        break;
      case BlockStorage::raw:
        if (block.payload.size() != expected) {
          return CodecStatus::corrupt_input;
        }
        std::copy(block.payload.begin(), block.payload.end(), output.begin());
        break;
      case BlockStorage::lz4:
        if (block.payload.empty() || block.payload.size() >= expected ||
            codec.decompress(block.payload, output) != CodecStatus::success) {
          return CodecStatus::corrupt_input;
        }
        break;
      }
      combined =
          combine_token(combined, token_for_block(output, static_cast<std::uint64_t>(index)));
    }
    token = combined;
    return CodecStatus::success;
  } catch (const std::bad_alloc&) {
    return CodecStatus::internal_failure;
  }
}

struct HostBudgetLedger::Impl {
  struct Record {
    HostBudgetCategory category = HostBudgetCategory::conversion_scratch;
    std::uint64_t bytes = 0;
  };

  mutable std::mutex mutex;
  std::uint64_t limit = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t next_id = 1;
  HostBudgetSnapshot counters;
  std::unordered_map<std::uint64_t, Record> reservations;
};

namespace {

[[nodiscard]] std::uint64_t& category_counter(HostBudgetSnapshot& value,
                                              const HostBudgetCategory category) noexcept {
  switch (category) {
  case HostBudgetCategory::authoritative:
    return value.authoritative_bytes;
  case HostBudgetCategory::conversion_scratch:
    return value.conversion_scratch_bytes;
  case HostBudgetCategory::pinned_staging:
    return value.pinned_staging_bytes;
  case HostBudgetCategory::spill:
    return value.spill_bytes;
  }
  return value.conversion_scratch_bytes;
}

[[nodiscard]] std::uint64_t& category_peak_counter(HostBudgetSnapshot& value,
                                                   const HostBudgetCategory category) noexcept {
  switch (category) {
  case HostBudgetCategory::authoritative:
    return value.authoritative_peak_bytes;
  case HostBudgetCategory::conversion_scratch:
    return value.conversion_scratch_peak_bytes;
  case HostBudgetCategory::pinned_staging:
    return value.pinned_staging_peak_bytes;
  case HostBudgetCategory::spill:
    return value.spill_peak_bytes;
  }
  return value.conversion_scratch_peak_bytes;
}

} // namespace

HostBudgetLedger::HostBudgetLedger(const std::uint64_t limit_bytes)
    : impl_(std::make_unique<Impl>()) {
  impl_->limit = limit_bytes == 0 ? std::numeric_limits<std::uint64_t>::max() : limit_bytes;
  impl_->counters.limit_bytes = impl_->limit;
}

HostBudgetLedger::~HostBudgetLedger() = default;
HostBudgetLedger::HostBudgetLedger(HostBudgetLedger&&) noexcept = default;
HostBudgetLedger& HostBudgetLedger::operator=(HostBudgetLedger&&) noexcept = default;

HostBudgetStatus HostBudgetLedger::reserve(const HostBudgetCategory category,
                                           const std::uint64_t bytes,
                                           HostBudgetReservation& reservation) noexcept {
  reservation = {};
  if (!impl_) {
    return HostBudgetStatus::invalid_argument;
  }
  if (bytes == 0) {
    reservation.category = category;
    return HostBudgetStatus::success;
  }
  std::lock_guard lock(impl_->mutex);
  if (bytes > impl_->limit - impl_->counters.total_bytes) {
    const std::uint64_t room =
        std::numeric_limits<std::uint64_t>::max() - impl_->counters.rejected_bytes;
    impl_->counters.rejected_bytes += std::min(room, bytes);
    return HostBudgetStatus::limit_exceeded;
  }
  if (impl_->next_id == 0) {
    return HostBudgetStatus::reservation_overflow;
  }
  const std::uint64_t id = impl_->next_id++;
  try {
    impl_->reservations.emplace(id, Impl::Record{category, bytes});
  } catch (const std::bad_alloc&) {
    return HostBudgetStatus::allocation_failure;
  }
  impl_->counters.total_bytes += bytes;
  category_counter(impl_->counters, category) += bytes;
  category_peak_counter(impl_->counters, category) =
      std::max(category_peak_counter(impl_->counters, category),
               category_counter(impl_->counters, category));
  impl_->counters.peak_bytes = std::max(impl_->counters.peak_bytes, impl_->counters.total_bytes);
  reservation = {id, category, bytes};
  return HostBudgetStatus::success;
}

HostBudgetStatus HostBudgetLedger::release(HostBudgetReservation& reservation) noexcept {
  if (!impl_ || (reservation.id == 0 && reservation.bytes != 0)) {
    return HostBudgetStatus::invalid_argument;
  }
  if (reservation.bytes == 0) {
    reservation = {};
    return HostBudgetStatus::success;
  }
  std::lock_guard lock(impl_->mutex);
  const auto found = impl_->reservations.find(reservation.id);
  if (found == impl_->reservations.end()) {
    return HostBudgetStatus::unknown_reservation;
  }
  if (found->second.category != reservation.category || found->second.bytes != reservation.bytes) {
    return HostBudgetStatus::reservation_mismatch;
  }
  impl_->counters.total_bytes -= reservation.bytes;
  category_counter(impl_->counters, reservation.category) -= reservation.bytes;
  impl_->reservations.erase(found);
  reservation = {};
  return HostBudgetStatus::success;
}

HostBudgetSnapshot HostBudgetLedger::snapshot() const noexcept {
  if (!impl_) {
    return {};
  }
  std::lock_guard lock(impl_->mutex);
  return impl_->counters;
}

HostBudgetStatus HostBudgetLedger::commit_authoritative(
    HostBudgetReservation& conversion_reservation, const std::uint64_t previous_authoritative_bytes,
    const std::uint64_t replacement_authoritative_bytes) noexcept {
  if (!impl_ ||
      (conversion_reservation.category != HostBudgetCategory::conversion_scratch &&
       conversion_reservation.category != HostBudgetCategory::spill) ||
      conversion_reservation.bytes < replacement_authoritative_bytes ||
      (conversion_reservation.id == 0 && conversion_reservation.bytes != 0)) {
    return HostBudgetStatus::reservation_mismatch;
  }
  std::lock_guard lock(impl_->mutex);
  if (previous_authoritative_bytes > impl_->counters.authoritative_bytes) {
    return HostBudgetStatus::reservation_mismatch;
  }
  if (conversion_reservation.bytes != 0) {
    const auto found = impl_->reservations.find(conversion_reservation.id);
    if (found == impl_->reservations.end() ||
        found->second.category != conversion_reservation.category ||
        found->second.bytes != conversion_reservation.bytes) {
      return HostBudgetStatus::reservation_mismatch;
    }
    impl_->reservations.erase(found);
  }
  const std::uint64_t unused_conversion_bytes =
      conversion_reservation.bytes - replacement_authoritative_bytes;
  category_counter(impl_->counters, conversion_reservation.category) -=
      conversion_reservation.bytes;
  impl_->counters.authoritative_bytes -= previous_authoritative_bytes;
  impl_->counters.authoritative_bytes += replacement_authoritative_bytes;
  impl_->counters.authoritative_peak_bytes =
      std::max(impl_->counters.authoritative_peak_bytes, impl_->counters.authoritative_bytes);
  impl_->counters.total_bytes -= previous_authoritative_bytes;
  impl_->counters.total_bytes -= unused_conversion_bytes;
  conversion_reservation = {};
  return HostBudgetStatus::success;
}

HostBudgetStatus
HostBudgetLedger::release_authoritative(const std::uint64_t authoritative_bytes) noexcept {
  if (!impl_) {
    return HostBudgetStatus::invalid_argument;
  }
  std::lock_guard lock(impl_->mutex);
  if (authoritative_bytes > impl_->counters.authoritative_bytes ||
      authoritative_bytes > impl_->counters.total_bytes) {
    return HostBudgetStatus::reservation_mismatch;
  }
  impl_->counters.authoritative_bytes -= authoritative_bytes;
  impl_->counters.total_bytes -= authoritative_bytes;
  return HostBudgetStatus::success;
}

std::string_view host_budget_status_name(const HostBudgetStatus status) noexcept {
  switch (status) {
  case HostBudgetStatus::success:
    return "success";
  case HostBudgetStatus::invalid_argument:
    return "invalid_argument";
  case HostBudgetStatus::limit_exceeded:
    return "limit_exceeded";
  case HostBudgetStatus::reservation_overflow:
    return "reservation_overflow";
  case HostBudgetStatus::allocation_failure:
    return "allocation_failure";
  case HostBudgetStatus::unknown_reservation:
    return "unknown_reservation";
  case HostBudgetStatus::reservation_mismatch:
    return "reservation_mismatch";
  }
  return "invalid";
}

std::string_view backing_status_name(const BackingStatus status) noexcept {
  switch (status) {
  case BackingStatus::success:
    return "success";
  case BackingStatus::invalid_configuration:
    return "invalid_configuration";
  case BackingStatus::invalid_argument:
    return "invalid_argument";
  case BackingStatus::duplicate_chunk:
    return "duplicate_chunk";
  case BackingStatus::chunk_not_found:
    return "chunk_not_found";
  case BackingStatus::invalid_state:
    return "invalid_state";
  case BackingStatus::stale_generation:
    return "stale_generation";
  case BackingStatus::generation_overflow:
    return "generation_overflow";
  case BackingStatus::range_overflow:
    return "range_overflow";
  case BackingStatus::range_out_of_bounds:
    return "range_out_of_bounds";
  case BackingStatus::host_budget_exceeded:
    return "host_budget_exceeded";
  case BackingStatus::allocation_failure:
    return "allocation_failure";
  case BackingStatus::codec_failure:
    return "codec_failure";
  case BackingStatus::corrupt_data:
    return "corrupt_data";
  case BackingStatus::not_beneficial:
    return "not_beneficial";
  case BackingStatus::internal_failure:
    return "internal_failure";
  }
  return "invalid";
}

// HostBackingStore implementation follows below. Keeping it in this translation unit ensures that
// immutable payload generations and budget reservations cannot be bypassed by runtime callers.

struct HostBackingStore::Impl {
  struct StoredBlock {
    BlockStorage storage = BlockStorage::implicit_zero;
    std::uint32_t uncompressed_bytes = 0;
    std::shared_ptr<const std::vector<std::byte>> payload;
    ContentToken token;
  };

  struct Image {
    BackingRepresentation representation = BackingRepresentation::invalid;
    std::uint64_t valid_bytes = 0;
    std::uint64_t generation = 0;
    std::uint64_t stored_payload_bytes = 0;
    std::uint64_t budget_charge_bytes = 0;
    ContentToken content_token;
    std::vector<StoredBlock> blocks;
  };

  struct LocalChunkHash {
    [[nodiscard]] std::size_t operator()(const ChunkKey& key) const noexcept {
      const std::uint64_t mixed = mix64(key.allocation_id.value) ^ mix64(key.chunk_index);
      return static_cast<std::size_t>(mixed);
    }
  };

  HostBackingConfig config;
  const BlockCodec* codec = nullptr;
  HostBudgetLedger ledger;
  mutable std::mutex mutex;
  std::unordered_map<ChunkKey, std::shared_ptr<const Image>, LocalChunkHash> chunks;

  Impl(const HostBackingConfig value, const BlockCodec& value_codec)
      : config(value), codec(&value_codec), ledger(value.host_store_cap_bytes) {}

  [[nodiscard]] bool configured() const noexcept {
    return config.chunk_bytes != 0 &&
           config.chunk_bytes <=
               static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) &&
           codec != nullptr && codec->codec() == CompressionCodec::lz4;
  }

  [[nodiscard]] static HostChunkInfo info(const ChunkKey key, const Image& image) noexcept {
    return {key,
            image.representation,
            image.valid_bytes,
            image.generation,
            image.stored_payload_bytes,
            image.budget_charge_bytes,
            image.content_token};
  }

  [[nodiscard]] std::shared_ptr<const Image> snapshot(const ChunkKey key) const noexcept {
    std::lock_guard lock(mutex);
    const auto found = chunks.find(key);
    return found == chunks.end() ? std::shared_ptr<const Image>{} : found->second;
  }

  [[nodiscard]] BackingResult
  commit_existing(const ChunkKey key, const std::uint64_t expected_generation,
                  std::shared_ptr<const Image> replacement,
                  HostBudgetReservation* preallocated = nullptr) noexcept {
    HostBudgetReservation local_reservation;
    HostBudgetReservation& reservation =
        preallocated == nullptr ? local_reservation : *preallocated;
    if (preallocated == nullptr) {
      const HostBudgetStatus reserved = ledger.reserve(
          HostBudgetCategory::conversion_scratch, replacement->budget_charge_bytes, reservation);
      if (reserved != HostBudgetStatus::success) {
        return {reserved == HostBudgetStatus::limit_exceeded ? BackingStatus::host_budget_exceeded
                                                             : BackingStatus::allocation_failure,
                {}};
      }
    } else if ((reservation.category != HostBudgetCategory::conversion_scratch &&
                reservation.category != HostBudgetCategory::spill) ||
               reservation.bytes < replacement->budget_charge_bytes) {
      static_cast<void>(ledger.release(reservation));
      return {BackingStatus::internal_failure, {}};
    }

    std::lock_guard lock(mutex);
    const auto found = chunks.find(key);
    if (found == chunks.end()) {
      static_cast<void>(ledger.release(reservation));
      return {BackingStatus::chunk_not_found, {}};
    }
    if (found->second->generation != expected_generation) {
      const HostChunkInfo current = info(key, *found->second);
      static_cast<void>(ledger.release(reservation));
      return {BackingStatus::stale_generation, current};
    }
    const HostBudgetStatus committed = ledger.commit_authoritative(
        reservation, found->second->budget_charge_bytes, replacement->budget_charge_bytes);
    if (committed != HostBudgetStatus::success) {
      static_cast<void>(ledger.release(reservation));
      return {BackingStatus::internal_failure, info(key, *found->second)};
    }
    found->second = std::move(replacement);
    return {BackingStatus::success, info(key, *found->second)};
  }
};

namespace {

using StoreImage = HostBackingStore::Impl::Image;
using StoreBlock = HostBackingStore::Impl::StoredBlock;

[[nodiscard]] BackingStatus maximum_image_charge(const std::uint64_t valid_bytes,
                                                 std::uint64_t& charge) noexcept {
  std::uint64_t metadata = 0;
  if (!checked_multiply(block_count_for(valid_bytes), backing_block_charge, metadata) ||
      !checked_add(metadata, backing_base_charge, metadata) ||
      !checked_add(metadata, valid_bytes, charge)) {
    return BackingStatus::range_overflow;
  }
  return BackingStatus::success;
}

class ReservationGuard {
public:
  explicit ReservationGuard(HostBudgetLedger& ledger) noexcept : ledger_(&ledger) {}

  ~ReservationGuard() {
    if (ledger_ != nullptr) {
      static_cast<void>(ledger_->release(reservation_));
    }
  }

  ReservationGuard(const ReservationGuard&) = delete;
  ReservationGuard& operator=(const ReservationGuard&) = delete;

  [[nodiscard]] HostBudgetStatus reserve(const std::uint64_t bytes) noexcept {
    return ledger_->reserve(HostBudgetCategory::conversion_scratch, bytes, reservation_);
  }

  [[nodiscard]] HostBudgetReservation& reservation() noexcept {
    return reservation_;
  }

private:
  HostBudgetLedger* ledger_ = nullptr;
  HostBudgetReservation reservation_;
};

[[nodiscard]] BackingStatus reserve_candidate(const std::uint64_t valid_bytes,
                                              ReservationGuard& guard) noexcept {
  std::uint64_t charge = 0;
  const BackingStatus calculated = maximum_image_charge(valid_bytes, charge);
  if (calculated != BackingStatus::success) {
    return calculated;
  }
  const HostBudgetStatus reserved = guard.reserve(charge);
  if (reserved == HostBudgetStatus::success) {
    return BackingStatus::success;
  }
  return reserved == HostBudgetStatus::limit_exceeded ? BackingStatus::host_budget_exceeded
                                                      : BackingStatus::allocation_failure;
}

[[nodiscard]] BackingStatus charge_image(StoreImage& image) noexcept {
  if (image.representation == BackingRepresentation::invalid) {
    image.stored_payload_bytes = 0;
    image.budget_charge_bytes = 0;
    return BackingStatus::success;
  }
  std::uint64_t payload = 0;
  std::uint64_t payload_capacity = 0;
  for (const StoreBlock& block : image.blocks) {
    if (block.payload) {
      if (!checked_add(payload, static_cast<std::uint64_t>(block.payload->size()), payload) ||
          !checked_add(payload_capacity, static_cast<std::uint64_t>(block.payload->capacity()),
                       payload_capacity)) {
        return BackingStatus::range_overflow;
      }
    }
  }
  std::uint64_t metadata = 0;
  if (!checked_multiply(static_cast<std::uint64_t>(image.blocks.size()), backing_block_charge,
                        metadata) ||
      !checked_add(metadata, backing_base_charge, metadata) ||
      !checked_add(metadata, payload_capacity, image.budget_charge_bytes)) {
    return BackingStatus::range_overflow;
  }
  image.stored_payload_bytes = payload;
  return BackingStatus::success;
}

[[nodiscard]] ContentToken combine_blocks(const StoreImage& image) noexcept {
  ContentToken result = token_seed(image.valid_bytes);
  for (const StoreBlock& block : image.blocks) {
    result = combine_token(result, block.token);
  }
  return result;
}

[[nodiscard]] CodecStatus decode_block(const StoreBlock& block, const BlockCodec& codec,
                                       const std::span<std::byte> output) noexcept {
  if (output.size() != block.uncompressed_bytes) {
    return CodecStatus::invalid_argument;
  }
  switch (block.storage) {
  case BlockStorage::implicit_zero:
    if (block.payload && !block.payload->empty()) {
      return CodecStatus::corrupt_input;
    }
    std::fill(output.begin(), output.end(), std::byte{0});
    return CodecStatus::success;
  case BlockStorage::raw:
    if (!block.payload || block.payload->size() != output.size()) {
      return CodecStatus::corrupt_input;
    }
    std::copy(block.payload->begin(), block.payload->end(), output.begin());
    return CodecStatus::success;
  case BlockStorage::lz4:
    if (!block.payload || block.payload->empty()) {
      return CodecStatus::corrupt_input;
    }
    return codec.decompress(*block.payload, output);
  }
  return CodecStatus::corrupt_input;
}

[[nodiscard]] bool is_all_zero(const std::span<const std::byte> bytes) noexcept {
  return std::all_of(bytes.begin(), bytes.end(),
                     [](const std::byte value) { return value == std::byte{0}; });
}

[[nodiscard]] BackingStatus make_raw_block(const std::span<const std::byte> input,
                                           const std::uint64_t block_index,
                                           StoreBlock& output) noexcept {
  try {
    output.storage = BlockStorage::raw;
    output.uncompressed_bytes = static_cast<std::uint32_t>(input.size());
    output.payload = std::make_shared<const std::vector<std::byte>>(input.begin(), input.end());
    output.token = token_for_block(input, block_index);
    return BackingStatus::success;
  } catch (const std::bad_alloc&) {
    return BackingStatus::allocation_failure;
  }
}

[[nodiscard]] StoreBlock make_zero_block(const std::uint32_t size,
                                         const std::uint64_t block_index) noexcept {
  StoreBlock result;
  result.storage = BlockStorage::implicit_zero;
  result.uncompressed_bytes = size;
  result.token = token_for_zero_block(size, block_index);
  return result;
}

[[nodiscard]] BackingStatus make_encoded_block(const std::span<const std::byte> input,
                                               const std::uint64_t block_index,
                                               const BlockCodec& codec,
                                               StoreBlock& output) noexcept {
  if (is_all_zero(input)) {
    output = make_zero_block(static_cast<std::uint32_t>(input.size()), block_index);
    return BackingStatus::success;
  }
  const std::optional<std::size_t> bound = codec.maximum_compressed_bytes(input.size());
  if (!bound.has_value() || *bound == 0) {
    return BackingStatus::codec_failure;
  }
  try {
    std::vector<std::byte> encoded(*bound);
    std::size_t written = 0;
    const CodecStatus encoded_status = codec.compress(input, encoded, written);
    if (encoded_status != CodecStatus::success || written > encoded.size()) {
      return BackingStatus::codec_failure;
    }
    if (written >= input.size()) {
      return make_raw_block(input, block_index, output);
    }
    // The immutable backing charge is based on stored bytes. Repack out of the temporary
    // compressBound allocation so vector capacity cannot retain almost-raw memory while the
    // budget ledger reports only the compressed size.
    std::vector<std::byte> exact(written);
    std::copy_n(encoded.begin(), written, exact.begin());
    output.storage = BlockStorage::lz4;
    output.uncompressed_bytes = static_cast<std::uint32_t>(input.size());
    output.payload = std::make_shared<const std::vector<std::byte>>(std::move(exact));
    output.token = token_for_block(input, block_index);
    return BackingStatus::success;
  } catch (const std::bad_alloc&) {
    return BackingStatus::allocation_failure;
  }
}

[[nodiscard]] BackingStatus validate_image_blocks(const StoreImage& image,
                                                  const BlockCodec& codec) noexcept {
  const std::uint64_t expected_blocks = block_count_for(image.valid_bytes);
  if (image.blocks.size() != expected_blocks) {
    return BackingStatus::corrupt_data;
  }
  ContentToken token = token_seed(image.valid_bytes);
  try {
    std::vector<std::byte> decoded(static_cast<std::size_t>(compression_block_bytes));
    for (std::uint64_t index = 0; index < expected_blocks; ++index) {
      const StoreBlock& block = image.blocks[static_cast<std::size_t>(index)];
      const std::uint32_t expected = block_valid_bytes(image.valid_bytes, index);
      if (block.uncompressed_bytes != expected) {
        return BackingStatus::corrupt_data;
      }
      const std::span<std::byte> output{decoded.data(), expected};
      if (decode_block(block, codec, output) != CodecStatus::success) {
        return BackingStatus::corrupt_data;
      }
      const ContentToken actual = token_for_block(output, index);
      if (actual != block.token) {
        return BackingStatus::corrupt_data;
      }
      token = combine_token(token, actual);
    }
  } catch (const std::bad_alloc&) {
    return BackingStatus::allocation_failure;
  }
  return token == image.content_token ? BackingStatus::success : BackingStatus::corrupt_data;
}

[[nodiscard]] std::shared_ptr<StoreImage> make_zero_image(const std::uint64_t valid_bytes,
                                                          const std::uint64_t generation,
                                                          BackingStatus& status) noexcept {
  try {
    auto image = std::make_shared<StoreImage>();
    image->representation = BackingRepresentation::implicit_zero;
    image->valid_bytes = valid_bytes;
    image->generation = generation;
    image->content_token = token_seed(valid_bytes);
    const std::uint64_t count = block_count_for(valid_bytes);
    image->blocks.resize(static_cast<std::size_t>(count));
    for (std::uint64_t index = 0; index < count; ++index) {
      StoreBlock& block = image->blocks[static_cast<std::size_t>(index)];
      block = make_zero_block(block_valid_bytes(valid_bytes, index), index);
      image->content_token = combine_token(image->content_token, block.token);
    }
    status = charge_image(*image);
    return status == BackingStatus::success ? image : nullptr;
  } catch (const std::bad_alloc&) {
    status = BackingStatus::allocation_failure;
    return nullptr;
  }
}

[[nodiscard]] std::shared_ptr<StoreImage> make_raw_zero_image(const std::uint64_t valid_bytes,
                                                              const std::uint64_t generation,
                                                              BackingStatus& status) noexcept {
  try {
    auto image = std::make_shared<StoreImage>();
    image->representation = BackingRepresentation::raw;
    image->valid_bytes = valid_bytes;
    image->generation = generation;
    const std::uint64_t count = block_count_for(valid_bytes);
    image->blocks.resize(static_cast<std::size_t>(count));
    for (std::uint64_t index = 0; index < count; ++index) {
      StoreBlock& block = image->blocks[static_cast<std::size_t>(index)];
      block.storage = BlockStorage::raw;
      block.uncompressed_bytes = block_valid_bytes(valid_bytes, index);
      std::vector<std::byte> payload(static_cast<std::size_t>(block.uncompressed_bytes),
                                     std::byte{0});
      block.payload = std::make_shared<const std::vector<std::byte>>(std::move(payload));
      block.token = token_for_zero_block(block.uncompressed_bytes, index);
    }
    image->content_token = combine_blocks(*image);
    status = charge_image(*image);
    return status == BackingStatus::success ? image : nullptr;
  } catch (const std::bad_alloc&) {
    status = BackingStatus::allocation_failure;
    return nullptr;
  }
}

[[nodiscard]] std::shared_ptr<StoreImage> make_raw_image(const std::span<const std::byte> input,
                                                         const std::uint64_t generation,
                                                         BackingStatus& status) noexcept {
  try {
    auto image = std::make_shared<StoreImage>();
    image->representation = BackingRepresentation::raw;
    image->valid_bytes = static_cast<std::uint64_t>(input.size());
    image->generation = generation;
    const std::uint64_t count = block_count_for(image->valid_bytes);
    image->blocks.resize(static_cast<std::size_t>(count));
    for (std::uint64_t index = 0; index < count; ++index) {
      const std::size_t offset = static_cast<std::size_t>(index * compression_block_bytes);
      const std::size_t length =
          static_cast<std::size_t>(block_valid_bytes(image->valid_bytes, index));
      status = make_raw_block(input.subspan(offset, length), index,
                              image->blocks[static_cast<std::size_t>(index)]);
      if (status != BackingStatus::success) {
        return nullptr;
      }
    }
    image->content_token = combine_blocks(*image);
    status = charge_image(*image);
    return status == BackingStatus::success ? image : nullptr;
  } catch (const std::bad_alloc&) {
    status = BackingStatus::allocation_failure;
    return nullptr;
  }
}

[[nodiscard]] std::shared_ptr<StoreImage> make_invalid_image(const std::uint64_t valid_bytes,
                                                             const std::uint64_t generation,
                                                             BackingStatus& status) noexcept {
  try {
    auto image = std::make_shared<StoreImage>();
    image->representation = BackingRepresentation::invalid;
    image->valid_bytes = valid_bytes;
    image->generation = generation;
    status = BackingStatus::success;
    return image;
  } catch (const std::bad_alloc&) {
    status = BackingStatus::allocation_failure;
    return nullptr;
  }
}

[[nodiscard]] BackingStatus next_generation(const std::uint64_t current,
                                            std::uint64_t& next) noexcept {
  if (current == std::numeric_limits<std::uint64_t>::max()) {
    return BackingStatus::generation_overflow;
  }
  next = current + 1U;
  return BackingStatus::success;
}

} // namespace

std::optional<std::uint64_t> maximum_raw_backing_charge(const std::uint64_t valid_bytes) noexcept {
  if (valid_bytes == 0) {
    return std::nullopt;
  }
  std::uint64_t charge = 0;
  return maximum_image_charge(valid_bytes, charge) == BackingStatus::success
             ? std::optional<std::uint64_t>{charge}
             : std::nullopt;
}

struct PreparedRawBacking::Impl {
  std::uint64_t source_generation = 0;
  std::shared_ptr<StoreImage> image;
  std::vector<std::shared_ptr<std::vector<std::byte>>> mutable_payloads;
  bool filled = false;
};

PreparedRawBacking::PreparedRawBacking() = default;
PreparedRawBacking::~PreparedRawBacking() = default;
PreparedRawBacking::PreparedRawBacking(PreparedRawBacking&&) noexcept = default;
PreparedRawBacking& PreparedRawBacking::operator=(PreparedRawBacking&&) noexcept = default;

bool PreparedRawBacking::valid() const noexcept {
  return impl_ != nullptr && impl_->image != nullptr;
}

HostBackingStore::HostBackingStore(const HostBackingConfig config, const BlockCodec& lz4_codec)
    : impl_(std::make_unique<Impl>(config, lz4_codec)) {}

HostBackingStore::~HostBackingStore() = default;
HostBackingStore::HostBackingStore(HostBackingStore&&) noexcept = default;
HostBackingStore& HostBackingStore::operator=(HostBackingStore&&) noexcept = default;

const HostBackingConfig& HostBackingStore::config() const noexcept {
  return impl_->config;
}

HostBudgetLedger& HostBackingStore::budget() noexcept {
  return impl_->ledger;
}

const HostBudgetLedger& HostBackingStore::budget() const noexcept {
  return impl_->ledger;
}

BackingResult HostBackingStore::register_chunk(const ChunkKey key, const std::uint64_t valid_bytes,
                                               const BackingRepresentation initial_state) noexcept {
  if (!impl_ || !impl_->configured()) {
    return {BackingStatus::invalid_configuration, {}};
  }
  if (!key.allocation_id || valid_bytes == 0 || valid_bytes > impl_->config.chunk_bytes ||
      (initial_state != BackingRepresentation::invalid &&
       initial_state != BackingRepresentation::implicit_zero &&
       initial_state != BackingRepresentation::raw)) {
    return {BackingStatus::invalid_argument, {}};
  }
  BackingStatus status = BackingStatus::success;
  std::shared_ptr<StoreImage> image;
  if (initial_state == BackingRepresentation::invalid) {
    image = make_invalid_image(valid_bytes, 1, status);
  } else if (initial_state == BackingRepresentation::raw) {
    image = make_raw_zero_image(valid_bytes, 1, status);
  } else {
    image = make_zero_image(valid_bytes, 1, status);
  }
  if (!image) {
    return {status, {}};
  }
  HostBudgetReservation reservation;
  const HostBudgetStatus reserved = impl_->ledger.reserve(HostBudgetCategory::conversion_scratch,
                                                          image->budget_charge_bytes, reservation);
  if (reserved != HostBudgetStatus::success) {
    return {reserved == HostBudgetStatus::limit_exceeded ? BackingStatus::host_budget_exceeded
                                                         : BackingStatus::allocation_failure,
            {}};
  }
  std::lock_guard lock(impl_->mutex);
  if (impl_->chunks.contains(key)) {
    static_cast<void>(impl_->ledger.release(reservation));
    return {BackingStatus::duplicate_chunk, {}};
  }
  try {
    const auto [found, inserted] = impl_->chunks.emplace(key, image);
    if (!inserted) {
      static_cast<void>(impl_->ledger.release(reservation));
      return {BackingStatus::duplicate_chunk, {}};
    }
    if (impl_->ledger.commit_authoritative(reservation, 0, image->budget_charge_bytes) !=
        HostBudgetStatus::success) {
      impl_->chunks.erase(found);
      static_cast<void>(impl_->ledger.release(reservation));
      return {BackingStatus::internal_failure, {}};
    }
  } catch (const std::bad_alloc&) {
    static_cast<void>(impl_->ledger.release(reservation));
    return {BackingStatus::allocation_failure, {}};
  }
  return {BackingStatus::success, Impl::info(key, *image)};
}

BackingResult HostBackingStore::erase_chunk(const ChunkKey key,
                                            const std::uint64_t expected_generation) noexcept {
  if (!impl_) {
    return {BackingStatus::invalid_configuration, {}};
  }
  std::lock_guard lock(impl_->mutex);
  const auto found = impl_->chunks.find(key);
  if (found == impl_->chunks.end()) {
    return {BackingStatus::chunk_not_found, {}};
  }
  const HostChunkInfo old_info = Impl::info(key, *found->second);
  if (old_info.generation != expected_generation) {
    return {BackingStatus::stale_generation, old_info};
  }
  if (impl_->ledger.release_authoritative(old_info.budget_charge_bytes) !=
      HostBudgetStatus::success) {
    return {BackingStatus::internal_failure, old_info};
  }
  impl_->chunks.erase(found);
  return {BackingStatus::success, old_info};
}

BackingResult HostBackingStore::inspect(const ChunkKey key) const noexcept {
  if (!impl_) {
    return {BackingStatus::invalid_configuration, {}};
  }
  const std::shared_ptr<const StoreImage> image = impl_->snapshot(key);
  return image ? BackingResult{BackingStatus::success, Impl::info(key, *image)}
               : BackingResult{BackingStatus::chunk_not_found, {}};
}

BackingResult HostBackingStore::read(const ChunkKey key, const std::uint64_t offset,
                                     const std::span<std::byte> output) const noexcept {
  if (!impl_) {
    return {BackingStatus::invalid_configuration, {}};
  }
  const std::shared_ptr<const StoreImage> image = impl_->snapshot(key);
  if (!image) {
    return {BackingStatus::chunk_not_found, {}};
  }
  const HostChunkInfo current = Impl::info(key, *image);
  if (image->representation == BackingRepresentation::invalid) {
    return {BackingStatus::invalid_state, current};
  }
  if (!range_fits(offset, static_cast<std::uint64_t>(output.size()), image->valid_bytes)) {
    return {BackingStatus::range_out_of_bounds, current};
  }
  if (image->blocks.size() != block_count_for(image->valid_bytes) ||
      combine_blocks(*image) != image->content_token) {
    return {BackingStatus::corrupt_data, current};
  }
  if (output.empty()) {
    return {BackingStatus::success, current};
  }
  if (image->representation == BackingRepresentation::implicit_zero) {
    for (std::size_t index = 0; index < image->blocks.size(); ++index) {
      const StoreBlock& block = image->blocks[index];
      if (block.storage != BlockStorage::implicit_zero ||
          (block.payload && !block.payload->empty()) ||
          block.uncompressed_bytes !=
              block_valid_bytes(image->valid_bytes, static_cast<std::uint64_t>(index)) ||
          block.token !=
              token_for_zero_block(block.uncompressed_bytes, static_cast<std::uint64_t>(index))) {
        return {BackingStatus::corrupt_data, current};
      }
    }
    std::fill(output.begin(), output.end(), std::byte{0});
    return {BackingStatus::success, current};
  }
  try {
    std::vector<std::byte> decoded(static_cast<std::size_t>(compression_block_bytes));
    std::uint64_t cursor = offset;
    std::size_t destination = 0;
    while (destination < output.size()) {
      const std::uint64_t block_index = cursor / compression_block_bytes;
      const std::uint64_t within = cursor % compression_block_bytes;
      const StoreBlock& block = image->blocks[static_cast<std::size_t>(block_index)];
      const std::span<std::byte> block_output{decoded.data(), block.uncompressed_bytes};
      if (decode_block(block, *impl_->codec, block_output) != CodecStatus::success) {
        return {BackingStatus::corrupt_data, current};
      }
      if (token_for_block(block_output, block_index) != block.token) {
        return {BackingStatus::corrupt_data, current};
      }
      const std::size_t count = std::min<std::size_t>(
          output.size() - destination, static_cast<std::size_t>(block.uncompressed_bytes - within));
      std::copy_n(decoded.begin() + static_cast<std::ptrdiff_t>(within), count,
                  output.begin() + static_cast<std::ptrdiff_t>(destination));
      cursor += static_cast<std::uint64_t>(count);
      destination += count;
    }
  } catch (const std::bad_alloc&) {
    return {BackingStatus::allocation_failure, current};
  }
  return {BackingStatus::success, current};
}

BackingResult HostBackingStore::replace_raw(const ChunkKey key,
                                            const std::uint64_t expected_generation,
                                            const std::span<const std::byte> input,
                                            HostBudgetReservation* candidate_reservation) noexcept {
  if (!impl_) {
    return {BackingStatus::invalid_configuration, {}};
  }
  const std::shared_ptr<const StoreImage> previous = impl_->snapshot(key);
  if (!previous) {
    return {BackingStatus::chunk_not_found, {}};
  }
  const HostChunkInfo current = Impl::info(key, *previous);
  if (previous->generation != expected_generation) {
    return {BackingStatus::stale_generation, current};
  }
  if (input.size() != previous->valid_bytes) {
    return {BackingStatus::invalid_argument, current};
  }
  std::uint64_t generation = 0;
  const BackingStatus generation_status = next_generation(expected_generation, generation);
  if (generation_status != BackingStatus::success) {
    return {generation_status, current};
  }
  ReservationGuard local_reservation(impl_->ledger);
  HostBudgetReservation* reservation = candidate_reservation;
  if (reservation == nullptr) {
    const BackingStatus reserved = reserve_candidate(previous->valid_bytes, local_reservation);
    if (reserved != BackingStatus::success) {
      return {reserved, current};
    }
    reservation = &local_reservation.reservation();
  }
  BackingStatus status = BackingStatus::success;
  std::shared_ptr<StoreImage> replacement = make_raw_image(input, generation, status);
  return replacement
             ? impl_->commit_existing(key, expected_generation, std::move(replacement), reservation)
             : BackingResult{status, current};
}

BackingResult HostBackingStore::prepare_raw_replacement(const ChunkKey key,
                                                        const std::uint64_t expected_generation,
                                                        PreparedRawBacking& output) noexcept {
  output.impl_.reset();
  if (!impl_) {
    return {BackingStatus::invalid_configuration, {}};
  }
  const std::shared_ptr<const StoreImage> previous = impl_->snapshot(key);
  if (!previous) {
    return {BackingStatus::chunk_not_found, {}};
  }
  const HostChunkInfo current = Impl::info(key, *previous);
  if (previous->generation != expected_generation) {
    return {BackingStatus::stale_generation, current};
  }
  std::uint64_t generation = 0;
  const BackingStatus generation_status = next_generation(expected_generation, generation);
  if (generation_status != BackingStatus::success) {
    return {generation_status, current};
  }
  try {
    auto prepared = std::make_unique<PreparedRawBacking::Impl>();
    prepared->source_generation = expected_generation;
    prepared->image = std::make_shared<StoreImage>();
    prepared->image->representation = BackingRepresentation::raw;
    prepared->image->valid_bytes = previous->valid_bytes;
    prepared->image->generation = generation;
    const std::uint64_t count = block_count_for(previous->valid_bytes);
    prepared->image->blocks.resize(static_cast<std::size_t>(count));
    prepared->mutable_payloads.resize(static_cast<std::size_t>(count));
    for (std::uint64_t index = 0; index < count; ++index) {
      StoreBlock& block = prepared->image->blocks[static_cast<std::size_t>(index)];
      block.storage = BlockStorage::raw;
      block.uncompressed_bytes = block_valid_bytes(previous->valid_bytes, index);
      auto payload = std::make_shared<std::vector<std::byte>>(
          static_cast<std::size_t>(block.uncompressed_bytes));
      block.payload = payload;
      prepared->mutable_payloads[static_cast<std::size_t>(index)] = std::move(payload);
    }
    const BackingStatus charged = charge_image(*prepared->image);
    if (charged != BackingStatus::success) {
      return {charged, current};
    }
    output.impl_ = std::move(prepared);
    return {BackingStatus::success, current};
  } catch (const std::bad_alloc&) {
    return {BackingStatus::allocation_failure, current};
  }
}

BackingStatus HostBackingStore::fill_prepared_raw(PreparedRawBacking& prepared,
                                                  const std::span<const std::byte> input) noexcept {
  if (!prepared.impl_ || !prepared.impl_->image ||
      input.size() != prepared.impl_->image->valid_bytes ||
      prepared.impl_->mutable_payloads.size() != prepared.impl_->image->blocks.size()) {
    return BackingStatus::invalid_argument;
  }
  std::size_t offset = 0;
  for (std::size_t index = 0; index < prepared.impl_->image->blocks.size(); ++index) {
    StoreBlock& block = prepared.impl_->image->blocks[index];
    const std::shared_ptr<std::vector<std::byte>>& payload =
        prepared.impl_->mutable_payloads[index];
    if (!payload || payload->size() != block.uncompressed_bytes) {
      return BackingStatus::invalid_state;
    }
    std::copy_n(input.begin() + static_cast<std::ptrdiff_t>(offset), payload->size(),
                payload->begin());
    block.token = token_for_block(*payload, static_cast<std::uint64_t>(index));
    offset += payload->size();
  }
  prepared.impl_->image->content_token = combine_blocks(*prepared.impl_->image);
  prepared.impl_->filled = true;
  return BackingStatus::success;
}

BackingResult
HostBackingStore::commit_prepared_raw(const ChunkKey key, const std::uint64_t expected_generation,
                                      PreparedRawBacking& prepared,
                                      HostBudgetReservation* candidate_reservation) noexcept {
  if (!impl_ || !prepared.impl_ || !prepared.impl_->image || !prepared.impl_->filled ||
      prepared.impl_->source_generation != expected_generation ||
      candidate_reservation == nullptr) {
    return {BackingStatus::invalid_argument, {}};
  }
  std::shared_ptr<const StoreImage> replacement = prepared.impl_->image;
  BackingResult result = impl_->commit_existing(key, expected_generation, std::move(replacement),
                                                candidate_reservation);
  if (result) {
    prepared.impl_.reset();
  }
  return result;
}

BackingResult HostBackingStore::write(const ChunkKey key, const std::uint64_t expected_generation,
                                      const std::uint64_t offset,
                                      const std::span<const std::byte> input) noexcept {
  if (!impl_) {
    return {BackingStatus::invalid_configuration, {}};
  }
  const std::shared_ptr<const StoreImage> previous = impl_->snapshot(key);
  if (!previous) {
    return {BackingStatus::chunk_not_found, {}};
  }
  const HostChunkInfo current = Impl::info(key, *previous);
  if (previous->generation != expected_generation) {
    return {BackingStatus::stale_generation, current};
  }
  if (previous->representation == BackingRepresentation::invalid) {
    return {BackingStatus::invalid_state, current};
  }
  if (!range_fits(offset, static_cast<std::uint64_t>(input.size()), previous->valid_bytes)) {
    return {BackingStatus::range_out_of_bounds, current};
  }
  if (input.empty()) {
    return {BackingStatus::success, current};
  }
  if (offset == 0 && input.size() == previous->valid_bytes) {
    return replace_raw(key, expected_generation, input);
  }
  std::uint64_t generation = 0;
  const BackingStatus generation_status = next_generation(expected_generation, generation);
  if (generation_status != BackingStatus::success) {
    return {generation_status, current};
  }
  ReservationGuard reservation(impl_->ledger);
  const BackingStatus reserved = reserve_candidate(previous->valid_bytes, reservation);
  if (reserved != BackingStatus::success) {
    return {reserved, current};
  }

  try {
    auto replacement = std::make_shared<StoreImage>();
    replacement->representation = previous->representation == BackingRepresentation::raw
                                      ? BackingRepresentation::raw
                                      : BackingRepresentation::lz4_blocks;
    replacement->valid_bytes = previous->valid_bytes;
    replacement->generation = generation;
    const std::uint64_t count = block_count_for(previous->valid_bytes);
    if (previous->representation == BackingRepresentation::implicit_zero) {
      replacement->blocks.reserve(static_cast<std::size_t>(count));
      for (std::uint64_t index = 0; index < count; ++index) {
        replacement->blocks.push_back(
            make_zero_block(block_valid_bytes(previous->valid_bytes, index), index));
      }
    } else {
      replacement->blocks = previous->blocks;
    }

    const std::uint64_t first = offset / compression_block_bytes;
    const std::uint64_t last =
        (offset + static_cast<std::uint64_t>(input.size()) - 1U) / compression_block_bytes;
    std::size_t source_offset = 0;
    std::vector<std::byte> decoded(static_cast<std::size_t>(compression_block_bytes));
    for (std::uint64_t index = first; index <= last; ++index) {
      StoreBlock& block = replacement->blocks[static_cast<std::size_t>(index)];
      const std::span<std::byte> block_output{decoded.data(), block.uncompressed_bytes};
      if (decode_block(block, *impl_->codec, block_output) != CodecStatus::success) {
        return {BackingStatus::corrupt_data, current};
      }
      if (token_for_block(block_output, index) != block.token) {
        return {BackingStatus::corrupt_data, current};
      }
      const std::uint64_t block_begin = index * compression_block_bytes;
      const std::uint64_t write_begin = std::max(offset, block_begin);
      const std::uint64_t write_end = std::min(offset + static_cast<std::uint64_t>(input.size()),
                                               block_begin + block.uncompressed_bytes);
      const std::size_t within = static_cast<std::size_t>(write_begin - block_begin);
      const std::size_t length = static_cast<std::size_t>(write_end - write_begin);
      std::copy_n(input.begin() + static_cast<std::ptrdiff_t>(source_offset), length,
                  decoded.begin() + static_cast<std::ptrdiff_t>(within));
      BackingStatus status = BackingStatus::success;
      StoreBlock updated;
      if (replacement->representation == BackingRepresentation::raw) {
        status = make_raw_block(block_output, index, updated);
      } else {
        status = make_encoded_block(block_output, index, *impl_->codec, updated);
      }
      if (status != BackingStatus::success) {
        return {status, current};
      }
      block = std::move(updated);
      source_offset += length;
    }
    if (source_offset != input.size()) {
      return {BackingStatus::internal_failure, current};
    }
    replacement->content_token = combine_blocks(*replacement);
    const BackingStatus charged = charge_image(*replacement);
    if (charged != BackingStatus::success) {
      return {charged, current};
    }
    return impl_->commit_existing(key, expected_generation, std::move(replacement),
                                  &reservation.reservation());
  } catch (const std::bad_alloc&) {
    return {BackingStatus::allocation_failure, current};
  }
}

BackingResult
HostBackingStore::set_implicit_zero(const ChunkKey key,
                                    const std::uint64_t expected_generation) noexcept {
  if (!impl_) {
    return {BackingStatus::invalid_configuration, {}};
  }
  const std::shared_ptr<const StoreImage> previous = impl_->snapshot(key);
  if (!previous) {
    return {BackingStatus::chunk_not_found, {}};
  }
  const HostChunkInfo current = Impl::info(key, *previous);
  if (previous->generation != expected_generation) {
    return {BackingStatus::stale_generation, current};
  }
  std::uint64_t generation = 0;
  const BackingStatus generation_status = next_generation(expected_generation, generation);
  if (generation_status != BackingStatus::success) {
    return {generation_status, current};
  }
  ReservationGuard reservation(impl_->ledger);
  const HostBudgetStatus reserved = reservation.reserve(backing_base_charge);
  if (reserved != HostBudgetStatus::success) {
    return {reserved == HostBudgetStatus::limit_exceeded ? BackingStatus::host_budget_exceeded
                                                         : BackingStatus::allocation_failure,
            current};
  }
  BackingStatus status = BackingStatus::success;
  std::shared_ptr<StoreImage> replacement =
      make_zero_image(previous->valid_bytes, generation, status);
  return replacement ? impl_->commit_existing(key, expected_generation, std::move(replacement),
                                              &reservation.reservation())
                     : BackingResult{status, current};
}

BackingResult HostBackingStore::invalidate(const ChunkKey key,
                                           const std::uint64_t expected_generation) noexcept {
  if (!impl_) {
    return {BackingStatus::invalid_configuration, {}};
  }
  const std::shared_ptr<const StoreImage> previous = impl_->snapshot(key);
  if (!previous) {
    return {BackingStatus::chunk_not_found, {}};
  }
  const HostChunkInfo current = Impl::info(key, *previous);
  if (previous->generation != expected_generation) {
    return {BackingStatus::stale_generation, current};
  }
  std::uint64_t generation = 0;
  const BackingStatus generation_status = next_generation(expected_generation, generation);
  if (generation_status != BackingStatus::success) {
    return {generation_status, current};
  }
  BackingStatus status = BackingStatus::success;
  std::shared_ptr<StoreImage> replacement =
      make_invalid_image(previous->valid_bytes, generation, status);
  return replacement ? impl_->commit_existing(key, expected_generation, std::move(replacement))
                     : BackingResult{status, current};
}

BackingResult HostBackingStore::compress(const ChunkKey key,
                                         const std::uint64_t expected_generation) noexcept {
  if (!impl_) {
    return {BackingStatus::invalid_configuration, {}};
  }
  const std::shared_ptr<const StoreImage> previous = impl_->snapshot(key);
  if (!previous) {
    return {BackingStatus::chunk_not_found, {}};
  }
  const HostChunkInfo current = Impl::info(key, *previous);
  if (previous->generation != expected_generation) {
    return {BackingStatus::stale_generation, current};
  }
  if (previous->representation == BackingRepresentation::invalid) {
    return {BackingStatus::invalid_state, current};
  }
  if (previous->representation == BackingRepresentation::implicit_zero ||
      previous->representation == BackingRepresentation::lz4_blocks) {
    return {BackingStatus::not_beneficial, current};
  }
  std::uint64_t generation = 0;
  const BackingStatus generation_status = next_generation(expected_generation, generation);
  if (generation_status != BackingStatus::success) {
    return {generation_status, current};
  }
  ReservationGuard reservation(impl_->ledger);
  const BackingStatus reserved = reserve_candidate(previous->valid_bytes, reservation);
  if (reserved != BackingStatus::success) {
    return {reserved, current};
  }
  try {
    auto replacement = std::make_shared<StoreImage>();
    replacement->representation = BackingRepresentation::lz4_blocks;
    replacement->valid_bytes = previous->valid_bytes;
    replacement->generation = generation;
    replacement->blocks.resize(previous->blocks.size());
    std::vector<std::byte> decoded(static_cast<std::size_t>(compression_block_bytes));
    bool compressed_any = false;
    for (std::size_t index = 0; index < previous->blocks.size(); ++index) {
      const StoreBlock& source = previous->blocks[index];
      const std::span<std::byte> block_output{decoded.data(), source.uncompressed_bytes};
      if (decode_block(source, *impl_->codec, block_output) != CodecStatus::success) {
        return {BackingStatus::corrupt_data, current};
      }
      const BackingStatus encoded =
          make_encoded_block(block_output, static_cast<std::uint64_t>(index), *impl_->codec,
                             replacement->blocks[index]);
      if (encoded != BackingStatus::success) {
        return {encoded, current};
      }
      compressed_any = compressed_any || replacement->blocks[index].storage != BlockStorage::raw;
    }
    replacement->content_token = combine_blocks(*replacement);
    if (replacement->content_token != previous->content_token) {
      return {BackingStatus::corrupt_data, current};
    }
    const BackingStatus charged = charge_image(*replacement);
    if (charged != BackingStatus::success) {
      return {charged, current};
    }
    if (!compressed_any || replacement->stored_payload_bytes >= previous->valid_bytes) {
      return {BackingStatus::not_beneficial, current};
    }
    const BackingStatus validated = validate_image_blocks(*replacement, *impl_->codec);
    if (validated != BackingStatus::success) {
      return {validated, current};
    }
    return impl_->commit_existing(key, expected_generation, std::move(replacement),
                                  &reservation.reservation());
  } catch (const std::bad_alloc&) {
    return {BackingStatus::allocation_failure, current};
  }
}

BackingResult HostBackingStore::materialize_raw(const ChunkKey key,
                                                const std::uint64_t expected_generation) noexcept {
  if (!impl_) {
    return {BackingStatus::invalid_configuration, {}};
  }
  const std::shared_ptr<const StoreImage> previous = impl_->snapshot(key);
  if (!previous) {
    return {BackingStatus::chunk_not_found, {}};
  }
  const HostChunkInfo current = Impl::info(key, *previous);
  if (previous->generation != expected_generation) {
    return {BackingStatus::stale_generation, current};
  }
  if (previous->representation == BackingRepresentation::invalid) {
    return {BackingStatus::invalid_state, current};
  }
  if (previous->representation == BackingRepresentation::raw) {
    return {BackingStatus::success, current};
  }
  std::uint64_t generation = 0;
  const BackingStatus generation_status = next_generation(expected_generation, generation);
  if (generation_status != BackingStatus::success) {
    return {generation_status, current};
  }
  ReservationGuard reservation(impl_->ledger);
  const BackingStatus reserved = reserve_candidate(previous->valid_bytes, reservation);
  if (reserved != BackingStatus::success) {
    return {reserved, current};
  }
  try {
    auto replacement = std::make_shared<StoreImage>();
    replacement->representation = BackingRepresentation::raw;
    replacement->valid_bytes = previous->valid_bytes;
    replacement->generation = generation;
    const std::uint64_t count = block_count_for(previous->valid_bytes);
    replacement->blocks.resize(static_cast<std::size_t>(count));
    std::vector<std::byte> decoded(static_cast<std::size_t>(compression_block_bytes));
    for (std::uint64_t index = 0; index < count; ++index) {
      const std::uint32_t length = block_valid_bytes(previous->valid_bytes, index);
      const std::span<std::byte> block_output{decoded.data(), length};
      if (previous->representation == BackingRepresentation::implicit_zero) {
        std::fill(block_output.begin(), block_output.end(), std::byte{0});
      } else if (decode_block(previous->blocks[static_cast<std::size_t>(index)], *impl_->codec,
                              block_output) != CodecStatus::success) {
        return {BackingStatus::corrupt_data, current};
      }
      const BackingStatus made =
          make_raw_block(block_output, index, replacement->blocks[static_cast<std::size_t>(index)]);
      if (made != BackingStatus::success) {
        return {made, current};
      }
    }
    replacement->content_token = combine_blocks(*replacement);
    if (replacement->content_token != previous->content_token) {
      return {BackingStatus::corrupt_data, current};
    }
    const BackingStatus charged = charge_image(*replacement);
    if (charged != BackingStatus::success) {
      return {charged, current};
    }
    return impl_->commit_existing(key, expected_generation, std::move(replacement),
                                  &reservation.reservation());
  } catch (const std::bad_alloc&) {
    return {BackingStatus::allocation_failure, current};
  }
}

BackingResult
HostBackingStore::commit_lz4_blocks(const ChunkKey key, const std::uint64_t expected_generation,
                                    Lz4BlocksV1 candidate,
                                    HostBudgetReservation* candidate_reservation) noexcept {
  if (!impl_) {
    return {BackingStatus::invalid_configuration, {}};
  }
  const std::shared_ptr<const StoreImage> previous = impl_->snapshot(key);
  if (!previous) {
    return {BackingStatus::chunk_not_found, {}};
  }
  const HostChunkInfo current = Impl::info(key, *previous);
  if (previous->generation != expected_generation) {
    return {BackingStatus::stale_generation, current};
  }
  std::uint64_t generation = 0;
  const BackingStatus generation_status = next_generation(expected_generation, generation);
  if (generation_status != BackingStatus::success) {
    return {generation_status, current};
  }
  if (candidate.format_version != lz4_blocks_format_version ||
      candidate.valid_bytes != previous->valid_bytes || candidate.generation != generation ||
      candidate.blocks.size() != block_count_for(candidate.valid_bytes)) {
    return {BackingStatus::invalid_argument, current};
  }
  try {
    auto replacement = std::make_shared<StoreImage>();
    replacement->representation = BackingRepresentation::lz4_blocks;
    replacement->valid_bytes = candidate.valid_bytes;
    replacement->generation = candidate.generation;
    replacement->content_token = candidate.content_token;
    replacement->blocks.resize(candidate.blocks.size());
    std::vector<std::byte> decoded(static_cast<std::size_t>(compression_block_bytes));
    for (std::size_t index = 0; index < candidate.blocks.size(); ++index) {
      Lz4BlockV1& source = candidate.blocks[index];
      StoreBlock& destination = replacement->blocks[index];
      const std::uint32_t expected =
          block_valid_bytes(candidate.valid_bytes, static_cast<std::uint64_t>(index));
      if (source.uncompressed_bytes != expected ||
          (source.storage == BlockStorage::implicit_zero && !source.payload.empty()) ||
          (source.storage == BlockStorage::raw && source.payload.size() != expected) ||
          (source.storage == BlockStorage::lz4 && source.payload.empty())) {
        return {BackingStatus::corrupt_data, current};
      }
      destination.storage = source.storage;
      destination.uncompressed_bytes = source.uncompressed_bytes;
      if (!source.payload.empty()) {
        destination.payload =
            std::make_shared<const std::vector<std::byte>>(std::move(source.payload));
      }
      const std::span<std::byte> block_output{decoded.data(), expected};
      if (decode_block(destination, *impl_->codec, block_output) != CodecStatus::success) {
        return {BackingStatus::corrupt_data, current};
      }
      destination.token = token_for_block(block_output, static_cast<std::uint64_t>(index));
    }
    if (combine_blocks(*replacement) != replacement->content_token) {
      return {BackingStatus::corrupt_data, current};
    }
    const BackingStatus charged = charge_image(*replacement);
    if (charged != BackingStatus::success) {
      return {charged, current};
    }
    if (replacement->stored_payload_bytes > replacement->valid_bytes) {
      return {BackingStatus::corrupt_data, current};
    }
    // The loop above already decoded every block, recomputed every block token, and reconciled the
    // aggregate content token. Re-running validate_image_blocks() here would duplicate the complete
    // trust-boundary verification before the same immutable image is atomically committed.
    return impl_->commit_existing(key, expected_generation, std::move(replacement),
                                  candidate_reservation);
  } catch (const std::bad_alloc&) {
    return {BackingStatus::allocation_failure, current};
  }
}

BackingStatus HostBackingStore::export_lz4_blocks(const ChunkKey key,
                                                  Lz4BlocksV1& output) const noexcept {
  if (!impl_) {
    return BackingStatus::invalid_configuration;
  }
  const std::shared_ptr<const StoreImage> image = impl_->snapshot(key);
  if (!image) {
    return BackingStatus::chunk_not_found;
  }
  if (image->representation != BackingRepresentation::lz4_blocks) {
    return BackingStatus::invalid_state;
  }
  const BackingStatus validated = validate_image_blocks(*image, *impl_->codec);
  if (validated != BackingStatus::success) {
    return validated;
  }
  try {
    Lz4BlocksV1 result;
    result.valid_bytes = image->valid_bytes;
    result.generation = image->generation;
    result.content_token = image->content_token;
    result.blocks.reserve(image->blocks.size());
    for (const StoreBlock& block : image->blocks) {
      Lz4BlockV1 exported;
      exported.storage = block.storage;
      exported.uncompressed_bytes = block.uncompressed_bytes;
      if (block.payload) {
        exported.payload = *block.payload;
      }
      result.blocks.push_back(std::move(exported));
    }
    output = std::move(result);
    return BackingStatus::success;
  } catch (const std::bad_alloc&) {
    return BackingStatus::allocation_failure;
  }
}

struct CompressionCostModel::Impl {
  double alpha = 0.25;
  std::uint64_t generation = 0;
  std::array<CostPathMetrics, 3> paths{};
  std::uint32_t unfavorable_probes = 0;
  bool suppress_adaptive = false;
};

namespace {

[[nodiscard]] std::size_t path_index(const CompressionPath path) noexcept {
  switch (path) {
  case CompressionPath::raw:
    return 0;
  case CompressionPath::cpu_lz4_gpu_decode:
    return 1;
  case CompressionPath::nvcomp_gpu_codec:
    return 2;
  }
  return 0;
}

void observe_ewma(double& target, const double sample, const double alpha,
                  const bool first) noexcept {
  target = first ? sample : alpha * sample + (1.0 - alpha) * target;
}

[[nodiscard]] double required_margin(const double raw_total_us) noexcept {
  return std::max(50.0, raw_total_us * 0.10);
}

[[nodiscard]] std::uint64_t predicted_bytes(const double ratio,
                                            const std::uint64_t raw_bytes) noexcept {
  if (!std::isfinite(ratio) || ratio <= 0.0 || raw_bytes == 0) {
    return 0;
  }
  const long double value = static_cast<long double>(ratio) * static_cast<long double>(raw_bytes);
  if (value >= static_cast<long double>(std::numeric_limits<std::uint64_t>::max())) {
    return std::numeric_limits<std::uint64_t>::max();
  }
  return static_cast<std::uint64_t>(std::ceil(value));
}

} // namespace

CompressionCostModel::CompressionCostModel(const double ewma_alpha)
    : impl_(std::make_unique<Impl>()) {
  impl_->alpha = ewma_alpha;
}

std::optional<ReuseProbeForecast>
forecast_clean_reuse_probe(const double raw_transfer_us, const double candidate_encode_us,
                           const double candidate_decode_us, const double candidate_transfer_us,
                           const std::uint64_t reuse_count) noexcept {
  if (!finite_nonnegative(raw_transfer_us) || !finite_nonnegative(candidate_encode_us) ||
      !finite_nonnegative(candidate_decode_us) || !finite_nonnegative(candidate_transfer_us)) {
    return std::nullopt;
  }
  const double reuse_cycles = 1.0 + static_cast<double>(reuse_count);
  const double raw_total_us = reuse_cycles * raw_transfer_us;
  const double candidate_total_us =
      candidate_encode_us + reuse_cycles * (candidate_decode_us + candidate_transfer_us);
  if (!finite_nonnegative(raw_total_us) || !finite_nonnegative(candidate_total_us)) {
    return std::nullopt;
  }
  return ReuseProbeForecast{raw_total_us, candidate_total_us};
}

CompressionCostModel::~CompressionCostModel() = default;
CompressionCostModel::CompressionCostModel(CompressionCostModel&&) noexcept = default;
CompressionCostModel& CompressionCostModel::operator=(CompressionCostModel&&) noexcept = default;

bool CompressionCostModel::valid() const noexcept {
  return impl_ && std::isfinite(impl_->alpha) && impl_->alpha > 0.0 && impl_->alpha <= 1.0;
}

std::uint64_t CompressionCostModel::generation() const noexcept {
  return impl_ ? impl_->generation : 0;
}

void CompressionCostModel::reset(const std::uint64_t content_generation) noexcept {
  if (!impl_) {
    return;
  }
  impl_->generation = content_generation;
  impl_->unfavorable_probes = 0;
  impl_->suppress_adaptive = false;
}

bool CompressionCostModel::observe(const std::uint64_t content_generation,
                                   const CostObservation& observation) noexcept {
  if (!valid() || content_generation == 0 || !valid_path(observation.path) ||
      observation.raw_bytes == 0 || observation.stored_bytes == 0 ||
      observation.stored_bytes > observation.raw_bytes ||
      !finite_nonnegative(observation.encode_us) || !finite_nonnegative(observation.decode_us) ||
      !finite_nonnegative(observation.transfer_us) || !finite_nonnegative(observation.staging_us) ||
      !finite_nonnegative(observation.cpu_queue_us) ||
      !finite_nonnegative(observation.sm_opportunity_us) ||
      !std::isfinite(observation.gpu_occupancy) || observation.gpu_occupancy < 0.0 ||
      observation.gpu_occupancy > 1.0 || !std::isfinite(observation.cpu_availability) ||
      observation.cpu_availability < 0.0 || observation.cpu_availability > 1.0) {
    return false;
  }
  if (impl_->generation != content_generation) {
    reset(content_generation);
  }
  CostPathMetrics& metrics = impl_->paths[path_index(observation.path)];
  const bool first = metrics.samples == 0;
  const double raw = static_cast<double>(observation.raw_bytes);
  const double stored = static_cast<double>(observation.stored_bytes);
  observe_ewma(metrics.compression_ratio, stored / raw, impl_->alpha, first);
  observe_ewma(metrics.encode_us_per_raw_byte, observation.encode_us / raw, impl_->alpha, first);
  observe_ewma(metrics.decode_us_per_raw_byte, observation.decode_us / raw, impl_->alpha, first);
  observe_ewma(metrics.transfer_us_per_stored_byte, observation.transfer_us / stored, impl_->alpha,
               first);
  observe_ewma(metrics.staging_us_per_raw_byte, observation.staging_us / raw, impl_->alpha, first);
  observe_ewma(metrics.cpu_queue_us, observation.cpu_queue_us, impl_->alpha, first);
  observe_ewma(metrics.sm_opportunity_us_per_raw_byte, observation.sm_opportunity_us / raw,
               impl_->alpha, first);
  observe_ewma(metrics.gpu_occupancy, observation.gpu_occupancy, impl_->alpha, first);
  observe_ewma(metrics.cpu_availability, observation.cpu_availability, impl_->alpha, first);
  observe_ewma(metrics.reuse_count, static_cast<double>(observation.reuse_count), impl_->alpha,
               first);
  observe_ewma(metrics.dirty_rate, observation.dirty ? 1.0 : 0.0, impl_->alpha, first);
  if (metrics.samples != std::numeric_limits<std::uint32_t>::max()) {
    ++metrics.samples;
  }
  return true;
}

bool CompressionCostModel::record_probe(const std::uint64_t content_generation,
                                        const CompressionPath candidate_path,
                                        const double raw_total_us,
                                        const double candidate_total_us) noexcept {
  if (!valid() || content_generation == 0 || candidate_path == CompressionPath::raw ||
      !valid_path(candidate_path) || !finite_nonnegative(raw_total_us) ||
      !finite_nonnegative(candidate_total_us)) {
    return false;
  }
  if (impl_->generation != content_generation) {
    reset(content_generation);
  }
  const bool favorable = candidate_total_us + required_margin(raw_total_us) <= raw_total_us;
  if (impl_->suppress_adaptive) {
    // `never_compress` is content-generation scoped. Once three consecutive probes establish the
    // mark, a stray later sample must not revive compression for the same immutable bytes.
    return favorable;
  }
  if (favorable) {
    impl_->unfavorable_probes = 0;
  } else if (impl_->unfavorable_probes != std::numeric_limits<std::uint32_t>::max()) {
    ++impl_->unfavorable_probes;
  }
  if (impl_->unfavorable_probes >= 3) {
    impl_->suppress_adaptive = true;
  }
  return favorable;
}

CostPathMetrics CompressionCostModel::metrics(const CompressionPath path) const noexcept {
  return impl_ && valid_path(path) ? impl_->paths[path_index(path)] : CostPathMetrics{};
}

CostEstimate CompressionCostModel::estimate(const CompressionPath path,
                                            const std::uint64_t raw_bytes) const noexcept {
  CostEstimate result;
  result.path = path;
  if (!impl_ || !valid_path(path) || raw_bytes == 0) {
    return result;
  }
  const CostPathMetrics& value = impl_->paths[path_index(path)];
  result.confident = value.samples >= 2;
  result.predicted_stored_bytes = predicted_bytes(value.compression_ratio, raw_bytes);
  const double raw = static_cast<double>(raw_bytes);
  const double stored = static_cast<double>(result.predicted_stored_bytes);
  const double reuse_cycles = 1.0 + std::max(0.0, value.reuse_count);
  const double dirty_cycles = reuse_cycles * std::clamp(value.dirty_rate, 0.0, 1.0);
  const double cpu_availability = std::clamp(value.cpu_availability, 0.05, 1.0);
  const double gpu_contention = 1.0 + std::clamp(value.gpu_occupancy, 0.0, 1.0);
  const double transfer_read = value.transfer_us_per_stored_byte * stored;
  const double transfer_raw = value.transfer_us_per_stored_byte * raw;
  const double staging = value.staging_us_per_raw_byte * raw;

  if (path == CompressionPath::raw) {
    // Every reuse may require another raw H2D, and a dirty reuse adds a raw D2H write-back.
    result.predicted_total_us =
        reuse_cycles * (transfer_read + staging) + dirty_cycles * (transfer_raw + staging);
  } else {
    const bool cpu_encode = path == CompressionPath::cpu_lz4_gpu_decode;
    double encode = value.encode_us_per_raw_byte * raw;
    double decode = value.decode_us_per_raw_byte * raw;
    double cpu_queue = value.cpu_queue_us;
    double sm_opportunity = value.sm_opportunity_us_per_raw_byte * raw;
    if (cpu_encode) {
      // CPU scarcity slows the host encode and its observed queue, while decode still competes for
      // GPU SM time. A 5% floor keeps the prediction finite but strongly penalizes starvation.
      encode /= cpu_availability;
      cpu_queue /= cpu_availability;
      decode *= gpu_contention;
      sm_opportunity *= gpu_contention;
    } else {
      encode *= gpu_contention;
      decode *= gpu_contention;
      sm_opportunity *= gpu_contention;
      cpu_queue = 0.0;
    }

    const double initial_encode = encode + cpu_queue + (cpu_encode ? 0.0 : sm_opportunity);
    const double read_cycle = decode + transfer_read + staging + sm_opportunity;
    // CPU LZ4 must first retrieve dirty bytes raw before a host encode. nvCOMP encodes on device
    // and transfers only the stored representation.
    const double writeback_transfer = cpu_encode ? transfer_raw : transfer_read;
    const double writeback_cycle =
        encode + cpu_queue + writeback_transfer + staging + (cpu_encode ? 0.0 : sm_opportunity);
    result.predicted_total_us =
        initial_encode + reuse_cycles * read_cycle + dirty_cycles * writeback_cycle;
  }
  if (!finite_nonnegative(result.predicted_total_us)) {
    result.confident = false;
    result.predicted_total_us = 0.0;
  }
  return result;
}

CostDecision CompressionCostModel::decide(const CompressionMode mode, const std::uint64_t raw_bytes,
                                          const bool allow_cpu_path,
                                          const bool allow_gpu_path) const noexcept {
  CostDecision result;
  result.raw = estimate(CompressionPath::raw, raw_bytes);
  result.cpu_lz4_gpu_decode = estimate(CompressionPath::cpu_lz4_gpu_decode, raw_bytes);
  result.nvcomp_gpu_codec = estimate(CompressionPath::nvcomp_gpu_codec, raw_bytes);
  result.never_compress = impl_ && impl_->suppress_adaptive;
  result.required_savings_us = required_margin(result.raw.predicted_total_us);
  if (mode == CompressionMode::disabled || raw_bytes == 0 || !result.raw.confident) {
    result.calibration_required = mode != CompressionMode::disabled;
    return result;
  }

  const std::array candidates{result.cpu_lz4_gpu_decode, result.nvcomp_gpu_codec};
  CostEstimate selected = result.raw;
  bool found_confident_candidate = false;
  for (const CostEstimate& candidate : candidates) {
    const bool allowed =
        candidate.path == CompressionPath::cpu_lz4_gpu_decode ? allow_cpu_path : allow_gpu_path;
    if (!allowed || !candidate.confident || candidate.predicted_stored_bytes >= raw_bytes) {
      continue;
    }
    found_confident_candidate = true;
    if (mode == CompressionMode::capacity) {
      if (selected.path == CompressionPath::raw ||
          candidate.predicted_stored_bytes < selected.predicted_stored_bytes ||
          (candidate.predicted_stored_bytes == selected.predicted_stored_bytes &&
           static_cast<int>(candidate.path) < static_cast<int>(selected.path))) {
        selected = candidate;
      }
    } else if (!result.never_compress &&
               candidate.predicted_total_us + result.required_savings_us <=
                   result.raw.predicted_total_us &&
               (selected.path == CompressionPath::raw ||
                candidate.predicted_total_us < selected.predicted_total_us ||
                (candidate.predicted_total_us == selected.predicted_total_us &&
                 static_cast<int>(candidate.path) < static_cast<int>(selected.path)))) {
      selected = candidate;
    }
  }
  result.calibration_required = !found_confident_candidate;
  result.path = selected.path;
  result.margin_satisfied =
      selected.path != CompressionPath::raw &&
      selected.predicted_total_us + result.required_savings_us <= result.raw.predicted_total_us;
  result.capacity_override = mode == CompressionMode::capacity &&
                             selected.path != CompressionPath::raw && !result.margin_satisfied;
  return result;
}

bool CompressionCostModel::never_compress() const noexcept {
  return impl_ && impl_->suppress_adaptive;
}

std::uint32_t CompressionCostModel::unfavorable_probe_count() const noexcept {
  return impl_ ? impl_->unfavorable_probes : 0;
}

} // namespace xvram::residency
