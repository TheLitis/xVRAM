#include "platform/pageable_memory.hpp"

#include <limits>

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#else
#include <sys/mman.h>
#endif

namespace xvram::platform {

PageableMemory::~PageableMemory() {
  (void)release();
}

bool PageableMemory::allocate(const std::uint64_t bytes) noexcept {
  if (data_ != nullptr || bytes == 0 ||
      bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return false;
  }
#ifdef _WIN32
  data_ = VirtualAlloc(nullptr, static_cast<std::size_t>(bytes), MEM_RESERVE | MEM_COMMIT,
                       PAGE_READWRITE);
  if (data_ == nullptr) {
    return false;
  }
#else
  data_ = mmap(nullptr, static_cast<std::size_t>(bytes), PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (data_ == MAP_FAILED) {
    data_ = nullptr;
    return false;
  }
#endif
  bytes_ = static_cast<std::size_t>(bytes);
  return true;
}

bool PageableMemory::release() noexcept {
  if (data_ == nullptr) {
    return true;
  }
#ifdef _WIN32
  const bool released = VirtualFree(data_, 0, MEM_RELEASE) != FALSE;
#else
  const bool released = munmap(data_, bytes_) == 0;
#endif
  if (released) {
    data_ = nullptr;
    bytes_ = 0;
  }
  return released;
}

void* PageableMemory::data() noexcept {
  return data_;
}

const void* PageableMemory::data() const noexcept {
  return data_;
}

std::size_t PageableMemory::size() const noexcept {
  return bytes_;
}

} // namespace xvram::platform
