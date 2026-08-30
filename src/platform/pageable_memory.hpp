#pragma once

#include <cstddef>
#include <cstdint>

namespace xvram::platform {

class PageableMemory {
public:
  PageableMemory() = default;
  PageableMemory(const PageableMemory&) = delete;
  PageableMemory& operator=(const PageableMemory&) = delete;
  PageableMemory(PageableMemory&&) = delete;
  PageableMemory& operator=(PageableMemory&&) = delete;
  ~PageableMemory();

  [[nodiscard]] bool allocate(std::uint64_t bytes) noexcept;
  [[nodiscard]] bool release() noexcept;
  [[nodiscard]] void* data() noexcept;
  [[nodiscard]] const void* data() const noexcept;
  [[nodiscard]] std::size_t size() const noexcept;

private:
  void* data_ = nullptr;
  std::size_t bytes_ = 0;
};

} // namespace xvram::platform
