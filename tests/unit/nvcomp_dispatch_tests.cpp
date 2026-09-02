#include "platform/nvcomp/nvcomp_api.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>

namespace {

int failures = 0;
std::uint32_t fake_version = 5300U;

void check(const bool condition, const char* expression, const int line) {
  if (!condition) {
    std::cerr << "line " << line << ": check failed: " << expression << '\n';
    ++failures;
  }
}

#define CHECK(expression) check(static_cast<bool>(expression), #expression, __LINE__)

nvcompStatus_t fake_get_properties(nvcompProperties_t* const properties) {
  properties->version = fake_version;
  properties->cudart_version = 13030U;
  return nvcompSuccess;
}

const char* fake_get_status_string(nvcompStatus_t) {
  return "fake";
}

nvcompStatus_t fake_compress_alignments(nvcompBatchedLZ4CompressOpts_t,
                                        nvcompAlignmentRequirements_t*) {
  return nvcompSuccess;
}

nvcompStatus_t fake_compress_temp(std::size_t, std::size_t,
                                  nvcompBatchedLZ4CompressOpts_t, std::size_t* const bytes,
                                  std::size_t) {
  *bytes = 4096U;
  return nvcompSuccess;
}

nvcompStatus_t fake_compress_max_output(std::size_t bytes,
                                        nvcompBatchedLZ4CompressOpts_t,
                                        std::size_t* const output) {
  *output = bytes + 64U;
  return nvcompSuccess;
}

nvcompStatus_t fake_compress_async(const void* const*, const std::size_t*, std::size_t,
                                   std::size_t, void*, std::size_t, void* const*, std::size_t*,
                                   nvcompBatchedLZ4CompressOpts_t, nvcompStatus_t*,
                                   cudaStream_t) {
  return nvcompSuccess;
}

nvcompStatus_t fake_decompress_alignments(nvcompBatchedLZ4DecompressOpts_t,
                                          nvcompAlignmentRequirements_t*) {
  return nvcompSuccess;
}

nvcompStatus_t fake_decompress_temp(std::size_t, std::size_t,
                                    nvcompBatchedLZ4DecompressOpts_t,
                                    std::size_t* const bytes, std::size_t) {
  *bytes = 8192U;
  return nvcompSuccess;
}

nvcompStatus_t fake_decompress_async(const void* const*, const std::size_t*,
                                     const std::size_t*, std::size_t*, std::size_t, void*,
                                     std::size_t, void* const*, nvcompBatchedLZ4DecompressOpts_t,
                                     nvcompStatus_t*, cudaStream_t) {
  return nvcompSuccess;
}

xvram::nvcomp::NvcompDispatch valid_dispatch() {
  return {
      &fake_get_properties,
      &fake_get_status_string,
      &fake_compress_alignments,
      &fake_compress_temp,
      &fake_compress_max_output,
      &fake_compress_async,
      &fake_decompress_alignments,
      &fake_decompress_temp,
      &fake_decompress_async,
  };
}

void injected_dispatch_tests() {
  fake_version = 5300U;
  xvram::nvcomp::NvcompApi api(valid_dispatch(), "fake-nvcomp");
  CHECK(api.status() == xvram::nvcomp::NvcompLoadStatus::loaded);
  CHECK(api.has_lz4());
  CHECK(api.library_source() == xvram::nvcomp::NvcompLibrarySource::injected);
  CHECK(api.loaded_name() == "fake-nvcomp");
  CHECK(api.properties().version == 5300U);

  auto missing = valid_dispatch();
  missing.lz4_decompress_async = nullptr;
  xvram::nvcomp::NvcompApi incomplete(missing);
  CHECK(incomplete.status() == xvram::nvcomp::NvcompLoadStatus::symbols_missing);
  CHECK(!incomplete.has_lz4());

  fake_version = 6100U;
  xvram::nvcomp::NvcompApi incompatible(valid_dispatch());
  CHECK(incompatible.status() == xvram::nvcomp::NvcompLoadStatus::incompatible_version);
  fake_version = 5300U;

  CHECK(std::string_view{xvram::nvcomp::nvcomp_load_status_name(
            xvram::nvcomp::NvcompLoadStatus::loaded)} == "loaded");
  CHECK(std::string_view{xvram::nvcomp::nvcomp_library_source_name(
            xvram::nvcomp::NvcompLibrarySource::app_local)} == "app_local");
}

} // namespace

int main() {
  injected_dispatch_tests();
  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
    return 1;
  }
  std::cout << "nvCOMP dispatch tests passed\n";
  return 0;
}
