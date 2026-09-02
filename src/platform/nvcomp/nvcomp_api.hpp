#pragma once

#include "platform/dynamic_library.hpp"

#include <nvcomp/lz4.h>

#include <filesystem>
#include <string>
#include <vector>

namespace xvram::nvcomp {

struct NvcompDispatch {
  decltype(&::nvcompGetProperties) get_properties = nullptr;
  decltype(&::nvcompGetStatusString) get_status_string = nullptr;
  decltype(&::nvcompBatchedLZ4CompressGetRequiredAlignments)
      lz4_compress_get_required_alignments = nullptr;
  decltype(&::nvcompBatchedLZ4CompressGetTempSizeAsync) lz4_compress_get_temp_size = nullptr;
  decltype(&::nvcompBatchedLZ4CompressGetMaxOutputChunkSize)
      lz4_compress_get_max_output_size = nullptr;
  decltype(&::nvcompBatchedLZ4CompressAsync) lz4_compress_async = nullptr;
  decltype(&::nvcompBatchedLZ4DecompressGetRequiredAlignments)
      lz4_decompress_get_required_alignments = nullptr;
  decltype(&::nvcompBatchedLZ4DecompressGetTempSizeAsync) lz4_decompress_get_temp_size = nullptr;
  decltype(&::nvcompBatchedLZ4DecompressAsync) lz4_decompress_async = nullptr;
};

struct NvcompLoadOptions {
  // Empty requests normal system lookup followed by an app-local lookup beside xVRAM.
  // A non-empty path must be absolute.
  std::filesystem::path library;
};

enum class NvcompLoadStatus {
  loaded,
  library_unavailable,
  symbols_missing,
  incompatible_version,
  invalid_path,
};

enum class NvcompLibrarySource { unknown, system, app_local, explicit_path, injected };

struct NvcompLoadResult {
  NvcompLoadStatus status = NvcompLoadStatus::library_unavailable;
  bool attempted_now = false;
};

class NvcompApi {
public:
  NvcompApi() = default;
  explicit NvcompApi(NvcompDispatch dispatch, std::string loaded_name = "injected");

  [[nodiscard]] NvcompLoadResult load(const NvcompLoadOptions& options = {});
  [[nodiscard]] const NvcompDispatch& dispatch() const noexcept;
  [[nodiscard]] NvcompLoadStatus status() const noexcept;
  [[nodiscard]] NvcompLibrarySource library_source() const noexcept;
  [[nodiscard]] const std::string& loaded_name() const noexcept;
  [[nodiscard]] const std::string& error() const noexcept;
  [[nodiscard]] const std::vector<std::string>& missing_symbols() const noexcept;
  [[nodiscard]] const nvcompProperties_t& properties() const noexcept;
  [[nodiscard]] bool has_lz4() const noexcept;

  // Retain the module reference until process exit when a submitted nvCOMP operation cannot be
  // proven complete. This mirrors CUDA/cublas quarantine behavior.
  void abandon() noexcept;

private:
  void resolve_symbols();
  void validate_dispatch();
  void validate_version();

  template <typename Function> void require(Function& output, const char* name) {
    output = library_.symbol<Function>(name);
    if (output == nullptr) {
      missing_symbols_.emplace_back(name);
    }
  }

  platform::DynamicLibrary library_;
  NvcompDispatch dispatch_;
  NvcompLoadStatus status_ = NvcompLoadStatus::library_unavailable;
  NvcompLibrarySource source_ = NvcompLibrarySource::unknown;
  bool load_attempted_ = false;
  std::string loaded_name_;
  std::string error_;
  std::vector<std::string> missing_symbols_;
  nvcompProperties_t properties_{};
};

[[nodiscard]] const char* nvcomp_load_status_name(NvcompLoadStatus status) noexcept;
[[nodiscard]] const char* nvcomp_library_source_name(NvcompLibrarySource source) noexcept;

} // namespace xvram::nvcomp
