#pragma once

#include "platform/cublas/cublas_abi.hpp"
#include "platform/dynamic_library.hpp"

#include <compare>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace xvram::cublas {

struct CublasDispatch {
  abi::Create create = nullptr;
  abi::Destroy destroy = nullptr;
  abi::SetStream set_stream = nullptr;
  abi::GetVersion get_version = nullptr;
  abi::SetMathMode set_math_mode = nullptr;
  abi::SetWorkspace set_workspace = nullptr;
  abi::GemmEx gemm_ex = nullptr;
  abi::Dgemm dgemm = nullptr;
  abi::GetStatusString get_status_string = nullptr;

  abi::LtCreate lt_create = nullptr;
  abi::LtDestroy lt_destroy = nullptr;
  abi::LtGetVersion lt_get_version = nullptr;
  abi::LtMatmulDescCreate lt_matmul_desc_create = nullptr;
  abi::LtMatmulDescDestroy lt_matmul_desc_destroy = nullptr;
  abi::LtMatmulDescSetAttribute lt_matmul_desc_set_attribute = nullptr;
  abi::LtMatrixLayoutCreate lt_matrix_layout_create = nullptr;
  abi::LtMatrixLayoutDestroy lt_matrix_layout_destroy = nullptr;
  abi::LtMatrixLayoutSetAttribute lt_matrix_layout_set_attribute = nullptr;
  abi::LtMatmulPreferenceCreate lt_matmul_preference_create = nullptr;
  abi::LtMatmulPreferenceDestroy lt_matmul_preference_destroy = nullptr;
  abi::LtMatmulPreferenceSetAttribute lt_matmul_preference_set_attribute = nullptr;
  abi::LtMatmulAlgoGetHeuristic lt_matmul_algo_get_heuristic = nullptr;
  abi::LtMatmulAlgoGetHeuristicForStream lt_matmul_algo_get_heuristic_for_stream = nullptr;
  abi::LtMatmulAlgoCheckForStream lt_matmul_algo_check_for_stream = nullptr;
  abi::LtMatmul lt_matmul = nullptr;
  abi::GetStatusString lt_get_status_string = nullptr;
};

struct CublasLoadOptions {
  // Explicit paths are the app-local redistributable fallback. Both must be absolute and supplied
  // together. Empty paths request normal system lookup.
  std::filesystem::path core_library;
  std::filesystem::path lt_library;
};

enum class CublasLoadStatus {
  loaded,
  core_library_unavailable,
  lt_library_unavailable,
  baseline_symbols_missing,
  invalid_paths,
};

enum class CublasLtLoadStatus {
  loaded,
  library_unavailable,
  symbols_missing,
};

enum class CublasLibrarySource {
  unknown,
  system,
  app_local,
  explicit_path,
};

struct CublasLoadResult {
  CublasLoadStatus status = CublasLoadStatus::core_library_unavailable;
  bool attempted_now = false;
  CublasLtLoadStatus lt_status = CublasLtLoadStatus::library_unavailable;
};

class CublasApi {
public:
  CublasApi() = default;
  explicit CublasApi(CublasDispatch dispatch, std::string core_name = "injected",
                     std::string lt_name = "injected");

  [[nodiscard]] CublasLoadResult load(const CublasLoadOptions& options = {});
  [[nodiscard]] const CublasDispatch& dispatch() const noexcept;
  [[nodiscard]] CublasLoadStatus status() const noexcept;
  [[nodiscard]] const std::string& core_loaded_name() const noexcept;
  [[nodiscard]] const std::string& lt_loaded_name() const noexcept;
  [[nodiscard]] const std::string& error() const noexcept;
  [[nodiscard]] const std::string& lt_error() const noexcept;
  [[nodiscard]] const std::vector<std::string>& missing_symbols() const noexcept;
  [[nodiscard]] const std::vector<std::string>& missing_lt_symbols() const noexcept;
  [[nodiscard]] bool has_complete_baseline() const noexcept;
  [[nodiscard]] bool has_lt() const noexcept;
  [[nodiscard]] CublasLtLoadStatus lt_status() const noexcept;
  [[nodiscard]] CublasLibrarySource library_source() const noexcept;

  // set_workspace and get_status_string are optional across supported redistributable versions.
  [[nodiscard]] bool has_optional_workspace_api() const noexcept;

private:
  void resolve_symbols();
  void validate_dispatch();

  template <typename Function>
  void require(platform::DynamicLibrary& library, Function& output, const char* name) {
    output = library.symbol<Function>(name);
    if (output == nullptr) {
      missing_symbols_.emplace_back(name);
    }
  }

  platform::DynamicLibrary core_library_;
  platform::DynamicLibrary lt_library_;
  CublasDispatch dispatch_;
  CublasLoadStatus status_ = CublasLoadStatus::core_library_unavailable;
  CublasLtLoadStatus lt_status_ = CublasLtLoadStatus::library_unavailable;
  CublasLibrarySource library_source_ = CublasLibrarySource::unknown;
  bool load_attempted_ = false;
  std::string core_name_;
  std::string lt_name_;
  std::string error_;
  std::string lt_error_;
  std::vector<std::string> missing_symbols_;
  std::vector<std::string> missing_lt_symbols_;
};

[[nodiscard]] const char* cublas_load_status_name(CublasLoadStatus status) noexcept;
[[nodiscard]] const char* cublas_lt_load_status_name(CublasLtLoadStatus status) noexcept;
[[nodiscard]] const char* cublas_library_source_name(CublasLibrarySource source) noexcept;

struct LtMatrixSignature {
  abi::DataType data_type = abi::data_fp32;
  abi::LtOrder order = abi::lt_order_column_major;
  std::uint64_t rows = 0;
  std::uint64_t columns = 0;
  std::int64_t leading_dimension = 0;
  std::uint32_t minimum_alignment_bytes = 0;

  [[nodiscard]] auto operator<=>(const LtMatrixSignature&) const noexcept = default;
};

// This is the complete algorithm-cache key for ordinary, non-batched Phase 3 GEMM. A cache is
// owned by one Lt handle/context, so device identity and library version are intentionally scoped
// by the executor rather than repeated in every key.
struct LtMatmulSignature {
  abi::ComputeType compute_type = abi::compute_fp32;
  abi::DataType scale_type = abi::data_fp32;
  abi::Operation operation_a = abi::operation_none;
  abi::Operation operation_b = abi::operation_none;
  LtMatrixSignature a;
  LtMatrixSignature b;
  LtMatrixSignature c;
  LtMatrixSignature d;
  std::uint64_t workspace_limit_bytes = 0;

  [[nodiscard]] auto operator<=>(const LtMatmulSignature&) const noexcept = default;
};

struct LtMatmulRequest {
  LtMatmulSignature signature;
  const void* alpha = nullptr;
  const void* a = nullptr;
  const void* b = nullptr;
  const void* beta = nullptr;
  const void* c = nullptr;
  void* d = nullptr;
  void* workspace = nullptr;
  std::uint64_t workspace_bytes = 0;
  abi::Stream stream = nullptr;
};

struct CachedLtAlgorithm {
  abi::LtMatmulAlgorithm algorithm{};
  std::uint64_t workspace_bytes = 0;
  float waves_count = 0.0F;
};

struct LtAlgorithmCacheStatistics {
  std::uint64_t hits = 0;
  std::uint64_t misses = 0;
  std::uint64_t insertions = 0;
  std::uint64_t evictions = 0;
};

class LtAlgorithmCache {
public:
  explicit LtAlgorithmCache(std::size_t capacity = 256);

  [[nodiscard]] bool find(const LtMatmulSignature& signature, CachedLtAlgorithm& output);
  void insert(const LtMatmulSignature& signature, const CachedLtAlgorithm& algorithm);
  [[nodiscard]] bool erase(const LtMatmulSignature& signature) noexcept;
  void clear() noexcept;

  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] std::size_t capacity() const noexcept;
  [[nodiscard]] const LtAlgorithmCacheStatistics& statistics() const noexcept;

private:
  struct Entry {
    LtMatmulSignature signature;
    CachedLtAlgorithm algorithm;
    std::uint64_t last_use = 0;
  };

  std::vector<Entry> entries_;
  std::size_t capacity_ = 0;
  std::uint64_t use_clock_ = 0;
  LtAlgorithmCacheStatistics statistics_;
};

enum class LtExecutionCode {
  success,
  fallback_required,
  invalid_request,
  not_initialized,
  cublas_failure,
  cleanup_failure,
};

struct LtExecutionResult {
  LtExecutionCode code = LtExecutionCode::not_initialized;
  abi::Status status = abi::not_initialized;
  std::string_view operation;
  bool algorithm_cache_hit = false;
  std::uint64_t algorithm_workspace_bytes = 0;

  [[nodiscard]] explicit operator bool() const noexcept {
    return code == LtExecutionCode::success;
  }
};

struct CoreGemmRequest {
  abi::Operation operation_a = abi::operation_none;
  abi::Operation operation_b = abi::operation_none;
  int m = 0;
  int n = 0;
  int k = 0;
  const void* a = nullptr;
  abi::DataType a_type = abi::data_fp32;
  int lda = 0;
  const void* b = nullptr;
  abi::DataType b_type = abi::data_fp32;
  int ldb = 0;
  void* c = nullptr;
  abi::DataType c_type = abi::data_fp32;
  int ldc = 0;
  abi::ComputeType compute_type = abi::compute_fp32;
  abi::MathMode math_mode = abi::default_math;
  double alpha = 1.0;
  double beta = 0.0;
  void* workspace = nullptr;
  std::uint64_t workspace_bytes = 0;
  abi::Stream stream = nullptr;
};

struct CoreExecutionResult {
  abi::Status status = abi::not_initialized;
  std::string_view operation;

  [[nodiscard]] explicit operator bool() const noexcept {
    return status == abi::success;
  }
};

enum class PreferredGemmPath {
  none,
  cublas_lt,
  cublas_core,
};

struct PreferredGemmResult {
  PreferredGemmPath path = PreferredGemmPath::none;
  LtExecutionResult lt;
  CoreExecutionResult core;

  [[nodiscard]] explicit operator bool() const noexcept {
    return path == PreferredGemmPath::cublas_lt
               ? static_cast<bool>(lt)
               : (path == PreferredGemmPath::cublas_core && static_cast<bool>(core));
  }
};

[[nodiscard]] CoreExecutionResult execute_core_gemm(const CublasDispatch& dispatch,
                                                    abi::Handle handle,
                                                    const CoreGemmRequest& request) noexcept;

// Worker-thread-confined owner of one cuBLASLt handle and its algorithm cache. The handle is
// created on the caller's current CUDA context and must be closed before that context is released.
class LtMatmulExecutor {
public:
  explicit LtMatmulExecutor(const CublasDispatch& dispatch, std::size_t cache_capacity = 256);
  ~LtMatmulExecutor();

  LtMatmulExecutor(const LtMatmulExecutor&) = delete;
  LtMatmulExecutor& operator=(const LtMatmulExecutor&) = delete;
  LtMatmulExecutor(LtMatmulExecutor&&) = delete;
  LtMatmulExecutor& operator=(LtMatmulExecutor&&) = delete;

  [[nodiscard]] abi::Status initialize() noexcept;
  [[nodiscard]] abi::Status close() noexcept;
  [[nodiscard]] bool ready() const noexcept;
  [[nodiscard]] abi::LtHandle handle() const noexcept;
  [[nodiscard]] std::size_t library_version() const noexcept;
  [[nodiscard]] LtAlgorithmCache& cache() noexcept;
  [[nodiscard]] const LtAlgorithmCache& cache() const noexcept;

  [[nodiscard]] LtExecutionResult execute(const LtMatmulRequest& request);
  [[nodiscard]] PreferredGemmResult
  execute_preferred(const LtMatmulRequest& request, abi::Handle core_handle,
                    const std::optional<CoreGemmRequest>& core_fallback = std::nullopt);

private:
  const CublasDispatch* dispatch_ = nullptr;
  abi::LtHandle handle_ = nullptr;
  std::size_t library_version_ = 0;
  LtAlgorithmCache cache_;
};

[[nodiscard]] std::string_view lt_execution_code_name(LtExecutionCode code) noexcept;
[[nodiscard]] std::string_view preferred_gemm_path_name(PreferredGemmPath path) noexcept;

} // namespace xvram::cublas
