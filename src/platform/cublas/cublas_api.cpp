#include "platform/cublas/cublas_api.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <tuple>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

namespace xvram::cublas {
namespace {

[[nodiscard]] bool exactly_one_path(const CublasLoadOptions& options) noexcept {
  return options.core_library.empty() != options.lt_library.empty();
}

[[nodiscard]] std::filesystem::path module_directory() {
#ifdef _WIN32
  HMODULE module = nullptr;
  if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                             GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                         reinterpret_cast<LPCWSTR>(&module_directory), &module) == FALSE) {
    return {};
  }
  std::array<wchar_t, 32'768> path{};
  const DWORD length = GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
  return length == 0 || length >= path.size()
             ? std::filesystem::path{}
             : std::filesystem::path(std::wstring(path.data(), length)).parent_path();
#else
  Dl_info info{};
  if (dladdr(reinterpret_cast<const void*>(&module_directory), &info) == 0 ||
      info.dli_fname == nullptr) {
    return {};
  }
  return std::filesystem::path(info.dli_fname).parent_path();
#endif
}

[[nodiscard]] bool open_app_local(platform::DynamicLibrary& core, platform::DynamicLibrary& lt) {
  const std::filesystem::path directory = module_directory();
  if (directory.empty()) {
    return false;
  }
#ifdef _WIN32
  constexpr std::array core_names{L"cublas64_13.dll", L"cublas64_12.dll"};
  constexpr std::array lt_names{L"cublasLt64_13.dll", L"cublasLt64_12.dll"};
#else
  constexpr std::array core_names{"libcublas.so.13", "libcublas.so.12"};
  constexpr std::array lt_names{"libcublasLt.so.13", "libcublasLt.so.12"};
#endif
  for (std::size_t index = 0; index < core_names.size(); ++index) {
    const std::filesystem::path core_path = directory / core_names[index];
    const std::filesystem::path lt_path = directory / lt_names[index];
    if (!std::filesystem::is_regular_file(core_path) || !core.open_absolute(core_path)) {
      core.close();
      continue;
    }

    // cuBLASLt is an optional acceleration path. Keep a usable app-local core library even when
    // the matching Lt redistributable is absent or cannot be loaded.
    if (std::filesystem::is_regular_file(lt_path) && !lt.open_absolute(lt_path)) {
      lt.close();
    }
    return true;
  }
  return false;
}

} // namespace

CublasApi::CublasApi(CublasDispatch dispatch, std::string core_name, std::string lt_name)
    : dispatch_(dispatch), load_attempted_(true), core_name_(std::move(core_name)),
      lt_name_(std::move(lt_name)) {
  validate_dispatch();
}

CublasLoadResult CublasApi::load(const CublasLoadOptions& options) {
  if (load_attempted_) {
    return {status_, false, lt_status_};
  }
  load_attempted_ = true;

  if (exactly_one_path(options) ||
      (!options.core_library.empty() &&
       (!options.core_library.is_absolute() || !options.lt_library.is_absolute()))) {
    status_ = CublasLoadStatus::invalid_paths;
    error_ = "cuBLAS and cuBLASLt paths must both be absolute when explicitly supplied";
    return {status_, true, lt_status_};
  }

  const bool explicit_paths = !options.core_library.empty();
  const bool core_opened = explicit_paths
                               ? core_library_.open_absolute(options.core_library)
#ifdef _WIN32
                               : core_library_.open_system({"cublas64_13.dll", "cublas64_12.dll"});
#else
                               : core_library_.open_system(
                                     {"libcublas.so.13", "libcublas.so.12", "libcublas.so"});
#endif
  if (core_opened) {
    library_source_ =
        explicit_paths ? CublasLibrarySource::explicit_path : CublasLibrarySource::system;
  }
  if (!core_opened) {
    if (explicit_paths || !open_app_local(core_library_, lt_library_)) {
      status_ = CublasLoadStatus::core_library_unavailable;
      error_ = core_library_.error();
      return {status_, true, lt_status_};
    }
    library_source_ = CublasLibrarySource::app_local;
  }

  bool lt_opened =
      lt_library_.is_open() ||
      (explicit_paths ? lt_library_.open_absolute(options.lt_library)
#ifdef _WIN32
                      : lt_library_.open_system({"cublasLt64_13.dll", "cublasLt64_12.dll"}));
#else
                      : lt_library_.open_system(
                            {"libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so"}));
#endif
  if (!lt_opened) {
    lt_status_ = CublasLtLoadStatus::library_unavailable;
    lt_error_ = lt_library_.error();
  }

  core_name_ = core_library_.loaded_name();
  if (lt_opened) {
    lt_name_ = lt_library_.loaded_name();
  }
  resolve_symbols();
  validate_dispatch();
  return {status_, true, lt_status_};
}

const CublasDispatch& CublasApi::dispatch() const noexcept {
  return dispatch_;
}

CublasLoadStatus CublasApi::status() const noexcept {
  return status_;
}

const std::string& CublasApi::core_loaded_name() const noexcept {
  return core_name_;
}

const std::string& CublasApi::lt_loaded_name() const noexcept {
  return lt_name_;
}

const std::string& CublasApi::error() const noexcept {
  return error_;
}

const std::string& CublasApi::lt_error() const noexcept {
  return lt_error_;
}

const std::vector<std::string>& CublasApi::missing_symbols() const noexcept {
  return missing_symbols_;
}

const std::vector<std::string>& CublasApi::missing_lt_symbols() const noexcept {
  return missing_lt_symbols_;
}

bool CublasApi::has_complete_baseline() const noexcept {
  return status_ == CublasLoadStatus::loaded;
}

bool CublasApi::has_lt() const noexcept {
  return lt_status_ == CublasLtLoadStatus::loaded;
}

CublasLtLoadStatus CublasApi::lt_status() const noexcept {
  return lt_status_;
}

CublasLibrarySource CublasApi::library_source() const noexcept {
  return library_source_;
}

bool CublasApi::has_optional_workspace_api() const noexcept {
  return dispatch_.set_workspace != nullptr;
}

void CublasApi::resolve_symbols() {
  missing_symbols_.clear();
  require(core_library_, dispatch_.create, "cublasCreate_v2");
  require(core_library_, dispatch_.destroy, "cublasDestroy_v2");
  require(core_library_, dispatch_.set_stream, "cublasSetStream_v2");
  require(core_library_, dispatch_.get_version, "cublasGetVersion_v2");
  require(core_library_, dispatch_.set_math_mode, "cublasSetMathMode");
  require(core_library_, dispatch_.gemm_ex, "cublasGemmEx");
  require(core_library_, dispatch_.dgemm, "cublasDgemm_v2");
  dispatch_.set_workspace = core_library_.symbol<abi::SetWorkspace>("cublasSetWorkspace_v2");
  dispatch_.get_status_string = core_library_.symbol<abi::GetStatusString>("cublasGetStatusString");

  require(lt_library_, dispatch_.lt_create, "cublasLtCreate");
  require(lt_library_, dispatch_.lt_destroy, "cublasLtDestroy");
  require(lt_library_, dispatch_.lt_get_version, "cublasLtGetVersion");
  require(lt_library_, dispatch_.lt_matmul_desc_create, "cublasLtMatmulDescCreate");
  require(lt_library_, dispatch_.lt_matmul_desc_destroy, "cublasLtMatmulDescDestroy");
  require(lt_library_, dispatch_.lt_matmul_desc_set_attribute, "cublasLtMatmulDescSetAttribute");
  require(lt_library_, dispatch_.lt_matrix_layout_create, "cublasLtMatrixLayoutCreate");
  require(lt_library_, dispatch_.lt_matrix_layout_destroy, "cublasLtMatrixLayoutDestroy");
  require(lt_library_, dispatch_.lt_matrix_layout_set_attribute,
          "cublasLtMatrixLayoutSetAttribute");
  require(lt_library_, dispatch_.lt_matmul_preference_create, "cublasLtMatmulPreferenceCreate");
  require(lt_library_, dispatch_.lt_matmul_preference_destroy, "cublasLtMatmulPreferenceDestroy");
  require(lt_library_, dispatch_.lt_matmul_preference_set_attribute,
          "cublasLtMatmulPreferenceSetAttribute");
  require(lt_library_, dispatch_.lt_matmul_algo_get_heuristic, "cublasLtMatmulAlgoGetHeuristic");
  dispatch_.lt_matmul_algo_get_heuristic_for_stream =
      lt_library_.symbol<abi::LtMatmulAlgoGetHeuristicForStream>(
          "cublasLtMatmulAlgoGetHeuristicForStream");
  dispatch_.lt_matmul_algo_check_for_stream =
      lt_library_.symbol<abi::LtMatmulAlgoCheckForStream>("cublasLtMatmulAlgoCheckForStream");
  require(lt_library_, dispatch_.lt_matmul, "cublasLtMatmul");
  dispatch_.lt_get_status_string =
      lt_library_.symbol<abi::GetStatusString>("cublasLtGetStatusString");
}

void CublasApi::validate_dispatch() {
  missing_symbols_.clear();
  missing_lt_symbols_.clear();
  if (dispatch_.create == nullptr) {
    missing_symbols_.emplace_back("cublasCreate_v2");
  }
  if (dispatch_.destroy == nullptr) {
    missing_symbols_.emplace_back("cublasDestroy_v2");
  }
  if (dispatch_.set_stream == nullptr) {
    missing_symbols_.emplace_back("cublasSetStream_v2");
  }
  if (dispatch_.get_version == nullptr) {
    missing_symbols_.emplace_back("cublasGetVersion_v2");
  }
  if (dispatch_.set_math_mode == nullptr) {
    missing_symbols_.emplace_back("cublasSetMathMode");
  }
  if (dispatch_.gemm_ex == nullptr) {
    missing_symbols_.emplace_back("cublasGemmEx");
  }
  if (dispatch_.dgemm == nullptr) {
    missing_symbols_.emplace_back("cublasDgemm_v2");
  }

  if (dispatch_.lt_create == nullptr) {
    missing_lt_symbols_.emplace_back("cublasLtCreate");
  }
  if (dispatch_.lt_destroy == nullptr) {
    missing_lt_symbols_.emplace_back("cublasLtDestroy");
  }
  if (dispatch_.lt_get_version == nullptr) {
    missing_lt_symbols_.emplace_back("cublasLtGetVersion");
  }
  if (dispatch_.lt_matmul_desc_create == nullptr) {
    missing_lt_symbols_.emplace_back("cublasLtMatmulDescCreate");
  }
  if (dispatch_.lt_matmul_desc_destroy == nullptr) {
    missing_lt_symbols_.emplace_back("cublasLtMatmulDescDestroy");
  }
  if (dispatch_.lt_matmul_desc_set_attribute == nullptr) {
    missing_lt_symbols_.emplace_back("cublasLtMatmulDescSetAttribute");
  }
  if (dispatch_.lt_matrix_layout_create == nullptr) {
    missing_lt_symbols_.emplace_back("cublasLtMatrixLayoutCreate");
  }
  if (dispatch_.lt_matrix_layout_destroy == nullptr) {
    missing_lt_symbols_.emplace_back("cublasLtMatrixLayoutDestroy");
  }
  if (dispatch_.lt_matrix_layout_set_attribute == nullptr) {
    missing_lt_symbols_.emplace_back("cublasLtMatrixLayoutSetAttribute");
  }
  if (dispatch_.lt_matmul_preference_create == nullptr) {
    missing_lt_symbols_.emplace_back("cublasLtMatmulPreferenceCreate");
  }
  if (dispatch_.lt_matmul_preference_destroy == nullptr) {
    missing_lt_symbols_.emplace_back("cublasLtMatmulPreferenceDestroy");
  }
  if (dispatch_.lt_matmul_preference_set_attribute == nullptr) {
    missing_lt_symbols_.emplace_back("cublasLtMatmulPreferenceSetAttribute");
  }
  if (dispatch_.lt_matmul_algo_get_heuristic == nullptr &&
      dispatch_.lt_matmul_algo_get_heuristic_for_stream == nullptr) {
    missing_lt_symbols_.emplace_back("cublasLtMatmulAlgoGetHeuristic");
  }
  if (dispatch_.lt_matmul == nullptr) {
    missing_lt_symbols_.emplace_back("cublasLtMatmul");
  }

  if (missing_symbols_.empty()) {
    // The ordinary cuBLAS baseline is independently usable when cuBLASLt is not installed or is
    // from an incompatible redistributable. This is the compatibility fallback contract.
    status_ = CublasLoadStatus::loaded;
    error_.clear();
  } else {
    status_ = CublasLoadStatus::baseline_symbols_missing;
    error_ = "Required core cuBLAS symbols are missing";
  }

  if (missing_lt_symbols_.empty()) {
    lt_status_ = CublasLtLoadStatus::loaded;
    lt_error_.clear();
    return;
  }

  const bool any_lt_symbol = dispatch_.lt_create != nullptr || dispatch_.lt_destroy != nullptr ||
                             dispatch_.lt_get_version != nullptr ||
                             dispatch_.lt_matmul_desc_create != nullptr ||
                             dispatch_.lt_matmul_desc_destroy != nullptr ||
                             dispatch_.lt_matmul_desc_set_attribute != nullptr ||
                             dispatch_.lt_matrix_layout_create != nullptr ||
                             dispatch_.lt_matrix_layout_destroy != nullptr ||
                             dispatch_.lt_matrix_layout_set_attribute != nullptr ||
                             dispatch_.lt_matmul_preference_create != nullptr ||
                             dispatch_.lt_matmul_preference_destroy != nullptr ||
                             dispatch_.lt_matmul_preference_set_attribute != nullptr ||
                             dispatch_.lt_matmul_algo_get_heuristic != nullptr ||
                             dispatch_.lt_matmul_algo_get_heuristic_for_stream != nullptr ||
                             dispatch_.lt_matmul != nullptr;
  if (lt_library_.is_open() || any_lt_symbol) {
    lt_status_ = CublasLtLoadStatus::symbols_missing;
    lt_error_ = "Required cuBLASLt symbols are missing";
  } else {
    lt_status_ = CublasLtLoadStatus::library_unavailable;
    if (lt_error_.empty()) {
      lt_error_ = "cuBLASLt library is unavailable";
    }
  }
}

const char* cublas_load_status_name(const CublasLoadStatus status) noexcept {
  switch (status) {
  case CublasLoadStatus::loaded:
    return "loaded";
  case CublasLoadStatus::core_library_unavailable:
    return "core_library_unavailable";
  case CublasLoadStatus::lt_library_unavailable:
    return "lt_library_unavailable";
  case CublasLoadStatus::baseline_symbols_missing:
    return "baseline_symbols_missing";
  case CublasLoadStatus::invalid_paths:
    return "invalid_paths";
  }
  return "invalid";
}

const char* cublas_lt_load_status_name(const CublasLtLoadStatus status) noexcept {
  switch (status) {
  case CublasLtLoadStatus::loaded:
    return "loaded";
  case CublasLtLoadStatus::library_unavailable:
    return "library_unavailable";
  case CublasLtLoadStatus::symbols_missing:
    return "symbols_missing";
  }
  return "invalid";
}

const char* cublas_library_source_name(const CublasLibrarySource source) noexcept {
  switch (source) {
  case CublasLibrarySource::unknown:
    return "unknown";
  case CublasLibrarySource::system:
    return "system";
  case CublasLibrarySource::app_local:
    return "app_local";
  case CublasLibrarySource::explicit_path:
    return "explicit";
  }
  return "invalid";
}

LtAlgorithmCache::LtAlgorithmCache(const std::size_t capacity) : capacity_(capacity) {
  entries_.reserve(capacity_);
}

bool LtAlgorithmCache::find(const LtMatmulSignature& signature, CachedLtAlgorithm& output) {
  const auto found = std::find_if(entries_.begin(), entries_.end(),
                                  [&](const Entry& entry) { return entry.signature == signature; });
  if (found == entries_.end()) {
    ++statistics_.misses;
    return false;
  }
  ++statistics_.hits;
  found->last_use = ++use_clock_;
  output = found->algorithm;
  return true;
}

void LtAlgorithmCache::insert(const LtMatmulSignature& signature,
                              const CachedLtAlgorithm& algorithm) {
  if (capacity_ == 0) {
    return;
  }
  const auto found = std::find_if(entries_.begin(), entries_.end(),
                                  [&](const Entry& entry) { return entry.signature == signature; });
  if (found != entries_.end()) {
    found->algorithm = algorithm;
    found->last_use = ++use_clock_;
    return;
  }
  if (entries_.size() == capacity_) {
    const auto victim = std::min_element(
        entries_.begin(), entries_.end(),
        [](const Entry& left, const Entry& right) { return left.last_use < right.last_use; });
    entries_.erase(victim);
    ++statistics_.evictions;
  }
  entries_.push_back(Entry{signature, algorithm, ++use_clock_});
  ++statistics_.insertions;
}

bool LtAlgorithmCache::erase(const LtMatmulSignature& signature) noexcept {
  const auto found = std::find_if(entries_.begin(), entries_.end(),
                                  [&](const Entry& entry) { return entry.signature == signature; });
  if (found == entries_.end()) {
    return false;
  }
  entries_.erase(found);
  return true;
}

void LtAlgorithmCache::clear() noexcept {
  entries_.clear();
  use_clock_ = 0;
}

std::size_t LtAlgorithmCache::size() const noexcept {
  return entries_.size();
}

std::size_t LtAlgorithmCache::capacity() const noexcept {
  return capacity_;
}

const LtAlgorithmCacheStatistics& LtAlgorithmCache::statistics() const noexcept {
  return statistics_;
}

namespace {

[[nodiscard]] bool is_operation_valid(const abi::Operation operation) noexcept {
  return operation == abi::operation_none || operation == abi::operation_transpose;
}

[[nodiscard]] bool is_data_type_valid(const abi::DataType type) noexcept {
  return type == abi::data_fp16 || type == abi::data_bf16 || type == abi::data_fp32 ||
         type == abi::data_fp64;
}

[[nodiscard]] bool is_order_valid(const abi::LtOrder order) noexcept {
  return order == abi::lt_order_column_major || order == abi::lt_order_row_major;
}

[[nodiscard]] bool is_power_of_two(const std::uint32_t value) noexcept {
  return value != 0 && (value & (value - 1U)) == 0;
}

[[nodiscard]] std::uint32_t pointer_alignment(const void* pointer) noexcept {
  const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(pointer);
  if (address == 0) {
    return 0;
  }
  std::uint32_t alignment = 1;
  while (alignment < 256U && (address % (static_cast<std::uintptr_t>(alignment) * 2U)) == 0) {
    alignment *= 2U;
  }
  return alignment;
}

[[nodiscard]] bool normalize_matrix(LtMatrixSignature& matrix, const void* pointer) noexcept {
  if (pointer == nullptr || !is_data_type_valid(matrix.data_type) ||
      !is_order_valid(matrix.order) || matrix.rows == 0 || matrix.columns == 0 ||
      matrix.leading_dimension <= 0) {
    return false;
  }
  const std::uint64_t required_leading =
      matrix.order == abi::lt_order_row_major ? matrix.columns : matrix.rows;
  if (static_cast<std::uint64_t>(matrix.leading_dimension) < required_leading) {
    return false;
  }
  const std::uint32_t actual_alignment = pointer_alignment(pointer);
  if (matrix.minimum_alignment_bytes == 0) {
    matrix.minimum_alignment_bytes = actual_alignment;
  }
  return is_power_of_two(matrix.minimum_alignment_bytes) &&
         matrix.minimum_alignment_bytes <= 256U &&
         matrix.minimum_alignment_bytes <= actual_alignment;
}

[[nodiscard]] std::pair<std::uint64_t, std::uint64_t>
operated_shape(const LtMatrixSignature& matrix, const abi::Operation operation) noexcept {
  return operation == abi::operation_none ? std::pair{matrix.rows, matrix.columns}
                                          : std::pair{matrix.columns, matrix.rows};
}

[[nodiscard]] bool is_type_combination_valid(const LtMatmulSignature& signature) noexcept {
  if (signature.a.data_type != signature.b.data_type ||
      signature.c.data_type != signature.d.data_type) {
    return false;
  }
  if (signature.a.data_type == abi::data_fp16 || signature.a.data_type == abi::data_bf16) {
    return (signature.compute_type == abi::compute_fp32 ||
            signature.compute_type == abi::compute_fp32_pedantic) &&
           signature.scale_type == abi::data_fp32 &&
           (signature.c.data_type == signature.a.data_type ||
            signature.c.data_type == abi::data_fp32);
  }
  if (signature.a.data_type == abi::data_fp32) {
    return (signature.compute_type == abi::compute_fp32 ||
            signature.compute_type == abi::compute_fp32_pedantic ||
            signature.compute_type == abi::compute_fast_tf32) &&
           signature.scale_type == abi::data_fp32 && signature.c.data_type == abi::data_fp32;
  }
  return signature.a.data_type == abi::data_fp64 && signature.compute_type == abi::compute_fp64 &&
         signature.scale_type == abi::data_fp64 && signature.c.data_type == abi::data_fp64;
}

[[nodiscard]] bool normalize_request(const LtMatmulRequest& request,
                                     LtMatmulSignature& signature) noexcept {
  signature = request.signature;
  if (!is_operation_valid(signature.operation_a) || !is_operation_valid(signature.operation_b) ||
      request.alpha == nullptr || request.beta == nullptr ||
      !normalize_matrix(signature.a, request.a) || !normalize_matrix(signature.b, request.b) ||
      !normalize_matrix(signature.c, request.c) || !normalize_matrix(signature.d, request.d) ||
      !is_type_combination_valid(signature) ||
      signature.workspace_limit_bytes > request.workspace_bytes ||
      request.workspace_bytes > std::numeric_limits<std::size_t>::max()) {
    return false;
  }
  if (request.workspace_bytes != 0 &&
      (request.workspace == nullptr || pointer_alignment(request.workspace) < 256U)) {
    return false;
  }
  const auto [a_rows, a_columns] = operated_shape(signature.a, signature.operation_a);
  const auto [b_rows, b_columns] = operated_shape(signature.b, signature.operation_b);
  return a_rows == signature.c.rows && a_rows == signature.d.rows &&
         b_columns == signature.c.columns && b_columns == signature.d.columns &&
         a_columns == b_rows && signature.c.rows == signature.d.rows &&
         signature.c.columns == signature.d.columns;
}

struct LtDescriptorSet {
  abi::LtMatmulDesc operation = nullptr;
  abi::LtMatrixLayout a = nullptr;
  abi::LtMatrixLayout b = nullptr;
  abi::LtMatrixLayout c = nullptr;
  abi::LtMatrixLayout d = nullptr;
  abi::LtMatmulPreference preference = nullptr;
};

[[nodiscard]] abi::Status destroy_descriptors(const CublasDispatch& dispatch,
                                              LtDescriptorSet& descriptors) noexcept {
  abi::Status first_error = abi::success;
  const auto remember = [&](const abi::Status status) {
    if (first_error == abi::success && status != abi::success) {
      first_error = status;
    }
  };
  if (descriptors.preference != nullptr) {
    remember(dispatch.lt_matmul_preference_destroy(descriptors.preference));
    descriptors.preference = nullptr;
  }
  if (descriptors.d != nullptr) {
    remember(dispatch.lt_matrix_layout_destroy(descriptors.d));
    descriptors.d = nullptr;
  }
  if (descriptors.c != nullptr) {
    remember(dispatch.lt_matrix_layout_destroy(descriptors.c));
    descriptors.c = nullptr;
  }
  if (descriptors.b != nullptr) {
    remember(dispatch.lt_matrix_layout_destroy(descriptors.b));
    descriptors.b = nullptr;
  }
  if (descriptors.a != nullptr) {
    remember(dispatch.lt_matrix_layout_destroy(descriptors.a));
    descriptors.a = nullptr;
  }
  if (descriptors.operation != nullptr) {
    remember(dispatch.lt_matmul_desc_destroy(descriptors.operation));
    descriptors.operation = nullptr;
  }
  return first_error;
}

[[nodiscard]] bool fallback_status(const abi::Status status) noexcept {
  return status == abi::not_supported || status == abi::architecture_mismatch;
}

[[nodiscard]] bool dispatch_supports_lt_execution(const CublasDispatch& dispatch) noexcept {
  return dispatch.lt_create != nullptr && dispatch.lt_destroy != nullptr &&
         dispatch.lt_matmul_desc_create != nullptr && dispatch.lt_matmul_desc_destroy != nullptr &&
         dispatch.lt_matmul_desc_set_attribute != nullptr &&
         dispatch.lt_matrix_layout_create != nullptr &&
         dispatch.lt_matrix_layout_destroy != nullptr &&
         dispatch.lt_matrix_layout_set_attribute != nullptr &&
         dispatch.lt_matmul_preference_create != nullptr &&
         dispatch.lt_matmul_preference_destroy != nullptr &&
         dispatch.lt_matmul_preference_set_attribute != nullptr &&
         (dispatch.lt_matmul_algo_get_heuristic != nullptr ||
          dispatch.lt_matmul_algo_get_heuristic_for_stream != nullptr) &&
         dispatch.lt_matmul != nullptr;
}

} // namespace

CoreExecutionResult execute_core_gemm(const CublasDispatch& dispatch, const abi::Handle handle,
                                      const CoreGemmRequest& request) noexcept {
  const auto fail = [](const abi::Status status, const std::string_view operation) {
    return CoreExecutionResult{status, operation};
  };
  if (handle == nullptr || dispatch.set_stream == nullptr || dispatch.set_math_mode == nullptr ||
      dispatch.gemm_ex == nullptr || dispatch.dgemm == nullptr || request.m <= 0 ||
      request.n <= 0 || request.k <= 0 || request.a == nullptr || request.b == nullptr ||
      request.c == nullptr || request.lda <= 0 || request.ldb <= 0 || request.ldc <= 0 ||
      !is_operation_valid(request.operation_a) || !is_operation_valid(request.operation_b) ||
      request.workspace_bytes > std::numeric_limits<std::size_t>::max() ||
      (request.workspace_bytes != 0 && request.workspace == nullptr)) {
    return fail(abi::invalid_value, "validate_core_request");
  }
  abi::Status status = dispatch.set_stream(handle, request.stream);
  if (status != abi::success) {
    return fail(status, "cublasSetStream_v2");
  }
  status = dispatch.set_math_mode(handle, request.math_mode);
  if (status != abi::success) {
    return fail(status, "cublasSetMathMode");
  }
  if (dispatch.set_workspace != nullptr && request.workspace_bytes != 0) {
    status = dispatch.set_workspace(handle, request.workspace,
                                    static_cast<std::size_t>(request.workspace_bytes));
    if (status != abi::success) {
      return fail(status, "cublasSetWorkspace_v2");
    }
  }

  if (request.compute_type == abi::compute_fp64) {
    if (request.a_type != abi::data_fp64 || request.b_type != abi::data_fp64 ||
        request.c_type != abi::data_fp64) {
      return fail(abi::invalid_value, "validate_core_types");
    }
    status = dispatch.dgemm(handle, request.operation_a, request.operation_b, request.m, request.n,
                            request.k, &request.alpha, static_cast<const double*>(request.a),
                            request.lda, static_cast<const double*>(request.b), request.ldb,
                            &request.beta, static_cast<double*>(request.c), request.ldc);
    return fail(status, "cublasDgemm_v2");
  }

  const float alpha = static_cast<float>(request.alpha);
  const float beta = static_cast<float>(request.beta);
  status = dispatch.gemm_ex(handle, request.operation_a, request.operation_b, request.m, request.n,
                            request.k, &alpha, request.a, request.a_type, request.lda, request.b,
                            request.b_type, request.ldb, &beta, request.c, request.c_type,
                            request.ldc, request.compute_type, abi::gemm_default);
  return fail(status, "cublasGemmEx");
}

LtMatmulExecutor::LtMatmulExecutor(const CublasDispatch& dispatch, const std::size_t cache_capacity)
    : dispatch_(&dispatch), cache_(cache_capacity) {}

LtMatmulExecutor::~LtMatmulExecutor() {
  (void)close();
}

abi::Status LtMatmulExecutor::initialize() noexcept {
  if (handle_ != nullptr) {
    return abi::success;
  }
  if (dispatch_ == nullptr || !dispatch_supports_lt_execution(*dispatch_)) {
    return abi::not_initialized;
  }
  const abi::Status status = dispatch_->lt_create(&handle_);
  if (status != abi::success) {
    handle_ = nullptr;
    return status;
  }
  library_version_ =
      dispatch_->lt_get_version != nullptr ? dispatch_->lt_get_version() : std::size_t{0};
  cache_.clear();
  return abi::success;
}

abi::Status LtMatmulExecutor::close() noexcept {
  cache_.clear();
  library_version_ = 0;
  if (handle_ == nullptr) {
    return abi::success;
  }
  const abi::LtHandle handle = std::exchange(handle_, nullptr);
  return dispatch_ != nullptr && dispatch_->lt_destroy != nullptr ? dispatch_->lt_destroy(handle)
                                                                  : abi::not_initialized;
}

bool LtMatmulExecutor::ready() const noexcept {
  return handle_ != nullptr;
}

abi::LtHandle LtMatmulExecutor::handle() const noexcept {
  return handle_;
}

std::size_t LtMatmulExecutor::library_version() const noexcept {
  return library_version_;
}

LtAlgorithmCache& LtMatmulExecutor::cache() noexcept {
  return cache_;
}

const LtAlgorithmCache& LtMatmulExecutor::cache() const noexcept {
  return cache_;
}

LtExecutionResult LtMatmulExecutor::execute(const LtMatmulRequest& request) {
  if (handle_ == nullptr || dispatch_ == nullptr) {
    return {LtExecutionCode::not_initialized, abi::not_initialized, "cublasLtCreate"};
  }

  LtMatmulSignature signature;
  if (!normalize_request(request, signature)) {
    return {LtExecutionCode::invalid_request, abi::invalid_value, "validate_lt_request"};
  }

  LtDescriptorSet descriptors;
  const auto finish = [&](const LtExecutionCode code, const abi::Status status,
                          const std::string_view operation, const bool cache_hit = false,
                          const std::uint64_t workspace = 0) {
    const abi::Status cleanup = destroy_descriptors(*dispatch_, descriptors);
    if (cleanup != abi::success) {
      return LtExecutionResult{LtExecutionCode::cleanup_failure, cleanup,
                               "cublasLt descriptor cleanup", cache_hit, workspace};
    }
    return LtExecutionResult{code, status, operation, cache_hit, workspace};
  };
  const auto fail_for_status = [&](const abi::Status status, const std::string_view operation) {
    return finish(fallback_status(status) ? LtExecutionCode::fallback_required
                                          : LtExecutionCode::cublas_failure,
                  status, operation);
  };

  abi::Status status = dispatch_->lt_matmul_desc_create(
      &descriptors.operation, signature.compute_type, signature.scale_type);
  if (status != abi::success) {
    return fail_for_status(status, "cublasLtMatmulDescCreate");
  }
  status = dispatch_->lt_matmul_desc_set_attribute(
      descriptors.operation, abi::lt_matmul_desc_transpose_a, &signature.operation_a,
      sizeof(signature.operation_a));
  if (status != abi::success) {
    return fail_for_status(status, "cublasLtMatmulDescSetAttribute(TRANSA)");
  }
  status = dispatch_->lt_matmul_desc_set_attribute(
      descriptors.operation, abi::lt_matmul_desc_transpose_b, &signature.operation_b,
      sizeof(signature.operation_b));
  if (status != abi::success) {
    return fail_for_status(status, "cublasLtMatmulDescSetAttribute(TRANSB)");
  }

  const auto create_layout = [&](const LtMatrixSignature& matrix,
                                 abi::LtMatrixLayout& output) -> abi::Status {
    abi::Status layout_status = dispatch_->lt_matrix_layout_create(
        &output, matrix.data_type, matrix.rows, matrix.columns, matrix.leading_dimension);
    if (layout_status == abi::success) {
      layout_status = dispatch_->lt_matrix_layout_set_attribute(
          output, abi::lt_matrix_layout_order, &matrix.order, sizeof(matrix.order));
    }
    return layout_status;
  };
  for (const auto& [matrix, output, name] :
       std::array{std::tuple{&signature.a, &descriptors.a, "A"},
                  std::tuple{&signature.b, &descriptors.b, "B"},
                  std::tuple{&signature.c, &descriptors.c, "C"},
                  std::tuple{&signature.d, &descriptors.d, "D"}}) {
    status = create_layout(*matrix, *output);
    if (status != abi::success) {
      const std::string_view operation = name[0] == 'A'   ? "cublasLtMatrixLayoutCreate(A)"
                                         : name[0] == 'B' ? "cublasLtMatrixLayoutCreate(B)"
                                         : name[0] == 'C' ? "cublasLtMatrixLayoutCreate(C)"
                                                          : "cublasLtMatrixLayoutCreate(D)";
      return fail_for_status(status, operation);
    }
  }

  CachedLtAlgorithm selected;
  const bool cache_hit = cache_.find(signature, selected);
  if (!cache_hit) {
    status = dispatch_->lt_matmul_preference_create(&descriptors.preference);
    if (status != abi::success) {
      return fail_for_status(status, "cublasLtMatmulPreferenceCreate");
    }
    const auto set_preference = [&](const abi::LtAttribute attribute, const auto& value) {
      return dispatch_->lt_matmul_preference_set_attribute(descriptors.preference, attribute,
                                                           &value, sizeof(value));
    };
    status =
        set_preference(abi::lt_preference_max_workspace_bytes, signature.workspace_limit_bytes);
    if (status == abi::success) {
      status = set_preference(abi::lt_preference_min_alignment_a_bytes,
                              signature.a.minimum_alignment_bytes);
    }
    if (status == abi::success) {
      status = set_preference(abi::lt_preference_min_alignment_b_bytes,
                              signature.b.minimum_alignment_bytes);
    }
    if (status == abi::success) {
      status = set_preference(abi::lt_preference_min_alignment_c_bytes,
                              signature.c.minimum_alignment_bytes);
    }
    if (status == abi::success) {
      status = set_preference(abi::lt_preference_min_alignment_d_bytes,
                              signature.d.minimum_alignment_bytes);
    }
    if (status != abi::success) {
      return fail_for_status(status, "cublasLtMatmulPreferenceSetAttribute");
    }

    std::array<abi::LtMatmulHeuristicResult, 8> results{};
    int returned = 0;
    status = dispatch_->lt_matmul_algo_get_heuristic_for_stream != nullptr
                 ? dispatch_->lt_matmul_algo_get_heuristic_for_stream(
                       handle_, descriptors.operation, descriptors.a, descriptors.b, descriptors.c,
                       descriptors.d, descriptors.preference, static_cast<int>(results.size()),
                       results.data(), &returned, request.stream)
                 : dispatch_->lt_matmul_algo_get_heuristic(
                       handle_, descriptors.operation, descriptors.a, descriptors.b, descriptors.c,
                       descriptors.d, descriptors.preference, static_cast<int>(results.size()),
                       results.data(), &returned);
    if (status != abi::success) {
      return fail_for_status(status, "cublasLtMatmulAlgoGetHeuristic");
    }
    const int result_count = std::clamp(returned, 0, static_cast<int>(results.size()));
    const auto usable =
        std::find_if(results.begin(), results.begin() + result_count,
                     [&](const abi::LtMatmulHeuristicResult& result) {
                       return result.state == abi::success &&
                              result.workspace_size <= signature.workspace_limit_bytes &&
                              result.workspace_size <= request.workspace_bytes;
                     });
    if (usable == results.begin() + result_count) {
      return finish(LtExecutionCode::fallback_required, abi::not_supported,
                    "cublasLtMatmulAlgoGetHeuristic(no algorithm)");
    }
    selected.algorithm = usable->algorithm;
    selected.workspace_bytes = static_cast<std::uint64_t>(usable->workspace_size);
    selected.waves_count = usable->waves_count;

    if (dispatch_->lt_matmul_algo_check_for_stream != nullptr) {
      abi::LtMatmulHeuristicResult checked{};
      status = dispatch_->lt_matmul_algo_check_for_stream(
          handle_, descriptors.operation, descriptors.a, descriptors.b, descriptors.c,
          descriptors.d, &selected.algorithm, &checked, request.stream);
      if (status != abi::success || checked.state != abi::success ||
          checked.workspace_size > signature.workspace_limit_bytes ||
          checked.workspace_size > request.workspace_bytes) {
        return fail_for_status(status != abi::success ? status : abi::not_supported,
                               "cublasLtMatmulAlgoCheckForStream");
      }
      selected.workspace_bytes = static_cast<std::uint64_t>(checked.workspace_size);
      selected.waves_count = checked.waves_count;
    }
  }

  status =
      dispatch_->lt_matmul(handle_, descriptors.operation, request.alpha, request.a, descriptors.a,
                           request.b, descriptors.b, request.beta, request.c, descriptors.c,
                           request.d, descriptors.d, &selected.algorithm, request.workspace,
                           static_cast<std::size_t>(request.workspace_bytes), request.stream);
  if (status != abi::success) {
    if (cache_hit) {
      (void)cache_.erase(signature);
    }
    return finish(fallback_status(status) ? LtExecutionCode::fallback_required
                                          : LtExecutionCode::cublas_failure,
                  status, "cublasLtMatmul", cache_hit, selected.workspace_bytes);
  }
  if (!cache_hit) {
    cache_.insert(signature, selected);
  }
  return finish(LtExecutionCode::success, abi::success, "cublasLtMatmul", cache_hit,
                selected.workspace_bytes);
}

PreferredGemmResult
LtMatmulExecutor::execute_preferred(const LtMatmulRequest& request, const abi::Handle core_handle,
                                    const std::optional<CoreGemmRequest>& core_fallback) {
  PreferredGemmResult result;
  result.lt = execute(request);
  if (result.lt) {
    result.path = PreferredGemmPath::cublas_lt;
    return result;
  }
  if ((result.lt.code == LtExecutionCode::fallback_required ||
       result.lt.code == LtExecutionCode::not_initialized) &&
      core_fallback.has_value()) {
    result.core = execute_core_gemm(*dispatch_, core_handle, *core_fallback);
    result.path = PreferredGemmPath::cublas_core;
  }
  return result;
}

std::string_view lt_execution_code_name(const LtExecutionCode code) noexcept {
  switch (code) {
  case LtExecutionCode::success:
    return "success";
  case LtExecutionCode::fallback_required:
    return "fallback_required";
  case LtExecutionCode::invalid_request:
    return "invalid_request";
  case LtExecutionCode::not_initialized:
    return "not_initialized";
  case LtExecutionCode::cublas_failure:
    return "cublas_failure";
  case LtExecutionCode::cleanup_failure:
    return "cleanup_failure";
  }
  return "invalid";
}

std::string_view preferred_gemm_path_name(const PreferredGemmPath path) noexcept {
  switch (path) {
  case PreferredGemmPath::none:
    return "none";
  case PreferredGemmPath::cublas_lt:
    return "cublas_lt";
  case PreferredGemmPath::cublas_core:
    return "cublas_core";
  }
  return "invalid";
}

} // namespace xvram::cublas
