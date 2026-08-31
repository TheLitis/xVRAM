#pragma once

#include <cstddef>
#include <cstdint>

#ifdef _WIN32
#define XVRAM_CUBLAS_CALL __stdcall
#else
#define XVRAM_CUBLAS_CALL
#endif

namespace xvram::cublas::abi {

// cuBLAS handles and all cuBLASLt descriptors are intentionally opaque here. Keeping this
// dispatch header independent from NVIDIA's development headers lets no-driver SDK consumers
// build without installing the CUDA Toolkit. The runtime still validates every required symbol
// before exposing a loaded dispatch.
struct Context;
struct LtContext;
struct LtMatmulDescriptor;
struct LtMatrixLayoutDescriptor;
struct LtMatmulPreferenceDescriptor;

using Handle = Context*;
using LtHandle = LtContext*;
using LtMatmulDesc = LtMatmulDescriptor*;
using LtMatrixLayout = LtMatrixLayoutDescriptor*;
using LtMatmulPreference = LtMatmulPreferenceDescriptor*;
using Stream = void*;
using Status = int;
using Operation = int;
using DataType = int;
using ComputeType = int;
using GemmAlgorithm = int;
using MathMode = int;
using LtAttribute = int;
using LtOrder = int;

inline constexpr Status success = 0;
inline constexpr Status not_initialized = 1;
inline constexpr Status allocation_failed = 3;
inline constexpr Status invalid_value = 7;
inline constexpr Status architecture_mismatch = 8;
inline constexpr Status mapping_error = 11;
inline constexpr Status execution_failed = 13;
inline constexpr Status internal_error = 14;
inline constexpr Status not_supported = 15;
inline constexpr Status license_error = 16;

inline constexpr Operation operation_none = 0;
inline constexpr Operation operation_transpose = 1;

// cudaDataType values are part of the CUDA public ABI and have been stable since their
// introduction. Only the Phase 3 floating-point subset is declared.
inline constexpr DataType data_fp32 = 0;
inline constexpr DataType data_fp64 = 1;
inline constexpr DataType data_fp16 = 2;
inline constexpr DataType data_bf16 = 14;

// cublasComputeType_t values used by Phase 3.
inline constexpr ComputeType compute_fp32 = 68;
inline constexpr ComputeType compute_fp32_pedantic = 69;
inline constexpr ComputeType compute_fp64 = 70;
inline constexpr ComputeType compute_fast_tf32 = 77;

inline constexpr MathMode default_math = 0;
inline constexpr MathMode pedantic_math = 2;
inline constexpr MathMode tf32_tensor_op_math = 3;
inline constexpr MathMode disallow_reduced_precision_reduction = 16;
inline constexpr GemmAlgorithm gemm_default = -1;

// CUDA 13.3 / libcublas 13.5 public cuBLASLt ABI values. These values are also present in
// supported CUDA 12 redistributables.
inline constexpr LtOrder lt_order_column_major = 0;
inline constexpr LtOrder lt_order_row_major = 1;
inline constexpr LtAttribute lt_matrix_layout_order = 1;
inline constexpr LtAttribute lt_matmul_desc_transpose_a = 3;
inline constexpr LtAttribute lt_matmul_desc_transpose_b = 4;
inline constexpr LtAttribute lt_preference_max_workspace_bytes = 1;
inline constexpr LtAttribute lt_preference_min_alignment_a_bytes = 5;
inline constexpr LtAttribute lt_preference_min_alignment_b_bytes = 6;
inline constexpr LtAttribute lt_preference_min_alignment_c_bytes = 7;
inline constexpr LtAttribute lt_preference_min_alignment_d_bytes = 8;

// cublasLtMatmulAlgo_t and cublasLtMatmulHeuristicResult_t are semi-opaque, trivially
// serializable public ABI structures. Their representation is required to retain heuristic
// selections without linking the CUDA Toolkit.
struct LtMatmulAlgorithm {
  std::uint64_t data[8]{};
};

struct LtMatmulHeuristicResult {
  LtMatmulAlgorithm algorithm{};
  std::size_t workspace_size = 0;
  Status state = success;
  float waves_count = 0.0F;
  int reserved[4]{};
};

static_assert(sizeof(LtMatmulAlgorithm) == 64);
static_assert(sizeof(LtMatmulHeuristicResult) == 96);

using Create = Status(XVRAM_CUBLAS_CALL*)(Handle*);
using Destroy = Status(XVRAM_CUBLAS_CALL*)(Handle);
using SetStream = Status(XVRAM_CUBLAS_CALL*)(Handle, Stream);
using GetVersion = Status(XVRAM_CUBLAS_CALL*)(Handle, int*);
using SetMathMode = Status(XVRAM_CUBLAS_CALL*)(Handle, MathMode);
using SetWorkspace = Status(XVRAM_CUBLAS_CALL*)(Handle, void*, std::size_t);
using GemmEx = Status(XVRAM_CUBLAS_CALL*)(Handle, Operation, Operation, int, int, int, const void*,
                                          const void*, DataType, int, const void*, DataType, int,
                                          const void*, void*, DataType, int, ComputeType,
                                          GemmAlgorithm);
using Dgemm = Status(XVRAM_CUBLAS_CALL*)(Handle, Operation, Operation, int, int, int, const double*,
                                         const double*, int, const double*, int, const double*,
                                         double*, int);
using GetStatusString = const char*(XVRAM_CUBLAS_CALL*)(Status);

using LtCreate = Status(XVRAM_CUBLAS_CALL*)(LtHandle*);
using LtDestroy = Status(XVRAM_CUBLAS_CALL*)(LtHandle);
using LtGetVersion = std::size_t(XVRAM_CUBLAS_CALL*)();
using LtMatmulDescCreate = Status(XVRAM_CUBLAS_CALL*)(LtMatmulDesc*, ComputeType, DataType);
using LtMatmulDescDestroy = Status(XVRAM_CUBLAS_CALL*)(LtMatmulDesc);
using LtMatmulDescSetAttribute = Status(XVRAM_CUBLAS_CALL*)(LtMatmulDesc, LtAttribute, const void*,
                                                            std::size_t);
using LtMatrixLayoutCreate = Status(XVRAM_CUBLAS_CALL*)(LtMatrixLayout*, DataType, std::uint64_t,
                                                        std::uint64_t, std::int64_t);
using LtMatrixLayoutDestroy = Status(XVRAM_CUBLAS_CALL*)(LtMatrixLayout);
using LtMatrixLayoutSetAttribute = Status(XVRAM_CUBLAS_CALL*)(LtMatrixLayout, LtAttribute,
                                                              const void*, std::size_t);
using LtMatmulPreferenceCreate = Status(XVRAM_CUBLAS_CALL*)(LtMatmulPreference*);
using LtMatmulPreferenceDestroy = Status(XVRAM_CUBLAS_CALL*)(LtMatmulPreference);
using LtMatmulPreferenceSetAttribute = Status(XVRAM_CUBLAS_CALL*)(LtMatmulPreference, LtAttribute,
                                                                  const void*, std::size_t);

using LtMatmulAlgoGetHeuristic = Status(XVRAM_CUBLAS_CALL*)(LtHandle, LtMatmulDesc, LtMatrixLayout,
                                                            LtMatrixLayout, LtMatrixLayout,
                                                            LtMatrixLayout, LtMatmulPreference, int,
                                                            LtMatmulHeuristicResult*, int*);
using LtMatmulAlgoGetHeuristicForStream = Status(XVRAM_CUBLAS_CALL*)(
    LtHandle, LtMatmulDesc, LtMatrixLayout, LtMatrixLayout, LtMatrixLayout, LtMatrixLayout,
    LtMatmulPreference, int, LtMatmulHeuristicResult*, int*, Stream);
using LtMatmulAlgoCheckForStream = Status(XVRAM_CUBLAS_CALL*)(LtHandle, LtMatmulDesc,
                                                              LtMatrixLayout, LtMatrixLayout,
                                                              LtMatrixLayout, LtMatrixLayout,
                                                              const LtMatmulAlgorithm*,
                                                              LtMatmulHeuristicResult*, Stream);
using LtMatmul = Status(XVRAM_CUBLAS_CALL*)(LtHandle, LtMatmulDesc, const void*, const void*,
                                            LtMatrixLayout, const void*, LtMatrixLayout,
                                            const void*, const void*, LtMatrixLayout, void*,
                                            LtMatrixLayout, const LtMatmulAlgorithm*, void*,
                                            std::size_t, Stream);

} // namespace xvram::cublas::abi

#undef XVRAM_CUBLAS_CALL
