# cuBLAS includes cuda_fp16.h even for this FP32-only, host-only facade. CUDA 13
# redistributes its nv/target dependency separately from the Runtime/CRT headers.
set(XVRAM_CUDA_CCCL_INCLUDE_DIR "" CACHE PATH
    "Optional matching CUDA CCCL include root containing nv/target")
set(XVRAM_CUDA_CCCL_HEADERS_DIR "${XVRAM_CUDA_CCCL_INCLUDE_DIR}")
if(NOT XVRAM_CUDA_CCCL_HEADERS_DIR)
  unset(XVRAM_CUDA_CCCL_HEADERS_DIR)
  find_path(XVRAM_CUDA_CCCL_HEADERS_DIR NAMES nv/target
            PATHS "${XVRAM_CUDA_INCLUDE_DIR}" NO_DEFAULT_PATH NO_CACHE)
endif()
if(NOT XVRAM_CUDA_CCCL_HEADERS_DIR AND XVRAM_FETCH_CUDA_HEADERS)
  if(NOT XVRAM_CUDA_HEADERS_VERSION EQUAL 13030)
    message(FATAL_ERROR
            "Incomplete CUDA Toolkit headers: set XVRAM_CUDA_CCCL_INCLUDE_DIR to the matching include directory containing nv/target")
  endif()
  # Matching cuda_cudart 13.3.29 and cuda_crt 13.3.33, pinned by NVIDIA's
  # https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.3.0.json
  if(WIN32)
    set(XVRAM_CCCL_HEADERS_URL
        "https://developer.download.nvidia.com/compute/cuda/redist/cccl/windows-x86_64/cccl-windows-x86_64-13.3.3.3.1-archive.zip")
    set(XVRAM_CCCL_HEADERS_HASH
        "SHA256=607dcfca31da168171fbdae5b7096ade646c4c2b1e0ff2899077dde0ccbdd6fb")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    set(XVRAM_CCCL_HEADERS_URL
        "https://developer.download.nvidia.com/compute/cuda/redist/cccl/linux-x86_64/cccl-linux-x86_64-13.3.3.3.1-archive.tar.xz")
    set(XVRAM_CCCL_HEADERS_HASH
        "SHA256=67746da12f16229ac4ebde78ce7895e42b069d1d3e2ae2d2d25f90bc43679d68")
  else()
    message(FATAL_ERROR
            "Automatic CUDA CCCL fetching supports Windows x64 and Linux x86_64 only; set XVRAM_CUDA_CCCL_INCLUDE_DIR")
  endif()
  message(STATUS "Fetching NVIDIA CCCL 13.3.3.3.1 headers")
  FetchContent_Declare(xvram_cuda_cccl_headers
                      URL "${XVRAM_CCCL_HEADERS_URL}"
                      URL_HASH "${XVRAM_CCCL_HEADERS_HASH}"
                      DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(xvram_cuda_cccl_headers)
  find_path(XVRAM_CUDA_CCCL_HEADERS_DIR NAMES nv/target
            PATHS "${xvram_cuda_cccl_headers_SOURCE_DIR}/include"
            NO_DEFAULT_PATH NO_CACHE)
endif()
if(NOT EXISTS "${XVRAM_CUDA_CCCL_HEADERS_DIR}/nv/target")
  message(FATAL_ERROR
          "nv/target is required by the cuBLAS facade. Set XVRAM_CUDA_CCCL_INCLUDE_DIR or enable XVRAM_FETCH_CUDA_HEADERS.")
endif()

function(xvram_check_cuda_compat_headers)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_REQUIRED_INCLUDES "${PROJECT_SOURCE_DIR}/include"
                             "${XVRAM_CUDA_INCLUDE_DIR}"
                             "${XVRAM_CUDA_CRT_HEADERS_DIR}"
                             "${XVRAM_CUDA_CCCL_HEADERS_DIR}"
                             "${XVRAM_CUBLAS_INCLUDE_DIR}")
  set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
  unset(XVRAM_CUDA_COMPAT_HEADERS_COMPILE CACHE)
  check_cxx_source_compiles(
    "#include <xvram/cuda_compat.hpp>\nint main() { return 0; }"
    XVRAM_CUDA_COMPAT_HEADERS_COMPILE)
  if(NOT XVRAM_CUDA_COMPAT_HEADERS_COMPILE)
    message(FATAL_ERROR
            "CUDA Runtime/CRT/CCCL and cuBLAS headers cannot compile the host-only facade; check the configured include directories and CMake configure log")
  endif()
  unset(XVRAM_CUDA_COMPAT_HEADERS_COMPILE CACHE)
endfunction()
xvram_check_cuda_compat_headers()
