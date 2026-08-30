cmake_minimum_required(VERSION 3.25)

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" XVRAM_VERSION LIMIT_COUNT 1)
if(NOT XVRAM_VERSION MATCHES "^([0-9]+)[.]([0-9]+)[.]([0-9]+)(-[0-9A-Za-z.-]+)?$")
  message(FATAL_ERROR "VERSION must contain a SemVer-compatible version")
endif()
set(XVRAM_PROJECT_VERSION "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")

project(
  xVRAM
  VERSION ${XVRAM_PROJECT_VERSION}
  DESCRIPTION "Software-managed tiered CUDA memory runtime"
  LANGUAGES CXX)

include(CTest)
include(CheckCXXSourceCompiles)
include(CMakePackageConfigHelpers)
include(FetchContent)
include(GNUInstallDirs)
find_package(Threads REQUIRED)

# The public SDK is shared while its implementation libraries remain static.  Keep
# those archives linkable into the shared object on ELF platforms as well as PE/COFF.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(XVRAM_BUILD_TESTS "Build xVRAM tests" ${BUILD_TESTING})
option(XVRAM_FETCH_CUDA_HEADERS
       "Download NVIDIA's official CUDA redistributable headers when no Toolkit is installed" ON)
option(XVRAM_FETCH_CUBLAS_REDIST
       "Download and stage NVIDIA's hash-pinned cuBLAS redistributable beside the SDK" OFF)
set(XVRAM_CUBLAS_REDIST_ROOT "" CACHE PATH
    "Existing official cuBLAS redistributable root to stage beside the SDK")
option(XVRAM_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR "xVRAM currently supports 64-bit targets only")
endif()
if(NOT WIN32 AND (NOT UNIX OR APPLE))
  message(FATAL_ERROR "xVRAM Phase 0 currently supports Windows and Linux only")
endif()

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Git QUIET)
set(XVRAM_GIT_COMMIT "unknown")
if(Git_FOUND)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse --short=12 HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RESULT_VARIABLE XVRAM_GIT_RESULT
    OUTPUT_VARIABLE XVRAM_GIT_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(XVRAM_GIT_RESULT EQUAL 0)
    set(XVRAM_GIT_COMMIT "${XVRAM_GIT_OUTPUT}")
  endif()
endif()

find_path(
  XVRAM_CUDA_INCLUDE_DIR
  NAMES cuda.h
  HINTS "$ENV{CUDA_PATH}"
  PATH_SUFFIXES include)

function(xvram_read_cuda_header_version include_dir output_variable)
  set(version "")
  if(EXISTS "${include_dir}/cuda.h")
    file(
      STRINGS "${include_dir}/cuda.h" version_line
      REGEX "^#define[ \t]+CUDA_VERSION[ \t]+[0-9]+"
      LIMIT_COUNT 1)
    if(version_line MATCHES "CUDA_VERSION[ \t]+([0-9]+)")
      set(version "${CMAKE_MATCH_1}")
    endif()
  endif()
  set(${output_variable} "${version}" PARENT_SCOPE)
endfunction()

if(XVRAM_CUDA_INCLUDE_DIR)
  xvram_read_cuda_header_version("${XVRAM_CUDA_INCLUDE_DIR}"
                                 XVRAM_CUDA_HEADERS_VERSION)
  if(NOT XVRAM_CUDA_HEADERS_VERSION OR XVRAM_CUDA_HEADERS_VERSION LESS 13000)
    message(
      STATUS
        "Ignoring CUDA headers older than 13.0: ${XVRAM_CUDA_INCLUDE_DIR} (CUDA_VERSION=${XVRAM_CUDA_HEADERS_VERSION})"
    )
    unset(XVRAM_CUDA_INCLUDE_DIR CACHE)
    unset(XVRAM_CUDA_INCLUDE_DIR)
    unset(XVRAM_CUDA_HEADERS_VERSION)
  endif()
endif()

if(NOT XVRAM_CUDA_INCLUDE_DIR AND XVRAM_FETCH_CUDA_HEADERS)
  message(STATUS "CUDA headers not found; fetching NVIDIA cuda_cudart 13.3.29 headers")
  if(WIN32)
    set(XVRAM_CUDART_HEADERS_URL
        "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-13.3.29-archive.zip"
    )
    set(XVRAM_CUDART_HEADERS_HASH
        "SHA256=1FEB7DD266813FFE8DBC24E115183A5AC35A4795C8D34ACA0DF85AB616B64D9C"
    )
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    set(XVRAM_CUDART_HEADERS_URL
        "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-13.3.29-archive.tar.xz"
    )
    set(XVRAM_CUDART_HEADERS_HASH
        "SHA256=1E59C4888267D27BA1A9BD0F3669A6439DB1334A96E754CD9013C7C73E18DC9D"
    )
  else()
    message(
      FATAL_ERROR
        "Automatic CUDA header fetching supports Windows x64 and Linux x86_64 only; install CUDA 13.x headers or set XVRAM_CUDA_INCLUDE_DIR"
    )
  endif()
  FetchContent_Declare(
    xvram_cuda_headers
    URL "${XVRAM_CUDART_HEADERS_URL}"
    URL_HASH "${XVRAM_CUDART_HEADERS_HASH}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(xvram_cuda_headers)
  find_path(
    XVRAM_CUDA_INCLUDE_DIR
    NAMES cuda.h
    PATHS "${xvram_cuda_headers_SOURCE_DIR}"
    PATH_SUFFIXES include
    NO_DEFAULT_PATH)
endif()

if(NOT XVRAM_CUDA_INCLUDE_DIR)
  message(FATAL_ERROR
          "cuda.h is required. Install CUDA Toolkit 13.3 or enable XVRAM_FETCH_CUDA_HEADERS.")
endif()

xvram_read_cuda_header_version("${XVRAM_CUDA_INCLUDE_DIR}" XVRAM_CUDA_HEADERS_VERSION)
if(NOT XVRAM_CUDA_HEADERS_VERSION OR XVRAM_CUDA_HEADERS_VERSION LESS 13000)
  message(FATAL_ERROR "xVRAM requires CUDA 13.0 or newer headers")
endif()

message(
  STATUS
    "Using CUDA headers: ${XVRAM_CUDA_INCLUDE_DIR} (CUDA_VERSION=${XVRAM_CUDA_HEADERS_VERSION})"
)

configure_file(cmake/version.hpp.in generated/xvram/version.hpp @ONLY)

set(XVRAM_SYNTHETIC_OVERLAP_MODULE_VERSION 1)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/src/platform/cuda/synthetic_overlap_v1.ptx"
     XVRAM_SYNTHETIC_OVERLAP_PTX)
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/src/platform/cuda/synthetic_overlap_v1.ptx"
     XVRAM_SYNTHETIC_OVERLAP_SHA256)
configure_file(cmake/synthetic_overlap_ptx.hpp.in
               generated/xvram/synthetic_overlap_ptx.hpp @ONLY)

set(XVRAM_VMM_TRANSFORM_MODULE_VERSION 1)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/src/vmm_poc/vmm_transform_v1.ptx"
     XVRAM_VMM_TRANSFORM_PTX)
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/src/vmm_poc/vmm_transform_v1.ptx"
     XVRAM_VMM_TRANSFORM_SHA256)
configure_file(cmake/vmm_transform_ptx.hpp.in
               generated/xvram/vmm_transform_ptx.hpp @ONLY)

set(XVRAM_RESIDENCY_WORKLOAD_MODULE_VERSION 1)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/src/residency/residency_workload_v1.ptx"
     XVRAM_RESIDENCY_WORKLOAD_PTX)
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/src/residency/residency_workload_v1.ptx"
     XVRAM_RESIDENCY_WORKLOAD_SHA256)
configure_file(cmake/residency_workload_ptx.hpp.in
               generated/xvram/residency_workload_ptx.hpp @ONLY)

find_path(
  XVRAM_NVML_INCLUDE_DIR
  NAMES nvml.h
  HINTS "$ENV{CUDA_PATH}"
  PATH_SUFFIXES include)

function(xvram_check_nvml_header include_dir output_variable)
  set(CMAKE_REQUIRED_INCLUDES "${include_dir}")
  set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
  unset(XVRAM_NVML_HEADER_FEATURES CACHE)
  check_cxx_source_compiles(
    [=[
      #include <nvml.h>
      using DriverModelFunction = decltype(&nvmlDeviceGetDriverModel_v2);
      static_assert(NVML_DRIVER_MCDM != NVML_DRIVER_WDDM);
      int main() { return sizeof(DriverModelFunction) > 0 ? 0 : 1; }
    ]=]
    XVRAM_NVML_HEADER_FEATURES)
  set(${output_variable} "${XVRAM_NVML_HEADER_FEATURES}" PARENT_SCOPE)
  unset(XVRAM_NVML_HEADER_FEATURES CACHE)
endfunction()

if(XVRAM_NVML_INCLUDE_DIR)
  xvram_check_nvml_header("${XVRAM_NVML_INCLUDE_DIR}" XVRAM_NVML_HEADER_COMPATIBLE)
  if(NOT XVRAM_NVML_HEADER_COMPATIBLE)
    message(
      STATUS
        "Ignoring NVML headers without MCDM/driver-model v2 support: ${XVRAM_NVML_INCLUDE_DIR}"
    )
    unset(XVRAM_NVML_INCLUDE_DIR CACHE)
    unset(XVRAM_NVML_INCLUDE_DIR)
    unset(XVRAM_NVML_HEADER_COMPATIBLE)
  endif()
endif()

if(NOT XVRAM_NVML_INCLUDE_DIR AND XVRAM_FETCH_CUDA_HEADERS)
  message(STATUS "NVML headers not found; fetching NVIDIA cuda_nvml_dev 13.3.29 headers")
  if(WIN32)
    set(XVRAM_NVML_HEADERS_URL
        "https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvml_dev/windows-x86_64/cuda_nvml_dev-windows-x86_64-13.3.29-archive.zip"
    )
    set(XVRAM_NVML_HEADERS_HASH
        "SHA256=4E1BD1C66DD66293C6D94B1ED2EFA5065654533B36CCBC7477DC1F466DC3D65C"
    )
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    set(XVRAM_NVML_HEADERS_URL
        "https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvml_dev/linux-x86_64/cuda_nvml_dev-linux-x86_64-13.3.29-archive.tar.xz"
    )
    set(XVRAM_NVML_HEADERS_HASH
        "SHA256=B62508923DADC0AC79FB1846EEB444296813CB077964E0C8348BA051BC0160AA"
    )
  else()
    message(
      FATAL_ERROR
        "Automatic NVML header fetching supports Windows x64 and Linux x86_64 only; install NVML headers or set XVRAM_NVML_INCLUDE_DIR"
    )
  endif()
  FetchContent_Declare(
    xvram_nvml_headers
    URL "${XVRAM_NVML_HEADERS_URL}"
    URL_HASH "${XVRAM_NVML_HEADERS_HASH}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(xvram_nvml_headers)
  find_path(
    XVRAM_NVML_INCLUDE_DIR
    NAMES nvml.h
    PATHS "${xvram_nvml_headers_SOURCE_DIR}"
    PATH_SUFFIXES include
    NO_DEFAULT_PATH)
endif()

if(NOT XVRAM_NVML_INCLUDE_DIR)
  message(FATAL_ERROR
          "nvml.h is required. Install CUDA Toolkit 13.3 or enable XVRAM_FETCH_CUDA_HEADERS.")
endif()

if(NOT DEFINED XVRAM_NVML_HEADER_COMPATIBLE)
  xvram_check_nvml_header("${XVRAM_NVML_INCLUDE_DIR}" XVRAM_NVML_HEADER_COMPATIBLE)
endif()
if(NOT XVRAM_NVML_HEADER_COMPATIBLE)
  message(FATAL_ERROR "xVRAM requires NVML headers with MCDM and driver-model v2 declarations")
endif()

message(STATUS "Using NVML headers: ${XVRAM_NVML_INCLUDE_DIR}")

function(xvram_enable_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /EHsc /permissive- /Zc:__cplusplus /utf-8)
    if(XVRAM_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE /WX)
    endif()
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion
                                             -Wsign-conversion)
    if(XVRAM_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE -Werror)
    endif()
  endif()
endfunction()

add_library(
  xvram_probe_lib STATIC
  src/base/json_writer.cpp
  src/base/size_parser.cpp
  src/platform/cuda/cuda_api.cpp
  src/platform/cuda/cuda_driver.cpp
  src/platform/dynamic_library.cpp
  src/platform/pageable_memory.cpp
  src/platform/nvml/nvml.cpp
  src/platform/system_info.cpp
  src/probe/probe.cpp
  src/probe/report_json.cpp
  src/probe/report_text.cpp)

if(WIN32)
  target_sources(xvram_probe_lib PRIVATE src/platform/windows/dxgi_memory.cpp)
  target_link_libraries(xvram_probe_lib PRIVATE dxgi ole32 shell32)
else()
  target_sources(xvram_probe_lib PRIVATE src/platform/other/dxgi_memory_stub.cpp)
  target_link_libraries(xvram_probe_lib PRIVATE ${CMAKE_DL_LIBS})
endif()

target_compile_features(xvram_probe_lib PUBLIC cxx_std_20)
target_include_directories(
  xvram_probe_lib
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>"
         "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(xvram_probe_lib SYSTEM PRIVATE "${XVRAM_CUDA_INCLUDE_DIR}")
target_include_directories(xvram_probe_lib SYSTEM PRIVATE "${XVRAM_NVML_INCLUDE_DIR}")
xvram_enable_warnings(xvram_probe_lib)

add_executable(xvram-probe src/probe/main.cpp)
target_link_libraries(xvram-probe PRIVATE xvram_probe_lib)
xvram_enable_warnings(xvram-probe)

add_library(xvram_worker_process STATIC src/platform/worker_process.cpp)
target_compile_features(xvram_worker_process PUBLIC cxx_std_20)
target_include_directories(xvram_worker_process PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(xvram_worker_process PRIVATE Threads::Threads)
xvram_enable_warnings(xvram_worker_process)

add_library(
  xvram_residency_core STATIC
  src/residency/budget.cpp
  src/residency/core.cpp
  src/residency/metrics.cpp
  src/residency/policy.cpp
  src/residency/runtime.cpp
  src/residency/scenario.cpp)
target_compile_features(xvram_residency_core PUBLIC cxx_std_20)
target_include_directories(xvram_residency_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(xvram_residency_core SYSTEM PRIVATE "${XVRAM_CUDA_INCLUDE_DIR}")
target_link_libraries(xvram_residency_core PRIVATE xvram_probe_lib)
xvram_enable_warnings(xvram_residency_core)

add_library(
  xvram_vmm_poc STATIC
  src/vmm_poc/executor.cpp
  src/vmm_poc/outcome.cpp
  src/vmm_poc/report_json.cpp
  src/vmm_poc/report_text.cpp
  src/vmm_poc/worker_controller.cpp
  src/vmm_poc/worker_protocol.cpp
  src/vmm_poc/workload.cpp)
target_compile_features(xvram_vmm_poc PUBLIC cxx_std_20)
target_include_directories(
  xvram_vmm_poc
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>"
         "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(xvram_vmm_poc SYSTEM PRIVATE "${XVRAM_CUDA_INCLUDE_DIR}")
target_link_libraries(xvram_vmm_poc PRIVATE xvram_probe_lib xvram_worker_process Threads::Threads)
xvram_enable_warnings(xvram_vmm_poc)

add_executable(xvram-vmm-poc src/vmm_poc/main.cpp)
target_link_libraries(xvram-vmm-poc PRIVATE xvram_vmm_poc)
target_include_directories(xvram-vmm-poc PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(xvram-vmm-poc SYSTEM PRIVATE "${XVRAM_CUDA_INCLUDE_DIR}")
xvram_enable_warnings(xvram-vmm-poc)

add_library(
  xvram_residency STATIC
  src/residency/executor.cpp
  src/residency/report_json.cpp
  src/residency/report_text.cpp
  src/residency/worker_controller.cpp
  src/residency/worker_protocol.cpp
  src/residency/workload.cpp)
target_compile_features(xvram_residency PUBLIC cxx_std_20)
target_include_directories(
  xvram_residency
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>"
         "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(xvram_residency SYSTEM PRIVATE "${XVRAM_CUDA_INCLUDE_DIR}")
target_link_libraries(
  xvram_residency
  PRIVATE xvram_probe_lib xvram_residency_core xvram_worker_process Threads::Threads)
xvram_enable_warnings(xvram_residency)

add_executable(xvram-cache-bench src/residency/main.cpp)
target_link_libraries(xvram-cache-bench PRIVATE xvram_residency)
target_include_directories(xvram-cache-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(xvram-cache-bench SYSTEM PRIVATE "${XVRAM_CUDA_INCLUDE_DIR}")
xvram_enable_warnings(xvram-cache-bench)

add_library(
  xvram_gemm_core STATIC
  src/gemm/planner.cpp
  src/platform/cublas/cublas_api.cpp)
target_compile_features(xvram_gemm_core PUBLIC cxx_std_20)
target_include_directories(xvram_gemm_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(xvram_gemm_core PRIVATE xvram_probe_lib xvram_residency_core)
xvram_enable_warnings(xvram_gemm_core)

add_library(
  xvram_sdk SHARED
  src/sdk/api.cpp
  src/sdk/runtime.cpp
  src/sdk/status.cpp)
set_target_properties(
  xvram_sdk
  PROPERTIES OUTPUT_NAME xvram
             EXPORT_NAME xvram
             VERSION "${PROJECT_VERSION}"
             SOVERSION "${PROJECT_VERSION_MAJOR}"
             CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN YES)
target_compile_definitions(xvram_sdk PRIVATE XVRAM_BUILDING_LIBRARY
                           INTERFACE XVRAM_USING_LIBRARY)
target_compile_features(xvram_sdk PUBLIC cxx_std_20)
target_include_directories(
  xvram_sdk
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
         "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(xvram_sdk SYSTEM PRIVATE "${XVRAM_CUDA_INCLUDE_DIR}")
target_link_libraries(
  xvram_sdk
  PRIVATE xvram_probe_lib xvram_residency_core xvram_gemm_core Threads::Threads)
xvram_enable_warnings(xvram_sdk)

add_library(xVRAM::xvram ALIAS xvram_sdk)

add_library(
  xvram_gemm_bench STATIC
  src/gemm_bench/executor.cpp
  src/gemm_bench/pattern.cpp
  src/gemm_bench/sizing.cpp
  src/gemm_bench/report_json.cpp
  src/gemm_bench/report_text.cpp
  src/gemm_bench/worker_controller.cpp
  src/gemm_bench/worker_protocol.cpp)
target_compile_features(xvram_gemm_bench PUBLIC cxx_std_20)
target_include_directories(
  xvram_gemm_bench
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(xvram_gemm_bench SYSTEM PRIVATE "${XVRAM_CUDA_INCLUDE_DIR}")
target_link_libraries(
  xvram_gemm_bench
  PRIVATE xvram_sdk xvram_probe_lib xvram_gemm_core xvram_worker_process Threads::Threads)
xvram_enable_warnings(xvram_gemm_bench)

add_executable(xvram-gemm-bench src/gemm_bench/main.cpp)
target_link_libraries(xvram-gemm-bench PRIVATE xvram_gemm_bench)
target_include_directories(xvram-gemm-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(xvram-gemm-bench SYSTEM PRIVATE "${XVRAM_CUDA_INCLUDE_DIR}")
xvram_enable_warnings(xvram-gemm-bench)

if(XVRAM_FETCH_CUBLAS_REDIST)
  if(WIN32)
    set(XVRAM_CUBLAS_REDIST_URL
        "https://developer.download.nvidia.com/compute/cuda/redist/libcublas/windows-x86_64/libcublas-windows-x86_64-13.5.1.27-archive.zip")
    set(XVRAM_CUBLAS_REDIST_HASH
        "SHA256=C946E1C825E05895747A95ED4FEE18030B08052C09783B9B7B19818FD2E31F58")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    set(XVRAM_CUBLAS_REDIST_URL
        "https://developer.download.nvidia.com/compute/cuda/redist/libcublas/linux-x86_64/libcublas-linux-x86_64-13.5.1.27-archive.tar.xz")
    set(XVRAM_CUBLAS_REDIST_HASH
        "SHA256=35A898360520D6101FFCAF36C0D04496B54D4FC2AFB82F7FCE44218C54513808")
  else()
    message(FATAL_ERROR "Automatic cuBLAS staging supports Windows/Linux x86-64 only")
  endif()
  FetchContent_Declare(
    xvram_cublas_redist
    URL "${XVRAM_CUBLAS_REDIST_URL}"
    URL_HASH "${XVRAM_CUBLAS_REDIST_HASH}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(xvram_cublas_redist)
  set(XVRAM_CUBLAS_REDIST_ROOT "${xvram_cublas_redist_SOURCE_DIR}")
endif()

if(XVRAM_CUBLAS_REDIST_ROOT)
  if(WIN32)
    file(GLOB_RECURSE XVRAM_CUBLAS_CORE_LIBRARY
         LIST_DIRECTORIES FALSE "${XVRAM_CUBLAS_REDIST_ROOT}/cublas64_13.dll")
    file(GLOB_RECURSE XVRAM_CUBLAS_LT_LIBRARY
         LIST_DIRECTORIES FALSE "${XVRAM_CUBLAS_REDIST_ROOT}/cublasLt64_13.dll")
  else()
    file(GLOB_RECURSE XVRAM_CUBLAS_CORE_LIBRARY
         LIST_DIRECTORIES FALSE "${XVRAM_CUBLAS_REDIST_ROOT}/libcublas.so.13")
    file(GLOB_RECURSE XVRAM_CUBLAS_LT_LIBRARY
         LIST_DIRECTORIES FALSE "${XVRAM_CUBLAS_REDIST_ROOT}/libcublasLt.so.13")
  endif()
  list(LENGTH XVRAM_CUBLAS_CORE_LIBRARY XVRAM_CUBLAS_CORE_COUNT)
  list(LENGTH XVRAM_CUBLAS_LT_LIBRARY XVRAM_CUBLAS_LT_COUNT)
  if(NOT XVRAM_CUBLAS_CORE_COUNT EQUAL 1 OR NOT XVRAM_CUBLAS_LT_COUNT EQUAL 1)
    message(FATAL_ERROR "cuBLAS redistributable root does not contain one core/Lt runtime pair")
  endif()
  add_custom_command(
    TARGET xvram_sdk POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${XVRAM_CUBLAS_CORE_LIBRARY}" "$<TARGET_FILE_DIR:xvram_sdk>"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${XVRAM_CUBLAS_LT_LIBRARY}" "$<TARGET_FILE_DIR:xvram_sdk>"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${XVRAM_CUBLAS_REDIST_ROOT}/LICENSE"
            "$<TARGET_FILE_DIR:xvram_sdk>/LICENSE.cublas.txt"
    VERBATIM)
  if(WIN32)
    set(XVRAM_CUBLAS_INSTALL_DESTINATION "${CMAKE_INSTALL_BINDIR}")
  else()
    # The ELF loader resolves app-local cuBLAS beside libxvram.so.
    set(XVRAM_CUBLAS_INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}")
  endif()
  install(FILES "${XVRAM_CUBLAS_CORE_LIBRARY}" "${XVRAM_CUBLAS_LT_LIBRARY}"
          DESTINATION "${XVRAM_CUBLAS_INSTALL_DESTINATION}")
  install(FILES "${XVRAM_CUBLAS_REDIST_ROOT}/LICENSE"
          DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/xvram"
          RENAME LICENSE.cublas.txt)
endif()

install(
  TARGETS xvram-probe xvram-vmm-poc xvram-cache-bench xvram-gemm-bench xvram_sdk
  EXPORT xVRAMTargets
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(FILES include/xvram/xvram.h DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/xvram")
install(FILES schemas/capability-report-v1.schema.json
              schemas/capability-report-v2.schema.json
              schemas/vmm-poc-report-v1.schema.json
              schemas/residency-cache-report-v1.schema.json
              schemas/residency-trace-record-v1.schema.json
              schemas/gemm-bench-report-v1.schema.json
        DESTINATION "${CMAKE_INSTALL_DATADIR}/xvram/schemas")

configure_package_config_file(
  cmake/xVRAMConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/xVRAMConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/xVRAM")
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/xVRAMConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion)
install(
  EXPORT xVRAMTargets
  FILE xVRAMTargets.cmake
  NAMESPACE xVRAM::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/xVRAM")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/xVRAMConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/xVRAMConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/xVRAM")

if(XVRAM_BUILD_TESTS)
  add_subdirectory(tests)
endif()
