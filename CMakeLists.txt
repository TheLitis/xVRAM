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
include(FetchContent)
include(GNUInstallDirs)

option(XVRAM_BUILD_TESTS "Build xVRAM tests" ${BUILD_TESTING})
option(XVRAM_FETCH_CUDA_HEADERS
       "Download NVIDIA's official CUDA redistributable headers when no Toolkit is installed" ON)
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

install(TARGETS xvram-probe RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(FILES schemas/capability-report-v1.schema.json
              schemas/capability-report-v2.schema.json
        DESTINATION "${CMAKE_INSTALL_DATADIR}/xvram/schemas")

if(XVRAM_BUILD_TESTS)
  add_subdirectory(tests)
endif()
