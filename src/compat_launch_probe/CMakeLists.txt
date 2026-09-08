cmake_minimum_required(VERSION 3.25)
project(xvram_compat_launch_probe LANGUAGES CXX)
if(NOT WIN32 OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR "The opt-in launch proxy profile requires Windows x64")
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_path(XVRAM_PROBE_CUDA_INCLUDE cuda.h HINTS "$ENV{CUDA_PATH}" PATH_SUFFIXES include REQUIRED)
set(XVRAM_PROBE_NATIVE_CUDART "" CACHE FILEPATH "Exact pinned llama.cpp b10819 cudart64_13.dll")
option(XVRAM_PROBE_BUILD_RUNTIME_PROXY "Build the separate negative-control Runtime proxy" OFF)
include(FetchContent)
FetchContent_Declare(probe_detours
  URL https://github.com/microsoft/Detours/archive/e4bfd6b03e50de46b47abfbd1e46b384f0c5f833.zip
  URL_HASH SHA256=fef0b17c5c3c9356fb8ba7366a3b196149daadd131fb6a92abeda1cb4d3b9b85
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(probe_detours)
add_library(probe_detours STATIC
  "${probe_detours_SOURCE_DIR}/src/detours.cpp"
  "${probe_detours_SOURCE_DIR}/src/modules.cpp"
  "${probe_detours_SOURCE_DIR}/src/disasm.cpp"
  "${probe_detours_SOURCE_DIR}/src/image.cpp"
  "${probe_detours_SOURCE_DIR}/src/creatwth.cpp"
  "${probe_detours_SOURCE_DIR}/src/disolx86.cpp"
  "${probe_detours_SOURCE_DIR}/src/disolx64.cpp"
  "${probe_detours_SOURCE_DIR}/src/disolia64.cpp"
  "${probe_detours_SOURCE_DIR}/src/disolarm.cpp"
  "${probe_detours_SOURCE_DIR}/src/disolarm64.cpp")
target_compile_options(probe_detours PRIVATE /W0)
target_compile_definitions(probe_detours PRIVATE WIN32_LEAN_AND_MEAN)
add_library(xvram_driver_launch_probe SHARED driver_probe.cpp ../platform/dynamic_library.cpp ../platform/sha256.cpp)
target_compile_features(xvram_driver_launch_probe PRIVATE cxx_std_20)
target_include_directories(xvram_driver_launch_probe PRIVATE ..)
target_include_directories(xvram_driver_launch_probe SYSTEM PRIVATE "${XVRAM_PROBE_CUDA_INCLUDE}" "${probe_detours_SOURCE_DIR}/src")
target_compile_definitions(xvram_driver_launch_probe PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
target_compile_options(xvram_driver_launch_probe PRIVATE /W4 /WX /permissive- /EHsc)
target_link_libraries(xvram_driver_launch_probe PRIVATE probe_detours)
set_target_properties(xvram_driver_launch_probe PROPERTIES CXX_EXTENSIONS OFF)
option(XVRAM_PROBE_CENSUS "Add opt-in simultaneous CUPTI census (requires official CUPTI 13.3)" OFF)
option(XVRAM_PROBE_LEGACY_RESOLVER "Observe the distinct public legacy resolver entry (trace v2)" OFF)
option(XVRAM_PROBE_WITNESS "Collect separate live library/module identity witnesses" OFF)
option(XVRAM_PROBE_PC_WITNESS "Experimental serialized PC identity diagnostic; not execution proof" OFF)
option(XVRAM_PROBE_MEMORY_WITNESS "Observe device allocation/VMM generations; not tensor memory proof" OFF)
option(XVRAM_PROBE_KERNEL_CAPTURE "Capture exact catalog-typed host parameter banks without tensor-bound inference" OFF)
option(XVRAM_PROBE_TEARDOWN_WITNESS "Retain typed library/context lifecycle observations through native teardown" OFF)
if(XVRAM_PROBE_TEARDOWN_WITNESS)
  if(NOT XVRAM_PROBE_CENSUS OR NOT XVRAM_PROBE_PC_WITNESS)
    message(FATAL_ERROR "Teardown witness requires the simultaneous census and execution witness")
  endif()
  target_sources(xvram_driver_launch_probe PRIVATE teardown_witness.cpp)
  target_compile_definitions(xvram_driver_launch_probe PRIVATE XVRAM_PROBE_TEARDOWN_WITNESS)
endif()
if(XVRAM_PROBE_KERNEL_CAPTURE)
  if(NOT XVRAM_PROBE_PC_WITNESS OR NOT XVRAM_PROBE_MEMORY_WITNESS)
    message(FATAL_ERROR "Typed kernel capture requires PC and memory witness profiles")
  endif()
  set(probe_capture_header "${CMAKE_CURRENT_BINARY_DIR}/audit-generated/capture_catalog.hpp")
  add_custom_command(OUTPUT "${probe_capture_header}"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/generate-audit-kernel-header.py"
            --catalog "${CMAKE_CURRENT_SOURCE_DIR}/../../python/xvram/compat_audit_kernel_capture_catalog.json"
            --output "${probe_capture_header}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/generate-audit-kernel-header.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/../../python/xvram/compat_audit_kernel_capture_catalog.json"
            "${CMAKE_CURRENT_SOURCE_DIR}/../../python/xvram/compat_audit_kernel_catalog.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/../../python/xvram/compat_audit_kernel_ranges.py"
    VERBATIM)
  target_sources(xvram_driver_launch_probe PRIVATE kernel_capture.cpp "${probe_capture_header}")
  target_include_directories(xvram_driver_launch_probe PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/audit-generated")
  target_compile_definitions(xvram_driver_launch_probe PRIVATE XVRAM_PROBE_KERNEL_CAPTURE)
endif()
if(XVRAM_PROBE_MEMORY_WITNESS)
  if(NOT XVRAM_PROBE_CENSUS)
    message(FATAL_ERROR "Memory witness requires the simultaneous Driver API census")
  endif()
  target_sources(xvram_driver_launch_probe PRIVATE memory_witness.cpp)
  target_compile_definitions(xvram_driver_launch_probe PRIVATE XVRAM_PROBE_MEMORY_WITNESS)
endif()
if(XVRAM_PROBE_PC_WITNESS)
  if(NOT XVRAM_PROBE_WITNESS)
    message(FATAL_ERROR "PC witness requires the identity witness profile")
  endif()
  target_sources(xvram_driver_launch_probe PRIVATE sampling.cpp execution_witness.cpp postmortem.cpp)
  target_compile_definitions(xvram_driver_launch_probe PRIVATE XVRAM_PROBE_PC_WITNESS)
endif()
if(XVRAM_PROBE_WITNESS)
  if(NOT XVRAM_PROBE_LEGACY_RESOLVER OR NOT XVRAM_PROBE_CENSUS)
    message(FATAL_ERROR "Identity witness requires legacy resolver and census")
  endif()
  target_sources(xvram_driver_launch_probe PRIVATE witness.cpp)
  target_compile_definitions(xvram_driver_launch_probe PRIVATE XVRAM_PROBE_WITNESS)
endif()
if(XVRAM_PROBE_LEGACY_RESOLVER)
  if(NOT XVRAM_PROBE_CENSUS)
    message(FATAL_ERROR "Legacy resolver observation requires the simultaneous census")
  endif()
  target_compile_definitions(xvram_driver_launch_probe PRIVATE XVRAM_PROBE_LEGACY_RESOLVER)
endif()
option(XVRAM_PROBE_FETCH_CUPTI "Fetch hash-pinned app-local CUPTI for the diagnostic only" OFF)
if(XVRAM_PROBE_CENSUS OR XVRAM_PROBE_BUILD_RUNTIME_PROXY)
  find_path(XVRAM_PROBE_CUDA_CRT_INCLUDE crt/host_defines.h HINTS "${XVRAM_PROBE_CUDA_INCLUDE}")
  if(NOT XVRAM_PROBE_CUDA_CRT_INCLUDE AND XVRAM_PROBE_FETCH_CUPTI)
    FetchContent_Declare(probe_cuda_crt
      URL https://developer.download.nvidia.com/compute/cuda/redist/cuda_crt/windows-x86_64/cuda_crt-windows-x86_64-13.3.33-archive.zip
      URL_HASH SHA256=752c528281a06a0ddf89237d760ffd6acde1b9cd59efc35803c2591127ef55f0
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(probe_cuda_crt)
    find_path(XVRAM_PROBE_CUDA_CRT_INCLUDE crt/host_defines.h PATHS "${probe_cuda_crt_SOURCE_DIR}/include" NO_DEFAULT_PATH REQUIRED)
  elseif(NOT XVRAM_PROBE_CUDA_CRT_INCLUDE)
    message(FATAL_ERROR "Set XVRAM_PROBE_CUDA_CRT_INCLUDE to the matching CUDA CRT header root")
  endif()
endif()
if(XVRAM_PROBE_CENSUS)
  if(XVRAM_PROBE_FETCH_CUPTI)
    FetchContent_Declare(probe_cupti
      URL https://developer.download.nvidia.com/compute/cuda/redist/cuda_cupti/windows-x86_64/cuda_cupti-windows-x86_64-13.3.75-archive.zip
      URL_HASH SHA256=298eb0dfbd7eb63b63aab0533b4ba1ab2542c2ea03842a744973c7f38d0982b3
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(probe_cupti)
    set(probe_cupti_root "${probe_cupti_SOURCE_DIR}")
  else()
    set(probe_cupti_root "$ENV{CUDA_PATH}/extras/CUPTI")
  endif()
  find_path(XVRAM_PROBE_CUPTI_INCLUDE cupti.h PATHS "${probe_cupti_root}/include" NO_DEFAULT_PATH REQUIRED)
  find_library(XVRAM_PROBE_CUPTI_LIBRARY cupti PATHS "${probe_cupti_root}/lib64" "${probe_cupti_root}/lib" "${probe_cupti_root}/lib/x64" NO_DEFAULT_PATH REQUIRED)
  find_file(XVRAM_PROBE_CUPTI_DLL cupti64_2026.2.1.dll PATHS "${probe_cupti_root}/lib64" "${probe_cupti_root}/lib" "${probe_cupti_root}/bin" NO_DEFAULT_PATH REQUIRED)
  file(SHA256 "${XVRAM_PROBE_CUPTI_DLL}" probe_cupti_hash)
  if(NOT probe_cupti_hash STREQUAL "9b10d2fafaff1a4dc9e447c4a1355fccb04ee024fa7e7d28c9e4c5ab53347faf")
    message(FATAL_ERROR "CUPTI diagnostic runtime hash mismatch")
  endif()
  target_sources(xvram_driver_launch_probe PRIVATE census.cpp)
  target_include_directories(xvram_driver_launch_probe SYSTEM PRIVATE "${XVRAM_PROBE_CUPTI_INCLUDE}" "${XVRAM_PROBE_CUDA_CRT_INCLUDE}")
  target_link_libraries(xvram_driver_launch_probe PRIVATE "${XVRAM_PROBE_CUPTI_LIBRARY}")
  target_compile_definitions(xvram_driver_launch_probe PRIVATE XVRAM_PROBE_CENSUS)
  set_target_properties(xvram_driver_launch_probe PROPERTIES OUTPUT_NAME xvram_driver_launch_census)
  add_custom_command(TARGET xvram_driver_launch_probe POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${XVRAM_PROBE_CUPTI_DLL}" "$<TARGET_FILE_DIR:xvram_driver_launch_probe>")
  if(XVRAM_PROBE_WITNESS)
    foreach(dependency IN ITEMS nvperf_host nvperf_target)
      find_file(probe_${dependency}_dll "${dependency}.dll" PATHS "${probe_cupti_root}/lib"
        "${probe_cupti_root}/lib64" NO_DEFAULT_PATH REQUIRED)
      file(SHA256 "${probe_${dependency}_dll}" dependency_hash)
      if(dependency STREQUAL "nvperf_host")
        set(expected_hash cde4143748734948838ede61b350db59a0d7e6ac6ef451e375a5702d85174a9b)
      else()
        set(expected_hash 049f858e512592e895b5789d0872f44ee7cbb57d3e4b8936303f4d7eb5b2b1a7)
      endif()
      if(NOT dependency_hash STREQUAL expected_hash)
        message(FATAL_ERROR "PC sampling dependency hash mismatch")
      endif()
    endforeach()
    add_executable(xvram-pc-sampling-probe pc_sampling_probe.cpp ../platform/dynamic_library.cpp ../platform/sha256.cpp)
    target_compile_features(xvram-pc-sampling-probe PRIVATE cxx_std_20)
    target_include_directories(xvram-pc-sampling-probe PRIVATE ..)
    target_include_directories(xvram-pc-sampling-probe SYSTEM PRIVATE "${XVRAM_PROBE_CUDA_INCLUDE}"
      "${XVRAM_PROBE_CUPTI_INCLUDE}" "${XVRAM_PROBE_CUDA_CRT_INCLUDE}")
    target_compile_definitions(xvram-pc-sampling-probe PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(xvram-pc-sampling-probe PRIVATE /W4 /WX /permissive- /EHsc)
    target_link_libraries(xvram-pc-sampling-probe PRIVATE "${XVRAM_PROBE_CUPTI_LIBRARY}")
    add_custom_command(TARGET xvram-pc-sampling-probe POST_BUILD
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${XVRAM_PROBE_CUPTI_DLL}" "$<TARGET_FILE_DIR:xvram-pc-sampling-probe>")
    foreach(dependency IN ITEMS nvperf_host nvperf_target)
      add_custom_command(TARGET xvram-pc-sampling-probe POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${probe_${dependency}_dll}" "$<TARGET_FILE_DIR:xvram-pc-sampling-probe>")
    endforeach()
  endif()
endif()
configure_file("${probe_detours_SOURCE_DIR}/LICENSE.md" "${CMAKE_CURRENT_BINARY_DIR}/Detours-LICENSE.md" COPYONLY)
if(XVRAM_PROBE_BUILD_RUNTIME_PROXY)
set(probe_def "${CMAKE_CURRENT_BINARY_DIR}/cudart.def")
set(native_def "${CMAKE_CURRENT_BINARY_DIR}/native.def")
# Regeneration is a build artifact operation, never a source/binary replacement.
if(EXISTS "${probe_def}")
  file(REMOVE "${probe_def}")
endif()
if(EXISTS "${native_def}")
  file(REMOVE "${native_def}")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../../python"
  "${Python3_EXECUTABLE}" -m xvram.compat_launch_probe exports --native "${XVRAM_PROBE_NATIVE_CUDART}"
  --output "${probe_def}" COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../../python"
  "${Python3_EXECUTABLE}" -m xvram.compat_launch_probe exports --native "${XVRAM_PROBE_NATIVE_CUDART}"
  --output "${native_def}" --import-definition COMMAND_ERROR_IS_FATAL ANY)
# LINK validates forwarded export symbols against an import library. Its module
# name MUST be the staged native name, never the proxy's own cudart64_13 name.
execute_process(COMMAND "${CMAKE_AR}" /nologo /machine:x64 "/def:${native_def}"
  "/out:${CMAKE_CURRENT_BINARY_DIR}/xvram_native.lib" COMMAND_ERROR_IS_FATAL ANY)
add_library(xvram_launch_probe SHARED proxy.cpp ../platform/dynamic_library.cpp ../platform/sha256.cpp "${probe_def}")
target_compile_features(xvram_launch_probe PRIVATE cxx_std_20)
target_include_directories(xvram_launch_probe PRIVATE ..)
target_include_directories(xvram_launch_probe SYSTEM PRIVATE "${XVRAM_PROBE_CUDA_INCLUDE}" "${XVRAM_PROBE_CUDA_CRT_INCLUDE}")
target_compile_definitions(xvram_launch_probe PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
target_compile_options(xvram_launch_probe PRIVATE /W4 /WX /permissive- /EHsc)
target_link_libraries(xvram_launch_probe PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/xvram_native.lib")
set_target_properties(xvram_launch_probe PROPERTIES OUTPUT_NAME cudart64_13 CXX_EXTENSIONS OFF)
endif()
# Intentionally no install rule and no production target/export changes.
