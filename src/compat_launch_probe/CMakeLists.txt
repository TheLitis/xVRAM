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
target_include_directories(xvram_launch_probe SYSTEM PRIVATE "${XVRAM_PROBE_CUDA_INCLUDE}")
target_compile_definitions(xvram_launch_probe PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
target_compile_options(xvram_launch_probe PRIVATE /W4 /WX /permissive- /EHsc)
target_link_libraries(xvram_launch_probe PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/xvram_native.lib")
set_target_properties(xvram_launch_probe PROPERTIES OUTPUT_NAME cudart64_13 CXX_EXTENSIONS OFF)
endif()
# Intentionally no install rule and no production target/export changes.
