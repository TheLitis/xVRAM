cmake_minimum_required(VERSION 3.25)

project(xvram_package_consumer LANGUAGES C CXX)

include(CTest)
find_package(xVRAM CONFIG REQUIRED)

# Only the explicit compat facade supplements the installed C ABI headers. Internal C++
# runtime/report models must remain private to the source tree.
get_target_property(xvram_package_include_roots xVRAM::xvram INTERFACE_INCLUDE_DIRECTORIES)
set(xvram_package_header_root_found FALSE)
set(xvram_package_expected_headers "torch_allocator.h;xvram.h;xvram_v2.h")
if(TARGET xVRAM::cuda_compat)
  list(APPEND xvram_package_expected_headers cuda_compat.h cuda_compat.hpp)
  list(SORT xvram_package_expected_headers)
endif()
foreach(xvram_package_include_root IN LISTS xvram_package_include_roots)
  if(EXISTS "${xvram_package_include_root}/xvram/xvram.h")
    set(xvram_package_header_root_found TRUE)
    file(GLOB_RECURSE xvram_package_installed_headers LIST_DIRECTORIES FALSE
         RELATIVE "${xvram_package_include_root}/xvram"
         "${xvram_package_include_root}/xvram/*.h"
         "${xvram_package_include_root}/xvram/*.hpp")
    list(SORT xvram_package_installed_headers)
    if(NOT xvram_package_installed_headers STREQUAL xvram_package_expected_headers)
      message(FATAL_ERROR
              "Unexpected installed xVRAM headers in ${xvram_package_include_root}/xvram: "
              "expected [${xvram_package_expected_headers}], "
              "found [${xvram_package_installed_headers}]")
    endif()
  endif()
endforeach()
if(NOT xvram_package_header_root_found)
  message(FATAL_ERROR "xVRAM::xvram has no installed include root containing xvram/xvram.h")
endif()

add_executable(xvram-package-consumer-c main.c)
target_link_libraries(xvram-package-consumer-c PRIVATE xVRAM::xvram)
add_test(NAME xvram.package.consumer-c COMMAND xvram-package-consumer-c)

if(TARGET xVRAM::cuda_compat)
  add_executable(xvram-package-consumer-cuda-compat cuda_compat.c)
  target_compile_features(xvram-package-consumer-cuda-compat PRIVATE c_std_11)
  target_link_libraries(xvram-package-consumer-cuda-compat PRIVATE xVRAM::cuda_compat)
  add_test(NAME xvram.package.consumer-cuda-compat COMMAND xvram-package-consumer-cuda-compat)

  # NVIDIA headers are an application build dependency, not part of the xVRAM package.
  if(XVRAM_BUILD_TREE)
    include("${XVRAM_BUILD_TREE}/compat-consumer-dependencies.cmake")
  endif()
  if(NOT dependency_XVRAM_CUDA_INCLUDE_DIR OR NOT dependency_XVRAM_CUBLAS_INCLUDE_DIR)
    message(FATAL_ERROR "Set XVRAM_BUILD_TREE to the configured build supplying official NVIDIA headers")
  endif()
  add_executable(xvram-package-consumer-cuda-facade ../compat_consumer/main.cpp)
  target_link_libraries(xvram-package-consumer-cuda-facade PRIVATE xVRAM::cuda_compat)
  target_include_directories(xvram-package-consumer-cuda-facade SYSTEM PRIVATE
    "${dependency_XVRAM_CUDA_INCLUDE_DIR}" "${dependency_XVRAM_CUBLAS_INCLUDE_DIR}"
    "${dependency_XVRAM_CUDA_CRT_HEADERS_DIR}" "${dependency_XVRAM_CUDA_CCCL_HEADERS_DIR}")
  if(MSVC)
    target_compile_options(xvram-package-consumer-cuda-facade PRIVATE /W4 /WX /permissive-)
  else()
    target_compile_options(xvram-package-consumer-cuda-facade PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
  add_test(NAME xvram.package.consumer-cuda-facade COMMAND xvram-package-consumer-cuda-facade)
endif()

add_executable(xvram-package-consumer-sdk-v2-c sdk_v2.c)
target_compile_features(xvram-package-consumer-sdk-v2-c PRIVATE c_std_11)
target_link_libraries(xvram-package-consumer-sdk-v2-c PRIVATE xVRAM::xvram)
add_test(NAME xvram.package.consumer-sdk-v2-c COMMAND xvram-package-consumer-sdk-v2-c)
set_tests_properties(xvram.package.consumer-sdk-v2-c PROPERTIES LABELS "installed-package;sdk-v2")

add_executable(xvram-package-consumer-cxx main.cpp)
target_compile_features(xvram-package-consumer-cxx PRIVATE cxx_std_20)
target_link_libraries(xvram-package-consumer-cxx PRIVATE xVRAM::xvram)
add_test(NAME xvram.package.consumer-cxx COMMAND xvram-package-consumer-cxx)

add_executable(xvram-package-consumer-torch-allocator torch_allocator.c)
target_link_libraries(xvram-package-consumer-torch-allocator PRIVATE xVRAM::torch_allocator)
add_test(NAME xvram.package.consumer-torch-allocator
         COMMAND xvram-package-consumer-torch-allocator)
