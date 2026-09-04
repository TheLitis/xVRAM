cmake_minimum_required(VERSION 3.25)

project(xvram_package_consumer LANGUAGES C CXX)

include(CTest)
find_package(xVRAM CONFIG REQUIRED)

# The installed SDK exposes only its C ABI headers. Internal C++ report models stay in the
# source tree; accidentally installing one would suggest an unsupported public C++ surface.
get_target_property(xvram_package_include_roots xVRAM::xvram INTERFACE_INCLUDE_DIRECTORIES)
set(xvram_package_header_root_found FALSE)
set(xvram_package_expected_headers "torch_allocator.h;xvram.h;xvram_v2.h")
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
