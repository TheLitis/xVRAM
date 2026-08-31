cmake_minimum_required(VERSION 3.25)

project(xvram_package_consumer LANGUAGES C CXX)

include(CTest)
find_package(xVRAM CONFIG REQUIRED)

add_executable(xvram-package-consumer-c main.c)
target_link_libraries(xvram-package-consumer-c PRIVATE xVRAM::xvram)
add_test(NAME xvram.package.consumer-c COMMAND xvram-package-consumer-c)

add_executable(xvram-package-consumer-cxx main.cpp)
target_compile_features(xvram-package-consumer-cxx PRIVATE cxx_std_20)
target_link_libraries(xvram-package-consumer-cxx PRIVATE xVRAM::xvram)
add_test(NAME xvram.package.consumer-cxx COMMAND xvram-package-consumer-cxx)
