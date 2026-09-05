# Explicit host-only integration. Never link cudart, cuBLAS, or the Driver into consumers.
find_path(XVRAM_CUBLAS_INCLUDE_DIR NAMES cublas_v2.h
          HINTS "${XVRAM_CUDA_INCLUDE_DIR}" "${XVRAM_CUBLAS_REDIST_ROOT}" "$ENV{CUDA_PATH}"
          PATH_SUFFIXES include)
if(NOT XVRAM_CUBLAS_INCLUDE_DIR)
  message(FATAL_ERROR
          "CUDA compatibility requires official cuBLAS headers. Set XVRAM_CUBLAS_INCLUDE_DIR "
          "to an installed Toolkit include directory or enable XVRAM_FETCH_CUBLAS_REDIST.")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/CudaCompatHeaders.cmake")

file(GENERATE OUTPUT "${PROJECT_BINARY_DIR}/compat-consumer-dependencies.cmake"
     CONTENT "set(dependency_XVRAM_CUDA_INCLUDE_DIR [==[${XVRAM_CUDA_INCLUDE_DIR}]==])\nset(dependency_XVRAM_CUBLAS_INCLUDE_DIR [==[${XVRAM_CUBLAS_INCLUDE_DIR}]==])\nset(dependency_XVRAM_CUDA_CRT_HEADERS_DIR [==[${XVRAM_CUDA_CRT_HEADERS_DIR}]==])\nset(dependency_XVRAM_CUDA_CCCL_HEADERS_DIR [==[${XVRAM_CUDA_CCCL_HEADERS_DIR}]==])\n")

add_library(xvram_cuda_compat_core STATIC
            src/cuda_compat/adapter.cpp
            src/cuda_compat/runtime_backend.cpp
            src/sdk/status.cpp)
target_compile_features(xvram_cuda_compat_core PUBLIC cxx_std_20)
target_include_directories(xvram_cuda_compat_core PUBLIC "${PROJECT_SOURCE_DIR}/include"
                                                       "${PROJECT_SOURCE_DIR}/src")
target_include_directories(xvram_cuda_compat_core SYSTEM PRIVATE "${XVRAM_CUDA_INCLUDE_DIR}")
target_link_libraries(xvram_cuda_compat_core PRIVATE xvram_gemm_core xvram_residency_core
                                                  xvram_probe_lib Threads::Threads)
set_target_properties(xvram_cuda_compat_core PROPERTIES CXX_VISIBILITY_PRESET hidden
                                                      VISIBILITY_INLINES_HIDDEN YES)
xvram_enable_warnings(xvram_cuda_compat_core)

add_library(xvram_cuda_compat SHARED src/cuda_compat/api.cpp)
set_target_properties(xvram_cuda_compat PROPERTIES OUTPUT_NAME xvram_cuda_compat
                                                 EXPORT_NAME cuda_compat
                                                 VERSION "${PROJECT_VERSION}" SOVERSION 1
                                                 CXX_VISIBILITY_PRESET hidden
                                                 VISIBILITY_INLINES_HIDDEN YES)
target_compile_definitions(xvram_cuda_compat PRIVATE XVRAM_BUILDING_CUDA_COMPAT
                                            INTERFACE XVRAM_USING_CUDA_COMPAT)
target_compile_features(xvram_cuda_compat PUBLIC cxx_std_20)
target_include_directories(xvram_cuda_compat PUBLIC
                           "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
                           "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_include_directories(xvram_cuda_compat PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(xvram_cuda_compat PRIVATE xvram_cuda_compat_core)
if(UNIX AND NOT APPLE)
  target_link_options(xvram_cuda_compat PRIVATE
                     "LINKER:--version-script=${PROJECT_SOURCE_DIR}/cmake/cuda_compat.exports.map")
  set_property(TARGET xvram_cuda_compat APPEND PROPERTY LINK_DEPENDS
               "${PROJECT_SOURCE_DIR}/cmake/cuda_compat.exports.map")
endif()
xvram_enable_warnings(xvram_cuda_compat)
add_library(xVRAM::cuda_compat ALIAS xvram_cuda_compat)

if(XVRAM_CUBLAS_REDIST_ROOT)
  file(SHA256 "${XVRAM_CUBLAS_CORE_LIBRARY}" XVRAM_COMPAT_CORE_SHA256)
  file(SHA256 "${XVRAM_CUBLAS_LT_LIBRARY}" XVRAM_COMPAT_LT_SHA256)
  get_filename_component(XVRAM_COMPAT_CORE_NAME "${XVRAM_CUBLAS_CORE_LIBRARY}" NAME)
  get_filename_component(XVRAM_COMPAT_LT_NAME "${XVRAM_CUBLAS_LT_LIBRARY}" NAME)
  file(GENERATE OUTPUT "${PROJECT_BINARY_DIR}/compat-cublas-provenance.json"
       CONTENT "{\"library_directory\":\"${XVRAM_CUBLAS_INSTALL_DESTINATION}\",\"core_name\":\"${XVRAM_COMPAT_CORE_NAME}\",\"core_sha256\":\"${XVRAM_COMPAT_CORE_SHA256}\",\"lt_name\":\"${XVRAM_COMPAT_LT_NAME}\",\"lt_sha256\":\"${XVRAM_COMPAT_LT_SHA256}\"}\n")
  add_custom_command(TARGET xvram_cuda_compat POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${XVRAM_CUBLAS_CORE_LIBRARY}"
            "$<TARGET_FILE_DIR:xvram_cuda_compat>"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${XVRAM_CUBLAS_LT_LIBRARY}"
            "$<TARGET_FILE_DIR:xvram_cuda_compat>"
    VERBATIM)
endif()

install(TARGETS xvram_cuda_compat EXPORT xVRAMTargets
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(FILES include/xvram/cuda_compat.h include/xvram/cuda_compat.hpp
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/xvram")

add_library(xvram_compat_bench STATIC
            src/compat_bench/options.cpp src/compat_bench/pattern.cpp
            src/compat_bench/final_validation.cpp
            src/compat_bench/report.cpp src/compat_bench/executor.cpp
            src/compat_bench/native_baseline.cpp src/compat_bench/worker_protocol.cpp
            src/compat_bench/worker_controller.cpp)
target_compile_features(xvram_compat_bench PUBLIC cxx_std_20)
target_include_directories(xvram_compat_bench PUBLIC "${PROJECT_SOURCE_DIR}/src"
                                                  "${PROJECT_SOURCE_DIR}/include")
target_include_directories(xvram_compat_bench SYSTEM PUBLIC "${XVRAM_CUDA_INCLUDE_DIR}"
                                                        "${XVRAM_CUDA_CRT_HEADERS_DIR}"
                                                        "${XVRAM_CUDA_CCCL_HEADERS_DIR}"
                                                        "${XVRAM_CUBLAS_INCLUDE_DIR}")
target_link_libraries(xvram_compat_bench PUBLIC xvram_cuda_compat xvram_probe_lib
                                      PRIVATE xvram_gemm_core xvram_worker_process Threads::Threads)
xvram_enable_warnings(xvram_compat_bench)
add_executable(xvram-compat-bench src/compat_bench/main.cpp)
target_link_libraries(xvram-compat-bench PRIVATE xvram_compat_bench)
if(UNIX AND NOT APPLE)
  set_target_properties(xvram-compat-bench PROPERTIES INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif()
xvram_enable_warnings(xvram-compat-bench)
install(TARGETS xvram-compat-bench RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(FILES schemas/cuda-compat-v1.schema.json schemas/cuda-compat-trace-v1.schema.json
        DESTINATION "${CMAKE_INSTALL_DATADIR}/xvram/schemas")
