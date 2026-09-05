add_test(NAME xvram.cuda-compat.export-contract
         COMMAND "${Python3_EXECUTABLE}"
                 "${CMAKE_CURRENT_SOURCE_DIR}/contract/cuda_compat_export_contract.py"
                 "$<TARGET_FILE:xvram_cuda_compat>")
add_executable(xvram-cuda-compat-tests unit/cuda_compat_tests.cpp)
target_link_libraries(xvram-cuda-compat-tests PRIVATE xvram_cuda_compat_core)
xvram_enable_warnings(xvram-cuda-compat-tests)
add_test(NAME xvram.cuda-compat.adapter-fake COMMAND xvram-cuda-compat-tests)
add_test(NAME xvram.cuda-compat.export-parser
         COMMAND "${Python3_EXECUTABLE}"
                 "${CMAKE_CURRENT_SOURCE_DIR}/contract/cuda_compat_export_contract.py" --self-test)

add_test(NAME xvram.cuda-compat.consumer-profile
         COMMAND "${Python3_EXECUTABLE}"
                 "${CMAKE_CURRENT_SOURCE_DIR}/contract/cuda_compat_consumer_contract.py"
                 --cmake "${CMAKE_COMMAND}"
                 --source "${CMAKE_CURRENT_SOURCE_DIR}/compat_consumer"
                 --library "$<TARGET_FILE:xvram_cuda_compat>"
                 --link-library "$<TARGET_LINKER_FILE:xvram_cuda_compat>"
                 --include "${PROJECT_SOURCE_DIR}/include"
                 --cuda-include "${XVRAM_CUDA_INCLUDE_DIR}"
                 --cublas-include "${XVRAM_CUBLAS_INCLUDE_DIR}"
                 --crt-include "${XVRAM_CUDA_CRT_HEADERS_DIR}"
                 --generator "${CMAKE_GENERATOR}"
                 --platform "${CMAKE_GENERATOR_PLATFORM}"
                 --make-program "${CMAKE_MAKE_PROGRAM}"
                 --compiler "${CMAKE_CXX_COMPILER}"
                 "--compiler-flags=${CMAKE_CXX_FLAGS}"
                 "--linker-flags=${CMAKE_EXE_LINKER_FLAGS}"
                 --configuration "$<CONFIG>"
                 --binary-root "${CMAKE_CURRENT_BINARY_DIR}/compat-consumers")
set_tests_properties(xvram.cuda-compat.consumer-profile PROPERTIES TIMEOUT 180 LABELS "no-driver")

add_executable(xvram-cuda-compat-load-helper helpers/cuda_compat_load_helper.cpp)
target_link_libraries(xvram-cuda-compat-load-helper PRIVATE xvram_gemm_core)
target_compile_features(xvram-cuda-compat-load-helper PRIVATE cxx_std_20)
xvram_enable_warnings(xvram-cuda-compat-load-helper)
