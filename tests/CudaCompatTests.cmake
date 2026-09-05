add_test(NAME xvram.cuda-compat.export-contract
         COMMAND "${Python3_EXECUTABLE}"
                 "${CMAKE_CURRENT_SOURCE_DIR}/contract/cuda_compat_export_contract.py"
                 "$<TARGET_FILE:xvram_cuda_compat>")
add_executable(xvram-cuda-compat-tests unit/cuda_compat_tests.cpp)
target_link_libraries(xvram-cuda-compat-tests PRIVATE xvram_cuda_compat_core)
xvram_enable_warnings(xvram-cuda-compat-tests)
add_test(NAME xvram.cuda-compat.adapter-fake COMMAND xvram-cuda-compat-tests)
add_executable(xvram-cuda-compat-facade-tests unit/cuda_compat_facade_tests.cpp)
target_compile_features(xvram-cuda-compat-facade-tests PRIVATE cxx_std_20)
target_include_directories(xvram-cuda-compat-facade-tests PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_include_directories(xvram-cuda-compat-facade-tests SYSTEM PRIVATE
                          "${XVRAM_CUDA_INCLUDE_DIR}" "${XVRAM_CUBLAS_INCLUDE_DIR}"
                          "${XVRAM_CUDA_CRT_HEADERS_DIR}")
target_link_libraries(xvram-cuda-compat-facade-tests PRIVATE Threads::Threads)
xvram_enable_warnings(xvram-cuda-compat-facade-tests)
add_test(NAME xvram.cuda-compat.facade-abi-rejection COMMAND xvram-cuda-compat-facade-tests)
add_test(NAME xvram.cuda-compat.facade-missing-members
         COMMAND xvram-cuda-compat-facade-tests --missing-members)
add_test(NAME xvram.cuda-compat.facade-wrong-version
         COMMAND xvram-cuda-compat-facade-tests --wrong-version)
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

add_executable(xvram-compat-core-tests helpers/compat_core_tests.cpp)
target_link_libraries(xvram-compat-core-tests PRIVATE xvram_compat_bench)
xvram_enable_warnings(xvram-compat-core-tests)
add_test(NAME xvram.cuda-compat.core COMMAND xvram-compat-core-tests)

add_executable(xvram-compat-worker-helper helpers/compat_worker_test_helper.cpp)
target_link_libraries(xvram-compat-worker-helper PRIVATE xvram_compat_bench)
xvram_enable_warnings(xvram-compat-worker-helper)
add_executable(xvram-compat-controller-tests helpers/compat_controller_tests.cpp)
target_link_libraries(xvram-compat-controller-tests PRIVATE xvram_compat_bench)
xvram_enable_warnings(xvram-compat-controller-tests)
add_test(NAME xvram.cuda-compat.controller
         COMMAND xvram-compat-controller-tests "$<TARGET_FILE:xvram-compat-worker-helper>")
set_tests_properties(xvram.cuda-compat.controller PROPERTIES TIMEOUT 30 LABELS "no-driver")

add_test(NAME xvram.cuda-compat.json-contract
         COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/contract/compat_contract.py"
                 --schema "${PROJECT_SOURCE_DIR}/schemas/cuda-compat-v1.schema.json"
                 --trace-schema "${PROJECT_SOURCE_DIR}/schemas/cuda-compat-trace-v1.schema.json"
                 --fixture-helper "$<TARGET_FILE:xvram-compat-core-tests>"
                 --bench "$<TARGET_FILE:xvram-compat-bench>")
set_tests_properties(xvram.cuda-compat.json-contract PROPERTIES TIMEOUT 120 LABELS "no-driver")
