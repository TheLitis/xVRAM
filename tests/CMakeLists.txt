find_package(Python3 COMPONENTS Interpreter REQUIRED)

if(XVRAM_BUILD_CUDA_COMPAT)
  include(CudaCompatTests.cmake)
endif()

if(WIN32)
  find_program(XVRAM_POWERSHELL_EXECUTABLE NAMES pwsh powershell REQUIRED)
  add_test(
    NAME xvram.acceptance.provenance
    COMMAND "${XVRAM_POWERSHELL_EXECUTABLE}" -NoLogo -NoProfile -NonInteractive
            -ExecutionPolicy Bypass -File
            "${CMAKE_CURRENT_SOURCE_DIR}/contract/acceptance_provenance_tests.ps1"
            -Helper "${PROJECT_SOURCE_DIR}/scripts/acceptance-provenance.ps1")
  set_tests_properties(xvram.acceptance.provenance PROPERTIES TIMEOUT 30 LABELS "no-driver")
endif()

add_executable(xvram-unit-tests unit/base_tests.cpp)
target_link_libraries(xvram-unit-tests PRIVATE xvram_probe_lib)
xvram_enable_warnings(xvram-unit-tests)

add_test(NAME xvram.unit COMMAND xvram-unit-tests)

add_executable(xvram-torch-allocator-tests unit/torch_allocator_tests.cpp)
target_link_libraries(xvram-torch-allocator-tests PRIVATE xvram_torch_allocator_core)
target_include_directories(xvram-torch-allocator-tests PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_include_directories(xvram-torch-allocator-tests SYSTEM PRIVATE "${XVRAM_CUDA_INCLUDE_DIR}")
target_compile_features(xvram-torch-allocator-tests PRIVATE cxx_std_20)
xvram_enable_warnings(xvram-torch-allocator-tests)
add_test(NAME xvram.torch-allocator.unit COMMAND xvram-torch-allocator-tests)

add_test(
  NAME xvram.torch-allocator.export-contract
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/contract/torch_allocator_export_contract.py"
          "$<TARGET_FILE:xvram_torch_allocator>")
add_test(
  NAME xvram.torch-allocator.export-parser
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/contract/torch_allocator_export_contract.py" --self-test)

add_test(
  NAME xvram.torch-runtime.export-parser
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/contract/torch_runtime_export_contract.py" --self-test)
if(TARGET xvram_torch_runtime)
  add_executable(
    xvram-torch-runtime-control-tests
    "${PROJECT_SOURCE_DIR}/src/torch_runtime/control_tests.cpp")
  target_link_libraries(xvram-torch-runtime-control-tests PRIVATE xvram_torch_runtime)
  target_include_directories(
    xvram-torch-runtime-control-tests PRIVATE "${PROJECT_SOURCE_DIR}/include")
  target_compile_features(xvram-torch-runtime-control-tests PRIVATE cxx_std_20)
  xvram_enable_warnings(xvram-torch-runtime-control-tests)
  add_test(NAME xvram.torch-runtime.control COMMAND xvram-torch-runtime-control-tests)
  list(GET XVRAM_TORCH_LIBRARY_DIRS 0 XVRAM_TORCH_PRIMARY_LIBRARY_DIR)
  set_tests_properties(
    xvram.torch-runtime.control
    PROPERTIES TIMEOUT 30
               ENVIRONMENT_MODIFICATION
               "PATH=path_list_prepend:$<TARGET_FILE_DIR:xvram_torch_runtime>;PATH=path_list_prepend:${XVRAM_TORCH_PRIMARY_LIBRARY_DIR}")
  add_test(
    NAME xvram.torch-runtime.export-contract
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/contract/torch_runtime_export_contract.py"
            "$<TARGET_FILE:xvram_torch_runtime>")
  add_test(
    NAME xvram.torch-runtime.stable-load
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/contract/torch_runtime_stable_load.py"
            "$<TARGET_FILE:xvram_torch_runtime>")
  set_tests_properties(
    xvram.torch-runtime.export-contract xvram.torch-runtime.stable-load
    PROPERTIES TIMEOUT 30)
  set_tests_properties(
    xvram.torch-runtime.stable-load
    PROPERTIES ENVIRONMENT_MODIFICATION
               "PATH=path_list_prepend:$<TARGET_FILE_DIR:xvram_torch_runtime>;PATH=path_list_prepend:${XVRAM_TORCH_PRIMARY_LIBRARY_DIR}")
endif()

add_test(
  NAME xvram.torch-python.unit
  COMMAND "${Python3_EXECUTABLE}" -m unittest discover -s
          "${CMAKE_CURRENT_SOURCE_DIR}/python" -p "test_*.py")
set_tests_properties(
  xvram.torch-python.unit
  PROPERTIES ENVIRONMENT
             "PYTHONPATH=${PROJECT_SOURCE_DIR}/python;XVRAM_TORCH_ALLOCATOR_LIBRARY=$<TARGET_FILE:xvram_torch_allocator>"
             TIMEOUT 60)

add_test(
  NAME xvram.torch-inference.help
  COMMAND "${Python3_EXECUTABLE}" -m xvram.torch_bench --help)
add_test(
  NAME xvram.torch-inference.version
  COMMAND "${Python3_EXECUTABLE}" -m xvram.torch_bench --version)
set_tests_properties(
  xvram.torch-inference.help xvram.torch-inference.version
  PROPERTIES ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR}/python" TIMEOUT 15)

add_executable(xvram-vmm-poc-unit-tests unit/vmm_poc_tests.cpp
                                          "${PROJECT_SOURCE_DIR}/src/vmm_poc/workload.cpp")
target_include_directories(xvram-vmm-poc-unit-tests PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_compile_features(xvram-vmm-poc-unit-tests PRIVATE cxx_std_20)
xvram_enable_warnings(xvram-vmm-poc-unit-tests)

add_test(NAME xvram.vmm-poc.unit COMMAND xvram-vmm-poc-unit-tests)

add_executable(xvram-vmm-poc-outcome-tests unit/vmm_outcome_tests.cpp)
target_link_libraries(xvram-vmm-poc-outcome-tests PRIVATE xvram_vmm_poc)
target_include_directories(xvram-vmm-poc-outcome-tests PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_include_directories(xvram-vmm-poc-outcome-tests
                           SYSTEM PRIVATE "${XVRAM_CUDA_INCLUDE_DIR}")
target_compile_features(xvram-vmm-poc-outcome-tests PRIVATE cxx_std_20)
xvram_enable_warnings(xvram-vmm-poc-outcome-tests)

add_test(NAME xvram.vmm-poc.outcome COMMAND xvram-vmm-poc-outcome-tests)

add_executable(xvram-vmm-poc-executor-fake-tests unit/vmm_executor_fake_tests.cpp)
target_link_libraries(xvram-vmm-poc-executor-fake-tests PRIVATE xvram_vmm_poc)
target_include_directories(xvram-vmm-poc-executor-fake-tests
                           PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_include_directories(xvram-vmm-poc-executor-fake-tests
                           SYSTEM PRIVATE "${XVRAM_CUDA_INCLUDE_DIR}")
target_compile_features(xvram-vmm-poc-executor-fake-tests PRIVATE cxx_std_20)
xvram_enable_warnings(xvram-vmm-poc-executor-fake-tests)

add_test(NAME xvram.vmm-poc.executor-fake COMMAND xvram-vmm-poc-executor-fake-tests)
set_tests_properties(xvram.vmm-poc.executor-fake PROPERTIES TIMEOUT 30)

add_executable(xvram-residency-core-tests unit/residency_core_tests.cpp)
target_link_libraries(xvram-residency-core-tests PRIVATE xvram_residency_core)
target_compile_features(xvram-residency-core-tests PRIVATE cxx_std_20)
xvram_enable_warnings(xvram-residency-core-tests)

add_test(NAME xvram.residency.core COMMAND xvram-residency-core-tests)

add_executable(xvram-nvcomp-dispatch-tests unit/nvcomp_dispatch_tests.cpp)
target_link_libraries(xvram-nvcomp-dispatch-tests PRIVATE xvram_residency_core)
target_include_directories(xvram-nvcomp-dispatch-tests PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_include_directories(
  xvram-nvcomp-dispatch-tests SYSTEM PRIVATE ${XVRAM_NVCOMP_INCLUDE_DIRS})
target_compile_features(xvram-nvcomp-dispatch-tests PRIVATE cxx_std_20)
xvram_enable_warnings(xvram-nvcomp-dispatch-tests)
add_test(NAME xvram.nvcomp.dispatch COMMAND xvram-nvcomp-dispatch-tests)

add_executable(xvram-nvcomp-load-test-helper helpers/nvcomp_load_test_helper.cpp)
target_link_libraries(xvram-nvcomp-load-test-helper PRIVATE xvram_residency_core)
target_include_directories(xvram-nvcomp-load-test-helper PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_include_directories(
  xvram-nvcomp-load-test-helper SYSTEM PRIVATE ${XVRAM_NVCOMP_INCLUDE_DIRS})
target_compile_definitions(
  xvram-nvcomp-load-test-helper
  PRIVATE "XVRAM_NVCOMP_LIBRARY_SHA256=\"${XVRAM_NVCOMP_LIBRARY_SHA256}\"")
target_compile_features(xvram-nvcomp-load-test-helper PRIVATE cxx_std_20)
xvram_enable_warnings(xvram-nvcomp-load-test-helper)
xvram_stage_nvcomp(xvram-nvcomp-load-test-helper)
add_test(NAME xvram.nvcomp.app-local-load COMMAND xvram-nvcomp-load-test-helper)

add_executable(xvram-nvcomp-pipeline-tests unit/nvcomp_pipeline_tests.cpp)
target_link_libraries(xvram-nvcomp-pipeline-tests PRIVATE xvram_residency_core)
target_include_directories(xvram-nvcomp-pipeline-tests PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_include_directories(
  xvram-nvcomp-pipeline-tests SYSTEM PRIVATE ${XVRAM_NVCOMP_INCLUDE_DIRS})
target_compile_features(xvram-nvcomp-pipeline-tests PRIVATE cxx_std_20)
xvram_enable_warnings(xvram-nvcomp-pipeline-tests)
add_test(NAME xvram.nvcomp.pipeline COMMAND xvram-nvcomp-pipeline-tests)

add_executable(xvram-compression-core-tests unit/compression_core_tests.cpp)
target_link_libraries(xvram-compression-core-tests PRIVATE xvram_residency_core)
target_include_directories(xvram-compression-core-tests PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_compile_features(xvram-compression-core-tests PRIVATE cxx_std_20)
xvram_enable_warnings(xvram-compression-core-tests)
add_test(NAME xvram.compression.core COMMAND xvram-compression-core-tests)

add_executable(xvram-cpu-codec-pool-tests unit/cpu_codec_pool_tests.cpp)
target_link_libraries(xvram-cpu-codec-pool-tests PRIVATE xvram_residency_core)
target_include_directories(xvram-cpu-codec-pool-tests PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_compile_features(xvram-cpu-codec-pool-tests PRIVATE cxx_std_20)
xvram_enable_warnings(xvram-cpu-codec-pool-tests)
add_test(NAME xvram.compression.cpu-codec-pool COMMAND xvram-cpu-codec-pool-tests)

add_executable(xvram-compression-options-tests unit/compression_options_tests.cpp)
target_link_libraries(xvram-compression-options-tests PRIVATE xvram_compression_bench)
target_compile_features(xvram-compression-options-tests PRIVATE cxx_std_20)
xvram_enable_warnings(xvram-compression-options-tests)
add_test(NAME xvram.compression.options COMMAND xvram-compression-options-tests)

add_executable(xvram-compression-executor-tests unit/compression_executor_tests.cpp)
target_link_libraries(xvram-compression-executor-tests PRIVATE xvram_compression_bench)
target_compile_features(xvram-compression-executor-tests PRIVATE cxx_std_20)
xvram_enable_warnings(xvram-compression-executor-tests)
add_test(NAME xvram.compression.executor COMMAND xvram-compression-executor-tests)

add_executable(xvram-compression-worker-test-helper helpers/compression_worker_test_helper.cpp)
target_link_libraries(xvram-compression-worker-test-helper PRIVATE xvram_compression_contract)
target_compile_features(xvram-compression-worker-test-helper PRIVATE cxx_std_20)
xvram_enable_warnings(xvram-compression-worker-test-helper)

add_executable(xvram-compression-worker-controller-tests
               unit/compression_worker_controller_tests.cpp)
target_link_libraries(xvram-compression-worker-controller-tests
                      PRIVATE xvram_compression_bench)
target_compile_features(xvram-compression-worker-controller-tests PRIVATE cxx_std_20)
add_dependencies(xvram-compression-worker-controller-tests xvram-compression-worker-test-helper)
xvram_enable_warnings(xvram-compression-worker-controller-tests)
add_test(
  NAME xvram.compression.worker-controller
  COMMAND xvram-compression-worker-controller-tests
          "$<TARGET_FILE:xvram-compression-worker-test-helper>")
set_tests_properties(xvram.compression.worker-controller PROPERTIES TIMEOUT 20)

add_test(
  NAME xvram.compression.controller-json-contract
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/contract/compression_controller_contract.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/../schemas/adaptive-compression-report-v1.schema.json"
          "$<TARGET_FILE:xvram-compression-worker-controller-tests>"
          "$<TARGET_FILE:xvram-compression-worker-test-helper>")
set_tests_properties(xvram.compression.controller-json-contract PROPERTIES TIMEOUT 30)

add_executable(xvram-compression-report-tests unit/compression_report_tests.cpp)
target_link_libraries(xvram-compression-report-tests PRIVATE xvram_compression_contract)
target_compile_features(xvram-compression-report-tests PRIVATE cxx_std_20)
xvram_enable_warnings(xvram-compression-report-tests)
add_test(NAME xvram.compression.report COMMAND xvram-compression-report-tests)

add_test(
  NAME xvram.compression.cpp-json-contract
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/contract/compression_cpp_contract.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/../schemas/adaptive-compression-report-v1.schema.json"
          "${CMAKE_CURRENT_SOURCE_DIR}/../schemas/compression-trace-record-v1.schema.json"
          "$<TARGET_FILE:xvram-compression-report-tests>")
set_tests_properties(xvram.compression.cpp-json-contract PROPERTIES TIMEOUT 20)

add_executable(xvram-compression-worker-protocol-tests
               unit/compression_worker_protocol_tests.cpp)
target_link_libraries(xvram-compression-worker-protocol-tests PRIVATE xvram_compression_contract)
target_compile_features(xvram-compression-worker-protocol-tests PRIVATE cxx_std_20)
xvram_enable_warnings(xvram-compression-worker-protocol-tests)
add_test(NAME xvram.compression.worker-protocol
         COMMAND xvram-compression-worker-protocol-tests)

add_executable(xvram-gemm-planner-tests unit/gemm_planner_tests.cpp)
target_link_libraries(xvram-gemm-planner-tests PRIVATE xvram_gemm_core)
target_include_directories(xvram-gemm-planner-tests PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_compile_features(xvram-gemm-planner-tests PRIVATE cxx_std_20)
xvram_enable_warnings(xvram-gemm-planner-tests)

add_test(NAME xvram.gemm.planner COMMAND xvram-gemm-planner-tests)

add_executable(xvram-gemm-pattern-tests unit/gemm_pattern_tests.cpp
                                        "${PROJECT_SOURCE_DIR}/src/gemm_bench/pattern.cpp")
target_include_directories(xvram-gemm-pattern-tests PRIVATE "${PROJECT_SOURCE_DIR}/src"
                                                            "${PROJECT_SOURCE_DIR}/include")
target_compile_features(xvram-gemm-pattern-tests PRIVATE cxx_std_20)
xvram_enable_warnings(xvram-gemm-pattern-tests)

add_test(NAME xvram.gemm.pattern COMMAND xvram-gemm-pattern-tests)

add_executable(xvram-gemm-sizing-tests unit/gemm_sizing_tests.cpp
                                       "${PROJECT_SOURCE_DIR}/src/gemm_bench/sizing.cpp")
target_include_directories(xvram-gemm-sizing-tests PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_compile_features(xvram-gemm-sizing-tests PRIVATE cxx_std_20)
xvram_enable_warnings(xvram-gemm-sizing-tests)

add_test(NAME xvram.gemm.sizing COMMAND xvram-gemm-sizing-tests)

add_executable(xvram-gemm-worker-protocol-tests unit/gemm_worker_protocol_tests.cpp)
target_link_libraries(xvram-gemm-worker-protocol-tests PRIVATE xvram_gemm_bench)
target_include_directories(xvram-gemm-worker-protocol-tests
                           PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_compile_features(xvram-gemm-worker-protocol-tests PRIVATE cxx_std_20)
xvram_enable_warnings(xvram-gemm-worker-protocol-tests)
add_test(NAME xvram.gemm.worker-protocol COMMAND xvram-gemm-worker-protocol-tests)

add_executable(xvram-sdk-abi-tests unit/sdk_abi_tests.cpp
                                     "${PROJECT_SOURCE_DIR}/src/sdk/api.cpp"
                                     "${PROJECT_SOURCE_DIR}/src/sdk/status.cpp")
target_include_directories(xvram-sdk-abi-tests PRIVATE "${PROJECT_SOURCE_DIR}/src"
                                                       "${PROJECT_SOURCE_DIR}/include")
target_compile_features(xvram-sdk-abi-tests PRIVATE cxx_std_20)
xvram_enable_warnings(xvram-sdk-abi-tests)

add_test(NAME xvram.sdk.abi COMMAND xvram-sdk-abi-tests)

# This executable has two deliberately separate modes. --self-test only loads the public v2 table
# and is safe in no-driver CI. The real CUDA path is not registered with CTest; the opt-in
# run-phase5-sdk-v2-hardware-smoke.ps1 gate builds and invokes it explicitly.
add_executable(xvram-sdk-v2-hardware-smoke helpers/sdk_v2_hardware_smoke.cpp)
target_link_libraries(xvram-sdk-v2-hardware-smoke PRIVATE xvram_sdk ${CMAKE_DL_LIBS})
target_compile_features(xvram-sdk-v2-hardware-smoke PRIVATE cxx_std_20)
xvram_enable_warnings(xvram-sdk-v2-hardware-smoke)
add_test(NAME xvram.sdk-v2.smoke-contract
         COMMAND xvram-sdk-v2-hardware-smoke --self-test)
set_tests_properties(
  xvram.sdk-v2.smoke-contract
  PROPERTIES LABELS "no-driver;sdk-v2"
             TIMEOUT 15
             ENVIRONMENT_MODIFICATION
             "PATH=path_list_prepend:$<TARGET_FILE_DIR:xvram_sdk>")

add_executable(xvram-sdk-deadline-tests unit/sdk_deadline_tests.cpp
                                        "${PROJECT_SOURCE_DIR}/src/sdk/status.cpp")
target_include_directories(xvram-sdk-deadline-tests PRIVATE "${PROJECT_SOURCE_DIR}/src"
                                                            "${PROJECT_SOURCE_DIR}/include")
target_compile_features(xvram-sdk-deadline-tests PRIVATE cxx_std_20)
xvram_enable_warnings(xvram-sdk-deadline-tests)

add_test(NAME xvram.sdk.deadline COMMAND xvram-sdk-deadline-tests)

add_test(
  NAME xvram.sdk.export-contract
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/contract/sdk_export_contract.py"
          "$<TARGET_FILE:xvram_sdk>")
add_test(
  NAME xvram.sdk.export-parser
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/contract/sdk_export_contract.py"
          --self-test)

add_executable(xvram-residency-workload-tests unit/residency_workload_tests.cpp)
target_link_libraries(xvram-residency-workload-tests PRIVATE xvram_residency)
target_include_directories(xvram-residency-workload-tests PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_compile_features(xvram-residency-workload-tests PRIVATE cxx_std_20)
xvram_enable_warnings(xvram-residency-workload-tests)

add_test(NAME xvram.residency.workload COMMAND xvram-residency-workload-tests)

add_executable(xvram-cache-worker-protocol-tests unit/cache_worker_protocol_tests.cpp)
target_link_libraries(xvram-cache-worker-protocol-tests PRIVATE xvram_residency)
target_include_directories(xvram-cache-worker-protocol-tests PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_compile_features(xvram-cache-worker-protocol-tests PRIVATE cxx_std_20)
xvram_enable_warnings(xvram-cache-worker-protocol-tests)

add_test(NAME xvram.residency.worker-protocol COMMAND xvram-cache-worker-protocol-tests)

add_executable(xvram-residency-executor-fake-tests unit/residency_executor_fake_tests.cpp)
target_link_libraries(xvram-residency-executor-fake-tests PRIVATE xvram_residency)
target_include_directories(xvram-residency-executor-fake-tests
                           PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_include_directories(xvram-residency-executor-fake-tests
                           SYSTEM PRIVATE ${XVRAM_NVCOMP_INCLUDE_DIRS})
target_compile_features(xvram-residency-executor-fake-tests PRIVATE cxx_std_20)
xvram_enable_warnings(xvram-residency-executor-fake-tests)

add_test(NAME xvram.residency.executor-fake COMMAND xvram-residency-executor-fake-tests)
set_tests_properties(xvram.residency.executor-fake PROPERTIES TIMEOUT 30)

find_package(Threads REQUIRED)

add_executable(
  xvram-worker-test-helper
  helpers/worker_test_helper.cpp
  "${PROJECT_SOURCE_DIR}/src/vmm_poc/worker_protocol.cpp")
target_include_directories(xvram-worker-test-helper PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_compile_features(xvram-worker-test-helper PRIVATE cxx_std_20)
target_link_libraries(xvram-worker-test-helper PRIVATE Threads::Threads)
xvram_enable_warnings(xvram-worker-test-helper)

add_executable(
  xvram-worker-controller-tests
  unit/worker_controller_tests.cpp
  "${PROJECT_SOURCE_DIR}/src/vmm_poc/worker_protocol.cpp"
  "${PROJECT_SOURCE_DIR}/src/vmm_poc/worker_controller.cpp")
target_include_directories(xvram-worker-controller-tests PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_compile_features(xvram-worker-controller-tests PRIVATE cxx_std_20)
target_link_libraries(xvram-worker-controller-tests PRIVATE Threads::Threads xvram_worker_process)
add_dependencies(xvram-worker-controller-tests xvram-worker-test-helper)
xvram_enable_warnings(xvram-worker-controller-tests)

add_test(
  NAME xvram.vmm-poc.worker-controller
  COMMAND xvram-worker-controller-tests "$<TARGET_FILE:xvram-worker-test-helper>")
set_tests_properties(xvram.vmm-poc.worker-controller PROPERTIES TIMEOUT 15)

add_executable(xvram-cache-worker-test-helper helpers/cache_worker_test_helper.cpp)
target_link_libraries(xvram-cache-worker-test-helper PRIVATE xvram_residency)
target_include_directories(xvram-cache-worker-test-helper PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_compile_features(xvram-cache-worker-test-helper PRIVATE cxx_std_20)
xvram_enable_warnings(xvram-cache-worker-test-helper)

add_executable(xvram-cache-worker-controller-tests unit/cache_worker_controller_tests.cpp)
target_link_libraries(xvram-cache-worker-controller-tests
                      PRIVATE xvram_residency xvram_worker_process Threads::Threads)
target_include_directories(xvram-cache-worker-controller-tests
                           PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_compile_features(xvram-cache-worker-controller-tests PRIVATE cxx_std_20)
add_dependencies(xvram-cache-worker-controller-tests xvram-cache-worker-test-helper)
xvram_enable_warnings(xvram-cache-worker-controller-tests)

add_test(
  NAME xvram.residency.worker-controller
  COMMAND xvram-cache-worker-controller-tests "$<TARGET_FILE:xvram-cache-worker-test-helper>")
set_tests_properties(xvram.residency.worker-controller PROPERTIES TIMEOUT 15)

add_executable(xvram-gemm-worker-test-helper helpers/gemm_worker_test_helper.cpp)
target_link_libraries(xvram-gemm-worker-test-helper PRIVATE xvram_gemm_bench)
target_include_directories(xvram-gemm-worker-test-helper PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_compile_features(xvram-gemm-worker-test-helper PRIVATE cxx_std_20)
xvram_enable_warnings(xvram-gemm-worker-test-helper)

add_executable(xvram-gemm-worker-controller-tests unit/gemm_worker_controller_tests.cpp)
target_link_libraries(xvram-gemm-worker-controller-tests
                      PRIVATE xvram_gemm_bench xvram_worker_process Threads::Threads)
target_include_directories(xvram-gemm-worker-controller-tests
                           PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_compile_features(xvram-gemm-worker-controller-tests PRIVATE cxx_std_20)
add_dependencies(xvram-gemm-worker-controller-tests xvram-gemm-worker-test-helper)
xvram_enable_warnings(xvram-gemm-worker-controller-tests)
add_test(
  NAME xvram.gemm.worker-controller
  COMMAND xvram-gemm-worker-controller-tests "$<TARGET_FILE:xvram-gemm-worker-test-helper>")
set_tests_properties(xvram.gemm.worker-controller PROPERTIES TIMEOUT 20)

add_executable(xvram-privacy-tests unit/privacy_tests.cpp)
target_link_libraries(xvram-privacy-tests PRIVATE xvram_probe_lib)
target_include_directories(xvram-privacy-tests PRIVATE "${PROJECT_SOURCE_DIR}/src")
xvram_enable_warnings(xvram-privacy-tests)

add_test(NAME xvram.privacy COMMAND xvram-privacy-tests)
add_test(NAME xvram.probe.help COMMAND xvram-probe --help)
add_test(NAME xvram.probe.version COMMAND xvram-probe --version)
add_test(NAME xvram.vmm-poc.help COMMAND xvram-vmm-poc --help)
add_test(NAME xvram.vmm-poc.version COMMAND xvram-vmm-poc --version)
add_test(NAME xvram.cache-bench.help COMMAND xvram-cache-bench --help)
add_test(NAME xvram.cache-bench.version COMMAND xvram-cache-bench --version)
add_test(NAME xvram.gemm-bench.help COMMAND xvram-gemm-bench --help)
add_test(NAME xvram.gemm-bench.version COMMAND xvram-gemm-bench --version)
add_test(NAME xvram.compression-bench.help COMMAND xvram-compression-bench --help)
add_test(NAME xvram.compression-bench.version COMMAND xvram-compression-bench --version)

add_executable(xvram-report-fixture contract/fixture_report.cpp)
target_link_libraries(xvram-report-fixture PRIVATE xvram_probe_lib)
target_sources(
  xvram-report-fixture
  PRIVATE "${PROJECT_SOURCE_DIR}/src/vmm_poc/report_json.cpp"
          "${PROJECT_SOURCE_DIR}/src/vmm_poc/report_text.cpp")
xvram_enable_warnings(xvram-report-fixture)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import jsonschema"
  RESULT_VARIABLE XVRAM_JSONSCHEMA_IMPORT_RESULT
  OUTPUT_QUIET
  ERROR_QUIET)
if(NOT XVRAM_JSONSCHEMA_IMPORT_RESULT EQUAL 0)
  message(
    FATAL_ERROR
      "XVRAM_BUILD_TESTS requires jsonschema; run: ${Python3_EXECUTABLE} -m pip install --requirement tests/requirements.txt"
  )
endif()

add_test(
  NAME xvram.probe.json-contract
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/contract/probe_contract.py"
          "$<TARGET_FILE:xvram-probe>"
          "${CMAKE_CURRENT_SOURCE_DIR}/../schemas/capability-report-v2.schema.json"
          "$<TARGET_FILE:xvram-report-fixture>")

add_test(
  NAME xvram.vmm-poc.json-contract
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/contract/vmm_poc_contract.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/../schemas/vmm-poc-report-v1.schema.json"
          "$<TARGET_FILE:xvram-report-fixture>")

add_test(
  NAME xvram.vmm-poc.cli-contract
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/contract/vmm_poc_cli_contract.py"
          "$<TARGET_FILE:xvram-vmm-poc>"
          "${CMAKE_CURRENT_SOURCE_DIR}/../schemas/vmm-poc-report-v1.schema.json")

add_test(
  NAME xvram.residency.json-contract
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/contract/residency_contract.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/../schemas/residency-cache-report-v1.schema.json"
          "${CMAKE_CURRENT_SOURCE_DIR}/../schemas/residency-trace-record-v1.schema.json")

add_test(
  NAME xvram.gemm.json-contract
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/contract/gemm_contract.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/../schemas/gemm-bench-report-v1.schema.json")

add_test(
  NAME xvram.compression.json-contract
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/contract/compression_contract.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/../schemas/adaptive-compression-report-v1.schema.json"
          "${CMAKE_CURRENT_SOURCE_DIR}/../schemas/compression-trace-record-v1.schema.json")

add_test(
  NAME xvram.compression.trace-contract
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/contract/compression_trace_contract.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/../schemas/compression-trace-record-v1.schema.json"
          "${CMAKE_CURRENT_SOURCE_DIR}/../scripts")

add_test(
  NAME xvram.compression.cli-contract
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/contract/compression_cli_contract.py"
          "$<TARGET_FILE:xvram-compression-bench>"
          "${CMAKE_CURRENT_SOURCE_DIR}/../schemas/adaptive-compression-report-v1.schema.json")
set_tests_properties(xvram.compression.cli-contract PROPERTIES TIMEOUT 20)

add_test(
  NAME xvram.gemm.cli-contract
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/contract/gemm_cli_contract.py"
          "$<TARGET_FILE:xvram-gemm-bench>"
          "${CMAKE_CURRENT_SOURCE_DIR}/../schemas/gemm-bench-report-v1.schema.json")
set_tests_properties(xvram.gemm.cli-contract PROPERTIES TIMEOUT 20)

add_test(
  NAME xvram.residency.cli-contract
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/contract/residency_cli_contract.py"
          "$<TARGET_FILE:xvram-cache-bench>"
          "${CMAKE_CURRENT_SOURCE_DIR}/../schemas/residency-cache-report-v1.schema.json")
set_tests_properties(xvram.residency.cli-contract PROPERTIES TIMEOUT 20)

add_test(
  NAME xvram.schema.hash-contract
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/contract/schema_hash_contract.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/../schemas/capability-report-v1.schema.json"
          "${CMAKE_CURRENT_SOURCE_DIR}/../schemas/capability-report-v2.schema.json"
          "${CMAKE_CURRENT_SOURCE_DIR}/../schemas/vmm-poc-report-v1.schema.json"
          "${CMAKE_CURRENT_SOURCE_DIR}/../schemas/residency-cache-report-v1.schema.json"
          "${CMAKE_CURRENT_SOURCE_DIR}/../schemas/residency-trace-record-v1.schema.json"
          "${CMAKE_CURRENT_SOURCE_DIR}/../schemas/gemm-bench-report-v1.schema.json"
          "${CMAKE_CURRENT_SOURCE_DIR}/../schemas/pytorch-inference-report-v1.schema.json"
          "${CMAKE_CURRENT_SOURCE_DIR}/../schemas/pytorch-inference-trace-record-v1.schema.json"
          "${CMAKE_CURRENT_SOURCE_DIR}/../schemas/adaptive-compression-report-v1.schema.json"
          "${CMAKE_CURRENT_SOURCE_DIR}/../schemas/compression-trace-record-v1.schema.json"
          "${CMAKE_CURRENT_SOURCE_DIR}/../schemas/pytorch-inference-report-v2.schema.json"
          "${CMAKE_CURRENT_SOURCE_DIR}/../schemas/pytorch-inference-trace-record-v2.schema.json")

add_test(
  NAME xvram.abi.hash-contract
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/contract/abi_hash_contract.py"
          "${PROJECT_SOURCE_DIR}/include/xvram/xvram.h"
          "${PROJECT_SOURCE_DIR}/include/xvram/torch_allocator.h"
          "${PROJECT_SOURCE_DIR}/include/xvram/internal/torch_runtime.h"
          "${PROJECT_SOURCE_DIR}/include/xvram/xvram_v2.h"
          "${PROJECT_SOURCE_DIR}/include/xvram/internal/torch_runtime_v2.h")

add_test(
  NAME xvram.torch-inference.json-contract
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/contract/torch_inference_contract.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/../schemas/pytorch-inference-report-v1.schema.json"
          "${CMAKE_CURRENT_SOURCE_DIR}/../schemas/pytorch-inference-trace-record-v1.schema.json")

add_test(
  NAME xvram.torch-inference-v2.json-contract
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/contract/torch_inference_v2_contract.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/../schemas/pytorch-inference-report-v2.schema.json"
          "${CMAKE_CURRENT_SOURCE_DIR}/../schemas/pytorch-inference-trace-record-v2.schema.json")
set_tests_properties(
  xvram.torch-inference-v2.json-contract
  PROPERTIES ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR}/python" TIMEOUT 15)
set_tests_properties(
  xvram.torch-inference.json-contract
  PROPERTIES ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR}/python" TIMEOUT 15)

add_test(
  NAME xvram.torch-inference.cli-contract
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/contract/torch_inference_cli_contract.py")
set_tests_properties(xvram.torch-inference.cli-contract PROPERTIES TIMEOUT 20)
