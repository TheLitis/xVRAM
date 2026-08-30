add_executable(xvram-unit-tests unit/base_tests.cpp)
target_link_libraries(xvram-unit-tests PRIVATE xvram_probe_lib)
xvram_enable_warnings(xvram-unit-tests)

add_test(NAME xvram.unit COMMAND xvram-unit-tests)

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

add_executable(xvram-report-fixture contract/fixture_report.cpp)
target_link_libraries(xvram-report-fixture PRIVATE xvram_probe_lib)
target_sources(
  xvram-report-fixture
  PRIVATE "${PROJECT_SOURCE_DIR}/src/vmm_poc/report_json.cpp"
          "${PROJECT_SOURCE_DIR}/src/vmm_poc/report_text.cpp")
xvram_enable_warnings(xvram-report-fixture)

find_package(Python3 COMPONENTS Interpreter REQUIRED)
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
