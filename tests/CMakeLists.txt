add_executable(xvram-unit-tests unit/base_tests.cpp)
target_link_libraries(xvram-unit-tests PRIVATE xvram_probe_lib)
xvram_enable_warnings(xvram-unit-tests)

add_test(NAME xvram.unit COMMAND xvram-unit-tests)

add_executable(xvram-privacy-tests unit/privacy_tests.cpp)
target_link_libraries(xvram-privacy-tests PRIVATE xvram_probe_lib)
target_include_directories(xvram-privacy-tests PRIVATE "${PROJECT_SOURCE_DIR}/src")
xvram_enable_warnings(xvram-privacy-tests)

add_test(NAME xvram.privacy COMMAND xvram-privacy-tests)
add_test(NAME xvram.probe.help COMMAND xvram-probe --help)
add_test(NAME xvram.probe.version COMMAND xvram-probe --version)

add_executable(xvram-report-fixture contract/fixture_report.cpp)
target_link_libraries(xvram-report-fixture PRIVATE xvram_probe_lib)
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
          "${CMAKE_CURRENT_SOURCE_DIR}/../schemas/capability-report-v1.schema.json"
          "$<TARGET_FILE:xvram-report-fixture>")
