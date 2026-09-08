# Exercise the actual shared-memory observer in an owned child process. The
# helper links only Win32 system libraries; no CUDA/CUPTI or GPU is required.
set(XVRAM_AUDIT_CAPTURE_HEADER "${CMAKE_CURRENT_BINARY_DIR}/audit-generated/capture_catalog.hpp")
add_custom_command(OUTPUT "${XVRAM_AUDIT_CAPTURE_HEADER}"
  COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/scripts/generate-audit-kernel-header.py"
          --catalog "${PROJECT_SOURCE_DIR}/python/xvram/compat_audit_kernel_capture_catalog.json"
          --output "${XVRAM_AUDIT_CAPTURE_HEADER}"
  DEPENDS "${PROJECT_SOURCE_DIR}/scripts/generate-audit-kernel-header.py"
          "${PROJECT_SOURCE_DIR}/python/xvram/compat_audit_kernel_capture_catalog.json"
          "${PROJECT_SOURCE_DIR}/python/xvram/compat_audit_kernel_catalog.py"
          "${PROJECT_SOURCE_DIR}/python/xvram/compat_audit_kernel_ranges.py"
  VERBATIM)
add_executable(xvram-typed-capture-tests "${PROJECT_SOURCE_DIR}/tests/compat_audit_typed_capture_tests.cpp"
                                       "${XVRAM_AUDIT_CAPTURE_HEADER}")
target_include_directories(xvram-typed-capture-tests PRIVATE "${PROJECT_SOURCE_DIR}/src"
  "${PROJECT_SOURCE_DIR}/src/compat_launch_probe" "${CMAKE_CURRENT_BINARY_DIR}/audit-generated")
target_compile_features(xvram-typed-capture-tests PRIVATE cxx_std_20)
xvram_enable_warnings(xvram-typed-capture-tests)
add_test(NAME xvram.compat-audit.typed-capture COMMAND xvram-typed-capture-tests)
set_tests_properties(xvram.compat-audit.typed-capture PROPERTIES TIMEOUT 30 LABELS "no-driver")
add_dependencies(xvram-compat-audit-core-tests xvram-typed-capture-tests)
add_executable(xvram-memory-registry-tests "${PROJECT_SOURCE_DIR}/tests/compat_audit_memory_registry.cpp")
target_include_directories(xvram-memory-registry-tests PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_compile_features(xvram-memory-registry-tests PRIVATE cxx_std_20)
xvram_enable_warnings(xvram-memory-registry-tests)
add_test(NAME xvram.compat-audit.memory-registry COMMAND xvram-memory-registry-tests)
set_tests_properties(xvram.compat-audit.memory-registry PROPERTIES TIMEOUT 30 LABELS "no-driver")
add_dependencies(xvram-compat-audit-core-tests xvram-memory-registry-tests)
if(WIN32)
  add_executable(xvram-postmortem-test-helper
    "${PROJECT_SOURCE_DIR}/tests/compat_audit_postmortem_observer.cpp"
    "${PROJECT_SOURCE_DIR}/src/compat_launch_probe/postmortem.cpp")
  target_include_directories(xvram-postmortem-test-helper PRIVATE "${PROJECT_SOURCE_DIR}/src")
  target_compile_features(xvram-postmortem-test-helper PRIVATE cxx_std_20)
  target_compile_definitions(xvram-postmortem-test-helper PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
  xvram_enable_warnings(xvram-postmortem-test-helper)
  # The Stable-ABI CI job deliberately builds only this subset before running
  # the audit suite; keep its no-driver observer coverage without another job.
  add_dependencies(xvram-compat-audit-core-tests xvram-postmortem-test-helper)
  set_property(TEST xvram.compat-audit.unit APPEND PROPERTY ENVIRONMENT
    "XVRAM_POSTMORTEM_TEST_EXE=$<TARGET_FILE:xvram-postmortem-test-helper>")
endif()
