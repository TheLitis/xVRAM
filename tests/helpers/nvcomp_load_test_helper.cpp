#include "platform/nvcomp/nvcomp_api.hpp"

#include <iostream>
#include <string>
#include <string_view>

#ifndef XVRAM_NVCOMP_LIBRARY_SHA256
#error "XVRAM_NVCOMP_LIBRARY_SHA256 must identify the staged nvCOMP shared library"
#endif

int main() {
  constexpr std::string_view expected_sha256{XVRAM_NVCOMP_LIBRARY_SHA256};
  if (expected_sha256.size() != 64U) {
    std::cerr << "configured nvCOMP shared-library SHA-256 is malformed\n";
    return 1;
  }

  xvram::nvcomp::NvcompApi api;
  const xvram::nvcomp::NvcompLoadResult loaded = api.load();
  if (loaded.status != xvram::nvcomp::NvcompLoadStatus::loaded) {
    std::cerr << "app-local nvCOMP load failed: " << api.error() << '\n';
    return 2;
  }
  if (api.library_source() != xvram::nvcomp::NvcompLibrarySource::app_local) {
    std::cerr << "default nvCOMP load did not use the app-local package\n";
    return 3;
  }
  if (!api.integrity_verified() || api.library_sha256() != expected_sha256) {
    std::cerr << "loaded nvCOMP bytes did not match the embedded library pin\n";
    return 4;
  }
  if (api.properties().version != 5300U || api.version_string() != "5.3.0.16") {
    std::cerr << "loaded nvCOMP did not pass the exact 5.3.0.16 identity gate\n";
    return 5;
  }

  xvram::nvcomp::NvcompApi mismatched;
  xvram::nvcomp::NvcompLoadOptions mismatch_options;
  mismatch_options.expected_library_sha256 = std::string(64U, '0');
  const xvram::nvcomp::NvcompLoadResult mismatch = mismatched.load(mismatch_options);
  if (mismatch.status != xvram::nvcomp::NvcompLoadStatus::integrity_failure ||
      mismatched.has_lz4() || mismatched.integrity_verified()) {
    std::cerr << "nvCOMP loader accepted a shared library that missed its SHA-256 pin\n";
    return 6;
  }

  std::cout << "{\"source\":\"app_local\",\"version\":" << api.properties().version
            << ",\"package_version\":\"" << api.version_string() << "\",\"library_sha256\":\""
            << api.library_sha256() << "\"}\n";
  return 0;
}
