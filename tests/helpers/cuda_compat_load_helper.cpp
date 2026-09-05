#include "platform/cublas/cublas_api.hpp"

#include <iostream>

int main() {
  xvram::cublas::CublasApi api;
  const auto loaded = api.load();
  if (loaded.status != xvram::cublas::CublasLoadStatus::loaded || !api.has_lt() ||
      !api.has_complete_baseline()) {
    std::cerr << "app-local cuBLAS core/Lt dispatch unavailable\n";
    return 1;
  }
  std::cout << "{\"source\":\"" << xvram::cublas::cublas_library_source_name(api.library_source())
            << "\",\"lt_version\":" << api.dispatch().lt_get_version() << "}\n";
  return api.library_source() == xvram::cublas::CublasLibrarySource::app_local ? 0 : 2;
}
