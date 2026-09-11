// External consumer of the installed C control table. No CUDA Toolkit headers,
// private xVRAM headers, native CUDA linkage, or changed runtime architecture.
#include <xvram/cuda_compat.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {
constexpr int m = 37;
constexpr int n = 19;
constexpr int k = 67;

struct Session {
  xvram_cuda_compat_api_v1 api{};
  bool attempted = false;
  bool closed = false;

  void check(xvram_status status, const char* operation) const {
    if (status != XVRAM_STATUS_SUCCESS) {
      std::cerr << operation << " failed (xVRAM status " << status << ")\n";
      throw std::runtime_error(operation);
    }
  }
  bool initialize() {
    check(xvram_cuda_compat_get_api(XVRAM_CUDA_COMPAT_ABI_VERSION_1, sizeof(api), &api), "get_api");
    const xvram_cuda_compat_config_v1 config = XVRAM_CUDA_COMPAT_CONFIG_V1_INIT;
    attempted = true;
    const auto status = api.initialize(&config);
    if (status != XVRAM_STATUS_SUCCESS) {
      std::cerr << "GPU initialization did not succeed (status " << status
                << "). Check driver, VRAM/RAM budget and app-local cuBLAS libraries.\n";
      return false;
    }
    return true;
  }
  bool close() {
    if (!attempted || closed)
      return true;
    closed = true;
    return api.shutdown() == XVRAM_STATUS_SUCCESS;
  }
  ~Session() {
    if (!close())
      std::cerr << "Session cleanup did not complete.\n";
  }
};

float a_value(int row, int column) {
  return static_cast<float>((row + 3 * column) % 17 - 8) / 16.0F;
}
float b_value(int row, int column) {
  return static_cast<float>((2 * row + column) % 13 - 6) / 16.0F;
}
} // namespace

int main(int argc, char** argv) {
  if (argc != 2 || std::string_view(argv[1]) != "--run") {
    std::cout << "xVRAM developer preview: small external FP32 GEMM example\n"
                 "Usage: xvram-preview-gemm --run\n"
                 "Uses the installed C API; requires an NVIDIA VMM-capable GPU, compatible\n"
                 "driver and app-local cuBLAS libraries. No API key or administrator required.\n"
                 "This small integration example is NOT an oversubscription benchmark.\n"
                 "For the isolated, validated large-data demo use preview.py oversubscribe.\n";
    return (argc == 1 || (argc == 2 && std::string_view(argv[1]) == "--help")) ? 0 : 64;
  }
  Session session;
  try {
    if (!session.initialize()) {
      // Initialization can fail without a GPU. Never count this as a passing GPU test.
      (void)session.close();
      return 23;
    }
    std::array<float, m * k> a{};
    std::array<float, k * n> b{};
    std::array<float, m * n> c{};
    for (int column = 0; column < k; ++column)
      for (int row = 0; row < m; ++row)
        a[static_cast<std::size_t>(row + column * m)] = a_value(row, column);
    for (int column = 0; column < n; ++column)
      for (int row = 0; row < k; ++row)
        b[static_cast<std::size_t>(row + column * k)] = b_value(row, column);
    void* device_a = nullptr;
    void* device_b = nullptr;
    void* device_c = nullptr;
    void* handle = nullptr;
    auto& api = session.api;
    session.check(api.malloc_device(&device_a, sizeof(a)), "allocate A");
    session.check(api.malloc_device(&device_b, sizeof(b)), "allocate B");
    session.check(api.malloc_device(&device_c, sizeof(c)), "allocate C");
    session.check(api.memcpy(device_a, a.data(), sizeof(a), XVRAM_CUDA_COMPAT_H2D), "copy A");
    session.check(api.memcpy(device_b, b.data(), sizeof(b), XVRAM_CUDA_COMPAT_H2D), "copy B");
    session.check(api.memcpy(device_c, c.data(), sizeof(c), XVRAM_CUDA_COMPAT_H2D), "copy C");
    session.check(api.blas_create(&handle), "create cuBLAS handle");
    const float alpha = 1.0F;
    const float beta = 0.0F;
    // The public operation constants describe non-transposed column-major matrices.
    session.check(api.blas_sgemm(handle, XVRAM_MATRIX_OP_N, XVRAM_MATRIX_OP_N, m, n, k, &alpha,
                                static_cast<const float*>(device_a), m,
                                static_cast<const float*>(device_b), k, &beta,
                                static_cast<float*>(device_c), m), "SGEMM");
    session.check(api.memcpy(c.data(), device_c, sizeof(c), XVRAM_CUDA_COMPAT_D2H), "read C");
    double maximum_error = 0.0;
    bool equal = true;
    for (int column = 0; column < n; ++column) {
      for (int row = 0; row < m; ++row) {
        double reference = 0.0;
        for (int inner = 0; inner < k; ++inner)
          reference += static_cast<double>(a_value(row, inner)) * b_value(inner, column);
        const double value = c[static_cast<std::size_t>(row + column * m)];
        const double error = std::abs(value - reference);
        equal = equal && std::isfinite(value) && error <= 1e-4 + 2e-5 * std::abs(reference);
        maximum_error = std::max(maximum_error, error);
      }
    }
    session.check(api.blas_destroy(handle), "destroy cuBLAS handle");
    session.check(api.free_device(device_c), "free C");
    session.check(api.free_device(device_b), "free B");
    session.check(api.free_device(device_a), "free A");
    if (!session.close()) {
      std::cerr << "Cleanup failed; the example is not successful.\n";
      return 27;
    }
    std::cout << "Checked " << m * n << " outputs against a full CPU FP64 reference.\n"
              << "Maximum absolute error: " << maximum_error << '\n'
              << (equal ? "PASS: small integration example; cleanup completed.\n"
                        : "FAIL: numerical verification.\n");
    return equal ? 0 : 24;
  } catch (const std::exception& error) {
    std::cerr << "Example failed: " << error.what() << '\n';
    return 27;
  }
}
