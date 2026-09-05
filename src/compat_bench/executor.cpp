#include "compat_bench/executor.hpp"
#include "compat_bench/native_baseline.hpp"
#include "platform/cuda/cuda_api.hpp"
#include "xvram/cuda_compat.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace xvram::compat_bench {
namespace {
namespace cc = xvram::cuda_compat;
using Clock = std::chrono::steady_clock;
constexpr std::uint64_t batch_elements = (16ULL << 20U) / sizeof(float);
struct Failure : std::exception {
  int code;
  std::string message;
  Failure(int value, std::string description) : code(value), message(std::move(description)) {}
  const char* what() const noexcept override {
    return message.c_str();
  }
};
int exit_for_status(const xvram_status status) {
  switch (status) {
  case XVRAM_STATUS_INVALID_ARGUMENT:
  case XVRAM_STATUS_INCOMPATIBLE_ABI:
  case XVRAM_STATUS_UNSUPPORTED:
  case XVRAM_STATUS_UNAVAILABLE:
    return 23;
  case XVRAM_STATUS_HOST_OUT_OF_MEMORY:
  case XVRAM_STATUS_DEVICE_OUT_OF_MEMORY:
  case XVRAM_STATUS_BUDGET_PRESSURE:
    return 25;
  case XVRAM_STATUS_CORRUPTION:
    return 24;
  case XVRAM_STATUS_TIMEOUT:
    return 26;
  case XVRAM_STATUS_INTERNAL:
    return 70;
  default:
    return 27;
  }
}
std::uint64_t checked_add(const std::uint64_t a, const std::uint64_t b) {
  if (b > UINT64_MAX - a)
    throw Failure(23, "operand storage size overflow");
  return a + b;
}
float* offset_pointer(void* pointer, const std::uint64_t elements) {
  return reinterpret_cast<float*>(reinterpret_cast<std::uintptr_t>(pointer) +
                                  elements * sizeof(float));
}
Shape choose_shape(const Options& o, const xvram_cuda_compat_telemetry_v1& t) {
  Shape s;
  s.padding = o.padding;
  s.offset = o.offset_elements;
  s.transpose_a = o.transpose_a;
  s.transpose_b = o.transpose_b;
  if (o.m) {
    s.m = *o.m;
    s.n = *o.n;
    s.k = *o.k;
  } else {
    s.m = 4096;
    s.n = 16;
    // The adapter already resolved live host headroom and charged pinned buffers once.
    // Leave only benchmark scratch here; do not subtract pinned memory from free RAM twice.
    const auto charged = checked_add(t.host_budget_bytes, 256ULL << 20U);
    const auto host = t.host_store_cap_bytes > charged ? t.host_store_cap_bytes - charged : 0;
    const auto target = o.logical_bytes.value_or(
        std::min(checked_add(t.total_vram_bytes, t.total_vram_bytes / 2U), host));
    const auto output = s.m * s.n * sizeof(float), step = (s.m + s.n) * sizeof(float);
    if (target <= output || (target - output) / step == 0)
      throw Failure(23, "logical size cannot fit the minimum declared geometry safely");
    // The requested bound is rounded down by less than one complete K-plane. No phantom bytes
    // are counted: every reported logical byte is an A/B/C matrix element used by SGEMM.
    s.k = (target - output) / step;
  }
  if (!matrix_storage(s, Operand::a) || !matrix_storage(s, Operand::b) ||
      !matrix_storage(s, Operand::c) || !operand_bytes(s))
    throw Failure(23, "matrix geometry/leading dimension/storage exceeds supported range");
  return s;
}
bool close_enough(const double actual, const double expected) {
  return std::isfinite(actual) && std::isfinite(expected) &&
         std::abs(actual - expected) <= 1e-4 + 2e-5 * std::abs(expected);
}
void optional_identifiers(Report& report) {
  if (!report.options.identifiers)
    return;
  cuda::CudaApi cuda;
  if (cuda.load().status != cuda::CudaApi::LoadStatus::loaded || cuda.init_(0) != CUDA_SUCCESS)
    return;
  CUdevice device = 0;
  if (cuda.device_get_(&device, report.options.device) != CUDA_SUCCESS)
    return;
  if (cuda.device_get_uuid_) {
    CUuuid uuid{};
    if (cuda.device_get_uuid_(&uuid, device) == CUDA_SUCCESS) {
      std::ostringstream text;
      text << "GPU-" << std::hex << std::setfill('0');
      for (std::size_t i = 0; i < sizeof(uuid.bytes); ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10)
          text << '-';
        text << std::setw(2) << static_cast<unsigned>(static_cast<unsigned char>(uuid.bytes[i]));
      }
      report.uuid = text.str();
    }
  }
  if (cuda.device_get_pci_bus_id_) {
    std::array<char, 64> pci{};
    if (cuda.device_get_pci_bus_id_(pci.data(), static_cast<int>(pci.size()), device) ==
        CUDA_SUCCESS)
      report.pci_bus_id = std::string(pci.data());
  }
}
} // namespace

Report run_executor(const Options& options, const ExecutionTrace& trace) {
  Report report = base_report(options);
  xvram_cuda_compat_api_v1 api{};
  const auto loaded = xvram_cuda_compat_get_api(XVRAM_CUDA_COMPAT_ABI_VERSION_1, sizeof(api), &api);
  if (loaded != XVRAM_STATUS_SUCCESS) {
    report.exit_code = 23;
    report.reason = "incompatible_abi";
    report.message = "compatibility API v1 is unavailable";
    return report;
  }
  const auto snapshot = [&] {
    xvram_cuda_compat_telemetry_v1 t{};
    t.struct_size = sizeof(t);
    if (api.get_telemetry(&t) == XVRAM_STATUS_SUCCESS) {
      report.telemetry = t;
      report.telemetry_observed = t.total_vram_bytes != 0;
    }
    return t;
  };
  const auto require = [&](const bool success, const char* operation) {
    if (success)
      return;
    xvram_error_info_v1 error = XVRAM_ERROR_INFO_V1_INIT;
    const auto status = api.get_error(&error);
    throw Failure(status == XVRAM_STATUS_SUCCESS ? exit_for_status(error.status) : 27,
                  std::string(operation) + ": " + error.message);
  };
  xvram_cuda_compat_config_v1 config = XVRAM_CUDA_COMPAT_CONFIG_V1_INIT;
  config.session.device_ordinal = options.device;
  config.session.chunk_size_bytes = options.chunk_bytes;
  config.session.cache_target_bytes = options.cache_bytes;
  config.session.device_headroom_bytes = options.headroom_bytes;
  config.session.workspace_cap_bytes = options.workspace_bytes;
  config.session.staging_slots = options.staging_slots;
  config.session.prefetch_distance = options.prefetch_distance;
  config.session.cache_policy =
      options.policy == "lru" ? XVRAM_CACHE_POLICY_LRU : XVRAM_CACHE_POLICY_CLOCK;
  config.session.budget_poll_ms = options.budget_poll_ms;
  config.stall_timeout_ms = options.stall_timeout_ms;
  if (options.cublas_library.size() >= sizeof(config.cublas_library) ||
      options.cublas_lt_library.size() >= sizeof(config.cublas_lt_library)) {
    report.exit_code = 23;
    report.reason = "configuration";
    report.message = "cuBLAS path exceeds the control ABI limit";
    return report;
  }
  std::memcpy(config.cublas_library, options.cublas_library.c_str(),
              options.cublas_library.size() + 1U);
  std::memcpy(config.cublas_lt_library, options.cublas_lt_library.c_str(),
              options.cublas_lt_library.size() + 1U);
  std::vector<void*> live_allocations;
  cublasHandle_t handle = nullptr;
  bool initialized = false;
  std::uint64_t next_operation = 0, next_allocation = 0;
  const auto emit = [&](std::string_view transition, std::uint64_t operation,
                        std::optional<std::uint64_t> allocation, std::uint64_t bytes,
                        std::string_view why) {
    if (trace)
      trace(transition, operation, allocation, bytes, why);
  };
  try {
    require(api.initialize(&config) == XVRAM_STATUS_SUCCESS, "initialize");
    initialized = true;
    static_cast<void>(snapshot());
    optional_identifiers(report);
    int devices = 0, device = -1;
    require(cc::cudaGetDeviceCount(&devices) == cudaSuccess, "cudaGetDeviceCount");
    require(cc::cudaGetDevice(&device) == cudaSuccess, "cudaGetDevice");
    if (devices <= options.device || device != options.device)
      throw Failure(27, "selected device does not match the adapter context");
    require(cc::cudaSetDevice(options.device) == cudaSuccess, "cudaSetDevice");
    require(cc::cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS, "cublasCreate");
    int version = 0;
    require(cc::cublasGetVersion(handle, &version) == CUBLAS_STATUS_SUCCESS, "cublasGetVersion");
    require(cc::cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH) == CUBLAS_STATUS_SUCCESS,
            "cublasSetMathMode");
    require(cc::cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST) == CUBLAS_STATUS_SUCCESS,
            "cublasSetPointerMode");
    require(cc::cublasSetStream(handle, nullptr) == CUBLAS_STATUS_SUCCESS, "cublasSetStream");
    std::vector<std::pair<std::string, Shape>> shapes;
    if (options.scenario == "suite") {
      for (unsigned a = 0; a < 2; ++a)
        for (unsigned b = 0; b < 2; ++b)
          shapes.emplace_back(std::string("smoke_") + (a ? "t" : "n") + (b ? "t" : "n"),
                              Shape{37, 19, 67, 5, 11, a != 0, b != 0});
    }
    if (options.scenario != "rejection")
      shapes.emplace_back("gemm", choose_shape(options, report.telemetry));
    for (const auto& [name, shape] : shapes) {
      Workload w;
      w.name = name;
      w.shape = shape;
      w.logical_bytes = *operand_bytes(shape);
      w.status = "running";
      const bool small = shape.m <= 128 && shape.n <= 128 && shape.k <= 512;
      w.reference_kind = small ? "cpu_fp64_full" : "cpu_fp64_analytic_periodic_v1";
      const auto before = snapshot();
      std::array<Matrix, 3> matrices{};
      std::array<void*, 3> pointers{};
      std::array<std::uint64_t, 3> allocation_ids{};
      const std::array<Operand, 3> operands{Operand::a, Operand::b, Operand::c};
      for (std::size_t i = 0; i < 3; ++i) {
        matrices[i] = *matrix_storage(shape, operands[i]);
        w.storage_bytes = checked_add(w.storage_bytes, matrices[i].elements * sizeof(float));
      }
      std::vector<float> staging(static_cast<std::size_t>(
          std::min(batch_elements,
                   std::max({matrices[0].elements, matrices[1].elements, matrices[2].elements}))));
      for (std::size_t i = 0; i < 3; ++i) {
        const auto bytes = matrices[i].elements * sizeof(float), op = ++next_operation;
        allocation_ids[i] = ++next_allocation;
        // Reserve ownership bookkeeping before a successful API allocation can escape.
        live_allocations.reserve(live_allocations.size() + 1U);
        emit("call", op, allocation_ids[i], bytes, "cudaMalloc");
        require(cc::cudaMalloc(&pointers[i], static_cast<std::size_t>(bytes)) == cudaSuccess,
                "cudaMalloc");
        live_allocations.push_back(pointers[i]);
        emit("return", op, allocation_ids[i], bytes, "cudaMalloc");
        for (std::uint64_t offset = 0; offset < matrices[i].elements;) {
          const auto count = std::min<std::uint64_t>(staging.size(), matrices[i].elements - offset);
          auto part = std::span<float>(staging.data(), static_cast<std::size_t>(count));
          fill_pattern(part, offset, shape, operands[i], options.seed);
          const auto copy_op = ++next_operation;
          emit("call", copy_op, allocation_ids[i], count * sizeof(float), "cudaMemcpyH2D");
          require(cc::cudaMemcpy(offset_pointer(pointers[i], offset), staging.data(),
                                 static_cast<std::size_t>(count * sizeof(float)),
                                 cudaMemcpyHostToDevice) == cudaSuccess,
                  "cudaMemcpy H2D");
          emit("return", copy_op, allocation_ids[i], count * sizeof(float), "cudaMemcpyH2D");
          offset += count;
        }
      }
      for (std::uint32_t pass = 0; pass < options.passes; ++pass) {
        const auto operation = ++next_operation;
        emit("call", operation, std::nullopt, w.logical_bytes, "cublasSgemm");
        const auto start = Clock::now();
        require(cc::cublasSgemm(
                    handle, shape.transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N,
                    shape.transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N, static_cast<int>(shape.m),
                    static_cast<int>(shape.n), static_cast<int>(shape.k), &options.alpha,
                    offset_pointer(pointers[0], shape.offset), static_cast<int>(matrices[0].ld),
                    offset_pointer(pointers[1], shape.offset), static_cast<int>(matrices[1].ld),
                    &options.beta, offset_pointer(pointers[2], shape.offset),
                    static_cast<int>(matrices[2].ld)) == CUBLAS_STATUS_SUCCESS,
                "cublasSgemm");
        w.pass_timings_ms.push_back(
            std::chrono::duration<double, std::milli>(Clock::now() - start).count());
        ++w.passes_completed;
        emit("return", operation, std::nullopt, w.logical_bytes, "cublasSgemm");
      }
      NativeBaseline baseline;
      if (small) {
        baseline = native_baseline(options, shape);
        if (!baseline.completed)
          throw Failure(27, baseline.error);
        w.native_baseline_equal = true;
        w.native_baseline_ms = baseline.milliseconds;
      }
      Digest actual_digest, expected_digest;
      for (std::uint64_t offset = 0; offset < matrices[2].elements;) {
        const auto count = std::min<std::uint64_t>(staging.size(), matrices[2].elements - offset);
        const auto operation = ++next_operation;
        emit("call", operation, allocation_ids[2], count * sizeof(float), "cudaMemcpyD2H");
        require(cc::cudaMemcpy(staging.data(), offset_pointer(pointers[2], offset),
                               static_cast<std::size_t>(count * sizeof(float)),
                               cudaMemcpyDeviceToHost) == cudaSuccess,
                "cudaMemcpy D2H");
        emit("return", operation, allocation_ids[2], count * sizeof(float), "cudaMemcpyD2H");
        for (std::uint64_t i = 0; i < count; ++i) {
          const auto index = offset + i;
          bool material = false;
          double expected = padding_value;
          if (index >= shape.offset) {
            const auto local = index - shape.offset, row = local % matrices[2].ld,
                       column = local / matrices[2].ld;
            material = row < shape.m && column < shape.n;
            if (material)
              expected = reference_result(shape, row, column, options.seed, options.alpha,
                                          options.beta, w.passes_completed, small);
          }
          const auto actual = staging[static_cast<std::size_t>(i)];
          actual_digest.add(actual);
          expected_digest.add(static_cast<float>(expected));
          bool okay = material ? close_enough(actual, expected)
                               : std::bit_cast<std::uint32_t>(actual) ==
                                     std::bit_cast<std::uint32_t>(padding_value);
          if (material) {
            ++w.output_elements_checked;
            const auto error = std::abs(static_cast<double>(actual) - expected);
            const auto finite_error =
                std::isfinite(error) ? error : std::numeric_limits<double>::max();
            w.max_absolute_error = std::max(w.max_absolute_error, finite_error);
            const auto relative = finite_error / std::max(std::abs(expected), 1e-30);
            w.max_relative_error =
                std::max(w.max_relative_error,
                         std::isfinite(relative) ? relative : std::numeric_limits<double>::max());
          } else
            ++w.padding_elements_checked;
          if (small && !close_enough(actual, baseline.c[static_cast<std::size_t>(index)])) {
            okay = false;
            w.native_baseline_equal = false;
          }
          if (!okay) {
            ++w.mismatches;
            if (!w.mismatch_offset)
              w.mismatch_offset = index * sizeof(float);
          }
        }
        offset += count;
      }
      w.digest = actual_digest.string();
      w.reference_digest = expected_digest.string();
      for (std::size_t i = 0; i < 3; ++i) {
        const auto operation = ++next_operation;
        emit("call", operation, allocation_ids[i], matrices[i].elements * sizeof(float),
             "cudaFree");
        require(cc::cudaFree(pointers[i]) == cudaSuccess, "cudaFree");
        live_allocations.erase(
            std::find(live_allocations.begin(), live_allocations.end(), pointers[i]));
        emit("return", operation, allocation_ids[i], matrices[i].elements * sizeof(float),
             "cudaFree");
      }
      const auto after = snapshot();
      w.tiles_retired = after.tiles_retired - before.tiles_retired;
      w.mappings = after.runtime.mappings - before.runtime.mappings;
      w.unmaps = after.runtime.unmaps - before.runtime.unmaps;
      w.h2d_bytes = after.runtime.bytes_h2d - before.runtime.bytes_h2d;
      w.d2h_bytes = after.runtime.bytes_d2h - before.runtime.bytes_d2h;
      w.evictions = after.runtime.clean_evictions + after.runtime.dirty_evictions -
                    before.runtime.clean_evictions - before.runtime.dirty_evictions;
      w.handle_reuses = after.runtime.handle_reuses - before.runtime.handle_reuses;
      w.status = w.mismatches == 0 ? "completed" : "corruption";
      report.workloads.push_back(std::move(w));
      if (report.workloads.back().mismatches != 0)
        throw Failure(24, "CPU reference, ordinary cuBLAS, or padding verification failed");
    }
    if (options.scenario == "suite" || options.scenario == "rejection") {
      const auto reject = [&](const auto& call) {
        const auto before = snapshot();
        const bool rejected = call();
        const auto after = snapshot();
        ++report.rejections_checked;
        if (!rejected || after.tiles_submitted != before.tiles_submitted ||
            after.runtime.mappings != before.runtime.mappings ||
            after.runtime.bytes_h2d != before.runtime.bytes_h2d ||
            after.runtime.bytes_d2h != before.runtime.bytes_d2h)
          ++report.rejections_failed;
        static_cast<void>(cc::cudaGetLastError());
      };
      reject([&] { return cc::cudaSetDevice(options.device + 1) != cudaSuccess; });
      reject([&] {
        return cc::cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE) !=
               CUBLAS_STATUS_SUCCESS;
      });
      reject([&] {
        return cc::cublasSetStream(handle, reinterpret_cast<cudaStream_t>(std::uintptr_t{1})) !=
               CUBLAS_STATUS_SUCCESS;
      });
      float foreign = 1.0F;
      reject([&] { return cc::cudaFree(&foreign) != cudaSuccess; });
      reject([&] {
        return cc::cudaMemcpy(&foreign, &foreign, sizeof(float), cudaMemcpyDeviceToDevice) !=
               cudaSuccess;
      });
      reject([&] {
        return cc::cudaMemcpy(&foreign, &foreign, sizeof(float), cudaMemcpyDefault) != cudaSuccess;
      });
      reject([&] {
        return cc::cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 0, 1, 1, &options.alpha, &foreign,
                               1, &foreign, 1, &options.beta, &foreign, 1) != CUBLAS_STATUS_SUCCESS;
      });
      reject([&] {
        return cc::cublasSgemm(handle, CUBLAS_OP_C, CUBLAS_OP_N, 1, 1, 1, &options.alpha, &foreign,
                               1, &foreign, 1, &options.beta, &foreign, 1) != CUBLAS_STATUS_SUCCESS;
      });
      if (report.rejections_failed != 0)
        throw Failure(27, "unsupported call was not rejected before submission");
    }
    require(cc::cudaDeviceSynchronize() == cudaSuccess, "cudaDeviceSynchronize");
    require(cc::cublasDestroy(handle) == CUBLAS_STATUS_SUCCESS, "cublasDestroy");
    handle = nullptr;
    report.exit_code = 0;
    report.reason = "proof_completed";
    report.message = "selected synchronous CUDA/cuBLAS compatibility proof completed";
  } catch (const Failure& failure) {
    report.exit_code = failure.code;
    report.reason = failure.code == 24   ? "reference_mismatch"
                    : failure.code == 25 ? "memory_or_budget"
                    : failure.code == 23 ? "prerequisite"
                                         : "execution_failure";
    report.message = failure.message;
  } catch (const std::bad_alloc&) {
    report.exit_code = 25;
    report.reason = "host_oom";
    report.message = "benchmark host allocation failed";
  } catch (const std::exception& exception) {
    report.exit_code = 70;
    report.reason = "internal_error";
    report.message = exception.what();
  } catch (...) {
    report.exit_code = 70;
    report.reason = "internal_error";
    report.message = "unknown benchmark exception";
  }
  if (initialized) {
    bool clean = true;
    for (void* pointer : live_allocations)
      clean = (cc::cudaFree(pointer) == cudaSuccess) && clean;
    if (handle)
      clean = (cc::cublasDestroy(handle) == CUBLAS_STATUS_SUCCESS) && clean;
    clean = (api.shutdown() == XVRAM_STATUS_SUCCESS) && clean;
    static_cast<void>(snapshot());
    if (!clean && report.exit_code == 0) {
      report.exit_code = 27;
      report.reason = "cleanup_failure";
      report.message = "compatibility shutdown failed";
    }
    complete_proof(report);
  } else {
    static_cast<void>(api.shutdown());
    static_cast<void>(snapshot());
  }
  return report;
}
} // namespace xvram::compat_bench
