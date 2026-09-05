#include "compat_bench/options.hpp"
#include "xvram/base/size_parser.hpp"
#include <charconv>
#include <cmath>
#include <limits>
#include <string_view>
#include <type_traits>

namespace xvram::compat_bench {
namespace {
bool unsigned_value(std::string_view value, std::uint64_t& output, int base = 10) {
  if (base == 16 && value.starts_with("0x"))
    value.remove_prefix(2);
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(), output, base);
  return parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size();
}
bool float_value(std::string_view value, float& output) {
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(), output);
  return parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size() &&
         std::isfinite(output);
}
} // namespace
bool parse_options(const int argc, char** argv, Options& o, std::string& error) {
  for (int i = 1; i < argc; ++i) {
    const std::string_view a(argv[i]);
    if (a == "--help" || a == "-h") {
      o.help = true;
      continue;
    }
    if (a == "--version") {
      o.version = true;
      continue;
    }
    if (a == "--worker") {
      o.worker = true;
      continue;
    }
    if (a == "--compact-json") {
      o.pretty = false;
      continue;
    }
    if (a == "--no-text") {
      o.text = false;
      continue;
    }
    if (a == "--include-identifiers") {
      o.identifiers = true;
      continue;
    }
    if (++i >= argc) {
      error = std::string(a) + " requires a value";
      return false;
    }
    const std::string_view v(argv[i]);
    const auto fail = [&] {
      error = "invalid value for " + std::string(a);
      return false;
    };
    std::uint64_t u = 0;
    if (a == "--scenario") {
      if (v != "suite" && v != "gemm" && v != "rejection")
        return fail();
      o.scenario = v;
    } else if (a == "--policy") {
      if (v != "clock" && v != "lru")
        return fail();
      o.policy = v;
    } else if (a == "--op-a" || a == "--op-b") {
      if (v != "n" && v != "t")
        return fail();
      (a == "--op-a" ? o.transpose_a : o.transpose_b) = v == "t";
    } else if (a == "--alpha" || a == "--beta") {
      if (!float_value(v, a == "--alpha" ? o.alpha : o.beta))
        return fail();
    } else if (a == "--json") {
      if (v.empty())
        return fail();
      o.json_path = v;
    } else if (a == "--trace") {
      if (v.empty() || v == "-")
        return fail();
      o.trace_path = v;
    } else if (a == "--cublas-library")
      o.cublas_library = v;
    else if (a == "--cublas-lt-library")
      o.cublas_lt_library = v;
    else if (a == "--test-worker")
      o.test_worker = v;
    else if (a == "--logical-size" || a == "--cache-target" || a == "--chunk-size" ||
             a == "--device-headroom" || a == "--workspace") {
      if (v == "auto" && a == "--logical-size") {
        o.logical_bytes.reset();
        continue;
      }
      if (v == "auto" && a == "--cache-target") {
        o.cache_bytes = 0;
        continue;
      }
      const auto parsed = parse_size(v);
      if (!parsed || (*parsed.bytes == 0 && a != "--workspace" && a != "--device-headroom"))
        return fail();
      u = *parsed.bytes;
      if (a == "--logical-size")
        o.logical_bytes = u;
      else if (a == "--cache-target")
        o.cache_bytes = u;
      else if (a == "--chunk-size")
        o.chunk_bytes = u;
      else if (a == "--workspace")
        o.workspace_bytes = u;
      else
        o.headroom_bytes = u;
    } else {
      if (!unsigned_value(v, u, a == "--seed" ? 16 : 10))
        return fail();
      if (a == "--device") {
        if (u > INT32_MAX)
          return fail();
        o.device = static_cast<std::int32_t>(u);
      } else if (a == "--m" || a == "--n" || a == "--k") {
        if (u == 0 || u > INT32_MAX)
          return fail();
        (a == "--m" ? o.m : a == "--n" ? o.n : o.k) = u;
      } else if (a == "--padding") {
        if (u > INT32_MAX)
          return fail();
        o.padding = u;
      } else if (a == "--offset-elements") {
        if (u > INT32_MAX)
          return fail();
        o.offset_elements = u;
      } else if (a == "--seed")
        o.seed = u;
      else if (a == "--passes") {
        if (u < 1 || u > 8)
          return fail();
        o.passes = static_cast<std::uint32_t>(u);
      } else if (a == "--staging-slots") {
        if (u < 2 || u > 8)
          return fail();
        o.staging_slots = static_cast<std::uint32_t>(u);
      } else if (a == "--prefetch-distance") {
        if (u > 8)
          return fail();
        o.prefetch_distance = static_cast<std::uint32_t>(u);
      } else if (a == "--budget-poll-ms") {
        if (u == 0 || u > 60000)
          return fail();
        o.budget_poll_ms = u;
      } else if (a == "--stall-timeout-ms") {
        if (u == 0 || u > 900000)
          return fail();
        o.stall_timeout_ms = u;
      } else if (a == "--timeout-seconds") {
        if (u == 0 || u > 86400)
          return fail();
        o.timeout = std::chrono::milliseconds(u * 1000);
      } else {
        error = "unknown argument: " + std::string(a);
        return false;
      }
    }
  }
  const auto dimensions = static_cast<unsigned>(o.m.has_value()) +
                          static_cast<unsigned>(o.n.has_value()) +
                          static_cast<unsigned>(o.k.has_value());
  if ((dimensions != 0 && dimensions != 3) || (dimensions != 0 && o.logical_bytes.has_value())) {
    error = "specify all M/N/K or logical-size, never both";
    return false;
  }
  if ((o.help || o.version) && argc != 2) {
    error = "help/version must be used alone";
    return false;
  }
  if (!o.text && !o.json_path && !o.worker) {
    error = "--no-text requires --json";
    return false;
  }
  if (o.cublas_library.empty() != o.cublas_lt_library.empty()) {
    error = "both cuBLAS library paths are required";
    return false;
  }
  if (o.json_path && o.trace_path && *o.json_path == *o.trace_path) {
    error = "JSON and trace output paths must differ";
    return false;
  }
  return true;
}
std::vector<std::string> worker_arguments(const Options& o) {
  std::vector<std::string> args{"--worker"};
  const auto put = [&](const char* key, auto value) {
    args.emplace_back(key);
    if constexpr (std::is_floating_point_v<decltype(value)>) {
      char buffer[64]{};
      const auto encoded =
          std::to_chars(buffer, buffer + sizeof(buffer), value, std::chars_format::general,
                        std::numeric_limits<decltype(value)>::max_digits10);
      args.emplace_back(buffer, encoded.ptr);
    } else {
      args.emplace_back(std::to_string(value));
    }
  };
  put("--device", o.device);
  put("--chunk-size", o.chunk_bytes);
  put("--device-headroom", o.headroom_bytes);
  put("--workspace", o.workspace_bytes);
  if (o.cache_bytes != 0)
    put("--cache-target", o.cache_bytes);
  if (o.logical_bytes)
    put("--logical-size", *o.logical_bytes);
  if (o.m) {
    put("--m", *o.m);
    put("--n", *o.n);
    put("--k", *o.k);
  }
  put("--padding", o.padding);
  put("--offset-elements", o.offset_elements);
  put("--passes", o.passes);
  put("--staging-slots", o.staging_slots);
  put("--prefetch-distance", o.prefetch_distance);
  put("--budget-poll-ms", o.budget_poll_ms);
  put("--stall-timeout-ms", o.stall_timeout_ms);
  put("--timeout-seconds", o.timeout.count() / 1000);
  put("--alpha", o.alpha);
  put("--beta", o.beta);
  char seed[17]{};
  const auto encoded = std::to_chars(seed, seed + 16, o.seed, 16);
  args.insert(args.end(), {"--seed", std::string(seed, encoded.ptr), "--policy", o.policy,
                           "--scenario", o.scenario, "--op-a", o.transpose_a ? "t" : "n", "--op-b",
                           o.transpose_b ? "t" : "n"});
  if (o.trace_path)
    args.insert(args.end(), {"--trace", "controller-owned"});
  if (o.identifiers)
    args.emplace_back("--include-identifiers");
  if (!o.cublas_library.empty())
    args.insert(args.end(),
                {"--cublas-library", o.cublas_library, "--cublas-lt-library", o.cublas_lt_library});
  if (!o.test_worker.empty())
    args.insert(args.end(), {"--test-worker", o.test_worker});
  return args;
}
const char* help_text() noexcept {
  return R"(xvram-compat-bench - isolated synchronous CUDA/cuBLAS compatibility proof
Options:
  --scenario <suite|gemm|rejection>  --device <ordinal>  --policy <clock|lru>
  --logical-size <auto|size>        or all --m <n> --n <n> --k <n>
  --op-a <n|t> --op-b <n|t>        --padding <elements> --offset-elements <n>
  --alpha <finite> --beta <finite> --passes <1..8> --seed <hex-u64>
  --chunk-size <size>              --cache-target <auto|size>
  --device-headroom <size>         --workspace <size> --staging-slots <2..8>
  --prefetch-distance <0..8>       --budget-poll-ms <n> --stall-timeout-ms <n>
  --timeout-seconds <n>            --trace <path> --json <path|->
  --compact-json --no-text --include-identifiers --version --help
  --cublas-library <absolute-path> --cublas-lt-library <absolute-path>
Defaults: suite, CLOCK, raw-safe 1.5x VRAM, M=4096 N=16, FP32, alpha=1.25 beta=.5,
two passes, 64MiB chunks, 512MiB headroom, 4MiB workspace, timeout 900 seconds.
Only explicitly integrated synchronous CUDA/cuBLAS calls are supported. No native fallback.
)";
}
std::filesystem::path utf8_path(const std::string_view text) {
  const auto* first = reinterpret_cast<const char8_t*>(text.data());
  return std::filesystem::path(std::u8string(first, first + text.size()));
}
} // namespace xvram::compat_bench
