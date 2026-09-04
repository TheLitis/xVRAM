#include "compression_bench/options.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(const bool condition, const char* expression, const int line) {
  if (!condition) {
    std::cerr << "line " << line << ": check failed: " << expression << '\n';
    ++failures;
  }
}

#define CHECK(expression) check(static_cast<bool>(expression), #expression, __LINE__)

[[nodiscard]] bool parse(std::vector<std::string> arguments,
                         xvram::compression::CliOptions& options, std::string& error) {
  std::vector<char*> argv;
  argv.reserve(arguments.size());
  for (std::string& item : arguments) {
    argv.push_back(item.data());
  }
  return xvram::compression::parse_cli(static_cast<int>(argv.size()), argv.data(), options, error);
}

void default_options_test() {
  xvram::compression::CliOptions options;
  std::string error;
  CHECK(parse({"xvram-compression-bench"}, options, error));
  CHECK(options.executor.device_ordinal == 0);
  CHECK(!options.executor.logical_bytes.has_value());
  CHECK(options.executor.chunk_bytes == 64ULL * 1024ULL * 1024ULL);
  CHECK(options.executor.compression_policy ==
        xvram::compression::RequestedCompressionPolicy::adaptive);
  CHECK(options.executor.path == xvram::compression::RequestedPath::automatic);
  CHECK(options.timeout == std::chrono::seconds(900));
}

void complete_options_test() {
  xvram::compression::CliOptions options;
  std::string error;
  CHECK(parse({"xvram-compression-bench",
               "--device",
               "2",
               "--logical-size",
               "3GiB",
               "--chunk-size",
               "32MiB",
               "--cache-target",
               "1GiB",
               "--compression-policy",
               "both",
               "--path",
               "gpu-lz4",
               "--codec",
               "lz4",
               "--policy",
               "both",
               "--scenario",
               "mixed",
               "--passes",
               "4",
               "--host-store-cap",
               "5GiB",
               "--host-headroom",
               "1GiB",
               "--codec-slots",
               "3",
               "--codec-workers",
               "4",
               "--timeout-seconds",
               "60",
               "--json",
               "-",
               "--trace",
               "trace.jsonl",
               "--include-identifiers"},
              options, error));
  CHECK(options.executor.device_ordinal == 2);
  CHECK(options.executor.logical_bytes == 3ULL * 1024ULL * 1024ULL * 1024ULL);
  CHECK(options.executor.compression_policy ==
        xvram::compression::RequestedCompressionPolicy::both);
  CHECK(options.executor.path == xvram::compression::RequestedPath::gpu_lz4);
  CHECK(options.executor.replacement_policy ==
        xvram::compression::RequestedReplacementPolicy::both);
  CHECK(options.executor.scenario == xvram::compression::ScenarioKind::mixed);
  CHECK(options.executor.trace_enabled);
  CHECK(options.executor.include_identifiers);
  CHECK(options.timeout == std::chrono::seconds(60));

  const std::vector<std::string> worker = xvram::compression::worker_arguments(options);
  CHECK(!worker.empty() && worker.front() == "--worker");
  CHECK(std::find(worker.begin(), worker.end(), "--trace-enabled") != worker.end());
  CHECK(std::find(worker.begin(), worker.end(), "--json") == worker.end());
  CHECK(std::find(worker.begin(), worker.end(), "--trace") == worker.end());
  CHECK(std::find(worker.begin(), worker.end(), "nvcomp_gpu_codec") != worker.end());
}

void rejection_tests() {
  {
    xvram::compression::CliOptions options;
    std::string error;
    CHECK(!parse({"bench", "--codec-slots", "1"}, options, error));
  }
  {
    xvram::compression::CliOptions options;
    std::string error;
    CHECK(!parse({"bench", "--trace", "-"}, options, error));
  }
  {
    xvram::compression::CliOptions options;
    std::string error;
    CHECK(!parse({"bench", "--no-text"}, options, error));
  }
  {
    xvram::compression::CliOptions options;
    std::string error;
    CHECK(!parse({"bench", "--help", "--version"}, options, error));
  }
}

} // namespace

int main() {
  default_options_test();
  complete_options_test();
  rejection_tests();
  if (failures != 0) {
    std::cerr << failures << " compression options test(s) failed\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
