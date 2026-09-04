#include "residency/cpu_codec_pool.hpp"
#include "residency/lz4_codec.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <thread>
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

using namespace std::chrono_literals;
using xvram::residency::AllocationId;
using xvram::residency::ChunkKey;
using xvram::residency::CodecStatus;
using xvram::residency::CpuCodecFunctions;
using xvram::residency::CpuCodecOperation;
using xvram::residency::CpuCodecPoolConfig;
using xvram::residency::CpuCodecPoolStatus;
using xvram::residency::CpuCodecRequest;
using xvram::residency::CpuCodecResult;
using xvram::residency::CpuCodecTicket;
using xvram::residency::CpuCodecWorkerPool;
using xvram::residency::Lz4BlockCodec;

constexpr ChunkKey key{AllocationId{7}, 3};

void configuration_and_status_tests() {
  Lz4BlockCodec codec;
  CpuCodecWorkerPool no_workers({0, 1}, codec);
  CpuCodecWorkerPool too_many_workers({9, 1}, codec);
  CpuCodecWorkerPool no_capacity({1, 0}, codec);
  CHECK(!no_workers.valid());
  CHECK(!too_many_workers.valid());
  CHECK(!no_capacity.valid());
  CHECK(xvram::residency::cpu_codec_operation_name(CpuCodecOperation::encode) == "encode");
  CHECK(xvram::residency::cpu_codec_pool_status_name(CpuCodecPoolStatus::queue_full) ==
        "queue_full");
}

void encode_decode_and_owned_input_test() {
  Lz4BlockCodec codec;
  CpuCodecWorkerPool pool({2, 4}, codec);
  CHECK(pool.valid());

  std::vector<std::byte> source(64 * 1024, std::byte{0x2A});
  CpuCodecTicket encode_ticket;
  CHECK(pool.submit_encode_copy(key, 11, source, encode_ticket) == CpuCodecPoolStatus::success);
  // submit_encode_copy owns a snapshot; later caller mutation cannot race the worker.
  source.assign(source.size(), std::byte{0x6B});

  CpuCodecResult encoded;
  CHECK(pool.wait(encode_ticket, 5s, &encoded) == CpuCodecPoolStatus::success);
  CHECK(encoded.ticket == encode_ticket);
  CHECK(encoded.output_bytes == encoded.output.size());
  CHECK(encoded.output_bytes < source.size());
  CHECK(!encoded.encoded_as_raw);
  CHECK(pool.poll(encode_ticket) == CpuCodecPoolStatus::stale_ticket);

  CpuCodecTicket decode_ticket;
  CHECK(pool.submit_decode(key, 11, std::move(encoded.output), source.size(), decode_ticket) ==
        CpuCodecPoolStatus::success);
  CpuCodecResult decoded;
  CHECK(pool.wait(decode_ticket, 5s, &decoded) == CpuCodecPoolStatus::success);
  CHECK(decoded.output.size() == source.size());
  CHECK(decoded.output != source);
  CHECK(decoded.output == std::vector<std::byte>(source.size(), std::byte{0x2A}));

  const auto telemetry = pool.telemetry();
  CHECK(telemetry.submitted == 2);
  CHECK(telemetry.encode_submitted == 1);
  CHECK(telemetry.decode_submitted == 1);
  CHECK(telemetry.completed == 2);
  CHECK(telemetry.retired == 2);
  CHECK(telemetry.failures == 0);
  CHECK(telemetry.outstanding == 0);
  CHECK(telemetry.queued == 0);
  CHECK(telemetry.running == 0);
  CHECK(pool.close() == CpuCodecPoolStatus::success);
  CHECK(pool.close() == CpuCodecPoolStatus::success);
}

void capacity_and_generation_safety_test() {
  Lz4BlockCodec codec;
  CpuCodecWorkerPool pool({1, 1}, codec);
  std::vector<std::byte> input(4096, std::byte{0x19});
  CpuCodecTicket first;
  CHECK(pool.submit_encode_copy(key, 21, input, first) == CpuCodecPoolStatus::success);
  // Completion observation does not release the bounded outstanding-job slot.
  CHECK(pool.wait(first, 5s) == CpuCodecPoolStatus::success);
  CpuCodecTicket rejected;
  CHECK(pool.submit_encode_copy(key, 22, input, rejected) == CpuCodecPoolStatus::queue_full);
  CHECK(!rejected);

  CpuCodecTicket altered = first;
  ++altered.source_generation;
  CHECK(pool.poll(altered) == CpuCodecPoolStatus::stale_ticket);
  altered = first;
  ++altered.pool_generation;
  CHECK(pool.poll(altered) == CpuCodecPoolStatus::stale_ticket);

  CpuCodecResult first_result;
  CHECK(pool.poll(first, &first_result) == CpuCodecPoolStatus::success);
  CpuCodecTicket second;
  CHECK(pool.submit_encode_copy(key, 22, input, second) == CpuCodecPoolStatus::success);
  CHECK(second.operation_id != first.operation_id);
  CpuCodecResult second_result;
  CHECK(pool.wait(second, 5s, &second_result) == CpuCodecPoolStatus::success);
  CHECK(pool.telemetry().queue_full == 1);
}

CpuCodecFunctions passthrough_functions() {
  CpuCodecFunctions functions;
  functions.maximum_compressed_bytes = [](const std::size_t bytes) { return bytes; };
  functions.compress = [](const std::span<const std::byte> input, const std::span<std::byte> output,
                          std::size_t& written) {
    std::copy(input.begin(), input.end(), output.begin());
    written = input.size();
    return CodecStatus::success;
  };
  functions.decompress = [](const std::span<const std::byte> input,
                            const std::span<std::byte> output) {
    if (input.size() != output.size()) {
      return CodecStatus::corrupt_input;
    }
    std::copy(input.begin(), input.end(), output.begin());
    return CodecStatus::success;
  };
  return functions;
}

void non_beneficial_encode_returns_owned_source_test() {
  CpuCodecWorkerPool pool({1, 2}, passthrough_functions());
  std::vector<std::byte> input(4096);
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = static_cast<std::byte>((index * 131U + 17U) & 0xffU);
  }
  CpuCodecTicket ticket;
  CHECK(pool.submit_encode_copy(key, 30, input, ticket) == CpuCodecPoolStatus::success);
  CpuCodecResult result;
  CHECK(pool.wait(ticket, 5s, &result) == CpuCodecPoolStatus::success);
  CHECK(result.encoded_as_raw);
  CHECK(result.output == input);
  CHECK(result.output_bytes == input.size());
  CHECK(pool.close() == CpuCodecPoolStatus::success);
}

void exception_and_codec_failure_tests() {
  CpuCodecFunctions throwing = passthrough_functions();
  throwing.compress = [](std::span<const std::byte>, std::span<std::byte>,
                         std::size_t&) -> CodecStatus {
    throw std::runtime_error("injected codec failure");
  };
  CpuCodecWorkerPool exception_pool({1, 2}, std::move(throwing));
  CpuCodecTicket ticket;
  CHECK(exception_pool.submit_encode_copy(key, 31, std::vector<std::byte>(128, std::byte{0x01}),
                                          ticket) == CpuCodecPoolStatus::success);
  CpuCodecResult result;
  CHECK(exception_pool.wait(ticket, 5s, &result) == CpuCodecPoolStatus::internal_failure);
  CHECK(result.status == CpuCodecPoolStatus::internal_failure);
  CHECK(result.codec_status == CodecStatus::internal_failure);
  CHECK(exception_pool.telemetry().exception_failures == 1);

  CpuCodecFunctions rejecting = passthrough_functions();
  rejecting.decompress = [](std::span<const std::byte>, std::span<std::byte>) {
    return CodecStatus::corrupt_input;
  };
  CpuCodecWorkerPool codec_failure_pool({1, 2}, std::move(rejecting));
  CHECK(codec_failure_pool.submit_decode(key, 32, std::vector<std::byte>(32, std::byte{0x4A}), 64,
                                         ticket) == CpuCodecPoolStatus::success);
  CHECK(codec_failure_pool.wait(ticket, 5s, &result) == CpuCodecPoolStatus::codec_failure);
  CHECK(result.codec_status == CodecStatus::corrupt_input);
  CHECK(result.output.empty());
}

void deterministic_close_drains_jobs_test() {
  std::atomic<std::uint32_t> calls{0};
  CpuCodecFunctions functions = passthrough_functions();
  functions.compress = [&calls](const std::span<const std::byte> input,
                                const std::span<std::byte> output, std::size_t& written) {
    std::this_thread::sleep_for(5ms);
    std::copy(input.begin(), input.end(), output.begin());
    written = input.size();
    calls.fetch_add(1, std::memory_order_relaxed);
    return CodecStatus::success;
  };
  CpuCodecWorkerPool pool({2, 4}, std::move(functions));
  std::vector<CpuCodecTicket> tickets(4);
  for (std::size_t index = 0; index < tickets.size(); ++index) {
    CHECK(pool.submit(CpuCodecRequest{CpuCodecOperation::encode, key,
                                      static_cast<std::uint64_t>(40 + index),
                                      std::vector<std::byte>(1024, std::byte{0x3C}), 0},
                      tickets[index]) == CpuCodecPoolStatus::success);
  }
  CHECK(pool.close() == CpuCodecPoolStatus::success);
  CHECK(calls.load(std::memory_order_relaxed) == tickets.size());
  for (const CpuCodecTicket& ticket : tickets) {
    CpuCodecResult result;
    CHECK(pool.poll(ticket, &result) == CpuCodecPoolStatus::success);
  }
  CpuCodecTicket rejected;
  CHECK(pool.submit_encode_copy(key, 99, std::vector<std::byte>(16), rejected) ==
        CpuCodecPoolStatus::closed);
  CHECK(pool.telemetry().completed == tickets.size());
}

void timeout_test() {
  CpuCodecFunctions functions = passthrough_functions();
  functions.compress = [](const std::span<const std::byte> input, const std::span<std::byte> output,
                          std::size_t& written) {
    std::this_thread::sleep_for(25ms);
    std::copy(input.begin(), input.end(), output.begin());
    written = input.size();
    return CodecStatus::success;
  };
  CpuCodecWorkerPool pool({1, 1}, std::move(functions));
  CpuCodecTicket ticket;
  CHECK(pool.submit_encode_copy(key, 77, std::vector<std::byte>(32), ticket) ==
        CpuCodecPoolStatus::success);
  CHECK(pool.wait(ticket, 0ms) == CpuCodecPoolStatus::timeout);
  CpuCodecResult result;
  CHECK(pool.wait(ticket, 5s, &result) == CpuCodecPoolStatus::success);
  CHECK(pool.wait(ticket, -1ms) == CpuCodecPoolStatus::invalid_argument);
}

} // namespace

int main() {
  configuration_and_status_tests();
  encode_decode_and_owned_input_test();
  capacity_and_generation_safety_test();
  non_beneficial_encode_returns_owned_source_test();
  exception_and_codec_failure_tests();
  deterministic_close_drains_jobs_test();
  timeout_test();
  return failures == 0 ? 0 : 1;
}
