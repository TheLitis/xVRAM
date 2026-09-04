#include "residency/cpu_codec_pool.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <limits>
#include <mutex>
#include <new>
#include <thread>
#include <unordered_map>
#include <utility>

namespace xvram::residency {
namespace {

std::atomic<std::uint64_t> next_pool_generation{1};
std::atomic<std::uint64_t> next_operation_id{1};

[[nodiscard]] std::uint64_t take_monotonic_id(std::atomic<std::uint64_t>& source) noexcept {
  std::uint64_t current = source.load(std::memory_order_relaxed);
  for (;;) {
    if (current == 0 || current == std::numeric_limits<std::uint64_t>::max()) {
      return 0;
    }
    if (source.compare_exchange_weak(current, current + 1, std::memory_order_relaxed,
                                     std::memory_order_relaxed)) {
      return current;
    }
  }
}

[[nodiscard]] bool add_saturated(std::uint64_t& value, const std::size_t increment) noexcept {
  const auto converted = static_cast<std::uint64_t>(increment);
  if (converted > std::numeric_limits<std::uint64_t>::max() - value) {
    value = std::numeric_limits<std::uint64_t>::max();
    return false;
  }
  value += converted;
  return true;
}

} // namespace

std::string_view cpu_codec_operation_name(const CpuCodecOperation operation) noexcept {
  switch (operation) {
  case CpuCodecOperation::encode:
    return "encode";
  case CpuCodecOperation::decode:
    return "decode";
  }
  return "unknown";
}

std::string_view cpu_codec_pool_status_name(const CpuCodecPoolStatus status) noexcept {
  switch (status) {
  case CpuCodecPoolStatus::success:
    return "success";
  case CpuCodecPoolStatus::not_ready:
    return "not_ready";
  case CpuCodecPoolStatus::timeout:
    return "timeout";
  case CpuCodecPoolStatus::invalid_configuration:
    return "invalid_configuration";
  case CpuCodecPoolStatus::invalid_argument:
    return "invalid_argument";
  case CpuCodecPoolStatus::queue_full:
    return "queue_full";
  case CpuCodecPoolStatus::stale_ticket:
    return "stale_ticket";
  case CpuCodecPoolStatus::closed:
    return "closed";
  case CpuCodecPoolStatus::allocation_failure:
    return "allocation_failure";
  case CpuCodecPoolStatus::codec_failure:
    return "codec_failure";
  case CpuCodecPoolStatus::internal_failure:
    return "internal_failure";
  }
  return "unknown";
}

bool CpuCodecFunctions::valid() const noexcept {
  return static_cast<bool>(maximum_compressed_bytes) && static_cast<bool>(compress) &&
         static_cast<bool>(decompress);
}

CpuCodecFunctions CpuCodecFunctions::from_block_codec(const BlockCodec& codec) {
  CpuCodecFunctions result;
  result.maximum_compressed_bytes = [&codec](const std::size_t input_bytes) {
    return codec.maximum_compressed_bytes(input_bytes);
  };
  result.compress = [&codec](const std::span<const std::byte> input,
                             const std::span<std::byte> output, std::size_t& written_bytes) {
    return codec.compress(input, output, written_bytes);
  };
  result.decompress = [&codec](const std::span<const std::byte> input,
                               const std::span<std::byte> output) {
    return codec.decompress(input, output);
  };
  return result;
}

struct CpuCodecWorkerPool::Impl {
  struct Job {
    CpuCodecRequest request;
    CpuCodecTicket ticket;
    CpuCodecResult result;
    bool running = false;
    bool complete = false;
    bool claimed = false;
  };

  CpuCodecPoolConfig config;
  CpuCodecFunctions functions;
  std::uint64_t pool_generation = 0;
  bool valid = false;
  bool accepting = false;
  bool closing = false;
  bool joined = false;
  std::size_t running = 0;
  mutable std::mutex mutex;
  std::mutex close_mutex;
  std::condition_variable work_ready;
  std::condition_variable completion_ready;
  std::deque<std::shared_ptr<Job>> queue;
  std::unordered_map<std::uint64_t, std::shared_ptr<Job>> jobs;
  std::vector<std::thread> workers;
  CpuCodecPoolTelemetry telemetry;

  Impl(const CpuCodecPoolConfig value_config, CpuCodecFunctions value_functions)
      : config(value_config), functions(std::move(value_functions)) {
    if (config.worker_count < cpu_codec_min_workers ||
        config.worker_count > cpu_codec_max_workers || config.queue_capacity == 0 ||
        config.queue_capacity > cpu_codec_max_queue_capacity || !functions.valid()) {
      return;
    }
    pool_generation = take_monotonic_id(next_pool_generation);
    if (pool_generation == 0) {
      return;
    }
    accepting = true;
    try {
      workers.reserve(config.worker_count);
      for (std::uint32_t index = 0; index < config.worker_count; ++index) {
        workers.emplace_back([this] { worker_loop(); });
      }
      valid = true;
    } catch (...) {
      accepting = false;
      closing = true;
      work_ready.notify_all();
      for (std::thread& worker : workers) {
        if (worker.joinable()) {
          worker.join();
        }
      }
      workers.clear();
    }
  }

  ~Impl() {
    static_cast<void>(close());
  }

  [[nodiscard]] bool ticket_matches(const CpuCodecTicket& ticket, const Job& job) const noexcept {
    return ticket.pool_generation == pool_generation && ticket == job.ticket;
  }

  void worker_loop() noexcept {
    for (;;) {
      std::shared_ptr<Job> job;
      {
        std::unique_lock lock(mutex);
        work_ready.wait(lock, [this] { return closing || !queue.empty(); });
        if (queue.empty()) {
          if (closing) {
            return;
          }
          continue;
        }
        job = std::move(queue.front());
        queue.pop_front();
        job->running = true;
        ++running;
        telemetry.peak_running =
            std::max(telemetry.peak_running, static_cast<std::uint64_t>(running));
      }

      execute(*job);

      {
        std::lock_guard lock(mutex);
        job->running = false;
        job->complete = true;
        --running;
        ++telemetry.completed;
        if (job->result.status != CpuCodecPoolStatus::success) {
          ++telemetry.failures;
        }
        static_cast<void>(add_saturated(telemetry.input_bytes, job->result.input_bytes));
        static_cast<void>(add_saturated(telemetry.output_bytes, job->result.output_bytes));
      }
      completion_ready.notify_all();
    }
  }

  void execute(Job& job) noexcept {
    CpuCodecResult result;
    result.ticket = job.ticket;
    result.input_bytes = job.request.input.size();
    try {
      if (job.request.operation == CpuCodecOperation::encode) {
        const std::optional<std::size_t> maximum =
            functions.maximum_compressed_bytes(job.request.input.size());
        if (!maximum || *maximum == 0) {
          result.status = CpuCodecPoolStatus::codec_failure;
          result.codec_status = CodecStatus::invalid_argument;
          job.result = std::move(result);
          return;
        }
        result.output.resize(*maximum);
        std::size_t written = 0;
        result.codec_status = functions.compress(job.request.input, result.output, written);
        if (result.codec_status != CodecStatus::success || written == 0 ||
            written > result.output.size()) {
          result.output.clear();
          result.status = CpuCodecPoolStatus::codec_failure;
          job.result = std::move(result);
          return;
        }
        // resize() is not permitted to retain compressBound capacity here. The result becomes
        // authoritative backing, whose admission is charged by its stored byte count; retaining
        // the oversized allocation would silently defeat the host-store cap on compressible data.
        std::vector<std::byte> exact(written);
        std::copy_n(result.output.begin(), written, exact.begin());
        result.output.swap(exact);
      } else if (job.request.operation == CpuCodecOperation::decode) {
        result.output.resize(job.request.decoded_bytes);
        result.codec_status = functions.decompress(job.request.input, result.output);
        if (result.codec_status != CodecStatus::success) {
          result.output.clear();
          result.status = CpuCodecPoolStatus::codec_failure;
          job.result = std::move(result);
          return;
        }
      } else {
        result.status = CpuCodecPoolStatus::invalid_argument;
        result.codec_status = CodecStatus::invalid_argument;
        job.result = std::move(result);
        return;
      }
      result.output_bytes = result.output.size();
      result.status = CpuCodecPoolStatus::success;
      job.result = std::move(result);
    } catch (const std::bad_alloc&) {
      result.output.clear();
      result.status = CpuCodecPoolStatus::allocation_failure;
      result.codec_status = CodecStatus::internal_failure;
      job.result = std::move(result);
      std::lock_guard lock(mutex);
      ++telemetry.exception_failures;
    } catch (...) {
      result.output.clear();
      result.status = CpuCodecPoolStatus::internal_failure;
      result.codec_status = CodecStatus::internal_failure;
      job.result = std::move(result);
      std::lock_guard lock(mutex);
      ++telemetry.exception_failures;
    }
  }

  [[nodiscard]] CpuCodecPoolStatus submit(CpuCodecRequest request,
                                          CpuCodecTicket& ticket) noexcept {
    ticket = {};
    if (!valid) {
      return CpuCodecPoolStatus::invalid_configuration;
    }
    if ((request.operation != CpuCodecOperation::encode &&
         request.operation != CpuCodecOperation::decode) ||
        !request.key.allocation_id || request.source_generation == 0 || request.input.empty() ||
        (request.operation == CpuCodecOperation::decode && request.decoded_bytes == 0) ||
        (request.operation == CpuCodecOperation::encode && request.decoded_bytes != 0)) {
      return CpuCodecPoolStatus::invalid_argument;
    }
    try {
      std::lock_guard lock(mutex);
      if (!accepting) {
        return CpuCodecPoolStatus::closed;
      }
      if (jobs.size() >= config.queue_capacity) {
        ++telemetry.queue_full;
        return CpuCodecPoolStatus::queue_full;
      }
      const std::uint64_t operation_id = take_monotonic_id(next_operation_id);
      if (operation_id == 0) {
        return CpuCodecPoolStatus::internal_failure;
      }
      auto job = std::make_shared<Job>();
      job->request = std::move(request);
      job->ticket = CpuCodecTicket{pool_generation, operation_id, job->request.key,
                                   job->request.source_generation, job->request.operation};
      job->result.ticket = job->ticket;
      const auto [position, inserted] = jobs.emplace(operation_id, job);
      if (!inserted) {
        return CpuCodecPoolStatus::internal_failure;
      }
      try {
        queue.push_back(job);
      } catch (...) {
        jobs.erase(position);
        throw;
      }
      ++telemetry.submitted;
      if (job->request.operation == CpuCodecOperation::encode) {
        ++telemetry.encode_submitted;
      } else {
        ++telemetry.decode_submitted;
      }
      telemetry.peak_outstanding =
          std::max(telemetry.peak_outstanding, static_cast<std::uint64_t>(jobs.size()));
      ticket = job->ticket;
      work_ready.notify_one();
      return CpuCodecPoolStatus::success;
    } catch (const std::bad_alloc&) {
      return CpuCodecPoolStatus::allocation_failure;
    } catch (...) {
      return CpuCodecPoolStatus::internal_failure;
    }
  }

  [[nodiscard]] CpuCodecPoolStatus collect(const CpuCodecTicket& ticket, CpuCodecResult* result,
                                           const bool wait_for_completion,
                                           const std::chrono::milliseconds timeout) noexcept {
    if (!valid) {
      return CpuCodecPoolStatus::invalid_configuration;
    }
    if (!ticket || ticket.pool_generation != pool_generation) {
      return CpuCodecPoolStatus::stale_ticket;
    }
    try {
      std::unique_lock lock(mutex);
      const auto position = jobs.find(ticket.operation_id);
      if (position == jobs.end() || !ticket_matches(ticket, *position->second)) {
        return CpuCodecPoolStatus::stale_ticket;
      }
      const std::shared_ptr<Job> job = position->second;
      if (!job->complete && wait_for_completion) {
        if (!completion_ready.wait_for(lock, timeout, [&job] { return job->complete; })) {
          return CpuCodecPoolStatus::timeout;
        }
      }
      if (!job->complete) {
        return CpuCodecPoolStatus::not_ready;
      }
      const CpuCodecPoolStatus status = job->result.status;
      if (result != nullptr) {
        if (job->claimed) {
          return CpuCodecPoolStatus::stale_ticket;
        }
        job->claimed = true;
        *result = std::move(job->result);
        jobs.erase(ticket.operation_id);
        ++telemetry.retired;
      }
      return status;
    } catch (const std::bad_alloc&) {
      return CpuCodecPoolStatus::allocation_failure;
    } catch (...) {
      return CpuCodecPoolStatus::internal_failure;
    }
  }

  [[nodiscard]] CpuCodecPoolStatus close() noexcept {
    std::lock_guard close_lock(close_mutex);
    if (!valid) {
      return CpuCodecPoolStatus::invalid_configuration;
    }
    if (joined) {
      return CpuCodecPoolStatus::success;
    }
    {
      std::lock_guard lock(mutex);
      accepting = false;
      closing = true;
    }
    work_ready.notify_all();
    for (std::thread& worker : workers) {
      if (worker.joinable()) {
        worker.join();
      }
    }
    workers.clear();
    joined = true;
    completion_ready.notify_all();
    return CpuCodecPoolStatus::success;
  }
};

CpuCodecWorkerPool::CpuCodecWorkerPool(const CpuCodecPoolConfig config, const BlockCodec& codec)
    : CpuCodecWorkerPool(config, CpuCodecFunctions::from_block_codec(codec)) {}

CpuCodecWorkerPool::CpuCodecWorkerPool(const CpuCodecPoolConfig config, CpuCodecFunctions functions)
    : impl_(std::make_unique<Impl>(config, std::move(functions))) {}

CpuCodecWorkerPool::~CpuCodecWorkerPool() = default;

bool CpuCodecWorkerPool::valid() const noexcept {
  return impl_->valid;
}

const CpuCodecPoolConfig& CpuCodecWorkerPool::config() const noexcept {
  return impl_->config;
}

CpuCodecPoolStatus CpuCodecWorkerPool::submit(CpuCodecRequest request,
                                              CpuCodecTicket& ticket) noexcept {
  return impl_->submit(std::move(request), ticket);
}

CpuCodecPoolStatus CpuCodecWorkerPool::submit_encode(const ChunkKey key,
                                                     const std::uint64_t source_generation,
                                                     std::vector<std::byte> input,
                                                     CpuCodecTicket& ticket) noexcept {
  return submit(
      CpuCodecRequest{CpuCodecOperation::encode, key, source_generation, std::move(input), 0},
      ticket);
}

CpuCodecPoolStatus CpuCodecWorkerPool::submit_encode_copy(const ChunkKey key,
                                                          const std::uint64_t source_generation,
                                                          const std::span<const std::byte> input,
                                                          CpuCodecTicket& ticket) noexcept {
  try {
    return submit_encode(key, source_generation, std::vector<std::byte>(input.begin(), input.end()),
                         ticket);
  } catch (const std::bad_alloc&) {
    ticket = {};
    return CpuCodecPoolStatus::allocation_failure;
  } catch (...) {
    ticket = {};
    return CpuCodecPoolStatus::internal_failure;
  }
}

CpuCodecPoolStatus CpuCodecWorkerPool::submit_decode(const ChunkKey key,
                                                     const std::uint64_t source_generation,
                                                     std::vector<std::byte> input,
                                                     const std::size_t decoded_bytes,
                                                     CpuCodecTicket& ticket) noexcept {
  return submit(CpuCodecRequest{CpuCodecOperation::decode, key, source_generation, std::move(input),
                                decoded_bytes},
                ticket);
}

CpuCodecPoolStatus CpuCodecWorkerPool::submit_decode_copy(const ChunkKey key,
                                                          const std::uint64_t source_generation,
                                                          const std::span<const std::byte> input,
                                                          const std::size_t decoded_bytes,
                                                          CpuCodecTicket& ticket) noexcept {
  try {
    return submit_decode(key, source_generation, std::vector<std::byte>(input.begin(), input.end()),
                         decoded_bytes, ticket);
  } catch (const std::bad_alloc&) {
    ticket = {};
    return CpuCodecPoolStatus::allocation_failure;
  } catch (...) {
    ticket = {};
    return CpuCodecPoolStatus::internal_failure;
  }
}

CpuCodecPoolStatus CpuCodecWorkerPool::poll(const CpuCodecTicket& ticket,
                                            CpuCodecResult* result) noexcept {
  return impl_->collect(ticket, result, false, std::chrono::milliseconds{0});
}

CpuCodecPoolStatus CpuCodecWorkerPool::wait(const CpuCodecTicket& ticket,
                                            const std::chrono::milliseconds timeout,
                                            CpuCodecResult* result) noexcept {
  if (timeout.count() < 0) {
    return CpuCodecPoolStatus::invalid_argument;
  }
  return impl_->collect(ticket, result, true, timeout);
}

CpuCodecPoolStatus CpuCodecWorkerPool::close() noexcept {
  return impl_->close();
}

CpuCodecPoolTelemetry CpuCodecWorkerPool::telemetry() const noexcept {
  std::lock_guard lock(impl_->mutex);
  CpuCodecPoolTelemetry result = impl_->telemetry;
  result.outstanding = static_cast<std::uint64_t>(impl_->jobs.size());
  result.queued = static_cast<std::uint64_t>(impl_->queue.size());
  result.running = static_cast<std::uint64_t>(impl_->running);
  return result;
}

} // namespace xvram::residency
