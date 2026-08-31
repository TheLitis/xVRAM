#include "platform/worker_process.hpp"

#include <array>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <mutex>
#include <string_view>
#include <thread>

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#else
#include <cerrno>
#include <csignal>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace xvram::platform {
namespace {

struct ReaderState {
  explicit ReaderState(WorkerStreamConsumer& input_consumer) : consumer(input_consumer) {}

  std::mutex mutex;
  std::condition_variable changed;
  WorkerStreamConsumer& consumer;
  std::string error;
  std::chrono::steady_clock::time_point last_progress = std::chrono::steady_clock::now();
  bool ended = false;
};

void consume_bytes(ReaderState& state, const char* data, const std::size_t size) {
  bool made_progress = false;
  std::string error;
  if (!state.consumer.consume(data, size, made_progress, error)) {
    std::lock_guard lock(state.mutex);
    if (state.error.empty()) {
      state.error = error.empty() ? "worker stream consumer rejected input" : std::move(error);
    }
    state.changed.notify_all();
    return;
  }
  if (made_progress) {
    std::lock_guard lock(state.mutex);
    state.last_progress = std::chrono::steady_clock::now();
    state.changed.notify_all();
  }
}

void finish_reader(ReaderState& state) {
  std::string error;
  const bool complete = state.consumer.finish(error);
  std::lock_guard lock(state.mutex);
  if (!complete && state.error.empty()) {
    state.error = error.empty() ? "worker stream ended incompletely" : std::move(error);
  }
  state.ended = true;
  state.changed.notify_all();
}

#ifdef _WIN32

[[nodiscard]] std::string windows_error(const char* operation) {
  return std::string(operation) + " failed with Windows error " +
         std::to_string(static_cast<unsigned long>(GetLastError()));
}

[[nodiscard]] std::wstring utf8_to_wide(const std::string_view text) {
  if (text.empty()) {
    return {};
  }
  if (text.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return {};
  }
  const int input_size = static_cast<int>(text.size());
  const int size =
      MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), input_size, nullptr, 0);
  if (size <= 0) {
    return {};
  }
  std::wstring output(static_cast<std::size_t>(size), L'\0');
  if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), input_size, output.data(),
                          size) != size) {
    return {};
  }
  return output;
}

[[nodiscard]] std::wstring quote_windows_argument(const std::wstring& argument) {
  if (!argument.empty() && argument.find_first_of(L" \t\n\v\"") == std::wstring::npos) {
    return argument;
  }
  std::wstring output(1, L'\"');
  std::size_t backslashes = 0;
  for (const wchar_t character : argument) {
    if (character == L'\\') {
      ++backslashes;
      continue;
    }
    if (character == L'\"') {
      output.append(backslashes * 2U + 1U, L'\\');
      output.push_back(L'\"');
    } else {
      output.append(backslashes, L'\\');
      output.push_back(character);
    }
    backslashes = 0;
  }
  output.append(backslashes * 2U, L'\\');
  output.push_back(L'\"');
  return output;
}

void windows_reader(HANDLE pipe, ReaderState& state) {
  std::array<char, 16384> buffer{};
  for (;;) {
    DWORD read = 0;
    if (!ReadFile(pipe, buffer.data(), static_cast<DWORD>(buffer.size()), &read, nullptr)) {
      const DWORD code = GetLastError();
      if (code != ERROR_BROKEN_PIPE) {
        std::lock_guard lock(state.mutex);
        if (state.error.empty()) {
          state.error = "worker pipe read failed with Windows error " +
                        std::to_string(static_cast<unsigned long>(code));
        }
      }
      break;
    }
    if (read == 0) {
      break;
    }
    consume_bytes(state, buffer.data(), static_cast<std::size_t>(read));
  }
  CloseHandle(pipe);
  finish_reader(state);
}

#else

void posix_reader(const int pipe_handle, ReaderState& state) {
  std::array<char, 16384> buffer{};
  for (;;) {
    const ssize_t read_size = read(pipe_handle, buffer.data(), buffer.size());
    if (read_size > 0) {
      consume_bytes(state, buffer.data(), static_cast<std::size_t>(read_size));
      continue;
    }
    if (read_size < 0 && errno == EINTR) {
      continue;
    }
    if (read_size < 0) {
      std::lock_guard lock(state.mutex);
      if (state.error.empty()) {
        state.error = "worker pipe read failed with errno " + std::to_string(errno);
      }
    }
    break;
  }
  close(pipe_handle);
  finish_reader(state);
}

#endif

} // namespace

IsolatedWorkerResult run_isolated_worker_process(const IsolatedWorkerOptions& options,
                                                 WorkerStreamConsumer& consumer) {
  IsolatedWorkerResult result;
  result.process_exit_code = options.failure_exit_code;
  if (options.executable.empty() || options.no_progress_timeout.count() <= 0 ||
      options.overall_timeout.count() <= 0 || options.timeout_exit_code < 0 ||
      options.failure_exit_code < 0) {
    result.error = "worker controller options are invalid";
    return result;
  }

  ReaderState reader_state(consumer);
  const auto started_at = std::chrono::steady_clock::now();
  reader_state.last_progress = started_at;

#ifdef _WIN32
  SECURITY_ATTRIBUTES security{};
  security.nLength = sizeof(security);
  security.bInheritHandle = TRUE;
  HANDLE pipe_read = nullptr;
  HANDLE pipe_write = nullptr;
  if (!CreatePipe(&pipe_read, &pipe_write, &security, 0)) {
    result.error = windows_error("CreatePipe");
    return result;
  }
  if (!SetHandleInformation(pipe_read, HANDLE_FLAG_INHERIT, 0)) {
    result.error = windows_error("SetHandleInformation");
    CloseHandle(pipe_read);
    CloseHandle(pipe_write);
    return result;
  }

  HANDLE job = CreateJobObjectW(nullptr, nullptr);
  if (job == nullptr) {
    result.error = windows_error("CreateJobObjectW");
    CloseHandle(pipe_read);
    CloseHandle(pipe_write);
    return result;
  }
  JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
  limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
  if (!SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits))) {
    result.error = windows_error("SetInformationJobObject");
    CloseHandle(job);
    CloseHandle(pipe_read);
    CloseHandle(pipe_write);
    return result;
  }

  const std::wstring executable = options.executable.wstring();
  std::wstring command_line = quote_windows_argument(executable);
  for (const auto& argument : options.arguments) {
    const std::wstring wide = utf8_to_wide(argument);
    if (wide.empty() && !argument.empty()) {
      result.error = "worker argument is not valid UTF-8";
      CloseHandle(job);
      CloseHandle(pipe_read);
      CloseHandle(pipe_write);
      return result;
    }
    command_line.push_back(L' ');
    command_line.append(quote_windows_argument(wide));
  }
  std::vector<wchar_t> mutable_command(command_line.begin(), command_line.end());
  mutable_command.push_back(L'\0');

  STARTUPINFOW startup{};
  startup.cb = sizeof(startup);
  startup.dwFlags = STARTF_USESTDHANDLES;
  startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
  startup.hStdOutput = pipe_write;
  startup.hStdError = GetStdHandle(STD_ERROR_HANDLE);
  PROCESS_INFORMATION process{};
  if (!CreateProcessW(executable.c_str(), mutable_command.data(), nullptr, nullptr, TRUE,
                      CREATE_SUSPENDED | CREATE_NO_WINDOW, nullptr, nullptr, &startup, &process)) {
    result.error = windows_error("CreateProcessW");
    CloseHandle(job);
    CloseHandle(pipe_read);
    CloseHandle(pipe_write);
    return result;
  }
  const auto terminate_and_close = [&](const unsigned int exit_code) {
    TerminateProcess(process.hProcess, exit_code);
    WaitForSingleObject(process.hProcess, INFINITE);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    CloseHandle(job);
    CloseHandle(pipe_read);
    CloseHandle(pipe_write);
  };
  if (!AssignProcessToJobObject(job, process.hProcess)) {
    result.error = windows_error("AssignProcessToJobObject");
    terminate_and_close(static_cast<unsigned int>(options.failure_exit_code));
    return result;
  }
  if (ResumeThread(process.hThread) == static_cast<DWORD>(-1)) {
    result.error = windows_error("ResumeThread");
    TerminateJobObject(job, static_cast<unsigned int>(options.failure_exit_code));
    WaitForSingleObject(process.hProcess, INFINITE);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    CloseHandle(job);
    CloseHandle(pipe_read);
    CloseHandle(pipe_write);
    return result;
  }
  CloseHandle(process.hThread);
  CloseHandle(pipe_write);
  result.started = true;
  std::thread reader(windows_reader, pipe_read, std::ref(reader_state));

  bool process_ended = false;
  for (;;) {
    const DWORD wait_result = WaitForSingleObject(process.hProcess, 25U);
    if (wait_result == WAIT_OBJECT_0) {
      process_ended = true;
      break;
    }
    if (wait_result == WAIT_FAILED) {
      result.error = windows_error("WaitForSingleObject");
      break;
    }
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard lock(reader_state.mutex);
    if (now - started_at >= options.overall_timeout ||
        now - reader_state.last_progress >= options.no_progress_timeout) {
      result.timed_out = true;
      break;
    }
    if (!reader_state.error.empty()) {
      break;
    }
  }
  if (!process_ended) {
    TerminateJobObject(job,
                       static_cast<unsigned int>(result.timed_out ? options.timeout_exit_code
                                                                  : options.failure_exit_code));
    WaitForSingleObject(process.hProcess, INFINITE);
  }
  DWORD exit_code = static_cast<DWORD>(options.failure_exit_code);
  if (GetExitCodeProcess(process.hProcess, &exit_code)) {
    result.process_exit_code = static_cast<int>(exit_code);
  }
  CloseHandle(process.hProcess);
  CloseHandle(job);
  reader.join();

#else
  int pipe_handles[2] = {-1, -1};
  if (pipe(pipe_handles) != 0) {
    result.error = "pipe failed with errno " + std::to_string(errno);
    return result;
  }
  std::vector<std::string> arguments;
  arguments.reserve(options.arguments.size() + 1U);
  arguments.push_back(options.executable.string());
  arguments.insert(arguments.end(), options.arguments.begin(), options.arguments.end());
  std::vector<char*> argv;
  argv.reserve(arguments.size() + 1U);
  for (auto& argument : arguments) {
    argv.push_back(argument.data());
  }
  argv.push_back(nullptr);

  const pid_t child = fork();
  if (child < 0) {
    result.error = "fork failed with errno " + std::to_string(errno);
    close(pipe_handles[0]);
    close(pipe_handles[1]);
    return result;
  }
  if (child == 0) {
    setpgid(0, 0);
    close(pipe_handles[0]);
    if (dup2(pipe_handles[1], STDOUT_FILENO) < 0) {
      _exit(127);
    }
    close(pipe_handles[1]);
    execv(argv[0], argv.data());
    _exit(127);
  }
  close(pipe_handles[1]);
  setpgid(child, child);
  result.started = true;
  std::thread reader(posix_reader, pipe_handles[0], std::ref(reader_state));

  bool process_ended = false;
  int wait_status = 0;
  for (;;) {
    const pid_t waited = waitpid(child, &wait_status, WNOHANG);
    if (waited == child) {
      process_ended = true;
      break;
    }
    if (waited < 0 && errno != EINTR) {
      result.error = "waitpid failed with errno " + std::to_string(errno);
      break;
    }
    const auto now = std::chrono::steady_clock::now();
    {
      std::unique_lock lock(reader_state.mutex);
      if (now - started_at >= options.overall_timeout ||
          now - reader_state.last_progress >= options.no_progress_timeout) {
        result.timed_out = true;
        break;
      }
      if (!reader_state.error.empty()) {
        break;
      }
      reader_state.changed.wait_for(lock, std::chrono::milliseconds(25));
    }
  }
  if (!process_ended) {
    kill(-child, SIGKILL);
    do {
      process_ended = waitpid(child, &wait_status, 0) == child;
    } while (!process_ended && errno == EINTR);
  }
  if (result.timed_out) {
    // POSIX SIGKILL is observable as 128 + SIGKILL, unlike the caller-selected termination code
    // used by Windows Job Objects. Expose the controller's versioned timeout code consistently;
    // the process has already been synchronously reaped at this point.
    result.process_exit_code = options.timeout_exit_code;
  } else if (WIFEXITED(wait_status)) {
    result.process_exit_code = WEXITSTATUS(wait_status);
  } else if (WIFSIGNALED(wait_status)) {
    result.process_exit_code = 128 + WTERMSIG(wait_status);
  }
  reader.join();
#endif

  {
    std::lock_guard lock(reader_state.mutex);
    if (!reader_state.error.empty() && result.error.empty()) {
      result.error = reader_state.error;
    }
    result.stream_error = !reader_state.error.empty();
  }
  return result;
}

} // namespace xvram::platform
