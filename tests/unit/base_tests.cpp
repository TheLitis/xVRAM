#include "xvram/base/json_writer.hpp"
#include "xvram/base/size_parser.hpp"

#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>

namespace {

int failures = 0;

void check(const bool condition, const char* expression, const int line) {
  if (!condition) {
    std::cerr << "line " << line << ": check failed: " << expression << '\n';
    ++failures;
  }
}

#define CHECK(expression) check(static_cast<bool>(expression), #expression, __LINE__)

void size_parser_tests() {
  CHECK(xvram::parse_size("0").bytes == 0);
  CHECK(xvram::parse_size("1KiB").bytes == 1024);
  CHECK(xvram::parse_size("64MiB").bytes == 64ULL * 1024ULL * 1024ULL);
  CHECK(xvram::parse_size("2g").bytes == 2ULL * 1024ULL * 1024ULL * 1024ULL);
  CHECK(!xvram::parse_size("-1MiB"));
  CHECK(!xvram::parse_size("1MB"));
  CHECK(!xvram::parse_size("18446744073709551616"));
  CHECK(xvram::format_bytes(1024) == "1.00 KiB");
}

void json_writer_tests() {
  std::ostringstream compact;
  xvram::JsonWriter writer(compact, false);
  writer.begin_object();
  writer.key("text");
  writer.value("line\n\"quoted\"");
  writer.key("items");
  writer.begin_array();
  writer.value(static_cast<std::uint64_t>(7));
  writer.value(true);
  writer.null_value();
  writer.end_array();
  writer.end_object();
  CHECK(compact.str() == R"({"text":"line\n\"quoted\"","items":[7,true,null]})");

  bool rejected_multiple_roots = false;
  try {
    writer.value(false);
  } catch (const std::logic_error&) {
    rejected_multiple_roots = true;
  }
  CHECK(rejected_multiple_roots);
}

} // namespace

int main() {
  size_parser_tests();
  json_writer_tests();
  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
    return 1;
  }
  std::cout << "all unit tests passed\n";
  return 0;
}
