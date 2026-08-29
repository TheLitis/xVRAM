#pragma once

#include <cstdint>
#include <ostream>
#include <string_view>
#include <vector>

namespace xvram {

class JsonWriter {
public:
  explicit JsonWriter(std::ostream& output, bool pretty = true);

  JsonWriter(const JsonWriter&) = delete;
  JsonWriter& operator=(const JsonWriter&) = delete;

  void begin_object();
  void end_object();
  void begin_array();
  void end_array();

  void key(std::string_view name);
  void value(std::string_view value);
  void value(const char* value);
  void value(bool value);
  void value(std::int64_t value);
  void value(std::uint64_t value);
  void value(double value);
  void null_value();

private:
  enum class ScopeKind { object, array };

  struct Scope {
    ScopeKind kind;
    bool first = true;
    bool expects_value = false;
  };

  void before_value();
  void finish_value();
  void newline_and_indent(std::size_t depth);
  void write_escaped(std::string_view value);

  std::ostream& output_;
  std::vector<Scope> scopes_;
  bool pretty_;
  bool root_written_ = false;
};

} // namespace xvram
