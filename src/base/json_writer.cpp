#include "xvram/base/json_writer.hpp"

#include <cmath>
#include <iomanip>
#include <limits>
#include <stdexcept>

namespace xvram {

JsonWriter::JsonWriter(std::ostream& output, const bool pretty)
    : output_(output), pretty_(pretty) {}

void JsonWriter::begin_object() {
  before_value();
  output_ << '{';
  scopes_.push_back({ScopeKind::object});
}

void JsonWriter::end_object() {
  if (scopes_.empty() || scopes_.back().kind != ScopeKind::object || scopes_.back().expects_value) {
    throw std::logic_error("invalid JSON object close");
  }
  const bool had_values = !scopes_.back().first;
  scopes_.pop_back();
  if (had_values && pretty_) {
    newline_and_indent(scopes_.size());
  }
  output_ << '}';
  finish_value();
}

void JsonWriter::begin_array() {
  before_value();
  output_ << '[';
  scopes_.push_back({ScopeKind::array});
}

void JsonWriter::end_array() {
  if (scopes_.empty() || scopes_.back().kind != ScopeKind::array) {
    throw std::logic_error("invalid JSON array close");
  }
  const bool had_values = !scopes_.back().first;
  scopes_.pop_back();
  if (had_values && pretty_) {
    newline_and_indent(scopes_.size());
  }
  output_ << ']';
  finish_value();
}

void JsonWriter::key(const std::string_view name) {
  if (scopes_.empty() || scopes_.back().kind != ScopeKind::object || scopes_.back().expects_value) {
    throw std::logic_error("JSON key outside object or without a value");
  }

  Scope& scope = scopes_.back();
  if (!scope.first) {
    output_ << ',';
  }
  if (pretty_) {
    newline_and_indent(scopes_.size());
  }
  write_escaped(name);
  output_ << (pretty_ ? ": " : ":");
  scope.first = false;
  scope.expects_value = true;
}

void JsonWriter::value(const std::string_view value) {
  before_value();
  write_escaped(value);
  finish_value();
}

void JsonWriter::value(const char* const value) {
  if (value == nullptr) {
    null_value();
    return;
  }
  this->value(std::string_view(value));
}

void JsonWriter::value(const bool value) {
  before_value();
  output_ << (value ? "true" : "false");
  finish_value();
}

void JsonWriter::value(const std::int64_t value) {
  before_value();
  output_ << value;
  finish_value();
}

void JsonWriter::value(const std::uint64_t value) {
  before_value();
  output_ << value;
  finish_value();
}

void JsonWriter::value(const double value) {
  before_value();
  if (!std::isfinite(value)) {
    output_ << "null";
  } else {
    output_ << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
  }
  finish_value();
}

void JsonWriter::null_value() {
  before_value();
  output_ << "null";
  finish_value();
}

void JsonWriter::before_value() {
  if (scopes_.empty()) {
    if (root_written_) {
      throw std::logic_error("multiple JSON root values");
    }
    root_written_ = true;
    return;
  }

  Scope& scope = scopes_.back();
  if (scope.kind == ScopeKind::object) {
    if (!scope.expects_value) {
      throw std::logic_error("JSON object value without key");
    }
    return;
  }

  if (!scope.first) {
    output_ << ',';
  }
  if (pretty_) {
    newline_and_indent(scopes_.size());
  }
  scope.first = false;
}

void JsonWriter::finish_value() {
  if (!scopes_.empty() && scopes_.back().kind == ScopeKind::object) {
    scopes_.back().expects_value = false;
  }
}

void JsonWriter::newline_and_indent(const std::size_t depth) {
  output_ << '\n';
  for (std::size_t index = 0; index < depth * 2; ++index) {
    output_ << ' ';
  }
}

void JsonWriter::write_escaped(const std::string_view value) {
  static constexpr char hex[] = "0123456789abcdef";
  output_ << '"';
  for (const char raw_character : value) {
    const auto character = static_cast<unsigned char>(raw_character);
    switch (character) {
    case '"':
      output_ << "\\\"";
      break;
    case '\\':
      output_ << "\\\\";
      break;
    case '\b':
      output_ << "\\b";
      break;
    case '\f':
      output_ << "\\f";
      break;
    case '\n':
      output_ << "\\n";
      break;
    case '\r':
      output_ << "\\r";
      break;
    case '\t':
      output_ << "\\t";
      break;
    default:
      if (character < 0x20U) {
        output_ << "\\u00" << hex[(character >> 4U) & 0xFU] << hex[character & 0xFU];
      } else {
        output_ << static_cast<char>(character);
      }
      break;
    }
  }
  output_ << '"';
}

} // namespace xvram
