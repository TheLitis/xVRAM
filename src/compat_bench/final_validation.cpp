#include "compat_bench/final_validation.hpp"
#include "compat_bench/report_schema_v1.hpp"
#include "xvram/base/json_writer.hpp"
#include <charconv>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace xvram::compat_bench {
namespace {
enum class Kind { null, boolean, unsigned_number, signed_number, real, string, object, array };
struct Value {
  Kind kind = Kind::null;
  bool boolean = false;
  std::uint64_t unsigned_number = 0;
  std::int64_t signed_number = 0;
  double real = 0;
  std::string string;
  std::map<std::string, Value> object;
  std::vector<Value> array;
};
[[noreturn]] void invalid() {
  throw std::runtime_error("worker final JSON is malformed or violates its typed report contract");
}
class Parser {
public:
  explicit Parser(std::string_view input) : input_(input) {}
  Value parse() {
    auto result = value(0);
    space();
    if (at_ != input_.size())
      invalid();
    return result;
  }

private:
  void space() {
    while (at_ < input_.size() && (input_[at_] == ' ' || input_[at_] == '\n' ||
                                   input_[at_] == '\r' || input_[at_] == '\t'))
      ++at_;
  }
  bool take(char c) {
    space();
    if (at_ < input_.size() && input_[at_] == c) {
      ++at_;
      return true;
    }
    return false;
  }
  void require(char c) {
    if (!take(c))
      invalid();
  }
  void literal(std::string_view text) {
    if (input_.substr(at_, text.size()) != text)
      invalid();
    at_ += text.size();
  }
  std::uint32_t hex4() {
    if (input_.size() - at_ < 4)
      invalid();
    std::uint32_t out = 0;
    for (unsigned i = 0; i < 4; ++i) {
      const char c = input_[at_++];
      out *= 16;
      if (c >= '0' && c <= '9')
        out += static_cast<unsigned>(c - '0');
      else if (c >= 'a' && c <= 'f')
        out += static_cast<unsigned>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F')
        out += static_cast<unsigned>(c - 'A' + 10);
      else
        invalid();
    }
    return out;
  }
  static void utf8(std::string& out, std::uint32_t c) {
    if (c <= 0x7f)
      out.push_back(static_cast<char>(c));
    else if (c <= 0x7ff) {
      out.push_back(static_cast<char>(0xc0U | (c >> 6U)));
      out.push_back(static_cast<char>(0x80U | (c & 63U)));
    } else if (c <= 0xffff) {
      out.push_back(static_cast<char>(0xe0U | (c >> 12U)));
      out.push_back(static_cast<char>(0x80U | ((c >> 6U) & 63U)));
      out.push_back(static_cast<char>(0x80U | (c & 63U)));
    } else {
      out.push_back(static_cast<char>(0xf0U | (c >> 18U)));
      out.push_back(static_cast<char>(0x80U | ((c >> 12U) & 63U)));
      out.push_back(static_cast<char>(0x80U | ((c >> 6U) & 63U)));
      out.push_back(static_cast<char>(0x80U | (c & 63U)));
    }
  }
  std::string string() {
    require('"');
    std::string out;
    while (at_ < input_.size()) {
      const auto c = static_cast<unsigned char>(input_[at_++]);
      if (c == '"')
        return out;
      if (c < 0x20)
        invalid();
      if (c == '\\') {
        if (at_ == input_.size())
          invalid();
        const char escape = input_[at_++];
        switch (escape) {
        case '"':
        case '\\':
        case '/':
          out.push_back(escape);
          break;
        case 'b':
          out.push_back('\b');
          break;
        case 'f':
          out.push_back('\f');
          break;
        case 'n':
          out.push_back('\n');
          break;
        case 'r':
          out.push_back('\r');
          break;
        case 't':
          out.push_back('\t');
          break;
        case 'u': {
          auto code = hex4();
          if (code >= 0xd800 && code <= 0xdbff) {
            if (input_.substr(at_, 2) != "\\u")
              invalid();
            at_ += 2;
            const auto low = hex4();
            if (low < 0xdc00 || low > 0xdfff)
              invalid();
            code = 0x10000 + ((code - 0xd800) << 10U) + (low - 0xdc00);
          } else if (code >= 0xdc00 && code <= 0xdfff)
            invalid();
          utf8(out, code);
          break;
        }
        default:
          invalid();
        }
      } else if (c < 0x80)
        out.push_back(static_cast<char>(c));
      else {
        unsigned extra = 0;
        std::uint32_t code = 0, minimum = 0;
        if (c >= 0xc2 && c <= 0xdf) {
          extra = 1;
          code = c & 31U;
          minimum = 0x80;
        } else if (c >= 0xe0 && c <= 0xef) {
          extra = 2;
          code = c & 15U;
          minimum = 0x800;
        } else if (c >= 0xf0 && c <= 0xf4) {
          extra = 3;
          code = c & 7U;
          minimum = 0x10000;
        } else
          invalid();
        if (input_.size() - at_ < extra)
          invalid();
        for (unsigned i = 0; i < extra; ++i) {
          const auto tail = static_cast<unsigned char>(input_[at_++]);
          if ((tail & 0xc0U) != 0x80U)
            invalid();
          code = (code << 6U) | (tail & 63U);
        }
        if (code < minimum || code > 0x10ffff || (code >= 0xd800 && code <= 0xdfff))
          invalid();
        utf8(out, code);
      }
    }
    invalid();
  }
  Value number() {
    Value out;
    const auto begin = at_;
    const bool negative = input_[at_] == '-';
    if (negative)
      ++at_;
    if (at_ == input_.size())
      invalid();
    if (input_[at_] == '0')
      ++at_;
    else {
      if (input_[at_] < '1' || input_[at_] > '9')
        invalid();
      while (at_ < input_.size() && input_[at_] >= '0' && input_[at_] <= '9')
        ++at_;
    }
    bool floating = false;
    if (at_ < input_.size() && input_[at_] == '.') {
      floating = true;
      ++at_;
      const auto digits = at_;
      while (at_ < input_.size() && input_[at_] >= '0' && input_[at_] <= '9')
        ++at_;
      if (at_ == digits)
        invalid();
    }
    if (at_ < input_.size() && (input_[at_] == 'e' || input_[at_] == 'E')) {
      floating = true;
      ++at_;
      if (at_ < input_.size() && (input_[at_] == '+' || input_[at_] == '-'))
        ++at_;
      const auto digits = at_;
      while (at_ < input_.size() && input_[at_] >= '0' && input_[at_] <= '9')
        ++at_;
      if (at_ == digits)
        invalid();
    }
    const auto parse = [&](auto& destination) {
      const auto result = std::from_chars(input_.data() + begin, input_.data() + at_, destination);
      if (result.ec != std::errc{} || result.ptr != input_.data() + at_)
        invalid();
    };
    if (floating) {
      out.kind = Kind::real;
      parse(out.real);
      if (!std::isfinite(out.real))
        invalid();
    } else if (negative) {
      out.kind = Kind::signed_number;
      parse(out.signed_number);
    } else {
      out.kind = Kind::unsigned_number;
      parse(out.unsigned_number);
    }
    return out;
  }
  Value value(unsigned depth) {
    if (depth > 32 || ++nodes_ > 100000)
      invalid();
    space();
    if (at_ == input_.size())
      invalid();
    Value out;
    if (input_[at_] == '{') {
      ++at_;
      out.kind = Kind::object;
      if (take('}'))
        return out;
      for (;;) {
        auto key = string();
        require(':');
        auto child = value(depth + 1);
        if (!out.object.emplace(std::move(key), std::move(child)).second)
          invalid();
        if (take('}'))
          return out;
        require(',');
      }
    }
    if (input_[at_] == '[') {
      ++at_;
      out.kind = Kind::array;
      if (take(']'))
        return out;
      for (;;) {
        out.array.push_back(value(depth + 1));
        if (take(']'))
          return out;
        require(',');
      }
    }
    if (input_[at_] == '"') {
      out.kind = Kind::string;
      out.string = string();
      return out;
    }
    if (input_[at_] == 't') {
      literal("true");
      out.kind = Kind::boolean;
      out.boolean = true;
      return out;
    }
    if (input_[at_] == 'f') {
      literal("false");
      out.kind = Kind::boolean;
      out.boolean = false;
      return out;
    }
    if (input_[at_] == 'n') {
      literal("null");
      return out;
    }
    return number();
  }
  std::string_view input_;
  std::size_t at_ = 0, nodes_ = 0;
};
void keys(const Value& object, std::string_view names) {
  if (object.kind != Kind::object)
    invalid();
  std::set<std::string> wanted;
  std::istringstream words{std::string(names)};
  for (std::string word; words >> word;)
    wanted.insert(word);
  if (wanted.size() != object.object.size())
    invalid();
  for (const auto& [key, child] : object.object) {
    static_cast<void>(child);
    if (!wanted.contains(key))
      invalid();
  }
}
const Value& member(const Value& object, const char* key, Kind kind) {
  if (object.kind != Kind::object)
    invalid();
  const auto it = object.object.find(key);
  if (it == object.object.end() || it->second.kind != kind)
    invalid();
  return it->second;
}
void write(JsonWriter& writer, const Value& value) {
  switch (value.kind) {
  case Kind::null:
    writer.null_value();
    break;
  case Kind::boolean:
    writer.value(value.boolean);
    break;
  case Kind::unsigned_number:
    writer.value(value.unsigned_number);
    break;
  case Kind::signed_number:
    writer.value(value.signed_number);
    break;
  case Kind::real:
    writer.value(value.real);
    break;
  case Kind::string:
    writer.value(value.string);
    break;
  case Kind::array:
    writer.begin_array();
    for (const auto& child : value.array)
      write(writer, child);
    writer.end_array();
    break;
  case Kind::object:
    writer.begin_object();
    for (const auto& [key, child] : value.object) {
      writer.key(key);
      write(writer, child);
    }
    writer.end_object();
    break;
  }
}
const Value* property(const Value& object, const char* key) {
  const auto found = object.object.find(key);
  return found == object.object.end() ? nullptr : &found->second;
}
bool equal_primitive(const Value& a, const Value& b) {
  if (a.kind != b.kind)
    return false;
  switch (a.kind) {
  case Kind::null:
    return true;
  case Kind::boolean:
    return a.boolean == b.boolean;
  case Kind::unsigned_number:
    return a.unsigned_number == b.unsigned_number;
  case Kind::signed_number:
    return a.signed_number == b.signed_number;
  case Kind::real:
    return a.real == b.real;
  case Kind::string:
    return a.string == b.string;
  default:
    return false;
  }
}
bool has_type(const Value& value, const std::string& name) {
  if (name == "null")
    return value.kind == Kind::null;
  if (name == "boolean")
    return value.kind == Kind::boolean;
  if (name == "integer")
    return value.kind == Kind::unsigned_number || value.kind == Kind::signed_number;
  if (name == "number")
    return value.kind == Kind::unsigned_number || value.kind == Kind::signed_number ||
           value.kind == Kind::real;
  if (name == "string")
    return value.kind == Kind::string;
  if (name == "array")
    return value.kind == Kind::array;
  if (name == "object")
    return value.kind == Kind::object;
  return false;
}
bool validate_schema(const Value& value, const Value& schema) {
  if (const auto* type = property(schema, "type")) {
    bool okay = false;
    if (type->kind == Kind::string)
      okay = has_type(value, type->string);
    else
      for (const auto& item : type->array)
        okay = okay || has_type(value, item.string);
    if (!okay)
      return false;
  }
  if (const auto* fixed = property(schema, "const"))
    if (!equal_primitive(value, *fixed))
      return false;
  if (const auto* choices = property(schema, "enum")) {
    bool found = false;
    for (const auto& choice : choices->array)
      found = found || equal_primitive(value, choice);
    if (!found)
      return false;
  }
  if (const auto* minimum = property(schema, "minimum")) {
    const double bound = minimum->kind == Kind::unsigned_number
                             ? static_cast<double>(minimum->unsigned_number)
                             : minimum->real;
    if (value.kind == Kind::signed_number && static_cast<double>(value.signed_number) < bound)
      return false;
    if (value.kind == Kind::unsigned_number && static_cast<double>(value.unsigned_number) < bound)
      return false;
    if (value.kind == Kind::real && value.real < bound)
      return false;
  }
  if (value.kind == Kind::object) {
    if (const auto* required = property(schema, "required"))
      for (const auto& name : required->array)
        if (!value.object.contains(name.string))
          return false;
    if (const auto* properties = property(schema, "properties")) {
      for (const auto& [key, child] : value.object) {
        const auto found = properties->object.find(key);
        if (found == properties->object.end()) {
          if (const auto* additional = property(schema, "additionalProperties"))
            if (!additional->boolean)
              return false;
        } else if (!validate_schema(child, found->second))
          return false;
      }
    }
  }
  if (value.kind == Kind::array) {
    if (const auto* maximum = property(schema, "maxItems"))
      if (value.array.size() > maximum->unsigned_number)
        return false;
    if (const auto* items = property(schema, "items"))
      for (const auto& child : value.array)
        if (!validate_schema(child, *items))
          return false;
  }
  if (value.kind == Kind::string) {
    if (const auto* pattern = property(schema, "pattern")) {
      // The contract deliberately has one digest pattern and no user-supplied regexes.
      if (pattern->string != "^([a-f0-9]{32})?$")
        return false;
      if (!value.string.empty()) {
        if (value.string.size() != 32)
          return false;
        for (char c : value.string)
          if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
            return false;
      }
    }
    if (const auto* format = property(schema, "format")) {
      if (format->string != "date-time" || value.string.size() != 20)
        return false;
      for (std::size_t i = 0; i < 20; ++i) {
        const char c = value.string[i];
        if (i == 4 || i == 7) {
          if (c != '-')
            return false;
        } else if (i == 10) {
          if (c != 'T')
            return false;
        } else if (i == 13 || i == 16) {
          if (c != ':')
            return false;
        } else if (i == 19) {
          if (c != 'Z')
            return false;
        } else if (c < '0' || c > '9')
          return false;
      }
    }
  }
  if (const auto* all = property(schema, "allOf"))
    for (const auto& constraint : all->array) {
      if (const auto* condition = property(constraint, "if")) {
        if (validate_schema(value, *condition)) {
          if (const auto* then = property(constraint, "then"))
            if (!validate_schema(value, *then))
              return false;
        }
      } else if (!validate_schema(value, constraint))
        return false;
    }
  return true;
}
} // namespace
bool finalize_json(const std::string_view input, const std::int32_t exit_code, const bool pretty,
                   std::string& output, std::string& error) {
  try {
    if (input.empty() || input.size() > 1024U * 1024U)
      invalid();
    auto root = Parser(input).parse();
    keys(root,
         "schema_version report_type generated_at_utc build system device configuration "
         "compatibility workloads execution cache verification proof outcome cleanup diagnostics");
    if (member(root, "schema_version", Kind::unsigned_number).unsigned_number != 1 ||
        member(root, "report_type", Kind::string).string != "xvram.cuda_compat")
      invalid();
    const auto& outcome = member(root, "outcome", Kind::object);
    keys(outcome, "status exit_code reason message");
    if (exit_code < 0 || member(outcome, "exit_code", Kind::unsigned_number).unsigned_number !=
                             static_cast<std::uint64_t>(exit_code))
      invalid();
    const std::string status = exit_code == 0    ? "completed"
                               : exit_code == 23 ? "skipped"
                               : exit_code == 24 ? "corruption"
                               : exit_code == 25 ? "oom"
                               : exit_code == 26 ? "timeout"
                                                 : "failed";
    if (member(outcome, "status", Kind::string).string != status)
      invalid();
    static_cast<void>(member(outcome, "reason", Kind::string));
    static_cast<void>(member(outcome, "message", Kind::string));
    const auto& proof = member(root, "proof", Kind::object);
    keys(proof, "supported_calls_routed reference_equal padding_preserved "
                "rejections_before_submission stable_addresses no_physical_aliases "
                "mapping_access_balanced mapping_unmap_balanced event_safe zero_unsafe_activity "
                "bounded_cache real_oversubscription_reuse cleanup_complete");
    for (const auto& [key, value] : proof.object) {
      static_cast<void>(key);
      if (value.kind != Kind::null && value.kind != Kind::boolean)
        invalid();
      if (exit_code == 0 && (value.kind != Kind::boolean || !value.boolean))
        invalid();
    }
    auto& cleanup = root.object.at("cleanup");
    keys(cleanup, "operations_drained events_drained allocations_released handles_destroyed "
                  "reservations_freed adapter_closed worker_terminated trace_closed");
    for (const auto& [key, value] : cleanup.object) {
      if (value.kind != Kind::null && value.kind != Kind::boolean)
        invalid();
      if (key == "worker_terminated") {
        if (value.kind != Kind::null)
          invalid();
      } else if (exit_code == 0 && (value.kind != Kind::boolean || !value.boolean))
        invalid();
    }
    const auto& diagnostics = member(root, "diagnostics", Kind::array);
    if (exit_code == 0 && !diagnostics.array.empty())
      invalid();
    for (const auto& item : diagnostics.array)
      if (item.kind != Kind::string)
        invalid();
    for (const char* key : {"build", "system", "device", "configuration", "compatibility",
                            "execution", "cache", "verification"})
      static_cast<void>(member(root, key, Kind::object));
    static_cast<void>(member(root, "workloads", Kind::array));
    cleanup.object["worker_terminated"].kind = Kind::boolean;
    cleanup.object["worker_terminated"].boolean = true;
    static const Value schema = Parser(report_schema_v1).parse();
    if (!validate_schema(root, schema))
      invalid();
    std::ostringstream stream;
    JsonWriter writer(stream, pretty);
    write(writer, root);
    output = stream.str();
    return true;
  } catch (const std::exception& exception) {
    error = exception.what();
    return false;
  } catch (...) {
    error = "worker final JSON validation failed";
    return false;
  }
}
} // namespace xvram::compat_bench
