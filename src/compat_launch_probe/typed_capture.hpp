#pragma once
#include "core.hpp"
#include <array>
#include <cstring>
#include <limits>
#include <span>
#include <string_view>

namespace xvram::launch_probe::typed_capture {
enum class ValueType { ptr, i64, u64, i32, u32, f32, boolean };
struct Field { const char* name; ValueType type; std::size_t offset; };
struct Argument { const char* name; std::size_t offset, bytes; std::span<const Field> fields; };
struct Kernel { std::uint32_t id; const char* symbol; std::span<const Argument> arguments; bool supported; };
inline std::size_t width(ValueType type) {
  switch(type) {
    case ValueType::ptr: case ValueType::i64: case ValueType::u64: return 8;
    case ValueType::i32: case ValueType::u32: case ValueType::f32: return 4;
    case ValueType::boolean: return 1;
  }
  throw std::runtime_error("capture_unknown_field_type");
}

// Mirrors only reviewed source structs, never opaque library Params. Their
// field offsets are checked independently against the checked-in source ABI.
struct Fusion {
  const void *x_bias, *gate, *gate_bias, *x_scale, *gate_scale;
  std::int32_t glu_op;
  float glu_limit;
};
static_assert(sizeof(void*)==8 && sizeof(Fusion)==48 && offsetof(Fusion,gate_scale)==32 &&
              offsetof(Fusion,glu_op)==40 && offsetof(Fusion,glu_limit)==44);
struct Softmax {
  std::int64_t nheads;
  std::uint32_t n_head_log2;
  std::int64_t ncols, nrows_x, nrows_y, ne00, ne01, ne02, ne03, nb11, nb12, nb13, ne12, ne13;
  float scale, max_bias, m0, m1;
};
static_assert(sizeof(Softmax)==128 && offsetof(Softmax,ncols)==16 &&
              offsetof(Softmax,ne13)==104 && offsetof(Softmax,scale)==112 && offsetof(Softmax,m1)==124);

inline bool matches(const Kernel& kernel, const Snapshot& snapshot) {
  if(!kernel.supported || snapshot.name!=kernel.symbol || snapshot.parameters.size()!=kernel.arguments.size()) return false;
  for(std::size_t i=0;i<kernel.arguments.size();++i)
    if(snapshot.parameters[i].offset!=kernel.arguments[i].offset || snapshot.parameters[i].bytes!=kernel.arguments[i].bytes)
      return false;
  return true;
}

// Reader performs bounded fault-aware reads from the *host* argument bank.
// Sink.pointer resolves device values to allocation IDs; no raw pointer value
// reaches Sink.scalar. All ABI validation precedes every parameter dereference.
// Fields, never padding or opaque blobs, are emitted from the copied snapshot.
template<class Reader, class Sink>
void capture(const Kernel& kernel, const Snapshot& snapshot, const void* parameters,
             const void* extra, Reader&& read, Sink& sink) {
  if(!matches(kernel,snapshot) || !parameters || extra || kernel.arguments.size()>64)
    throw std::runtime_error("capture_unsupported_bank_or_abi");
  for(const auto& argument:kernel.arguments) {
    if(!argument.bytes || argument.bytes>128 || argument.fields.empty()) throw std::runtime_error("capture_argument_limit");
    std::size_t end=0;
    for(const auto& field:argument.fields) {
      const auto size=width(field.type);
      if(!field.name || field.offset<end || field.offset>argument.bytes || size>argument.bytes-field.offset)
        throw std::runtime_error("capture_field_layout");
      end=field.offset+size;
    }
  }
  std::array<std::uintptr_t,64> bank{};
  read(parameters,bank.data(),kernel.arguments.size()*sizeof(std::uintptr_t));
  // Snapshot all host bytes before publishing any argument to the trace. The
  // caller's original pointers/bank/extra are forwarded unchanged afterwards.
  std::array<std::array<std::byte,128>,64> copies{};
  for(std::size_t i=0;i<kernel.arguments.size();++i) {
    if(!bank[i]) throw std::runtime_error("capture_null_parameter");
    read(reinterpret_cast<const void*>(bank[i]),copies[i].data(),kernel.arguments[i].bytes);
  }
  for(std::size_t i=0;i<kernel.arguments.size();++i) {
    const auto& argument=kernel.arguments[i];
    for(const auto& field:argument.fields) {
      std::uint64_t bits=0;
      std::memcpy(&bits,copies[i].data()+field.offset,width(field.type));
      if(field.type==ValueType::boolean && bits>1) throw std::runtime_error("capture_invalid_boolean");
      if(field.type==ValueType::ptr) sink.pointer(i,argument.name,field.name,bits);
      else sink.scalar(i,argument.name,field.name,field.type,bits);
    }
  }
}
}
