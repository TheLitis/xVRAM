"""Bounded, observation-only CUDA compatibility evidence analysis.

This module neither loads native libraries nor executes the inspected program. API
storage ranges are not tensor footprints. Missing evidence always remains unknown.
The current CUPTI observer does not capture typed kernel arguments; consequently
its captures cannot establish an execution-ready working set on their own.
"""

import datetime
import hashlib
import json
import math
import re
import struct
from collections import Counter
from pathlib import Path


U64_MAX = (1 << 64) - 1
MAX_RECORD_BYTES = 1024 * 1024
MAX_TRACE_BYTES = 2 * 1024 * 1024 * 1024
MAX_RECORDS = 8000000
MAX_STATE_ITEMS = 100000
MAX_METADATA_BYTES = 64 * 1024 * 1024
MAX_ITEMS = 100000
TRACE_TYPE = "xvram.cuda_compat_audit_trace"
REPORT_TYPE = "xvram.cuda_compat_audit"
_HEX_ADDRESS = re.compile(r"0x[0-9a-fA-F]{6,}")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class AuditInputError(ValueError):
    """An input exceeded a bound or violated the evidence format."""


def _uint(value, label="integer"):
    if type(value) is not int or not 0 <= value <= U64_MAX:
        raise AuditInputError("invalid_" + label)
    return value


def _add(left, right):
    left, right = _uint(left), _uint(right)
    if right > U64_MAX - left:
        raise AuditInputError("range_overflow")
    return left + right


def _mul(left, right):
    left, right = _uint(left), _uint(right)
    if left and right > U64_MAX // left:
        raise AuditInputError("range_overflow")
    return left * right


def range_union_bytes(ranges):
    """Union explicit (offset, byte_count) ranges; never infer tensor sizes."""
    spans = sorted((offset, _add(offset, size)) for offset, size in ranges)
    total, stop = 0, 0
    for start, end in spans:
        if end > stop:
            total = _add(total, end - max(start, stop))
            stop = end
    return total


def _text(value, limit=4096):
    if not isinstance(value, str) or len(value) > limit or "\x00" in value:
        raise AuditInputError("invalid_text")
    if _HEX_ADDRESS.search(value):
        raise AuditInputError("raw_address_disallowed")
    return value


def _hash_file(stream):
    digest = hashlib.sha256()
    stream.seek(0)
    while True:
        block = stream.read(1024 * 1024)
        if not block:
            return digest.hexdigest()
        digest.update(block)


class _Reader:
    def __init__(self, stream, size, budget=MAX_METADATA_BYTES):
        self.stream, self.size, self.budget = stream, size, budget
        self.digest = hashlib.sha256()

    def read(self, count):
        _uint(count)
        if count > self.budget or _add(self.stream.tell(), count) > self.size:
            raise AuditInputError("truncated_or_unbounded_metadata")
        data = self.stream.read(count)
        if len(data) != count:
            raise AuditInputError("truncated_metadata")
        self.budget -= count
        self.digest.update(data)
        return data

    def unpack(self, fmt):
        return struct.unpack("<" + fmt, self.read(struct.calcsize("<" + fmt)))

    def string(self, limit=MAX_RECORD_BYTES):
        length = self.unpack("Q")[0]
        if length > limit:
            raise AuditInputError("string_limit")
        try:
            return self.read(length).decode("utf-8", errors="strict")
        except UnicodeError as error:
            raise AuditInputError("invalid_utf8") from error


def _pe_metadata(stream, size):
    reader = _Reader(stream, size, 16 * 1024 * 1024)

    def at(offset, count):
        stream.seek(_uint(offset))
        return reader.read(count)

    dos = at(0, 64)
    if dos[:2] != b"MZ":
        raise AuditInputError("not_pe")
    offset = struct.unpack_from("<I", dos, 60)[0]
    head = at(offset, 24)
    if head[:4] != b"PE\0\0":
        raise AuditInputError("invalid_pe_signature")
    machine, section_count = struct.unpack_from("<HH", head, 4)
    optional_size = struct.unpack_from("<H", head, 20)[0]
    if not 1 <= section_count <= 96 or optional_size > 4096:
        raise AuditInputError("pe_header_limit")
    optional = at(_add(offset, 24), optional_size)
    if len(optional) < 96:
        raise AuditInputError("invalid_optional_header")
    magic = struct.unpack_from("<H", optional)[0]
    if magic not in (0x10B, 0x20B):
        raise AuditInputError("unsupported_pe_format")
    directory_offset, thunk_size = (112, 8) if magic == 0x20B else (96, 4)
    if len(optional) < directory_offset:
        raise AuditInputError("invalid_optional_header")
    directory_count = struct.unpack_from("<I", optional, directory_offset - 4)[0]
    if directory_count > 16 or directory_offset + 8 * directory_count > len(optional):
        raise AuditInputError("invalid_directory_count")
    headers_size = struct.unpack_from("<I", optional, 60)[0]
    if headers_size > size:
        raise AuditInputError("invalid_headers_size")
    sections = []
    for index in range(section_count):
        section = at(offset + 24 + optional_size + 40 * index, 40)
        virtual_size, rva, raw_size, raw_offset = struct.unpack_from("<IIII", section, 8)
        if _add(raw_offset, raw_size) > size or _add(rva, max(virtual_size, raw_size)) > (1 << 32):
            raise AuditInputError("invalid_section_range")
        sections.append((rva, raw_size, raw_offset))

    def resolve(rva, count):
        end = _add(rva, count)
        matches = []
        if end <= headers_size:
            matches.append(rva)
        for base, length, raw in sections:
            if base <= rva and end <= base + length:
                matches.append(raw + rva - base)
        if len(matches) != 1:
            raise AuditInputError("unmapped_or_ambiguous_rva")
        return matches[0]

    def cstring(rva):
        data = bytearray()
        for delta in range(512):
            byte = at(resolve(_add(rva, delta), 1), 1)
            if byte == b"\0":
                try:
                    return _text(data.decode("ascii"), 512)
                except UnicodeError as error:
                    raise AuditInputError("invalid_import_name") from error
            data.extend(byte)
        raise AuditInputError("import_name_limit")

    imports, total_symbols = [], 0
    for directory_index, delayed, stride in ((1, False, 20), (13, True, 32)):
        if directory_count <= directory_index:
            continue
        rva, length = struct.unpack_from("<II", optional, directory_offset + 8 * directory_index)
        if not rva and not length:
            continue
        if not rva or length < stride:
            raise AuditInputError("invalid_import_directory")
        terminated = False
        for index in range(min(length // stride, 1024)):
            entry = at(resolve(_add(rva, index * stride), stride), stride)
            if not any(entry):
                terminated = True
                break
            values = struct.unpack("<" + "I" * (stride // 4), entry)
            if delayed:
                if values[0] != 1:
                    raise AuditInputError("unsupported_delay_import_address_mode")
                name_rva, thunk_rva = values[1], values[4] or values[3]
            else:
                name_rva, thunk_rva = values[3], values[0] or values[4]
            if not thunk_rva:
                raise AuditInputError("missing_import_thunk")
            symbols = []
            for symbol_index in range(16385):
                thunk = int.from_bytes(at(resolve(_add(thunk_rva, symbol_index * thunk_size), thunk_size), thunk_size), "little")
                if not thunk:
                    break
                total_symbols += 1
                if total_symbols > 16384:
                    raise AuditInputError("import_symbol_limit")
                ordinal_bit = 1 << (thunk_size * 8 - 1)
                if thunk & ordinal_bit:
                    symbols.append("#" + str(thunk & 0xFFFF))
                else:
                    if thunk > 0xFFFFFFFF:
                        raise AuditInputError("invalid_import_rva")
                    symbols.append(cstring(_add(thunk, 2)))
            imports.append({"library": cstring(name_rva), "delayed": delayed, "symbols": symbols})
        if not terminated:
            raise AuditInputError("unterminated_import_directory")
    return {"machine": machine, "bits": thunk_size * 8, "imports": imports,
            "dynamic_resolution_possible": any(symbol in ("GetProcAddress", "LoadLibraryA", "LoadLibraryW", "LoadLibraryExA", "LoadLibraryExW") for item in imports for symbol in item["symbols"]),
            "absence_of_import_proves_static_linkage": False}


def inventory(paths, *, include_paths=False):
    """Stream hashes and inspect PE imports without loading executable code."""
    paths = list(paths)
    if len(paths) > 128:
        raise AuditInputError("artifact_limit")
    result = []
    for index, value in enumerate(paths):
        path = Path(value)
        item = {"artifact_id": "artifact-{:04d}".format(index + 1),
                "name": _text(path.name, 512), "path": str(path.resolve()) if include_paths else None,
                "sha256": None, "size_bytes": None, "format": "unknown",
                "evidence": "unresolved", "pe": None, "errors": []}
        try:
            with path.open("rb") as stream:
                before = path.stat()
                item["size_bytes"] = _uint(before.st_size)
                item["sha256"] = _hash_file(stream)
                stream.seek(0)
                signature = stream.read(4)
                item["format"] = "pe" if signature[:2] == b"MZ" else "gguf" if signature == b"GGUF" else "other"
                if item["format"] == "pe":
                    item["pe"] = _pe_metadata(stream, before.st_size)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                    raise AuditInputError("artifact_changed_during_read")
                item["evidence"] = "observed"
        except (OSError, AuditInputError) as error:
            item["errors"].append(str(error) if isinstance(error, AuditInputError) else "artifact_unreadable")
        result.append(item)
    return result


# Block layouts are source-proven by the pinned upstream ggml definitions, not
# inferred from tensor offsets or allocation sizes. Unknown types stay unknown.
GGML_LAYOUT_SOURCE = "https://github.com/ggml-org/llama.cpp/blob/b10819/ggml/src/ggml-common.h"
_LAYOUTS = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20),
            6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 36),
            10: (256, 84), 11: (256, 110), 12: (256, 144),
            13: (256, 176), 14: (256, 210), 15: (256, 292),
            24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8),
            28: (1, 8), 30: (1, 2)}
_VALUE_FORMATS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "B", 10: "Q", 11: "q", 12: "d"}


def read_gguf_metadata(path):
    """Read bounded GGUF v2/v3 metadata, never model weight payloads."""
    path = Path(path)
    try:
        with path.open("rb") as stream:
            before = path.stat()
            reader = _Reader(stream, before.st_size)
            if reader.read(4) != b"GGUF":
                raise AuditInputError("not_gguf")
            version, tensors_count, metadata_count = reader.unpack("IQQ")
            if version not in (2, 3):
                raise AuditInputError("unsupported_gguf_version_or_endianness")
            if tensors_count > MAX_ITEMS or metadata_count > MAX_ITEMS:
                raise AuditInputError("gguf_item_limit")

            def value(kind, retain=True):
                if kind == 8:
                    text = reader.string()
                    return text if retain else None
                if kind == 9:
                    subtype, length = reader.unpack("IQ")
                    if subtype == 9 or length > 1000000:
                        raise AuditInputError("gguf_array_limit")
                    if subtype in _VALUE_FORMATS:
                        count = _mul(length, struct.calcsize("<" + _VALUE_FORMATS[subtype]))
                        while count:
                            chunk = min(count, MAX_RECORD_BYTES)
                            reader.read(chunk)
                            count -= chunk
                    elif subtype == 8:
                        for _ in range(length):
                            reader.string()
                    else:
                        raise AuditInputError("unknown_gguf_value_type")
                    return None
                if kind not in _VALUE_FORMATS:
                    raise AuditInputError("unknown_gguf_value_type")
                scalar = reader.unpack(_VALUE_FORMATS[kind])[0]
                if kind == 7 and scalar not in (0, 1):
                    raise AuditInputError("invalid_gguf_bool")
                if isinstance(scalar, float) and not math.isfinite(scalar):
                    raise AuditInputError("nonfinite_gguf_value")
                return scalar if retain else None

            selected, keys = {}, set()
            for _ in range(metadata_count):
                key = reader.string(4096)
                if key in keys:
                    raise AuditInputError("duplicate_gguf_key")
                keys.add(key)
                kind = reader.unpack("I")[0]
                retain = key in ("general.architecture", "general.alignment", "general.file_type")
                scalar = value(kind, retain)
                if retain:
                    selected[key] = scalar
            alignment = selected.get("general.alignment", 32)
            if type(alignment) is not int or not 1 <= alignment <= MAX_RECORD_BYTES or alignment & (alignment - 1):
                raise AuditInputError("invalid_gguf_alignment")
            tensors, names, unknown, known_bytes = [], set(), set(), 0
            for _ in range(tensors_count):
                name = _text(reader.string(4096))
                dimensions = reader.unpack("I")[0]
                if name in names or not 1 <= dimensions <= 4:
                    raise AuditInputError("invalid_gguf_tensor_header")
                names.add(name)
                shape = list(reader.unpack("Q" * dimensions))
                if any(dimension == 0 for dimension in shape):
                    raise AuditInputError("invalid_gguf_tensor_shape")
                element_count = 1
                for dimension in shape:
                    element_count = _mul(element_count, dimension)
                tensor_type, offset = reader.unpack("IQ")
                if offset % alignment:
                    raise AuditInputError("unaligned_gguf_tensor")
                byte_count = None
                if tensor_type in _LAYOUTS:
                    block, block_bytes = _LAYOUTS[tensor_type]
                    if shape[0] % block:
                        raise AuditInputError("invalid_quantized_row")
                    byte_count = _mul(element_count // block, block_bytes)
                    known_bytes = _add(known_bytes, byte_count)
                else:
                    unknown.add(tensor_type)
                tensors.append({"name": name, "shape": shape, "ggml_type": tensor_type,
                                "offset_bytes": offset, "size_bytes": byte_count,
                                "evidence": "source_proven" if byte_count is not None else "unresolved"})
            metadata_end = stream.tell()
            data_offset = _add(metadata_end, alignment - 1) // alignment * alignment
            if data_offset > before.st_size:
                raise AuditInputError("missing_gguf_data_alignment")
            ranges = []
            for tensor in tensors:
                start = _add(data_offset, tensor["offset_bytes"])
                if start > before.st_size:
                    raise AuditInputError("gguf_tensor_out_of_file")
                if tensor["size_bytes"] is not None:
                    if _add(start, tensor["size_bytes"]) > before.st_size:
                        raise AuditInputError("gguf_tensor_out_of_file")
                    ranges.append((start, tensor["size_bytes"]))
            if range_union_bytes(ranges) != known_bytes:
                raise AuditInputError("overlapping_gguf_tensors")
            after = path.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                raise AuditInputError("artifact_changed_during_read")
            architecture = selected.get("general.architecture")
            if architecture is not None:
                architecture = _text(architecture, 256)
            return {"version": version, "file_size_bytes": before.st_size,
                    "metadata_sha256": reader.digest.hexdigest(), "metadata_bytes_read": metadata_end,
                    "tensor_count": tensors_count, "metadata_count": metadata_count,
                    "architecture": architecture, "alignment": alignment, "data_offset_bytes": data_offset,
                    "known_tensor_bytes": known_bytes, "unknown_tensor_types": sorted(unknown),
                    "layout_source": GGML_LAYOUT_SOURCE, "tensors": tensors,
                    "evidence": "source_proven" if not unknown else "unresolved"}
    except OSError as error:
        raise AuditInputError("model_unreadable") from error


def combine_gguf_metadata(shards):
    """Combine shard metadata while keeping tensor offsets shard-relative."""
    if not isinstance(shards, list) or not 1 <= len(shards) <= 128:
        raise AuditInputError("invalid_model_shards")
    combined = dict(shards[0])
    tensors, names = [], set()
    for index, shard in enumerate(shards):
        required = {"version", "file_size_bytes", "metadata_bytes_read", "tensor_count", "metadata_count", "known_tensor_bytes", "metadata_sha256", "unknown_tensor_types", "tensors", "evidence"}
        if not isinstance(shard, dict) or not required <= shard.keys():
            raise AuditInputError("invalid_model_shard_metadata")
        if shard.get("version") != combined.get("version") or (shard.get("architecture") is not None and combined.get("architecture") is not None and shard["architecture"] != combined["architecture"]):
            raise AuditInputError("inconsistent_model_shards")
        if combined.get("architecture") is None:
            combined["architecture"] = shard.get("architecture")
        for tensor in shard.get("tensors", []):
            if tensor["name"] in names:
                raise AuditInputError("duplicate_sharded_tensor_name")
            names.add(tensor["name"])
            tensors.append(dict(tensor, shard_index=index))
    for field in ("file_size_bytes", "metadata_bytes_read", "tensor_count", "metadata_count", "known_tensor_bytes"):
        combined[field] = 0
        for shard in shards:
            combined[field] = _add(combined[field], shard[field])
    combined["metadata_sha256"] = hashlib.sha256("".join(shard["metadata_sha256"] for shard in shards).encode("ascii")).hexdigest()
    combined["unknown_tensor_types"] = sorted({kind for shard in shards for kind in shard["unknown_tensor_types"]})
    combined["data_offset_bytes"] = None  # Offsets have no cross-shard coordinate system.
    combined["tensors"] = tensors
    combined["evidence"] = "source_proven" if all(shard["evidence"] == "source_proven" for shard in shards) else "unresolved"
    return combined


_ENVELOPE = {"schema_version", "report_type", "sequence", "kind", "timestamp_ns"}
_API_FIELDS = {"domain", "callback_id", "correlation_id", "thread_id", "symbol", "status", "detail_known", "op", "context_id", "allocation_id", "generation", "offset_bytes", "size_bytes", "range_known", "memory_kind", "stream_id", "event_id", "module_id", "function_id", "physical_allocation_id", "handle_offset_bytes", "parameter_bytes", "kernel_name", "src_pitch_bytes", "dst_pitch_bytes", "width_bytes", "height", "shared_bytes"}
_API_FIELDS |= {prefix + suffix for prefix in ("src_", "dst_") for suffix in ("allocation_id", "generation", "offset_bytes", "range_known")}
_GEOMETRY = {prefix + axis for prefix in ("grid_", "block_") for axis in ("x", "y", "z")}
_API_FIELDS |= _GEOMETRY
_RESOLVER_FIELDS = {"requested_symbol", "requested_version", "resolver_flags", "query_status", "entry_point_id"}
_RESOLVERS = {("driver", "cuGetProcAddress"), ("driver", "cuGetProcAddress_v2"),
              ("runtime", "cudaGetDriverEntryPoint"), ("runtime", "cudaGetDriverEntryPointByVersion"),
              ("runtime", "cudaGetDriverEntryPoint_ptsz"), ("runtime", "cudaGetDriverEntryPointByVersion_ptsz")}
_KIND_FIELDS = {
    "session": {"collector_version", "process_id", "max_record_bytes", "kernel_arguments_captured", "tensor_bounds_known", "cublas_api_visibility", "timestamp_clock", "activity_clock", "runtime_parameter_bytes_not_kernel_arguments", "run_id", "complete"},
    "api_enter": _API_FIELDS - {"status"}, "api_exit": _API_FIELDS,
    "activity": {"activity_kind", "activity_kind_id", "context_id", "stream_id", "correlation_id", "start_ns", "end_ns", "size_bytes", "name", "detail_known", "shared_bytes", "module_id", "function_id", "allocation_id", "generation", "offset_bytes", "range_known", "memory_kind_id", "operation_id"} | _GEOMETRY,
    "resource": {"resource_kind", "operation", "context_id", "stream_id", "module_id", "size_bytes", "callback_id", "detail_known"},
    "gap": {"reason", "lost_records"},
    "summary": {"callback_records", "activity_records", "resource_records", "dropped_records", "unknown_callback_details", "unknown_activity_kinds", "serialization_errors", "collector_errors", "complete", "safely_finalized", "terminal_checkpoint", "buffers_requested", "buffers_completed", "incomplete_activities"},
    "kernel_binding": {"domain", "correlation_id", "contract_id", "kernel_name", "bindings", "scratch_bytes"},
}
_STRINGS = {"kind", "report_type", "domain", "symbol", "op", "memory_kind", "kernel_name", "activity_kind", "name", "resource_kind", "operation", "reason", "collector_version", "cublas_api_visibility", "timestamp_clock", "activity_clock", "run_id", "terminal_checkpoint", "contract_id"}
_BOOLS = {"detail_known", "range_known", "src_range_known", "dst_range_known", "kernel_arguments_captured", "tensor_bounds_known", "runtime_parameter_bytes_not_kernel_arguments", "complete", "safely_finalized"}
_OPS = {"allocate", "free", "host_allocate", "host_free", "host_register", "host_unregister", "copy", "memset", "vmm_reserve", "vmm_free", "vmm_create", "vmm_release", "vmm_map", "vmm_unmap", "vmm_access", "launch", "module_function", "stream", "event", "other"}


def validate_trace_record(record):
    """Validate the strict, address-redacted normalized record contract."""
    if not isinstance(record, dict) or record.get("kind") not in _KIND_FIELDS:
        raise AuditInputError("invalid_record_kind")
    kind = record["kind"]
    version = record.get("schema_version")
    extra = _RESOLVER_FIELDS if version == 2 and kind in ("api_enter", "api_exit") else set()
    if not _ENVELOPE <= record.keys() or record.keys() - (_ENVELOPE | _KIND_FIELDS[kind] | extra):
        raise AuditInputError("invalid_record_fields")
    if version not in (1, 2) or type(version) is not int or record["report_type"] != TRACE_TYPE:
        raise AuditInputError("invalid_trace_version")
    required = {"api_enter": {"domain", "callback_id", "correlation_id", "thread_id", "symbol", "detail_known"}, "api_exit": {"domain", "callback_id", "correlation_id", "thread_id", "symbol", "detail_known", "status"}, "gap": {"reason", "lost_records"}, "kernel_binding": _KIND_FIELDS["kernel_binding"], "session": _KIND_FIELDS["session"] - {"run_id", "complete"}, "summary": _KIND_FIELDS["summary"], "activity": {"activity_kind", "detail_known"}, "resource": {"resource_kind", "operation", "detail_known"}}[kind]
    if not required <= record.keys():
        raise AuditInputError("missing_record_fields")
    for key, value in record.items():
        if key == "requested_symbol":
            _text(value)
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
                raise AuditInputError("invalid_requested_symbol")
        elif key == "requested_version":
            if type(value) is not int or not -(1 << 31) <= value < (1 << 32):
                raise AuditInputError("invalid_requested_version")
        elif key in _STRINGS:
            _text(value)
        elif key in _BOOLS:
            if type(value) is not bool:
                raise AuditInputError("invalid_boolean")
        elif key in ("status", "query_status"):
            if type(value) is not int or not -(1 << 31) <= value < (1 << 31):
                raise AuditInputError("invalid_native_status")
        elif key == "bindings":
            if not isinstance(value, list) or not 1 <= len(value) <= 256:
                raise AuditInputError("invalid_kernel_bindings")
            for binding in value:
                if not isinstance(binding, dict) or set(binding) != {"tensor_name", "allocation_id", "generation", "offset_bytes", "size_bytes", "access"}:
                    raise AuditInputError("invalid_kernel_binding")
                _text(binding["tensor_name"])
                if binding["access"] not in ("read", "write", "read_write"):
                    raise AuditInputError("invalid_binding_access")
                for field in ("allocation_id", "generation", "offset_bytes", "size_bytes"):
                    _uint(binding[field])
        else:
            _uint(value, key)
    is_resolver = (record.get("domain"), record.get("symbol")) in _RESOLVERS
    resolver_symbol = record.get("symbol", "")
    if resolver_symbol.endswith("_ptsz"):
        resolver_symbol = resolver_symbol[:-5]
    if record.keys() & _RESOLVER_FIELDS:
        if not is_resolver or record.get("op") != "other" or not record.get("detail_known"):
            raise AuditInputError("unexpected_resolver_fields")
        if "requested_symbol" not in record or "resolver_flags" not in record:
            raise AuditInputError("missing_resolver_input")
        if resolver_symbol != "cudaGetDriverEntryPoint" and "requested_version" not in record:
            raise AuditInputError("missing_resolver_version")
        if resolver_symbol == "cudaGetDriverEntryPoint" and "requested_version" in record:
            raise AuditInputError("unexpected_resolver_version")
        if record.keys() & {"query_status", "entry_point_id"}:
            if kind != "api_exit" or record.get("status") != 0:
                raise AuditInputError("premature_resolver_output")
        if "query_status" in record and resolver_symbol == "cuGetProcAddress":
            raise AuditInputError("unexpected_resolver_query_status")
        if "entry_point_id" in record and (record["entry_point_id"] == 0 or record.get("query_status", 0) != 0):
            raise AuditInputError("invalid_resolver_entry_point")
    elif version == 2 and is_resolver and record.get("detail_known"):
        raise AuditInputError("missing_resolver_input")
    if record["sequence"] == 0:
        raise AuditInputError("invalid_sequence")
    if "domain" in record and record["domain"] not in ("runtime", "driver"):
        raise AuditInputError("invalid_domain")
    if "op" in record and record["op"] not in _OPS:
        raise AuditInputError("invalid_operation")
    if "activity_kind" in record and record["activity_kind"] not in ("kernel", "api_runtime", "api_driver", "memcpy", "memset", "memory", "function", "unknown"):
        raise AuditInputError("invalid_activity_kind")
    if "resource_kind" in record and record["resource_kind"] not in ("context", "stream", "module", "other"):
        raise AuditInputError("invalid_resource_kind")
    if kind == "resource" and record["operation"] not in ("create", "destroy", "load", "unload", "other"):
        raise AuditInputError("invalid_resource_operation")
    if "start_ns" in record and "end_ns" in record and record["end_ns"] < record["start_ns"]:
        raise AuditInputError("invalid_activity_interval")


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AuditInputError("duplicate_json_key")
        result[key] = value
    return result


def _trace_records(path, *, digest=None):
    total = 0
    try:
        with Path(path).open("rb") as stream:
            for index in range(MAX_RECORDS + 1):
                line = stream.readline(MAX_RECORD_BYTES + 1)
                if not line:
                    return
                total += len(line)
                if len(line) > MAX_RECORD_BYTES or total > MAX_TRACE_BYTES or index == MAX_RECORDS:
                    raise AuditInputError("trace_limit")
                if digest is not None:
                    digest.update(line)
                try:
                    record = json.loads(line, object_pairs_hook=_json_object,
                                        parse_constant=lambda _: (_ for _ in ()).throw(AuditInputError("nonfinite_json")))
                    validate_trace_record(record)
                except (ValueError, UnicodeError, RecursionError) as error:
                    if isinstance(error, AuditInputError):
                        raise
                    raise AuditInputError("malformed_json_record") from error
                yield record
    except OSError as error:
        raise AuditInputError("trace_unreadable") from error


def _source_catalog(profile, issues):
    contracts, entries = {}, []
    candidates = profile.get("source_contracts", [])
    if not isinstance(candidates, list) or len(candidates) > 1024:
        raise AuditInputError("source_contract_limit")
    for candidate in candidates:
        try:
            required = {"contract_id", "kernel_name", "source_path", "source_sha256", "source_url", "upstream_commit", "tensor_names", "scratch_bytes"}
            if not isinstance(candidate, dict) or set(candidate) != required:
                raise AuditInputError("invalid_source_contract")
            contract_id = _text(candidate["contract_id"], 256)
            kernel_name = _text(candidate["kernel_name"])
            digest = candidate["source_sha256"]
            if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                raise AuditInputError("invalid_source_hash")
            if not re.fullmatch(r"[0-9a-f]{40}", candidate["upstream_commit"]):
                raise AuditInputError("unpinned_source_contract")
            url = _text(candidate["source_url"])
            if not url.startswith("https://") or candidate["upstream_commit"] not in url:
                raise AuditInputError("unpinned_source_contract")
            source_path = Path(candidate["source_path"])
            if source_path.stat().st_size > MAX_METADATA_BYTES:
                raise AuditInputError("source_contract_limit")
            with source_path.open("rb") as stream:
                if _hash_file(stream) != digest:
                    raise AuditInputError("source_hash_mismatch")
            names = candidate["tensor_names"]
            if not isinstance(names, list) or not names or len(names) > 256 or len(set(names)) != len(names):
                raise AuditInputError("invalid_contract_tensor_names")
            for name in names:
                _text(name)
            _uint(candidate["scratch_bytes"])
            if contract_id in contracts:
                raise AuditInputError("duplicate_source_contract")
            contracts[contract_id] = candidate
            entries.append({"contract_id": contract_id, "kernel_name": kernel_name,
                            "source_sha256": digest, "source_url": url,
                            "upstream_commit": candidate["upstream_commit"], "evidence": "source_proven"})
        except (OSError, AuditInputError, TypeError) as error:
            issues[str(error) if isinstance(error, AuditInputError) else "source_contract_unreadable"] += 1
    return contracts, entries


def analyze(trace_paths, inventory_data, model_metadata, profile, *, provenance=None, cleanup=None):
    """Return a strict v1 report. A completed audit is distinct from a GO verdict.

    Optional source contracts must be pinned and hash-verified. They cannot replace
    actual per-launch bindings. The normalized observer currently emits none.
    """
    issues, counts, kernel_types = Counter(), Counter(), Counter()
    api_symbols, operations, resources, activity_kernels = Counter(), Counter(), Counter(), Counter()
    profile = dict(profile)
    profile_id = _text(profile.get("profile_id", "unspecified"), 256)
    cache_budget = profile.get("cache_budget_bytes")
    if cache_budget is not None:
        cache_budget = _uint(cache_budget)
    scratch_reserve = profile.get("scratch_reserve_bytes")
    if scratch_reserve is not None:
        scratch_reserve = _uint(scratch_reserve)
    else:
        issues["scratch_reserve_unknown"] += 1
    chunk_bytes = _uint(profile.get("chunk_bytes", 64 * 1024 * 1024))
    if not chunk_bytes:
        raise AuditInputError("invalid_chunk_bytes")
    contracts, source_catalog = _source_catalog(profile, issues)
    paths = list(trace_paths)
    if len(paths) > 64:
        raise AuditInputError("trace_file_limit")
    if not paths:
        issues["missing_trace"] += 1
    live_peak, copy_bytes, working_peak = 0, 0, 0
    resolved_kernels, invalid_input, summaries_complete = 0, False, 0
    if isinstance(model_metadata, list):
        model_metadata = combine_gguf_metadata(model_metadata)
    tensors = {tensor["name"]: tensor for tensor in (model_metadata or {}).get("tensors", [])}
    for path in paths:
        pending, live, generations, event_records, bindings, launches = {}, {}, {}, {}, {}, {}
        activity_correlations = Counter()
        binding_reference_count = 0
        record_counts, sequence, session, summary, live_bytes = Counter(), 0, None, None, 0
        try:
            for record in _trace_records(path):
                if binding_reference_count + sum(map(len, (pending, live, generations, event_records, bindings, launches, activity_correlations, kernel_types, api_symbols, activity_kernels, resources))) > MAX_STATE_ITEMS:
                    raise AuditInputError("trace_state_limit")
                counts["records"] += 1
                kind = record["kind"]
                if record["sequence"] != sequence + 1:
                    issues["sequence_gap"] += 1
                sequence = record["sequence"]
                if summary is not None:
                    issues["records_after_summary"] += 1
                if session is None and kind != "session":
                    issues["missing_initial_session"] += 1
                record_counts[kind] += 1
                counts[kind] += 1
                if kind == "session":
                    if session is not None or sequence != 1:
                        issues["multiple_or_misplaced_session"] += 1
                    session = record
                    if record["cublas_api_visibility"] == "unavailable":
                        issues["cublas_api_visibility_unavailable"] += 1
                    if record["timestamp_clock"] != "steady_clock" or record["activity_clock"] != "cupti":
                        issues["unknown_timestamp_clock"] += 1
                    continue
                if kind == "gap":
                    issues["collector_gap"] += 1
                    counts["gap_lost_records"] += record["lost_records"]
                    if record["reason"] == "cublas_api_not_observed_by_cupti":
                        issues["cublas_api_visibility_unavailable"] += 1
                    continue
                if kind == "summary":
                    summary = record
                    continue
                if record.get("detail_known") is False:
                    issues["unknown_record_details"] += 1
                if kind == "kernel_binding":
                    # A source hash establishes artifact identity, not a trusted
                    # bridge from module/function ABI to actual launch arguments.
                    # v1 has no such bridge; declarations cannot grant readiness.
                    issues["typed_module_function_bridge_unverified"] += 1
                    key = (record["domain"], record["correlation_id"])
                    if key in bindings:
                        issues["duplicate_kernel_binding"] += 1
                    # Capture the lifetime state at the evidence checkpoint. A
                    # later free must not make an earlier launch appear invalid.
                    references = {(item["allocation_id"], item["generation"]) for item in record["bindings"]}
                    binding_reference_count += len(references)
                    bindings[key] = (record, {ref: live[ref] for ref in references if ref in live})
                    continue
                if kind == "api_enter":
                    api_symbols[(record["domain"], record["symbol"])] += 1
                    key = (record["domain"], record["correlation_id"], record["thread_id"])
                    if key in pending:
                        issues["duplicate_api_enter"] += 1
                    pending[key] = record
                    continue
                if kind == "api_exit":
                    key = (record["domain"], record["correlation_id"], record["thread_id"])
                    entered = pending.pop(key, None)
                    if entered is None or entered["symbol"] != record["symbol"]:
                        issues["unmatched_api_exit"] += 1
                    if record["status"] != 0:
                        counts["failed_api_calls"] += 1
                        continue
                    operation = record.get("op", (entered or {}).get("op", "other"))
                    operations[operation] += 1
                    merged = dict(entered or {})
                    merged.update(record)
                    allocation = (merged.get("allocation_id"), merged.get("generation"))
                    if operation in ("allocate", "vmm_reserve"):
                        if None in allocation or not merged.get("range_known") or "size_bytes" not in merged:
                            issues["unresolved_allocation_range"] += 1
                        elif allocation in live or allocation[1] <= generations.get(allocation[0], -1):
                            issues["allocation_identity_repeated_or_nested_api"] += 1
                        else:
                            if any(item[0] == allocation[0] for item in live):
                                issues["overlapping_allocation_lifetime"] += 1
                            live[allocation] = merged["size_bytes"]
                            generations[allocation[0]] = allocation[1]
                            live_bytes = _add(live_bytes, merged["size_bytes"])
                            live_peak = max(live_peak, live_bytes)
                    elif operation in ("free", "vmm_free"):
                        if allocation not in live:
                            issues["free_identity_missing_or_nested_api"] += 1
                        else:
                            live_bytes -= live.pop(allocation)
                    elif operation in ("copy", "memset"):
                        size = merged.get("size_bytes")
                        if size is None:
                            issues["unresolved_copy_range"] += 1
                        else:
                            copy_bytes = _add(copy_bytes, size)
                            for prefix in (("src_", "dst_") if operation == "copy" else ("",)):
                                if prefix + "allocation_id" not in merged:
                                    continue  # Host endpoints need not be device allocations.
                                span = size
                                if "height" in merged and merged["height"]:
                                    width = merged.get("width_bytes")
                                    pitch = merged.get(prefix + "pitch_bytes")
                                    if width is None or pitch is None or width > pitch:
                                        issues["unresolved_pitched_range"] += 1
                                        continue
                                    span = _add(_mul(merged["height"] - 1, pitch), width)
                                ref = (merged.get(prefix + "allocation_id"), merged.get(prefix + "generation"))
                                if not merged.get(prefix + "range_known") or prefix + "offset_bytes" not in merged:
                                    issues["unresolved_copy_range"] += 1
                                elif ref not in live or _add(merged[prefix + "offset_bytes"], span) > live[ref]:
                                    issues["copy_outside_live_allocation"] += 1
                    elif operation == "launch":
                        launch_key = (record["domain"], record["correlation_id"])
                        if session and session["kernel_arguments_captured"] and session["tensor_bounds_known"]:
                            launches[launch_key] = merged
                        else:
                            counts["kernel_launches"] += 1
                            issues["missing_kernel_binding"] += 1
                            issues["kernel_activity_correlation_unresolved"] += 1
                        name = merged.get("kernel_name", merged["symbol"])
                        kernel_types[name] += 1
                    elif operation in ("vmm_create", "vmm_release", "vmm_map", "vmm_unmap", "vmm_access"):
                        issues["vmm_alias_contract_unresolved"] += 1
                    elif operation == "event":
                        issues["asynchronous_ordering_unresolved"] += 1
                        event = merged.get("event_id")
                        symbol = merged["symbol"].lower()
                        if event is None:
                            issues["event_identity_unresolved"] += 1
                        elif "record" in symbol:
                            event_records[event] = event_records.get(event, 0) + 1
                        elif "wait" in symbol or "synchronize" in symbol:
                            if event not in event_records:
                                issues["event_wait_without_record"] += 1
                        elif "destroy" in symbol:
                            event_records.pop(event, None)
                    elif operation == "other":
                        issues["api_semantics_unresolved"] += 1
                elif kind == "activity" and record["activity_kind"] == "kernel":
                    counts["kernel_activities"] += 1
                    activity_kernels[record.get("name", "Unknown")] += 1
                    if "correlation_id" not in record:
                        issues["kernel_activity_without_correlation"] += 1
                    elif session and session["kernel_arguments_captured"] and session["tensor_bounds_known"]:
                        activity_correlations[record["correlation_id"]] += 1
                elif kind == "resource":
                    resources[(record["resource_kind"], record["operation"])] += 1
            if pending:
                issues["unclosed_api_calls"] += len(pending)
            if session is None:
                issues["missing_session"] += 1
            if summary is None:
                issues["missing_summary"] += 1
            else:
                for field in ("dropped_records", "unknown_callback_details", "unknown_activity_kinds", "serialization_errors", "collector_errors", "incomplete_activities"):
                    counts[field] += summary[field]
                    if summary[field]:
                        issues[field] += summary[field]
                expected = {"callback_records": record_counts["api_enter"] + record_counts["api_exit"], "activity_records": record_counts["activity"], "resource_records": record_counts["resource"]}
                if any(summary[field] != count for field, count in expected.items()):
                    issues["summary_counter_mismatch"] += 1
                if summary["buffers_requested"] != summary["buffers_completed"]:
                    issues["uncompleted_activity_buffers"] += 1
                if not summary["complete"] or not summary["safely_finalized"] or summary["terminal_checkpoint"] != "explicit_finalize":
                    issues["collector_not_safely_finalized"] += 1
                else:
                    summaries_complete += 1
            if live:
                issues["allocations_live_at_trace_end"] += len(live)
            for key, launch in launches.items():
                counts["kernel_launches"] += 1
                if activity_correlations.pop(key[1], 0) != 1:
                    issues["kernel_activity_correlation_unresolved"] += 1
                bound = bindings.pop(key, None)
                if bound is None:
                    issues["missing_kernel_binding"] += 1
                    continue
                binding, at_binding = bound
                contract = contracts.get(binding["contract_id"])
                if not session or not session["kernel_arguments_captured"] or not session["tensor_bounds_known"]:
                    issues["typed_kernel_arguments_unavailable"] += 1
                    continue
                if contract is None or contract["kernel_name"] != binding["kernel_name"] or launch.get("kernel_name") != binding["kernel_name"]:
                    issues["kernel_source_contract_unresolved"] += 1
                    continue
                if sorted(contract["tensor_names"]) != sorted(item["tensor_name"] for item in binding["bindings"]):
                    issues["kernel_tensor_contract_mismatch"] += 1
                    continue
                ranges, valid = {}, True
                for item in binding["bindings"]:
                    tensor = tensors.get(item["tensor_name"])
                    ref = (item["allocation_id"], item["generation"])
                    if tensor is None or tensor.get("size_bytes") is None or tensor["size_bytes"] != item["size_bytes"]:
                        issues["kernel_tensor_extent_unresolved"] += 1
                        valid = False
                    elif ref not in at_binding or _add(item["offset_bytes"], item["size_bytes"]) > at_binding[ref]:
                        issues["kernel_binding_outside_lifetime"] += 1
                        valid = False
                    else:
                        ranges.setdefault(ref, []).append((item["offset_bytes"], item["size_bytes"]))
                if binding["scratch_bytes"] != contract["scratch_bytes"]:
                    issues["kernel_scratch_unresolved"] += 1
                    valid = False
                if valid:
                    rounded = []
                    for spans in ranges.values():
                        rounded.append(range_union_bytes([(offset // chunk_bytes * chunk_bytes,
                            _add(_add(offset, size), chunk_bytes - 1) // chunk_bytes * chunk_bytes - offset // chunk_bytes * chunk_bytes)
                            for offset, size in spans]))
                    working = _add(sum(rounded), binding["scratch_bytes"])
                    working_peak = max(working_peak, working)
                    resolved_kernels += 1
            if bindings:
                issues["binding_without_launch"] += len(bindings)
            if activity_correlations:
                issues["kernel_activity_without_launch"] += sum(activity_correlations.values())
        except AuditInputError as error:
            issues[str(error)] += 1
            invalid_input = True
    if not counts["kernel_launches"] or not counts["kernel_activities"]:
        issues["empty_kernel_coverage"] += 1
    if counts["kernel_launches"] != counts["kernel_activities"]:
        issues["kernel_callback_activity_mismatch"] += 1
    if not inventory_data or any(item.get("evidence") != "observed" for item in inventory_data):
        issues["artifact_inventory_unresolved"] += 1
    if not model_metadata or model_metadata.get("evidence") != "source_proven" or not tensors:
        issues["model_metadata_unresolved"] += 1
    cleanup = dict(cleanup or {})
    cleanup_view = {key: cleanup.get(key) for key in ("controller_reaped", "collector_finalized", "complete")}
    if any(value is not True for value in cleanup_view.values()):
        issues["cleanup_not_proven"] += 1
    known_working = (counts["kernel_launches"] > 0 and resolved_kernels == counts["kernel_launches"]
                     and not issues["typed_module_function_bridge_unverified"])
    if not known_working:
        resolved_kernels = 0
    required_bytes = _add(working_peak, scratch_reserve) if known_working and scratch_reserve is not None else None
    if not cache_budget:
        issues["cache_budget_unspecified"] += 1
    elif required_bytes is not None and required_bytes > cache_budget:
        issues["working_set_exceeds_cache_budget"] += 1
    # Provenance is descriptive only and cannot grant execution readiness.
    provenance = dict(provenance or {})
    provenance_view = {}
    for field in ("profile_id", "model_revision", "upstream_commit", "collector_version", "analyzer_version", "observation_file", "observation_file_sha256"):
        value = provenance.get(field)
        provenance_view[field] = _text(value, 512) if value is not None else None
    provenance_view["binary_sha256"] = [item["sha256"] for item in inventory_data if item.get("sha256")]
    provenance_view["source_catalog"] = source_catalog
    source_evidence = profile.get("source_evidence", [])
    if not isinstance(source_evidence, list) or len(source_evidence) > 256:
        raise AuditInputError("source_evidence_limit")
    provenance_view["source_evidence"] = []
    for item in source_evidence:
        if not isinstance(item, dict) or set(item) != {"evidence_id", "status", "source", "claim", "limitation"} or item["status"] not in ("observed", "source_proven", "unresolved"):
            raise AuditInputError("invalid_source_evidence")
        provenance_view["source_evidence"].append({key: _text(value) for key, value in item.items()})
    versions = provenance.get("tool_versions", {})
    if not isinstance(versions, dict) or len(versions) > 64:
        raise AuditInputError("invalid_tool_versions")
    provenance_view["tool_versions"] = {_text(key, 128): _text(value, 512) for key, value in versions.items()}
    report_inventory = []
    for item in inventory_data:
        # Never allow caller-provided path fields into the default report.
        clean = {key: item.get(key) for key in ("artifact_id", "name", "sha256", "size_bytes", "format", "evidence", "pe", "errors")}
        report_inventory.append(clean)
    model_fields = ("version", "file_size_bytes", "metadata_sha256", "metadata_bytes_read", "tensor_count", "metadata_count", "architecture", "alignment", "data_offset_bytes", "known_tensor_bytes", "unknown_tensor_types", "layout_source", "evidence")
    model_view = {key: (model_metadata or {}).get(key) for key in model_fields}
    coverage = {key: counts[key] for key in ("records", "session", "api_enter", "api_exit", "activity", "resource", "gap", "kernel_launches", "kernel_activities", "failed_api_calls", "dropped_records", "unknown_callback_details", "unknown_activity_kinds", "serialization_errors", "collector_errors", "incomplete_activities", "gap_lost_records")}
    coverage.update({"trace_files": len(paths), "complete_trace_files": summaries_complete,
                     "resolved_kernel_launches": resolved_kernels,
                     "kernel_types": [{"name": name, "launches": count, "semantic_ranges": "source_proven" if known_working else "unresolved"} for name, count in sorted(kernel_types.items())],
                     "evidence": "observed"})
    coverage["api_symbols"] = [{"domain": domain, "symbol": symbol, "calls": count} for (domain, symbol), count in sorted(api_symbols.items())]
    coverage["operations"] = [{"op": operation, "successful_calls": count} for operation, count in sorted(operations.items())]
    coverage["resources"] = [{"resource_kind": kind, "operation": operation, "records": count} for (kind, operation), count in sorted(resources.items())]
    coverage["activity_kernel_types"] = [{"name": name, "activities": count,
        "argument_abi": "unresolved", "read_write_ranges": "unresolved",
        "padding": "unresolved", "indirect_access": "unresolved", "scratch": "unresolved",
        "module_function_hash_identity": "unresolved"} for name, count in sorted(activity_kernels.items())]
    unresolved = [{"code": code, "count": count, "evidence": "unresolved"} for code, count in sorted(issues.items()) if count]
    go = not unresolved and known_working
    return {"schema_version": 1, "report_type": REPORT_TYPE,
            "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
            "provenance": provenance_view,
            "configuration": {"profile_id": profile_id, "cache_budget_bytes": cache_budget, "scratch_reserve_bytes": scratch_reserve, "chunk_bytes": chunk_bytes,
                              "max_record_bytes": MAX_RECORD_BYTES, "max_trace_bytes": MAX_TRACE_BYTES, "max_records": MAX_RECORDS,
                              "raw_addresses_serialized": False, "weight_payloads_read_by_metadata_parser": False},
            "inventory": report_inventory, "model": model_view, "coverage": coverage,
            "memory_bounds": {"observed_peak_live_allocation_bytes": live_peak, "observed_copy_bytes": copy_bytes,
                              "source_proven_peak_kernel_working_set_bytes": working_peak if known_working else None,
                              "required_cache_bytes_including_scratch_reserve": required_bytes,
                              "allocation_sizes_are_tensor_bounds": False,
                              "observed_allocation_metric": "storage_extents_not_physical_vram",
                              "evidence": "source_proven" if known_working else "unresolved"},
            "unresolved": unresolved,
            "decision": {"verdict": "GO" if go else "NO-GO", "execution_ready": go,
                         "scope": "pinned_observed_workload_only", "interoperability_impossible_proven": False},
            "outcome": {"status": "invalid_input" if invalid_input else "completed", "exit_code": 64 if invalid_input else 0},
            "cleanup": cleanup_view}
