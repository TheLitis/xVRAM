#!/usr/bin/env python3
"""Generate fieldwise capture tables from the exact reviewed, hash-pinned catalog.

This is build-time metadata generation only. It never opens a CUDA process,
dereferences an argument, or promotes typed capture candidates into proof.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

# The source tree path is anchored to this file, not the working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from xvram.compat_audit_kernel_catalog import BASELINE_FAMILIES, expected_layout

CATALOG_SHA256 = "b8fd7a2fcf2aa61bb8de5ac7c85ea7f3e8b9aaff446fb899af3b7986af033494"
MAX_CATALOG_BYTES = 1024 * 1024
SCALAR_TYPES = {"ptr", "i64", "u64", "i32", "u32", "f32", "bool"}


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("capture_catalog_duplicate_key")
        value[key] = item
    return value


def load_catalog(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_CATALOG_BYTES + 1)
    if len(raw) > MAX_CATALOG_BYTES:
        raise ValueError("capture_catalog_size_limit")
    if hashlib.sha256(raw).hexdigest() != CATALOG_SHA256:
        raise ValueError("capture_catalog_unreviewed_hash")
    result = json.loads(raw, object_pairs_hook=unique_object,
                        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("capture_catalog_nonfinite")))
    validate_catalog(result)
    return result


def validate_catalog(value):
    if (type(value) is not dict or type(value.get("schema_version")) is not int or value["schema_version"] != 1 or
            value.get("catalog_type") != "xvram.cuda_kernel_capture_catalog" or value.get("version") != "0.1.0-dev"):
        raise ValueError("capture_catalog_contract")
    rows = value.get("kernels")
    if type(rows) is not list or len(rows) != 39:
        raise ValueError("capture_catalog_coverage")
    names = set()
    for index, (row, family) in enumerate(zip(rows, BASELINE_FAMILIES), 1):
        if type(row) is not dict or type(row.get("catalog_id")) is not int or row["catalog_id"] != index:
            raise ValueError("capture_catalog_order")
        name = row.get("symbol")
        if not isinstance(name, str) or not name.isascii() or not name or len(name) > 1024 or name in names:
            raise ValueError("capture_catalog_symbol")
        names.add(name)
        if row.get("source_family") != family or row.get("source_semantics_bound_to_cubin") is not False or row.get("memory_bounds_proven") is not False:
            raise ValueError("capture_catalog_source_scope")
        expected_status = "typed_capture_candidate" if family else "unsupported_opaque_library"
        if row.get("status") != expected_status or row.get("arguments") != (expected_layout(family) if family else []):
            raise ValueError("capture_catalog_field_contract")
        if family and row.get("observed_parameter_count") != len(row["arguments"]):
            raise ValueError("capture_catalog_argument_count")


def fields(argument):
    kind = argument["type"]
    if kind in SCALAR_TYPES:
        return [("value", "boolean" if kind == "bool" else kind, 0)]
    if kind == "uint3":
        return [(name, "u32", index*4) for index, name in enumerate(("x", "y", "z"))]
    if kind == "f32x2":
        return [(name, "f32", index*4) for index, name in enumerate(("v0", "v1"))]
    if kind in ("fusion_struct", "soft_max_params_struct"):
        return [(field["name"], field["type"], field["offset_bytes"]) for field in argument["fields"]]
    raise ValueError("capture_catalog_unknown_type")


def render(value):
    validate_catalog(value)
    lines = ["// Generated from the reviewed xVRAM capture catalog; do not edit.", "#pragma once",
             '#include "typed_capture.hpp"', "#include <array>", "#include <span>", "",
             "namespace xvram::launch_probe::typed_capture {",
             f'inline constexpr const char catalog_sha256[] = "{CATALOG_SHA256}";', ""]
    for kernel in value["kernels"]:
        kid = kernel["catalog_id"]
        for argument in kernel["arguments"]:
            aid = argument["ordinal"]
            members = fields(argument)
            lines.append(f"inline constexpr std::array<Field, {len(members)}> fields_{kid}_{aid}{{{{")
            lines.extend(f"  {{{json.dumps(name)}, ValueType::{kind}, {offset}}}," for name, kind, offset in members)
            lines.append("}};")
        arguments = kernel["arguments"]
        if arguments:
            lines.append(f"inline constexpr std::array<Argument, {len(arguments)}> arguments_{kid}{{{{")
            for arg in arguments:
                aid = arg["ordinal"]
                lines.append(f'  {{{json.dumps(arg["name"])}, {arg["offset_bytes"]}, {arg["size_bytes"]}, '
                             f'std::span<const Field>{{fields_{kid}_{aid}}}}},')
            lines.append("}};")
        else:
            lines.append(f"inline constexpr std::array<Argument, 0> arguments_{kid}{{}};")
        lines.append("")
    lines.append("inline constexpr std::array<Kernel, 39> kernels{{")
    for kernel in value["kernels"]:
        supported = "true" if kernel["source_family"] else "false"
        kid = kernel["catalog_id"]
        lines.append(f'  {{{kid}, {json.dumps(kernel["symbol"])}, std::span<const Argument>{{arguments_{kid}}}, {supported}}},')
    lines.extend(("}};", "} // namespace xvram::launch_probe::typed_capture", ""))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = render(load_catalog(args.catalog))
    output = Path(args.output)
    if output.suffix not in (".hpp", ".h"):
        raise ValueError("capture_header_output_extension")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists() or output.read_text(encoding="utf-8") != result:
        output.write_text(result, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
