"""Strict typed launch snapshots; allocation identity is NOT tensor bounds.

The catalog pins source-reviewed field layouts, not binary-source equivalence.
Even a completely captured file cannot establish GPU retirement, indirect index
contents, observer lifetime or cubin semantics. All four overall proofs stay
false. Unknown/opaque calls are explicit observations, never guessed layouts.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

from .compat_audit_identity import _uint, records, digest
from .compat_launch_probe import _object, analyze_trace, TRACE_CAP

RECORD_TYPE = "xvram.cuda_kernel_arguments"
CATALOG_SHA256 = "b8fd7a2fcf2aa61bb8de5ac7c85ea7f3e8b9aaff446fb899af3b7986af033494"
BYTE_CAP, CALL_CAP, FIELD_CAP = 512*1024*1024, 100000, 4000000
COMMON = {"schema_version", "record_type", "sequence", "kind", "call"}


def catalog():
    raw = Path(__file__).with_name("compat_audit_kernel_capture_catalog.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != CATALOG_SHA256:
        raise ValueError("arguments_catalog_hash")
    return json.loads(raw, object_pairs_hook=_object)["kernels"]


def field_layout(kernel):
    result = []
    for arg in kernel["arguments"]:
        kind = arg["type"]
        fields = [("value", kind)]
        if kind == "uint3":
            fields = [(name, "u32") for name in ("x", "y", "z")]
        elif kind == "f32x2":
            fields = [(name, "f32") for name in ("v0", "v1")]
        elif kind in ("fusion_struct", "soft_max_params_struct"):
            fields = [(field["name"], field["type"]) for field in arg["fields"]]
        result += [(arg["ordinal"], arg["name"], name, typ) for name, typ in fields]
    return result


def _fields(row, names):
    if set(row) != COMMON | set(names.split()):
        raise ValueError("arguments_fields")


def _bool(value):
    if type(value) is not bool:
        raise ValueError("arguments_boolean")


def _false(value):
    if value is not False:
        raise ValueError("arguments_unproven_claim")


def _integer(value, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("arguments_integer")


def _field(row, expected):
    if (row.get("ordinal"), row.get("argument"), row.get("field"), row.get("value_type")) != expected:
        raise ValueError("arguments_field_order_or_type")
    _uint(row["ordinal"], maximum=63)
    common = "ordinal argument field value_type"
    typ = row["value_type"]
    if typ == "ptr":
        resolution = row.get("resolution")
        names = common + " resolution memory_bounds_proven"
        if resolution == "allocation_identity":
            names += " allocation_id generation offset_bytes allocation_bytes lookup_bytes memory_revision memory_sequence alignment base_mod_alignment mapped readable writable"
        elif resolution not in ("null", "unresolved"):
            raise ValueError("arguments_pointer_resolution")
        _fields(row, names)
        _false(row["memory_bounds_proven"])
        if resolution == "allocation_identity":
            for key in ("allocation_id", "generation", "memory_revision", "memory_sequence"):
                _uint(row[key], 1)
            size = _uint(row["allocation_bytes"], 1)
            _uint(row["offset_bytes"], maximum=size-1)
            if type(row["lookup_bytes"]) is not int or row["lookup_bytes"] != 1:
                raise ValueError("arguments_lookup_not_tensor")
            if type(row["alignment"]) is not int or row["alignment"] != 16:
                raise ValueError("arguments_alignment")
            _uint(row["base_mod_alignment"], maximum=15)
            for key in ("mapped", "readable", "writable"):
                _bool(row[key])
            if not row["mapped"] and (row["readable"] or row["writable"]):
                raise ValueError("arguments_unmapped_access")
        return int(resolution == "unresolved")
    _fields(row, common + (" bits_u32" if typ == "f32" else " value"))
    if typ == "f32":
        _uint(row["bits_u32"], maximum=2**32-1)
    elif typ == "bool":
        _bool(row["value"])
    elif typ in ("u32", "u64"):
        _uint(row["value"], maximum=2**(32 if typ == "u32" else 64)-1)
    elif typ in ("i32", "i64"):
        bits = 32 if typ == "i32" else 64
        _integer(row["value"], -(2**(bits-1)), 2**(bits-1)-1)
    else:
        raise ValueError("arguments_scalar_type")
    return 0


def analyze_rows(rows):
    kernels = catalog()
    counts = Counter(records=0, calls=0, returns=0, fields=0, unsupported=0,
                     capture_faults=0, unresolved_pointers=0, errors=0)
    calls, active, summary = {}, None, None
    last_call = 0
    for sequence, row in enumerate(rows, 1):
        if (type(row) is not dict or row.get("record_type") != RECORD_TYPE or
                type(row.get("schema_version")) is not int or row["schema_version"] != 1 or
                _uint(row.get("sequence"), 1) != sequence):
            raise ValueError("arguments_profile_or_sequence")
        if summary is not None:
            raise ValueError("arguments_after_summary")
        counts["records"] += 1
        kind, call = row.get("kind"), _uint(row.get("call"), maximum=CALL_CAP)
        if sequence == 1 and kind != "session":
            raise ValueError("arguments_session_missing")
        if kind == "session":
            _fields(row, "catalog_sha256 byte_cap call_cap field_cap memory_observer_enabled memory_bounds_proven source_semantics_bound_to_cubin")
            if sequence != 1 or call or row["catalog_sha256"] != CATALOG_SHA256:
                raise ValueError("arguments_session")
            for key, expected in (("byte_cap", BYTE_CAP), ("call_cap", CALL_CAP), ("field_cap", FIELD_CAP)):
                if type(row[key]) is not int or row[key] != expected:
                    raise ValueError("arguments_cap")
            _bool(row["memory_observer_enabled"])
            _false(row["memory_bounds_proven"]); _false(row["source_semantics_bound_to_cubin"])
        elif kind == "begin":
            _fields(row, "catalog_id grid_x grid_y grid_z block_x block_y block_z shared_bytes device memory_observer_healthy")
            if active is not None or not call or call <= last_call:
                raise ValueError("arguments_call_order")
            last_call = call
            kid = _uint(row["catalog_id"], maximum=len(kernels))
            for key in ("grid_x", "grid_y", "grid_z", "block_x", "block_y", "block_z", "shared_bytes"):
                _uint(row[key], maximum=2**32-1)
            _integer(row["device"], -1, 2**31-1)
            _bool(row["memory_observer_healthy"])
            active = {"begin": row, "fields": [], "unresolved": 0, "status": None}
            active["expected"] = field_layout(kernels[kid-1]) if kid else []
            counts["calls"] += 1
        elif kind == "field":
            if active is None or call != last_call or active["status"] is not None:
                raise ValueError("arguments_field_lifetime")
            index = len(active["fields"])
            if index >= len(active["expected"]):
                raise ValueError("arguments_extra_or_opaque_field")
            active["unresolved"] += _field(row, active["expected"][index])
            active["fields"].append(row)
            counts["fields"] += 1
            if counts["fields"] > FIELD_CAP:
                raise ValueError("arguments_field_cap")
        elif kind == "capture_end":
            _fields(row, "status fields unresolved_pointers memory_bounds_proven")
            if active is None or call != last_call or active["status"] is not None:
                raise ValueError("arguments_capture_end_order")
            status, kid = row["status"], active["begin"]["catalog_id"]
            valid = {"captured", "unsupported_symbol", "unsupported_opaque_library", "unsupported_abi",
                     "unsupported_argument_bank", "capture_fault"}
            if type(status) is not str or status not in valid:
                raise ValueError("arguments_capture_status")
            if status == "unsupported_symbol":
                if kid:
                    raise ValueError("arguments_symbol_status")
            elif status == "unsupported_opaque_library":
                if kid < 36:
                    raise ValueError("arguments_opaque_status")
            elif not 1 <= kid <= 35:
                raise ValueError("arguments_supported_status")
            if status == "captured":
                if len(active["fields"]) != len(active["expected"]):
                    raise ValueError("arguments_missing_fields")
            elif active["fields"]:
                raise ValueError("arguments_partial_failed_capture")
            if (_uint(row["fields"], maximum=FIELD_CAP) != len(active["fields"]) or
                    _uint(row["unresolved_pointers"]) != active["unresolved"]):
                raise ValueError("arguments_capture_counts")
            _false(row["memory_bounds_proven"])
            active["status"] = status
            counts["unsupported"] += status.startswith("unsupported_")
            counts["capture_faults"] += status == "capture_fault"
            counts["unresolved_pointers"] += active["unresolved"]
        elif kind == "return":
            _fields(row, "result")
            if active is None or call != last_call or active["status"] is None:
                raise ValueError("arguments_return_order")
            _integer(row["result"], -(2**31), 2**31-1)
            active["result"] = row["result"]
            del active["expected"]
            calls[call], active = active, None
            counts["returns"] += 1
        elif kind == "summary":
            _fields(row, "calls returns fields unsupported capture_faults unresolved_pointers errors terminal_complete memory_bounds_proven source_semantics_bound_to_cubin")
            if call or active is not None:
                raise ValueError("arguments_summary_order")
            for key in ("calls", "returns", "fields", "unsupported", "capture_faults", "unresolved_pointers", "errors"):
                if _uint(row[key]) != counts[key]:
                    raise ValueError("arguments_summary_counts")
            _false(row["terminal_complete"]); _false(row["memory_bounds_proven"])
            _false(row["source_semantics_bound_to_cubin"])
            summary = row
        else:
            raise ValueError("arguments_unknown_or_incomplete_record")
    if summary is None:
        raise ValueError("arguments_summary_missing")
    return dict(counts), calls


def bind_launches(calls, launch_path):
    """Join every snapshot to validated actual host-launch metadata and return.

    This is a call-ID + exact name/layout join, not an Activity/cubin proof.
    Re-hashing detects mutation during the separately bounded validation pass.
    """
    launch_path = Path(launch_path)
    validated = analyze_trace(launch_path)
    if validated["counts"]["calls_returned"] != len(calls):
        raise ValueError("arguments_launch_count")
    kernels = catalog()
    names = {item["symbol"] for item in kernels}
    observed, active, layout = set(), None, []
    for row in records(launch_path, "xvram.cuda_launch_probe", cap=TRACE_CAP, versions=(1, 2),
                       max_records=FIELD_CAP+3*CALL_CAP+2):
        if row["kind"] == "begin":
            call = row["call_id"]
            if call not in calls or call in observed or not row["metadata"]:
                raise ValueError("arguments_launch_identity")
            observed.add(call)
            active, layout = calls[call], []
            kid = active["begin"]["catalog_id"]
            if kid:
                if row["kernel_name"] != kernels[kid-1]["symbol"]:
                    raise ValueError("arguments_launch_catalog_name")
            elif row["kernel_name"] in names:
                raise ValueError("arguments_launch_unknown_name")
        elif row["kind"] == "parameter":
            layout.append((row["offset_bytes"], row["size_bytes"]))
        elif row["kind"] == "end":
            if active is None or row["result"] != active["result"]:
                raise ValueError("arguments_launch_return")
            kid, status = active["begin"]["catalog_id"], active["status"]
            if 1 <= kid <= 35:
                expected = [(arg["offset_bytes"], arg["size_bytes"]) for arg in kernels[kid-1]["arguments"]]
                if (layout == expected) == (status == "unsupported_abi"):
                    raise ValueError("arguments_launch_catalog_abi")
            active = None
    if observed != set(calls) or digest(launch_path, TRACE_CAP) != validated["sha256"]:
        raise ValueError("arguments_launch_changed_or_missing")
    return validated["sha256"]


def analyze(path, launch_path=None):
    before = digest(path, BYTE_CAP)
    counts, calls = analyze_rows(records(path, RECORD_TYPE, cap=BYTE_CAP, max_records=FIELD_CAP+3*CALL_CAP+2))
    launch_sha = bind_launches(calls, launch_path) if launch_path is not None else None
    if digest(path, BYTE_CAP) != before:
        raise ValueError("arguments_input_changed")
    return {"schema_version": 1, "report_type": "xvram.cuda_kernel_argument_observation",
            "trace_sha256": before, "launch_trace_sha256": launch_sha, "catalog_sha256": CATALOG_SHA256,
            "observed_host_calls_joined": launch_sha is not None,
            "counts": counts, "catalog_ids": sorted({row["begin"]["catalog_id"] for row in calls.values()}),
            "memory_bounds_proven": False, "native_launches_cubin_bound": False,
            "stream_event_order_proven": False, "trace_complete": False}
