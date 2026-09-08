"""Bounded native identity witnesses. Observations never imply execution GO."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from .compat_launch_probe import _object

CAP = 64 * 1024 * 1024
COMMON = {"schema_version", "record_type", "sequence", "kind"}


def _uint(value, minimum=0, maximum=2**64-1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("witness_integer")
    return value


def _fields(row, names):
    if set(row) != COMMON | set(names.split()):
        raise ValueError("witness_fields")


def _false(value):
    if value is not False:
        raise ValueError("witness_unsupported_proof")


def records(path, record_type, *, cap=CAP, versions=(1,)):
    total = 0
    with Path(path).open("rb") as stream:
        for sequence, raw in enumerate(iter(lambda: stream.readline(65537), b""), 1):
            total += len(raw)
            if len(raw) > 65536 or not raw.endswith(b"\n") or total > cap or sequence > 1500000:
                raise ValueError("witness_framing")
            row = json.loads(raw, object_pairs_hook=_object,
                             parse_constant=lambda _: (_ for _ in ()).throw(ValueError("witness_nonfinite")))
            if (type(row) is not dict or row.get("record_type") != record_type
                    or type(row.get("schema_version")) is not int or row["schema_version"] not in versions
                    or _uint(row.get("sequence"), 1) != sequence):
                raise ValueError("witness_profile_or_sequence")
            yield row


def digest(path, cap=CAP):
    value = hashlib.sha256()
    total = 0
    with Path(path).open("rb") as stream:
        while data := stream.read(1024*1024):
            total += len(data)
            if total > cap:
                raise ValueError("witness_capacity")
            value.update(data)
    return value.hexdigest()


def identity(path, census_path, calls):
    libraries, apis = {}, {}
    counts = Counter(records=0, launches=0, library_loads=0, library_unloads=0,
                     library_errors=0, library_linked_launches=0, function_only_launches=0)
    summary = None
    for row in records(path, "xvram.cuda_identity_witness"):
        if summary is not None:
            raise ValueError("witness_after_summary")
        counts["records"] += 1
        kind = row["kind"]
        if counts["records"] == 1 and kind != "session":
            raise ValueError("witness_session_missing")
        if kind == "session":
            _fields(row, "terminal_complete")
            _false(row["terminal_complete"])
            if counts["records"] != 1:
                raise ValueError("witness_duplicate_session")
        elif kind == "library_api":
            _fields(row, "api_id symbol result library_id library_generation")
            api_id = _uint(row["api_id"], 1, 500000)
            symbol = row["symbol"]
            if api_id in apis or symbol not in ("cuLibraryLoadData", "cuLibraryLoadFromFile", "cuLibraryUnload"):
                raise ValueError("witness_library_api")
            result = _uint(row["result"], 0, 2**31-1)
            lib, generation = _uint(row["library_id"], maximum=100000), _uint(row["library_generation"], maximum=100000)
            apis[api_id] = (symbol, result)
            if result:
                counts["library_errors"] += 1
                if lib or generation:
                    raise ValueError("witness_failed_library_output")
                continue
            if not lib or not generation:
                raise ValueError("witness_library_zero")
            previous, live = libraries.get(lib, (0, False))
            if symbol == "cuLibraryUnload":
                if not live or generation != previous:
                    raise ValueError("witness_stale_unload")
                libraries[lib] = generation, False
                counts["library_unloads"] += 1
            else:
                if live or generation != previous + 1:
                    raise ValueError("witness_library_generation")
                libraries[lib] = generation, True
                counts["library_loads"] += 1
        elif kind == "launch_identity":
            _fields(row, "call_id library_id library_generation module_observation_id library_module_equal cubin_binding_proven")
            _false(row["cubin_binding_proven"])
            if _uint(row["call_id"], 1, 100000) != counts["launches"] + 1:
                raise ValueError("witness_launch_sequence")
            _uint(row["module_observation_id"], 1, 100000)
            lib, generation = _uint(row["library_id"], maximum=100000), _uint(row["library_generation"], maximum=100000)
            if lib:
                if libraries.get(lib) != (generation, True) or row["library_module_equal"] is not True:
                    raise ValueError("witness_unbound_library")
                counts["library_linked_launches"] += 1
            else:
                if generation or row["library_module_equal"] is not False:
                    raise ValueError("witness_function_only")
                counts["function_only_launches"] += 1
            counts["launches"] += 1
        elif kind == "summary":
            _fields(row, "errors terminal_complete")
            _false(row["terminal_complete"])
            if _uint(row["errors"]):
                raise ValueError("witness_native_error")
            summary = row
        else:
            raise ValueError("witness_kind")
    if not summary or not counts["launches"] or counts["launches"] != calls:
        raise ValueError("witness_incomplete")
    # The sidecar API id is the census's actual instance id, not a timestamp guess.
    seen = set()
    for row in records(census_path, "xvram.cuda_launch_census", cap=256*1024*1024, versions=(2,)):
        if row["kind"] != "api_exit" or row["domain"] != "driver":
            continue
        if row["symbol"] not in ("cuLibraryLoadData", "cuLibraryLoadFromFile", "cuLibraryUnload"):
            continue
        key = row["api_id"]
        if key in seen or apis.get(key) != (row["symbol"], row["result"]):
            raise ValueError("witness_census_library_mismatch")
        seen.add(key)
    if seen != set(apis):
        raise ValueError("witness_orphan_library_api")
    return dict(counts)


def make_report(probe, census, path, census_path):
    # Controller passes reports freshly recomputed from the child traces.
    report = dict(schema_version=1, report_type="xvram.cuda_identity_evidence",
                  version="0.1.0-dev", observation=None, diagnostics=[],
                  proof=dict(memory_bounds=False, cubin_binding=False, device_ordering=False,
                             trace_completeness=False), exit_code=27)
    if probe["capture"]["timed_out"]:
        report["exit_code"] = 26
    try:
        initial, census_initial = digest(path), digest(census_path, 256*1024*1024)
        counts = identity(path, census_path, probe["trace"]["counts"]["calls_returned"])
        if initial != digest(path) or census_initial != digest(census_path, 256*1024*1024):
            raise ValueError("witness_changed")
        report["observation"] = dict(sha256=initial, census_sha256=census_initial, counts=counts)
        if probe["exit_code"] == census["exit_code"] == 0:
            report["exit_code"] = 0
    except (ValueError, KeyError, TypeError, OSError, UnicodeError):
        report["diagnostics"].append("identity_witness_invalid_or_incomplete")
    return report
