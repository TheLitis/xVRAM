"""Offline cubin parameter evidence; never a runtime binding or GO authority.

Only ordinary ELF64 symbol tables are decoded locally. CUDA-specific parameter
attributes are taken from the pinned NVIDIA cuobjdump text, not guessed private
ELF attribute numbers. Neither tool output nor static names establish the module
or argument values of a captured launch.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import struct
import sys
import tempfile

from . import compat_audit_sources as source_index
from .compat_audit_binary_contract import (
    BACKEND_SHA256, TOOL_SHA256, TOOL_VERSION, LIMITATIONS as CONTRACT_LIMITATIONS,
    validate_report,
)
from .compat_audit_sources import _SAFE_NAME, _HEX_ADDRESS

MAX_CUBIN_BYTES = 64 * 1024 * 1024
MAX_SECTIONS = 65535
MAX_SYMBOLS = 100000
MAX_PARAMETERS = 128
MAX_PARAMETER_BYTES = 65536
MAX_DUMP_LINE = 65536
MAX_CUBINS = 1024
MAX_EXTRACTED_BYTES = 512 * 1024 * 1024
MAX_EVIDENCE_BYTES = 4 * 1024 * 1024
LIMITATIONS = list(CONTRACT_LIMITATIONS)


class BinaryAuditError(ValueError):
    """Fixed diagnostic code without binary contents, paths or native values."""


def _span(data, offset, size):
    if offset < 0 or size < 0 or offset > len(data) or size > len(data) - offset:
        raise BinaryAuditError("elf_span_out_of_bounds")
    return memoryview(data)[offset:offset + size]


def _cstring(table, offset):
    if not 0 <= offset < len(table):
        raise BinaryAuditError("elf_string_offset_out_of_bounds")
    end = table.find(b"\0", offset, min(len(table), offset + 4097))
    if end < 0:
        raise BinaryAuditError("elf_string_unterminated_or_too_long")
    try:
        return table[offset:end].decode("ascii")
    except UnicodeError as error:
        raise BinaryAuditError("elf_string_not_ascii") from error


def elf_functions(data: bytes) -> dict:
    """Read ELF64/CUDA symbols with exact table bounds; reject unsupported forms."""
    if not isinstance(data, bytes) or not 64 <= len(data) <= MAX_CUBIN_BYTES:
        raise BinaryAuditError("cubin_size_limit")
    if data[:7] != b"\x7fELF\x02\x01\x01":
        raise BinaryAuditError("unsupported_elf_encoding")
    (kind, machine, version, _entry, _phoff, shoff, flags, ehsize,
     _phentsize, _phnum, shentsize, shnum, shstrndx) = struct.unpack_from("<HHIQQQIHHHHHH", data, 16)
    if kind != 2 or machine != 190 or version != 1 or ehsize != 64:
        raise BinaryAuditError("unsupported_cuda_elf_header")
    if shentsize != 64 or not 0 < shnum <= MAX_SECTIONS or not 0 < shstrndx < shnum:
        raise BinaryAuditError("unsupported_elf_section_table")
    table = _span(data, shoff, shnum * shentsize)
    sections = [struct.unpack_from("<IIQQQQIIQQ", table, index * 64) for index in range(shnum)]
    names_section = sections[shstrndx]
    if names_section[1] != 3:
        raise BinaryAuditError("elf_section_names_not_strings")
    names = bytes(_span(data, names_section[4], names_section[5]))
    section_names = [_cstring(names, section[0]) for section in sections]
    if len(set(section_names[1:])) != len(section_names[1:]):
        raise BinaryAuditError("duplicate_elf_section_name")
    for section in sections:
        # SHT_NOBITS has a virtual size but no corresponding bytes in the file.
        if section[1] != 8:
            _span(data, section[4], section[5])
    symtabs = [section for section in sections if section[1] == 2]
    if len(symtabs) != 1:
        raise BinaryAuditError("elf_symbol_table_not_unique")
    symtab = symtabs[0]
    if symtab[9] != 24 or symtab[5] % 24 or symtab[5] // 24 > MAX_SYMBOLS:
        raise BinaryAuditError("invalid_elf_symbol_table")
    if not 0 < symtab[6] < shnum or sections[symtab[6]][1] != 3:
        raise BinaryAuditError("invalid_elf_symbol_strings")
    strtab = sections[symtab[6]]
    strings = bytes(_span(data, strtab[4], strtab[5]))
    functions = {}
    for index in range(symtab[5] // 24):
        name_offset, info, _other, section_index, value, size = struct.unpack_from(
            "<IBBHQQ", data, symtab[4] + index * 24)
        if info & 15 != 2 or section_index == 0:
            continue
        if section_index >= shnum:
            raise BinaryAuditError("unsupported_function_section_index")
        name = _cstring(strings, name_offset)
        if section_names[section_index] != ".text." + name:
            continue  # Not an independently named CUDA function section.
        if name in functions:
            raise BinaryAuditError("duplicate_elf_function")
        section = sections[section_index]
        if (section[1] != 1 or not section[2] & 4 or value < section[3]
                or value - section[3] > section[5] or size > section[5] - (value - section[3])):
            raise BinaryAuditError("invalid_elf_function_span")
        if not _SAFE_NAME.fullmatch(name) or _HEX_ADDRESS.search(name):
            raise BinaryAuditError("invalid_elf_function")
        functions[name] = {"symbol_index": index, "section_index": section_index,
                           "code_bytes": size}
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data),
            "elf_flags": flags, "functions": functions}


class ParameterDump:
    """Streaming reader for selected .nv.info sections in cuobjdump 13.3 text.

    The tool's architecture/header and whole-process success are checked by the
    caller. A selected attribute may not be truncated, duplicated, conflicting or
    out of range. Pointee alignment/space annotations are deliberately NOT used
    to infer pointer types, access directions or struct members.
    """
    def __init__(self, names):
        if not isinstance(names, (list, tuple, set, frozenset)) or len(names) > 4096:
            raise BinaryAuditError("selected_name_limit")
        if any(not isinstance(name, str) or not _SAFE_NAME.fullmatch(name) or _HEX_ADDRESS.search(name) for name in names):
            raise BinaryAuditError("invalid_selected_name")
        self.names = frozenset(names)
        if len(self.names) != len(names):
            raise BinaryAuditError("duplicate_selected_name")
        self.layouts = {}
        self.current = None
        self.attribute = None
        self.parameters = {}
        self.bank_size = None
        self.seen_sections = set()

    def _finish_section(self):
        if self.current is None:
            return
        if self.attribute is not None:
            raise BinaryAuditError("truncated_parameter_attribute")
        if self.bank_size is None or not self.parameters:
            raise BinaryAuditError("missing_parameter_layout")
        ordered = [self.parameters[index] for index in sorted(self.parameters)]
        if sorted(self.parameters) != list(range(len(ordered))):
            raise BinaryAuditError("noncontiguous_parameter_ordinals")
        end = 0
        for parameter in ordered:
            if parameter["offset_bytes"] < end:
                raise BinaryAuditError("overlapping_parameter_ranges")
            end = parameter["offset_bytes"] + parameter["size_bytes"]
            if end > self.bank_size:
                raise BinaryAuditError("parameter_exceeds_bank")
        if end != self.bank_size:
            raise BinaryAuditError("parameter_bank_extent_mismatch")
        self.layouts[self.current] = {"parameter_bytes": self.bank_size, "parameters": ordered}
        self.current = None

    def line(self, raw: bytes):
        if len(raw) > MAX_DUMP_LINE:
            raise BinaryAuditError("dump_line_limit")
        try:
            line = raw.decode("ascii").strip()
        except UnicodeError as error:
            raise BinaryAuditError("non_ascii_tool_output") from error
        if line.startswith(".nv.info."):
            self._finish_section()
            name = line[len(".nv.info."):]
            self.current = name if name in self.names else None
            self.attribute, self.parameters, self.bank_size = None, {}, None
            if self.current is not None:
                if name in self.seen_sections:
                    raise BinaryAuditError("duplicate_parameter_section")
                self.seen_sections.add(name)
            return
        if line.startswith(".") and not line.startswith(".nv.info."):
            self._finish_section()
            return
        if self.current is None:
            return
        if line.startswith("Attribute:"):
            if self.attribute is not None:
                raise BinaryAuditError("truncated_parameter_attribute")
            value = line.partition(":")[2].strip()
            if value in ("EIATTR_KPARAM_INFO", "EIATTR_CBANK_PARAM_SIZE"):
                self.attribute = value
            return
        if self.attribute is None or not line.startswith("Value:"):
            return
        value = line.partition(":")[2].strip()
        if self.attribute == "EIATTR_CBANK_PARAM_SIZE":
            if self.bank_size is not None or not re.fullmatch(r"0x[0-9a-fA-F]{1,8}", value):
                raise BinaryAuditError("invalid_parameter_bank_size")
            self.bank_size = int(value, 16)
            if not 0 < self.bank_size <= MAX_PARAMETER_BYTES:
                raise BinaryAuditError("parameter_bank_size_limit")
        else:
            match = re.fullmatch(
                r"Index\s*:\s*0x([0-9a-fA-F]{1,8})\s+Ordinal\s*:\s*0x([0-9a-fA-F]{1,8})\s+"
                r"Offset\s*:\s*0x([0-9a-fA-F]{1,8})\s+Size\s*:\s*0x([0-9a-fA-F]{1,8})", value)
            if not match:
                raise BinaryAuditError("invalid_parameter_attribute")
            index, ordinal, offset, size = [int(part, 16) for part in match.groups()]
            if index != 0 or ordinal >= MAX_PARAMETERS or not 0 < size <= MAX_PARAMETER_BYTES or offset > MAX_PARAMETER_BYTES - size:
                raise BinaryAuditError("parameter_attribute_limit")
            if ordinal in self.parameters:
                raise BinaryAuditError("duplicate_parameter_ordinal")
            self.parameters[ordinal] = {"ordinal": ordinal, "offset_bytes": offset, "size_bytes": size}
        self.attribute = None

    def finish(self):
        self._finish_section()
        if set(self.layouts) != self.names:
            raise BinaryAuditError("selected_parameter_section_missing")
        return self.layouts


def _file_bytes(path, limit):
    try:
        path = Path(path)
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise BinaryAuditError("input_link_not_supported")
        # In particular, a malformed worker result must not block its controller
        # while opening a FIFO after the contained process tree has been reaped.
        if not stat.S_ISREG(path.stat().st_mode):
            raise BinaryAuditError("input_not_regular")
        with path.open("rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise BinaryAuditError("input_not_regular")
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise BinaryAuditError("input_size_limit")
        return data
    except OSError as error:
        raise BinaryAuditError("input_unreadable") from error


def _file_hash(path, limit):
    return hashlib.sha256(_file_bytes(path, limit)).hexdigest()


def _observations(paths):
    if not 1 <= len(paths) <= 16:
        raise BinaryAuditError("report_count_limit")
    reports, counts, seen = [], Counter(), set()
    for path in paths:
        digest, values, total = source_index.read_report(path)
        if digest in seen:
            raise BinaryAuditError("duplicate_report")
        seen.add(digest)
        reports.append({"sha256": digest, "kernel_activities": total})
        for name, count in values.items():
            counts[name] = source_index._add(counts[name], count)
        if len(counts) > 4096:
            raise BinaryAuditError("observed_name_limit")
    if not counts:
        raise BinaryAuditError("no_observed_kernel_names")
    return sorted(reports, key=lambda item: item["sha256"]), counts


def parse_elf_list(raw):
    if len(raw) > 1024 * 1024:
        raise BinaryAuditError("elf_list_limit")
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeError as error:
        raise BinaryAuditError("elf_list_encoding") from error
    selected, names, indices = [], set(), set()
    for line in lines:
        if not line.strip():
            continue
        match = re.fullmatch(r"ELF file\s+([1-9][0-9]{0,5}): (ggml-cuda\.([1-9][0-9]{0,5})\.sm_([0-9]{2,3}[af]?)\.cubin)", line.strip())
        if not match or match[1] != match[3]:
            raise BinaryAuditError("invalid_elf_list_entry")
        index, name = int(match[1]), match[2]
        if name in names or index in indices:
            raise BinaryAuditError("duplicate_elf_list_entry")
        names.add(name)
        indices.add(index)
        if match[4] == "86":
            selected.append((index, name))
    if not 1 <= len(selected) <= MAX_CUBINS:
        raise BinaryAuditError("sm86_cubin_count_limit")
    return selected


def _empty_report():
    return {
        "schema_version": 1, "report_type": "xvram.cuda_compat_binary_evidence",
        "architecture": "sm_86", "version": "0.1.0-dev",
        "provenance": {"upstream_commit": source_index.COMMIT,
                       "backend_name": "ggml-cuda.dll", "backend_sha256": None,
                       "tool_name": "cuobjdump", "tool_sha256": None,
                       "tool_version": None, "reports": []},
        "modules": [], "kernels": [],
        "coverage": {"sm86_cubins": 0, "candidate_modules": 0, "observed_types": 0,
                     "static_layout_types": 0, "missing_types": 0, "ambiguous_types": 0,
                     "runtime_bound_types": 0, "complete_profile_coverage": False},
        "tools": {"invocations": 0, "stdout_bytes": 0, "all_direct_children_reaped": False,
                  "all_pipes_drained": False},
        "decision": {"verdict": "NO-GO", "execution_ready": False, "oversubscription_proof": False},
        "outcome": {"status": "failed", "exit_code": 27},
        "cleanup": {"worker_reaped": False, "process_tree_drained": False, "scratch_removed": False},
        "diagnostics": [], "limitations": LIMITATIONS.copy(),
    }


def inspect_worker(config):
    """Called ONLY under run_process containment; no Driver/runtime API loads."""
    from .compat_audit_tools import run_tool
    tool, backend, scratch = (Path(config[key]) for key in ("tool", "backend", "scratch"))
    report = _empty_report()
    if _file_hash(tool, 64 * 1024 * 1024) != TOOL_SHA256:
        raise BinaryAuditError("unpinned_cuobjdump")
    if backend.name != "ggml-cuda.dll" or _file_hash(backend, 256 * 1024 * 1024) != BACKEND_SHA256:
        raise BinaryAuditError("unpinned_backend")
    reports, counts = _observations(config["reports"])
    report["provenance"].update(backend_sha256=BACKEND_SHA256, tool_sha256=TOOL_SHA256,
                                tool_version=TOOL_VERSION, reports=reports)
    stats = report["tools"]

    def invoke(arguments, **kwargs):
        result = run_tool([str(tool), *arguments], cwd=scratch, timeout_seconds=30, **kwargs)
        stats["invocations"] += 1
        stats["stdout_bytes"] += result.stdout_bytes
        stats["all_direct_children_reaped"] = result.direct_child_reaped
        stats["all_pipes_drained"] = result.pipes_drained
        # Diagnostics are never copied into the report. Even a successful exit
        # with a warning invalidates this deliberately fixed inspection profile.
        if result.stderr.strip():
            raise BinaryAuditError("cuobjdump_diagnostic_output")
        return result

    version = invoke(["--version"], stdout_limit=8192).stdout
    if b"release 13.3, V13.3.73" not in version:
        raise BinaryAuditError("cuobjdump_version_mismatch")
    listing = invoke(["--list-elf", str(backend)], stdout_limit=1024 * 1024).stdout
    selected = parse_elf_list(listing)
    extract = scratch / "cubins"
    extract.mkdir()
    # Exactly one fresh output directory, owned by the outer controller. NVIDIA
    # extraction writes binary artifacts, never source/application replacements.
    result = run_tool([str(tool), "--extract-elf", "sm_86", str(backend)], cwd=extract,
                      timeout_seconds=30, stdout_limit=1024 * 1024)
    stats["invocations"] += 1
    stats["stdout_bytes"] += result.stdout_bytes
    if result.stderr.strip():
        raise BinaryAuditError("cuobjdump_diagnostic_output")
    if {path.name for path in extract.iterdir()} != {name for _, name in selected}:
        raise BinaryAuditError("extracted_file_set_mismatch")
    matches = {name: [] for name in counts}
    total_size = 0
    modules = []
    for index, filename in selected:
        path = extract / filename
        data = _file_bytes(path, MAX_CUBIN_BYTES)
        total_size += len(data)
        if total_size > MAX_EXTRACTED_BYTES:
            raise BinaryAuditError("extracted_size_limit")
        elf = elf_functions(data)
        wanted = sorted(set(counts).intersection(elf["functions"]))
        layouts = {}
        if wanted:
            parser = ParameterDump(wanted)
            header_count = 0
            dump_hash = hashlib.sha256()

            def consume(line):
                nonlocal header_count
                dump_hash.update(line)
                if line.strip().startswith(b"64-bit ELF:"):
                    match = re.fullmatch(
                        rb"64-bit ELF: type=ET_EXEC, ABI=([0-9]+), sm=86, toolkit=13\.3, flags=0x([0-9a-fA-F]+)", line.strip())
                    if not match or int(match[2], 16) != elf["elf_flags"]:
                        raise BinaryAuditError("dump_elf_identity_mismatch")
                    header_count += 1
                parser.line(line)

            invoke(["--dump-elf", str(path)], stdout_limit=128 * 1024 * 1024,
                   consume_stdout_line=consume, line_limit=MAX_DUMP_LINE)
            if header_count != 1:
                raise BinaryAuditError("dump_elf_header_not_unique")
            layouts = parser.finish()
        if _file_hash(path, MAX_CUBIN_BYTES) != elf["sha256"]:
            raise BinaryAuditError("cubin_changed_during_analysis")
        module = {"module_index": index, "sha256": elf["sha256"], "size_bytes": len(data),
                  "elf_flags": elf["elf_flags"], "symbol_functions": len(elf["functions"]),
                  "selected_functions": len(wanted),
                  "parameter_dump_sha256": dump_hash.hexdigest() if wanted else None}
        modules.append(module)
        for name in wanted:
            matches[name].append({"module_index": index, "cubin_sha256": elf["sha256"],
                                  **elf["functions"][name], **layouts[name]})
    for name, candidates in sorted(matches.items()):
        report["kernels"].append({"name": name, "activities": counts[name],
                                  "status": "static_layout_only" if len(candidates) == 1 else "ambiguous" if candidates else "missing",
                                  "candidates": candidates, "runtime_binding_proven": False,
                                  "memory_bounds_proven": False})
    statuses = Counter(item["status"] for item in report["kernels"])
    report["modules"] = modules
    report["coverage"].update(sm86_cubins=len(modules), candidate_modules=sum(bool(item["selected_functions"]) for item in modules),
                               observed_types=len(counts), static_layout_types=statuses["static_layout_only"],
                               missing_types=statuses["missing"], ambiguous_types=statuses["ambiguous"])
    after_reports, after_counts = _observations(config["reports"])
    if after_reports != reports or after_counts != counts or _file_hash(tool, 64 * 1024 * 1024) != TOOL_SHA256 or _file_hash(backend, 256 * 1024 * 1024) != BACKEND_SHA256:
        raise BinaryAuditError("input_changed_during_analysis")
    report["outcome"].update(status="completed", exit_code=0)
    return report


def _worker(config_path):
    from .compat_audit_tools import AuditToolError
    report = _empty_report()
    try:
        config = source_index._json_bytes(_file_bytes(config_path, 1024 * 1024))
        report = inspect_worker(config)
    except (BinaryAuditError, source_index.SourceIndexError, AuditToolError) as error:
        code = 26 if isinstance(error, AuditToolError) and error.code == "tool_timeout" else 27 if isinstance(error, AuditToolError) else 23
        report["outcome"].update(status="timeout" if code == 26 else "failed" if code == 27 else "rejected", exit_code=code)
        report["diagnostics"] = [str(error)]
    except Exception:
        report["diagnostics"] = ["binary_worker_failed"]
    try:
        validate_report(report, allow_pending_cleanup=True)
    except ValueError:
        report = _empty_report()
        report["diagnostics"] = ["invalid_binary_worker_report"]
    with Path(config_path).with_name("result.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=True, allow_nan=False)
    return report["outcome"]["exit_code"]


def main(argv=None):
    from .compat_audit import _Parser, clean_capture_environment
    from .compat_audit_capture import run_process
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) == 2 and arguments[0] == "--worker":
        return _worker(Path(arguments[1]))
    parser = _Parser(description=__doc__)
    parser.add_argument("--backend", type=Path, required=True, help="pinned official ggml-cuda.dll")
    parser.add_argument("--cuobjdump", type=Path, required=True, help="pinned Windows CUDA 13.3.73 tool")
    parser.add_argument("--report", action="append", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True, help="external directory for fresh temporary extraction")
    parser.add_argument("--timeout-seconds", type=int, choices=range(1, 901), default=300,
                        metavar="N", help="overall worker deadline: 1..900 seconds (default: 300)")
    parser.add_argument("--json", required=True, help="new report path or -; never overwrite evidence")
    args = parser.parse_args(arguments)
    report = _empty_report()
    temporary_path = None
    safe_to_remove = True
    worker_cleanup = {"worker_reaped": False, "process_tree_drained": False}
    try:
        if args.json != "-" and Path(args.json).exists():
            raise BinaryAuditError("output_exists")
        if not 1 <= len(args.report) <= 16:
            raise BinaryAuditError("report_count_limit")
        work_root = args.work_root.resolve()
        repository = Path(__file__).resolve().parents[2]
        if (repository / "CMakeLists.txt").is_file() and (work_root == repository or repository in work_root.parents):
            raise BinaryAuditError("extraction_root_inside_source_repository")
        work_root.mkdir(parents=True, exist_ok=True)
        temporary_path = Path(tempfile.mkdtemp(prefix="xvram-binary-audit-", dir=work_root))
        if temporary_path.resolve().parent != work_root or not temporary_path.name.startswith("xvram-binary-audit-"):
            safe_to_remove = False
            raise BinaryAuditError("unsafe_scratch_directory")
        config_path = temporary_path / "plan.json"
        # Preserve link provenance; byte reads, pin checks and potentially slow
        # input I/O all occur under the worker deadline, before invoking a tool.
        config = {"backend": str(args.backend.absolute()), "tool": str(args.cuobjdump.absolute()),
                  "scratch": str(temporary_path), "reports": [str(path.absolute()) for path in args.report]}
        with config_path.open("x", encoding="utf-8") as stream:
            json.dump(config, stream)
        safe_to_remove = False
        capture = run_process([sys.executable, "-m", "xvram.compat_audit_binary", "--worker", str(config_path)],
                              output_dir=temporary_path / "control", cwd=temporary_path,
                              timeout_seconds=args.timeout_seconds, environment=clean_capture_environment())
        worker_cleanup = {"worker_reaped": capture["controller_reaped"],
                          "process_tree_drained": capture["process_tree_drained"]}
        safe_to_remove = all(worker_cleanup.values())
        if capture["timed_out"]:
            report["outcome"].update(status="timeout", exit_code=26)
            report["diagnostics"] = ["binary_inspection_timeout"]
        else:
            try:
                result = source_index._json_bytes(_file_bytes(temporary_path / "result.json", MAX_EVIDENCE_BYTES))
                validate_report(result, allow_pending_cleanup=True)
                if result["outcome"]["exit_code"] != capture["exit_code"]:
                    raise ValueError("worker_exit_mismatch")
            except (ValueError, source_index.SourceIndexError):
                report["diagnostics"] = ["invalid_binary_worker_result"]
            else:
                report = result
        if (capture["errors"] and not capture["timed_out"]) or capture.get("output_truncated", False) or not safe_to_remove:
            report["outcome"].update(status="failed", exit_code=27)
            report["diagnostics"].append("binary_worker_containment_failed")
    except (BinaryAuditError, source_index.SourceIndexError) as error:
        report["outcome"].update(status="rejected", exit_code=23)
        report["diagnostics"].append(str(error))
    except (OSError, ValueError):
        report["outcome"].update(status="failed", exit_code=27)
        report["diagnostics"].append("binary_inspection_io_or_cleanup_failed")
    finally:
        report["cleanup"].update(worker_cleanup)
        report["cleanup"]["scratch_removed"] = False
        # Do not remove files from a worker whose tree is still unconfirmed.
        # Retain that exact scratch directory as a diagnostic quarantine boundary.
        if temporary_path is not None and safe_to_remove:
            try:
                _remove_scratch(temporary_path, work_root)
                report["cleanup"]["scratch_removed"] = True
            except (OSError, BinaryAuditError):
                report["outcome"].update(status="failed", exit_code=27)
                report["diagnostics"].append("binary_scratch_cleanup_failed")
    try:
        validate_report(report)
    except ValueError:
        cleanup = report["cleanup"].copy()
        report = _empty_report()
        report["cleanup"].update(cleanup)
        report["diagnostics"] = ["invalid_binary_final_report"]
        # Only controller-produced cleanup is retained; an inconsistent worker
        # result cannot preserve evidence or promote a failed analysis to success.
        if not report["cleanup"]["worker_reaped"]:
            report["cleanup"]["process_tree_drained"] = False
        validate_report(report)
    try:
        data = json.dumps(report, ensure_ascii=True, allow_nan=False, indent=2) + "\n"
        if args.json == "-":
            sys.stdout.write(data)
        else:
            with Path(args.json).open("x", encoding="utf-8") as stream:
                stream.write(data)
    except OSError:
        print("binary_evidence_output_failed", file=sys.stderr)
        return 74
    return report["outcome"]["exit_code"]


def _remove_scratch(path, work_root):
    """Delete only the exact fresh extraction directory after confirmed drain."""
    if (path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())
            or path.resolve().parent != work_root.resolve()
            or not path.name.startswith("xvram-binary-audit-")):
        raise BinaryAuditError("unsafe_scratch_cleanup_target")
    shutil.rmtree(path)


if __name__ == "__main__":
    raise SystemExit(main())
