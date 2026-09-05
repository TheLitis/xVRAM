"""Offline observation refinement; never an interoperability GO authority."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import sys

from .compat_audit import _Parser
from .compat_audit_analysis import AuditInputError, MAX_TRACE_BYTES
from .compat_audit_correlations import correlate_trace
from .compat_audit_routing import summarize_routing

LIMITATIONS = ["host_nesting_is_not_device_ordering", "resolver_results_do_not_prove_invocation_routing",
              "kernel_module_argument_and_tensor_bounds_unresolved", "terminal_and_activity_loss_preclude_complete_coverage",
              "native_cpu_offload_is_not_xvram_oversubscription"]


def _digest(path):
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise AuditInputError("trace_not_regular_file")
            total = 0
            while block := stream.read(4 * 1024 * 1024):
                total += len(block)
                if total > MAX_TRACE_BYTES:
                    raise AuditInputError("trace_limit")
                digest.update(block)
    except OSError as error:
        raise AuditInputError("trace_unreadable") from error
    return digest.hexdigest()


def observations(trace_paths):
    paths = list(trace_paths)
    if not 1 <= len(paths) <= 16 or len({Path(path).resolve() for path in paths}) != len(paths):
        raise AuditInputError("invalid_trace_list")
    traces = []
    for index, path in enumerate(paths, 1):
        before = _digest(path)
        correlations = correlate_trace(path)
        routing = summarize_routing(path)
        unchanged = before == _digest(path)
        if not unchanged:
            correlations["launch_records_reconciled"] = False
            correlations["trace_complete"] = False
            correlations["trace_issues"].append({"code": "trace_changed_during_analysis", "count": 1})
            routing["evidence"] = "unresolved"
            routing["issues"].append({"code": "trace_changed_during_analysis", "count": 1})
        traces.append({"trace_id": index, "sha256": before, "input_unchanged": unchanged,
                       "correlations": correlations, "routing": routing})
    return {"schema_version": 1, "report_type": "xvram.cuda_compat_audit_observations",
            "traces": traces, "decision": {"verdict": "NO-GO", "execution_ready": False,
                                            "oversubscription_proof": False}, "limitations": LIMITATIONS}


def main(argv=None):
    parser = _Parser(description=__doc__)
    parser.add_argument("--trace", action="append", type=Path, required=True)
    parser.add_argument("--json", default="-", help="new output path, or - for stdout; existing files are not overwritten")
    args = parser.parse_args(argv)
    try:
        result = observations(args.trace)
        data = json.dumps(result, ensure_ascii=True, allow_nan=False, indent=2) + "\n"
        if args.json == "-":
            sys.stdout.write(data)
        else:
            with Path(args.json).open("x", encoding="utf-8") as stream:
                stream.write(data)
        return 0
    except AuditInputError as error:
        print(str(error), file=sys.stderr)
        return 64
    except (OSError, ValueError):
        print("observation_output_failure", file=sys.stderr)
        return 74


if __name__ == "__main__":
    raise SystemExit(main())
