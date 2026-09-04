from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import jsonschema

from xvram.torch_report import (
    EXIT_COMPLETED,
    EXIT_CORRUPTION,
    EXIT_INTERNAL,
    EXIT_OUTPUT,
    EXIT_PREREQUISITE,
    EXIT_PRESSURE,
    EXIT_RUNTIME,
    EXIT_TIMEOUT,
    empty_report,
    finalize_proof,
    success_semantics,
    validate_report_envelope,
    validate_trace_record,
)


def configuration() -> dict[str, object]:
    return {
        "device": 0,
        "model": "llama2-like",
        "model_ratio": 2.28,
        "layers": 47,
        "batch": 1,
        "sequence": 32,
        "hidden": 4096,
        "intermediate": 11008,
        "heads": 32,
        "dtype": "float16",
        "policy": "clock",
        "cache_target_bytes": 7 * 1024**3,
        "chunk_size_bytes": 64 * 1024**2,
        "device_headroom_bytes": 512 * 1024**2,
        "scratch_cap_bytes": 512 * 1024**2,
        "prefetch_distance": 2,
        "sdpa_backend": "math",
        "seed": "0x585652414d503035",
        "compression": "capacity",
        "compression_codec": "lz4",
        "state_pattern": "structured",
        "host_store_cap_bytes": 24 * 1024**3,
        "host_headroom_bytes": 4 * 1024**3,
        "compression_scratch_cap_bytes": 256 * 1024**2,
        "codec_slots": 2,
        "codec_workers": 2,
    }


def completed() -> dict[str, object]:
    report = empty_report(
        configuration(),
        exit_code=EXIT_COMPLETED,
        status="completed",
        message="compressed inference completed",
    )
    gib = 1024**3
    report["device"]["total_vram_bytes"] = 8 * gib
    report["model"].update(
        logical_bytes=18 * gib,
        parameter_bytes=17 * gib,
        activation_bytes=1 * gib,
    )
    report["graph"].update(
        hash="a" * 64,
        backend_hash="a" * 64,
        node_count=2,
        region_count=2,
    )
    report["execution"].update(
        leases_acquired=2,
        leases_sealed=2,
        leases_retired=2,
        events_recorded=5,
        events_retired=5,
        live_views_peak=2,
        regions_completed=2,
        region_timings_ms=[1.0, 1.1],
        trace_complete=True,
    )
    report["cache"].update(
        maps=4,
        set_access=4,
        unmaps=4,
        misses=4,
        h2d_bytes=4096,
        evictions=2,
        frame_reuses=2,
    )
    report["verification"].update(
        reference_digest="b" * 64,
        output_digest="b" * 64,
    )
    report["compression"].update(
        logical_bytes=18 * gib,
        stored_bytes=7 * gib,
        stored_peak_bytes=8 * gib,
        raw_bytes=1 * gib,
        compressed_bytes=6 * gib,
        host_budget_bytes=8 * gib,
        host_budget_peak_bytes=9 * gib,
        conversion_scratch_peak_bytes=128 * 1024**2,
        logical_h2d_bytes=4096,
        pcie_h2d_bytes=2048,
        pcie_h2d_payload_bytes=2048,
        pcie_h2d_metadata_bytes=0,
        logical_d2h_bytes=0,
        pcie_d2h_bytes=0,
        pcie_d2h_payload_bytes=0,
        pcie_d2h_metadata_bytes=0,
        rejected_candidate_logical_d2h_bytes=0,
        raw_path_decisions=1,
        cpu_lz4_gpu_decode_decisions=3,
        compression_attempts=3,
        compression_commits=3,
        decompression_attempts=3,
        decompression_commits=3,
        generations_created=4,
        generations_committed=4,
        codec_events_recorded=3,
        codec_events_retired=3,
        workspace_peak_bytes=128 * 1024**2,
    )
    report["proof"]["stable_addresses"] = True
    for key in report["cleanup"]:
        report["cleanup"][key] = True
    return finalize_proof(report)


def check_acceptance_device_budget() -> None:
    validator_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "validate-phase5-pytorch-acceptance.py"
    )
    spec = importlib.util.spec_from_file_location("phase5_pytorch_acceptance", validator_path)
    assert spec is not None and spec.loader is not None
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    mib = 1024**2
    report = {
        "device": {"total_vram_bytes": 8192 * mib},
        "configuration": {
            "device_headroom_bytes": 512 * mib,
            "scratch_cap_bytes": 512 * mib,
        },
        "cache": {"target_bytes": 7680 * mib, "resident_peak_bytes": 6912 * mib},
        "compression": {"workspace_peak_bytes": 128 * mib, "slot_peak_bytes": 128 * mib},
    }
    # This exactly fills the aggregate target. Adding its reserved components
    # again would incorrectly reject a valid near-full-budget hardware run.
    validator._validate_device_budget(report)
    shrunk = copy.deepcopy(report)
    shrunk["cache"]["target_bytes"] = 6144 * mib
    # The final live target is below the already-retired historical residency
    # peak. That is a valid budget shrink, not evidence of an earlier overflow.
    validator._validate_device_budget(shrunk)
    mutations = (
        ("cache", "target_bytes", 7680 * mib + 1),
        ("cache", "resident_peak_bytes", 6912 * mib + 1),
        ("configuration", "scratch_cap_bytes", 512 * mib + 1),
        ("compression", "workspace_peak_bytes", 128 * mib + 1),
        ("compression", "slot_peak_bytes", 128 * mib + 1),
    )
    for section, name, value in mutations:
        invalid = copy.deepcopy(report)
        invalid[section][name] = value
        try:
            validator._validate_device_budget(invalid)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"hardware gate accepted overbudget {section}.{name}")


def main() -> int:
    check_acceptance_device_budget()
    report_schema = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    trace_schema = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(report_schema)
    jsonschema.Draft202012Validator.check_schema(trace_schema)
    validate_report = jsonschema.Draft202012Validator(report_schema).validate
    validate_trace = jsonschema.Draft202012Validator(trace_schema).validate

    success = completed()
    validate_report(success)
    validate_report_envelope(success)
    ok, errors = success_semantics(success)
    if not ok:
        raise AssertionError(errors)

    for exit_code, status in (
        (EXIT_PREREQUISITE, "skipped"),
        (EXIT_CORRUPTION, "corruption"),
        (EXIT_TIMEOUT, "timeout"),
        (EXIT_RUNTIME, "failed"),
        (EXIT_INTERNAL, "failed"),
        (EXIT_OUTPUT, "failed"),
    ):
        validate_report(
            empty_report(
                configuration(),
                exit_code=exit_code,
                status=status,
                message="fixture",
            )
        )

    for message in ("host out of memory", "device out of memory"):
        validate_report(
            empty_report(
                configuration(),
                exit_code=EXIT_PRESSURE,
                status="oom",
                message=message,
            )
        )

    cleanup_failure = empty_report(
        configuration(),
        exit_code=EXIT_RUNTIME,
        status="failed",
        message="cleanup failure",
    )
    cleanup_failure["cleanup"]["worker_reaped"] = True
    cleanup_failure["cleanup"]["mappings_unmapped"] = False
    cleanup_failure["cleanup"]["complete"] = False
    validate_report(cleanup_failure)

    trace = {
        "schema_version": 2,
        "record_type": "xvram.pytorch_trace",
        "sequence": 1,
        "monotonic_ns": 123,
        "kind": "codec",
        "region_id": 1,
        "allocation_id": 7,
        "operation": "decode_retired",
        "bytes": 4096,
        "reason": "capacity",
        "chunk_index": 2,
        "source_generation": 3,
        "target_generation": 4,
        "slot_generation": 5,
        "representation": "lz4_blocks",
        "codec_path": "cpu_lz4_gpu_decode",
    }
    validate_trace(trace)
    validate_trace_record(trace)

    success["diagnostics"].append(
        {"stage": "codec", "code": "leak", "message": "stream_handle: 0x1234"}
    )
    try:
        validate_report_envelope(success)
    except ValueError:
        pass
    else:
        raise AssertionError("v2 report accepted a serialized stream handle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
