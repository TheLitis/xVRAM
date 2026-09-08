from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from xvram import compat_audit_kernel_arguments as arguments


def fixture(kid=1, status="captured"):
    rows = []

    def row(kind, call=0, **values):
        item = dict(schema_version=1, record_type=arguments.RECORD_TYPE,
                    sequence=len(rows)+1, kind=kind, call=call, **values)
        rows.append(item)
        return item

    row("session", catalog_sha256=arguments.CATALOG_SHA256, byte_cap=arguments.BYTE_CAP,
        call_cap=arguments.CALL_CAP, field_cap=arguments.FIELD_CAP, memory_observer_enabled=True,
        memory_bounds_proven=False, source_semantics_bound_to_cubin=False)
    row("begin", 1, catalog_id=kid, grid_x=2, grid_y=3, grid_z=4,
        block_x=32, block_y=1, block_z=1, shared_bytes=128, device=0, memory_observer_healthy=True)
    fields = arguments.field_layout(arguments.catalog()[kid-1]) if kid and status == "captured" else []
    for ordinal, arg, field, typ in fields:
        item = row("field", 1, ordinal=ordinal, argument=arg, field=field, value_type=typ)
        if typ == "ptr":
            item.update(resolution="allocation_identity", memory_bounds_proven=False,
                        allocation_id=7, generation=3, offset_bytes=17, allocation_bytes=4096,
                        lookup_bytes=1, memory_revision=19, memory_sequence=23, alignment=16,
                        base_mod_alignment=0, mapped=True, readable=True, writable=True)
        elif typ == "f32":
            item["bits_u32"] = 0x3f800000
        else:
            item["value"] = True if typ == "bool" else -1 if typ.startswith("i") else 1
    row("capture_end", 1, status=status, fields=len(fields), unresolved_pointers=0, memory_bounds_proven=False)
    row("return", 1, result=0)
    row("summary", calls=1, returns=1, fields=len(fields), unsupported=int(status.startswith("unsupported_")),
        capture_faults=int(status == "capture_fault"), unresolved_pointers=0, errors=0,
        terminal_complete=False, memory_bounds_proven=False, source_semantics_bound_to_cubin=False)
    return rows


class KernelArgumentTests(unittest.TestCase):
    def test_every_reviewed_layout_captures_fields_without_proving_bounds(self):
        for kid in range(1, 36):
            with self.subTest(kid=kid):
                counts, calls = arguments.analyze_rows(fixture(kid))
                self.assertEqual(counts["calls"], counts["returns"])
                self.assertEqual(counts["unsupported"], 0)
                self.assertEqual(calls[1]["status"], "captured")
                self.assertGreater(counts["fields"], 0)

    def test_opaque_and_unsupported_are_explicit(self):
        for kid, status in [(36, "unsupported_opaque_library"), (39, "unsupported_opaque_library"),
                            (0, "unsupported_symbol"), (1, "unsupported_abi"),
                            (1, "unsupported_argument_bank"), (1, "capture_fault")]:
            with self.subTest(status=status):
                counts, calls = arguments.analyze_rows(fixture(kid, status))
                self.assertEqual(counts["fields"], 0)
                self.assertEqual(calls[1]["status"], status)

    def test_wrong_status_cannot_relabel_catalog(self):
        for kid, status in [(1, "unsupported_symbol"), (1, "unsupported_opaque_library"),
                            (36, "captured"), (0, "capture_fault")]:
            with self.subTest(status=status), self.assertRaises(ValueError):
                arguments.analyze_rows(fixture(kid, status))

    def test_extra_pointer_or_native_handle_field_rejected(self):
        for key in ("address", "pointer", "stream_handle", "native_handle"):
            rows = fixture()
            rows[2][key] = 0x7fedcba987654321
            with self.subTest(key=key), self.assertRaises(ValueError):
                arguments.analyze_rows(rows)

    def test_pointer_cannot_be_reclassified_as_scalar(self):
        rows = fixture()
        rows[2] = {key: value for key, value in rows[2].items()
                   if key in arguments.COMMON | {"ordinal", "argument", "field", "value_type"}}
        rows[2].update(value_type="u64", value=0x7fedcba987654321)
        with self.assertRaises(ValueError):
            arguments.analyze_rows(rows)

    def test_null_and_unresolved_do_not_fabricate_allocation(self):
        for resolution in ("null", "unresolved"):
            rows = fixture()
            rows[2] = {key: value for key, value in rows[2].items()
                       if key in arguments.COMMON | {"ordinal", "argument", "field", "value_type", "memory_bounds_proven"}}
            rows[2]["resolution"] = resolution
            rows[-3]["unresolved_pointers"] = rows[-1]["unresolved_pointers"] = int(resolution == "unresolved")
            counts, _ = arguments.analyze_rows(rows)
            self.assertEqual(counts["unresolved_pointers"], int(resolution == "unresolved"))

    def test_identity_range_overflow_and_generation(self):
        for key, value in (("generation", 0), ("allocation_id", True), ("offset_bytes", 4096),
                           ("allocation_bytes", 2**64), ("lookup_bytes", 64),
                           ("memory_sequence", 0), ("alignment", 8), ("base_mod_alignment", 16)):
            rows = fixture(); rows[2][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                arguments.analyze_rows(rows)

    def test_proof_escalation_rejected(self):
        for index, key in ((0, "memory_bounds_proven"), (0, "source_semantics_bound_to_cubin"),
                           (2, "memory_bounds_proven"), (-1, "terminal_complete"),
                           (-1, "memory_bounds_proven"), (-1, "source_semantics_bound_to_cubin")):
            rows = fixture(); rows[index][key] = True
            with self.subTest(key=key), self.assertRaises(ValueError):
                arguments.analyze_rows(rows)

    def test_missing_or_reordered_field_rejected(self):
        rows = fixture(); rows[2], rows[3] = rows[3], rows[2]
        for i, row in enumerate(rows, 1):
            row["sequence"] = i
        with self.assertRaises(ValueError):
            arguments.analyze_rows(rows)
        rows = fixture(); del rows[2]
        for i, row in enumerate(rows, 1):
            row["sequence"] = i
        with self.assertRaises(ValueError):
            arguments.analyze_rows(rows)

    def test_lifetime_and_summary_counts(self):
        for mutation in (lambda rows: rows[-1].update(calls=2),
                         lambda rows: rows[-1].update(fields=0),
                         lambda rows: rows[-1].update(errors=1),
                         lambda rows: rows[-2].update(call=2),
                         lambda rows: rows[-3].update(status="capture_fault"),
                         lambda rows: rows[1].update(call=0),
                         lambda rows: rows[0].update(catalog_sha256="0"*64)):
            rows = fixture(); mutation(rows)
            with self.assertRaises(ValueError):
                arguments.analyze_rows(rows)
        with self.assertRaises(ValueError):
            arguments.analyze_rows(fixture()[:-1])
        rows = fixture(); rows.append(deepcopy(rows[-1])); rows[-1]["sequence"] += 1
        with self.assertRaises(ValueError):
            arguments.analyze_rows(rows)

    def test_f32_bit_pattern_and_integer_ranges(self):
        rows = fixture(34)
        floating = next(row for row in rows if row.get("value_type") == "f32")
        floating["bits_u32"] = 0x7fc00000  # exact NaN storage is valid JSON integer
        arguments.analyze_rows(rows)
        floating["bits_u32"] = 2**32
        with self.assertRaises(ValueError):
            arguments.analyze_rows(rows)
        rows = fixture(34)
        signed = next(row for row in rows if row.get("value_type") == "i32")
        signed["value"] = 2**31
        with self.assertRaises(ValueError):
            arguments.analyze_rows(rows)

    def test_framing_privacy_and_output_proof_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"arguments.jsonl"
            text = "".join(json.dumps(row)+"\n" for row in fixture())
            path.write_text(text, encoding="utf-8")
            report = arguments.analyze(path)
            for key in ("memory_bounds_proven", "native_launches_cubin_bound", "stream_event_order_proven", "trace_complete"):
                self.assertIs(report[key], False)
            path.write_text(text[:-1], encoding="utf-8")
            with self.assertRaises(ValueError):
                arguments.analyze(path)
            path.write_text(text.replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1', 1), encoding="utf-8")
            with self.assertRaises(ValueError):
                arguments.analyze(path)

    def test_exact_launch_metadata_join(self):
        rows = fixture()
        _, calls = arguments.analyze_rows(rows)
        kernel = arguments.catalog()[0]
        launch = [{"schema_version": 2, "record_type": "xvram.cuda_launch_probe", "sequence": 1,
                   "kind": "setup", "call_id": 0, "hooks": 3},
                  {"schema_version": 2, "record_type": "xvram.cuda_launch_probe", "sequence": 2,
                   "kind": "begin", "call_id": 1, "api": "cuLaunchKernel_resolved_legacy",
                   "handle_kind": "contextless_kernel", "metadata": True, "module_resolved": True,
                   "kernel_name": kernel["symbol"], "parameter_count": len(kernel["arguments"])}]
        for index, arg in enumerate(kernel["arguments"]):
            launch.append({"schema_version": 2, "record_type": "xvram.cuda_launch_probe", "sequence": len(launch)+1,
                           "kind": "parameter", "call_id": 1, "index": index,
                           "offset_bytes": arg["offset_bytes"], "size_bytes": arg["size_bytes"]})
        launch.append({"schema_version": 2, "record_type": "xvram.cuda_launch_probe", "sequence": len(launch)+1,
                       "kind": "end", "call_id": 1, "result": 0})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"launch.jsonl"
            def write():
                path.write_text("".join(json.dumps(row)+"\n" for row in launch), encoding="utf-8")
            write()
            self.assertEqual(len(arguments.bind_launches(calls, path)), 64)
            launch[1]["kernel_name"] = "wrong_kernel"
            write()
            with self.assertRaises(ValueError):
                arguments.bind_launches(calls, path)
            launch[1]["kernel_name"] = kernel["symbol"]
            launch[-1]["result"] = 1
            write()
            with self.assertRaises(ValueError):
                arguments.bind_launches(calls, path)


if __name__ == "__main__":
    unittest.main()
