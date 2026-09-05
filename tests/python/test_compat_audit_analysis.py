"""No-driver, bounded-input and fail-closed compatibility audit tests."""

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import jsonschema

from xvram import compat_audit_analysis as audit


ROOT = Path(__file__).resolve().parents[2]


def string(value):
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def gguf(tensor_type=0, shape=(4,), tensor_name="weight", architecture="tiny", offset=0):
    metadata = b"" if architecture is None else string("general.architecture") + struct.pack("<I", 8) + string(architecture)
    header = b"GGUF" + struct.pack("<IQQ", 3, 1, int(architecture is not None)) + metadata
    header += string(tensor_name) + struct.pack("<I", len(shape)) + struct.pack("<" + "Q" * len(shape), *shape) + struct.pack("<IQ", tensor_type, offset)
    return header + bytes((-len(header)) % 32) + bytes(256)


def pe(with_import=True):
    data = bytearray(2048)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 60, 128)
    data[128:132] = b"PE\0\0"
    struct.pack_into("<HH", data, 132, 0x8664, 1)
    struct.pack_into("<H", data, 148, 240)
    struct.pack_into("<H", data, 152, 0x20B)
    struct.pack_into("<I", data, 212, 512)
    struct.pack_into("<I", data, 260, 16)
    if with_import:
        struct.pack_into("<II", data, 272, 4096, 40)
    struct.pack_into("<IIII", data, 400, 1536, 4096, 1536, 512)
    if with_import:
        struct.pack_into("<IIIII", data, 512, 4196, 0, 0, 4176, 4196)
        data[592:605] = b"KERNEL32.dll\0"
        struct.pack_into("<QQ", data, 612, 4216, 0)
        data[632:649] = b"\0\0GetProcAddress\0"
    return bytes(data)


def session(typed=False):
    return {"kind": "session", "collector_version": "0.1", "process_id": 1,
            "max_record_bytes": audit.MAX_RECORD_BYTES, "complete": False,
            "kernel_arguments_captured": typed, "tensor_bounds_known": typed,
            "runtime_parameter_bytes_not_kernel_arguments": True,
            "cublas_api_visibility": "source_proven" if typed else "unavailable",
            "timestamp_clock": "steady_clock", "activity_clock": "cupti"}


def api(correlation, operation, **kwargs):
    common = {"domain": "driver", "callback_id": 1, "correlation_id": correlation,
              "thread_id": 1, "symbol": "test_" + operation, "op": operation,
              "detail_known": True}
    common.update(kwargs)
    return [dict(common, kind="api_enter"), dict(common, kind="api_exit", status=0)]


def complete(records):
    return records + [{"kind": "summary", "callback_records": sum(item["kind"] in ("api_enter", "api_exit") for item in records),
                       "activity_records": sum(item["kind"] == "activity" for item in records),
                       "resource_records": sum(item["kind"] == "resource" for item in records),
                       "dropped_records": 0, "unknown_callback_details": 0,
                       "unknown_activity_kinds": 0, "serialization_errors": 0,
                       "collector_errors": 0, "complete": True, "safely_finalized": True,
                       "terminal_checkpoint": "explicit_finalize", "buffers_requested": 0,
                       "buffers_completed": 0, "incomplete_activities": 0}]


class AuditAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads((ROOT / "schemas/cuda-compat-audit-v1.schema.json").read_text())
        cls.trace_schema = json.loads((ROOT / "schemas/cuda-compat-audit-trace-v1.schema.json").read_text())
        jsonschema.Draft202012Validator.check_schema(cls.schema)
        jsonschema.Draft202012Validator.check_schema(cls.trace_schema)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.model_path = self.directory / "model.gguf"
        self.model_path.write_bytes(gguf())
        self.model = audit.read_gguf_metadata(self.model_path)
        self.binary = self.directory / "application.exe"
        self.binary.write_bytes(pe())
        self.inventory = audit.inventory([self.binary])
        self.profile = {"profile_id": "test", "cache_budget_bytes": 256 * 1024 * 1024,
                        "scratch_reserve_bytes": 0}
        self.cleanup = {"controller_reaped": True, "collector_finalized": True, "complete": True}

    def records(self, records):
        path = self.directory / "trace.jsonl"
        result = []
        for sequence, record in enumerate(records, 1):
            item = {"schema_version": 1, "report_type": audit.TRACE_TYPE,
                    "sequence": sequence, "timestamp_ns": sequence}
            item.update(record)
            result.append(item)
        path.write_text("".join(json.dumps(record) + "\n" for record in result), encoding="utf-8")
        return path, result

    def report(self, records, **kwargs):
        path, _ = self.records(records)
        result = audit.analyze([path], self.inventory, self.model, self.profile,
                               cleanup=self.cleanup, **kwargs)
        jsonschema.validate(result, self.schema)
        return result

    def codes(self, result):
        return {issue["code"] for issue in result["unresolved"]}

    def test_pe_hash_import_inventory_without_native_loading(self):
        item = self.inventory[0]
        self.assertEqual(item["evidence"], "observed")
        self.assertEqual(item["sha256"], hashlib.sha256(self.binary.read_bytes()).hexdigest())
        self.assertEqual(item["pe"]["imports"][0]["symbols"], ["GetProcAddress"])
        self.assertTrue(item["pe"]["dynamic_resolution_possible"])
        self.assertFalse(item["pe"]["absence_of_import_proves_static_linkage"])
        self.assertIsNone(item["path"])

    def test_no_import_is_not_static_linkage_proof(self):
        self.binary.write_bytes(pe(False))
        result = audit.inventory([self.binary])[0]
        self.assertEqual(result["pe"]["imports"], [])
        self.assertFalse(result["pe"]["absence_of_import_proves_static_linkage"])

    def test_truncated_and_malformed_pe_fail_closed(self):
        for data in (b"MZ", b"MZ" + bytes(62), pe()[:700]):
            with self.subTest(length=len(data)):
                self.binary.write_bytes(data)
                result = audit.inventory([self.binary])[0]
                self.assertEqual(result["evidence"], "unresolved")
                self.assertTrue(result["errors"])

    def test_import_thunk_outside_section_rejected(self):
        data = bytearray(pe())
        struct.pack_into("<I", data, 512, 0xFFFFFFFE)
        self.binary.write_bytes(data)
        self.assertEqual(audit.inventory([self.binary])[0]["evidence"], "unresolved")

    def test_inventory_missing_file_has_redacted_error(self):
        result = audit.inventory([self.directory / "secret" / "absent.exe"])[0]
        self.assertEqual(result["errors"], ["artifact_unreadable"])
        self.assertNotIn(str(self.directory), json.dumps(result))

    def test_metadata_does_not_read_weights(self):
        self.assertEqual(self.model["known_tensor_bytes"], 16)
        self.assertEqual(self.model["tensors"][0]["shape"], [4])
        self.assertLess(self.model["metadata_bytes_read"], self.model["data_offset_bytes"] + 1)
        original = self.model["metadata_sha256"]
        data = bytearray(self.model_path.read_bytes())
        data[-1] = 9
        self.model_path.write_bytes(data)
        self.assertEqual(audit.read_gguf_metadata(self.model_path)["metadata_sha256"], original)

    def test_quantized_row_uses_block_layout_not_padding(self):
        self.model_path.write_bytes(gguf(12, (256,)))
        result = audit.read_gguf_metadata(self.model_path)
        self.assertEqual(result["known_tensor_bytes"], 144)

    def test_unknown_tensor_type_has_unknown_extent(self):
        self.model_path.write_bytes(gguf(999))
        result = audit.read_gguf_metadata(self.model_path)
        self.assertEqual(result["evidence"], "unresolved")
        self.assertIsNone(result["tensors"][0]["size_bytes"])
        self.assertEqual(result["known_tensor_bytes"], 0)

    def test_gguf_malformed_and_overflow_inputs(self):
        cases = (b"GGUF", b"GGUF" + struct.pack("<IQQ", 99, 0, 0),
                 b"GGUF" + struct.pack("<IQQ", 3, audit.U64_MAX, 0),
                 gguf(shape=(audit.U64_MAX, 2)), gguf(12, (4,)), gguf(offset=1))
        for data in cases:
            with self.subTest(length=len(data)):
                self.model_path.write_bytes(data)
                with self.assertRaises(audit.AuditInputError):
                    audit.read_gguf_metadata(self.model_path)

    def test_shards_allow_architecture_only_on_first(self):
        other = self.directory / "second.gguf"
        other.write_bytes(gguf(tensor_name="second", architecture=None))
        combined = audit.combine_gguf_metadata([self.model, audit.read_gguf_metadata(other)])
        self.assertEqual(combined["architecture"], "tiny")
        self.assertEqual(combined["tensor_count"], 2)
        self.assertIsNone(combined["data_offset_bytes"])
        self.assertEqual([tensor["shard_index"] for tensor in combined["tensors"]], [0, 1])

    def test_shards_reject_duplicate_names_and_malformed(self):
        for shards in ([{}], [self.model, self.model], []):
            with self.subTest(shards=len(shards)):
                with self.assertRaises(audit.AuditInputError):
                    audit.combine_gguf_metadata(shards)

    def test_explicit_range_union_and_overflow(self):
        self.assertEqual(audit.range_union_bytes([(0, 8), (4, 8), (12, 0)]), 12)
        self.assertEqual(audit.range_union_bytes([(8, 4), (0, 4)]), 8)
        for spans in ([(audit.U64_MAX, 1)], [(-1, 1)], [(True, 1)]):
            with self.assertRaises(audit.AuditInputError):
                audit.range_union_bytes(spans)

    def test_missing_and_empty_trace_never_go(self):
        result = audit.analyze([], self.inventory, self.model, self.profile)
        jsonschema.validate(result, self.schema)
        self.assertEqual(result["decision"]["verdict"], "NO-GO")
        self.assertIn("missing_trace", self.codes(result))
        result = self.report(complete([session()]))
        self.assertIn("empty_kernel_coverage", self.codes(result))
        self.assertIsNone(result["memory_bounds"]["source_proven_peak_kernel_working_set_bytes"])

    def test_real_observer_session_and_all_record_kinds_validate(self):
        records = complete([session(), *api(1, "other"),
                            {"kind": "activity", "activity_kind": "unknown", "detail_known": False},
                            {"kind": "resource", "resource_kind": "context", "operation": "create", "context_id": 1, "detail_known": True},
                            {"kind": "gap", "reason": "cublas_api_not_observed_by_cupti", "lost_records": 0}])
        _, normalized = self.records(records)
        for record in normalized:
            audit.validate_trace_record(record)
            jsonschema.validate(record, self.trace_schema)

    def test_raw_address_unknown_fields_bool_integer_rejected(self):
        _, records = self.records([session()])
        for changes in ({"raw_pointer": 100}, {"process_id": True}, {"run_id": "pointer 0x12345678"}, {"sequence": audit.U64_MAX + 1}):
            with self.subTest(changes=changes):
                with self.assertRaises(audit.AuditInputError):
                    audit.validate_trace_record(dict(records[0], **changes))
                with self.assertRaises(jsonschema.ValidationError):
                    jsonschema.validate(dict(records[0], **changes), self.trace_schema)

    def test_malformed_json_and_duplicate_keys_invalid_input(self):
        path = self.directory / "broken.jsonl"
        for payload in ("{", '{"kind":"session","kind":"gap"}', '{"value":NaN}', "[[]]"):
            path.write_text(payload)
            result = audit.analyze([path], self.inventory, self.model, self.profile)
            self.assertEqual(result["outcome"]["exit_code"], 64)
            self.assertFalse(result["decision"]["execution_ready"])

    def test_line_and_state_limits_are_fail_closed(self):
        path = self.directory / "large.jsonl"
        path.write_bytes(b" " * 257)
        with mock.patch.object(audit, "MAX_RECORD_BYTES", 256):
            result = audit.analyze([path], [], {}, {})
            self.assertIn("trace_limit", self.codes(result))
        with mock.patch.object(audit, "MAX_STATE_ITEMS", 1):
            result = self.report(complete([session(), *api(1, "allocate", allocation_id=1, generation=1, size_bytes=16, range_known=True)]))
            self.assertIn("trace_state_limit", self.codes(result))

    def test_sequence_gap_and_record_after_summary(self):
        records = complete([session()]) + [{"kind": "gap", "reason": "after", "lost_records": 0, "sequence": 9}]
        result = self.report(records)
        self.assertTrue({"sequence_gap", "records_after_summary"} <= self.codes(result))

    def test_dropped_and_incomplete_summary_no_go(self):
        records = complete([session()])
        records[-1].update(dropped_records=3, complete=False, safely_finalized=False,
                           terminal_checkpoint="process_exit_unflushed", callback_records=12)
        result = self.report(records)
        self.assertTrue({"dropped_records", "collector_not_safely_finalized", "summary_counter_mismatch"} <= self.codes(result))

    def test_allocation_lifetime_generation_and_copy_bounds(self):
        records = [session(), *api(1, "allocate", allocation_id=1, generation=1, size_bytes=16, range_known=True),
                   *api(2, "copy", size_bytes=8, dst_allocation_id=1, dst_generation=1, dst_offset_bytes=12, dst_range_known=True),
                   *api(3, "free", allocation_id=1, generation=1),
                   *api(4, "copy", size_bytes=1, src_allocation_id=1, src_generation=1, src_offset_bytes=0, src_range_known=True),
                   *api(5, "allocate", allocation_id=1, generation=1, size_bytes=16, range_known=True)]
        result = self.report(complete(records))
        self.assertIn("copy_outside_live_allocation", self.codes(result))
        self.assertIn("allocation_identity_repeated_or_nested_api", self.codes(result))
        self.assertEqual(result["memory_bounds"]["observed_peak_live_allocation_bytes"], 16)
        self.assertIsNone(result["memory_bounds"]["source_proven_peak_kernel_working_set_bytes"])

    def test_failed_allocation_has_no_lifetime_effect(self):
        records = api(1, "allocate", allocation_id=1, generation=1, size_bytes=16, range_known=True)
        records[-1]["status"] = 2
        result = self.report(complete([session(), *records]))
        self.assertEqual(result["memory_bounds"]["observed_peak_live_allocation_bytes"], 0)
        self.assertEqual(result["coverage"]["failed_api_calls"], 1)

    def test_pitched_copy_overflow_is_invalid(self):
        records = [session(), *api(1, "allocate", allocation_id=1, generation=1, size_bytes=16, range_known=True),
                   *api(2, "copy", size_bytes=8, height=3, width_bytes=2, dst_pitch_bytes=audit.U64_MAX,
                        dst_allocation_id=1, dst_generation=1, dst_offset_bytes=0, dst_range_known=True)]
        result = self.report(complete(records))
        self.assertEqual(result["outcome"]["exit_code"], 64)
        self.assertIn("range_overflow", self.codes(result))

    def test_event_wait_and_vmm_alias_remain_unresolved(self):
        records = [session(), *api(1, "event", symbol="cuStreamWaitEvent", event_id=1, stream_id=2),
                   *api(2, "vmm_map", allocation_id=1, physical_allocation_id=2, size_bytes=16)]
        result = self.report(complete(records))
        self.assertTrue({"event_wait_without_record", "vmm_alias_contract_unresolved", "asynchronous_ordering_unresolved"} <= self.codes(result))

    def test_kernel_callback_layers_are_not_tensor_footprints(self):
        records = [session(), *api(1, "launch", kernel_name="callback_kernel"),
                   {"kind": "activity", "activity_kind": "kernel", "detail_known": True, "correlation_id": 1, "name": "actual_gpu_kernel", "start_ns": 100, "end_ns": 200}]
        result = self.report(complete(records))
        self.assertEqual(result["coverage"]["activity_kernel_types"][0]["name"], "actual_gpu_kernel")
        self.assertEqual(result["coverage"]["activity_kernel_types"][0]["read_write_ranges"], "unresolved")
        self.assertIn("missing_kernel_binding", self.codes(result))

    def test_typed_declarations_and_source_hash_cannot_manufacture_go(self):
        source = self.directory / "kernel.cpp"
        source.write_text("reviewed source still does not bind actual arguments")
        commit = "a" * 40
        self.profile["source_contracts"] = [{"contract_id": "tiny", "kernel_name": "tiny", "source_path": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "source_url": "https://example.com/" + commit,
            "upstream_commit": commit, "tensor_names": ["weight"], "scratch_bytes": 0}]
        binding = {"kind": "kernel_binding", "domain": "driver", "correlation_id": 2, "contract_id": "tiny", "kernel_name": "tiny", "scratch_bytes": 0,
                   "bindings": [{"tensor_name": "weight", "allocation_id": 1, "generation": 1, "offset_bytes": 0, "size_bytes": 16, "access": "read"}]}
        records = [session(True), *api(1, "allocate", allocation_id=1, generation=1, size_bytes=16, range_known=True), binding,
                   *api(2, "launch", kernel_name="tiny"),
                   {"kind": "activity", "activity_kind": "kernel", "detail_known": True, "correlation_id": 2, "name": "tiny"},
                   *api(3, "free", allocation_id=1, generation=1)]
        result = self.report(complete(records))
        self.assertIn("typed_module_function_bridge_unverified", self.codes(result))
        self.assertFalse(result["decision"]["execution_ready"])
        self.assertIsNone(result["memory_bounds"]["source_proven_peak_kernel_working_set_bytes"])
        self.assertEqual(result["coverage"]["resolved_kernel_launches"], 0)
        self.assertEqual(result["coverage"]["kernel_types"][0]["semantic_ranges"], "unresolved")
        self.assertNotIn(str(self.directory), json.dumps(result))

    def test_budget_and_scratch_unknown_remain_null(self):
        result = audit.analyze([], self.inventory, self.model, {"cache_budget_bytes": None, "scratch_reserve_bytes": None})
        jsonschema.validate(result, self.schema)
        self.assertIsNone(result["configuration"]["cache_budget_bytes"])
        self.assertIsNone(result["configuration"]["scratch_reserve_bytes"])

    def test_report_discards_inventory_paths_and_unknown_provenance(self):
        inventory = audit.inventory([self.binary], include_paths=True)
        result = audit.analyze([], inventory, self.model, self.profile, provenance={"secret_path": str(self.directory)})
        self.assertNotIn(str(self.directory), json.dumps(result))
        altered = dict(result, secret="extra")
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(altered, self.schema)

    def test_schema_u64_boundary_is_exact(self):
        _, records = self.records([session()])
        record = dict(records[0], timestamp_ns=audit.U64_MAX)
        jsonschema.validate(record, self.trace_schema)
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(dict(record, timestamp_ns=audit.U64_MAX + 1), self.trace_schema)


if __name__ == "__main__":
    unittest.main()
