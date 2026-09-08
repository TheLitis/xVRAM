import json
import copy
from pathlib import Path
import struct
import tempfile
import unittest
import jsonschema
from unittest import mock

from xvram import compat_launch_probe as probe


def image():
    data = bytearray(2048)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 60, 128)
    data[128:132] = b"PE\0\0"
    struct.pack_into("<HH", data, 132, 0x8664, 1)
    struct.pack_into("<H", data, 148, 240)
    struct.pack_into("<H", data, 152, 0x20B)
    struct.pack_into("<I", data, 260, 16)
    struct.pack_into("<II", data, 264, 4096, 128)
    struct.pack_into("<IIII", data, 400, 1536, 4096, 1536, 512)
    struct.pack_into("<IIIIII", data, 528, 1, 2, 2, 4136, 4144, 4152)
    struct.pack_into("<II", data, 552, 4500, 4501)
    struct.pack_into("<II", data, 560, 4160, 4177)
    struct.pack_into("<HH", data, 568, 0, 1)
    data[576:593] = b"cudaLaunchKernel\0"
    data[593:613] = b"cudaLaunchKernelExC\0"
    return bytes(data)


def trace(metadata=True):
    common = {"schema_version": 1, "record_type": "xvram.cuda_launch_probe", "call_id": 1}
    records = [{**common, "sequence": 1, "kind": "begin", "api": "cudaLaunchKernel",
                "metadata": metadata, "module_resolved": metadata, "kernel_name": "kernel" if metadata else "",
                "parameter_count": 1 if metadata else 0}]
    if metadata:
        records.append({**common, "sequence": 2, "kind": "parameter", "index": 0, "offset_bytes": 0, "size_bytes": 8})
    records.append({**common, "sequence": len(records) + 1, "kind": "end", "result": 0})
    return records


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "trace.jsonl"

    def analyze(self, records):
        self.path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        return probe.analyze_trace(self.path)

    def test_pe_named_ordinals(self):
        self.assertEqual(probe.pe_exports(image()), [("cudaLaunchKernel", 1), ("cudaLaunchKernelExC", 2)])

    def test_pe_corruptions(self):
        for offset, fmt, value in [(60, "<I", 2**32-1), (132, "<H", 0x14c), (148, "<H", 0),
                                   (152, "<H", 0x10b), (536, "<I", 4097), (568, "<H", 2),
                                   (552, "<I", 4096), (408, "<I", 2**32-1)]:
            with self.subTest(offset=offset):
                data = bytearray(image()); struct.pack_into(fmt, data, offset, value)
                with self.assertRaises(ValueError): probe.pe_exports(data)
        for length in (0, 64, 151, 400, 570):
            with self.assertRaises(ValueError): probe.pe_exports(image()[:length])

    def test_export_generation_is_pinned(self):
        self.path.write_bytes(image())
        with self.assertRaisesRegex(ValueError, "hash_mismatch"): probe.export_definition(self.path)
        with mock.patch.object(probe, "NATIVE_SHA256", probe.hashlib.sha256(image()).hexdigest()):
            definition = probe.export_definition(self.path)
            self.assertIn("cudaLaunchKernel=xvram_probe_launch @1 PRIVATE", definition)
            self.assertIn("LIBRARY xvram_native_cudart64_13.dll", probe.export_definition(self.path, import_definition=True))

    def test_routing_and_metadata(self):
        for metadata in (True, False):
            report = self.analyze(trace(metadata))
            self.assertEqual(report["counts"]["calls_returned"], 1)
            self.assertEqual(report["counts"]["metadata_calls"], int(metadata))
            self.assertEqual(len(report["layouts"]), int(metadata))

    def test_metadata_snapshot_not_name_cache(self):
        records = trace()
        second = trace()
        for r in second:
            r["sequence"] += 3; r["call_id"] = 2
        second[1]["size_bytes"] = 4
        report = self.analyze(records + second)
        self.assertEqual(len(report["layouts"]), 2)

    def test_strict_records(self):
        cases = [(0, "sequence", 2), (0, "call_id", True), (0, "schema_version", True),
                 (0, "metadata", 1), (0, "module_resolved", False), (0, "kernel_name", "0x12345678"),
                 (0, "parameter_count", 257), (0, "api", "cuLaunchKernelEx"),
                 (1, "index", 1), (1, "size_bytes", 0), (1, "offset_bytes", 65536),
                 (2, "call_id", 2), (2, "result", -1), (1, "raw_address", 123)]
        for index, key, value in cases:
            with self.subTest(key=key, value=value):
                records = trace(); records[index][key] = value
                with self.assertRaises(ValueError): self.analyze(records)

    def test_empty_truncated_oversized_duplicate(self):
        for records in ([], trace()[:-1], trace()[1:], [trace()[0], trace()[2]]):
            with self.assertRaises(ValueError): self.analyze(records)
        for data in (b"{}", b"x" * 65537 + b"\n", b'{"kind":"a","kind":"b"}\n'):
            self.path.write_bytes(data)
            with self.assertRaises(ValueError): probe.analyze_trace(self.path)

    def test_native_error_observed_not_success(self):
        records = trace(); records[-1]["result"] = 700
        self.assertEqual(self.analyze(records)["counts"]["native_errors"], 1)

    def test_stage_refuses_overwrite_and_changed_profile(self):
        with mock.patch.object(probe, "verify_binary_directory", return_value=[]):
            with self.assertRaises(ValueError): probe.stage(Path(self.temp.name), self.path, Path(self.temp.name))

    def capture(self):
        return dict(exit_code=0, timed_out=False, controller_reaped=True, process_tree_drained=True,
                    elapsed_ms=10, stdout_sha256="a" * 64, output_truncated=False, errors=[])

    def schema(self, name):
        return json.loads((Path(__file__).resolve().parents[2] / "schemas" / name).read_text(encoding="utf-8"))

    def test_report_success_and_faults(self):
        self.analyze(trace())
        capture = self.capture()
        report = probe.make_report("metadata", "runtime_proxy", {"cudart64_13.dll": "a" * 64}, capture,
                                   self.path, microbatch=128)
        schema = self.schema("cuda-launch-probe-report-v1.schema.json")
        jsonschema.validate(report, schema)
        self.assertEqual(report["exit_code"], 0)
        for key, value in [("exit_code", 1), ("controller_reaped", False), ("process_tree_drained", False),
                           ("output_truncated", True), ("errors", ["child_failed"])]:
            with self.subTest(key=key):
                failed = probe.make_report("metadata", "runtime_proxy", {"cudart64_13.dll": "a" * 64},
                                           {**capture, key: value}, self.path, microbatch=128)
                self.assertEqual(failed["exit_code"], 27)
                jsonschema.validate(failed, schema)
        timeout = probe.make_report("metadata", "runtime_proxy", {"cudart64_13.dll": "a" * 64},
                                    {**capture, "timed_out": True, "exit_code": 26}, self.path, microbatch=128)
        self.assertEqual(timeout["exit_code"], 26)
        jsonschema.validate(timeout, schema)
        for mutate in (lambda r: r["proof"].update(cubin_binding=True),
                       lambda r: r["trace"]["counts"].update(parameters=9),
                       lambda r: r["trace"]["layouts"][0].update(calls=2),
                       lambda r: r["capture"].update(controller_reaped=False)):
            broken = copy.deepcopy(report); mutate(broken)
            with self.assertRaises(ValueError): probe.validate_report(broken)
        report["raw_address"] = 123
        with self.assertRaises(jsonschema.ValidationError): jsonschema.validate(report, schema)

    def test_driver_setup_and_record_schema(self):
        records = trace()
        setup = {"schema_version": 1, "record_type": "xvram.cuda_launch_probe", "sequence": 1,
                 "call_id": 0, "kind": "setup", "hooks": 2}
        records[0]["api"] = "cuLaunchKernel_resolved_ptsz"
        with self.assertRaises(ValueError): self.analyze(records)
        for record in records: record["sequence"] += 1
        records.insert(0, setup)
        self.assertTrue(self.analyze(records)["driver_setup_observed"])
        schema = self.schema("cuda-launch-probe-trace-v1.schema.json")
        for record in records: jsonschema.validate(record, schema)
        report = probe.make_report("metadata", "driver_entry_probe", {"nvcuda.dll": "a" * 64},
                                   self.capture(), self.path, microbatch=1)
        self.assertEqual(report["exit_code"], 0)

    def test_controller_hang_is_contained(self):
        import sys
        output = Path(self.temp.name) / "capture"
        output.mkdir()
        result = probe.run_process([sys.executable, "-c", "import time; time.sleep(30)"],
                                   output_dir=output, timeout_seconds=1)
        self.assertTrue(result["timed_out"])
        self.assertTrue(result["controller_reaped"])
        self.assertTrue(result["process_tree_drained"])
        report = probe.make_report("metadata", "driver_entry_probe", {"nvcuda.dll": "a" * 64},
                                   result, output / "missing.jsonl", microbatch=1)
        jsonschema.validate(report, self.schema("cuda-launch-probe-report-v1.schema.json"))
        self.assertEqual(report["exit_code"], 26)


if __name__ == "__main__":
    unittest.main()
