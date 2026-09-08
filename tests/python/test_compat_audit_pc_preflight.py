import copy
import json
import unittest
from pathlib import Path
import jsonschema

from xvram import compat_pc_preflight as pc


class PCPreflightTests(unittest.TestCase):
    def setUp(self):
        self.capture = dict(exit_code=0, timed_out=False, controller_reaped=True,
                            process_tree_drained=True, output_truncated=False, errors=[])
        self.rows = [dict(profiler_initialize=0, device_support_result=0, device_support_level=3),
                     dict(stage="before_empty_kernel"),
                     dict(launch=0, synchronize=0, data=0, unload=0, disable=0, destroy=0)]

    def report(self):
        report = pc.make_report(self.capture, "\n".join(map(json.dumps, self.rows)),
                                {"xvram-pc-sampling-probe.exe": "a"*64, **pc.PINS})
        schema = Path(__file__).resolve().parents[2] / "schemas/cuda-pc-preflight-v1.schema.json"
        jsonschema.validate(report, json.loads(schema.read_text()))
        return report

    def test_permission_is_checked_by_data_not_enable(self):
        self.rows[-1].update(data=35, disable=35)
        self.capture["exit_code"] = 27
        report = self.report()
        self.assertEqual(report["outcome"], dict(status="permission_required", exit_code=23))
        self.assertFalse(any(report["proof"].values()))

    def test_ready_is_not_execution_proof(self):
        report = self.report()
        self.assertEqual(report["outcome"]["status"], "ready")
        self.assertFalse(any(report["proof"].values()))

    def test_failed_stage_or_counter_does_not_become_permission_success(self):
        for key in self.rows[-1]:
            original = copy.deepcopy(self.rows)
            self.rows[-1][key] = -1
            self.assertEqual(self.report()["outcome"]["exit_code"], 27)
            self.rows = original

    def test_crash_missing_reap_and_truncation_fail_closed(self):
        for key, value in (("exit_code", 1), ("controller_reaped", False), ("process_tree_drained", False),
                           ("output_truncated", True), ("errors", ["error"])):
            original = copy.deepcopy(self.capture)
            self.capture[key] = value
            self.assertEqual(self.report()["outcome"]["exit_code"], 27)
            self.capture = original
        self.capture["timed_out"] = True
        self.assertEqual(self.report()["outcome"]["exit_code"], 26)

    def test_corrupt_output_and_privacy_fields(self):
        text = "\n".join(map(json.dumps, self.rows))
        for invalid in ("", text*2, text.replace('"data": 0', '"data": true'),
                        text.replace('"data": 0', '"data": 0, "data": 0'),
                        text.replace('"data": 0', '"data": 0, "pointer": 123'), "x"*4097):
            with self.assertRaises(ValueError):
                pc.parse_smoke(invalid)


if __name__ == "__main__":
    unittest.main()
