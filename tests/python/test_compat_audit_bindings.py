"""Module hashes are useful evidence, never an implicit namespace bridge."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import jsonschema
from xvram import compat_audit_bindings as binding
from xvram.compat_audit_bindings_contract import validate_report, empty_report
from xvram.compat_audit_analysis import validate_trace_record, AuditInputError
from test_compat_audit_analysis import session, complete, api
from test_compat_audit_correlations import launch, activity
from test_compat_audit_binary_contract import _completed

ROOT=Path(__file__).resolve().parents[2]

def records():
    load=api(1,"other",domain="driver",symbol="cuModuleLoadData",context_id=2)
    load[1].update(module_id=20,module_generation=1)
    lookup=api(2,"module_function",domain="driver",symbol="cuModuleGetFunction",
               context_id=2,module_id=20,module_generation=1,kernel_name="test_kernel")
    lookup[1].update(function_id=4,function_generation=1)
    launched=launch("driver",10,module_id=20,module_generation=1,function_generation=1)
    return [session(), *load,
            {"kind":"resource","resource_kind":"module","operation":"load","context_id":2,
             "module_id":50,"module_generation":1,"size_bytes":512,"detail_known":True,"module_copy_queued":True},
            {"kind":"module_hash","context_id":2,"module_id":50,"module_generation":1,"source_sequence":4,
             "size_bytes":512,"sha256":"c"*64},
            *lookup,*launched,activity(),activity("api_driver")]

class BindingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.trace=self.root/"trace.jsonl"; self.binary=self.root/"binary.json"
        self.binary.write_text(json.dumps(_completed()),encoding="utf-8")
        self.trace_schema=json.loads((ROOT/"schemas/cuda-compat-audit-trace-v3.schema.json").read_text())
        self.report_schema=json.loads((ROOT/"schemas/cuda-compat-launch-bindings-v1.schema.json").read_text())
        self.trace_validator=jsonschema.Draft202012Validator(self.trace_schema)
        self.report_validator=jsonschema.Draft202012Validator(self.report_schema)

    def write(self, items=None, terminal=True):
        items=records() if items is None else items
        if terminal:
            items=complete(items)
            queued=sum(r.get("module_copy_queued",False) for r in items)
            hashed=[r for r in items if r["kind"]=="module_hash"]
            items[-1].update(module_copies_submitted=queued,module_copies_retired=len(hashed),
                module_hash_records=len(hashed),module_copy_active_bytes=0,
                module_copy_total_bytes=512*queued,module_copy_peak_bytes=512 if queued else 0)
        values=[dict(schema_version=3,report_type="xvram.cuda_compat_audit_trace",sequence=i,timestamp_ns=i,**r)
                for i,r in enumerate(items,1)]
        self.trace.write_text("".join(json.dumps(r)+"\n" for r in values),encoding="utf-8")
        return values

    def analyze(self, items=None, terminal=True):
        values=self.write(items,terminal)
        for r in values:
            validate_trace_record(r); self.trace_validator.validate(r)
        result=binding.analyze(self.trace,self.binary)
        validate_report(result,allow_pending_cleanup=True)
        self.report_validator.validate(result)
        return result

    def test_module_hash_and_native_lookup_do_not_bridge_namespaces(self):
        result=self.analyze()
        self.assertEqual(result["coverage"]["static_module_matches"],1)
        self.assertEqual(result["coverage"]["native_launch_links"],1)
        self.assertEqual(result["coverage"]["proven_launch_bindings"],0)
        self.assertEqual(result["decision"]["verdict"],"NO-GO")
        self.assertTrue(result["integrity"]["launch_records_reconciled"])

    def test_missing_terminal_preserves_only_observations(self):
        result=self.analyze(terminal=False)
        self.assertEqual(result["coverage"]["hashed_modules"],1)
        self.assertFalse(result["integrity"]["trace_complete"])

    def test_wrong_hash_cannot_match_static_module(self):
        items=records(); items[4]["sha256"]="f"*64
        self.assertEqual(self.analyze(items)["coverage"]["static_module_matches"],0)

    def test_module_hash_wrong_generation_or_size(self):
        for key,value in (("module_generation",2),("module_id",20),("size_bytes",511),("source_sequence",3)):
            with self.subTest(key=key):
                items=records(); items[4][key]=value
                result=self.analyze(items)
                self.assertFalse(result["integrity"]["input_valid"])
                self.assertEqual(result["coverage"]["hashed_modules"],0)

    def test_delayed_hash_after_unload_retains_original_generation(self):
        items=records(); digest=items.pop(4)
        items.insert(4,{"kind":"resource","resource_kind":"module","operation":"unload","context_id":2,
                        "module_id":50,"module_generation":1,"size_bytes":512,"detail_known":True})
        items.append(digest)
        result=self.analyze(items)
        self.assertEqual(result["modules"][0]["unload_sequence"],5)
        self.assertEqual(result["coverage"]["hashed_modules"],1)

    def test_module_reuse_duplicate_context_and_missing_hash(self):
        for context,generation in ((2,1),(2,2),(3,1)):
            items=records(); additional=copy.deepcopy(items[3]); additional.update(context_id=context,module_generation=generation)
            items.append(additional)
            result=self.analyze(items)
            self.assertFalse(result["integrity"]["input_valid"])
            self.assertFalse(result["decision"]["complete_binding_coverage"])

    def test_failed_lookup_and_stale_function_are_not_native_links(self):
        for update in ({"status":1},{"function_generation":2},{"module_generation":2}):
            items=records(); items[6].update(update)
            self.assertEqual(self.analyze(items)["coverage"]["native_launch_links"],0)

    def test_buffered_function_cannot_claim_current_generation(self):
        item=dict(schema_version=3,report_type="xvram.cuda_compat_audit_trace",kind="activity",sequence=1,timestamp_ns=1,
                  activity_kind="function",context_id=2,module_id=50,function_id=4,function_index=2,module_generation=1,detail_known=True)
        with self.assertRaises(AuditInputError): validate_trace_record(item)

    def test_non_monotonic_truncated_oversized_and_raw_fields(self):
        values=self.write()
        variants=[values[0]|{"sequence":2}, values[3]|{"address":12345}, values[4]|{"sha256":"0x123abc"}]
        for value in variants:
            self.trace.write_text(json.dumps(value)+"\n",encoding="utf-8")
            with self.assertRaises(ValueError): binding.analyze(self.trace,self.binary)
        for payload in ('{"schema_version":', ' '* (1024*1024+1)):
            self.trace.write_text(payload,encoding="utf-8")
            with self.assertRaises(ValueError): binding.analyze(self.trace,self.binary)

    def test_old_versions_refuse_new_fields(self):
        for version in (1,2):
            for value in self.write()[:5]:
                value["schema_version"]=version
                if value["kind"] in ("resource","module_hash"):
                    with self.assertRaises(ValueError): validate_trace_record(value)

    def test_input_changed_and_uncompleted_static_report(self):
        self.write()
        with mock.patch.object(binding,"_file_hash",return_value="0"*64):
            with self.assertRaises(AuditInputError): binding.analyze(self.trace,self.binary)
        self.binary.write_text(json.dumps(empty_report()))
        with self.assertRaises(ValueError): binding.analyze(self.trace,self.binary)

    def test_report_counter_privacy_and_false_success_rejected(self):
        report=self.analyze()
        for section,key,value in (("coverage","proven_launch_bindings",1),("decision","execution_ready",True),
                                  ("coverage","hashed_modules",2),("provenance","trace_sha256","0x123abc")):
            invalid=copy.deepcopy(report); invalid[section][key]=value
            with self.assertRaises(ValueError): validate_report(invalid,allow_pending_cleanup=True)
        with self.assertRaises(ValueError): validate_report(report)

    def test_existing_audit_schemas_frozen(self):
        pins={"binary-evidence-v1":"eab2f88f20f3ac55fc0bbbd9af2d7600b9c9fb375124c2be49a69d51d0aacdf0",
              "observations-v1":"fc05efe763ca28ac1b5e69b268ac44475e411b58b3bc92d12d9ec2ea1beb0da2",
              "trace-v1":"e7922d80d39b55c914c88739e5d97b84d117a3e67540b0507a6f1cebfef7a63f",
              "trace-v2":"d85e4ba28888e55c92a765183e7f0c6b57e9041b4d02586b7a06ef2b4aefa2fc"}
        for suffix,digest in pins.items():
            self.assertEqual(hashlib.sha256((ROOT/f"schemas/cuda-compat-audit-{suffix}.schema.json").read_bytes()).hexdigest(),digest)
        self.assertEqual(hashlib.sha256((ROOT/"schemas/cuda-compat-audit-v1.schema.json").read_bytes()).hexdigest(),
                         "07b7d9284f3a091745b1bfc0126eac00cdcf d75bb0ee11496b5c17a94b351b5e".replace(" ",""))

    def test_real_controller_success_and_output_refusal(self):
        self.write(); output=self.root/"result.json"
        args=["--trace",str(self.trace),"--binary-evidence",str(self.binary),"--json",str(output)]
        self.assertEqual(binding.main(args),0)
        result=json.loads(output.read_text()); validate_report(result)
        self.assertTrue(all(result["cleanup"].values()))
        before=output.read_bytes(); self.assertEqual(binding.main(args),74); self.assertEqual(output.read_bytes(),before)

    def test_real_controller_crash_and_timeout_reap(self):
        from xvram import compat_audit_capture as capture
        run=capture.run_process
        self.write()
        for payload,code in (("raise SystemExit(17)",27),("import time; time.sleep(30)",26)):
            output=self.root/f"failure-{code}.json"
            def replace(command,**kwargs):
                return run([sys.executable,"-c",payload],**kwargs)
            with mock.patch.object(capture,"run_process",side_effect=replace):
                self.assertEqual(binding.main(["--trace",str(self.trace),"--binary-evidence",str(self.binary),
                                               "--json",str(output),"--timeout-seconds","1"]),code)
            result=json.loads(output.read_text()); validate_report(result)
            self.assertTrue(all(result["cleanup"].values()))
            self.assertFalse(result["decision"]["execution_ready"])

    def test_controller_rejects_truncated_oversized_and_contradictory_results(self):
        from xvram import compat_audit_capture as capture
        valid=self.analyze()
        invalid=copy.deepcopy(valid); invalid["decision"]["execution_ready"]=True
        for index,(payload,exit_code) in enumerate(((b"{",0),(b" "*(4*1024*1024+1),0),
                        (json.dumps(valid).encode(),17),(json.dumps(invalid).encode(),0))):
            output=self.root/f"invalid-{index}.json"
            def fake(command,**kwargs):
                Path(command[-1]).with_name("result.json").write_bytes(payload)
                return {"controller_reaped":True,"process_tree_drained":True,"timed_out":False,
                        "exit_code":exit_code,"errors":[],"output_truncated":False}
            with mock.patch.object(capture,"run_process",side_effect=fake):
                self.assertEqual(binding.main(["--trace",str(self.trace),"--binary-evidence",str(self.binary),"--json",str(output)]),27)
            report=json.loads(output.read_text()); validate_report(report)
            self.assertEqual(report["coverage"]["hashed_modules"],0)

    def test_output_io_error_is_74(self):
        self.write()
        self.assertEqual(binding.main(["--trace",str(self.trace),"--binary-evidence",str(self.binary),
                                       "--json",str(self.root/"absent"/"result.json")]),74)

    def test_failed_staging_admission_never_claims_complete_trace(self):
        items=records(); items[3]["module_copy_queued"]=False; del items[4]
        result=self.analyze(items)
        self.assertEqual(result["coverage"]["hashed_modules"],0)
        self.assertIn("module_copy_not_queued",result["integrity"]["issues"])
        self.assertFalse(result["integrity"]["trace_complete"])

if __name__=="__main__": unittest.main()
