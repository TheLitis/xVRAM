"""Synthetic official-layout SQLite fixtures; no Nsight installation or GPU needed."""
from __future__ import annotations

import hashlib
import contextlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

from xvram.compat_audit_nsys import summarize_nsys


class AuditNsightTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="xvram-nsys-summary-")
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "native.sqlite"

    @contextlib.contextmanager
    def database(self):
        connection = sqlite3.connect(self.path)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def fixture(self):
        with self.database() as connection:
            connection.executescript("""
                CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT);
                INSERT INTO StringIds VALUES(1, 'quant_kernel<float>');
                INSERT INTO StringIds VALUES(2, 'cudaLaunchKernel');
                INSERT INTO StringIds VALUES(3, 'cuMemMap');
                INSERT INTO StringIds VALUES(4, 'kernel 0xdeadbeef1234 handle=0000123456789abc');
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
                    start INTEGER,end INTEGER,demangledName INTEGER,mangledName INTEGER,
                    streamId INTEGER,contextId INTEGER,correlationId INTEGER);
                INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(100,150,1,NULL,998877,776655,554433);
                INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(120,190,NULL,1,998877,776655,554433);
                CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,nameId INTEGER);
                INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(10,30,2);
                CREATE TABLE CUPTI_ACTIVITY_KIND_DRIVER(start INTEGER,end INTEGER,nameId INTEGER);
                INSERT INTO CUPTI_ACTIVITY_KIND_DRIVER VALUES(20,60,3);
                CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,bytes INTEGER,
                    srcAddress INTEGER,dstAddress INTEGER,streamId INTEGER);
                INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(30,40,4096,998877665544,887766554433,778899);
                CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET(start INTEGER,end INTEGER,bytes INTEGER,address INTEGER);
                INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES(40,45,128,665544332211);
            """)

    def test_aggregates_fixed_known_fields_without_modifying_database(self):
        self.fixture()
        before = self.path.read_bytes()
        entries_before = sorted(path.name for path in self.path.parent.iterdir())
        result = summarize_nsys(self.path)
        self.assertEqual(result["report_type"], "xvram.cuda_compat_audit_nsys")
        self.assertFalse(result["coverage_complete"])
        self.assertEqual(result["source"]["sha256"], hashlib.sha256(before).hexdigest())
        self.assertEqual(result["counts"], {"kernels": 2, "runtime": 1, "driver": 1, "memcpy": 1, "memset": 1})
        self.assertEqual(result["kernels"][0]["name"], "quant_kernel<float>")
        self.assertEqual(result["kernels"][0]["count"], 2)
        self.assertEqual(result["kernels"][0]["total_duration_ns"], 120)
        self.assertEqual(result["kernels"][0]["min_duration_ns"], 50)
        self.assertEqual(result["kernels"][0]["max_duration_ns"], 70)
        self.assertIn("not_application_wall_time", result["timing_scope"])
        self.assertEqual(result["apis"]["runtime"][0]["name"], "cudaLaunchKernel")
        self.assertEqual(result["apis"]["driver"][0]["name"], "cuMemMap")
        self.assertEqual(result["apis"]["cublas"], [])
        self.assertIn("cuBLAS_api_visibility_not_established", result["unresolved"])
        self.assertEqual(result["memory"][0]["total_bytes"], 4096)
        encoded = json.dumps(result, allow_nan=False)
        for forbidden in ("998877", "776655", "554433", "665544332211", str(self.path), "streamId", "contextId", "srcAddress"):
            self.assertNotIn(forbidden, encoded)
        self.assertEqual(before, self.path.read_bytes())
        self.assertEqual(entries_before, sorted(path.name for path in self.path.parent.iterdir()))

    def test_native_values_in_symbol_text_are_redacted(self):
        self.fixture()
        with self.database() as connection:
            connection.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET demangledName=4")
        result = summarize_nsys(self.path)
        encoded = json.dumps(result)
        self.assertNotIn("0xdeadbeef1234", encoded)
        self.assertNotIn("0000123456789abc", encoded)
        self.assertIn("native-value-redacted", encoded)
        self.assertFalse(result["coverage_complete"])

    def test_missing_and_malformed_inputs_are_unknown(self):
        missing = summarize_nsys(self.path)
        self.assertIn("database_unavailable", missing["unresolved"])
        self.assertTrue(all(value is None for value in missing["counts"].values()))
        self.path.write_bytes(b"not a sqlite database" * 10)
        result = summarize_nsys(self.path)
        self.assertIn("database_header_invalid", result["unresolved"])
        self.assertFalse(result["coverage_complete"])

    def test_absent_or_unsupported_tables_never_mean_verified_zero_activity(self):
        with self.database() as connection:
            connection.execute("CREATE TABLE unrelated(value TEXT)")
            connection.execute("CREATE VIEW CUPTI_ACTIVITY_KIND_KERNEL AS SELECT value FROM unrelated")
            connection.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER, nameId INTEGER)")
        result = summarize_nsys(self.path)
        self.assertEqual(result["tables"]["kernels"], "failed")
        self.assertIn("unsupported_table_kind", result["unresolved"])
        self.assertEqual(result["tables"]["runtime"], "unsupported_columns")
        self.assertEqual(result["tables"]["driver"], "missing")
        self.assertIsNone(result["counts"]["driver"])
        self.assertFalse(result["coverage_complete"])

    def test_row_name_file_and_string_limits_are_explicit(self):
        self.fixture()
        with mock.patch("xvram.compat_audit_nsys.MAX_DATABASE_BYTES", 16):
            self.assertIn("database_size_outside_bound", summarize_nsys(self.path)["unresolved"])
        with mock.patch("xvram.compat_audit_nsys.MAX_TABLE_ROWS", 1):
            result = summarize_nsys(self.path)
        self.assertEqual(result["counts"]["kernels"], 1)
        self.assertEqual(result["tables"]["kernels"], "truncated")
        self.assertIn("kernels_row_limit_exceeded", result["unresolved"])
        with self.database() as connection:
            connection.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET demangledName=4 WHERE start=120")
        with mock.patch("xvram.compat_audit_nsys.MAX_DISTINCT_NAMES", 1):
            result = summarize_nsys(self.path)
        self.assertEqual(result["counts"]["kernels"], sum(item["count"] for item in result["kernels"]))
        self.assertIn("kernels_distinct_name_limit_exceeded", result["unresolved"])
        with mock.patch("xvram.compat_audit_nsys.MAX_STRING_ROWS", 1):
            result = summarize_nsys(self.path)
        self.assertIn("referenced_names_unobserved", result["unresolved"])
        self.assertIn("string_row_limit_exceeded", result["unresolved"])

    def test_invalid_durations_and_missing_transfer_bytes_remain_unknown(self):
        self.fixture()
        with self.database() as connection:
            connection.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET end=start-1")
            connection.execute("UPDATE CUPTI_ACTIVITY_KIND_MEMCPY SET bytes=NULL")
        result = summarize_nsys(self.path)
        self.assertEqual(result["kernels"][0]["invalid_duration_count"], 2)
        self.assertIsNone(result["kernels"][0]["total_duration_ns"])
        self.assertIsNone(result["memory"][0]["total_bytes"])
        self.assertIn("kernels_invalid_duration", result["unresolved"])
        self.assertIn("memcpy_byte_observation_unavailable", result["unresolved"])

    def test_live_wal_is_not_ignored_or_removed(self):
        self.fixture()
        sidecar = Path(str(self.path) + "-wal")
        sidecar.write_bytes(b"unfinalized sidecar")
        result = summarize_nsys(self.path)
        self.assertIn("database_not_finalized_sidecar_present", result["unresolved"])
        self.assertEqual(sidecar.read_bytes(), b"unfinalized sidecar")
        self.assertFalse(result["coverage_complete"])

    def test_direct_names_unknown_references_and_name_length_bound(self):
        with self.database() as connection:
            connection.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,name TEXT)")
            connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(1,2,'direct_kernel')")
        result = summarize_nsys(self.path)
        self.assertEqual(result["kernels"][0]["name"], "direct_kernel")
        with self.database() as connection:
            connection.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET name=?", ("x" * 3000,))
        result = summarize_nsys(self.path)
        self.assertIn("name_length_limit_exceeded", result["unresolved"])
        self.assertLess(len(json.dumps(result)), 10000)

    def test_expired_normalization_deadline_is_not_complete(self):
        self.fixture()
        with mock.patch("xvram.compat_audit_nsys.MAX_QUERY_SECONDS", 0):
            result = summarize_nsys(self.path)
        self.assertIn("normalization_deadline_exceeded", result["unresolved"])
        self.assertFalse(result["coverage_complete"])

    def test_runtime_callchain_rows_are_not_reported_as_extra_api_calls(self):
        self.fixture()
        with self.database() as connection:
            connection.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_RUNTIME ADD COLUMN callchainId INTEGER")
            connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(100,900,2,123456)")
        result = summarize_nsys(self.path)
        self.assertEqual(result["counts"]["runtime"], 1)
        self.assertEqual(result["apis"]["runtime"][0]["total_duration_ns"], 20)
        self.assertNotIn("123456", json.dumps(result))

    def test_mixed_runtime_table_preserves_driver_calls_with_explicit_family_counts(self):
        self.fixture()
        with self.database() as connection:
            connection.executescript("""
                CREATE TABLE ENUM_NSYS_EVENT_CLASS(id INTEGER,name TEXT,label TEXT);
                INSERT INTO ENUM_NSYS_EVENT_CLASS VALUES(0,'TRACE_PROCESS_EVENT_CUDA_RUNTIME','CUDA runtime');
                INSERT INTO ENUM_NSYS_EVENT_CLASS VALUES(1,'TRACE_PROCESS_EVENT_CUDA_DRIVER','CUDA driver');
                ALTER TABLE CUPTI_ACTIVITY_KIND_RUNTIME ADD COLUMN eventClass INTEGER;
                UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET eventClass=0;
                INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(40,80,3,1);
                DROP TABLE CUPTI_ACTIVITY_KIND_DRIVER;
            """)
        result = summarize_nsys(self.path)
        self.assertEqual(result["counts"]["runtime"], 2)
        self.assertEqual(result["tables"]["driver"], "missing")
        self.assertIsNone(result["api_class_counts"]["driver"])
        self.assertEqual(result["api_class_counts"]["runtime"], {
            "runtime": 1, "driver": 1, "cublas": 0, "cudnn": 0, "other": 0, "unknown": 0})
        self.assertEqual({row["name"] for row in result["apis"]["runtime"]}, {"cudaLaunchKernel", "cuMemMap"})
        self.assertIn("not_api_families", result["api_grouping"])

    def test_profiler_warnings_are_preserved_as_codes_without_native_text(self):
        self.fixture()
        with self.database() as connection:
            connection.executescript("""
                CREATE TABLE ENUM_DIAGNOSTIC_SEVERITY_LEVEL(id INTEGER,name TEXT,label TEXT);
                INSERT INTO ENUM_DIAGNOSTIC_SEVERITY_LEVEL VALUES(1,'Info','Info');
                INSERT INTO ENUM_DIAGNOSTIC_SEVERITY_LEVEL VALUES(2,'Warning','Warning');
                CREATE TABLE DIAGNOSTIC_EVENT(severity INTEGER,text TEXT,globalPid INTEGER);
            """)
            messages = [(1, "Profiling has started.", 9988776655),
                        (2, "Installed CUDA driver version (13.4) is not supported by this build of Nsight Systems. CUDA trace will be collected using libraries for driver version 13.3", 9988776655),
                        (1, "CUDA hardware tracing is not supported on this system. A legacy (software instrumented) trace was collected instead.", 9988776655),
                        (1, r"Loaded CUPTI library: C:\private-directory\cupti64.dll pointer=0xdeadbeef", 9988776655),
                        (1, "Profiling has stopped.", 9988776655)]
            connection.executemany("INSERT INTO DIAGNOSTIC_EVENT VALUES(?,?,?)", messages)
        result = summarize_nsys(self.path)
        diagnostic = result["profiler_diagnostics"]
        self.assertEqual(diagnostic["status"], "observed")
        self.assertEqual(diagnostic["count"], 5)
        self.assertEqual(diagnostic["severity_counts"], {"info": 4, "warning": 1, "error": 0, "verbose": 0, "unknown": 0})
        self.assertEqual(set(diagnostic["codes"]), {"driver_version_outside_profiler_support",
                         "legacy_software_instrumentation", "profiling_start_observed", "profiling_stop_observed"})
        self.assertIn("profiler_driver_version_outside_support", result["unresolved"])
        self.assertIn("profiler_diagnostic_warning_observed", result["unresolved"])
        self.assertFalse(result["coverage_complete"])
        for forbidden in ("9988776655", "private-directory", "0xdeadbeef", "globalPid", "13.4"):
            self.assertNotIn(forbidden, json.dumps(result))

    def test_diagnostic_and_enumeration_limits_remain_unknown(self):
        self.fixture()
        with self.database() as connection:
            connection.executescript("""
                CREATE TABLE ENUM_NSYS_EVENT_CLASS(id INTEGER,name TEXT);
                INSERT INTO ENUM_NSYS_EVENT_CLASS VALUES(1,'TRACE_PROCESS_EVENT_CUDA_DRIVER');
                INSERT INTO ENUM_NSYS_EVENT_CLASS VALUES(1,'TRACE_PROCESS_EVENT_CUDA_RUNTIME');
                CREATE TABLE DIAGNOSTIC_EVENT(severity INTEGER,text TEXT);
                INSERT INTO DIAGNOSTIC_EVENT VALUES(999,'unrecognized event'),(999,'second event');
            """)
        with mock.patch("xvram.compat_audit_nsys.MAX_DIAGNOSTIC_ROWS", 1):
            result = summarize_nsys(self.path)
        self.assertIn("api_class_mapping_failed_or_query_deadline_exceeded", result["unresolved"])
        self.assertIn("profiler_diagnostic_row_limit_exceeded", result["unresolved"])
        self.assertEqual(result["profiler_diagnostics"]["status"], "truncated")
        self.assertEqual(result["profiler_diagnostics"]["count"], 1)
        self.assertEqual(result["profiler_diagnostics"]["severity_counts"]["unknown"], 1)
        self.assertFalse(result["coverage_complete"])


if __name__ == "__main__":
    unittest.main()
