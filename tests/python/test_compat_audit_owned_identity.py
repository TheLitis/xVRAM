import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from xvram import compat_audit_owned_identity as identity
from xvram.compat_audit_capture import run_process

CLEAN = {"controller_reaped": True, "process_tree_drained": True, "timed_out": False,
         "exit_code": 0, "errors": []}


def fixture():
    row = dict.fromkeys(identity.FIELDS, 0)
    row.update(magic=identity.MAGIC, version=1, bytes=identity.FORMAT.size, nonce=17, state=2,
               pid=1234, creation_filetime=100000, exit_filetime=200000, frequency=10000000,
               prepared_qpc=10, start_before_qpc=20, start_after_qpc=21,
               finish_before_qpc=30, finish_after_qpc=31)
    return row


def decode(row, capture=None, **changes):
    params = dict(nonce=17, frequency=10000000, reaped_qpc=40)
    params.update(changes)
    return identity.decode(identity._pack(row), CLEAN if capture is None else capture, **params)


class OwnedIdentityTests(unittest.TestCase):
    def test_private_identity_decode_and_redacted_representation(self):
        value = decode(fixture())
        self.assertEqual((value.native_pid, value.creation_filetime, value.exit_filetime), (1234, 100000, 200000))
        self.assertNotIn("1234", repr(value))
        self.assertNotIn("100000", repr(value))

    def test_clean_reap_required_even_with_valid_payload(self):
        for key, value in (("controller_reaped", False), ("process_tree_drained", False),
                           ("timed_out", True), ("exit_code", 27), ("exit_code", False),
                           ("errors", ["controller_failure"])):
            capture = dict(CLEAN); capture[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                decode(fixture(), capture)

    def test_missing_and_corrupt_header_rejected(self):
        for data in (b"", b"\0"*identity.FORMAT.size, identity._pack(fixture())[:-1]):
            with self.assertRaises(ValueError):
                identity.decode(data, CLEAN, nonce=17, frequency=10000000, reaped_qpc=40)
        for key, value in (("magic", 1), ("version", 2), ("bytes", 16)):
            row = fixture(); row[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                decode(row)

    def test_partial_start_timeout_bad_generation_and_clock(self):
        for key, value in (("state", 0), ("state", 1), ("pid", 0), ("pid", 2**32),
                           ("creation_filetime", 0), ("exit_filetime", 99999), ("exit_code", 27),
                           ("nonce", 18), ("start_before_qpc", 9), ("finish_before_qpc", 19),
                           ("finish_after_qpc", 41), ("frequency", 1)):
            row = fixture(); row[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                decode(row)
        for params in (dict(nonce=True), dict(frequency=False), dict(reaped_qpc=30)):
            with self.assertRaises(ValueError):
                decode(fixture(), **params)

    def test_torn_payload_checksum_rejected(self):
        data = bytearray(identity._pack(fixture()))
        data[40] ^= 1
        with self.assertRaisesRegex(ValueError, "checksum"):
            identity.decode(data, CLEAN, nonce=17, frequency=10000000, reaped_qpc=40)

    def test_disabled_hooks_do_not_inspect_process(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            identity.start(object())
            identity.finish(object())

    @unittest.skipUnless(os.name == "nt", "original Windows process-handle observation")
    def test_real_owned_short_child_no_pid_reopen_and_reap(self):
        process = None
        with identity.Ledger() as ledger, mock.patch.dict(os.environ, ledger.environment()), \
                mock.patch.object(identity, "_active", None):
            try:
                process = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read(1)"],
                                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           text=True)
                identity.start(process)
                with self.assertRaisesRegex(ValueError, "still_running"):
                    identity.finish(process)
                with self.assertRaises(ValueError):
                    ledger.snapshot_after_reap(CLEAN)
                with self.assertRaisesRegex(ValueError, "wrong_child"):
                    identity.finish(object())
                process.communicate("x", timeout=5)
                self.assertEqual(process.returncode, 0)
                identity.finish(process)
                value = ledger.snapshot_after_reap(CLEAN)
                self.assertEqual(value.native_pid, process.pid)
                self.assertGreaterEqual(value.exit_filetime, value.creation_filetime)
                self.assertLessEqual(value.finish_after_qpc, value.reaped_qpc)
                self.assertNotIn(str(process.pid), repr(value))
                self.assertEqual(process.poll(), 0)
                with self.assertRaisesRegex(ValueError, "duplicate_finish"):
                    identity.finish(process)
            finally:
                if process is not None:
                    if process.poll() is None:
                        process.kill()
                    process.communicate(timeout=5)

    @unittest.skipUnless(os.name == "nt", "Windows named mapping")
    def test_bad_mapping_header_fails_before_original_handle_query(self):
        with identity.Ledger() as ledger, mock.patch.dict(os.environ, ledger.environment()), \
                mock.patch.object(identity, "_active", None), \
                mock.patch.object(identity, "_times", side_effect=AssertionError("must_not_read_handle")):
            ledger.mapping[:] = b"\0"*identity.FORMAT.size
            with self.assertRaisesRegex(ValueError, "header"):
                identity.start(object())

    @unittest.skipUnless(os.name == "nt", "Windows bounded capture identity integration")
    def test_bounded_capture_reconciles_private_identity_without_report_expansion(self):
        with tempfile.TemporaryDirectory() as directory, identity.Ledger() as ledger:
            environment = dict(os.environ)
            environment.pop(identity.ENVIRONMENT, None)
            command = [sys.executable, "-c", "print('owned identity integration')"]
            ordinary = run_process(command, output_dir=Path(directory)/"ordinary", timeout_seconds=30,
                                   environment=environment)
            self.assertEqual(ordinary["exit_code"], 0, ordinary)
            environment.update(ledger.environment())
            observed = run_process(command, output_dir=Path(directory)/"observed", timeout_seconds=30,
                                   environment=environment)
            self.assertEqual(observed["exit_code"], 0, observed)
            self.assertTrue(observed["controller_reaped"])
            self.assertTrue(observed["process_tree_drained"])
            private = ledger.snapshot_after_reap(observed)
            self.assertGreater(private.native_pid, 0)
            self.assertGreaterEqual(private.exit_filetime, private.creation_filetime)
            self.assertEqual(set(observed), set(ordinary))
            self.assertEqual(observed["stdout_sha256"], ordinary["stdout_sha256"])
            forbidden = {"pid", "native_pid", "process_id", "creation_filetime", "exit_filetime",
                         "process_handle", "native_handle", "owned_identity", "identity_mapping"}

            def inspect(value):
                if type(value) is dict:
                    self.assertFalse(forbidden.intersection(value))
                    for item in value.values():
                        inspect(item)
                elif type(value) is list:
                    for item in value:
                        inspect(item)
                elif type(value) is int:
                    self.assertNotEqual(value, private.native_pid)
            inspect(observed)
            encoded = json.dumps(observed)
            self.assertNotIn(str(private.creation_filetime), encoded)
            self.assertNotIn(str(private.exit_filetime), encoded)
            self.assertNotIn(ledger.name, encoded)
            self.assertEqual((Path(directory)/"observed"/"stdout.txt").read_text().strip(),
                             "owned identity integration")

    @unittest.skipUnless(os.name == "nt", "Windows bounded capture identity integration")
    def test_bounded_capture_nonzero_exit_reaps_but_refuses_identity_proof(self):
        with tempfile.TemporaryDirectory() as directory, identity.Ledger() as ledger:
            environment = dict(os.environ); environment.update(ledger.environment())
            observed = run_process([sys.executable, "-c", "raise SystemExit(7)"], output_dir=Path(directory),
                                   timeout_seconds=30, environment=environment)
            self.assertEqual(observed["exit_code"], 7, observed)
            self.assertTrue(observed["controller_reaped"])
            self.assertTrue(observed["process_tree_drained"])
            # Bootstrap finish observed the actual failed child, but public
            # clean-reap admission must reject it, not certify a process proof.
            row = identity._unpack(ledger.mapping[:])
            self.assertEqual((row["state"], row["exit_code"]), (2, 7))
            with self.assertRaisesRegex(ValueError, "not_cleanly_reaped"):
                ledger.snapshot_after_reap(observed)

    @unittest.skipUnless(os.name == "nt", "Windows bounded capture identity integration")
    def test_bounded_capture_timeout_reaps_and_refuses_incomplete_identity(self):
        with tempfile.TemporaryDirectory() as directory, identity.Ledger() as ledger:
            environment = dict(os.environ); environment.update(ledger.environment())
            observed = run_process([sys.executable, "-c", "import threading; threading.Event().wait(60)"],
                                   output_dir=Path(directory), timeout_seconds=0.25, environment=environment)
            self.assertEqual(observed["exit_code"], 26, observed)
            self.assertTrue(observed["timed_out"])
            self.assertTrue(observed["controller_reaped"])
            self.assertTrue(observed["process_tree_drained"])
            # Do not assume the new interpreter reached Popen before deadline.
            # Either prepared or started is incomplete; both must fail closed.
            self.assertIn(identity._unpack(ledger.mapping[:])["state"], (0, 1))
            with self.assertRaisesRegex(ValueError, "not_cleanly_reaped"):
                ledger.snapshot_after_reap(observed)


if __name__ == "__main__":
    unittest.main()
