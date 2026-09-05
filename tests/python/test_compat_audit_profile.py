"""Offline integrity checks for the bounded compatibility audit input catalog."""

import importlib.resources
import json
import pathlib
import re
import shutil
import subprocess
import tempfile
import unittest


class CompatAuditProfileTests(unittest.TestCase):
    def setUp(self):
        resource = importlib.resources.files("xvram").joinpath("compat_audit_profile.json")
        self.profile = json.loads(resource.read_text(encoding="utf-8"))

    def test_exact_upstream_release_and_extracted_identity(self):
        upstream = self.profile["upstream"]
        self.assertEqual(upstream["tag"], "b10819")
        self.assertEqual(upstream["commit"], "6a1a922d269908a29cbd4b49c27e6a8e7fd10fae")
        self.assertEqual(len(upstream["files"]), 54)
        self.assertEqual(
            upstream["files"]["llama-cli.exe"],
            "b6ff33d8174d39a819bb0ca6e7c81085aad1f2ab201f0e0fea4b710009c1446c",
        )
        self.assertEqual(
            upstream["files"]["llama-completion.exe"],
            "10678184d600ff60f6b3962771b6b2bef48477755236a02e2f7f7f00c0d1fa8d",
        )
        self.assertEqual(
            upstream["files"]["ggml-cuda.dll"],
            "88350839e27a43212a52cf6686562b2b6ef1498c59391dfb8cda49f3ebd89a62",
        )
        for name, digest in upstream["files"].items():
            self.assertEqual(pathlib.PureWindowsPath(name).name, name)
            self.assertIn(pathlib.PureWindowsPath(name).suffix, (".dll", ".exe"))
            self.assertRegex(digest, r"^[a-f0-9]{64}$")

    def test_two_hashed_archives_from_pinned_official_release(self):
        archives = self.profile["upstream"]["archives"]
        self.assertEqual(len(archives), 2)
        self.assertEqual(len({entry["name"] for entry in archives}), 2)
        for entry in archives:
            self.assertEqual(
                entry["url"],
                "https://github.com/ggml-org/llama.cpp/releases/download/b10819/"
                + entry["name"],
            )
            self.assertRegex(entry["sha256"], r"^[a-f0-9]{64}$")
            self.assertGreater(entry["bytes"], 0)

    def test_complete_shards_for_exact_model_revisions(self):
        expected = {
            "14b": ("b466e1f8c07172155743e8e1307507d8a4f91fbd", 3, 8988110496),
            "32b": ("a15e3cc10f8bbb2c0af6f8f1f34a32e3b060c09d", 5, 19851336384),
        }
        self.assertEqual(set(self.profile["models"]), set(expected))
        for key, (revision, count, byte_count) in expected.items():
            model = self.profile["models"][key]
            self.assertEqual(model["repo"], "Qwen/Qwen2.5-" + key.upper() + "-Instruct-GGUF")
            self.assertEqual(model["revision"], revision)
            self.assertEqual(len(model["files"]), count)
            self.assertEqual(sum(entry["bytes"] for entry in model["files"]), byte_count)
            for index, entry in enumerate(model["files"], 1):
                self.assertEqual(
                    entry["name"],
                    "qwen2.5-{}-instruct-q4_k_m-{:05d}-of-{:05d}.gguf".format(key, index, count),
                )
                self.assertRegex(entry["sha256"], r"^[a-f0-9]{64}$")
                self.assertGreater(entry["bytes"], 0)

    def test_source_catalog_never_claims_complete_kernel_contract(self):
        entries = self.profile["source_evidence"]
        self.assertEqual(len(entries), len({entry["evidence_id"] for entry in entries}))
        self.assertTrue(any(entry["status"] == "unresolved" for entry in entries))
        footprints = next(
            entry for entry in entries if entry["evidence_id"] == "quantized-kernel-access-contracts"
        )
        self.assertEqual(footprints["status"], "unresolved")
        for entry in entries:
            self.assertIn(entry["status"], ("source_proven", "unresolved"))
            self.assertTrue(entry["limitation"])
            self.assertTrue(entry["claim"])
            self.assertTrue(re.match(r"^https://(?:api[.])?github[.]com/", entry["source"]))

    def test_downloader_preserves_existing_invalid_archive_without_network(self):
        shell = shutil.which("pwsh") or shutil.which("powershell")
        if shell is None:
            self.skipTest("PowerShell unavailable; manifest validation still runs without it")
        script = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "fetch-phase6b-audit-dependencies.ps1"
        with tempfile.TemporaryDirectory(prefix="xvram-audit-dependencies-") as temporary:
            destination = pathlib.Path(temporary)
            archive = destination / self.profile["upstream"]["archives"][0]["name"]
            archive.write_bytes(b"invalid existing archive must remain untouched")
            observed = subprocess.run(
                [
                    shell, "-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                    "-File", str(script), "-DependencyRoot", str(destination), "-Models", "none", "-VerifyOnly",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20,
                check=False,
            )
            self.assertNotEqual(observed.returncode, 0)
            self.assertIn("wrong size", observed.stdout + observed.stderr)
            self.assertEqual(archive.read_bytes(), b"invalid existing archive must remain untouched")
            self.assertEqual(list(destination.iterdir()), [archive])


if __name__ == "__main__":
    unittest.main()
