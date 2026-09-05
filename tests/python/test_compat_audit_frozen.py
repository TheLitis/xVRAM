"""Phase 6b diagnostics cannot alter the previously delivered native contracts.

Phase 6a pins were recorded from ef6d52f3aae7b2a0783a7775016a74c877188a36.
Older headers/schemas reuse their existing hash catalogs without changing them.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
FROZEN_PHASE6A = {
    "include/xvram/cuda_compat.h": "7fb5f8bf4ff3115176e550992cefa422dc04d9bfb18a4f722337f33af2746163",
    "include/xvram/cuda_compat.hpp": "2babf9c739210ae4b9f64dbefdfb34207ecc434a381c4c0fcfe4f20b367a2003",
    "schemas/cuda-compat-v1.schema.json": "7477cfd04a5edd4e46169bf7a2ce1753b29078a907fcbf96ea3be17b6fc8f8b3",
    "schemas/cuda-compat-trace-v1.schema.json": "f0432e562083202cc5db79b04afd70f8fb812dc3ff1e3e05abc96c604a8788bc",
    "cmake/cuda_compat.exports.map": "53bd60b6ef0f05a7ff26adf948e0c07acc6971310d4e751dd17dbe4dc5dbe531",
    "tests/contract/cuda_compat_export_contract.py": "b6f00b4f7b4204b66f963e57c7710eed14007b7f1958fd83671536ce087599cb",
}
FROZEN_PRIOR_FILTERS = {
    "cmake/xvram.exports.map": "89e7e593deabaae28430ed0120451fc9d316fa12e9301bd0f0332539874b8bd3",
    "cmake/xvram_torch_allocator.exports.map": "ba97bc524f8651492e54dac9ad55e6a0710915a26f679aa8952d048fc2c90344",
    "cmake/xvram_torch_runtime.exports.map": "6006aa7535bec00769ecb1bdc79939e31edd03e601bf6624bc32276de3991f1c",
    "tests/contract/schema_hash_contract.py": "7041e50774582a9b46a7eea585f0c0f9cbc8962cabce3dcc43318db32a8ea31a",
    "tests/contract/abi_hash_contract.py": "066a3cf831312a5db245fbdc88b83834f9b3a6e5397ac1c54ec00d694c18626d",
}
PRIOR_SCHEMAS = (
    "capability-report-v1.schema.json",
    "capability-report-v2.schema.json",
    "vmm-poc-report-v1.schema.json",
    "residency-cache-report-v1.schema.json",
    "residency-trace-record-v1.schema.json",
    "gemm-bench-report-v1.schema.json",
    "pytorch-inference-report-v1.schema.json",
    "pytorch-inference-trace-record-v1.schema.json",
    "adaptive-compression-report-v1.schema.json",
    "compression-trace-record-v1.schema.json",
    "pytorch-inference-report-v2.schema.json",
    "pytorch-inference-trace-record-v2.schema.json",
)
PRIOR_HEADERS = (
    "include/xvram/xvram.h",
    "include/xvram/torch_allocator.h",
    "include/xvram/internal/torch_runtime.h",
    "include/xvram/xvram_v2.h",
    "include/xvram/internal/torch_runtime_v2.h",
)


def _catalog(name):
    spec = importlib.util.spec_from_file_location(
        "_xvram_audit_" + name, ROOT / "tests" / "contract" / (name + ".py")
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("frozen contract catalog unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EXPECTED


class AuditFrozenContractTests(unittest.TestCase):
    def assert_frozen(self, pins):
        for relative, expected in pins.items():
            with self.subTest(contract=relative):
                self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), expected)

    def test_phase6a_header_facade_schemas_and_export_filter(self):
        self.assert_frozen(FROZEN_PHASE6A)

    def test_existing_export_filters_and_hash_catalogs_remain_unchanged(self):
        self.assert_frozen(FROZEN_PRIOR_FILTERS)

    def test_earlier_report_and_trace_schemas_reuse_existing_pins(self):
        expected = _catalog("schema_hash_contract")
        self.assertEqual(len(PRIOR_SCHEMAS), len(expected))
        self.assert_frozen({"schemas/" + name: digest for name, digest in zip(PRIOR_SCHEMAS, expected)})

    def test_earlier_native_abi_headers_reuse_existing_pins(self):
        expected = _catalog("abi_hash_contract")
        self.assertEqual({Path(relative).name for relative in PRIOR_HEADERS}, set(expected))
        self.assert_frozen({relative: expected[Path(relative).name] for relative in PRIOR_HEADERS})


if __name__ == "__main__":
    unittest.main()
