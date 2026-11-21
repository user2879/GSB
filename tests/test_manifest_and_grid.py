import json
import os
import tempfile
import unittest

from configmesh.errors import ErrorCode, BusinessError
from configmesh.fixedpoint import Grid, quantize, derive_qvalue, error_bound, compare_derived
from configmesh.manifest import load_registry, load_manifest


class ManifestValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.registry_path = os.path.join(self.tmp, "registry.json")
        self.manifest_path = os.path.join(self.tmp, "manifest.json")
        with open(self.registry_path, "w", encoding="utf-8") as fh:
            json.dump({"nodes": [
                {"node_id": "n1", "port": 18201},
                {"node_id": "n2", "port": 18202},
            ]}, fh)
        self.registry = load_registry(self.registry_path)

    def _load_manifest(self, payload):
        with open(self.manifest_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        return load_manifest(self.manifest_path, self.registry)

    def test_valid_manifest_loads(self):
        versions = self._load_manifest({"versions": [{
            "version": 1,
            "items": [{"key": "a", "value": 1.0},
                      {"key": "b", "derived_from": {"a": 2.0}}],
            "targets": ["n1"],
            "strategy": {"mode": "full", "shard_count": 1},
        }]})
        self.assertEqual([v.version for v in versions], [1])

    def test_stable_violation_listing_and_all_or_nothing(self):
        with self.assertRaises(BusinessError) as ctx:
            self._load_manifest({"versions": [
                {"version": 2, "items": [{"key": "bad key", "value": 1}],
                 "targets": ["missing_node"], "strategy": {"mode": "weird"}},
                {"version": 1, "items": [{"key": "a", "value": 1}],
                 "targets": ["n1"], "strategy": {"mode": "full"}},
            ]})
        self.assertEqual(ctx.exception.code, ErrorCode.E_MANIFEST)
        entries = [d["entry"] for d in ctx.exception.details]
        self.assertEqual(entries, sorted(entries))
        self.assertGreaterEqual(len(entries), 4)

    def test_cycle_has_dedicated_code(self):
        with self.assertRaises(BusinessError) as ctx:
            self._load_manifest({"versions": [{
                "version": 1,
                "items": [
                    {"key": "a", "derived_from": {"b": 1.0}},
                    {"key": "b", "derived_from": {"a": 1.0}},
                ],
                "targets": ["n1"],
                "strategy": {"mode": "full"},
            }]})
        self.assertEqual(ctx.exception.code, ErrorCode.E_CYCLE)

    def test_corrupt_json_is_e_manifest(self):
        with open(self.manifest_path, "w", encoding="utf-8") as fh:
            fh.write("{broken")
        with self.assertRaises(BusinessError) as ctx:
            load_manifest(self.manifest_path, self.registry)
        self.assertEqual(ctx.exception.code, ErrorCode.E_MANIFEST)

    def test_duplicate_registry_port_rejected(self):
        with open(self.registry_path, "w", encoding="utf-8") as fh:
            json.dump({"nodes": [{"node_id": "a", "port": 18300},
                                 {"node_id": "b", "port": 18300}]}, fh)
        with self.assertRaises(BusinessError) as ctx:
            load_registry(self.registry_path)
        self.assertEqual(ctx.exception.code, ErrorCode.E_MANIFEST)


class FixedPointTests(unittest.TestCase):
    def test_grid_rejects_illegal_parameters(self):
        for bad in (-1, 10, 1.5, True, "3"):
            with self.assertRaises(BusinessError) as ctx:
                Grid.create(bad)  # type: ignore[arg-type]
            self.assertEqual(ctx.exception.code, ErrorCode.E_GRID)

    def test_truncation_quantization_and_integer_derive(self):
        grid = Grid.create(2)
        self.assertEqual(quantize(1.239, grid), 123)
        self.assertEqual(quantize(-1.239, grid), -123)
        q = derive_qvalue({"a": quantize(2, grid)}, {"a": quantize(3, grid)}, grid)
        self.assertEqual(q, 600)

    def test_error_bound_and_stable_comparison(self):
        grid = Grid.create(2)
        coeffs = {"a": quantize(1.4, grid), "b": quantize(-0.3, grid)}
        inputs = {"a": quantize(2.0, grid), "b": quantize(5.0, grid)}
        bound = error_bound(coeffs, inputs, grid)
        self.assertIn("bound_q", bound)
        self.assertTrue(all("contribution_pct" in t for t in bound["terms"]))
        decision = compare_derived("a", 100, 1, "b", 100, 1)
        self.assertFalse(decision["decisive"])
        self.assertEqual(decision["tie_break_winner"], "a")
        decision2 = compare_derived("a", 1000, 1, "b", 100, 1)
        self.assertTrue(decision2["decisive"])
        self.assertEqual(decision2["verdict"], "GT")

    def test_saturation_clamps(self):
        from configmesh.fixedpoint import saturate_int64, INT64_MAX, INT64_MIN
        self.assertEqual(saturate_int64(10**100), INT64_MAX)
        self.assertEqual(saturate_int64(-10**100), INT64_MIN)


if __name__ == "__main__":
    unittest.main()
