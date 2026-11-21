import json
import os
import tempfile
import unittest

from configbench.errors import ManifestError
from configbench.manifest import load_manifest, load_registry, load_sample_decl

REG = {"nodes": [
    {"node_id": "node-alpha", "host": "127.0.0.1", "port": 47201,
     "group_key": "region", "scale": 3},
    {"node_id": "node-beta", "host": "127.0.0.1", "port": 47202,
     "group_key": "region", "scale": 4},
]}


class ManifestValidationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.reg_path = os.path.join(self.tmp, "registry.json")
        with open(self.reg_path, "w") as f:
            json.dump(REG, f)
        self.registry = load_registry(self.reg_path)

    def _load(self, manifest_obj):
        p = os.path.join(self.tmp, "manifest.json")
        with open(p, "w") as f:
            json.dump(manifest_obj, f)
        return load_manifest(p, self.registry)

    def base_manifest(self, **over):
        m = {
            "latency_budget_ms": 500,
            "versions": [{"version": 1, "items": {
                "a": {"value_type": "int", "value": 1,
                      "targets": ["node-alpha"]}}}],
        }
        m.update(over)
        return m

    def test_valid_load(self):
        m = self._load(self.base_manifest())
        self.assertEqual(m.latest.version, 1)
        self.assertEqual(m.latest.items["a"].targets, ("node-alpha",))

    def test_duplicate_version_rejected(self):
        m = self.base_manifest()
        m["versions"].append({"version": 1, "items": dict(m["versions"][0]["items"])})
        with self.assertRaises(ManifestError) as ctx:
            self._load(m)
        self.assertEqual(ctx.exception.code, "E_MANIFEST")
        self.assertIn("version", ctx.exception.field)

    def test_non_monotonic_version_rejected(self):
        m = self.base_manifest()
        m["versions"].append({"version": 1, "items": {}})
        m["versions"][1]["version"] = 1
        with self.assertRaises(ManifestError) as ctx:
            self._load(self.base_manifest(versions=[
                {"version": 2, "items": m["versions"][0]["items"]},
                {"version": 1, "items": m["versions"][0]["items"]}]))
        self.assertEqual(ctx.exception.code, "E_MANIFEST")
        self.assertIn("单调递增", ctx.exception.basis)

    def test_bad_key_rejected(self):
        m = self.base_manifest(versions=[{"version": 1, "items": {
            "1bad": {"value_type": "int", "value": 1, "targets": ["node-alpha"]}}}])
        with self.assertRaises(ManifestError) as ctx:
            self._load(m)
        self.assertIn("1bad", ctx.exception.field)

    def test_unsupported_value_type(self):
        m = self.base_manifest(versions=[{"version": 1, "items": {
            "a": {"value_type": "string", "value": "x", "targets": ["node-alpha"]}}}])
        with self.assertRaises(ManifestError) as ctx:
            self._load(m)
        self.assertIn("value_type", ctx.exception.field)

    def test_unknown_target_rejected(self):
        m = self.base_manifest(versions=[{"version": 1, "items": {
            "a": {"value_type": "int", "value": 1, "targets": ["node-ghost"]}}}])
        with self.assertRaises(ManifestError) as ctx:
            self._load(m)
        self.assertIn("node-ghost", ctx.exception.field)

    def test_duplicate_port_rejected(self):
        bad = {"nodes": [
            {"node_id": "n1", "host": "127.0.0.1", "port": 47300,
             "group_key": "g", "scale": 1},
            {"node_id": "n2", "host": "127.0.0.1", "port": 47300,
             "group_key": "g", "scale": 1}]}
        p = os.path.join(self.tmp, "r2.json")
        with open(p, "w") as f:
            json.dump(bad, f)
        with self.assertRaises(ManifestError) as ctx:
            load_registry(p)
        self.assertIn("port", ctx.exception.field)

    def test_missing_derive_ref_rejected(self):
        m = self.base_manifest(versions=[{"version": 1, "items": {
            "a": {"value_type": "int", "value": 1, "targets": ["node-alpha"],
                  "derives_from": [{"key": "ghost", "coef": {"raw": 1, "scale": 0}}]}}}])
        with self.assertRaises(ManifestError) as ctx:
            self._load(m)
        self.assertIn("derives_from", ctx.exception.field)

    def test_cycle_rejected(self):
        m = self.base_manifest(versions=[{"version": 1, "items": {
            "a": {"value_type": "int", "value": 1, "targets": ["node-alpha"],
                  "derives_from": [{"key": "b", "coef": {"raw": 1, "scale": 0}}]},
            "b": {"value_type": "int", "value": 2, "targets": ["node-alpha"],
                  "derives_from": [{"key": "a", "coef": {"raw": 1, "scale": 0}}]}}}])
        with self.assertRaises(ManifestError) as ctx:
            self._load(m)
        self.assertIn("环", ctx.exception.message)

    def test_sample_decl_validation(self):
        p = os.path.join(self.tmp, "s.json")
        with open(p, "w") as f:
            json.dump({"seed": 1, "n_samples": -1, "n_groups": 2,
                       "value_range": 10}, f)
        with self.assertRaises(ManifestError):
            load_sample_decl(p)


if __name__ == "__main__":
    unittest.main()
