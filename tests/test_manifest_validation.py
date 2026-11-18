"""要求1：清单与注册表加载、强校验、E_MANIFEST 字段路径。"""
import json
import tempfile
import unittest
from pathlib import Path

from ccdplatform.errors import ManifestError
from ccdplatform.manifest import load_manifest, load_nodes

VALID_NODES = {
    "nodes": [
        {"node_id": "node-a", "host": "127.0.0.1", "port": 21001},
        {"node_id": "node-b", "host": "127.0.0.1", "port": 21002},
    ]
}


def write(tmp: Path, name: str, data) -> Path:
    p = tmp / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


class ManifestValidationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.nodes_path = write(self.root, "nodes.json", VALID_NODES)
        self.nodes = load_nodes(self.nodes_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid_manifest_loads(self):
        manifest = {
            "versions": [
                {
                    "version": 1,
                    "strategy": {"batch_size": 2, "timeout_ms": 1000, "max_retries": 1},
                    "items": [
                        {"key": "a", "value": 1, "targets": ["node-a"]},
                        {
                            "key": "b",
                            "value": 2.0,
                            "targets": ["node-b"],
                            "derivation": {"refs": [["a", 1.0]]},
                        },
                    ],
                }
            ]
        }
        versions = load_manifest(write(self.root, "m.json", manifest), set(self.nodes))
        self.assertEqual(len(versions), 1)
        self.assertTrue(versions[0].items[1].is_derived)

    def test_non_monotonic_version_rejected(self):
        manifest = {"versions": [
            {"version": 2, "strategy": {"batch_size": 1, "timeout_ms": 1, "max_retries": 0},
             "items": [{"key": "a", "value": 1, "targets": ["node-a"]}]},
            {"version": 1, "strategy": {"batch_size": 1, "timeout_ms": 1, "max_retries": 0},
             "items": [{"key": "a", "value": 1, "targets": ["node-a"]}]},
        ]}
        with self.assertRaises(ManifestError) as ctx:
            load_manifest(write(self.root, "m.json", manifest), set(self.nodes))
        self.assertEqual(ctx.exception.code, "E_MANIFEST")
        self.assertIn("path", ctx.exception.details)

    def test_duplicate_version_rejected(self):
        manifest = {"versions": [
            {"version": 1, "strategy": {"batch_size": 1, "timeout_ms": 1, "max_retries": 0},
             "items": [{"key": "a", "value": 1, "targets": ["node-a"]}]},
            {"version": 1, "strategy": {"batch_size": 1, "timeout_ms": 1, "max_retries": 0},
             "items": [{"key": "a", "value": 2, "targets": ["node-a"]}]},
        ]}
        with self.assertRaises(ManifestError) as ctx:
            load_manifest(write(self.root, "m.json", manifest), set(self.nodes))
        self.assertEqual(ctx.exception.code, "E_MANIFEST")

    def test_unsupported_value_type(self):
        manifest = {"versions": [
            {"version": 1, "strategy": {"batch_size": 1, "timeout_ms": 1, "max_retries": 0},
             "items": [{"key": "a", "value": "1.0", "targets": ["node-a"]}]},
        ]}
        with self.assertRaises(ManifestError) as ctx:
            load_manifest(write(self.root, "m.json", manifest), set(self.nodes))
        self.assertEqual(ctx.exception.details["path"].endswith(".value"), True)
        self.assertEqual(ctx.exception.details["version"], 1)

    def test_unknown_target_node(self):
        manifest = {"versions": [
            {"version": 1, "strategy": {"batch_size": 1, "timeout_ms": 1, "max_retries": 0},
             "items": [{"key": "a", "value": 1, "targets": ["ghost"]}]},
        ]}
        with self.assertRaises(ManifestError) as ctx:
            load_manifest(write(self.root, "m.json", manifest), set(self.nodes))
        self.assertIn("targets", ctx.exception.details["path"])

    def test_missing_strategy_field(self):
        manifest = {"versions": [
            {"version": 1, "strategy": {"batch_size": 1, "timeout_ms": 1},
             "items": [{"key": "a", "value": 1, "targets": ["node-a"]}]},
        ]}
        with self.assertRaises(ManifestError) as ctx:
            load_manifest(write(self.root, "m.json", manifest), set(self.nodes))
        self.assertIn("max_retries", ctx.exception.details["path"])

    def test_boolean_value_rejected(self):
        manifest = {"versions": [
            {"version": 1, "strategy": {"batch_size": 1, "timeout_ms": 1, "max_retries": 0},
             "items": [{"key": "a", "value": True, "targets": ["node-a"]}]},
        ]}
        with self.assertRaises(ManifestError):
            load_manifest(write(self.root, "m.json", manifest), set(self.nodes))

    def test_port_conflict_rejected(self):
        bad = {"nodes": [
            {"node_id": "x", "port": 21009},
            {"node_id": "y", "port": 21009},
        ]}
        with self.assertRaises(ManifestError):
            load_nodes(write(self.root, "bad.json", bad))


if __name__ == "__main__":
    unittest.main()
