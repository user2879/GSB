import json
import tempfile
import unittest
from pathlib import Path

from configreg.errors import ManifestError
from configreg.manifest import key_set_delta, load_manifest_pair, load_registry


REGISTRY = {
    "nodes": [
        {"node_id": "node_a", "host": "127.0.0.1", "port": 21001},
        {"node_id": "node_b", "host": "127.0.0.1", "port": 21002},
    ]
}


def write(tmp: Path, name: str, payload: dict) -> Path:
    path = tmp / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class ManifestValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.registry_path = write(self.root, "registry.json", REGISTRY)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _manifests(self, baseline: dict, candidate: dict):
        base_path = write(self.root, "baseline.json", baseline)
        cand_path = write(self.root, "candidate.json", candidate)
        return load_manifest_pair(base_path, cand_path, self.registry_path)

    def test_valid_pair_and_key_delta(self) -> None:
        baseline = {
            "version": 1,
            "strategy": "rolling",
            "items": {
                "a": {"derived": False, "value": 1, "targets": ["node_a"]},
                "b": {"derived": True, "coeffs": {"a": 2}, "targets": ["node_a"]},
            },
        }
        candidate = {
            "version": 2,
            "strategy": "canary",
            "items": {
                "a": {"derived": False, "value": 2, "targets": ["node_a", "node_b"]},
                "c": {"derived": False, "value": 3, "targets": ["node_b"]},
            },
        }
        base, cand, registry = self._manifests(baseline, candidate)
        self.assertEqual(registry.node_ids(), ["node_a", "node_b"])
        self.assertEqual(key_set_delta(base, cand)["added"], ["c"])
        self.assertEqual(key_set_delta(base, cand)["removed"], ["b"])

    def test_cycle_rejected(self) -> None:
        manifest = {
            "version": 1,
            "strategy": "broadcast",
            "items": {
                "a": {"derived": True, "coeffs": {"b": 1}, "targets": ["node_a"]},
                "b": {"derived": True, "coeffs": {"a": 1}, "targets": ["node_a"]},
            },
        }
        with self.assertRaises(ManifestError) as ctx:
            self._manifests(manifest, {**manifest, "version": 2})
        self.assertEqual(ctx.exception.code, "E_MANIFEST")
        self.assertTrue(any("环" in v["problem"] for v in ctx.exception.violations))

    def test_unknown_target_and_duplicate_rejected(self) -> None:
        manifest = {
            "version": 1,
            "strategy": "broadcast",
            "items": {"a": {"derived": False, "value": 1, "targets": ["node_a", "node_a", "ghost"]}},
        }
        with self.assertRaises(ManifestError) as ctx:
            self._manifests(manifest, {**manifest, "version": 2})
        fields = {v["field"] for v in ctx.exception.violations}
        self.assertTrue(any("targets" in field for field in fields))

    def test_duplicate_port_rejected(self) -> None:
        bad_registry = {
            "nodes": [
                {"node_id": "n1", "port": 22001},
                {"node_id": "n2", "port": 22001},
            ]
        }
        path = write(self.root, "bad.json", bad_registry)
        with self.assertRaises(ManifestError):
            load_registry(path)

    def test_version_must_increase(self) -> None:
        manifest = {
            "version": 3,
            "strategy": "broadcast",
            "items": {"a": {"derived": False, "value": 1, "targets": ["node_a"]}},
        }
        with self.assertRaises(ManifestError):
            self._manifests(manifest, {**manifest, "version": 3})


if __name__ == "__main__":
    unittest.main()
