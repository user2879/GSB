from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gsbcache.content import NodeStore, artifact_digest, artifact_filename

from tests.helpers import make_sample


class DeterministicContentTests(unittest.TestCase):
    def test_same_node_key_is_byte_stable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _path, spec, _ = make_sample(Path(td))
            node = spec.node_by_id["node-a"]
            first = artifact_digest(node, "src:main")
            second = artifact_digest(node, "src:main")
            self.assertEqual(first, second)
            self.assertEqual(first, spec.artifacts["src:main"].expected_digest)
            store1 = NodeStore(Path(td) / "n1", spec, node)
            store2 = NodeStore(Path(td) / "n2", spec, node)
            self.assertEqual(store1.digest("src:main"), store2.digest("src:main"))
            self.assertEqual(store1.digest("src:main"), store1.digest("src:main"))

    def test_seed_mismatch_reports_expected_and_actual(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _path, spec, _ = make_sample(Path(td))
            node = spec.node_by_id["node-c"]
            store = NodeStore(Path(td) / "node-c", spec, node)
            detail = store.reusability("app:bin")
            self.assertFalse(detail["reusable"])
            self.assertEqual(detail["reason"], "digest_mismatch")
            self.assertEqual(detail["node_id"], "node-c")
            self.assertEqual(detail["expected_digest"], spec.artifacts["app:bin"].expected_digest)
            self.assertEqual(detail["actual_digest"], artifact_digest(node, "app:bin"))
            self.assertNotEqual(detail["expected_digest"], detail["actual_digest"])

    def test_reference_closure_mismatch_blocks_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _path, spec, _ = make_sample(Path(td))
            node = spec.node_by_id["node-a"]
            store = NodeStore(Path(td) / "node-a", spec, node)
            target_path = store.path_for("src:main")
            store.ensure("src:main")
            target_path.write_bytes(b"tampered-reference-content")
            detail = store.reusability("app:bin")
            self.assertFalse(detail["reusable"])
            self.assertEqual(detail["reason"], "reference_digest_mismatch")
            self.assertIn("src:main", detail["bad_references"])
            self.assertEqual(store.persisted_digest("src:main"), __import__("hashlib").sha256(b"tampered-reference-content").hexdigest())

    def test_persisted_file_name_is_key_hash(self) -> None:
        name = artifact_filename("src:main")
        self.assertTrue(name.startswith("artifact-"))
        self.assertTrue(name.endswith(".bin"))
        self.assertEqual(len(name), len("artifact-") + 64 + len(".bin"))


if __name__ == "__main__":
    unittest.main()
