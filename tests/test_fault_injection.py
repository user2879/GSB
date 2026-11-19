import json
import tempfile
import unittest
from pathlib import Path

from cache_invalidation.engine import Engine
from cache_invalidation.errors import EGraph, EInherit, ESnapshot
from cache_invalidation.fault_injection import (
    inject_annotation_cycle,
    inject_missing_inheritance,
    inject_truncated_snapshot,
)
from cache_invalidation.loader import load_declarations
from cache_invalidation.model import EntryID
from cache_invalidation.snapshot import decode_snapshot_file
from cache_invalidation.store import Store
from tests.support import sample_graph, write_json


ROOT = Path(__file__).resolve().parent.parent


class FaultInjectionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.work = Path(self.temp.name)
        self.ns_path = write_json(
            self.work / "namespaces.json",
            json.loads((ROOT / "samples" / "namespaces.json").read_text()))
        self.entry_path = write_json(
            self.work / "entries.json",
            json.loads((ROOT / "samples" / "entries.json").read_text()))

    def tearDown(self):
        self.temp.cleanup()

    def test_annotation_cycle_injection_e_graph(self):
        paths = inject_annotation_cycle(
            self.ns_path, self.entry_path, self.work / "cycle")
        with self.assertRaises(EGraph) as caught:
            load_declarations(paths.namespace_path, paths.entry_path)
        self.assertEqual(caught.exception.code, "E_GRAPH")
        self.assertIn("cycle", caught.exception.details)

    def test_truncated_snapshot_injection_e_snapshot(self):
        engine = Engine(sample_graph(), store=Store())
        engine.explicit_invalidate(EntryID("catalog", 100), 100)
        snapshot = self.work / "snapshot.json"
        engine.save_snapshot(snapshot, 3)
        truncated = inject_truncated_snapshot(
            snapshot, self.work / "truncate")
        with self.assertRaises(ESnapshot) as caught:
            decode_snapshot_file(truncated)
        self.assertEqual(caught.exception.code, "E_SNAPSHOT")

    def test_missing_inheritance_injection_e_inherit(self):
        paths = inject_missing_inheritance(
            self.ns_path, self.entry_path, self.work / "inherit")
        with self.assertRaises(EInherit) as caught:
            load_declarations(paths.namespace_path, paths.entry_path)
        self.assertEqual(caught.exception.code, "E_INHERIT")
        self.assertEqual(
            caught.exception.details["missing_parent"],
            "missing_upstream_namespace")

    def test_injection_does_not_modify_original_inputs(self):
        original_ns = self.ns_path.read_text(encoding="utf-8")
        original_entries = self.entry_path.read_text(encoding="utf-8")
        inject_annotation_cycle(
            self.ns_path, self.entry_path, self.work / "cycle")
        self.assertEqual(
            self.ns_path.read_text(encoding="utf-8"), original_ns)
        self.assertEqual(
            self.entry_path.read_text(encoding="utf-8"),
            original_entries)


if __name__ == "__main__":
    unittest.main()
