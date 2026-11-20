import json
import tempfile
import unittest
from pathlib import Path

from colbench.sample_data import generate
from colbench.snapshot import load_snapshot, restore_snapshot
from colbench.versioning import (BudgetParams, ParameterVersion,
                                 ViewState, ViewStore, VersionRegistry)
from colbench.workbench import Workbench


class VersioningTests(unittest.TestCase):
    def test_budget_version_monotonic_and_stale(self):
        tmp = Path(tempfile.mkdtemp())
        registry = VersionRegistry(tmp / "v.json")
        v1 = registry.observe_budget(BudgetParams(4, 8192, 2))
        v2 = registry.observe_budget(BudgetParams(4, 8192, 2))
        self.assertEqual(v1.budget, v2.budget)
        v3 = registry.observe_budget(BudgetParams(1, 1024, 1))
        self.assertEqual(v3.budget, v1.budget + 1)
        self.assertEqual(v3.encoding, v1.encoding)

    def test_view_replacement_lists_blocks_and_clears(self):
        store = ViewStore()
        version = ParameterVersion(1, 1, 1)
        first = ViewState(version=version,
                          trajectories=[{"segment_id": "s", "row_group": 0,
                                         "column": "a", "block_index": 0}])
        store.replace(first)
        next_version = ParameterVersion(2, 1, 1)
        second = ViewState(version=next_version,
                           trajectories=[{"segment_id": "s", "row_group": 1,
                                          "column": "a", "block_index": 0}])
        report = store.replace(second)
        self.assertEqual(report.cleared_trajectory_count, 1)
        self.assertEqual(len(report.affected_block_keys), 2)
        self.assertEqual(store.state.stale_trajectories,
                         first.trajectories)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.bench = Workbench(self.tmp)
        self.bench.build_from_specs(generate(self.tmp / "samples"))

    def test_snapshot_roundtrip_and_digest(self):
        version = self.bench.registry.observe_budget(
            BudgetParams(2, 4096, 1))
        path = self.tmp / "snap.json"
        from colbench import snapshot
        bundle = snapshot.save_snapshot(
            path, self.bench.segments, version,
            [{"segment_id": "s", "row_group": 0, "column": "a",
              "block_index": 0}], [{"segment_id": "s", "row_group": 0,
                                    "column": "a"}], {"tasks": 1})
        loaded = load_snapshot(path)
        self.assertEqual(loaded.content_digest, bundle.content_digest)

    def test_snapshot_mismatch_replaces_with_backup(self):
        version = self.bench.registry.observe_budget(
            BudgetParams(2, 4096, 1))
        path = self.tmp / "snap.json"
        from colbench import snapshot
        snapshot.save_snapshot(path, self.bench.segments, version, [], [],
                               {"tasks": 0})
        changed = ParameterVersion(version.budget + 1, version.encoding,
                                   version.format)
        report = restore_snapshot(path, self.bench.segments, changed)
        self.assertEqual(report["action"], "REPLACED")
        self.assertTrue(Path(report["backup"]).exists())
        self.assertTrue(report["reason"]["version_changed"])

    def test_tampered_snapshot_digest_fails(self):
        from colbench.errors import SnapshotError
        version = self.bench.registry.observe_budget(
            BudgetParams(2, 4096, 1))
        path = self.tmp / "tamper.json"
        from colbench import snapshot
        snapshot.save_snapshot(path, self.bench.segments, version, [], [],
                               {"tasks": 0})
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["statistics"]["tasks"] = 99
        path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaises(SnapshotError):
            load_snapshot(path)


if __name__ == "__main__":
    unittest.main()
