import json
import tempfile
import unittest
from pathlib import Path

from graphidx import (
    IndexIntegrityError,
    MaintenanceSession,
    ReplayError,
    replay_snapshot,
)

from tests.helpers import make_session, sample_changesets


class SnapshotCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.snapshot = Path(self.tmp.name) / "session.json"
        self.session = make_session()
        for cs in sample_changesets():
            self.session.apply_changeset(cs)
        self.session.save_snapshot(self.snapshot)

    def tearDown(self):
        self.tmp.cleanup()


class TestSnapshotRestore(SnapshotCase):
    def test_snapshot_contains_required_sections(self):
        snap = json.loads(self.snapshot.read_text(encoding="utf-8"))
        for key in (
            "graph", "index_def_version", "entries", "views",
            "invalidation_trace", "recompute_ledger", "changesets", "params",
        ):
            self.assertIn(key, snap)
        self.assertIn("digest", snap["graph"])
        self.assertIn("version", snap["graph"])

    def test_restore_in_sync_no_replacement(self):
        other = make_session()
        for cs in sample_changesets():
            other.apply_changeset(cs)
        report = other.restore(self.snapshot)
        self.assertFalse(report["replaced"])
        self.assertEqual(report["changed_entries"], [])
        self.assertEqual(report["changed_views"], [])

    def test_restore_diverged_wholesale_replacement(self):
        diverged = make_session()
        diverged.apply_changeset({"id": "csX", "ops": [
            {"op": "set_edge_weight", "edge": "e2", "weight": 99},
        ]})
        before_digest = diverged.graph.content_digest()
        report = diverged.restore(self.snapshot)
        self.assertTrue(report["replaced"])
        self.assertNotEqual(before_digest, diverged.graph.content_digest())
        self.assertEqual(diverged.graph.content_digest(), self.session.graph.content_digest())
        # 变化清单稳定且含新旧版本
        self.assertTrue(report["changed_entries"])
        for item in report["changed_entries"]:
            self.assertIn("old_version", item)
            self.assertIn("new_version", item)
        keys = [(i["index"], i["key"]) for i in report["changed_entries"]]
        self.assertEqual(keys, sorted(keys))
        # 恢复后条目状态与快照逐条一致
        snap = json.loads(self.snapshot.read_text(encoding="utf-8"))
        for name, entries in snap["entries"].items():
            for key, stored in entries.items():
                entry = diverged.indexes.entries[name][key]
                self.assertEqual(entry.entry_version, stored["entry_version"])
                self.assertEqual(entry.digest, stored["digest"])

    def test_tampered_entry_digest_raises_e_index(self):
        snap = json.loads(self.snapshot.read_text(encoding="utf-8"))
        snap["entries"]["by_label"]["Person"]["digest"] = "sha256:" + "0" * 64
        self.snapshot.write_text(json.dumps(snap), encoding="utf-8")
        with self.assertRaises(IndexIntegrityError) as ctx:
            MaintenanceSession.from_snapshot(self.snapshot)
        self.assertEqual(ctx.exception.code, "E_INDEX")
        self.assertEqual(ctx.exception.details[0]["element"], "by_label:Person")
        self.assertEqual(ctx.exception.details[0]["field"], "digest")


class TestReplay(SnapshotCase):
    def test_replay_ok(self):
        report = replay_snapshot(str(self.snapshot))
        self.assertEqual(report["status"], "ok")
        self.assertGreater(report["checks"], 0)

    def test_replay_detects_tampered_trace(self):
        snap = json.loads(self.snapshot.read_text(encoding="utf-8"))
        snap["invalidation_trace"][0]["affected_entries"] = [["by_label", "City"]]
        self.snapshot.write_text(json.dumps(snap), encoding="utf-8")
        with self.assertRaises(ReplayError) as ctx:
            replay_snapshot(str(self.snapshot))
        err = ctx.exception
        self.assertEqual(err.code, "E_REPLAY")
        self.assertEqual(err.details[0]["element"], "trace:cs1")
        self.assertEqual(err.details[0]["field"], "affected_entries")

    def test_replay_detects_tampered_entry_version(self):
        snap = json.loads(self.snapshot.read_text(encoding="utf-8"))
        # 修改条目版本（不动摘要，绕过完整性校验，由重放比对发现）
        entry = snap["entries"]["by_weight"]["1"]
        entry["entry_version"] += 10
        self.snapshot.write_text(json.dumps(snap), encoding="utf-8")
        with self.assertRaises(ReplayError) as ctx:
            replay_snapshot(str(self.snapshot))
        self.assertEqual(ctx.exception.details[0]["element"], "by_weight:1")
        self.assertEqual(ctx.exception.details[0]["field"], "entry_version")

    def test_replay_detects_tampered_ledger(self):
        snap = json.loads(self.snapshot.read_text(encoding="utf-8"))
        snap["recompute_ledger"][0]["target"] = "by_label:City"
        self.snapshot.write_text(json.dumps(snap), encoding="utf-8")
        with self.assertRaises(ReplayError) as ctx:
            replay_snapshot(str(self.snapshot))
        self.assertEqual(ctx.exception.details[0]["field"], "target")


if __name__ == "__main__":
    unittest.main()
