import tempfile
import unittest
from pathlib import Path

from configlease.engine import (
    SNAPSHOT_VERSION,
    ReplayMismatchError,
    VersionConflictError,
    assert_snapshot_compatible,
    content_hash,
    replay_lease_records,
)
from configlease.manifest import Revision
from configlease.storage import Store


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / "db.sqlite3")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_revision_and_cursor_roundtrip(self):
        rev = {"rev_seq": 1, "parent": 0, "changes": {"a": 1}, "kind": "normal"}
        self.store.upsert_revision(rev)
        self.store.put_cursor("n1", 1, {"a": 1})
        self.assertEqual(self.store.list_revisions()[0]["rev_seq"], 1)
        self.assertEqual(self.store.list_cursors()["n1"]["applied_rev"], 1)

    def test_lease_lifecycle_and_active_query(self):
        lease = {
            "lease_id": "L1", "request_id": "R1", "node_id": "n1", "rev_seq": 1,
            "low_seq": 1, "high_seq": 1, "keys": ["a"],
            "issued_ms": 0, "deadline_ms": 10,
        }
        self.store.put_lease(lease, "active")
        self.assertEqual(len(self.store.active_leases()), 1)
        self.store.update_lease_status("L1", "applied")
        self.assertEqual(len(self.store.active_leases()), 0)

    def test_event_retrieval_by_node_and_rev(self):
        self.store.insert_event({"ts": 1.0, "session_id": "s", "event": "e",
                                 "node_id": "n1", "rev_seq": 2, "code": "E_REV"})
        self.store.insert_event({"ts": 2.0, "session_id": "s", "event": "e",
                                 "node_id": "n2", "rev_seq": 2, "code": "E_KEY"})
        self.assertEqual(len(self.store.query_events(rev_seq=2)), 2)
        self.assertEqual(len(self.store.query_events(node_id="n1")), 1)
        self.assertEqual(self.store.query_events(code="E_KEY")[0]["node_id"], "n2")

    def test_snapshot_save_and_load(self):
        state = {"x": 1}
        digest = content_hash(state)
        self.store.save_snapshot("snap", SNAPSHOT_VERSION, digest, state, 123)
        loaded = self.store.load_snapshot("snap")
        self.assertEqual(loaded["content_hash"], digest)
        self.assertEqual(loaded["state"], state)


class SnapshotReplayTests(unittest.TestCase):
    def _revisions(self):
        return [
            Revision(1, 0, {"a": 1}),
            Revision(2, 1, {"b": 2}),
        ]

    def _state(self):
        return {
            "superseded": [],
            "leases": [
                {"lease_id": "L1", "node_id": "n1", "rev_seq": 1,
                 "low_seq": 1, "high_seq": 1, "keys": ["a"], "status": "applied"},
                {"lease_id": "L2", "node_id": "n1", "rev_seq": 2,
                 "low_seq": 2, "high_seq": 2, "keys": ["b"], "status": "applied"},
            ],
            "cursors": {"n1": {"applied_rev": 2, "view": {"a": 1, "b": 2}}},
            "views": {"n1": {"a": 1, "b": 2}},
            "budgets": {
                "global": {"limit": 8, "consumed": 0},
                "nodes": {"n1": {"limit": 4, "consumed": 2}},
            },
        }

    def test_replay_matches_snapshot(self):
        out = replay_lease_records(self._state()["leases"], self._revisions(),
                                   self._state())
        self.assertEqual(out["cursors"]["n1"], 2)
        self.assertEqual(out["node_consumed"]["n1"], 2)

    def test_replay_detects_cursor_divergence(self):
        state = self._state()
        state["cursors"]["n1"]["applied_rev"] = 99
        with self.assertRaises(ReplayMismatchError) as ctx:
            replay_lease_records(state["leases"], self._revisions(), state)
        self.assertEqual(ctx.exception.code, "E_REPLAY")
        self.assertEqual(ctx.exception.context["field"], "applied_rev")

    def test_version_conflict_detected(self):
        state = self._state()
        snapshot = {"version": SNAPSHOT_VERSION,
                    "content_hash": content_hash(state)}
        with self.assertRaises(VersionConflictError):
            assert_snapshot_compatible(snapshot, {"different": True})


if __name__ == "__main__":
    unittest.main()
