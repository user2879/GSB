from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gsb.lease import LeaseManager
from gsb.snapshot import SessionSnapshot, ViewStore


class SnapshotViewTests(unittest.TestCase):
    def test_atomic_save_load_identifies_format(self) -> None:
        snapshot = SessionSnapshot(
            capacity_version=7,
            spec_digest="abc",
            watermarks={"budget_q": 1000},
            decisions=[{"shard_id": "s1"}],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.json"
            snapshot.save_atomic(path)
            loaded = SessionSnapshot.load(path)
        self.assertEqual(loaded.snapshot_format, 1)
        self.assertEqual(loaded.capacity_version, 7)
        self.assertEqual(loaded.snapshot_digest, snapshot.snapshot_digest)

    def test_capacity_bump_replaces_entire_view_and_marks_old_stale(self) -> None:
        store = ViewStore()
        old = store.replace(SessionSnapshot(capacity_version=1, spec_digest="old"))
        new = store.replace(SessionSnapshot(capacity_version=2, spec_digest="new"))

        self.assertTrue(old.stale)
        self.assertFalse(new.stale)
        visible = store.visible_view()
        self.assertEqual(visible["capacity_version"], 2)
        self.assertEqual(visible["spec_digest"], "new")
        self.assertEqual(len(store.stale_generations()), 1)

    def test_capacity_revocation_and_expiry_release_once(self) -> None:
        manager = LeaseManager(ttl_ms=10_000)
        old = manager.grant(
            shard_id="old",
            node_id="n1",
            capacity_version=1,
            resource_units=4,
            now_ms=0,
        )
        manager.grant(
            shard_id="new",
            node_id="n1",
            capacity_version=2,
            resource_units=4,
            now_ms=0,
        )
        revoked = manager.revoke_old_versions({"n1": 2})
        self.assertEqual([lease.shard_id for lease in revoked], ["old"])
        expiry = manager.expire_due(now_ms=20_000)
        expired_ids = {lease.shard_id for lease in expiry}
        self.assertNotIn("old", expired_ids)
        self.assertIn("new", expired_ids)
        first = manager.release_once(old.lease_id, "capacity_revoked", status="revoked")
        second = manager.release_once(old.lease_id, "expired", status="expired")
        self.assertIsNotNone(first)
        self.assertIsNone(second)


if __name__ == "__main__":
    unittest.main()
