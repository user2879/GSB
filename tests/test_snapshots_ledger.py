from __future__ import annotations

import json
import os
import stat
import unittest

from memdiag.errors import ReplayError, SnapshotError
from memdiag.ledger import Ledger
from memdiag.plan import apply_plan_changes
from memdiag.services import replay_session, run_full_session, run_incremental_session
from memdiag.snapshots import SnapshotStore
from tests.helpers import TemporaryDirectoryTestCase, fixture_plan, fixture_samples, read_fixture_json


class SnapshotAndLedgerTests(TemporaryDirectoryTestCase):
    def setUp(self):
        super().setUp()
        self.ledger = Ledger(self.temp_dir / "ledger.sqlite3")
        self.store = SnapshotStore(self.temp_dir / "snapshots")
        self.plan = fixture_plan()
        self.samples = fixture_samples()

    def tearDown(self):
        self.ledger.close()
        super().tearDown()

    def test_ledger_has_integer_format_and_content_digest(self):
        result = run_full_session(
            self.plan,
            self.samples,
            self.ledger,
            snapshot_store=self.store,
            save_snapshot=True,
            session_id="full-1",
        )
        meta = self.ledger.current_meta()
        self.assertEqual(meta["format_version"], 3)
        self.assertEqual(meta["content_digest"], result["run"].content_digest)
        self.assertEqual(meta["plan_version"], 1)
        layer_count = self.ledger.connection.execute(
            "SELECT COUNT(*) AS value FROM current_layers"
        ).fetchone()["value"]
        self.assertEqual(layer_count, len(result["run"].layers))

    def test_snapshot_is_read_only_and_restore_is_wholesale(self):
        first = run_full_session(
            self.plan,
            self.samples,
            self.ledger,
            snapshot_store=self.store,
            save_snapshot=True,
            snapshot_label="v1",
            session_id="full-v1",
        )
        v1_info = first["snapshot"]
        self.assertTrue(v1_info["read_only"])
        snapshot_dir_mode = os.stat(v1_info["path"]).st_mode
        snapshot_file_mode = os.stat(os.path.join(v1_info["path"], "snapshot.json")).st_mode
        self.assertEqual(snapshot_dir_mode & 0o222, 0)
        self.assertEqual(snapshot_file_mode & 0o222, 0)

        changes = read_fixture_json("changes_scale.json")
        new_plan, changed, _ = apply_plan_changes(self.plan, changes)
        second = run_incremental_session(
            new_plan,
            self.samples,
            first["run"],
            changed,
            self.ledger,
            snapshot_store=self.store,
            save_snapshot=True,
            snapshot_label="v2",
            session_id="inc-v2",
        )
        self.assertEqual(self.ledger.current_meta()["plan_version"], 2)
        restore = self.store.restore(v1_info["snapshot_id"], self.ledger)
        self.assertTrue(restore["replaced"])
        self.assertEqual(restore["replacement_mode"], "wholesale")
        self.assertFalse(restore["comparison"]["identical"])
        self.assertTrue(restore["backup_snapshot"]["read_only"])
        current = self.ledger.load_current_run()
        self.assertEqual(current.plan.plan_version, 1)
        self.assertEqual(current.content_digest, first["run"].content_digest)
        current_layer_ids = [
            row["layer_id"]
            for row in self.ledger.connection.execute(
                "SELECT layer_id FROM current_layers ORDER BY layer_index"
            ).fetchall()
        ]
        self.assertEqual(current_layer_ids, list(first["run"].layers))
        self.assertNotEqual(current.content_digest, second["run"].content_digest)

    def test_snapshot_tampering_is_detected(self):
        result = run_full_session(
            self.plan,
            self.samples,
            self.ledger,
            snapshot_store=self.store,
            save_snapshot=True,
            session_id="tamper-source",
        )
        path = result["snapshot"]["path"]
        snapshot_file = os.path.join(path, "snapshot.json")
        os.chmod(path, 0o755)
        os.chmod(snapshot_file, 0o644)
        with open(snapshot_file, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["content_digest"] = "0" * 64
        with open(snapshot_file, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        with self.assertRaises(SnapshotError):
            self.store.load(result["snapshot"]["snapshot_id"])

    def test_replay_detects_first_modified_commit_field(self):
        result = run_full_session(
            self.plan,
            self.samples,
            self.ledger,
            snapshot_store=self.store,
            save_snapshot=True,
            session_id="replay-source",
        )
        with self.ledger.transaction() as connection:
            connection.execute(
                "UPDATE layer_commits SET output_digest=? WHERE session_id=? AND layer_id=?",
                ("deadbeef" * 8, "replay-source", "branch_b"),
            )
        with self.assertRaises(ReplayError) as caught:
            replay_session(self.ledger, "replay-source")
        self.assertEqual(caught.exception.code, "E_REPLAY")
        self.assertEqual(caught.exception.details["layer_id"], "branch_b")
        self.assertEqual(caught.exception.details["field"], "output_digest")
        self.assertIn("basis", caught.exception.details)

    def test_snapshot_list_is_stable(self):
        first = run_full_session(
            self.plan,
            self.samples,
            self.ledger,
            snapshot_store=self.store,
            save_snapshot=True,
            snapshot_label="a",
            session_id="list-1",
        )
        changes = read_fixture_json("changes_scale.json")
        new_plan, changed, _ = apply_plan_changes(self.plan, changes)
        second = run_incremental_session(
            new_plan,
            self.samples,
            first["run"],
            changed,
            self.ledger,
            snapshot_store=self.store,
            save_snapshot=True,
            snapshot_label="b",
            session_id="list-2",
        )
        snapshots = self.store.list()
        self.assertEqual([item.plan_version for item in snapshots], [1, 2])
        self.assertEqual([item.snapshot_id for item in snapshots], sorted(item.snapshot_id for item in snapshots))
        self.assertEqual({item.content_digest for item in snapshots}, {first["run"].content_digest, second["run"].content_digest})


if __name__ == "__main__":
    unittest.main()
