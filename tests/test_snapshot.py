from __future__ import annotations

import json
import tempfile
import unittest

from mvcc_workbench.engine import DiagnosticEngine
from mvcc_workbench.errors import SnapshotError
from mvcc_workbench.loader import load_spec
from mvcc_workbench.snapshot import (
    create_snapshot,
    load_snapshot,
    restore_snapshot,
    save_snapshot,
    session_identity,
    verify_snapshot_replay,
)


class SnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = load_spec("data/demo_large.json")
        self.round = DiagnosticEngine(self.spec).run(verify_consistency=False)

    def test_save_load_and_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/snapshot.json"
            snapshot = create_snapshot(
                self.spec,
                self.round.plan.to_dict(),
                self.round.results,
                ledger_cursor=9,
                snapshot_id="snap-1",
            )
            saved = save_snapshot(path, snapshot)
            loaded = load_snapshot(path)
            self.assertEqual(saved["content_digest"], loaded.content_digest)
            verification = verify_snapshot_replay(loaded)
            self.assertTrue(verification.ok)
            expected = session_identity(self.spec, self.round.results, ledger_cursor=9)
            restored, replay = restore_snapshot(path, expected_identity=expected)
            self.assertEqual(restored.snapshot_id, "snap-1")
            self.assertTrue(replay.ok)

    def test_digest_mismatch_reports_e_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/snapshot.json"
            snapshot = create_snapshot(
                self.spec,
                self.round.plan.to_dict(),
                self.round.results,
                ledger_cursor=0,
                snapshot_id="snap-tamper",
            )
            save_snapshot(path, snapshot)
            with open(path, encoding="utf-8") as handle:
                value = json.load(handle)
            value["results"][0]["visible_version"] = 999999
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(value, handle)
            with self.assertRaises(SnapshotError) as raised:
                load_snapshot(path)
            self.assertEqual(raised.exception.code, "E_SNAPSHOT")
            self.assertEqual(
                raised.exception.details["anomaly"], "SNAPSHOT_DIGEST_MISMATCH"
            )

    def test_identity_mismatch_requires_whole_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/snapshot.json"
            snapshot = create_snapshot(
                self.spec,
                self.round.plan.to_dict(),
                self.round.results,
                ledger_cursor=2,
                snapshot_id="snap-replace",
            )
            save_snapshot(path, snapshot)
            wrong = session_identity(self.spec, self.round.results, ledger_cursor=3)
            with self.assertRaises(SnapshotError) as raised:
                restore_snapshot(path, expected_identity=wrong)
            self.assertEqual(raised.exception.details["action"], "REPLACE_ALL")
            replaced, replay = restore_snapshot(
                path, expected_identity=wrong, force_replace=True
            )
            self.assertEqual(replaced.snapshot_id, "snap-replace")
            self.assertEqual(replaced.identity.ledger_cursor, 2)
            self.assertTrue(replay.ok)


if __name__ == "__main__":
    unittest.main()
