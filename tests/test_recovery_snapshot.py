from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from txnmsg.content import snapshot_digest
from txnmsg.errors import SnapshotError
from txnmsg.replay_verify import replay_verify
from txnmsg.snapshots import (
    create_snapshot,
    list_snapshots,
    load_snapshot,
    next_snapshot_path,
    restore_snapshot,
)
from txnmsg.store import Ledger

from tests.helpers import extra_transaction, make_ledger


class SimulatedCrash(RuntimeError):
    pass


def crash_hook():
    raise SimulatedCrash("forced termination boundary")


class RecoverySnapshotTests(unittest.TestCase):
    def test_stale_prefix_cache_is_repaired_after_commit_event_crash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.sqlite"
            ledger = make_ledger(path)
            try:
                ledger.advance_state("t1", "prepared")
                ledger.advance_state("t1", "committing")
                with self.assertRaises(SimulatedCrash):
                    ledger.advance_state("t1", "committed", crash_hook=crash_hook)
            finally:
                ledger.close()
            with Ledger(path, auto_recover=False, readonly=True) as readonly:
                stale = readonly.persisted_prefix_rows()["orders"]
                self.assertEqual(stale["computed_event_seq"], 2)
                self.assertEqual(stale["prefix"], -1)
            with Ledger(path) as reopened:
                self.assertIsNotNone(reopened.last_recovery)
                self.assertTrue(reopened.last_recovery.persisted_prefix_repaired)
                self.assertEqual(reopened.query_prefixes()[0].prefix, 1)
                self.assertFalse(reopened.last_recovery.as_dict()["exit_report_available"])

    def test_committing_crash_is_unknown_and_excluded_from_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.sqlite"
            ledger = make_ledger(path)
            try:
                ledger.advance_state("t1", "prepared")
                ledger.advance_state("t1", "committing")
                ledger.advance_state("t1", "committed")
                ledger.advance_state("t2", "prepared")
                with self.assertRaises(SimulatedCrash):
                    ledger.advance_state("t2", "committing", crash_hook=crash_hook)
            finally:
                ledger.close()
            with Ledger(path) as reopened:
                report = reopened.last_recovery
                self.assertIsNotNone(report)
                self.assertEqual(len(report.unknown_transactions), 1)
                unknown = report.unknown_transactions[0]
                self.assertEqual(unknown.transaction_id, "t2")
                self.assertEqual(unknown.partition_id, "orders")
                self.assertEqual(unknown.intervals, ((2, 3),))
                orders = reopened.query_prefixes()[0]
                self.assertEqual(orders.prefix, 1)
                self.assertEqual(orders.truncated_position, 2)
                self.assertEqual(orders.truncated_reason, "unknown")
                self.assertEqual(orders.unknown_intervals, ((2, 3),))

                snapshot = create_snapshot(reopened, next_snapshot_path(directory))
                replay = replay_verify(reopened, snapshot.path)
                self.assertEqual(replay.status, "PASS")
                self.assertEqual(
                    [item.transaction_id for item in replay.unknown_transactions], ["t2"]
                )

                reopened.advance_state("t2", "committed")
                self.assertEqual(reopened.unknown_transactions(), ())
                self.assertEqual(reopened.query_prefixes()[0].prefix, 3)

    def test_snapshot_restore_is_wholesale_and_backup_is_readonly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "ledger.sqlite"
            ledger = make_ledger(database)
            try:
                ledger.advance_many(
                    [
                        ("t1", "prepared"),
                        ("t1", "committing"),
                        ("t1", "committed"),
                        ("t2", "prepared"),
                        ("t2", "committing"),
                        ("t2", "committed"),
                    ]
                )
                snapshot = create_snapshot(ledger, next_snapshot_path(root))
                snapshot_digest_value = ledger.computed_ledger_digest()
                extra = extra_transaction()
                ledger.ingest(extra)
                ledger.advance_many(
                    [
                        ("t4", "prepared"),
                        ("t4", "committing"),
                        ("t4", "committed"),
                    ]
                )
                pre_restore_digest = ledger.computed_ledger_digest()
                self.assertNotEqual(snapshot_digest_value, pre_restore_digest)
                self.assertEqual(ledger.query_prefixes()[0].prefix, 4)

                restored = restore_snapshot(ledger, snapshot.path)
                self.assertTrue(restored.replaced)
                self.assertEqual(restored.snapshot_ledger_digest, snapshot_digest_value)
                self.assertEqual(restored.prior_ledger_digest, pre_restore_digest)
                self.assertIn("t4", restored.diff.transaction_removed)
                self.assertEqual(ledger.query_prefixes()[0].prefix, 3)
                self.assertNotIn("t4", [t.transaction_id for t in ledger.transactions()])
                self.assertEqual(ledger.computed_ledger_digest(), snapshot_digest_value)
                self.assertIsNotNone(restored.backup_path)
                backup = Path(restored.backup_path)
                self.assertTrue(backup.exists())
                self.assertEqual(backup.stat().st_mode & 0o222, 0)

                with Ledger(backup, auto_recover=False, readonly=True) as backup_ledger:
                    self.assertIn("t4", [t.transaction_id for t in backup_ledger.transactions()])
                    self.assertEqual(backup_ledger.computed_ledger_digest(), pre_restore_digest)

                replay = replay_verify(ledger, snapshot.path)
                self.assertEqual(replay.status, "PASS")
                names = [item.name for item in list_snapshots(root)]
                self.assertEqual(names, sorted(names))
                os.chmod(backup, 0o644)
            finally:
                ledger.close()

    def test_snapshot_ledger_digest_mismatch_reports_e_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = make_ledger(root / "ledger.sqlite")
            try:
                ledger.advance_state("t1", "prepared")
                snapshot = create_snapshot(ledger, next_snapshot_path(root))
            finally:
                ledger.close()
            payload = json.loads(Path(snapshot.path).read_text(encoding="utf-8"))
            payload["ledger_digest"] = "sha256:" + "0" * 64
            payload["content_digest"] = snapshot_digest(payload)
            tampered = root / "tampered.json"
            tampered.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(SnapshotError) as raised:
                load_snapshot(tampered)
            self.assertEqual(raised.exception.code, "E_SNAPSHOT")
            self.assertEqual(raised.exception.field, "ledger_digest")

    def test_snapshot_content_digest_mismatch_reports_e_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = make_ledger(root / "ledger.sqlite")
            try:
                ledger.advance_state("t1", "prepared")
                snapshot = create_snapshot(ledger, next_snapshot_path(root))
            finally:
                ledger.close()
            payload = json.loads(Path(snapshot.path).read_text(encoding="utf-8"))
            payload["created_at"] = "tampered"
            tampered = root / "tampered-content.json"
            tampered.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(SnapshotError) as raised:
                load_snapshot(tampered)
            self.assertEqual(raised.exception.field, "content_digest")

    def test_snapshot_versions_and_event_cursor_are_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = make_ledger(root / "ledger.sqlite")
            try:
                ledger.advance_state("t1", "prepared")
                ledger.advance_state("t1", "committing")
                ledger.advance_state("t1", "committed")
                snapshot = create_snapshot(ledger, next_snapshot_path(root))
                self.assertEqual(snapshot.event_cursor, 3)
                self.assertEqual(snapshot.partition_set_version, 1)
                self.assertEqual(snapshot.transaction_set_version, 1)
                self.assertEqual(len(snapshot.prefixes), 2)
                payload = json.loads(Path(snapshot.path).read_text(encoding="utf-8"))
                self.assertEqual(payload["event_cursor"], 3)
                self.assertTrue(payload["ledger_digest"].startswith("sha256:"))
                self.assertTrue(payload["content_digest"].startswith("sha256:"))
            finally:
                ledger.close()


if __name__ == "__main__":
    unittest.main()
