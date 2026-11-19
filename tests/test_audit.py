from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from batch_audit.audit import AuditMismatch, run_audit
from batch_audit.engine import Scheduler, WorkspacePaths

from .helpers import load_json, make_workspace


class AuditTests(unittest.TestCase):
    def _complete_workspace(self, root: Path) -> WorkspacePaths:
        ws, _ = make_workspace(root)
        paths = WorkspacePaths.under(ws)
        Scheduler(paths).run()
        return paths

    def test_clean_audit_digest_is_repeatable(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            paths = self._complete_workspace(Path(temp))
            first = run_audit(ledger_path=paths.ledger, feed_path=paths.feed)
            second = run_audit(ledger_path=paths.ledger, feed_path=paths.feed)
            self.assertEqual(first["recomputed_digest"], second["recomputed_digest"])
            self.assertEqual(first["comparisons"]["assignments_compared"], 10)

    def test_tampered_output_digest_reports_sample_slot_and_field(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            paths = self._complete_workspace(Path(temp))
            with sqlite3.connect(paths.ledger) as conn:
                conn.execute(
                    "UPDATE assignments SET sample_output_digest = 'tampered' "
                    "WHERE batch_slot = 2 AND rank = 1"
                )
            with self.assertRaises(AuditMismatch) as caught:
                run_audit(ledger_path=paths.ledger, feed_path=paths.feed)
            mismatch = caught.exception.as_dict()["first_mismatch"]
            self.assertEqual(mismatch["batch_slot"], 2)
            self.assertEqual(mismatch["field"], "sample_output_digest")
            self.assertIsNotNone(mismatch["sample_id"])
            self.assertNotEqual(mismatch["expected"], mismatch["actual"])

    def test_feed_tamper_is_detected_before_assignment_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            paths = self._complete_workspace(Path(temp))
            feed = load_json(paths.feed)
            for entry in feed["entries"]:
                if not entry["admitted"]:
                    entry["reject_reason"] = "changed_reason"
                    break
            paths.feed.write_text(
                __import__("json").dumps(feed, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(AuditMismatch) as caught:
                run_audit(ledger_path=paths.ledger, feed_path=paths.feed)
            mismatch = caught.exception.as_dict()["first_mismatch"]
            self.assertEqual(mismatch["field"], "feed_digest")
            self.assertIsNone(mismatch["sample_id"])

    def test_ledger_content_digest_tamper_is_reported_after_business_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            paths = self._complete_workspace(Path(temp))
            with sqlite3.connect(paths.ledger) as conn:
                conn.execute(
                    "UPDATE ledger_header SET content_digest = 'broken' WHERE singleton = 1"
                )
            with self.assertRaises(AuditMismatch) as caught:
                run_audit(ledger_path=paths.ledger, feed_path=paths.feed)
            mismatch = caught.exception.as_dict()["first_mismatch"]
            self.assertEqual(mismatch["field"], "ledger_content_digest")

    def test_array_data_tamper_is_detected_from_embedded_ledger_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            paths = self._complete_workspace(Path(temp))
            with sqlite3.connect(paths.ledger) as conn:
                raw = conn.execute(
                    "SELECT data FROM arrays WHERE name = 'W1' LIMIT 1"
                ).fetchone()[0]
                changed = bytes([raw[0] ^ 0x01]) + raw[1:]
                conn.execute("UPDATE arrays SET data = ? WHERE name = 'W1'", (changed,))
            with self.assertRaises(AuditMismatch) as caught:
                run_audit(ledger_path=paths.ledger, feed_path=paths.feed)
            mismatch = caught.exception.as_dict()["first_mismatch"]
            self.assertEqual(mismatch["field"], "arrays[W1].digest")


if __name__ == "__main__":
    unittest.main()
