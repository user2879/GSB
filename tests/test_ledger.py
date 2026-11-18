from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from quota_app.codec import parse_document
from quota_app.derive import resolve_effective
from quota_app.errors import E_FORMAT, E_REPLAY, ConfigFormatError
from quota_app.ledger import Ledger

from .helpers import document, document_raw, event_records, write_events


class LedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = self.root / "ledger.sqlite3"
        self.events_path = write_events(self.root / "events.ndjson")
        from quota_app.events import load_events

        self.events = load_events(self.events_path)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_ingest_batches_persist_and_replay(self) -> None:
        doc = document(3)
        with Ledger(self.db) as ledger:
            ledger.register_config(doc)
            report = ledger.ingest_events(doc, self.events, batch_size=3)
            self.assertEqual(report.batches_committed, 4)
            self.assertEqual(report.judgments_written, 10)
            self.assertEqual(report.decisions, {"ALLOW": 6, "RATE_LIMITED": 2, "OVER_QUOTA": 2})
            replay = ledger.verify_replay()
            self.assertTrue(replay.ok)
            self.assertEqual(replay.judgments_compared, 10)
            batches = ledger.connection.execute(
                "SELECT status, COUNT(*) AS count FROM event_batches GROUP BY status"
            ).fetchall()
            self.assertEqual([(row["status"], row["count"]) for row in batches], [("committed", 4)])
            progress = ledger.current_progress()
            self.assertEqual(progress["next_index"], 10)
            view = ledger.current_view()
            self.assertEqual(len(view["judgments"]), 10)
            self.assertEqual(view["error_bounds"], {"single": 0, "aggregate": 0})

    def test_failed_batch_rolls_back_results_usage_cursor_and_inputs(self) -> None:
        doc = document(3)
        calls = {"count": 0}

        def crash_hook(stage: str, batch_id: int) -> None:
            if stage == "before_commit":
                calls["count"] += 1
                raise RuntimeError("simulated forced termination before commit")

        with Ledger(self.db) as ledger:
            ledger.register_config(doc)
            with self.assertRaises(RuntimeError):
                ledger.ingest_events(doc, self.events, batch_size=3, crash_hook=crash_hook)
            # The pending batch row survives, but all transactional data rolls back.
            pending = ledger.connection.execute(
                "SELECT status FROM event_batches ORDER BY batch_id"
            ).fetchall()
            self.assertEqual([row["status"] for row in pending], ["pending"])
            for table in ("judgments", "current_judgments", "event_inputs", "current_usage"):
                count = ledger.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                self.assertEqual(count, 0, table)
            progress = ledger.current_progress()
            self.assertEqual(progress["next_index"], 0)

            recovery = ledger.recover_incomplete()
            self.assertEqual(len(recovery.unknown_batches), 1)
            self.assertFalse(recovery.exit_report_available)
            self.assertTrue(recovery.rerun_required)
            status = ledger.connection.execute(
                "SELECT status FROM event_batches ORDER BY batch_id"
            ).fetchone()["status"]
            self.assertEqual(status, "unknown")

            rerun = ledger.ingest_events(doc, self.events, batch_size=3)
            self.assertEqual(rerun.judgments_written, 10)
            self.assertTrue(ledger.verify_replay().ok)

    def test_crash_after_committed_batch_preserves_prior_batch(self) -> None:
        doc = document(3)
        crash_on_second = {"seen": 0}

        def hook(stage: str, batch_id: int) -> None:
            if stage == "before_commit":
                crash_on_second["seen"] += 1
                if crash_on_second["seen"] == 2:
                    raise RuntimeError("simulated termination in second batch")

        with Ledger(self.db) as ledger:
            ledger.register_config(doc)
            with self.assertRaises(RuntimeError):
                ledger.ingest_events(doc, self.events, batch_size=3, crash_hook=hook)
            progress = ledger.current_progress()
            self.assertEqual(progress["next_index"], 3)
            self.assertEqual(
                ledger.connection.execute(
                    "SELECT COUNT(*) FROM current_judgments"
                ).fetchone()[0],
                3,
            )
            recovery = ledger.recover_incomplete()
            self.assertEqual(len(recovery.unknown_batches), 1)
            rerun = ledger.ingest_events(doc, self.events, batch_size=3)
            self.assertEqual(rerun.judgments_written, 7)
            replay = ledger.verify_replay()
            self.assertTrue(replay.ok)
            self.assertEqual(replay.judgments_compared, 10)

    def test_config_revision_replaces_current_view_and_marks_stale(self) -> None:
        doc3 = document(3)
        with Ledger(self.db) as ledger:
            ledger.register_config(doc3)
            ledger.ingest_events(doc3, self.events, batch_size=5)
            old_judgments = len(ledger.load_current_judgments(3))
            self.assertEqual(old_judgments, 10)

            raw = document_raw(4)
            raw["volumes"][1]["capacity_bytes"] = 800
            raw["volumes"][3]["ops_per_sec"] = 9
            doc4 = parse_document(raw)
            replaced = ledger.register_config(doc4)
            self.assertTrue(replaced)
            self.assertEqual(ledger.active_revision(), 4)
            self.assertEqual(ledger.load_current_judgments(4), [])
            self.assertEqual(ledger.load_current_usage(4), {})
            self.assertEqual(ledger.current_progress()["next_index"], 0)
            stale_count = ledger.connection.execute(
                "SELECT COUNT(*) FROM judgments WHERE config_revision = 3 AND stale = 1"
            ).fetchone()[0]
            self.assertEqual(stale_count, 10)
            versions = ledger.list_config_versions()
            self.assertEqual([(row["revision"], row["stale"]) for row in versions], [(3, 1), (4, 0)])
            replacements = ledger.list_replacements()
            self.assertEqual(replacements[-1]["affected_volumes"], ["child", "mirror"])
            self.assertEqual(
                replacements[-1]["changed_fields"],
                [
                    {"volume_id": "child", "fields": ["capacity_bytes", "effective_capacity_bytes"]},
                    {"volume_id": "mirror", "fields": ["ops_per_sec"]},
                ],
            )

            ledger.ingest_events(doc4, self.events, batch_size=4)
            self.assertTrue(ledger.verify_replay().ok)
            self.assertEqual(len(ledger.load_current_judgments(4)), 10)

    def test_revision_reuse_with_different_content_is_rejected(self) -> None:
        doc = document(3)
        with Ledger(self.db) as ledger:
            ledger.register_config(doc)
            raw = document_raw(3)
            raw["volumes"][0]["capacity_bytes"] = 999
            changed = parse_document(raw)
            with self.assertRaises(ConfigFormatError) as context:
                ledger.register_config(changed)
            self.assertEqual(context.exception.code, E_FORMAT)

    def test_unknown_event_volume_is_rejected_before_batch(self) -> None:
        doc = document(3)
        from quota_app.models import AccessEvent

        bad_events = list(self.events)
        bad_events[0] = AccessEvent(1, 0, "missing", 1, 1)
        with Ledger(self.db) as ledger:
            ledger.register_config(doc)
            with self.assertRaises(ConfigFormatError) as context:
                ledger.ingest_events(doc, bad_events, batch_size=3)
            self.assertEqual(context.exception.code, E_FORMAT)
            self.assertEqual(context.exception.volume_id, "missing")
            count = ledger.connection.execute("SELECT COUNT(*) FROM event_batches").fetchone()[0]
            self.assertEqual(count, 0)

    def test_tampered_replay_is_e_replay_with_basis(self) -> None:
        doc = document(3)
        with Ledger(self.db) as ledger:
            ledger.register_config(doc)
            ledger.ingest_events(doc, self.events, batch_size=5)
            ledger.connection.execute(
                "UPDATE current_judgments SET decision = 'OVER_QUOTA' WHERE seq = 1"
            )
            ledger.connection.commit()
            replay = ledger.verify_replay()
            self.assertFalse(replay.ok)
            self.assertEqual(replay.errors[0].code, E_REPLAY)
            self.assertEqual(replay.errors[0].volume_id, "root")
            self.assertIn("decision", replay.errors[0].details["difference_fields"])

    def test_tampered_usage_snapshot_is_e_replay(self) -> None:
        doc = document(3)
        with Ledger(self.db) as ledger:
            ledger.register_config(doc)
            ledger.ingest_events(doc, self.events, batch_size=5)
            ledger.connection.execute(
                "UPDATE current_usage SET cumulative_write_bytes = 9999 WHERE volume_id = 'root'"
            )
            ledger.connection.commit()
            replay = ledger.verify_replay()
            self.assertFalse(replay.ok)
            self.assertEqual(replay.errors[0].code, E_REPLAY)
            self.assertEqual(replay.errors[0].volume_id, "root")
            self.assertIn(
                "cumulative_write_bytes", replay.errors[0].details["difference_fields"]
            )

            ledger.connection.execute("DELETE FROM current_usage")
            ledger.connection.commit()
            with self.assertRaises(Exception) as context:
                ledger.ingest_events(doc, self.events, batch_size=5)
            self.assertEqual(getattr(context.exception, "code", None), E_REPLAY)

    def test_upgrade_checkpoint_recorded_in_ledger(self) -> None:
        payload = {"checkpoint_version": 1, "entries": {"x.json": {"status": "completed"}}}
        with Ledger(self.db) as ledger:
            ledger.record_upgrade_checkpoint(payload)
            row = ledger.connection.execute(
                "SELECT payload_json FROM upgrade_checkpoints ORDER BY checkpoint_id DESC LIMIT 1"
            ).fetchone()
            self.assertEqual(json.loads(row["payload_json"]), payload)


if __name__ == "__main__":
    unittest.main()
