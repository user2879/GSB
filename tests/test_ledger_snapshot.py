from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from quota_workbench.canonical import make_writable, write_canonical
from quota_workbench.errors import WorkbenchError
from quota_workbench.snapshot import load_snapshot, restore_snapshot, save_snapshot
from quota_workbench.session import WorkbenchSession
from tests.helpers import ROOT, SAMPLES, changed_spec, sample_spec


class LedgerSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="quota-workbench-test-")
        self.base = Path(self.temp.name)

    def tearDown(self) -> None:
        for child in sorted(self.base.rglob("*"), reverse=True):
            try:
                make_writable(child)
            except OSError:
                pass
        self.temp.cleanup()

    def open_session(self, name: str = "ledger.sqlite3", events: str = "normal") -> WorkbenchSession:
        return WorkbenchSession.open(
            SAMPLES / "spec.json", SAMPLES / f"{events}.ndjson", self.base / name
        )

    def test_ledger_commits_and_replays(self) -> None:
        session = self.open_session()
        try:
            results = session.run_all()
            self.assertEqual(len(results), 4)
            self.assertEqual(session.cursor, 4)
            self.assertEqual(len(session.ledger.event_rows()), 4)
            self.assertEqual(len(session.ledger.preemptions_for_event(1)), 1)
            report = session.verify_replay()
            self.assertEqual(report["status"], "REPLAY_OK")
            self.assertEqual(report["event_count"], 4)
            self.assertEqual(report["head_digest"], session.ledger.head_digest)
            ledger_info = session.ledger.conn.execute(
                "SELECT format_version FROM ledger_info WHERE singleton = 1"
            ).fetchone()
            self.assertEqual(ledger_info["format_version"], 1)
        finally:
            session.close()

    def test_tampered_normalized_matrix_reports_first_replay_location(self) -> None:
        session = self.open_session()
        try:
            session.run_all()
            session.ledger.conn.execute(
                "UPDATE tenant_matrices SET entitlement_units = entitlement_units + 1 "
                "WHERE event_id = 1 AND iteration = 0 AND tenant_id = 'tenant-alpha'"
            )
            with self.assertRaises(WorkbenchError) as caught:
                session.verify_replay()
            error = caught.exception
            self.assertEqual(error.code, "E_REPLAY")
            self.assertEqual(error.object_id, "1")
            self.assertEqual(error.details["tenant_id"], "tenant-alpha")
            self.assertIn("entitlement_units", error.field)
            self.assertEqual(error.details["first_inconsistent_event_id"], 1)
        finally:
            session.close()

    def test_tampered_iteration_and_preemption_indexes_are_detected(self) -> None:
        session = self.open_session("index-tamper.sqlite3")
        try:
            session.run_next()
            session.ledger.conn.execute(
                "UPDATE iterations SET terminal_reason = 'FORGED' "
                "WHERE event_id = 1 AND iteration = 1"
            )
            with self.assertRaises(WorkbenchError) as caught:
                session.verify_replay()
            self.assertEqual(caught.exception.code, "E_REPLAY")
            self.assertIn("iterations[1].terminal_reason", caught.exception.field)
            session.ledger.conn.execute(
                "UPDATE iterations SET terminal_reason = 'NO_EXCESS' "
                "WHERE event_id = 1 AND iteration = 1"
            )
            session.ledger.conn.execute(
                "UPDATE preemptions SET selection_basis_json = '{}' "
                "WHERE event_id = 1 AND sequence = 1"
            )
            with self.assertRaises(WorkbenchError) as caught:
                session.verify_replay()
            self.assertEqual(caught.exception.code, "E_REPLAY")
            self.assertIn("selection_basis", caught.exception.field)
        finally:
            session.close()

    def test_crash_after_prepare_marks_unknown_and_reruns(self) -> None:
        db_path = self.base / "crash.sqlite3"
        code = (
            "from quota_workbench.session import WorkbenchSession\n"
            f"s = WorkbenchSession.open({str(SAMPLES / 'spec.json')!r}, "
            f"{str(SAMPLES / 'normal.ndjson')!r}, {str(db_path)!r})\n"
            "s.run_next(crash_after_prepare=True)\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 137, completed.stderr)
        session = WorkbenchSession.open(
            SAMPLES / "spec.json", SAMPLES / "normal.ndjson", db_path
        )
        try:
            self.assertEqual(session.cursor, 0)
            recovery = session.recovery_report()
            self.assertEqual(recovery["status"], "UNKNOWN_REQUIRES_RERUN")
            self.assertEqual(recovery["unknown_event_ids"], [1])
            self.assertEqual(session.ledger.event_rows(), [])
            result = session.run_next()
            self.assertIsNotNone(result)
            self.assertEqual(session.cursor, 1)
            self.assertEqual(session.ledger.unknown_attempts(), [])
        finally:
            session.close()
        reopened = WorkbenchSession.open(
            SAMPLES / "spec.json", SAMPLES / "normal.ndjson", db_path
        )
        try:
            self.assertEqual(reopened.recovery_report()["status"], "CLEAN")
            self.assertEqual(reopened.cursor, 1)
        finally:
            reopened.close()

    def test_snapshot_restore_remaining_and_reference_match(self) -> None:
        session = self.open_session("snapshot-source.sqlite3", events="replay")
        restored_session = None
        reference = None
        try:
            session.run_next()
            session.run_next()
            snapshot_path = self.base / "snapshot.json"
            snapshot = save_snapshot(session, snapshot_path)
            self.assertEqual(snapshot.ledger_cursor, 2)
            self.assertFalse(snapshot_path.stat().st_mode & stat.S_IWUSR)
            with self.assertRaises(WorkbenchError) as caught:
                save_snapshot(session, snapshot_path)
            self.assertEqual(caught.exception.code, "E_SNAPSHOT")

            restored = restore_snapshot(
                snapshot_path,
                SAMPLES / "replay.ndjson",
                self.base / "restored.sqlite3",
                current_declaration_path=SAMPLES / "spec_context_changed.json",
            )
            restored_session = restored.session
            self.assertEqual(restored_session.cursor, 2)
            self.assertEqual(restored.change_report["mode"], "WHOLE_REPLACE")
            self.assertEqual(restored.change_report["affected_tenant_ids"], ["tenant-alpha"])
            self.assertIn("cluster_total_units", restored.change_report["changed_fields"])
            self.assertEqual(restored_session.verify_replay()["status"], "REPLAY_OK")
            while not restored_session.completed:
                self.assertIsNotNone(restored_session.run_next())

            reference = WorkbenchSession.open(
                SAMPLES / "spec.json", SAMPLES / "replay.ndjson", self.base / "reference.sqlite3"
            )
            reference.run_all()
            self.assertEqual(restored_session.ledger.head_digest, reference.ledger.head_digest)
            self.assertEqual(restored_session.state_digest(), reference.state_digest())
            self.assertEqual(restored_session.verify_replay()["status"], "REPLAY_OK")

            valid = json.loads(snapshot_path.read_text(encoding="utf-8"))
            valid["declaration"]["cluster_total_units"] += 1
            bad_path = self.base / "bad-snapshot.json"
            write_canonical(bad_path, valid)
            with self.assertRaises(WorkbenchError) as caught:
                load_snapshot(bad_path)
            self.assertEqual(caught.exception.code, "E_SNAPSHOT")
            self.assertEqual(caught.exception.field, "content_digest")
        finally:
            session.close()
            if restored_session is not None:
                restored_session.close()
            if reference is not None:
                reference.close()

    def test_snapshot_of_empty_ledger_can_be_restored(self) -> None:
        session = self.open_session("empty-source.sqlite3")
        try:
            snapshot = save_snapshot(session, self.base / "empty-snapshot.json")
            self.assertEqual(snapshot.ledger_cursor, 0)
        finally:
            session.close()
        restored = restore_snapshot(
            self.base / "empty-snapshot.json",
            SAMPLES / "normal.ndjson",
            self.base / "empty-restored.sqlite3",
        )
        try:
            self.assertEqual(restored.session.cursor, 0)
            self.assertIsNotNone(restored.session.run_next())
            self.assertEqual(restored.session.cursor, 1)
        finally:
            restored.session.close()

    def test_context_change_is_unreproducible_not_replay_failure(self) -> None:
        db_path = self.base / "context.sqlite3"
        session = self.open_session("context.sqlite3")
        try:
            session.run_next()
        finally:
            session.close()
        changed_path = self.base / "changed-spec.json"
        write_canonical(changed_path, changed_spec())
        with self.assertRaises(WorkbenchError) as caught:
            WorkbenchSession.open(changed_path, SAMPLES / "normal.ndjson", db_path)
        self.assertEqual(caught.exception.code, "E_CONTEXT")
        self.assertTrue(caught.exception.details["unreproducible"])

    def test_event_stream_change_is_unreproducible(self) -> None:
        db_path = self.base / "events.sqlite3"
        session = self.open_session("events.sqlite3")
        try:
            session.run_next()
        finally:
            session.close()
        changed_events = self.base / "changed.ndjson"
        lines = (SAMPLES / "normal.ndjson").read_text(encoding="utf-8").splitlines()
        lines[0] = lines[0].replace('"pressure_units":35', '"pressure_units":36')
        changed_events.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with self.assertRaises(WorkbenchError) as caught:
            WorkbenchSession.open(SAMPLES / "spec.json", changed_events, db_path)
        self.assertEqual(caught.exception.code, "E_CONTEXT")
        self.assertEqual(caught.exception.details["reason"], "EVENT_STREAM_CHANGED")


if __name__ == "__main__":
    unittest.main()
