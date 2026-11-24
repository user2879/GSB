from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from lateness_ledger.canonical import canonical_json, digest_object
from lateness_ledger.engine import run_engine
from lateness_ledger.errors import ReplayError
from lateness_ledger.ledger import (
    LedgerReader,
    LedgerWriter,
    create_snapshot,
    recover_unknown_windows,
    restore_snapshot,
)
from lateness_ledger.replay import replay_and_compare
from lateness_ledger.ui_model import load_dashboard
from tests.helpers import ROOT, sample_bundle


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LedgerReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="ledger-test-")
        cls.directory = Path(cls.temp.name)
        cls.ledger = cls.directory / "live.sqlite"
        cls.bundle = sample_bundle("source_max")
        with LedgerWriter(cls.ledger, reset_existing=True) as writer:
            cls.run_result = run_engine(cls.bundle, writer=writer)
        cls.snapshot = cls.directory / "snapshot.sqlite"
        cls.snapshot_record = create_snapshot(cls.ledger, cls.snapshot)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_per_window_ledger_and_summary(self) -> None:
        reader = LedgerReader(self.ledger)
        meta = reader.meta()
        self.assertEqual(meta["format_version"], 1)
        self.assertEqual(meta["run_status"], "complete")
        self.assertEqual(meta["event_cursor"], 359)
        windows = reader.windows()
        self.assertEqual(len(windows), 36)
        self.assertEqual([item["status"] for item in windows], ["finalized"] * 36)
        self.assertEqual(
            sorted(int(item["finalization_order"]) for item in windows),
            list(range(1, 37)),
        )
        evidence = reader.late_evidence()
        self.assertGreater(len(evidence), 0)
        self.assertTrue(
            all(item["evidence"]["window_id"] == item["window_id"] for item in evidence)
        )
        self.assertTrue(
            any(
                item["content"]["displacement_trajectory"]
                for item in windows
                if item["content"] is not None
            )
        )

    def test_replay_matches_every_window_and_watermark(self) -> None:
        report = replay_and_compare(self.bundle, str(self.ledger))
        self.assertEqual(report["result"], "MATCH")
        self.assertEqual(report["windows_checked"], 36)
        self.assertGreater(report["watermark_rows_checked"], 0)

    def test_replay_locates_first_field_with_partition_and_boundary(self) -> None:
        tampered = self.directory / "tampered.sqlite"
        shutil.copy2(self.snapshot, tampered)
        connection = sqlite3.connect(tampered)
        row = connection.execute(
            "SELECT window_id,content_json FROM windows "
            "ORDER BY finalization_order LIMIT 1"
        ).fetchone()
        payload = json.loads(row[1])
        payload["stats"]["count"] += 1
        new_content = canonical_json(payload)
        new_digest = digest_object(payload)
        connection.execute(
            "UPDATE windows SET content_json=?,digest=? WHERE window_id=?",
            (new_content, new_digest, row[0]),
        )
        connection.commit()
        connection.close()
        with self.assertRaises(ReplayError) as caught:
            replay_and_compare(self.bundle, str(tampered))
        first = caught.exception.details["first_mismatch"]
        self.assertIn(first["field"], {"window.digest", "window.stats"})
        self.assertEqual(first["window_id"], row[0])
        self.assertIsNotNone(first["partition_label"])
        self.assertIn("start_ms", first["window_boundary"])
        self.assertIn("end_ms", first["window_boundary"])

    def test_snapshot_whole_replacement_and_diff(self) -> None:
        target = self.directory / "restore-target.sqlite"
        shutil.copy2(self.snapshot, target)
        connection = sqlite3.connect(target)
        deleted = connection.execute(
            "SELECT window_id FROM windows ORDER BY window_id LIMIT 1"
        ).fetchone()[0]
        upgraded = connection.execute(
            "SELECT window_id FROM windows ORDER BY window_id LIMIT 1 OFFSET 1"
        ).fetchone()[0]
        connection.execute("DELETE FROM windows WHERE window_id=?", (deleted,))
        connection.execute(
            "UPDATE windows SET scale=4,rules_version='legacy-v0' WHERE window_id=?",
            (upgraded,),
        )
        connection.commit()
        connection.close()
        report = restore_snapshot(target, self.snapshot)
        self.assertEqual(
            report["replacement"], "whole_window_conclusion_and_watermark_state"
        )
        self.assertEqual(report["added"], [deleted])
        self.assertEqual(report["removed"], [])
        self.assertEqual(report["scale_upgraded"], [upgraded])
        self.assertEqual(LedgerReader(target).meta()["run_status"], "complete")
        self.assertEqual(len(LedgerReader(target).windows()), 36)
        self.assertTrue(
            all(window["scale"] == 3 for window in LedgerReader(target).windows())
        )
        self.assertEqual(
            len(LedgerReader(target).watermark_points()),
            len(self.run_result.watermarks),
        )

    def test_ui_projection_is_read_only_and_stable(self) -> None:
        before = file_digest(self.ledger)
        first = load_dashboard(self.ledger)
        second = load_dashboard(self.ledger)
        after = file_digest(self.ledger)
        self.assertEqual(before, after)
        self.assertEqual(canonical_json(first), canonical_json(second))
        self.assertEqual(len(first["window_rows"]), 36)
        self.assertGreater(len(first["partition_rows"]), 0)
        self.assertGreater(len(first["error_rows"]), 0)

    def test_crash_marks_planned_windows_unknown_then_replay_requires_rerun(self) -> None:
        crash_ledger = self.directory / "crash.sqlite"
        command = [
            sys.executable,
            "-m",
            "lateness_ledger",
            "run",
            "--sources",
            str(ROOT / "samples" / "sources_source_max.json"),
            "--rules",
            str(ROOT / "samples" / "window_rules.json"),
            "--budgets",
            str(ROOT / "samples" / "derivation_budget.json"),
            "--ledger",
            str(crash_ledger),
            "--crash-after",
            "2",
        ]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 137)
        unknown = recover_unknown_windows(crash_ledger)
        self.assertGreater(len(unknown), 0)
        report = LedgerReader(crash_ledger).status_report()
        self.assertEqual(report["meta"]["run_status"], "incomplete_recovered")
        self.assertEqual(len(report["finalized"]), 2)
        with self.assertRaises(ReplayError) as caught:
            replay_and_compare(self.bundle, str(crash_ledger))
        self.assertEqual(caught.exception.details["required_action"], "rerun")
        meta = LedgerReader(crash_ledger).meta()
        self.assertNotIn("summary_json", meta)
        self.assertNotEqual(meta["run_status"], "complete")


if __name__ == "__main__":
    unittest.main()
