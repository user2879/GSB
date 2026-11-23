from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from planner_app.engine import PlanEngine
from planner_app.errors import AppError
from planner_app.snapshots import read_snapshot


ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT / "samples" / "workspace.json"


class SnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.engine = PlanEngine(
            WORKSPACE,
            ledger_path=self.base / "ledger.sqlite",
            snapshot_dir=self.base / "snapshots",
            log_path=self.base / "app.jsonl",
            budget_ms=1000,
        )
        self.engine.submit_edits(
            [{"type": "push_filter", "target": "filter"}], query_name="q_status"
        )
        self.snapshot = self.engine.create_snapshot("q_status", self.base / "snapshot.json")
        self.snapshot_path = self.base / "snapshot.json"

    def test_restore_is_whole_replacement_and_replays_ledger(self) -> None:
        result = self.engine.restore_snapshot(self.snapshot_path)
        self.assertTrue(result["restored"])
        self.assertTrue(result["replacement"])
        self.assertEqual(result["replay"]["status"], "MATCH")
        self.assertEqual(
            result["views"]["q_status"]["plan_summary"],
            self.snapshot["logical_plan_summary"],
        )

    def test_content_digest_tamper_is_e_snapshot_and_snapshot_not_merged(self) -> None:
        document = read_snapshot(self.snapshot_path)
        document["content_digest"] = "f" * 64
        tampered = self.base / "tampered.json"
        tampered.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
        with self.assertRaises(AppError) as raised:
            self.engine.restore_snapshot(tampered)
        self.assertEqual(raised.exception.code, "E_SNAPSHOT")
        differences = raised.exception.details["differences"]
        self.assertEqual(differences[0]["field"], "content_digest")
        self.assertTrue(raised.exception.details["snapshot_ignored"])
        self.assertEqual(raised.exception.details["replacement_mode"], "whole_current_ledger_view")
        self.assertTrue(self.engine.current_view("q_status")["plan_summary"])

    def test_version_mismatch_lists_stable_difference_fields(self) -> None:
        document = read_snapshot(self.snapshot_path)
        document["sql_version"] = int(document["sql_version"]) + 1
        tampered = self.base / "version-tampered.json"
        tampered.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
        with self.assertRaises(AppError) as raised:
            self.engine.restore_snapshot(tampered)
        self.assertEqual(raised.exception.code, "E_SNAPSHOT")
        fields = [entry["field"] for entry in raised.exception.details["differences"]]
        self.assertIn("content_digest", fields)
        self.assertIn("sql_version", fields)
        self.assertEqual(fields, sorted(fields, key=lambda value: value.encode("utf-8")))


if __name__ == "__main__":
    unittest.main()
