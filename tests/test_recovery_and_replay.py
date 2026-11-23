from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from planner_app.engine import PlanEngine
from planner_app.errors import AppError
from planner_app.ledger import Ledger


ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT / "samples" / "workspace.json"


class RecoveryAndReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.ledger_path = self.base / "ledger.sqlite"
        self.snapshot_dir = self.base / "snapshots"
        self.log_path = self.base / "app.jsonl"
        self.engine = PlanEngine(
            WORKSPACE,
            ledger_path=self.ledger_path,
            snapshot_dir=self.snapshot_dir,
            log_path=self.log_path,
            budget_ms=1000,
        )

    def test_running_attempt_is_unknown_after_restart_and_not_applied(self) -> None:
        self.engine.submit_edits(
            [{"type": "prune_columns", "target": "project", "columns": ["min_amount"]}],
            query_name="q_status",
        )
        committed_summary = self.engine.current_view("q_status")["plan_summary"]
        cursor = self.engine.ledger.next_cursor(self.engine.workspace_id)
        attempt_id = self.engine.ledger.begin_attempt(
            workspace_id=self.engine.workspace_id,
            cursor=cursor,
            query_name="q_status",
            edits=[{"type": "push_filter", "target": "filter"}],
            parameter_version=99,
            next_edit_seq=2,
        )
        self.assertGreater(attempt_id, 0)
        # Abrupt termination is modelled by discarding this engine without commit.
        restarted = PlanEngine(
            WORKSPACE,
            ledger_path=self.ledger_path,
            snapshot_dir=self.snapshot_dir,
            log_path=self.log_path,
            budget_ms=1000,
        )
        self.assertEqual(restarted.current_view("q_status")["plan_summary"], committed_summary)
        unknown = restarted.recovery["unknown_attempts"]
        self.assertEqual(len(unknown), 1)
        self.assertEqual(unknown[0]["status"], "UNKNOWN")
        self.assertTrue(unknown[0]["requires_rerun"])
        self.assertFalse(restarted.recovery["exit_report_available"])

    def test_tampered_plan_bytes_are_rejected_as_replay_not_silently_loaded(self) -> None:
        self.engine.submit_edits(
            [{"type": "push_filter", "target": "filter"}], query_name="q_status"
        )
        latest = self.engine.ledger.batch_by_id(self.engine.states["q_status"].last_batch_id)
        self.assertIsNotNone(latest)
        corrupted = {
            **latest,
            "plan_bytes": b'{"format":1,"root":{"id":"scan","kind":"scan","props":{},"children":[],"version":1,"summary":"x","estimated_rows":1.0}}',
        }
        corrupted["content_hash"] = Ledger._batch_content_hash(corrupted)
        with self.engine.ledger.connection(write=True) as connection:
            connection.execute(
                "UPDATE plan_batches SET plan_bytes=?, content_hash=? WHERE batch_id=?",
                (corrupted["plan_bytes"], corrupted["content_hash"], latest["batch_id"]),
            )
            connection.commit()
        with self.assertRaises(AppError) as raised:
            PlanEngine(
                WORKSPACE,
                ledger_path=self.ledger_path,
                snapshot_dir=self.snapshot_dir,
                log_path=self.log_path,
                budget_ms=1000,
            )
        self.assertEqual(raised.exception.code, "E_REPLAY")
        self.assertIsNotNone(raised.exception.details.get("first_divergence"))
        self.assertGreaterEqual(raised.exception.edit_seq or 0, 1)

    def test_committed_batch_with_interrupted_verification_is_completed_on_restart(self) -> None:
        self.engine.submit_edits(
            [{"type": "push_filter", "target": "filter"}], query_name="q_status"
        )
        latest = self.engine.ledger.batch_by_id(self.engine.states["q_status"].last_batch_id)
        with self.engine.ledger.connection(write=True) as connection:
            connection.execute(
                "UPDATE plan_batches SET verify_status='PENDING', verify_json='{\"status\":\"PENDING\"}' WHERE batch_id=?",
                (latest["batch_id"],),
            )
            connection.commit()
        restarted = PlanEngine(
            WORKSPACE,
            ledger_path=self.ledger_path,
            snapshot_dir=self.snapshot_dir,
            log_path=self.log_path,
            budget_ms=1000,
        )
        self.assertTrue(restarted.recovery["pending_verification_batches"])
        self.assertTrue(
            any(item["status"] == "MATCH" for item in restarted.recovery["recovery_verifications"])
        )
        recovered = restarted.ledger.batch_by_id(latest["batch_id"])
        self.assertEqual(recovered["verify_status"], "MATCH")
        self.assertEqual(restarted.current_view("q_status")["compile_status"], "READY")


if __name__ == "__main__":
    unittest.main()
