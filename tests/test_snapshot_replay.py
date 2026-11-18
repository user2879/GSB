from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from slowquery_workbench import Workbench, WorkbenchError

from tests.helpers import SAMPLE_QUERY, new_sample_workbench


class SnapshotReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _diagnosed(self) -> tuple[Workbench, Path, str]:
        workbench = new_sample_workbench(self.root)
        result = workbench.diagnose(
            SAMPLE_QUERY,
            timeout_ms=500,
            budget_ms=5000,
            threshold_ms=0,
            threshold_rows=0,
        )
        snapshot = workbench.save_snapshot(self.root / "snapshot.sqlite")
        return workbench, snapshot, result["query_id"]

    def test_restore_reports_version_diff_and_replaces_whole_state(self) -> None:
        workbench, snapshot, query_id = self._diagnosed()
        snapshot_versions = dict(workbench.versions())
        workbench.append_rows(
            "orders",
            ["id", "customer_id", "status", "amount", "created_at"],
            [(999994, 1, "paid", 1.0, "2024-12-31")],
        )
        workbench.add_index("orders", ["amount"], name="idx_orders_after_snapshot")
        self.assertNotEqual(workbench.versions()["data_version"], snapshot_versions["data_version"])
        with self.assertRaises(WorkbenchError) as captured:
            workbench.restore_snapshot(snapshot)
        self.assertEqual(captured.exception.code, "E_VERSION")
        differences = captured.exception.details["differences"]
        self.assertEqual(
            [item["field"] for item in differences],
            ["data_version", "index_set_hash", "index_version"],
        )
        self.assertEqual(workbench.versions(), snapshot_versions)
        restored_conclusion = workbench.session.conclusion(query_id)
        self.assertFalse(restored_conclusion["is_stale"])
        count = workbench.conn.execute("SELECT COUNT(*) AS n FROM orders").fetchone()["n"]
        self.assertEqual(count, 20_000)
        report = workbench.replay_conclusions(repeats=2, timeout_ms=500, budget_ms=8000)
        self.assertTrue(report["matched"])
        workbench.close()

    def test_equal_version_snapshot_restores_without_conflict(self) -> None:
        workbench, snapshot, _query_id = self._diagnosed()
        result = workbench.restore_snapshot(snapshot)
        self.assertTrue(result["replaced"])
        self.assertEqual(result["differences"], [])
        workbench.close()

    def test_missing_candidate_raises_e_replay_with_query_id_and_reason(self) -> None:
        workbench, snapshot, query_id = self._diagnosed()
        workbench.restore_snapshot(snapshot)
        conclusion = workbench.session.conclusion(query_id)
        candidate_id = conclusion["candidates"][0]["candidate_id"]
        with workbench.conn:
            workbench.conn.execute("DELETE FROM __sw_candidates WHERE candidate_id = ?", (candidate_id,))
        with self.assertRaises(WorkbenchError) as captured:
            workbench.replay_conclusions(repeats=1, timeout_ms=500, budget_ms=8000)
        self.assertEqual(captured.exception.code, "E_REPLAY")
        unreproducible = captured.exception.details["unreproducible"]
        self.assertEqual(unreproducible[0]["query_id"], query_id)
        self.assertIn("candidate deleted", unreproducible[0]["reason"])
        workbench.close()

    def test_version_change_makes_replay_unreproducible(self) -> None:
        workbench, _snapshot, query_id = self._diagnosed()
        workbench.add_index("orders", ["status"], name="idx_orders_status_replay")
        with self.assertRaises(WorkbenchError) as captured:
            workbench.replay_conclusions(repeats=1, timeout_ms=500, budget_ms=8000)
        self.assertEqual(captured.exception.code, "E_REPLAY")
        self.assertEqual(captured.exception.details["unreproducible"][0]["query_id"], query_id)
        self.assertIn("index_version", captured.exception.details["unreproducible"][0]["reason"])
        workbench.close()


if __name__ == "__main__":
    unittest.main()
