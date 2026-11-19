from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from vecflow.errors import AppError
from vecflow.ledger import ExecutionLedger, SnapshotStore
from vecflow.logging_utils import JsonLogger
from vecflow.service import ExecutionService
from tests.helpers import make_workspace


ROOT = Path(__file__).resolve().parents[1]


class LedgerReplayTests(unittest.TestCase):
    def _execute(self, root: Path):
        workspace = make_workspace(root / "ws", delay_ms=2)
        ledger = ExecutionLedger(root / "ledger.sqlite")
        service = ExecutionService(workspace, ledger, snapshot_store=SnapshotStore(root / "snapshots"), logger=JsonLogger(root / "logs"))
        report = service.execute()
        return workspace, ledger, service, report

    def test_ledger_format_records_and_consistent_replay(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, ledger, service, report = self._execute(root)
            try:
                batch = ledger.latest_batch(recover=False)
                self.assertEqual(batch.format_version, 1)
                self.assertEqual(batch.status, "committed")
                self.assertEqual(batch.result_digest, report.overall_digest)
                self.assertEqual(len(batch.shards), 3)
                self.assertTrue(all(shard.wait_duration_ms is not None for shard in batch.shards))
                self.assertIsNotNone(batch.cumulative_digest)
                replay = service.replay(batch_id=batch.batch_id, save_snapshot=False)
                self.assertEqual(replay["compared_shards"], 3)
                self.assertEqual(replay["unknown_rerun_shards"], [])
                source = ledger.get_batch(batch.batch_id)
                self.assertEqual(source.replay_status, "consistent")
            finally:
                ledger.close()

    def test_replay_reports_first_mismatch_with_basis(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, ledger, service, _ = self._execute(root)
            try:
                batch = ledger.latest_batch(recover=False)
                first = batch.shards[0]
                with sqlite3.connect(root / "ledger.sqlite") as conn:
                    conn.execute(
                        "UPDATE shard_records SET result_digest = ? WHERE batch_id = ? AND shard_id = ?",
                        ("0" * 64, batch.batch_id, first.shard_id),
                    )
                    conn.commit()
                with self.assertRaises(AppError) as raised:
                    service.replay(batch_id=batch.batch_id, save_snapshot=False)
                self.assertEqual(raised.exception.code, "E_REPLAY")
                self.assertEqual(raised.exception.details["shard_id"], first.shard_id)
                self.assertEqual(raised.exception.details["field"], "result_digest")
                self.assertIn("first mismatch", raised.exception.details["basis"])
            finally:
                ledger.close()

    def test_snapshot_restore_whole_replaces_newer_state(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            workspace, ledger, service, report = self._execute(root)
            try:
                snapshot = root / "snapshots" / "session.json"
                self.assertTrue(snapshot.exists())
                old_version = workspace.parameter_version
                old_digest = workspace.content_digest
                workspace.update_global_budget(1, persist=False)
                self.assertNotEqual(workspace.content_digest, old_digest)
                restored = service.restore_snapshot(snapshot)
                self.assertTrue(restored.replaced)
                self.assertIn("content_digest", restored.mismatched_fields)
                self.assertIn("parameter_version", restored.mismatched_fields)
                self.assertEqual(workspace.parameter_version, old_version)
                self.assertEqual(workspace.content_digest, old_digest)
                self.assertIsNotNone(workspace.current_run)
            finally:
                ledger.close()

    def test_real_forced_crash_recovery_has_unknown_and_no_exit_report(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ledger_path = root / "ledger.sqlite"
            data = ROOT / "samples" / "data_blocks.json"
            tasks = ROOT / "samples" / "tasks.json"
            run = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "vecflow",
                    "run",
                    "--data",
                    str(data),
                    "--tasks",
                    str(tasks),
                    "--ledger",
                    str(ledger_path),
                    "--snapshot-dir",
                    str(root / "snapshots"),
                    "--log-dir",
                    str(root / "logs"),
                    "--force-crash-after",
                    "2",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(run.returncode, 137, run.stderr)
            recover = subprocess.run(
                [sys.executable, "-m", "vecflow", "recover", "--ledger", str(ledger_path), "--log-dir", str(root / "logs")],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(recover.returncode, 0, recover.stderr)
            recovered = json.loads(recover.stdout)
            self.assertFalse(recovered["exit_report_available"])
            self.assertIsNone(recovered["exit_report"])
            statuses = [shard["status"] for shard in recovered["latest"]["shards"]]
            self.assertIn("unknown", statuses)
            self.assertIn("completed", statuses)

            replay = subprocess.run(
                [sys.executable, "-m", "vecflow", "replay", "--ledger", str(ledger_path), "--snapshot-dir", str(root / "snapshots"), "--log-dir", str(root / "logs")],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(replay.returncode, 0, replay.stderr + replay.stdout)
            replay_payload = json.loads(replay.stdout)
            self.assertTrue(replay_payload["ok"])
            self.assertGreater(len(replay_payload["replay"]["unknown_rerun_shards"]), 0)
            self.assertEqual(replay_payload["source_batch"]["replay_status"], "consistent")


if __name__ == "__main__":
    unittest.main()
