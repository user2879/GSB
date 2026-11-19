from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

from vecflow.ledger import ExecutionLedger, SnapshotStore
from vecflow.logging_utils import JsonLogger
from vecflow.service import ExecutionService
from tests.helpers import make_workspace


class ServiceAndStaleTests(unittest.TestCase):
    def test_repeated_execution_has_byte_identical_results(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            workspace = make_workspace(root / "ws", delay_ms=8)
            ledger = ExecutionLedger(root / "ledger.sqlite")
            try:
                service = ExecutionService(
                    workspace,
                    ledger,
                    snapshot_store=SnapshotStore(root / "snapshots"),
                    logger=JsonLogger(root / "logs"),
                )
                first = service.execute()
                second = service.execute()
            finally:
                ledger.close()
            self.assertEqual(first.status, "committed")
            self.assertEqual(second.status, "committed")
            self.assertEqual(first.result_bytes, second.result_bytes)
            self.assertEqual(first.overall_digest, second.overall_digest)
            for task_id in first.task_results:
                self.assertEqual(first.task_results[task_id].canonical_bytes, second.task_results[task_id].canonical_bytes)

    def test_parameter_change_invalidates_running_and_queued_shards(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            workspace = make_workspace(root / "ws", budget=4, threads=2, quota=2, delay_ms=140)
            ledger = ExecutionLedger(root / "ledger.sqlite")
            service = ExecutionService(workspace, ledger, snapshot_store=SnapshotStore(root / "snapshots"), logger=JsonLogger(root / "logs"))
            holder: dict[str, object] = {}

            def run() -> None:
                holder["report"] = service.execute(save_snapshot=False)

            thread = threading.Thread(target=run)
            thread.start()
            deadline = time.time() + 3
            while time.time() < deadline:
                if workspace._active_runs:
                    statuses = workspace._active_runs[0].snapshot_statuses()
                    if "running" in statuses.values() and "queued" in statuses.values():
                        break
                time.sleep(0.005)
            result = workspace.update_global_budget(3, persist=False)
            thread.join(timeout=5)
            report = holder.get("report")
            self.assertIsNotNone(report)
            self.assertEqual(report.status, "stale")
            self.assertTrue(report.stale)
            self.assertEqual(report.task_results, {})
            self.assertTrue(result.affected_running)
            self.assertTrue(result.affected_queued)
            self.assertTrue(result.stale_run_ids)
            stale_batch = ledger.latest_batch(recover=False)
            self.assertEqual(stale_batch.status, "stale")
            self.assertFalse(any(shard.status in {"queued", "running"} for shard in stale_batch.shards))

            replacement = service.execute(save_snapshot=False)
            self.assertEqual(replacement.status, "committed")
            self.assertEqual(workspace.current_run.run_id, replacement.run_id)
            self.assertGreater(replacement.parameter_version, report.parameter_version)
            workspace.update_global_budget(4, persist=False)
            self.assertTrue(replacement.stale)
            replacement_batch = ledger.get_batch(replacement.batch_id)
            self.assertEqual(replacement_batch.status, "stale")
            self.assertFalse(replacement_batch.exit_report_allowed)
            ledger.close()


if __name__ == "__main__":
    unittest.main()
