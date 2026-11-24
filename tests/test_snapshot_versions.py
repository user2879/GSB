from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from columnar_diag.errors import AppError
from columnar_diag.models import BudgetConfig, TaskResult
from columnar_diag.sample import build_sample_spec, sample_budget
from columnar_diag.snapshot import load_snapshot, restore_snapshot, save_snapshot
from columnar_diag.util import atomic_write_json
from columnar_diag.verifier import authoritative_replay
from columnar_diag.workspace import Workspace


class SnapshotAndVersionTests(unittest.TestCase):
    def test_snapshot_whole_replacement_cursor_resume_and_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = Workspace.create(
                root,
                build_sample_spec(),
                budget=BudgetConfig(**sample_budget()),
                overwrite=True,
            )
            try:
                workspace.scan_pending(batch_size=5, max_tasks=10, delay_ms=1.0)
                snapshot_path = root / "snapshot.json"
                saved = save_snapshot(workspace, snapshot_path)
                workspace.scan_pending(batch_size=5)
                restored = restore_snapshot(workspace, snapshot_path)
                self.assertTrue(restored["whole_view_replacement"])
                self.assertEqual(
                    workspace.ledger.current_generation()["cursor"], 10
                )
                workspace.scan_pending(batch_size=5, delay_ms=1.0)
                replay = authoritative_replay(workspace)
                self.assertTrue(replay["ok"])
                self.assertEqual(saved["content_digest"], restored["content_digest"])

                tampered_path = root / "tampered.json"
                tampered = load_snapshot(snapshot_path)
                tampered["statistics"]["completed_rows"] += 1
                atomic_write_json(tampered_path, tampered)
                with self.assertRaises(AppError) as context:
                    load_snapshot(tampered_path)
                self.assertEqual(context.exception.code, "E_SNAPSHOT")
            finally:
                workspace.close()

    def test_ledger_divergence_is_reported_at_first_block(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace.create(
                Path(directory),
                build_sample_spec(),
                budget=BudgetConfig(**sample_budget()),
                overwrite=True,
            )
            try:
                workspace.run_full_scan(batch_size=8)
                row = workspace.ledger.current_task_rows()[2]
                with workspace.ledger.conn:
                    workspace.ledger.conn.execute(
                        "UPDATE scan_ledger SET result_digest=? WHERE ledger_id=?",
                        ("f" * 64, row["ledger_id"]),
                    )
                with self.assertRaises(AppError) as context:
                    authoritative_replay(workspace)
                error = context.exception
                self.assertEqual(error.code, "E_DIVERGE")
                self.assertEqual(error.block_ordinal, row["block_ordinal"])
                self.assertEqual(error.field, "result_digest")
                with self.assertRaises(AppError) as replay_context:
                    authoritative_replay(workspace, error_code="E_REPLAY")
                self.assertEqual(replay_context.exception.code, "E_REPLAY")
            finally:
                workspace.close()

    def test_batch_commit_is_atomic_after_forced_termination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace.create(
                Path(directory),
                build_sample_spec(),
                budget=BudgetConfig(**sample_budget()),
                overwrite=True,
            )
            try:
                workspace.run_full_scan(batch_size=5)
                before = len(workspace.ledger.current_batches())
                task = workspace.bundle.tasks[0]
                result = TaskResult(
                    task=task,
                    completed_rows=0,
                    decode_memory_bytes=0,
                    queue_started_ms=0.0,
                    wait_ended_ms=0.0,
                    run_started_ms=0.0,
                    run_ended_ms=0.0,
                    duration_ms=0.0,
                    block_digest="fault",
                    result_digest="fault",
                    result_json={"fault": True},
                    stats={},
                )
                with self.assertRaises(AppError) as context:
                    workspace.ledger.commit_batch(
                        generation_id=int(
                            workspace.ledger.current_generation()["generation_id"]
                        ),
                        reason="forced-termination",
                        task_results=[result],
                        query_results={},
                        plans={},
                        cursor_before=0,
                        cursor_after=1,
                        tasks_total=len(workspace.bundle.tasks),
                        budget_version=workspace.versions["budget_params_version"],
                        encoding_version=workspace.versions[
                            "encoding_selection_version"
                        ],
                        segment_digest=workspace.image.file_digest,
                        plan_digest=workspace.bundle.digest,
                        fail_after_rows=1,
                    )
                self.assertEqual(context.exception.code, "E_LEDGER")
                self.assertEqual(len(workspace.ledger.current_batches()), before)
                self.assertTrue(authoritative_replay(workspace)["ok"])
            finally:
                workspace.close()

    def test_budget_and_encoding_versions_are_monotonic_and_stale_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace.create(
                Path(directory),
                build_sample_spec(),
                budget=BudgetConfig(**sample_budget()),
                overwrite=True,
            )
            try:
                workspace.run_full_scan(batch_size=6)
                old_generation = int(
                    workspace.ledger.current_generation()["generation_id"]
                )
                budget_result = workspace.change_budget(
                    BudgetConfig(3, 100, 2), batch_size=6
                )
                self.assertTrue(budget_result["version_changed"])
                self.assertGreaterEqual(budget_result["budget_params_version"], 2)
                current = workspace.ledger.current_generation()
                self.assertNotEqual(current["generation_id"], old_generation)
                rows = workspace.ledger.generation_rows()
                old_row = next(
                    row for row in rows if row["generation_id"] == old_generation
                )
                self.assertEqual(old_row["status"], "stale")
                self.assertEqual(old_row["is_current"], 0)
                encoding_result = workspace.change_encoding_policy(
                    column="region", encoding="raw", compressed=False
                )
                self.assertTrue(encoding_result["version_changed"])
                self.assertGreaterEqual(
                    encoding_result["encoding_selection_version"], 2
                )
                self.assertTrue(authoritative_replay(workspace)["ok"])
                self.assertEqual(
                    sum(
                        int(row["is_current"])
                        for row in workspace.ledger.generation_rows()
                    ),
                    1,
                )
            finally:
                workspace.close()

    def test_restoring_older_budget_snapshot_restores_recorded_budget_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = Workspace.create(
                root,
                build_sample_spec(),
                budget=BudgetConfig(**sample_budget()),
                overwrite=True,
            )
            try:
                workspace.run_full_scan(batch_size=6)
                snapshot_path = root / "budget-v1.json"
                save_snapshot(workspace, snapshot_path)
                workspace.change_budget(BudgetConfig(2, 100, 2), batch_size=6)
                self.assertEqual(
                    workspace.versions["budget_params_version"], 2
                )
                restored = restore_snapshot(workspace, snapshot_path)
                self.assertTrue(restored["budget_restored_from_snapshot"])
                self.assertEqual(
                    workspace.versions["budget_params_version"], 1
                )
                self.assertEqual(workspace.budget.global_parallelism, 4)
                self.assertTrue(authoritative_replay(workspace)["ok"])
            finally:
                workspace.close()


if __name__ == "__main__":
    unittest.main()
