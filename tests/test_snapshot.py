from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from exp_tracker.errors import ReplayError
from exp_tracker.service import ExperimentService

from .helpers import current_record, mixed_document


class SnapshotTests(unittest.TestCase):
    def test_snapshot_metadata_contains_versions_digests_checkpoint_and_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            service = ExperimentService.initialize(workspace, mixed_document())
            snapshot = service.snapshot("metadata")
            self.assertEqual(snapshot["snapshot_id"], 1)
            self.assertEqual(snapshot["format_versions"], [3])
            self.assertEqual(snapshot["source_format_versions"], [1, 2, 3])
            self.assertEqual(snapshot["aggregation_versions"], ["agg-v1"])
            self.assertEqual(snapshot["checkpoint_path"], "upgrade-checkpoint.json")
            self.assertEqual(snapshot["checkpoint_position"], 2)
            self.assertEqual(snapshot["run_count"], 3)
            self.assertEqual(len(snapshot["library_digest"]), 64)

            with sqlite3.connect(workspace / "snapshots.sqlite3") as connection:
                rows = connection.execute(
                    "SELECT run_id, content_digest, metric_digest FROM snapshot_runs "
                    "WHERE snapshot_id = 1 AND is_current = 1 ORDER BY run_id"
                ).fetchall()
            self.assertEqual([row[0] for row in rows], ["baseline", "ensemble", "tuned"])
            self.assertTrue(all(len(row[1]) == 64 and len(row[2]) == 64 for row in rows))

    def test_restore_detects_differences_and_replaces_entire_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            service = ExperimentService.initialize(workspace, mixed_document())
            snapshot = service.snapshot("before-change")
            service.commit_record(current_record(run_id="baseline", run_version=2, learning_rate=0.02))
            extra = current_record(
                run_id="extra", run_version=1, seed=99, learning_rate=0.009, steps=8
            )
            service.commit_record(extra)
            self.assertEqual(len(service.list_runs()), 4)

            restored = service.restore_snapshot(snapshot["snapshot_id"], replay=True)
            difference = restored["restore"]
            self.assertEqual(difference["mode"], "whole_replace")
            self.assertEqual(difference["removed_from_current"], ["extra"])
            self.assertEqual([item["run_id"] for item in difference["changed"]], ["baseline"])
            self.assertTrue(difference["is_different"])
            self.assertTrue(restored["state_replaced"])
            baseline = service.run_detail("baseline")["run"]
            self.assertEqual(baseline["run_version"], 1)
            self.assertEqual([run["run_id"] for run in service.list_runs()], ["baseline", "ensemble", "tuned"])
            self.assertEqual(restored["replay"]["status"], "pass")
            self.assertEqual(
                [item["run_id"] for item in restored["replay"]["results"]],
                ["baseline", "ensemble", "tuned"],
            )
            checkpoint = json.loads(
                (workspace / "upgrade-checkpoint.json").read_text(encoding="utf-8")
            )
            self.assertEqual(checkpoint["status"], "complete")
            self.assertEqual(checkpoint["position"], 2)

    def test_restore_with_identical_state_reports_no_differences(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = ExperimentService.initialize(Path(directory), mixed_document())
            snapshot = service.snapshot("same")
            restored = service.restore_snapshot(snapshot["snapshot_id"], replay=True)
            self.assertFalse(restored["restore"]["is_different"])
            self.assertEqual(restored["restore"]["changed"], [])

    def test_missing_snapshot_metric_digest_is_replay_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            service = ExperimentService.initialize(workspace, mixed_document())
            snapshot = service.snapshot("missing-digest")
            with sqlite3.connect(workspace / "snapshots.sqlite3") as connection:
                connection.execute(
                    "UPDATE snapshot_runs SET metric_digest = NULL "
                    "WHERE snapshot_id = ? AND run_id = 'baseline'",
                    (snapshot["snapshot_id"],),
                )
            with self.assertRaises(ReplayError) as caught:
                service.replay_snapshot(snapshot["snapshot_id"])
            self.assertEqual(caught.exception.code, "E_REPLAY")
            self.assertEqual(caught.exception.details["run_id"], "baseline")
            self.assertIn("run_version=1", caught.exception.details["version"])
            self.assertIn("missing", caught.exception.details["basis"])

    def test_unsupported_snapshot_aggregation_version_is_replay_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            service = ExperimentService.initialize(workspace, mixed_document())
            snapshot = service.snapshot("agg-version")
            with sqlite3.connect(workspace / "snapshots.sqlite3") as connection:
                record_json = connection.execute(
                    "SELECT record_json FROM snapshot_runs "
                    "WHERE snapshot_id = ? AND run_id = 'baseline'",
                    (snapshot["snapshot_id"],),
                ).fetchone()[0]
                raw = json.loads(record_json)
                raw["aggregation"]["version"] = "agg-v2"
                connection.execute(
                    "UPDATE snapshot_runs SET record_json = ?, aggregation_version = 'agg-v2' "
                    "WHERE snapshot_id = ? AND run_id = 'baseline'",
                    (json.dumps(raw, sort_keys=True), snapshot["snapshot_id"]),
                )
            with self.assertRaises(ReplayError) as caught:
                service.replay_snapshot(snapshot["snapshot_id"])
            self.assertEqual(caught.exception.details["run_id"], "baseline")
            self.assertIn("agg-v2", caught.exception.details["basis"])

    def test_mismatched_snapshot_summary_is_replay_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            service = ExperimentService.initialize(workspace, mixed_document())
            snapshot = service.snapshot("bad-summary")
            with sqlite3.connect(workspace / "snapshots.sqlite3") as connection:
                connection.execute(
                    "UPDATE snapshot_runs SET metric_digest = ? "
                    "WHERE snapshot_id = ? AND run_id = 'baseline'",
                    ("0" * 64, snapshot["snapshot_id"]),
                )
            with self.assertRaises(ReplayError) as caught:
                service.replay_snapshot(snapshot["snapshot_id"])
            self.assertEqual(caught.exception.details["run_id"], "baseline")
            self.assertIn("summary", caught.exception.details["basis"])
            self.assertEqual(caught.exception.details["expected_digest"], "0" * 64)


if __name__ == "__main__":
    unittest.main()

