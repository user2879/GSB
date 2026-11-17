from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from exp_tracker.errors import RunError
from exp_tracker.service import ExperimentService

from .helpers import current_record, mixed_document


class ReplacementTests(unittest.TestCase):
    def test_new_version_replaces_whole_record_and_retains_readonly_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            service = ExperimentService.initialize(workspace, mixed_document())
            previous = service.run_detail("baseline")["run"]
            replacement = current_record(
                run_id="baseline",
                run_version=2,
                learning_rate=0.02,
                extra_hyperparameters={"dropout": 0.08},
            )
            difference = service.commit_record(replacement)

            self.assertTrue(difference["whole_record_replaced"])
            self.assertEqual(difference["old_version"], 1)
            self.assertEqual(difference["new_version"], 2)
            changed = {change["field"]: change for change in difference["changed_fields"]}
            self.assertIn("parameters.learning_rate", changed)
            self.assertEqual(changed["parameters.learning_rate"]["before"], 0.01)
            self.assertEqual(changed["parameters.learning_rate"]["after"], 0.02)
            self.assertIn("parameters.hyperparameters.dropout", changed)
            self.assertNotEqual(difference["metric_changes"]["changed_step_count"], 0)
            self.assertNotEqual(difference["before_metric_digest"], difference["after_metric_digest"])

            detail = service.run_detail("baseline")
            self.assertEqual(detail["run"]["run_version"], 2)
            self.assertEqual([item["run_version"] for item in detail["history"]], [1])
            self.assertEqual(detail["history"][0]["final_metrics"], previous["final_metrics"])
            self.assertTrue(detail["history"][0]["read_only"])
            self.assertFalse(detail["run"]["read_only"])

            current_keys = set(
                json.loads((workspace / "records.json").read_text(encoding="utf-8"))["runs"][0]
            )
            self.assertEqual(
                current_keys,
                {
                    "format_version",
                    "run_id",
                    "run_version",
                    "parameters",
                    "step_metrics",
                    "timestamp",
                    "aggregation",
                    "parent_run_id",
                    "final_metrics",
                    "provenance",
                },
            )
            backup_files = list((workspace / "backups").glob("current-baseline-v1-*.json"))
            self.assertEqual(len(backup_files), 1)
            if os.name != "nt":
                self.assertEqual(backup_files[0].stat().st_mode & 0o777, 0o444)
            report = json.loads(
                (workspace / "reports" / "replacements" / "baseline-v2.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(report["whole_record_replaced"])
            self.assertEqual(report["metric_changes"], difference["metric_changes"])

    def test_aggregation_change_is_reported_and_not_merged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = ExperimentService.initialize(Path(directory), mixed_document())
            replacement = current_record(
                run_id="baseline",
                run_version=2,
                learning_rate=0.01,
                aggregation="mean_last_5",
            )
            difference = service.commit_record(replacement)
            changed = {change["field"] for change in difference["changed_fields"]}
            self.assertIn("aggregation.name", changed)
            self.assertEqual(difference["aggregation"]["after"]["name"], "mean_last_5")
            current = service.run_detail("baseline")["run"]
            self.assertEqual(current["aggregation"]["name"], "mean_last_5")
            self.assertEqual(
                current["final_metrics"],
                replacement["final_metrics"],
            )

    def test_non_monotonic_version_is_run_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = ExperimentService.initialize(Path(directory), mixed_document())
            replacement = current_record(run_id="baseline", run_version=1)
            with self.assertRaises(RunError) as caught:
                service.commit_record(replacement)
            self.assertEqual(caught.exception.code, "E_RUN")
            self.assertEqual(caught.exception.details["constraint"], "run_version_monotonic")
            self.assertEqual(caught.exception.details["run_id"], "baseline")
            self.assertEqual(service.run_detail("baseline")["run"]["run_version"], 1)

    def test_parent_cycle_on_replacement_is_run_error_and_library_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = ExperimentService.initialize(Path(directory), mixed_document())
            before = service.run_detail("baseline")["run"]["content_digest"]
            replacement = current_record(
                run_id="baseline",
                run_version=2,
                parent_run_id="ensemble",
            )
            with self.assertRaises(RunError) as caught:
                service.commit_record(replacement)
            self.assertEqual(caught.exception.details["constraint"], "parent_acyclic")
            self.assertIn("baseline", caught.exception.details["chain"])
            self.assertEqual(service.run_detail("baseline")["run"]["content_digest"], before)

    def test_new_run_must_start_at_version_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = ExperimentService.initialize(Path(directory), mixed_document())
            with self.assertRaises(RunError) as caught:
                service.commit_record(current_record(run_id="new-run", run_version=2))
            self.assertEqual(caught.exception.details["constraint"], "run_version_monotonic")


if __name__ == "__main__":
    unittest.main()

