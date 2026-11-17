from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from exp_tracker.errors import FormatError
from exp_tracker.repository import UpgradeInterrupted
from exp_tracker.service import ExperimentService

from .helpers import mixed_document


class UpgradeTests(unittest.TestCase):
    def test_open_upgrades_all_legacy_records_and_writes_readonly_backups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            service = ExperimentService.initialize(workspace, mixed_document())
            self.assertEqual(service.upgrade_result.status, "complete")
            self.assertEqual(
                [entry["key"] for entry in service.upgrade_result.processed],
                ["baseline@v1", "tuned@v1"],
            )
            self.assertEqual(service.library.format_versions(), [3])
            self.assertEqual(service.library.source_format_versions(), [1, 2, 3])

            checkpoint = json.loads(
                (workspace / "upgrade-checkpoint.json").read_text(encoding="utf-8")
            )
            self.assertEqual(checkpoint["status"], "complete")
            self.assertEqual(checkpoint["position"], 2)
            self.assertEqual(len(checkpoint["completed"]), 2)
            self.assertIsNone(checkpoint["active"])

            records = json.loads((workspace / "records.json").read_text(encoding="utf-8"))
            self.assertEqual({record["format_version"] for record in records["runs"]}, {3})
            backups = list((workspace / "backups").glob("baseline-v1-f1-*.json"))
            self.assertEqual(len(backups), 1)
            if os.name != "nt":
                self.assertEqual(backups[0].stat().st_mode & 0o777, 0o444)

    def test_interrupted_upgrade_resumes_from_checkpoint_without_repeating(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            ExperimentService.initialize(workspace, mixed_document())

            # Recreate the legacy library and deliberately stop after one record.
            from exp_tracker.service import reset_session_artifacts

            reset_session_artifacts(workspace)
            from exp_tracker.repository import RecordRepository
            from exp_tracker.codec import parse_import_document

            library = parse_import_document(mixed_document())
            repository = RecordRepository(workspace)
            repository.initialize(library)
            with self.assertRaises(UpgradeInterrupted) as caught:
                repository.load(stop_after=1)
            self.assertEqual(caught.exception.entry["key"], "baseline@v1")

            first_checkpoint = json.loads(
                (workspace / "upgrade-checkpoint.json").read_text(encoding="utf-8")
            )
            self.assertEqual(first_checkpoint["status"], "running")
            self.assertEqual(first_checkpoint["position"], 1)
            self.assertEqual(
                [entry["key"] for entry in first_checkpoint["completed"]],
                ["baseline@v1"],
            )
            partial = json.loads((workspace / "records.json").read_text(encoding="utf-8"))
            versions = {
                record["run_id"]: record["format_version"] for record in partial["runs"]
            }
            self.assertEqual(versions, {"baseline": 3, "tuned": 2, "ensemble": 3})

            service = ExperimentService.open(workspace)
            self.assertEqual(service.upgrade_result.recovered, [])
            self.assertEqual(
                [entry["key"] for entry in service.upgrade_result.processed],
                ["tuned@v1"],
            )
            checkpoint = json.loads(
                (workspace / "upgrade-checkpoint.json").read_text(encoding="utf-8")
            )
            self.assertEqual(checkpoint["status"], "complete")
            self.assertEqual(
                [entry["key"] for entry in checkpoint["completed"]],
                ["baseline@v1", "tuned@v1"],
            )
            self.assertEqual(
                len(list((workspace / "backups").glob("baseline-v1-f1-*.json"))), 1
            )

    def test_crash_between_record_write_and_checkpoint_completion_is_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            ExperimentService.initialize(workspace, mixed_document())
            from exp_tracker.service import reset_session_artifacts

            reset_session_artifacts(workspace)
            from exp_tracker.codec import parse_import_document
            from exp_tracker.repository import RecordRepository

            repository = RecordRepository(workspace)
            repository.initialize(parse_import_document(mixed_document()))
            with self.assertRaises(UpgradeInterrupted):
                repository.load(stop_after=1)

            checkpoint_path = workspace / "upgrade-checkpoint.json"
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            completed_entry = checkpoint["completed"][0]
            checkpoint["completed"] = []
            checkpoint["position"] = 0
            checkpoint["active"] = completed_entry
            checkpoint["status"] = "running"
            checkpoint_path.write_text(
                json.dumps(checkpoint, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )

            service = ExperimentService.open(workspace)
            self.assertEqual(service.upgrade_result.recovered, ["baseline@v1"])
            self.assertEqual(
                [entry["key"] for entry in service.upgrade_result.processed],
                ["tuned@v1"],
            )
            self.assertEqual(
                len(list((workspace / "backups").glob("baseline-v1-f1-*.json"))), 1
            )

    def test_unknown_version_rejects_before_any_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            document = mixed_document()
            document["runs"][0]["format_version"] = 9
            records_path = workspace / "records.json"
            records_path.write_text(
                json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8"
            )
            original = records_path.read_bytes()
            with self.assertRaises(FormatError) as caught:
                ExperimentService.open(workspace)
            self.assertEqual(caught.exception.code, "E_FORMAT")
            self.assertEqual(caught.exception.details["path"], "$.runs[0].format_version")
            self.assertEqual(records_path.read_bytes(), original)
            self.assertFalse((workspace / "upgrade-checkpoint.json").exists())
            self.assertEqual(
                [path for path in (workspace / "backups").glob("*.json")], []
            )

    def test_completed_upgrade_is_not_reprocessed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            ExperimentService.initialize(workspace, mixed_document())
            service = ExperimentService.open(workspace)
            self.assertEqual(service.upgrade_result.status, "not_needed")
            self.assertEqual(service.upgrade_result.processed, [])


if __name__ == "__main__":
    unittest.main()

