from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from exp_tracker.codec import parse_library, parse_record
from exp_tracker.errors import FormatError, RunError
from exp_tracker.metrics import record_metric_digest
from exp_tracker.records import build_v1_record, build_v2_record
from exp_tracker.service import ExperimentService

from .helpers import clone, current_record, mixed_document


class CodecTests(unittest.TestCase):
    def test_three_versions_normalize_to_same_internal_metrics(self) -> None:
        current = current_record()
        records = [
            parse_record(build_v1_record(current)),
            parse_record(build_v2_record(current)),
            parse_record(current),
        ]
        self.assertEqual(
            {record_metric_digest(record) for record in records},
            {record_metric_digest(records[0])},
        )
        self.assertTrue(all(record.step_metrics == records[0].step_metrics for record in records))
        self.assertTrue(all(record.final_metrics == records[0].final_metrics for record in records))
        self.assertTrue(all(record.aggregation.name == "last" for record in records))
        self.assertEqual(records[0].provenance.source_format_version, 1)
        self.assertEqual(records[1].provenance.source_format_version, 2)
        self.assertEqual(records[2].provenance.source_format_version, 3)

    def test_unknown_newer_version_is_rejected_with_effective_path(self) -> None:
        document = mixed_document()
        document["runs"][0]["format_version"] = 4
        with self.assertRaises(FormatError) as caught:
            parse_library(document)
        self.assertEqual(caught.exception.code, "E_FORMAT")
        self.assertEqual(caught.exception.details["path"], "$.runs[0].format_version")
        self.assertEqual(caught.exception.details["effective_path"], "$.runs[0].format_version")
        self.assertEqual(caught.exception.details["supported"], [1, 2, 3])

    def test_missing_or_wrong_typed_field_is_format_error(self) -> None:
        record = build_v2_record(current_record())
        del record["timestamp"]
        with self.assertRaises(FormatError) as caught:
            parse_record(record, path="$.runs[0]")
        self.assertEqual(caught.exception.details["path"], "$.runs[0].timestamp")

        record = build_v2_record(current_record())
        record["parameters"]["seed"] = "11"
        with self.assertRaises(FormatError) as caught:
            parse_record(record, path="$.runs[0]")
        self.assertEqual(caught.exception.details["path"], "$.runs[0].parameters.seed")

    def test_invalid_import_is_atomic_and_writes_nothing(self) -> None:
        document = mixed_document()
        document["runs"][1]["timestamp"] = "not-an-integer"
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            with self.assertRaises(FormatError) as caught:
                ExperimentService.initialize(workspace, document)
            self.assertEqual(caught.exception.code, "E_FORMAT")
            self.assertFalse((workspace / "records.json").exists())
            self.assertFalse((workspace / "upgrade-checkpoint.json").exists())
            self.assertFalse((workspace / "snapshots.sqlite3").exists())

    def test_missing_parent_is_run_error_with_chain(self) -> None:
        document = mixed_document()
        document["runs"][2]["parent_run_id"] = "missing-parent"
        with self.assertRaises(RunError) as caught:
            parse_library(document)
        self.assertEqual(caught.exception.code, "E_RUN")
        self.assertEqual(caught.exception.details["run_id"], "ensemble")
        self.assertEqual(caught.exception.details["chain"], ["ensemble", "missing-parent"])
        self.assertEqual(caught.exception.details["constraint"], "parent_exists")

    def test_parent_cycle_is_run_error(self) -> None:
        first = current_record(run_id="a", parent_run_id="b")
        second = current_record(run_id="b", parent_run_id="a", seed=12)
        document = {
            "library_schema": 1,
            "library_id": "cycle",
            "runs": [first, second],
            "history": [],
        }
        with self.assertRaises(RunError) as caught:
            parse_library(document)
        self.assertEqual(caught.exception.details["constraint"], "parent_acyclic")
        self.assertEqual(caught.exception.details["chain"][0], caught.exception.details["chain"][-1])
        self.assertEqual(set(caught.exception.details["chain"]), {"a", "b"})

    def test_duplicate_current_run_id_is_run_error(self) -> None:
        document = mixed_document()
        duplicate = clone(document["runs"][0])
        duplicate["run_version"] = 2
        document["runs"].append(duplicate)
        with self.assertRaises(RunError) as caught:
            parse_library(document)
        self.assertEqual(caught.exception.details["constraint"], "run_id_unique")
        self.assertEqual(caught.exception.details["run_id"], "baseline")

    def test_run_versions_must_be_monotonic(self) -> None:
        current = current_record(run_id="baseline", run_version=1)
        history = clone(current)
        document = {
            "library_schema": 1,
            "library_id": "versions",
            "runs": [current],
            "history": [history],
        }
        with self.assertRaises(RunError) as caught:
            parse_library(document)
        self.assertEqual(caught.exception.details["constraint"], "run_version_monotonic")
        self.assertEqual(caught.exception.details["run_id"], "baseline")


if __name__ == "__main__":
    unittest.main()
