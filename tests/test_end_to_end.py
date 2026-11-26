from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from intentlab.errors import Codes
from intentlab.snapshot import SnapshotStore

from tests.helpers import ROOT, prepare_project, run, write_json


class EndToEndTests(unittest.TestCase):
    def test_full_sample_reports_intent_alignment_boundary_conflict_and_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = prepare_project(tmp, include_edits=True)
            first = run(paths)
            report = first.report
            self.assertEqual(report["summary"]["queries_processed"], 24)
            self.assertEqual(report["summary"]["queries_pending"], 0)
            self.assertEqual(report["summary"]["segments"], 14)
            self.assertEqual(report["summary"]["boundary_intents"], 1)
            self.assertEqual(report["summary"]["business_code_counts"], {})
            self.assertEqual(report["issues"], [])
            self.assertTrue(report["snapshot"]["committed"])
            self.assertEqual(report["rules"]["active_revision_ids"], ["r07"])
            self.assertEqual(report["rules"]["trigger_count"], 9)
            self.assertTrue(all(item["score"]["predicted_intent"] == item["ground_truth_intent"] for item in report["queries"]))
            q018 = next(item for item in report["queries"] if item["query_id"] == "q018")
            self.assertTrue(q018["score"]["boundary"])
            conflict_segments = [segment for segment in report["segments"] if segment["conflicts"]]
            self.assertEqual(len(conflict_segments), 1)
            self.assertEqual(conflict_segments[0]["query_ids"], ["q016", "q017", "q018"])
            self.assertTrue(
                all(stage_count == 0 for item in report["invalidation_by_revision"].values() for stage_count in item.values())
            )
            self.assertTrue(Path(first.json_path).exists())
            self.assertTrue(Path(first.markdown_path).exists())

            handle = SnapshotStore(paths.snapshot_root).load_current()
            self.assertIsNotNone(handle)
            assert handle is not None
            self.assertEqual(handle.manifest["stages"], ["rules", "match", "score", "merge"])
            self.assertTrue(handle.manifest["complete"])

            second = run(paths)
            self.assertEqual(second.report["summary"]["match_reuse_hits"], 24)
            self.assertEqual(second.report["summary"]["score_reuse_hits"], 24)
            self.assertEqual(second.report["summary"]["merge_reuse_hits"], 14)
            self.assertEqual(second.report["summary"]["reuse_hits"], 62)
            self.assertEqual(second.report["ledger"]["invalidations"], [])
            self.assertEqual(
                second.report["recovery"]["committed_generation"],
                report["snapshot"]["generation"],
            )

    def test_invalid_query_file_keeps_valid_part_and_reports_business_codes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = prepare_project(tmp, include_edits=False)
            invalid = json.loads((ROOT / "input" / "invalid_queries.json").read_text(encoding="utf-8"))
            write_json(paths.queries_path, invalid)
            result = run(paths)
            self.assertEqual(result.report["summary"]["queries_loaded"], 2)
            codes = result.report["summary"]["business_code_counts"]
            self.assertEqual(codes[Codes.QUERY_ID_DUPLICATE], 1)
            self.assertEqual(codes[Codes.QUERY_TERM_UNKNOWN], 1)
            self.assertEqual(codes[Codes.SLOT_RANGE], 1)
            self.assertEqual(len(result.report["issues"]), 3)
            self.assertTrue(result.report["snapshot"]["committed"])

    def test_cli_run_and_validate_are_usable_without_services(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = prepare_project(tmp, include_edits=True)
            common = [
                "--config",
                str(paths.config_path),
                "--vocabulary",
                str(paths.vocabulary_path),
                "--queries",
                str(paths.queries_path),
                "--rules",
                str(paths.rules_path),
                "--edits",
                str(paths.edits_path),
                "--snapshots",
                str(paths.snapshot_root),
                "--reports",
                str(paths.report_root),
            ]
            validate = subprocess.run(
                [sys.executable, "-m", "intentlab", "validate", *common],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60,
                check=False,
            )
            self.assertEqual(validate.returncode, 0, validate.stderr + validate.stdout)
            validation_payload = json.loads(validate.stdout)
            self.assertEqual(validation_payload["queries"], 24)
            self.assertEqual(validation_payload["active_revision_ids"], ["r07"])
            self.assertEqual(validation_payload["issues"], [])

            run_result = subprocess.run(
                [sys.executable, "-m", "intentlab", "run", *common],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60,
                check=False,
            )
            self.assertEqual(run_result.returncode, 0, run_result.stderr + run_result.stdout)
            self.assertIn("processed=24/24", run_result.stdout)
            self.assertIn("committed=True", run_result.stdout)


if __name__ == "__main__":
    unittest.main()
