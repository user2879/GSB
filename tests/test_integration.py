from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from search_intent_workbench.runner import run_workbench_demo
from scripts.generate_samples import generate


ROOT = Path(__file__).resolve().parents[1]


class EndToEndTests(unittest.TestCase):
    def test_generator_is_byte_for_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            generate(Path(first), 20261123)
            generate(Path(second), 20261123)
            first_files = {
                path.relative_to(first): path.read_bytes()
                for path in Path(first).rglob("*")
                if path.is_file()
            }
            second_files = {
                path.relative_to(second): path.read_bytes()
                for path in Path(second).rglob("*")
                if path.is_file()
            }
            self.assertEqual(sorted(first_files), sorted(second_files))
            self.assertEqual(first_files, second_files)

    def test_demo_passes_and_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            summary = run_workbench_demo(
                data_dir=ROOT / "data" / "base",
                injections_file=ROOT / "data" / "injected" / "injections.json",
                snapshot_root=root / "snapshots",
                artifact_dir=root / "artifacts",
                workers=4,
                with_faults=True,
            )
            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(summary["global_version"], 9)
            self.assertEqual(summary["query_count"], 17)
            self.assertEqual(summary["command_count"], 8)
            self.assertTrue(summary["all_commands_equal"])
            self.assertTrue(summary["serial_concurrent_equal"])
            self.assertTrue(summary["recovery_ok"])
            self.assertEqual(summary["fault_hits"], ["I01", "I02", "I03", "I04"])

            report_path = root / "artifacts" / "diagnostic_report.json"
            html_path = root / "artifacts" / "diagnostic_report.html"
            self.assertTrue(report_path.is_file())
            self.assertTrue(html_path.is_file())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["query_count"], 17)
            self.assertIn("intent_distribution", report)
            self.assertIn("boundary_intent_samples", report)
            self.assertGreater(len(report["per_query"]), 0)
            self.assertIn("Query Intent Diagnosis Workbench", html_path.read_text(encoding="utf-8"))

            evolution = json.loads(
                (root / "artifacts" / "command_evolution.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(evolution), 8)
            self.assertTrue(all(item["incremental_equals_full"] for item in evolution))
            recovery = json.loads(
                (root / "artifacts" / "recovery_check.json").read_text(encoding="utf-8")
            )
            self.assertEqual(recovery[0]["status"], "RESUME")
            self.assertEqual(recovery[2]["status"], "E_SNAPSHOT_STALE")
            self.assertEqual(recovery[2]["artifact_count"], 0)
            self.assertEqual(recovery[4]["status"], "RESUME_STAGING")


if __name__ == "__main__":
    unittest.main()
