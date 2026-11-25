from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fixedseason.experiment import (
    compare_reports,
    run_fresh_build,
    run_incremental_experiment,
)
from fixedseason.loader import load_config, load_series_jsonl
from fixedseason.synthetic import generate_example_workspace


class IncrementalExperimentTests(unittest.TestCase):
    def test_edit_then_incremental_final_equals_fresh_final(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture_dir = root / "fixtures"
            paths = generate_example_workspace(fixture_dir, force=True)
            series, series_issues = load_series_jsonl(paths["series"])
            config, config_issues = load_config(paths["config"])
            assert config is not None
            incremental = run_incremental_experiment(
                root / "incremental",
                series,
                config,
                initial_issues=series_issues + config_issues,
                commands_path=paths["commands"],
                mode="concurrent",
                fresh=True,
            )
            self.assertEqual(len(incremental.reports), 10)
            self.assertEqual(incremental.final_config.version, 9)
            self.assertEqual(incremental.trace[4]["applied"], False)
            self.assertEqual(
                [issue.code for issue in incremental.issues], ["E_COMMAND_INVALID"]
            )
            fresh = run_fresh_build(
                root / "fresh",
                incremental.final_series,
                incremental.final_config,
                validation_issues=incremental.issues,
                mode="concurrent",
            )
            equal, differences = compare_reports(incremental.final_report, fresh)
            self.assertTrue(equal, differences[:5])
            self.assertEqual(
                [item["series_id"] for item in incremental.final_report["series"]],
                [item["series_id"] for item in fresh["series"]],
            )
            self.assertEqual(
                incremental.final_report["failures"], fresh["failures"]
            )
            self.assertEqual(
                incremental.final_report["pending"], fresh["pending"]
            )
            self.assertEqual(
                incremental.final_report["reuse_ledger"], fresh["reuse_ledger"]
            )
            # The replaced series must be invalidated while unaffected series
            # are served from persistent stage products in that step.
            replace_report = incremental.reports[2]
            self.assertEqual(
                {
                    item["series_id"]: item["action"]
                    for item in replace_report["cache_ledger"]
                    if item["stage"] == "coefficients"
                }["sku-alpha"],
                "miss",
            )
            self.assertEqual(
                {
                    item["series_id"]: item["action"]
                    for item in replace_report["cache_ledger"]
                    if item["stage"] == "coefficients"
                }["sku-bravo"],
                "hit",
            )
            final_path = Path(temp) / "incremental" / "reports" / "step-009-v9.json"
            self.assertTrue(final_path.exists())
            snapshot_manifest = json.loads(
                (
                    Path(temp)
                    / "incremental"
                    / "snapshot"
                    / "current"
                    / "manifest.json"
                ).read_text(encoding="utf-8")
            )
            self.assertTrue(snapshot_manifest["complete"])


if __name__ == "__main__":
    unittest.main()
