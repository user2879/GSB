from __future__ import annotations

import unittest

from feature_pipeline.config import load_pipeline_config
from feature_pipeline.mini_yaml import loads
from feature_pipeline.pipeline import run_pipeline

from .helpers import TestWorkspace


class ConfigAndSampleTests(unittest.TestCase):
    def test_yaml_subset_and_inline_values(self) -> None:
        payload = loads(
            """
            seed: 7
            columns:
              - name: a
                type: category
                values: [x, y, z]
                missing_rate: 0.0
            nested:
              enabled: true
              values: [1, 2.5]
            """
        )
        self.assertEqual(payload["seed"], 7)
        self.assertEqual(payload["columns"][0]["values"], ["x", "y", "z"])
        self.assertTrue(payload["nested"]["enabled"])

    def test_validation_collects_business_codes_without_stopping(self) -> None:
        workspace = TestWorkspace()
        try:
            columns = workspace.columns
            columns["columns"][-1]["type"] = "text"
            workspace.write_columns(columns)
            config = workspace.config
            config["crossing"]["whitelist"] = ["region", "ghost", "tenure"]
            config["crossing"]["order"] = 0
            config["crossing"]["cardinality_budget"] = 0
            config["precision"]["quantization"]["tier"] = "ultra"
            workspace.write_config(config)
            loaded = load_pipeline_config(workspace.config_path, root=workspace.root)
            codes = {item["code"] for item in loaded.failures}
            self.assertIn("E_UNKNOWN_COLUMN", codes)
            self.assertIn("E_TYPE_MISMATCH", codes)
            self.assertIn("E_CROSS_ORDER", codes)
            self.assertIn("E_CARDINALITY_BUDGET", codes)
            self.assertIn("E_QUANTIZATION_PROFILE", codes)
            self.assertEqual([item["seq"] for item in loaded.failures], list(range(1, len(loaded.failures) + 1)))
            self.assertEqual(
                [descriptor["name"] for descriptor in loaded.descriptors],
                ["region", "device", "channel", "tier", "tenure", "spend"],
            )
        finally:
            workspace.close()

    def test_sample_generation_is_reused_and_reproducible(self) -> None:
        workspace = TestWorkspace()
        try:
            first = workspace.prepare()
            second = workspace.prepare()
            forced = workspace.prepare(force=True)
            self.assertFalse(first["reused"])
            self.assertTrue(second["reused"])
            self.assertEqual(first["checksum"], second["checksum"])
            self.assertEqual(first["checksum"], forced["checksum"])
            self.assertEqual(len(first["column_order"]), 7)
        finally:
            workspace.close()

    def test_failed_run_still_reports_other_loaded_columns(self) -> None:
        workspace = TestWorkspace()
        try:
            config = workspace.config
            config["crossing"]["whitelist"] = ["region", "ghost"]
            config["crossing"]["order"] = 0
            config["crossing"]["cardinality_budget"] = 0
            config["precision"]["quantization"]["tier"] = "invalid"
            workspace.write_config(config)
            report = run_pipeline(workspace.config_path, root=workspace.root)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["loaded_columns"], ["channel", "device", "region", "score", "spend", "tenure", "tier"])
            self.assertEqual(report["rows_loaded"], 120)
            self.assertIn("E_UNKNOWN_COLUMN", {item["code"] for item in report["failures"]})
        finally:
            workspace.close()


if __name__ == "__main__":
    unittest.main()
