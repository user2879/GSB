from __future__ import annotations

import unittest

from .helpers import TestWorkspace, report_for


class ConcurrencyAndRecoveryTests(unittest.TestCase):
    def test_serial_and_parallel_results_are_identical(self) -> None:
        serial_workspace = TestWorkspace()
        parallel_workspace = TestWorkspace()
        try:
            serial_config = serial_workspace.config
            serial_config["runtime"]["max_workers"] = 1
            serial_workspace.write_config(serial_config)
            parallel_config = parallel_workspace.config
            parallel_config["runtime"]["max_workers"] = 4
            parallel_workspace.write_config(parallel_config)
            serial_workspace.prepare()
            parallel_workspace.prepare()
            serial = serial_workspace.run()
            parallel = parallel_workspace.run()
            self.assertEqual(serial["semantic_fingerprint"], parallel["semantic_fingerprint"])
            self.assertEqual(serial["stage_versions"], parallel["stage_versions"])
            self.assertEqual(
                serial["diagnostics"]["stage_diffs"], parallel["diagnostics"]["stage_diffs"]
            )
            self.assertLessEqual(serial["runtime"]["max_observed_workers"], 1)
            self.assertGreaterEqual(serial["runtime"]["max_observed_workers"], 1)
            self.assertLessEqual(parallel["runtime"]["max_observed_workers"], 4)
            self.assertGreater(parallel["runtime"]["max_observed_workers"], 1)
        finally:
            serial_workspace.close()
            parallel_workspace.close()

    def test_broken_current_pointer_and_abandoned_staging_recover(self) -> None:
        workspace = TestWorkspace()
        try:
            workspace.prepare()
            baseline = workspace.run()
            stage_root = workspace.root / "cache" / "stages" / "standardization"
            (stage_root / "current").unlink()
            stale = workspace.root / ".tmp" / ".stage-build-standardization-stale"
            stale.mkdir(parents=True)
            (stale / "partial.txt").write_text("incomplete", encoding="utf-8")
            recovered = workspace.run()
            self.assertFalse(stale.exists())
            actions = {(item["action"], item.get("stage")) for item in recovered["recovery_actions"]}
            self.assertIn(("restore_current_pointer", "standardization"), actions)
            self.assertEqual(recovered["semantic_fingerprint"], baseline["semantic_fingerprint"])
            self.assertEqual(recovered["stage_versions"], baseline["stage_versions"])
            self.assertTrue((stage_root / "current").exists())
        finally:
            workspace.close()

    def test_failed_override_is_reported_without_pipeline_stages(self) -> None:
        workspace = TestWorkspace()
        try:
            workspace.prepare()
            workspace.replace_override("region", ["east"])
            report = workspace.run()
            self.assertEqual(report["status"], "failed")
            self.assertIn("E_DATA", {item["code"] for item in report["failures"]})
        finally:
            workspace.close()


if __name__ == "__main__":
    unittest.main()
