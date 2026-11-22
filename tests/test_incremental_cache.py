from __future__ import annotations

import unittest

from .helpers import TestWorkspace, ledger_is_sorted, report_for


class IncrementalCacheTests(unittest.TestCase):
    def test_edit_sequence_matches_full_rebuild(self) -> None:
        workspace = TestWorkspace()
        full_workspace = None
        try:
            workspace.prepare()
            baseline = workspace.run()
            self.assertEqual(baseline["status"], "ok")

            warm = workspace.run()
            self.assertEqual(warm["diagnostics"]["cache_summary"]["reused"], 36)
            self.assertEqual(warm["semantic_fingerprint"], baseline["semantic_fingerprint"])
            self.assertEqual(warm["global_version"], baseline["global_version"])

            workspace.replace_override(
                "region",
                [["east", "north", "south", "west"][index % 4] for index in range(workspace.columns["rows"])],
            )
            replaced = workspace.run()
            reused_cross = {
                event["sort_key"].split("/")[1]
                for event in replaced["diagnostics"]["reuse_ledger"]
                if event["stage"] == "cross_encoding"
            }
            self.assertIn("channel+device", reused_cross)
            self.assertIn("channel+tier", reused_cross)
            self.assertIn("device+tier", reused_cross)
            self.assertNotIn("channel+region", reused_cross)
            invalid_reduction = {
                event["sort_key"].split("/")[1]
                for event in replaced["diagnostics"]["invalidation_ledger"]
                if event["stage"] == "dimensionality_reduction"
            }
            self.assertEqual(invalid_reduction, {"pca", "random_projection", "variance"})

            config = workspace.config
            config["crossing"]["order"] = 3
            workspace.write_config(config)
            self.assertEqual(workspace.run()["status"], "ok")

            config = workspace.config
            config["crossing"]["whitelist"] = ["region", "device", "channel"]
            workspace.write_config(config)
            self.assertEqual(workspace.run()["status"], "ok")

            config = workspace.config
            config["crossing"]["cardinality_budget"] = 4
            workspace.write_config(config)
            self.assertEqual(workspace.run()["status"], "ok")

            config = workspace.config
            config["reduction"]["pca"]["dimensions"] = 2
            workspace.write_config(config)
            pca_changed = workspace.run()
            invalid_dim_paths = {
                event["sort_key"].split("/")[1]
                for event in pca_changed["diagnostics"]["invalidation_ledger"]
                if event["stage"] == "dimensionality_reduction"
            }
            self.assertEqual(invalid_dim_paths, {"pca"})
            reused_dim_paths = {
                event["sort_key"].split("/")[1]
                for event in pca_changed["diagnostics"]["reuse_ledger"]
                if event["stage"] == "dimensionality_reduction"
            }
            self.assertEqual(reused_dim_paths, {"random_projection", "variance"})

            config = workspace.config
            config["precision"]["quantization"]["tier"] = "coarse"
            workspace.write_config(config)
            incremental = workspace.run()
            self.assertEqual(incremental["status"], "ok")

            full_workspace = workspace.clone(cache="cache_full", report="reports/full.json")
            full_workspace.prepare()
            full = full_workspace.run()
            self.assertEqual(incremental["semantic_fingerprint"], full["semantic_fingerprint"])
            self.assertEqual(incremental["global_version"], full["global_version"])
            self.assertEqual(incremental["stage_versions"], full["stage_versions"])
            self.assertEqual(incremental["cross_encoding"], full["cross_encoding"])
            self.assertEqual(incremental["standardization"], full["standardization"])
            self.assertEqual(incremental["reduction"], full["reduction"])
            self.assertEqual(incremental["diagnostics"]["stage_diffs"], full["diagnostics"]["stage_diffs"])
            self.assertEqual(incremental["diagnostics"]["boundary_cases"], full["diagnostics"]["boundary_cases"])
            self.assertEqual(incremental["diagnostics"]["dependency_ledger"], full["diagnostics"]["dependency_ledger"])
            self.assertEqual(incremental["failures"], full["failures"])
            self.assertTrue(ledger_is_sorted(incremental, "reuse_ledger"))
            self.assertTrue(ledger_is_sorted(incremental, "invalidation_ledger"))
            self.assertTrue(ledger_is_sorted(full, "reuse_ledger"))
            self.assertTrue(ledger_is_sorted(full, "invalidation_ledger"))
        finally:
            workspace.close()
            if full_workspace is not None:
                full_workspace.close()


if __name__ == "__main__":
    unittest.main()
