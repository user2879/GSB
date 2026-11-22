from __future__ import annotations

import unittest

from feature_pipeline.metrics import attribute_boundary_cases

from .helpers import TestWorkspace


class PrecisionTests(unittest.TestCase):
    def test_four_stage_differences_and_first_exceeding_stage(self) -> None:
        workspace = TestWorkspace()
        try:
            workspace.prepare()
            report = workspace.run()
            self.assertEqual(
                [item["stage"] for item in report["diagnostics"]["stage_summary"]],
                ["cross_encoding", "standardization", "projection", "final_metrics"],
            )
            summary = {item["stage"]: item for item in report["diagnostics"]["stage_summary"]}
            self.assertEqual(summary["cross_encoding"]["max_abs_diff"], 0.0)
            self.assertGreater(summary["standardization"]["max_abs_diff"], 0.0)
            self.assertEqual(report["diagnostics"]["first_exceeded_stage"], "standardization")
            self.assertEqual(len(report["precision_fingerprints"]), 2)
        finally:
            workspace.close()

    def test_actual_quantization_boundary_attribution(self) -> None:
        workspace = TestWorkspace()
        try:
            workspace.prepare()
            baseline = workspace.run()
            pca = next(item for item in baseline["reduction"]["paths"] if item["path"] == "pca")
            single_scores = pca["tiers"]["single"]["scores"]
            double_scores = pca["tiers"]["double"]["scores"]
            row = min(range(len(double_scores)), key=lambda index: abs(double_scores[index]))
            threshold = (single_scores[row] + double_scores[row]) / 2.0
            config = workspace.config
            config["metrics"]["decision_threshold"] = threshold
            config["paths"]["cache"] = "cache_boundary"
            config["paths"]["report"] = "reports/boundary.json"
            workspace.write_config(config)
            report = workspace.run()
            cases = report["diagnostics"]["boundary_cases"]
            self.assertTrue(cases)
            self.assertIn("quantization", {case["source"] for case in cases})
            self.assertTrue(all(case["path"] in {"pca", "random_projection", "variance"} for case in cases))
        finally:
            workspace.close()

    def test_boundary_attribution_distinguishes_approximation(self) -> None:
        cases = attribute_boundary_cases(
            path="pca",
            single={"decisions": [0], "raw_decisions": [0], "scores": [0.1]},
            double={"decisions": [1], "raw_decisions": [1], "scores": [0.2]},
        )
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["source"], "dimensional_approximation")
        self.assertTrue(cases[0]["approximation_changed"])


if __name__ == "__main__":
    unittest.main()
