from __future__ import annotations

import unittest

from rope_extrap.analysis import diagnostic_consistency_view
from rope_extrap.engine import DiagnosticEngine, run_full_diagnosis
from rope_extrap.errors import BusinessError
from tests.helpers import load_config_and_samples


class EngineTests(unittest.TestCase):
    def test_full_report_has_required_sections_and_sorted_violations(self) -> None:
        config, samples, failures = load_config_and_samples(count=12)
        report = run_full_diagnosis(config, samples, loader_failures=failures)
        for key in (
            "samples",
            "score_statistics",
            "error_statistics",
            "over_tolerance_samples",
            "dimension_sensitivity_ranking",
            "failures",
            "business_code_counts",
            "cache",
        ):
            self.assertIn(key, report)
        self.assertEqual(report["summary"]["completed_samples"], 12)
        self.assertTrue(report["summary"]["is_complete"])
        values = [item["max_deviation"] for item in report["over_tolerance_samples"]]
        self.assertEqual(values, sorted(values, reverse=True))
        for sample in report["samples"]:
            self.assertIn("position_vector_summary", sample)
            self.assertIn("cache_keys", sample)
            self.assertIn("trace", sample)
            self.assertTrue(sample["scores"])
            self.assertEqual(sample["sample_interval"], "extrapolation" if sample["has_extrapolation"] else "in_training")

    def test_concurrency_is_capped_and_matches_serial_results(self) -> None:
        config, samples, _ = load_config_and_samples(count=8)
        concurrent = run_full_diagnosis(config.changed(concurrency=4), samples)
        serial = run_full_diagnosis(config.changed(concurrency=1), samples)
        self.assertLessEqual(concurrent["run"]["engine_max_observed_concurrency"], 4)
        self.assertEqual(concurrent["run"]["engine_max_observed_concurrency"], 4)
        self.assertEqual(serial["run"]["engine_max_observed_concurrency"], 1)
        concurrent_view = diagnostic_consistency_view(concurrent)
        serial_view = diagnostic_consistency_view(serial)
        concurrent_view["config"]["concurrency"] = 0
        serial_view["config"]["concurrency"] = 0
        concurrent_view["config"]["config_fingerprint"] = ""
        serial_view["config"]["config_fingerprint"] = ""
        concurrent_view["run"].pop("configured_concurrency", None)
        serial_view["run"].pop("configured_concurrency", None)
        self.assertEqual(concurrent_view, serial_view)

    def test_stage_exception_is_isolated(self) -> None:
        config, samples, _ = load_config_and_samples(count=4)

        def fail_one(sample):
            if sample.sample_id == "s001":
                raise BusinessError(
                    "E_INJECTED",
                    "injected stage failure",
                    stage="attention_score",
                    sample_id=sample.sample_id,
                )

        engine = DiagnosticEngine(
            config,
            samples,
            stage_hooks={"attention_score": fail_one},
        )
        plan = engine.run_full()
        report = engine.build_report(plan=plan)
        self.assertEqual(report["summary"]["completed_samples"], 3)
        self.assertEqual(report["summary"]["failed_samples"], 1)
        self.assertTrue(report["summary"]["is_complete"])
        self.assertEqual([row["sample_id"] for row in report["samples"]], ["s000", "s002", "s003"])
        self.assertEqual(report["failures"][0]["code"], "E_INJECTED")
        self.assertEqual(report["failures"][0]["stage"], "attention_score")
        self.assertEqual(report["cache"]["audit"]["summary"]["failed_samples"], ["s001"])


if __name__ == "__main__":
    unittest.main()
