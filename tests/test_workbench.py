"""Deterministic, incremental and recovery tests for the time-series workbench."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from tsbench.cli import compare_reports
from tsbench.config import load_config
from tsbench.errors import E_CONFIG_INVALID, E_SERIES_INVALID, ConfigError
from tsbench.model import error_metrics, fit_ols, run_gradient_descent
from tsbench.pipeline import SeriesProcessor
from tsbench.runner import WorkbenchRunner
from tsbench.synthetic import apply_operation, generate_series
from tsbench.transforms import build_feature_matrix, preprocess
from tsbench.updates import load_update_plan

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, "inputs", "experiment.yaml")
UPDATES = os.path.join(ROOT, "inputs", "updates.json")


def load_runner(output_dir: str, config_path: str = CONFIG) -> WorkbenchRunner:
    config = load_config(config_path)
    runner = WorkbenchRunner(config, output_dir)
    runner.initialize_series()
    return runner


class DeterminismTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="tsbench-")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_generation_is_seed_deterministic(self) -> None:
        config = load_config(CONFIG)
        from datetime import date

        first = generate_series(config.series_specs[0], date(2024, 1, 1))
        second = generate_series(config.series_specs[0], date(2024, 1, 1))
        self.assertEqual(first.values, second.values)
        self.assertEqual(first.timestamps, second.timestamps)

    def test_ols_and_gd_are_deterministic(self) -> None:
        rng = np.random.default_rng(3)
        x = rng.normal(size=(60, 4))
        y = rng.normal(size=60)
        tier = {"epochs": 25, "learning_rate": 0.01, "batch_size": 8}
        beta_a = fit_ols(x, y)["beta"]
        beta_b = fit_ols(x, y)["beta"]
        np.testing.assert_array_equal(beta_a, beta_b)
        trace_a = run_gradient_descent(x, y, tier)
        trace_b = run_gradient_descent(x, y, tier)
        self.assertEqual(
            [t["loss"] for t in trace_a], [t["loss"] for t in trace_b]
        )

    def test_error_metrics_finite(self) -> None:
        metrics = error_metrics(np.array([1.0, 2.0]), np.array([1.5, 1.5]))
        for key in ("mae", "rmse", "mape"):
            self.assertTrue(np.isfinite(metrics[key]))


class IncrementalConsistencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="tsbench-")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_full_equals_incremental_field_by_field(self) -> None:
        batches = load_update_plan(UPDATES)

        full_dir = os.path.join(self.tmp, "full")
        full_runner = load_runner(full_dir)
        full_runner.materialize_final_state(batches)
        full_report = full_runner.run_once(9999, {}, {}, {})

        inc_dir = os.path.join(self.tmp, "incr")
        inc_runner = load_runner(inc_dir)
        inc_reports = inc_runner.apply_batches(batches)
        final_report = inc_reports[-1]

        result = compare_reports(full_report, final_report)
        self.assertTrue(result["identical"], result["mismatches"])
        self.assertEqual(result["forecast_steps_covered"], 2)

        # Every fold must expose exactly two forecast steps for every series.
        for series in final_report["series_results"]:
            for fold in series["backtest"]["folds"]:
                self.assertEqual(len(fold["predictions"]), 2)
                self.assertEqual(len(fold["actuals"]), 2)

    def test_reuse_grows_after_first_batch(self) -> None:
        batches = load_update_plan(UPDATES)
        runner = load_runner(os.path.join(self.tmp, "incr"))
        reports = runner.apply_batches(batches)
        self.assertEqual(len(reports[0]["reuse_list"]), 0)
        self.assertGreater(len(reports[1]["reuse_list"]), 0)
        self.assertGreater(len(reports[2]["reuse_list"]), 0)

    def test_history_replace_only_invalidates_dependent_artifacts(self) -> None:
        runner = load_runner(os.path.join(self.tmp, "incr"))
        baseline = runner.run_once(0, {}, {}, {})
        before = {
            item["series_id"]: item["preprocess"]["fingerprint"]
            for item in baseline["series_results"]
        }

        series = runner.series_map["S-B"]
        apply_operation(
            series,
            {
                "type": "replace_range",
                "start": 30,
                "end": 31,
                "values": [series.values[30] + 1.0, series.values[31] - 1.0],
            },
            runner.config.block_size,
        )
        after_report = runner.run_once(1, {"S-B": ["frag-00001"]}, {"S-B": [30, 31]},
                                       {"S-B": ["replace_range"]})
        after = {
            item["series_id"]: item["preprocess"]["fingerprint"]
            for item in after_report["series_results"]
        }
        # Untouched series keep identical fingerprints and only reuse entries.
        self.assertEqual(before["S-A"], after["S-A"])
        self.assertEqual(before["S-C"], after["S-C"])
        self.assertNotEqual(before["S-B"], after["S-B"])
        audit_sa = next(
            item for item in after_report["series_results"] if item["series_id"] == "S-A"
        )
        self.assertTrue(
            all(label.startswith("preprocess:") or label.startswith("features:")
                or label.startswith("model:") or label.startswith("backtest:")
                for label in audit_sa["audit"]["reused"])
        )
        self.assertEqual(audit_sa["audit"]["recomputed"], [])


class ConcurrencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="tsbench-")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_concurrency_does_not_change_results(self) -> None:
        batches = load_update_plan(UPDATES)
        reports = {}
        for workers in (1, 4):
            runner = load_runner(os.path.join(self.tmp, f"w{workers}"))
            runner.params["concurrency"] = workers
            runner.materialize_final_state(batches)
            reports[workers] = runner.run_once(9999, {}, {}, {})
        result = compare_reports(reports[1], reports[4])
        self.assertTrue(result["identical"], result["mismatches"])

    def test_result_order_is_series_id_sorted(self) -> None:
        runner = load_runner(os.path.join(self.tmp, "order"))
        runner.params["concurrency"] = 4
        report = runner.run_once(0, {}, {}, {})
        ids = [item["series_id"] for item in report["series_results"]]
        self.assertEqual(ids, sorted(ids))


class ValidationTests(unittest.TestCase):
    def test_invalid_seasonal_period_raises_config_error(self) -> None:
        with self.assertRaises(ConfigError) as context:
            load_config(os.path.join(ROOT, "inputs", "invalid", "bad_config.yaml"))
        self.assertEqual(context.exception.code, E_CONFIG_INVALID)

    def test_missing_section_raises_config_error(self) -> None:
        with self.assertRaises(ConfigError) as context:
            load_config(os.path.join(ROOT, "inputs", "invalid", "missing_field.yaml"))
        self.assertEqual(context.exception.code, E_CONFIG_INVALID)

    def test_unknown_series_goes_to_failure_list(self) -> None:
        tmp = tempfile.mkdtemp(prefix="tsbench-")
        try:
            runner = load_runner(os.path.join(tmp, "x"))
            plan = load_update_plan(
                os.path.join(ROOT, "inputs", "invalid", "unknown_series_updates.json")
            )
            runner.apply_batches(plan)
            codes = {item["code"] for item in runner.failure_records}
            self.assertIn(E_SERIES_INVALID, codes)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_non_finite_value_rejected(self) -> None:
        runner = load_runner(tempfile.mkdtemp(prefix="tsbench-"))
        series = runner.series_map["S-A"]
        with self.assertRaises(Exception):
            apply_operation(
                series,
                {"type": "tail_correction", "values": [float("nan")]},
                runner.config.block_size,
            )


class RecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="tsbench-")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_config_mismatch_blocks_recovery(self) -> None:
        output = os.path.join(self.tmp, "state")
        runner = load_runner(output)
        runner.run_once(0, {}, {}, {})
        changed_path = os.path.join(self.tmp, "changed.yaml")
        with open(CONFIG, encoding="utf-8") as handle:
            text = handle.read()
        with open(changed_path, "w", encoding="utf-8") as handle:
            handle.write(text.replace("seasonal_period: 7", "seasonal_period: 9"))
        other = load_runner(output, changed_path)
        with self.assertRaises(ConfigError) as context:
            other.state.check_recoverable(other.config.config_hash, "run")
        self.assertEqual(context.exception.code, "E_CONFIG_MISMATCH")

    def test_recovery_point_reflects_manifest(self) -> None:
        output = os.path.join(self.tmp, "state")
        runner = load_runner(output)
        runner.run_once(0, {}, {}, {})
        point = runner.state.recovery_point()
        self.assertTrue(point["recovered"])
        self.assertEqual(point["global_version"], 1)
        self.assertEqual(
            point["completed_series"], sorted(runner.series_map)
        )


class FeatureMatrixTests(unittest.TestCase):
    def test_lag_matrix_alignment(self) -> None:
        rng = np.random.default_rng(11)
        x = np.cumsum(rng.normal(size=50))
        params = {
            "lag_orders": [1, 3],
            "diff_order": 1,
            "seasonal_period": 7,
            "seasonal_terms": 2,
            "rolling_window": 6,
            "forecast_steps": 2,
            "feature_rules": ["lag", "rolling_mean"],
        }
        pre = preprocess(x, params)
        feat = build_feature_matrix(pre, params)
        self.assertEqual(feat["matrix"].shape[1], 3)  # two lags + rolling mean
        self.assertEqual(feat["matrix"].shape[0], len(feat["row_indices"]))
        self.assertTrue(all(idx >= 3 for idx in feat["row_indices"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
