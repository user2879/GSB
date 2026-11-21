from __future__ import annotations

import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

from ts_pipeline.compare import compare_reports
from dataclasses import replace

from ts_pipeline.config import ALLOWED_DECIMAL_PLACES, ALLOWED_QUANTIZATION_LEVELS, QuantizationConfig, load_config
from ts_pipeline.generator import (
    generate_fixture,
    replace_sequence_values,
    rewrite_config,
)
from ts_pipeline.injections import load_injections
from ts_pipeline.models import (
    AutoregressiveModel,
    MovingAverageModel,
    NaivePersistenceModel,
    TrainingContext,
)
from ts_pipeline.pipeline import PipelineOptions, run_pipeline
from ts_pipeline.quantization import evaluate_window, quantize_value
from ts_pipeline.samples import load_samples
from ts_pipeline.windows import build_window


def _run(
    config_path: Path,
    state_dir: Path,
    report_path: Path,
    *,
    reset: bool = False,
    stop_after_stage: str | None = None,
):
    if reset and state_dir.exists():
        shutil.rmtree(state_dir)
    return run_pipeline(
        PipelineOptions(
            config_path=config_path,
            state_dir=state_dir,
            report_path=report_path,
            stop_after_stage=stop_after_stage,
        )
    )


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class PipelineTests(unittest.TestCase):
    def test_config_samples_and_window_imputation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = generate_fixture(root / "data", force=True)
            loaded = load_config(fixture.config_path)
            self.assertFalse(loaded.failures)
            assert loaded.config is not None
            samples = load_samples(loaded.config.samples_path)
            self.assertFalse(samples.failures)
            injections = load_injections(
                loaded.config.injection_manifest_path,
                loaded.config,
                samples.samples,
            )
            self.assertFalse(injections.failures)

            sample = samples.samples["s-alpha"]
            first = build_window(sample, 0, loaded.config, ())
            second = build_window(sample, 0, loaded.config, ())
            self.assertEqual(first, second)
            self.assertTrue(first["imputation_records"])
            for record in first["imputation_records"]:
                self.assertTrue(record["source_positions_in_window"])
                self.assertTrue(
                    all(
                        position < loaded.config.backtest.window_length
                        for position in record["source_positions_in_window"]
                    )
                )
                self.assertEqual(record["source_window"]["series_id"], "s-alpha")
                self.assertEqual(
                    record["source_window"]["end_index_exclusive"],
                    loaded.config.backtest.window_length,
                )

    def test_three_models_are_deterministic_and_ar_handles_singular_matrix(self):
        values = (3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0)
        timestamps = tuple(index * 1000 for index in range(len(values)))
        context = TrainingContext(
            series_id="s-constant",
            window_index=0,
            train_start_index=0,
            train_end_index=len(values),
            train_start_ms=0,
            train_end_ms=timestamps[-1],
            values=values,
        )
        model_config = None
        with tempfile.TemporaryDirectory() as temporary:
            fixture = generate_fixture(Path(temporary) / "data", force=True)
            loaded_config = load_config(fixture.config_path)
            assert loaded_config.config is not None
            model_config = loaded_config.config.models
        assert model_config is not None
        for model in (
            NaivePersistenceModel(),
            MovingAverageModel(),
            AutoregressiveModel(),
        ):
            first = model.fit(values, timestamps, context, model_config)
            second = model.fit(values, timestamps, context, model_config)
            self.assertEqual(first, second)
            self.assertEqual(model.predict(first, 4), model.predict(second, 4))
            self.assertIn("training_window", first)
            self.assertIn("fitted_params", first)
            self.assertIn("residual_sequence", first)
        singular_config = replace(model_config, autoregressive_regularization=0.0)
        ar_artifact = AutoregressiveModel().fit(values, timestamps, context, singular_config)
        self.assertTrue(math.isfinite(ar_artifact["fitted_params"]["intercept"]))
        self.assertTrue(
            all(math.isfinite(value) for value in ar_artifact["fitted_params"]["coefficients"])
        )
        self.assertEqual(
            ar_artifact["fitted_params"]["solver"],
            "normal_equations_pseudoinverse",
        )

    def test_quantization_and_metric_accumulation(self):
        config = QuantizationConfig(fixed_decimal_places=2, quantization_level=1)
        self.assertEqual(quantize_value(1.235, config).bucket, 124)
        self.assertEqual(quantize_value(-1.235, config).bucket, -124)
        metrics = evaluate_window([1.234, -2.345], [1.0, -2.0], config, tolerance=0.2)
        self.assertEqual(metrics["points"][0]["prediction_fixed"], 123)
        self.assertEqual(metrics["points"][1]["prediction_fixed"], -235)
        self.assertEqual(metrics["points"][0]["actual_fixed"], 100)
        self.assertEqual(metrics["points"][1]["actual_fixed"], -200)
        self.assertAlmostEqual(metrics["mae"], (23 + 35) / 2 / 100.0)
        self.assertTrue(metrics["tolerance_exceeded"])
        self.assertEqual(metrics["first_exceeded"]["step"], 0)
        self.assertEqual(metrics["first_exceeded"]["step_number"], 1)
        self.assertEqual(ALLOWED_DECIMAL_PLACES, (0, 1, 2, 3, 4, 5, 6))
        self.assertIn(2, ALLOWED_QUANTIZATION_LEVELS)

    def test_full_run_report_and_injections(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = generate_fixture(root / "data", force=True)
            result = _run(
                fixture.config_path,
                root / "state",
                root / "report.json",
                reset=True,
            )
            self.assertEqual(result.exit_code, 0)
            report = result.report
            self.assertEqual(report["status"], "completed")
            self.assertEqual(set(report["models"]), {
                "autoregressive",
                "moving_average",
                "naive_persistence",
            })
            for model in report["models"].values():
                self.assertTrue(model["windows"])
                for window in model["windows"]:
                    self.assertIn("training_window", window)
                    self.assertIn("fitted_params", window)
                    self.assertIn("residual_sequence", window)
            hits = {hit["injection_id"]: hit for hit in report["injections"]["hits"]}
            self.assertEqual(hits["INJ-001"]["business_code"], "E_SAMPLE_INVALID")
            self.assertEqual(hits["INJ-002"]["business_code"], "E_SAMPLE_INVALID")
            self.assertEqual(hits["INJ-003"]["business_code"], "E_CONFIG_INVALID")
            self.assertEqual(hits["INJ-004"]["business_code"], "E_CONFIG_INVALID")
            self.assertEqual(hits["INJ-004"]["models"], ["moving_average"])
            self.assertEqual(report["injections"]["unhit_injection_ids"], [])
            codes = {failure["code"] for failure in report["failures"]}
            self.assertIn("E_SAMPLE_INVALID", codes)
            self.assertIn("E_CONFIG_INVALID", codes)
            failure_keys = [
                (
                    failure["sequence_id"] or "",
                    failure["window_index"] if failure["window_index"] is not None else -1,
                    failure["model_name"] or "",
                    failure["stage"],
                    failure["code"],
                )
                for failure in report["failures"]
            ]
            self.assertEqual(failure_keys, sorted(failure_keys))
            self.assertIn("snapshot", report)
            self.assertIn("stage_timings_ms", report)

    def test_incremental_edits_match_full_rebuild_and_reuse_layers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = generate_fixture(root / "data", force=True)
            state = root / "state"
            initial = _run(fixture.config_path, state, root / "initial.json", reset=True)
            self.assertEqual(initial.exit_code, 0)

            replace_sequence_values(fixture.samples_path, "s-beta", seed=fixture.seed)
            sequence_report = _run(fixture.config_path, state, root / "sequence.json").report
            invalidation = sequence_report["incremental_audit"]["invalidation"]
            self.assertGreater(invalidation["invalidated_count"], 0)
            self.assertGreater(invalidation["unchanged_count"], 0)
            self.assertEqual(
                {item["series_id"] for item in invalidation["invalidated_tasks"]},
                {"s-beta"},
            )
            self.assertEqual(
                {item["reason"] for item in invalidation["invalidated_tasks"]},
                {"sequence_content_changed"},
            )

            rewrite_config(
                fixture.config_path,
                models={
                    "moving_average_window": 4,
                    "feature_lag_order": 4,
                    "autoregressive_regularization": 1.0e-5,
                },
            )
            model_report = _run(fixture.config_path, state, root / "models.json").report
            model_invalidation = model_report["incremental_audit"]["invalidation"]
            self.assertEqual(
                {item["model_name"] for item in model_invalidation["invalidated_tasks"]},
                {"moving_average", "autoregressive"},
            )
            self.assertEqual(
                {item["reason"] for item in model_invalidation["invalidated_tasks"]},
                {"model_parameters_changed"},
            )

            rewrite_config(fixture.config_path, quantization={"quantization_level": 2})
            quantization_report = _run(
                fixture.config_path, state, root / "quantization.json"
            ).report
            self.assertEqual(
                quantization_report["incremental_audit"]["training_computed"],
                [],
            )
            self.assertTrue(quantization_report["incremental_audit"]["training_reused"])

            rewrite_config(
                fixture.config_path,
                backtest={"window_length": 10, "step": 3},
            )
            final_incremental = _run(
                fixture.config_path, state, root / "final_incremental.json"
            ).report
            final_dataset = root / "final_dataset"
            shutil.copytree(root / "data", final_dataset)
            full = _run(
                final_dataset / "config.yaml",
                root / "full_state",
                root / "full.json",
                reset=True,
            ).report
            parity = compare_reports(final_incremental, full)
            self.assertTrue(parity["equal"], json.dumps(parity["differences"], indent=2))

    def test_concurrency_budget_and_quota_are_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = generate_fixture(root / "data", force=True)
            report4 = _run(
                fixture.config_path,
                root / "state4",
                root / "report4.json",
                reset=True,
            ).report
            concurrent_dataset = root / "concurrent_data"
            shutil.copytree(root / "data", concurrent_dataset)
            rewrite_config(concurrent_dataset / "config.yaml", scheduler={"concurrency": 1})
            report1 = _run(
                concurrent_dataset / "config.yaml",
                root / "state1",
                root / "report1.json",
                reset=True,
            ).report
            self.assertTrue(compare_reports(report1, report4)["equal"])

            quota_dataset = root / "quota_data"
            shutil.copytree(root / "data", quota_dataset)
            rewrite_config(
                quota_dataset / "config.yaml",
                scheduler={"per_series_point_quota": 16, "batch_point_budget": 500000},
            )
            quota = _run(
                quota_dataset / "config.yaml",
                root / "quota_state",
                root / "quota.json",
                reset=True,
            ).report
            self.assertIn(
                "E_SERIES_QUOTA_EXHAUSTED",
                {failure["code"] for failure in quota["failures"]},
            )

            budget_dataset = root / "budget_data"
            shutil.copytree(root / "data", budget_dataset)
            rewrite_config(
                budget_dataset / "config.yaml",
                scheduler={"per_series_point_quota": 500000, "batch_point_budget": 20},
            )
            budget = _run(
                budget_dataset / "config.yaml",
                root / "budget_state",
                root / "budget.json",
                reset=True,
            ).report
            self.assertIn(
                "E_POINT_BUDGET_EXHAUSTED",
                {failure["code"] for failure in budget["failures"]},
            )
            self.assertIsNotNone(budget["scheduler"]["pending_cursor"])

    def test_snapshot_interrupt_resume_matches_full(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = generate_fixture(root / "data", force=True)
            full = _run(
                fixture.config_path,
                root / "full_state",
                root / "full.json",
                reset=True,
            ).report
            interrupted = _run(
                fixture.config_path,
                root / "state",
                root / "interrupted.json",
                reset=True,
                stop_after_stage="windows",
            )
            self.assertEqual(interrupted.exit_code, 0)
            self.assertEqual(interrupted.report["status"], "interrupted")
            manifest = _read(root / "state" / "snapshot" / "manifest.json")
            self.assertEqual(manifest["stages"]["load"]["status"], "complete")
            self.assertEqual(manifest["stages"]["windows"]["status"], "complete")
            self.assertNotIn("metrics", manifest["stages"])
            resumed = _run(
                fixture.config_path,
                root / "state",
                root / "resumed.json",
            ).report
            self.assertTrue(compare_reports(resumed, full)["equal"])
            self.assertTrue(resumed["incremental_audit"]["window_reused"])
            self.assertEqual(resumed["incremental_audit"]["window_computed"], [])

    def test_only_committed_snapshot_artifacts_are_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = generate_fixture(root / "data", force=True)
            state = root / "state"
            first = _run(fixture.config_path, state, root / "first.json", reset=True)
            self.assertEqual(first.exit_code, 0)
            manifest_path = state / "snapshot" / "manifest.json"
            manifest = _read(manifest_path)
            window_entries = [
                entry
                for entry in manifest["cache_inventory"]
                if entry["namespace"] == "window"
            ]
            self.assertTrue(window_entries)
            removed = window_entries[0]
            manifest["cache_inventory"] = [
                entry
                for entry in manifest["cache_inventory"]
                if not (
                    entry["namespace"] == removed["namespace"]
                    and entry["key"] == removed["key"]
                )
            ]
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            second = _run(fixture.config_path, state, root / "second.json").report
            computed = {
                (item["series_id"], item["window_index"])
                for item in second["incremental_audit"]["window_computed"]
            }
            reused = {
                (item["series_id"], item["window_index"])
                for item in second["incremental_audit"]["window_reused"]
            }
            self.assertEqual(len(computed), 1)
            self.assertTrue(reused)
            self.assertFalse(computed & reused)

    def test_stale_snapshot_and_invalid_inputs_continue(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = generate_fixture(root / "data", force=True)
            samples_path = fixture.samples_path
            with samples_path.open("a", encoding="utf-8") as handle:
                handle.write('{"sequence_id":"bad-line","timestamps_ms":[1,3],"values":[1,"oops"]}\n')
            injection_data = _read(fixture.injections_path)
            injection_data["injections"].append(
                {
                    "id": "INJ-BAD-REF",
                    "kind": "non_finite_observation",
                    "sequence_id": "does-not-exist",
                    "window_index": 0,
                    "position": 0,
                }
            )
            fixture.injections_path.write_text(json.dumps(injection_data), encoding="utf-8")
            first = _run(
                fixture.config_path,
                root / "state",
                root / "first.json",
                reset=True,
            ).report
            self.assertIn(
                "E_SAMPLE_INVALID",
                {failure["code"] for failure in first["failures"]},
            )
            self.assertIn(
                "E_CONFIG_INVALID",
                {failure["code"] for failure in first["failures"]},
            )
            self.assertTrue(
                any(
                    failure["stage"] == "injection_validation"
                    for failure in first["failures"]
                )
            )
            self.assertEqual(set(hit["business_code"] for hit in first["injections"]["hits"]),
                             {"E_SAMPLE_INVALID", "E_CONFIG_INVALID"})

            rewrite_config(fixture.config_path, quantization={"quantization_level": 2})
            second = _run(
                fixture.config_path,
                root / "state",
                root / "second.json",
            ).report
            self.assertEqual(second["snapshot"]["status_code"], "E_SNAPSHOT_STALE")
            self.assertTrue(second["snapshot"]["stale"])


if __name__ == "__main__":
    unittest.main()
