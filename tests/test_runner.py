from __future__ import annotations

import copy
import json
import tempfile
import threading
import unittest
from pathlib import Path

import yaml

from tabular_lab.errors import E_SNAPSHOT_STALE, LabError
from tabular_lab.runner import run_experiment
from tests.helpers import LabFixture, model_by_name


def write_config(fixture: LabFixture, config: dict, name: str) -> Path:
    path = fixture.root / name
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


class RunnerTests(unittest.TestCase):
    def test_clean_run_then_full_cache_hit_and_snapshot_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LabFixture(Path(temporary))
            state = fixture.root / "state"
            first = run_experiment(fixture.config_path, state_dir=state)
            self.assertEqual(first["status"], "ok")
            self.assertEqual([model["name"] for model in first["models"]], ["elastic_balanced", "lasso_compact", "ridge_core"])
            self.assertEqual(first["validation_summary"]["model_count"], 3)
            self.assertEqual(first["snapshot"]["consumed_epochs"], 18)
            for model in first["models"]:
                self.assertEqual(model["status"], "max_epochs")
                self.assertEqual(len(model["loss_trajectory"]), 6)
                self.assertIn("coefficient_snapshots", model)
            self.assertEqual(set(first["snapshot"]["stage_versions"]), {"load", "standardize", "train", "convert"})

            manifest = json.loads(Path(first["snapshot"]["path"], "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["global_version"], 1)
            self.assertEqual(manifest["completed_models"], ["elastic_balanced", "lasso_compact", "ridge_core"])
            self.assertEqual(manifest["failed_models"], [])
            self.assertEqual(manifest["pending_models"], [])
            self.assertEqual(manifest["consumed_epochs"], 18)
            self.assertEqual(manifest["stage_versions"], first["snapshot"]["stage_versions"])
            self.assertEqual(
                manifest["model_refs"]["ridge_core"]["feature_names"],
                ["income", "tenure", "utilization", "bureau_score", "region=south", "region=west"],
            )
            self.assertIn("sample_fingerprint", manifest["model_refs"]["ridge_core"])

            second = run_experiment(fixture.config_path, state_dir=state)
            self.assertEqual(second["run_mode"], "incremental")
            self.assertEqual(second["reuse_list"], ["elastic_balanced", "lasso_compact", "ridge_core"])
            self.assertEqual(second["retrained_models"], [])
            self.assertEqual(second["reconverted_models"], [])
            self.assertEqual(second["snapshot"]["global_version"], 2)
            self.assertEqual(first["semantic_digest"], second["semantic_digest"])
            self.assertEqual(first["snapshot"]["stage_versions"], second["snapshot"]["stage_versions"])

    def test_incremental_edit_sequence_matches_clean_full_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LabFixture(Path(temporary))
            incremental_state = fixture.root / "incremental"
            base_config = copy.deepcopy(fixture.config)
            base_config["data"] = str(fixture.data_path)
            run_experiment(fixture.config_path, state_dir=incremental_state)

            penalty_config = copy.deepcopy(base_config)
            for model in penalty_config["models"]:
                if model["name"] == "lasso_compact":
                    model["penalty"] = 0.031
            penalty_path = write_config(fixture, penalty_config, "penalty.yaml")
            penalty_report = run_experiment(penalty_path, state_dir=incremental_state)
            self.assertEqual(penalty_report["retrained_models"], ["lasso_compact"])
            self.assertEqual(penalty_report["reused_models"], ["elastic_balanced", "ridge_core"])

            v1_config = copy.deepcopy(penalty_config)
            v1_config["exchange_version"] = "v1"
            v1_path = write_config(fixture, v1_config, "v1.yaml")
            v1_report = run_experiment(v1_path, state_dir=incremental_state)
            self.assertEqual(v1_report["audit"]["training_reused"], ["elastic_balanced", "lasso_compact", "ridge_core"])
            self.assertEqual(v1_report["reconverted_models"], ["elastic_balanced", "lasso_compact", "ridge_core"])
            self.assertTrue(Path(v1_report["snapshot"]["path"], "exports/models.v1.txt").exists())
            self.assertEqual(len(list(Path(v1_report["snapshot"]["path"], "sidecars").glob("*.json"))), 3)

            variant_data = copy.deepcopy(fixture.data_spec)
            for column in variant_data["columns"]:
                if column["name"] == "utilization":
                    column["variant"] = 1
            variant_data_path = fixture.root / "data-variant.yaml"
            variant_data_path.write_text(yaml.safe_dump(variant_data, sort_keys=False), encoding="utf-8")
            column_config = copy.deepcopy(v1_config)
            column_config["data"] = str(variant_data_path)
            column_path = write_config(fixture, column_config, "column.yaml")
            column_report = run_experiment(column_path, state_dir=incremental_state)
            self.assertEqual(column_report["retrained_models"], ["elastic_balanced", "ridge_core"])
            self.assertEqual(column_report["audit"]["training_reused"], ["lasso_compact"])

            final_config = copy.deepcopy(column_config)
            for model in final_config["models"]:
                if model["name"] == "ridge_core":
                    model["features"] = [feature for feature in model["features"] if feature != "region"]
            final_path = write_config(fixture, final_config, "final.yaml")
            final_incremental = run_experiment(final_path, state_dir=incremental_state)
            self.assertEqual(final_incremental["retrained_models"], ["ridge_core"])

            full_state = fixture.root / "full"
            final_full = run_experiment(final_path, state_dir=full_state)
            self.assertEqual(final_incremental["semantic_digest"], final_full["semantic_digest"])
            self.assertEqual(final_incremental["snapshot"]["stage_versions"], final_full["snapshot"]["stage_versions"])
            self.assertEqual(final_incremental["failures"], final_full["failures"])
            for left, right in zip(final_incremental["models"], final_full["models"]):
                self.assertEqual(left["coefficients"], right["coefficients"])
                self.assertEqual(left["loss_trajectory"], right["loss_trajectory"])
                self.assertEqual(left["validation_metrics"], right["validation_metrics"])

            incremental_path = Path(final_incremental["snapshot"]["path"])
            full_path = Path(final_full["snapshot"]["path"])
            self.assertEqual(
                (incremental_path / "exports/models.v1.txt").read_bytes(),
                (full_path / "exports/models.v1.txt").read_bytes(),
            )
            incremental_sidecars = {
                path.name: path.read_bytes() for path in (incremental_path / "sidecars").glob("*.json")
            }
            full_sidecars = {path.name: path.read_bytes() for path in (full_path / "sidecars").glob("*.json")}
            self.assertEqual(incremental_sidecars, full_sidecars)

    def test_whitelist_order_is_a_full_cache_hit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LabFixture(Path(temporary))
            state = fixture.root / "state"
            first = run_experiment(fixture.config_path, state_dir=state)
            reordered = copy.deepcopy(fixture.config)
            reordered["models"] = list(reversed(reordered["models"]))
            for model in reordered["models"]:
                model["features"] = list(reversed(model["features"]))
            reordered_path = write_config(fixture, reordered, "reordered.yaml")
            second = run_experiment(reordered_path, state_dir=state)
            self.assertEqual(first["config_hash"], second["config_hash"])
            self.assertEqual(second["reuse_list"], ["elastic_balanced", "lasso_compact", "ridge_core"])
            self.assertEqual(second["retrained_models"], [])
            self.assertEqual(first["semantic_digest"], second["semantic_digest"])

    def test_global_budget_and_model_quota_are_stable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LabFixture(Path(temporary))
            config = copy.deepcopy(fixture.config)
            config["budget"] = {"global_epochs": 10, "per_model_epochs": 8}
            path = write_config(fixture, config, "budget.yaml")
            report = run_experiment(path, state_dir=fixture.root / "state")
            self.assertEqual(report["status"], "partial")
            self.assertEqual(report["exit_code"], 4)
            self.assertEqual([failure["model"] for failure in report["failures"]], ["lasso_compact", "ridge_core"])
            self.assertEqual(report["failures"][0]["code"], "E_MODEL_BUDGET_EXHAUSTED")
            self.assertEqual(report["failures"][1]["code"], "E_TRAIN_BUDGET_EXHAUSTED")
            self.assertEqual(report["pending_models"], ["ridge_core"])
            self.assertEqual(report["validation_summary"]["model_count"], 1)
            self.assertEqual(report["snapshot"]["completed_models"], ["elastic_balanced"])
            self.assertEqual(report["snapshot"]["consumed_epochs"], 8)
            self.assertEqual(report["diagnostics"]["budget"]["reservations"]["lasso_compact"], 2)
            self.assertEqual(report["diagnostics"]["budget"]["reservations"]["ridge_core"], 0)

    def test_concurrent_and_serial_runs_are_equivalent_with_isolated_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LabFixture(Path(temporary))
            base = copy.deepcopy(fixture.config)
            base["training"]["epochs"] = 5
            base["training"]["snapshot_every_batches"] = 2
            base["faults"] = [
                {
                    "model": "lasso_compact",
                    "stage": "conversion",
                    "exception": "RuntimeError",
                    "message": "isolated conversion failure",
                }
            ]
            serial = copy.deepcopy(base)
            serial["runtime"]["max_concurrency"] = 1
            parallel = copy.deepcopy(base)
            parallel["runtime"]["max_concurrency"] = 4
            serial_path = write_config(fixture, serial, "serial.yaml")
            parallel_path = write_config(fixture, parallel, "parallel.yaml")

            serial_report = run_experiment(serial_path, state_dir=fixture.root / "serial-state")
            parallel_report = run_experiment(parallel_path, state_dir=fixture.root / "parallel-state")
            self.assertEqual(serial_report["semantic_digest"], parallel_report["semantic_digest"])
            self.assertEqual(serial_report["config_hash"], parallel_report["config_hash"])
            self.assertEqual([item["model"] for item in serial_report["failures"]], ["lasso_compact"])
            self.assertEqual([item["model"] for item in parallel_report["failures"]], ["lasso_compact"])
            self.assertEqual(serial_report["failures"][0]["stage"], "conversion")
            self.assertEqual(serial_report["validation_summary"]["model_count"], 2)
            for left, right in zip(serial_report["models"], parallel_report["models"]):
                self.assertEqual(left["coefficients"], right["coefficients"])
                self.assertEqual(left["loss_trajectory"], right["loss_trajectory"])

    def test_interrupted_run_commits_then_resume_matches_clean_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LabFixture(Path(temporary))
            state = fixture.root / "resume-state"
            stop = threading.Event()
            stop.set()
            interrupted = run_experiment(fixture.config_path, state_dir=state, stop_event=stop)
            self.assertEqual(interrupted["status"], "interrupted")
            self.assertEqual(interrupted["exit_code"], 130)
            self.assertEqual(interrupted["pending_models"], ["elastic_balanced", "lasso_compact", "ridge_core"])
            manifest = json.loads(Path(interrupted["snapshot"]["path"], "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["final_status"], "interrupted")

            resumed = run_experiment(fixture.config_path, state_dir=state)
            full = run_experiment(fixture.config_path, state_dir=fixture.root / "full-state")
            self.assertEqual(resumed["semantic_digest"], full["semantic_digest"])
            for left, right in zip(resumed["models"], full["models"]):
                self.assertEqual(left["coefficients"], right["coefficients"])
                self.assertEqual(left["loss_trajectory"], right["loss_trajectory"])

    def test_strict_resume_rejects_stale_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LabFixture(Path(temporary))
            state = fixture.root / "state"
            run_experiment(fixture.config_path, state_dir=state)
            changed = copy.deepcopy(fixture.config)
            changed["models"][1]["penalty"] = 0.02
            changed_path = write_config(fixture, changed, "changed.yaml")
            with self.assertRaises(LabError) as context:
                run_experiment(changed_path, state_dir=state, resume=True)
            self.assertEqual(context.exception.code, E_SNAPSHOT_STALE)
            incremental = run_experiment(changed_path, state_dir=state)
            self.assertEqual(incremental["retrained_models"], ["lasso_compact"])

    def test_v1_conversion_tolerance_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LabFixture(Path(temporary))
            config = copy.deepcopy(fixture.config)
            config["exchange_version"] = "v1"
            config["conversion_tolerance"] = 0.0
            path = write_config(fixture, config, "strict-tolerance.yaml")
            report = run_experiment(path, state_dir=fixture.root / "state")
            self.assertEqual(report["status"], "partial")
            self.assertEqual(report["validation_summary"]["model_count"], 0)
            self.assertEqual(len(report["failures"]), 3)
            self.assertTrue(all(failure["stage"] == "conversion" for failure in report["failures"]))
            self.assertTrue(all(failure["code"] == "E_MODEL_EXCEPTION" for failure in report["failures"]))
            self.assertEqual(report["audit"]["training_retrained"], ["elastic_balanced", "lasso_compact", "ridge_core"])


if __name__ == "__main__":
    unittest.main()
