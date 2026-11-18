from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from probcal.calibration import build_calibration_stage
from probcal.config import fixed_point
from probcal.metrics import _calibration_metric, _threshold_metric
from probcal.models import build_model_stage
from probcal.pipeline import core_digest, report_core
from probcal.service import edit_service, run_service
from probcal.snapshot import SnapshotStore


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "calibration.yaml"
BASE_SPEC = ROOT / "data" / "data_spec.yaml"
BASE_OVERRIDES = ROOT / "data" / "sample_overrides.yaml"


def create_workspace(
    root: Path,
    workers: int = 4,
    mutate_config=None,
) -> dict[str, Path]:
    workspace = root / f"workspace-w{workers}"
    workspace.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    config["concurrency"]["max_workers"] = workers
    config["data"]["spec_file"] = str(BASE_SPEC.resolve())
    config["data"]["overrides_file"] = str((workspace / "sample_overrides.yaml").resolve())
    if mutate_config is not None:
        mutate_config(config)
    config_path = workspace / "calibration.yaml"
    config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    shutil.copyfile(BASE_OVERRIDES, workspace / "sample_overrides.yaml")
    return {
        "workspace": workspace,
        "config": config_path,
        "overrides": workspace / "sample_overrides.yaml",
    }


def write_edit(path: Path, operations: list[dict]) -> Path:
    path.write_text(
        yaml.safe_dump({"operations": operations}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def run_ok(config: Path, state: Path):
    report = run_service(config, state)
    if report["status"] != "ok":
        raise AssertionError(json.dumps(report, ensure_ascii=False, indent=2))
    return report


class ConfigValidationTests(unittest.TestCase):
    def test_required_validation_errors_are_aggregated_without_crash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            def mutate(config):
                config["features"]["numeric"].append("does_not_exist")
                config["features"]["numeric"].append("region")
                config["target"] = "missing_target"
                config["calibrators"]["histogram"]["bins"] = 0
                config["evaluation"]["threshold_step"] = "0"
                config["concurrency"]["max_workers"] = 0

            paths = create_workspace(Path(temporary), workers=1, mutate_config=mutate)
            report = run_service(paths["config"], Path(temporary) / "state")
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["business_code"], "E_CONFIG_INVALID")
            self.assertTrue(all(item["code"] == "E_CONFIG_INVALID" for item in report["failures"]))
            fields = {item["field"] for item in report["failures"]}
            self.assertIn("features.numeric.does_not_exist", fields)
            self.assertIn("features.numeric.region", fields)
            self.assertIn("target", fields)
            self.assertIn("calibrators.histogram.bins", fields)
            self.assertIn("evaluation.threshold_step", fields)
            self.assertIn("concurrency.max_workers", fields)
            self.assertFalse((Path(temporary) / "state" / "manifest.json").exists())


class DeterminismAndCalibrationTests(unittest.TestCase):
    def test_two_full_runs_are_field_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = create_workspace(Path(temporary), workers=1)
            first = run_ok(paths["config"], Path(temporary) / "state-a")
            second = run_ok(paths["config"], Path(temporary) / "state-b")
            self.assertEqual(core_digest(first), core_digest(second))
            for stage in ("data", "model", "calibration", "metrics"):
                self.assertEqual(
                    first["snapshot"]["stage_versions"][stage]["cache_key"],
                    second["snapshot"]["stage_versions"][stage]["cache_key"],
                )
            self.assertTrue(all(model["loss_trajectory"] for model in first["models"]))

    def test_probabilities_and_metrics_are_fixed_and_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = create_workspace(Path(temporary), workers=4)
            report = run_ok(paths["config"], Path(temporary) / "state")
            for calibration in report["calibrations"]:
                for record in calibration["probabilities"]:
                    self.assertGreaterEqual(record["calibrated_probability_fixed"], 0)
                    self.assertLessEqual(record["calibrated_probability_fixed"], 1_000_000)
                    self.assertEqual(len(record["calibrated_probability"].split(".")[1]), 6)
            for metric in report["scheme_metrics"]:
                self.assertGreaterEqual(metric["ece_fixed"], 0)
                self.assertLessEqual(metric["ece_fixed"], 1_000_000)
                self.assertGreaterEqual(metric["mce_fixed"], 0)
                self.assertLessEqual(metric["mce_fixed"], 1_000_000)
            isotonic = [item for item in report["calibrations"] if item["scheme"] == "isotonic"]
            self.assertTrue(isotonic)
            for calibration in isotonic:
                values = [item["value_fixed"] for item in calibration["bin_details"]]
                self.assertEqual(values, sorted(values))

    def test_imputation_statistics_use_training_fold_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = create_workspace(root, workers=1)
            run_ok(paths["config"], root / "state")
            store = SnapshotStore(root / "state")
            manifest = store.load_manifest()
            data = store.load_artifact("data", manifest["stages"]["data"]["cache_key"])
            train_rows = {row["sample_id"]: row for row in data["train"]["rows"]}
            for name in data["preprocessing"]["numeric_medians"]:
                values = sorted(
                    float(row["values"][name])
                    for row in train_rows.values()
                    if row["values"][name] is not None
                )
                midpoint = len(values) // 2
                expected = (
                    values[midpoint]
                    if len(values) % 2
                    else (values[midpoint - 1] + values[midpoint]) / 2.0
                )
                self.assertEqual(data["preprocessing"]["numeric_medians"][name], expected)
            for name in data["preprocessing"]["categorical_modes"]:
                counts = {}
                for row in train_rows.values():
                    value = row["values"][name]
                    if value is not None:
                        counts[value] = counts.get(value, 0) + 1
                expected = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
                self.assertEqual(data["preprocessing"]["categorical_modes"][name], expected)

    def test_boundary_rules_are_explicit(self) -> None:
        records = [
            {"sample_id": 1, "calibrated_probability_fixed": 500},
            {"sample_id": 2, "calibrated_probability_fixed": 499},
            {"sample_id": 3, "calibrated_probability_fixed": 500},
        ]
        labels = [1, 0, 0]
        metric = _calibration_metric(records, labels, 4, 1000, 1000)
        self.assertEqual(metric["boundary_sample_ids"], [1, 3])
        self.assertEqual(metric["bin_details"][2]["count"], 2)
        self.assertEqual(metric["bin_details"][1]["count"], 1)
        threshold = _threshold_metric(records, labels, 500, 1000, 1000, 3)
        self.assertEqual(threshold["boundary_sample_ids"], [1, 3])
        self.assertEqual(threshold["confusion"], {"tp": 1, "fp": 1, "tn": 1, "fn": 0})
        self.assertEqual(threshold["accuracy"], "0.667")

    def test_disabled_schemes_are_absent_not_empty_placeholders(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            def mutate(config):
                config["calibrators"]["raw"]["enabled"] = False
                config["calibrators"]["histogram"]["enabled"] = False

            paths = create_workspace(Path(temporary), workers=2, mutate_config=mutate)
            report = run_ok(paths["config"], Path(temporary) / "state")
            schemes = {item["scheme"] for item in report["calibrations"]}
            self.assertEqual(schemes, {"platt", "isotonic"})
            store = SnapshotStore(Path(temporary) / "state")
            manifest = store.load_manifest()
            key = manifest["stages"]["calibration"]["cache_key"]
            artifact = store.load_artifact("calibration", key)
            self.assertEqual(artifact["enabled_schemes"], ["platt", "isotonic"])
            self.assertTrue(all(item["scheme"] not in ("raw", "histogram") for item in artifact["results"]))

    def test_model_and_calibration_fit_do_not_read_validation_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = create_workspace(Path(temporary), workers=2)
            state = Path(temporary) / "state"
            report = run_ok(paths["config"], state)
            store = SnapshotStore(state)
            manifest = store.load_manifest()
            data_artifact = store.load_artifact("data", manifest["stages"]["data"]["cache_key"])
            model_artifact = store.load_artifact("model", manifest["stages"]["model"]["cache_key"])
            changed = copy.deepcopy(data_artifact)
            changed["validation"]["labels"] = [1 - label for label in changed["validation"]["labels"]]
            config = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
            fixed = fixed_point(config)
            model_again = build_model_stage(
                changed,
                config["models"],
                fixed,
                model_artifact["model_key"],
                1,
            )
            self.assertEqual(model_again["models"], model_artifact["models"])
            calibration_again = build_calibration_stage(
                changed,
                model_artifact,
                config["calibrators"],
                fixed,
                manifest["stages"]["calibration"]["cache_key"],
                1,
            )
            original_calibration = store.load_artifact(
                "calibration", manifest["stages"]["calibration"]["cache_key"]
            )
            self.assertEqual(calibration_again["results"], original_calibration["results"])

    def test_inversion_detection_contains_sample_pair_and_bin(self) -> None:
        from probcal.calibration import _find_inversions

        records = [
            {"sample_id": 1, "raw_probability_fixed": 100, "calibrated_probability_fixed": 500},
            {"sample_id": 2, "raw_probability_fixed": 200, "calibrated_probability_fixed": 400},
        ]
        inversions = _find_inversions(records, "histogram", 10, 1000)
        self.assertEqual(len(inversions), 1)
        self.assertEqual(inversions[0]["left_sample_id"], 1)
        self.assertEqual(inversions[0]["right_sample_id"], 2)
        self.assertEqual(inversions[0]["right_bin"], 2)


class IncrementalEquivalenceTests(unittest.TestCase):
    def test_edit_sequence_matches_full_rebuild_after_every_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = create_workspace(root, workers=4)
            incremental_state = root / "incremental-state"
            initial = run_ok(paths["config"], incremental_state)
            self._assert_full_rebuild_matches(paths, initial, root, 0)

            edits = [
                [
                    {"op": "switch_calibration", "scheme": "raw", "enabled": False},
                    {"op": "switch_calibration", "scheme": "platt", "enabled": True},
                    {"op": "set_bins", "scheme": "histogram", "bins": 12},
                    {"op": "set_bins", "target": "evaluation", "bins": 10},
                ],
                [
                    {"op": "set_regularization", "scheme": "platt", "strength": "0.002"},
                    {"op": "set_regularization", "scheme": "isotonic", "strength": "1.0"},
                ],
                [{"op": "set_threshold_step", "step": "0.025"}],
                [
                    {
                        "op": "replace_sample",
                        "sample_id": 17,
                        "values": {
                            "age": 39.5,
                            "income": 70123.25,
                            "income_redundant": 72232.95,
                            "region": "east",
                            "segment": "premium",
                            "risk_score": 1.75,
                        },
                        "target": 1,
                    }
                ],
            ]
            expected_reuse = [
                ["data", "model"],
                ["data", "model"],
                ["data", "model", "calibration"],
                [],
            ]
            expected_recompute = [
                ["calibration", "metrics"],
                ["calibration", "metrics"],
                ["metrics"],
                ["data", "model", "calibration", "metrics"],
            ]
            for index, operations in enumerate(edits, start=1):
                edit_path = write_edit(root / f"edit-{index}.yaml", operations)
                report = edit_service(paths["config"], incremental_state, edit_path)
                self.assertEqual(report["status"], "ok", report)
                self.assertEqual(report["execution"]["reused_stages"], expected_reuse[index - 1])
                self.assertEqual(
                    report["execution"]["recomputed_stages"], expected_recompute[index - 1]
                )
                if index == 3:
                    self.assertEqual(
                        [(item["stage"], item["input_key"]) for item in report["execution"]["invalidated_inputs"]],
                        [("metrics", "threshold_step")],
                    )
                self._assert_full_rebuild_matches(paths, report, root, index)

    def test_disabled_scheme_parameter_change_does_not_invalidate_stages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def disable(config):
                config["calibrators"]["histogram"]["enabled"] = False

            paths = create_workspace(root, workers=1, mutate_config=disable)
            state = root / "state"
            initial = run_ok(paths["config"], state)
            edit = write_edit(
                root / "disabled-edit.yaml",
                [{"op": "set_bins", "scheme": "histogram", "bins": 12}],
            )
            report = edit_service(paths["config"], state, edit)
            self.assertEqual(report["status"], "ok")
            self.assertEqual(
                report["execution"]["reused_stages"],
                ["data", "model", "calibration", "metrics"],
            )
            self.assertEqual(report["execution"]["recomputed_stages"], [])
            self.assertEqual(report["execution"]["invalidated_stages"], [])
            self.assertEqual(report["snapshot"]["version"], initial["snapshot"]["version"] + 1)

    def _assert_full_rebuild_matches(
        self, paths: dict[str, Path], incremental: dict, root: Path, index: int
    ) -> None:
        full_state = root / f"full-state-{index}"
        full = run_ok(paths["config"], full_state)
        self.assertEqual(core_digest(incremental), core_digest(full))
        self.assertEqual(report_core(incremental), report_core(full))
        for stage in ("data", "model", "calibration", "metrics"):
            self.assertEqual(
                incremental["snapshot"]["stage_versions"][stage]["cache_key"],
                full["snapshot"]["stage_versions"][stage]["cache_key"],
            )


class ConcurrencyTests(unittest.TestCase):
    def test_one_worker_and_four_workers_are_field_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            one_paths = create_workspace(root / "one", workers=1)
            four_paths = create_workspace(root / "four", workers=4)
            one = run_ok(one_paths["config"], root / "state-one")
            four = run_ok(four_paths["config"], root / "state-four")
            self.assertEqual(core_digest(one), core_digest(four))
            for stage in ("data", "model", "calibration", "metrics"):
                self.assertEqual(
                    one["snapshot"]["stage_versions"][stage]["cache_key"],
                    four["snapshot"]["stage_versions"][stage]["cache_key"],
                )
            self.assertEqual(one["models"], four["models"])
            self.assertEqual(one["calibrations"], four["calibrations"])
            one_store = SnapshotStore(root / "state-one")
            four_store = SnapshotStore(root / "state-four")
            one_manifest = one_store.load_manifest()
            four_manifest = four_store.load_manifest()
            self.assertEqual(
                one_store.load_artifact("model", one_manifest["stages"]["model"]["cache_key"]),
                four_store.load_artifact("model", four_manifest["stages"]["model"]["cache_key"]),
            )
            self.assertEqual(
                one_store.load_artifact(
                    "calibration", one_manifest["stages"]["calibration"]["cache_key"]
                ),
                four_store.load_artifact(
                    "calibration", four_manifest["stages"]["calibration"]["cache_key"]
                ),
            )


class SnapshotAndFailureTests(unittest.TestCase):
    def test_manifest_versions_and_atomic_artifact_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = create_workspace(root, workers=2)
            report = run_ok(paths["config"], root / "state")
            store = SnapshotStore(root / "state")
            manifest = store.load_manifest()
            self.assertEqual(manifest["global_version"], 1)
            self.assertEqual(manifest["config_hash"], report["snapshot"]["config_hash"])
            self.assertIsNotNone(manifest["data_fingerprint"])
            self.assertEqual(set(manifest["stages"]), {"data", "model", "calibration", "metrics"})
            for stage, record in manifest["stages"].items():
                artifact = store.load_artifact(stage, record["cache_key"])
                self.assertIsNotNone(artifact)
                self.assertTrue(record["version"])
            self.assertTrue((root / "state" / "snapshots" / "v000000001" / "manifest.json").exists())
            self.assertTrue((root / "state" / "snapshots" / "v000000001" / "report.json").exists())
            self.assertFalse(any((root / "state" / "cache").rglob(".tmp-*")))
            self.assertTrue(report["snapshot_layout"]["same_device"])

    def test_external_config_edit_and_version_mismatch_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = create_workspace(root, workers=2)
            state = root / "state"
            run_ok(paths["config"], state)
            original = paths["config"].read_bytes()
            config = yaml.safe_load(original)
            config["evaluation"]["threshold_step"] = "0.025"
            paths["config"].write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            mismatch = run_service(paths["config"], state)
            self.assertEqual(mismatch["status"], "failed")
            self.assertEqual(mismatch["business_code"], "E_CONFIG_MISMATCH")
            paths["config"].write_bytes(original)
            wrong_version = run_service(paths["config"], state, expect_version=99)
            self.assertEqual(wrong_version["business_code"], "E_CONFIG_MISMATCH")
            initial_wrong_version = run_service(
                paths["config"], root / "fresh-state", expect_version=1
            )
            self.assertEqual(initial_wrong_version["business_code"], "E_CONFIG_MISMATCH")

    def test_pre_requested_stop_commits_interrupted_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            import threading

            root = Path(temporary)
            paths = create_workspace(root, workers=1)
            stop = threading.Event()
            stop.set()
            report = run_service(paths["config"], root / "state", stop_event=stop)
            self.assertEqual(report["status"], "interrupted")
            self.assertEqual(report["execution"]["pending_stages"], ["data", "model", "calibration", "metrics"])
            self.assertTrue((root / "state" / "manifest.json").exists())
            self.assertTrue((root / "state" / "snapshots" / "v000000001" / "report.json").exists())


if __name__ == "__main__":
    unittest.main()
