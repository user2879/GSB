from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcw.config import Candidate, candidate_from_params, load_experiment_config
from tcw.data import generate_dataset, prepare_dataset
from tcw.errors import (
    E_CONFIG_INVALID,
    E_CONFIG_MISMATCH,
    E_DBSCAN_RADIUS_INVALID,
    E_HYPERPARAM_OUT_OF_RANGE,
    E_INPUT_NON_FINITE,
    E_CATEGORY_UNMAPPED,
    BusinessError,
)
from tcw.evaluation import build_ranking, evaluate_fit
from tcw.injection import InjectionContext, load_injection_config, run_injections
from tcw.models import fit_dbscan, fit_kmeans
from tcw.pipeline import execute_experiment
from tcw.preprocessing import fit_transform
from tcw.snapshot import SnapshotStore
from tcw.util import canonical_json, sha256_value


CONFIG_PATH = ROOT / "config" / "experiment.yaml"
DATA_PATH = ROOT / "config" / "data_description.yaml"
INJECTION_PATH = ROOT / "config" / "injections.yaml"


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _reduced_raw_config(concurrency: int = 4) -> dict:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["data"]["description"] = str(DATA_PATH)
    raw["execution"]["concurrency"] = concurrency
    raw["execution"]["snapshot_commit_interval"] = 1
    raw["clustering"]["candidates"] = {
        "kmeans": raw["clustering"]["candidates"]["kmeans"][:1],
        "dbscan": raw["clustering"]["candidates"]["dbscan"][1:2],
        "gaussian_mixture": raw["clustering"]["candidates"]["gaussian_mixture"][:1],
    }
    return raw


def _comparable_report(report: dict) -> dict:
    keys = [
        "candidate_ranking",
        "metrics_table",
        "diagnostics",
        "injection_hits",
        "failure_list",
        "business_codes",
        "injection_hash",
        "dataset",
        "preprocessing",
    ]
    return {key: report[key] for key in keys}


class WorkbenchTests(unittest.TestCase):
    def test_invalid_config_entries_fail_but_valid_remain(self) -> None:
        raw = _reduced_raw_config(concurrency=0)
        raw["experiment"]["seed"] = "not-a-seed"
        raw.pop("target", None)
        raw["preprocessing"]["include_columns"].append("ghost_column")
        raw["preprocessing"]["expected_types"]["region"] = "numeric"
        raw["clustering"]["candidates"]["kmeans"].append(
            {"n_clusters": 0, "max_iter": 20, "tol": 0.0001}
        )
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            config_path = temp_path / "invalid_but_recoverable.yaml"
            _write_yaml(config_path, raw)
            config = load_experiment_config(config_path, DATA_PATH)
            messages = " | ".join(failure.message for failure in config.failures)
            self.assertIn("target structure", messages)
            self.assertIn("unknown column: ghost_column", messages)
            self.assertIn("type mismatch for region", messages)
            self.assertIn("n_clusters", messages)
            self.assertIn("concurrency", messages)
            self.assertIn("experiment.seed", messages)
            self.assertTrue(config.failures)
            self.assertTrue(all(failure.code == E_CONFIG_INVALID for failure in config.failures))
            self.assertEqual(config.seed, 20261118)
            self.assertEqual(config.execution["concurrency"], 1)
            self.assertNotIn("ghost_column", config.preprocessing["include_columns"])
            self.assertNotIn("region", config.preprocessing["include_columns"])
            self.assertTrue(all(candidate.params.get("n_clusters", 1) > 0 for candidate in config.candidates))

            injection = load_injection_config(None)
            outcome = execute_experiment(
                config,
                injection,
                temp_path / "snapshots",
                temp_path / "report.json",
            )
            self.assertEqual(outcome.exit_code, 1)
            self.assertGreater(len(outcome.report["candidate_ranking"]), 0)
            self.assertEqual(outcome.report["status"], "completed_with_failures")
            self.assertTrue(all(entry["code"] == E_CONFIG_INVALID for entry in outcome.report["failure_list"]))
            self.assertTrue((temp_path / "report.json").is_file())

    def test_preprocessing_is_elementwise_repeatable(self) -> None:
        config = load_experiment_config(CONFIG_PATH)
        dataset = generate_dataset(config.description, seed=config.seed)
        prepared = prepare_dataset(dataset, config.preprocessing["fit_fraction"])
        first = fit_transform(prepared, config.preprocessing)
        second = fit_transform(prepared, config.preprocessing)
        self.assertEqual(first.matrix, second.matrix)
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(first.output_hash, second.output_hash)
        self.assertTrue(np.all(np.isfinite(np.asarray(first.matrix, dtype=float))))
        self.assertEqual(prepared.numeric_fill_values.keys(), {
            "age", "income", "spend", "tenure", "age_copy", "spend_copy"
        })
        for feature in first.stats["features"].values():
            self.assertEqual(feature["accumulation_order"], "training_row_ascending")

        quantile_config = copy.deepcopy(config.preprocessing)
        quantile_config["method"] = "quantile"
        quantile_first = fit_transform(prepared, quantile_config)
        quantile_second = fit_transform(prepared, quantile_config)
        self.assertEqual(quantile_first.matrix, quantile_second.matrix)
        self.assertNotEqual(first.fingerprint, quantile_first.fingerprint)
        self.assertEqual(quantile_first.stats["kind"], "quantile")
        for feature in quantile_first.stats["features"].values():
            self.assertEqual(feature["accumulation_order"], "sorted_value_ascending")

    def test_injection_hits_are_unique_and_repeatable(self) -> None:
        config = load_experiment_config(CONFIG_PATH)
        injection = load_injection_config(INJECTION_PATH)
        dataset = generate_dataset(config.description, seed=config.seed)
        prepared = prepare_dataset(dataset, config.preprocessing["fit_fraction"])
        preprocessed = fit_transform(prepared, config.preprocessing)
        context = InjectionContext(
            dataset=dataset,
            prepared=prepared,
            matrix=np.asarray(preprocessed.matrix, dtype=float),
            candidates=config.candidates,
        )
        hits_first, failures_first = run_injections(injection, context)
        hits_second, failures_second = run_injections(injection, context)
        self.assertEqual(canonical_json(hits_first), canonical_json(hits_second))
        self.assertEqual(
            [failure.to_dict() for failure in sorted(failures_first, key=lambda item: item.sort_key())],
            [failure.to_dict() for failure in sorted(failures_second, key=lambda item: item.sort_key())],
        )
        ids = [hit["injection_id"] for hit in hits_first]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 5)
        self.assertTrue(all(hit["hit"] for hit in hits_first))
        actual_codes = {hit["injection_id"]: hit["actual_code"] for hit in hits_first}
        self.assertEqual(actual_codes["INJ-NONFINITE-001"], E_INPUT_NON_FINITE)
        self.assertEqual(actual_codes["INJ-MISSING-CATEGORY-002"], E_CATEGORY_UNMAPPED)
        self.assertEqual(actual_codes["INJ-KMEANS-ZERO-CLUSTERS-003"], E_HYPERPARAM_OUT_OF_RANGE)
        self.assertEqual(actual_codes["INJ-DBSCAN-NEGATIVE-RADIUS-004"], E_DBSCAN_RADIUS_INVALID)
        self.assertEqual(actual_codes["INJ-GMM-COVARIANCE-005"], E_HYPERPARAM_OUT_OF_RANGE)

    def test_concurrency_does_not_change_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            serial_raw = _reduced_raw_config(concurrency=1)
            parallel_raw = _reduced_raw_config(concurrency=4)
            serial_config_path = temp_path / "serial.yaml"
            parallel_config_path = temp_path / "parallel.yaml"
            _write_yaml(serial_config_path, serial_raw)
            _write_yaml(parallel_config_path, parallel_raw)
            serial = execute_experiment(
                load_experiment_config(serial_config_path, DATA_PATH),
                load_injection_config(None),
                temp_path / "serial-snapshots",
                temp_path / "serial-report.json",
            )
            parallel = execute_experiment(
                load_experiment_config(parallel_config_path, DATA_PATH),
                load_injection_config(None),
                temp_path / "parallel-snapshots",
                temp_path / "parallel-report.json",
            )
            self.assertEqual(_comparable_report(serial.report), _comparable_report(parallel.report))
            self.assertNotEqual(serial.report["config_hash"], parallel.report["config_hash"])
            self.assertEqual(serial.exit_code, 0)
            self.assertEqual(parallel.exit_code, 0)
            self.assertEqual(len(serial.report["candidate_ranking"]), 3)
            methods = {entry["method"] for entry in serial.report["candidate_ranking"]}
            self.assertEqual(methods, {"kmeans", "dbscan", "gaussian_mixture"})
            gmm = next(entry for entry in serial.report["candidate_ranking"] if entry["method"] == "gaussian_mixture")
            responsibilities = gmm["assignments"]["responsibilities"]
            self.assertIsNotNone(responsibilities)
            self.assertTrue(all(abs(sum(row) - 1.0) < 1e-9 for row in responsibilities))
            report_files = list((temp_path / "parallel-report.json").parent.glob(".*.tmp"))
            self.assertEqual(report_files, [])

    def test_snapshot_resume_and_mismatch_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            config_path = temp_path / "config.yaml"
            _write_yaml(config_path, _reduced_raw_config(concurrency=2))
            config = load_experiment_config(config_path, DATA_PATH)
            injection = load_injection_config(None)
            snapshot_dir = temp_path / "snapshots"
            first = execute_experiment(
                config,
                injection,
                snapshot_dir,
                temp_path / "first.json",
            )
            resumed = execute_experiment(
                config,
                injection,
                snapshot_dir,
                temp_path / "resumed.json",
                resume=True,
            )
            self.assertEqual(_comparable_report(first.report), _comparable_report(resumed.report))
            self.assertEqual(
                resumed.report["execution"]["restored_stages"],
                ["preprocess", "clustering", "evaluation"],
            )
            manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(set(manifest["stages"]), {"preprocess", "clustering", "evaluation"})
            for stage, entry in manifest["stages"].items():
                artifact = snapshot_dir / entry["path"] / "artifact.json"
                self.assertTrue(artifact.is_file(), stage)
                self.assertEqual(sha256_value(json.loads(artifact.read_text(encoding="utf-8"))), entry["payload_hash"])

            wrong_version = SnapshotStore(
                snapshot_dir,
                manifest["snapshot_version"] + 1,
                manifest["config_hash"],
                manifest["injection_hash"],
                manifest["data_hash"],
            )
            with self.assertRaises(BusinessError) as version_error:
                wrong_version.verify_manifest()
            self.assertEqual(version_error.exception.code, E_CONFIG_MISMATCH)

            wrong_config = SnapshotStore(
                snapshot_dir,
                manifest["snapshot_version"],
                "wrong-config-hash",
                manifest["injection_hash"],
                manifest["data_hash"],
            )
            with self.assertRaises(BusinessError) as config_error:
                wrong_config.verify_manifest()
            self.assertEqual(config_error.exception.code, E_CONFIG_MISMATCH)

            wrong_injection = SnapshotStore(
                snapshot_dir,
                manifest["snapshot_version"],
                manifest["config_hash"],
                "wrong-injection-hash",
                manifest["data_hash"],
            )
            with self.assertRaises(BusinessError) as injection_error:
                wrong_injection.verify_manifest()
            self.assertEqual(injection_error.exception.code, E_CONFIG_MISMATCH)

    def test_empty_cluster_and_all_noise_candidates_are_retained(self) -> None:
        kmeans_matrix = np.asarray([[1.0], [0.0], [1.0], [0.0]], dtype=float)
        kmeans_params = {"n_clusters": 3, "max_iter": 20, "tol": 1e-9}
        kmeans = fit_kmeans(kmeans_matrix, kmeans_params, 0, "kmeans:test")
        self.assertTrue(kmeans.diagnostics["empty_clusters"])

        matrix = np.asarray([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0], [6.0, 6.0]], dtype=float)
        dbscan_params = {"eps": 0.01, "min_samples": 2}
        dbscan = fit_dbscan(matrix, dbscan_params, 0, "dbscan:test")
        self.assertTrue(dbscan.diagnostics["all_noise"])
        self.assertEqual(dbscan.diagnostics["noise_count"], matrix.shape[0])
        kmeans_candidate = candidate_from_params("kmeans", kmeans_params, kmeans_matrix.shape[0])
        dbscan_candidate = candidate_from_params("dbscan", dbscan_params, matrix.shape[0])
        evaluations = [
            evaluate_fit(kmeans_candidate, kmeans, kmeans_matrix),
            evaluate_fit(dbscan_candidate, dbscan, matrix),
        ]
        ranked = build_ranking(
            evaluations,
            {
                "weights": {"silhouette": 0.4, "calinski_harabasz": 0.3, "log_likelihood": 0.3},
                "primary": "composite",
            },
        )
        self.assertEqual(len(ranked), 2)
        self.assertEqual(sorted(evaluation.rank for evaluation in ranked), [1, 2])
        dbscan_eval = next(evaluation for evaluation in ranked if evaluation.candidate.method == "dbscan")
        self.assertIsNone(dbscan_eval.metrics["silhouette"])
        self.assertTrue(dbscan_eval.diagnostics["all_noise"])


if __name__ == "__main__":
    unittest.main()
