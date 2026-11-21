from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from clustering_workbench.cache import ResidentBlockCache, deterministic_recompute_probe
from clustering_workbench.config import load_config
from clustering_workbench.data import generate_dataset
from clustering_workbench.injections import InjectionManager
from clustering_workbench.methods import CandidateSpec, execute_method
from clustering_workbench.metrics import rank_candidates
from clustering_workbench.pipeline import WorkbenchPipeline
from clustering_workbench.preprocess import preprocess_dataset


ROOT = Path(__file__).resolve().parents[1]
DEMO_CONFIG = ROOT / "configs" / "demo.yaml"
DEMO_DATASET = ROOT / "samples" / "demo_dataset.yaml"


def run_pipeline(output_dir: str | Path, overrides: dict | None = None) -> dict:
    return WorkbenchPipeline(DEMO_CONFIG, overrides=overrides, output_dir=output_dir).run()


def write_config(path: Path, config: dict) -> None:
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


class WorkbenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.class_dir = Path(tempfile.mkdtemp(prefix="cw-tests-"))

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.class_dir, ignore_errors=True)

    def _context(self, injections: bool = False):
        config, config_path, failures = load_config(
            DEMO_CONFIG,
            overrides={"injections": []} if not injections else None,
        )
        manager = InjectionManager(config.get("injections", []))
        dataset, data_failures = generate_dataset(config, config_path, manager)
        self.assertIsNotNone(dataset)
        self.assertEqual([], [failure for failure in failures + data_failures if failure.severity == "error"])
        return config, dataset, manager

    def test_dataset_generation_is_deterministic_and_contains_requested_shapes(self) -> None:
        config, dataset, _ = self._context()
        _, dataset2, _ = self._context()
        self.assertEqual(dataset.content_fingerprint, dataset2.content_fingerprint)
        self.assertEqual(dataset.columns, dataset2.columns)
        self.assertEqual(240, dataset.rows_generated)
        self.assertIn("segment", dataset.target_values)
        self.assertEqual(3, len(set(dataset.target_values["segment"])))
        self.assertTrue(any(value is None for value in dataset.columns["order_value"]))
        self.assertGreaterEqual(len(set(dataset.columns["acquisition_channel"])), 30)
        for left, right in zip(dataset.columns["order_value"], dataset.columns["value_copy"]):
            self.assertEqual(left, right)

    def test_preprocess_is_elementwise_stable_and_method_switch_changes_artifact(self) -> None:
        config, dataset, _ = self._context()
        failures_a: list = []
        failures_b: list = []
        standard_a = preprocess_dataset(dataset, config, failures_a)
        standard_b = preprocess_dataset(dataset, config, failures_b)
        self.assertIsNotNone(standard_a)
        self.assertIsNotNone(standard_b)
        np.testing.assert_array_equal(standard_a.X, standard_b.X)
        self.assertEqual(standard_a.content_fingerprint, standard_b.content_fingerprint)
        self.assertEqual(standard_a.artifact_fingerprint, standard_b.artifact_fingerprint)
        self.assertIn("value_copy", [item["column"] for item in standard_a.dropped_columns])
        self.assertTrue(np.all(np.isfinite(standard_a.X)))

        quantile_config = copy.deepcopy(config)
        quantile_config["preprocessing"]["method"] = "quantile"
        quantile_failures: list = []
        quantile = preprocess_dataset(dataset, quantile_config, quantile_failures)
        self.assertIsNotNone(quantile)
        self.assertNotEqual(standard_a.content_fingerprint, quantile.content_fingerprint)
        self.assertNotEqual(standard_a.parameter_fingerprint, quantile.parameter_fingerprint)
        self.assertFalse(np.array_equal(standard_a.X, quantile.X))
        self.assertIn("quantile_reference_values", quantile.statistics["order_value"])

    def test_three_methods_are_deterministic(self) -> None:
        config, dataset, _ = self._context()
        failures: list = []
        result = preprocess_dataset(dataset, config, failures)
        self.assertIsNotNone(result)
        specifications = {
            "KMeans": CandidateSpec("KMeans", {"n_clusters": 3, "max_iter": 30, "tol": 1e-6}, "k", ("KMeans", "{}", "000"), 0, "k"),
            "DBSCAN": CandidateSpec("DBSCAN", {"eps": 0.5, "min_samples": 4}, "d", ("DBSCAN", "{}", "000"), 1, "d"),
            "GaussianMixture": CandidateSpec(
                "GaussianMixture",
                {"n_components": 3, "covariance_type": "full", "max_iter": 25, "tol": 1e-5, "reg_covar": 1e-5},
                "g",
                ("GaussianMixture", "{}", "000"),
                2,
                "g",
            ),
        }
        for name, spec in specifications.items():
            grant = 1 if name == "DBSCAN" else 30
            first = execute_method(spec, result.X, grant)
            second = execute_method(spec, result.X, grant)
            np.testing.assert_array_equal(first.labels, second.labels)
            self.assertEqual(first.trajectory, second.trajectory)
            self.assertEqual(len(result.X), len(first.labels))
            self.assertTrue(np.all(np.isfinite(first.labels.astype(float))))
        self.assertEqual(30, specifications["KMeans"].params["max_iter"])

    def test_empty_cluster_and_all_noise_are_retained_for_ranking(self) -> None:
        constant = np.ones((24, 3), dtype=np.float64)
        kmeans_spec = CandidateSpec("KMeans", {"n_clusters": 3, "max_iter": 4, "tol": 1e-6}, "k", ("KMeans", "{}", "000"), 0, "k")
        empty = execute_method(kmeans_spec, constant, 4)
        self.assertIn("EMPTY_CLUSTER", empty.diagnostic_flags)

        spread = np.arange(24, dtype=np.float64).reshape(12, 2)
        dbscan_spec = CandidateSpec("DBSCAN", {"eps": 0.01, "min_samples": 2}, "d", ("DBSCAN", "{}", "000"), 1, "d")
        noise = execute_method(dbscan_spec, spread, 1)
        self.assertIn("ALL_NOISE", noise.diagnostic_flags)

        rows = [
            {
                "method": "KMeans",
                "canonical_params": "{}",
                "occurrence": "000",
                "status": "converged",
                "metrics": {"silhouette": 0.1, "calinski_harabasz": 1.0, "density_log_likelihood": -2.0},
                "diagnostic_flags": ["EMPTY_CLUSTER"],
            },
            {
                "method": "DBSCAN",
                "canonical_params": "{}",
                "occurrence": "000",
                "status": "completed",
                "metrics": {"silhouette": None, "calinski_harabasz": None, "density_log_likelihood": None},
                "diagnostic_flags": ["ALL_NOISE", "METRIC_UNDEFINED"],
            },
        ]
        ranked = rank_candidates(rows, "silhouette")
        self.assertEqual(2, len(ranked))
        self.assertEqual("KMeans", ranked[0]["method"])
        self.assertEqual("DBSCAN", ranked[1]["method"])

    def test_gaussian_singular_covariance_is_regularized(self) -> None:
        base = np.linspace(-1.0, 1.0, 90, dtype=np.float64)
        matrix = np.column_stack([base, base, base * 0.5 + 0.01])
        spec = CandidateSpec(
            "GaussianMixture",
            {
                "n_components": 4,
                "covariance_type": "full",
                "max_iter": 20,
                "tol": 1e-6,
                "reg_covar": 1e-5,
            },
            "g",
            ("GaussianMixture", "{}", "000"),
            0,
            "g",
        )
        execution = execute_method(spec, matrix, 20)
        self.assertEqual("GaussianMixture", execution.diagnostics["algorithm"])
        self.assertEqual(90, len(execution.labels))
        self.assertTrue(np.all(np.isfinite(execution.labels.astype(float))))
        self.assertEqual(1e-5, execution.diagnostics["reg_covar"])

    def test_serial_and_parallel_are_fieldwise_equivalent_with_budget_pressure(self) -> None:
        serial_dir = Path(tempfile.mkdtemp(prefix="cw-serial-", dir=self.class_dir))
        parallel_dir = Path(tempfile.mkdtemp(prefix="cw-parallel-", dir=self.class_dir))
        overrides_serial = {
            "scheduler": {
                "max_workers": 1,
                "global_iteration_budget": 20,
                "per_candidate_iteration_quota": 4,
            }
        }
        overrides_parallel = copy.deepcopy(overrides_serial)
        overrides_parallel["scheduler"]["max_workers"] = 4
        serial = run_pipeline(serial_dir, overrides_serial)
        parallel = run_pipeline(parallel_dir, overrides_parallel)
        self.assertEqual(serial["determinism_digest"], parallel["determinism_digest"])
        self.assertEqual(serial["ranking"], parallel["ranking"])
        self.assertEqual(serial["scheduler"]["budget_log"], parallel["scheduler"]["budget_log"])
        self.assertEqual(serial["failures"], parallel["failures"])
        self.assertEqual(
            [row.get("labels") for row in serial["candidates"]],
            [row.get("labels") for row in parallel["candidates"]],
        )
        self.assertNotEqual(
            serial["scheduler"]["completion_order"],
            parallel["scheduler"]["completion_order"],
        )
        self.assertTrue(any(row["status"] == "budget_aborted" for row in serial["candidates"]))

    def test_cache_budget_does_not_change_computational_result(self) -> None:
        small_cache_dir = Path(tempfile.mkdtemp(prefix="cw-cache1-", dir=self.class_dir))
        large_cache_dir = Path(tempfile.mkdtemp(prefix="cw-cache8-", dir=self.class_dir))
        small = run_pipeline(
            small_cache_dir,
            {"scheduler": {"cache_resident_blocks": 1, "max_workers": 2}},
        )
        large = run_pipeline(
            large_cache_dir,
            {"scheduler": {"cache_resident_blocks": 8, "max_workers": 2}},
        )
        self.assertEqual(small["determinism_digest"], large["determinism_digest"])
        self.assertGreater(small["scheduler"]["cache"]["evictions"], large["scheduler"]["cache"]["evictions"])
        self.assertGreater(small["scheduler"]["cache"]["recomputes"], 0)

    def test_lru_eviction_recomputes_identically(self) -> None:
        cache = ResidentBlockCache(2)
        result = deterministic_recompute_probe(cache, rounds=4)
        self.assertEqual(0, result["values"]["probe:00"])
        self.assertEqual(1, result["values"]["probe:01"])
        self.assertEqual(4, result["values"]["probe:02"])
        self.assertEqual(9, result["values"]["probe:03"])
        self.assertGreater(result["summary"]["evictions"], 0)
        self.assertGreater(result["summary"]["recomputes"], 0)

    def test_incremental_reuse_only_rebuilds_affected_candidate(self) -> None:
        incremental_dir = Path(tempfile.mkdtemp(prefix="cw-inc-", dir=self.class_dir))
        fresh_dir = Path(tempfile.mkdtemp(prefix="cw-full-", dir=self.class_dir))
        base = run_pipeline(incremental_dir)
        base_report = json.loads((incremental_dir / "latest_report.json").read_text(encoding="utf-8"))
        changed_candidates = [
            {"n_components": 2, "covariance_type": "diag", "max_iter": 30, "tol": 1e-5, "reg_covar": 1e-6},
            {"n_components": 3, "covariance_type": "full", "max_iter": 35, "tol": 1e-5, "reg_covar": 1e-6},
            {"n_components": 3, "covariance_type": "diag", "max_iter": 35, "tol": 1e-5, "reg_covar": 1e-6},
        ]
        overrides = {"methods": {"GaussianMixture": {"enabled": True, "candidates": changed_candidates}}}
        incremental = run_pipeline(incremental_dir, overrides)
        fresh = run_pipeline(fresh_dir, overrides)
        self.assertEqual(base["determinism_digest"], base_report["determinism_digest"])
        self.assertEqual(incremental["determinism_digest"], fresh["determinism_digest"])
        self.assertEqual(incremental["ranking"], fresh["ranking"])
        self.assertEqual(incremental["reuse"]["counters"]["preprocess_compute_calls"], 0)
        self.assertEqual(incremental["reuse"]["counters"]["cluster_compute_calls"], 1)
        self.assertEqual(incremental["reuse"]["counters"]["evaluation_compute_calls"], 1)
        self.assertIn("SNAPSHOT_VERSION_INVALID", {failure["code"] for failure in incremental["failures"]})
        unchanged_base = {
            row["stable_key"]: row.get("labels")
            for row in base["candidates"]
            if "GaussianMixture" not in row["stable_key"]
        }
        unchanged_incremental = {
            row["stable_key"]: row.get("labels")
            for row in incremental["candidates"]
            if row["stable_key"] in unchanged_base
        }
        self.assertEqual(unchanged_base, unchanged_incremental)

    def test_partial_sample_replacement_invalidates_preprocess_and_all_candidates(self) -> None:
        output_dir = Path(tempfile.mkdtemp(prefix="cw-sample-", dir=self.class_dir))
        base = run_pipeline(output_dir)
        changed = run_pipeline(
            output_dir,
            {"data": {"sample_overrides": {"S00001": {"order_value": 987.5}}}},
        )
        self.assertNotEqual(
            base["preprocess"]["content_fingerprint"],
            changed["preprocess"]["content_fingerprint"],
        )
        self.assertGreaterEqual(changed["reuse"]["counters"]["preprocess_compute_calls"], 1)
        self.assertGreaterEqual(changed["reuse"]["counters"]["cluster_compute_calls"], 7)
        self.assertGreaterEqual(changed["reuse"]["counters"]["evaluation_compute_calls"], 7)

    def test_snapshot_resume_and_version_mismatch_are_fail_closed(self) -> None:
        output_dir = Path(tempfile.mkdtemp(prefix="cw-snapshot-", dir=self.class_dir))
        first = run_pipeline(output_dir)
        second = run_pipeline(output_dir)
        self.assertTrue(second["snapshots"]["valid"])
        self.assertTrue(second["snapshots"]["resume"])
        self.assertEqual(second["preprocess"]["source"], "snapshot")
        self.assertEqual(second["reuse"]["counters"]["cluster_compute_calls"], 0)
        self.assertEqual(second["reuse"]["counters"]["evaluation_compute_calls"], 0)
        self.assertEqual(first["determinism_digest"], second["determinism_digest"])

        invalid = run_pipeline(output_dir, {"snapshot": {"version": 2}})
        codes = {failure["code"] for failure in invalid["failures"]}
        self.assertIn("SNAPSHOT_VERSION_INVALID", codes)
        self.assertFalse(invalid["snapshots"]["valid"])
        self.assertTrue(any(path.name.startswith("invalid-") for path in (output_dir / "snapshots" / "invalidated").iterdir()))

    def test_all_injection_types_hit_and_map_to_business_codes(self) -> None:
        output_dir = Path(tempfile.mkdtemp(prefix="cw-inject-", dir=self.class_dir))
        report = run_pipeline(output_dir)
        self.assertEqual(360, report["dataset"]["rows_generated"])
        self.assertEqual(4, report["injections"]["hit"])
        self.assertEqual(0, report["injections"]["missed"])
        report_codes = {failure["code"] for failure in report["failures"]}
        for code in (
            "INJ_NON_FINITE_FEATURE",
            "INJ_HYPERPARAMETER_OUT_OF_RANGE",
            "INJ_THRESHOLD_BOUNDARY_EQUAL",
            "INJ_DATA_SIZE_OVER_LIMIT",
            "EVAL_ALL_NOISE",
        ):
            self.assertIn(code, report_codes)
        invalid = [row for row in report["candidates"] if row["status"] == "invalid"]
        self.assertEqual(1, len(invalid))
        self.assertEqual("INJ_HYPERPARAMETER_OUT_OF_RANGE", invalid[0]["invalid_code"])
        boundary_rows = [row for row in report["candidates"] if row.get("boundary_check")]
        self.assertEqual(1, len(boundary_rows))
        boundary = boundary_rows[0]["boundary_check"]
        self.assertTrue(boundary["boundary_hit"])
        self.assertEqual(boundary["score"], boundary["threshold"])
        self.assertEqual(0.0, boundary["delta"])
        all_noise = [row for row in report["candidates"] if "ALL_NOISE" in row.get("diagnostic_flags", [])]
        self.assertEqual(1, len(all_noise))
        self.assertIsNotNone(all_noise[0]["rank"])

    def test_validation_business_codes_do_not_abort_other_columns(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="cw-validation-", dir=self.class_dir))
        config = yaml.safe_load(DEMO_CONFIG.read_text(encoding="utf-8"))
        config["paths"]["dataset"] = str(DEMO_DATASET)
        config["paths"]["outputs"] = str(temp_dir / "outputs")
        config["injections"] = []
        config["data"]["include_columns"] = list(config["data"]["include_columns"]) + ["ghost_column"]
        config["data"]["expected_types"]["tenure_months"] = "categorical"
        config["methods"] = {
            "KMeans": {"enabled": True, "candidates": [{"n_clusters": 0}, {"n_clusters": 3}]},
            "DBSCAN": {"enabled": False, "candidates": []},
            "GaussianMixture": {"enabled": False, "candidates": []},
        }
        config["scheduler"]["max_workers"] = 0
        config["scheduler"]["global_iteration_budget"] = 0
        config_path = temp_dir / "validation.yaml"
        write_config(config_path, config)
        report = WorkbenchPipeline(config_path).run()
        codes = {failure["code"] for failure in report["failures"]}
        for expected in (
            "CFG_UNKNOWN_COLUMN",
            "CFG_TYPE_MISMATCH",
            "CFG_CLUSTER_ZERO",
            "CFG_CONCURRENCY_NONPOSITIVE",
            "CFG_GLOBAL_BUDGET_NONPOSITIVE",
        ):
            self.assertIn(expected, codes)
        self.assertGreater(report["preprocess"]["shape"][1], 0)

    def test_missing_target_structure_emits_stable_code(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="cw-target-", dir=self.class_dir))
        dataset = yaml.safe_load(DEMO_DATASET.read_text(encoding="utf-8"))
        dataset["dataset"].pop("structure", None)
        dataset_path = temp_dir / "dataset_no_structure.yaml"
        dataset_path.write_text(yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8")
        config = yaml.safe_load(DEMO_CONFIG.read_text(encoding="utf-8"))
        config["paths"]["dataset"] = str(dataset_path)
        config["paths"]["outputs"] = str(temp_dir / "outputs")
        config["data"]["target_columns"] = []
        config["injections"] = []
        config["snapshot"]["enabled"] = False
        config_path = temp_dir / "missing_target.yaml"
        write_config(config_path, config)
        report = WorkbenchPipeline(config_path).run()
        self.assertIn(
            "CFG_TARGET_STRUCTURE_MISSING",
            {failure["code"] for failure in report["failures"]},
        )


if __name__ == "__main__":
    unittest.main()
