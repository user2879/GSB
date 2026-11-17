from __future__ import annotations

import copy
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from adw.commands import apply_command
from adw.config import ALGORITHM_NAMES, load_config_bundle, validate_config
from adw.data import generate_dataset
from adw.engine import WorkbenchEngine
from adw.errors import ConfigValidationError
from adw.injection import load_injection_bundle
from adw.pipeline import DetectionPipeline, canonical_result_payload, run_full
from adw.preprocess import preprocess_dataset
from adw.storage import ContentCache, SnapshotManager


ROOT = Path(__file__).resolve().parents[1]


def temporary_config(bundle, directory: Path) -> Path:
    text = (ROOT / "config.yaml").read_text(encoding="utf-8")
    description = str(bundle.description_path)
    text = text.replace("description: data_description.yaml", f"description: {json.dumps(description)}")
    storage = {
        "snapshot_dir": str(directory / "snapshots"),
        "cache_dir": str(directory / "cache"),
        "report_path": str(directory / "report.json"),
        "injection_report_path": str(directory / "injection_report.json"),
    }
    originals = {
        "snapshot_dir": "  snapshot_dir: .workbench/snapshots",
        "cache_dir": "  cache_dir: .workbench/cache",
        "report_path": "  report_path: .workbench/report.json",
        "injection_report_path": "  injection_report_path: .workbench/injection_report.json",
    }
    for key, value in storage.items():
        text = text.replace(originals[key], f"  {key}: {json.dumps(value)}")
    config_path = directory / "config.yaml"
    config_path.write_text(text, encoding="utf-8")
    return config_path


class WorkbenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_config_bundle(ROOT / "config.yaml")
        assert not cls.bundle.failures, cls.bundle.failures
        cls.dataset = generate_dataset(cls.bundle.description)
        cls.injection = load_injection_bundle(ROOT / "injection.yaml")
        assert not cls.injection.failures, cls.injection.failures

    def run_pipeline(self, config, cache=None, injection_items=None, force_serial=False):
        pipeline = DetectionPipeline(
            self.dataset,
            config,
            cache=cache,
            injection_items=injection_items,
            force_serial=force_serial,
        )
        return pipeline.run()

    def test_dataset_and_preprocessing_are_deterministic(self):
        other = generate_dataset(self.bundle.description)
        self.assertEqual(self.dataset.fingerprint, other.fingerprint)
        self.assertEqual(self.dataset.rows, other.rows)
        self.assertEqual(self.dataset.sample_ids, [f"S{i:06d}" for i in range(1, 1401)])
        expected_anomalies = 98
        self.assertEqual(sum(self.dataset.target), expected_anomalies)
        self.assertTrue(any(any(value is None for key, value in row.items() if key != "is_anomaly") for row in self.dataset.rows))

        first = preprocess_dataset(self.dataset, self.bundle.config)
        second = preprocess_dataset(self.dataset, self.bundle.config)
        self.assertTrue(np.array_equal(first.matrix, second.matrix))
        self.assertTrue(np.all(np.isfinite(first.matrix)))
        self.assertEqual(first.columns, self.bundle.config["data"]["feature_columns"])
        self.assertEqual(sorted(first.category_maps), ["channel", "region"])
        for mapping in first.category_maps.values():
            self.assertEqual(list(mapping.values()), [float(i) for i in range(len(mapping))])

    def test_three_algorithms_are_deterministic_and_stably_sorted(self):
        first = run_full(self.dataset, self.bundle.config)
        second = run_full(self.dataset, self.bundle.config)
        self.assertEqual(first.status, "ok")
        self.assertEqual(second.status, "ok")
        self.assertEqual(canonical_result_payload(first.payload), canonical_result_payload(second.payload))
        for name in ALGORITHM_NAMES:
            ranking = first.payload["algorithm_rankings"][name]
            self.assertEqual(len(ranking), len(self.dataset.sample_ids))
            keys = [(-item["score"], item["sample_id"]) for item in ranking]
            self.assertEqual(keys, sorted(keys))
            self.assertTrue(all(np.isfinite(item["score"]) for item in ranking))
        samples = first.payload["samples"]
        self.assertEqual([item["sample_id"] for item in samples], self.dataset.sample_ids)
        for item in samples[:20]:
            self.assertEqual([part["algorithm"] for part in item["contributions"]], list(ALGORITHM_NAMES))
            weighted_sum = sum(part["weighted_contribution"] for part in item["contributions"])
            self.assertAlmostEqual(item["anomaly_score"], weighted_sum / len(ALGORITHM_NAMES), places=11)
        self.assertEqual(first.payload["summary"]["anomaly_count"], len(first.payload["anomalies"]))
        self.assertGreaterEqual(first.payload["summary"]["anomaly_count"], 1)

    def test_incremental_cache_reuse_and_exact_invalidation_scope(self):
        with tempfile.TemporaryDirectory(prefix="adw-cache-test-") as temp:
            cache = ContentCache(Path(temp), enabled=True)
            baseline = self.run_pipeline(self.bundle.config, cache=cache)
            self.assertEqual(baseline.status, "ok")
            self.assertEqual(baseline.diagnostics["recalculated_algorithms"], list(ALGORITHM_NAMES))

            repeated = self.run_pipeline(self.bundle.config, cache=cache)
            self.assertEqual(repeated.diagnostics["cache"]["hits"], 5)
            self.assertEqual(repeated.diagnostics["cache"]["misses"], 0)
            self.assertEqual(repeated.diagnostics["recalculated_algorithms"], [])
            self.assertEqual(canonical_result_payload(baseline.payload), canonical_result_payload(repeated.payload))

            hist_changed = copy.deepcopy(self.bundle.config)
            hist_changed["algorithms"]["histogram_density"]["params"]["bins"] = 20
            one_algorithm = self.run_pipeline(hist_changed, cache=cache)
            self.assertEqual(one_algorithm.diagnostics["recalculated_algorithms"], ["histogram_density"])
            self.assertEqual(
                one_algorithm.diagnostics["reused_algorithms"],
                ["random_partition_depth", "robust_mahalanobis"],
            )
            self.assertTrue(one_algorithm.diagnostics["preprocess_reused"])
            self.assertEqual(one_algorithm.diagnostics["cache"]["misses"], 2)
            self.assertEqual(
                canonical_result_payload(one_algorithm.payload),
                canonical_result_payload(run_full(self.dataset, hist_changed).payload),
            )

            threshold_changed = copy.deepcopy(hist_changed)
            threshold_changed["threshold"] = 0.61
            threshold_only = self.run_pipeline(threshold_changed, cache=cache)
            self.assertEqual(threshold_only.diagnostics["recalculated_algorithms"], [])
            self.assertEqual(len(threshold_only.diagnostics["reused_algorithms"]), 3)
            self.assertTrue(threshold_only.diagnostics["decision_reused"] is False)
            self.assertEqual(threshold_only.diagnostics["cache"]["misses"], 1)
            self.assertEqual(
                canonical_result_payload(threshold_only.payload),
                canonical_result_payload(run_full(self.dataset, threshold_changed).payload),
            )

            preprocessing_changed = copy.deepcopy(threshold_changed)
            preprocessing_changed["preprocessing"]["mode"] = "standardize"
            all_again = self.run_pipeline(preprocessing_changed, cache=cache)
            self.assertFalse(all_again.diagnostics["preprocess_reused"])
            self.assertEqual(all_again.diagnostics["recalculated_algorithms"], list(ALGORITHM_NAMES))
            self.assertEqual(
                canonical_result_payload(all_again.payload),
                canonical_result_payload(run_full(self.dataset, preprocessing_changed).payload),
            )

    def test_concurrency_does_not_change_results(self):
        config_one = copy.deepcopy(self.bundle.config)
        config_one["interaction"]["concurrency"] = 1
        config_four = copy.deepcopy(self.bundle.config)
        config_four["interaction"]["concurrency"] = 4
        serial = run_full(self.dataset, config_one)
        parallel_one = self.run_pipeline(config_one, cache=ContentCache(None, enabled=False))
        parallel_four = self.run_pipeline(config_four, cache=ContentCache(None, enabled=False))
        self.assertEqual(canonical_result_payload(serial.payload), canonical_result_payload(parallel_one.payload))
        self.assertEqual(canonical_result_payload(serial.payload), canonical_result_payload(parallel_four.payload))

    def test_interaction_sequence_matches_full_serial_run(self):
        config = copy.deepcopy(self.bundle.config)
        commands = [
            "algo disable random_partition_depth",
            "param set histogram_density bins 20",
            "preprocess set mode standardize",
            "threshold set 0.61",
            "algo enable random_partition_depth",
            "param set random_partition_depth trees 40",
        ]
        with tempfile.TemporaryDirectory(prefix="adw-seq-cache-") as temp:
            cache = ContentCache(Path(temp), enabled=True)
            outcome = self.run_pipeline(config, cache=cache)
            self.assertEqual(outcome.status, "ok")
            for line in commands:
                result = apply_command(config, self.bundle.description, line)
                config = result.config
                outcome = self.run_pipeline(config, cache=cache)
                self.assertEqual(outcome.status, "ok")
            full = run_full(self.dataset, config)
            self.assertEqual(canonical_result_payload(outcome.payload), canonical_result_payload(full.payload))
            with self.assertRaises(ConfigValidationError) as context:
                apply_command(config, self.bundle.description, "param set histogram_density bins 9")
            self.assertEqual(context.exception.failures[0]["code"], "E_CONFIG_INVALID")

    def test_budget_pending_then_settles_exactly(self):
        with tempfile.TemporaryDirectory(prefix="adw-budget-test-") as temp:
            config = copy.deepcopy(self.bundle.config)
            config["storage"] = {
                "snapshot_dir": str(Path(temp) / "snapshots"),
                "cache_dir": str(Path(temp) / "cache"),
                "report_path": str(Path(temp) / "report.json"),
                "injection_report_path": str(Path(temp) / "injection_report.json"),
            }
            engine = WorkbenchEngine(self.dataset, config, root=ROOT)
            engine.start()
            changed = copy.deepcopy(config)
            changed["threshold"] = 0.61
            response = engine.submit_config(
                changed,
                affected_algorithms=[],
                source_command="threshold set 0.61",
            )
            self.assertEqual(response["status"], "pending")
            self.assertEqual(response["code"], "E_BUDGET_PENDING")
            self.assertTrue(response["interaction"]["pending"])
            self.assertFalse(response["interaction"]["over_budget"])
            self.assertLessEqual(
                response["interaction"]["elapsed_ms"],
                response["interaction"]["budget_ms"] + 1.0,
            )
            settled = engine.settle_response()
            self.assertEqual(settled["status"], "ok")
            self.assertEqual(
                canonical_result_payload(settled["result"]),
                canonical_result_payload(run_full(self.dataset, changed).payload),
            )
            history = engine.report_payload()["command_responses"]
            self.assertTrue(any(item["code"] == "E_BUDGET_PENDING" for item in history))
            engine.write_report()
            shutil_report = json.loads((Path(temp) / "report.json").read_text(encoding="utf-8"))
            self.assertIn("command_responses", shutil_report)
            engine.request_shutdown()
            engine.join_worker()

    def test_snapshot_atomic_recovery_and_temp_file_ignoring(self):
        with tempfile.TemporaryDirectory(prefix="adw-snapshot-test-") as temp:
            manager = SnapshotManager(Path(temp))
            cache = ContentCache(Path(temp) / "cache", enabled=True)
            pipeline = DetectionPipeline(self.dataset, self.bundle.config, cache=cache, snapshots=manager)
            outcome = pipeline.run()
            self.assertEqual(outcome.status, "ok")
            scan = manager.scan(pipeline.data_fingerprint)
            self.assertEqual([item["stage"] for item in scan["valid_snapshots"]], ["preprocess", "score", "decision"])
            for item in scan["valid_snapshots"]:
                self.assertEqual(item["completed_samples"], 1400)
                self.assertTrue(item["artifact_version"])
                self.assertTrue(item["config_hash"])
                self.assertEqual(item["data_fingerprint"], pipeline.data_fingerprint)
            temp_file = Path(temp) / ".preprocess.json.tmp.crashed"
            temp_file.write_text('{"payload":', encoding="utf-8")
            scan = manager.scan(pipeline.data_fingerprint)
            self.assertIn(temp_file.name, scan["ignored_temp_files"])
            self.assertEqual(len(scan["valid_snapshots"]), 3)

            config = copy.deepcopy(self.bundle.config)
            config["storage"] = {
                "snapshot_dir": str(Path(temp)),
                "cache_dir": str(Path(temp) / "cache"),
                "report_path": str(Path(temp) / "report.json"),
                "injection_report_path": str(Path(temp) / "injection_report.json"),
            }
            restarted = WorkbenchEngine(self.dataset, config, root=ROOT)
            self.assertTrue(restarted.recovery["restored_from_complete_snapshots"])
            self.assertGreaterEqual(restarted.recovery["warmed_cache_entries"], 5)
            self.assertIn(temp_file.name, restarted.recovery["ignored_temp_files"])

    def test_injection_is_isolated_and_mapped_to_business_codes(self):
        incremental = self.run_pipeline(
            self.bundle.config,
            cache=ContentCache(None, enabled=False),
            injection_items=self.injection.data["items"],
        )
        full = run_full(self.dataset, self.bundle.config, injection_items=self.injection.data["items"])
        self.assertEqual(canonical_result_payload(incremental.payload), canonical_result_payload(full.payload))
        injection_report = incremental.diagnostics["injection"]
        self.assertEqual(injection_report["hits"], 4)
        self.assertEqual(injection_report["total"], 4)
        codes = {item["code"] for item in incremental.payload["failures"]}
        self.assertEqual(
            codes,
            {"E_DATA_LIMIT", "E_NON_FINITE_FEATURE", "E_PARAM_INVALID", "E_THRESHOLD_BOUNDARY"},
        )
        failed_ids = {item["sample_id"] for item in incremental.payload["failures"] if item["sample_id"]}
        self.assertEqual(failed_ids, {"S000007", "S000011", "S000019"})
        self.assertEqual(len(incremental.payload["samples"]), 1400)
        self.assertEqual(sum(1 for item in incremental.payload["samples"] if item["failed"]), 3)
        self.assertTrue(any(not item["failed"] and item["anomaly_score"] is not None for item in incremental.payload["samples"]))
        distributions = incremental.diagnostics["distributions"]
        self.assertEqual(set(distributions["raw"]), set(ALGORITHM_NAMES))
        self.assertGreater(distributions["anomaly_score"]["count"], 1000)

    def test_config_validation_failures_are_not_silent(self):
        cases = []

        def add(name, mutate, description=None):
            config = copy.deepcopy(self.bundle.config)
            mutate(config)
            cases.append((name, config, description or self.bundle.description))

        add("unknown-column", lambda c: c["data"]["feature_columns"].__setitem__(0, "not_a_column"))
        add("type-mismatch", lambda c: c["preprocessing"]["numeric_columns"].__setitem__(0, "channel"))
        add("concurrency", lambda c: c["interaction"].__setitem__("concurrency", 0))
        add("budget", lambda c: c["interaction"].__setitem__("latency_budget_ms", 0))
        missing_target = copy.deepcopy(self.bundle.description)
        missing_target["dataset"].pop("target")
        add("target-missing", lambda c: None, missing_target)
        bad_ratio = copy.deepcopy(self.bundle.description)
        bad_ratio["dataset"]["target"]["anomaly_ratio"] = 1.01
        add("anomaly-ratio", lambda c: None, bad_ratio)
        for name, config, description in cases:
            _, failures = validate_config(config, description)
            self.assertTrue(failures, name)
            self.assertTrue(all(item["code"] == "E_CONFIG_INVALID" for item in failures), name)

    def test_cli_script_and_full_comparison(self):
        with tempfile.TemporaryDirectory(prefix="adw-cli-test-") as temp:
            directory = Path(temp)
            config_path = temporary_config(self.bundle, directory)
            script_path = directory / "script.txt"
            script_path.write_text(
                "param set histogram_density bins 12\n"
                "settle\n"
                "threshold set 0.69\n"
                "settle\n"
                "compare full\n"
                "report\n"
                "quit\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [sys.executable, "-m", "adw", "--config", str(config_path), "--script", str(script_path)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=40,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            events = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
            self.assertEqual(events[0]["event"], "ready")
            comparisons = [item for item in events if item.get("status") == "ok" and "equal" in item]
            self.assertEqual(len(comparisons), 1)
            self.assertTrue(comparisons[0]["equal"], comparisons[0].get("differences"))
            self.assertEqual(events[-1]["event"], "exit")
            self.assertEqual(events[-1]["code"], 0)
            self.assertTrue((directory / "report.json").exists())

    def test_graceful_signal_completes_stage_snapshot_and_exits_zero(self):
        with tempfile.TemporaryDirectory(prefix="adw-signal-test-") as temp:
            directory = Path(temp)
            config_path = temporary_config(self.bundle, directory)
            script_path = directory / "script.txt"
            script_path.write_text("settle\n", encoding="utf-8")
            process = subprocess.Popen(
                [sys.executable, "-m", "adw", "--config", str(config_path), "--script", str(script_path)],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                ready_line = process.stdout.readline()
                self.assertTrue(ready_line, "CLI did not emit a ready event")
                self.assertEqual(json.loads(ready_line)["event"], "ready")
                time.sleep(0.04)
                process.send_signal(signal.SIGTERM)
                stdout, stderr = process.communicate(timeout=15)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()
            self.assertEqual(process.returncode, 0, stderr)
            events = [json.loads(line) for line in (ready_line + stdout).splitlines() if line.strip()]
            self.assertEqual(events[-1]["event"], "shutdown")
            self.assertEqual(events[-1]["code"], 0)
            self.assertTrue(events[-1]["graceful"])
            snapshots = list((directory / "snapshots").glob("*.json"))
            self.assertTrue(snapshots)

    def test_signal_at_interactive_prompt_exits_zero(self):
        with tempfile.TemporaryDirectory(prefix="adw-prompt-signal-test-") as temp:
            directory = Path(temp)
            config_path = temporary_config(self.bundle, directory)
            process = subprocess.Popen(
                [sys.executable, "-m", "adw", "--config", str(config_path)],
                cwd=ROOT,
                text=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                ready_line = process.stdout.readline()
                self.assertTrue(ready_line, "CLI did not emit a ready event")
                time.sleep(0.04)
                process.send_signal(signal.SIGTERM)
                stdout, stderr = process.communicate(timeout=15)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()
            self.assertEqual(process.returncode, 0, stderr)
            events = [json.loads(line) for line in (ready_line + stdout).splitlines() if line.strip()]
            self.assertEqual(events[-1]["event"], "shutdown")
            self.assertEqual(events[-1]["code"], 0)
            self.assertTrue(events[-1]["graceful"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
