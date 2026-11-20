from __future__ import annotations

import copy
import json
import math
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import yaml

from tree_workbench.binning import compute_binning
from tree_workbench.config import validate_experiment
from tree_workbench.data import prepare_data
from tree_workbench.engine import Engine
from tree_workbench.errors import SnapshotVersionChanged
from tree_workbench.snapshot import SnapshotManager
from tree_workbench.verify import run_verification


ROOT = Path(__file__).resolve().parents[1]


def load_samples():
    with (ROOT / "samples" / "experiment.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with (ROOT / "samples" / "data_spec.json").open("r", encoding="utf-8") as handle:
        spec = json.load(handle)
    with (ROOT / "samples" / "edit_plan.json").open("r", encoding="utf-8") as handle:
        edits = json.load(handle)
    return config, spec, edits


class ValidationTests(unittest.TestCase):
    def test_independent_business_codes(self):
        config, spec, _ = load_samples()
        cases = {
            "CFG_UNKNOWN_COLUMN": lambda c, s: c["data"]["features"].append("ghost_column"),
            "CFG_TYPE_MISMATCH": lambda c, s: s["columns"][0].__setitem__("type", "unsupported"),
            "CFG_TARGET_MISSING": lambda c, s: c["data"].__setitem__("target", "missing_target"),
            "CFG_TREE_COUNT_INVALID": lambda c, s: c["model"].__setitem__("tree_count", 0),
            "CFG_MAX_DEPTH_INVALID": lambda c, s: c["model"].__setitem__("max_depth", 0),
            "CFG_LEARNING_RATE_INVALID": lambda c, s: c["model"].__setitem__("learning_rate", -0.1),
            "CFG_BIN_COUNT_INVALID": lambda c, s: c["model"].__setitem__("bin_count", 0),
            "CFG_CONCURRENCY_INVALID": lambda c, s: c["runtime"].__setitem__("thread_pool_size", 0),
        }
        found_codes: set[str] = set()
        for expected_code, mutate in cases.items():
            config_copy = copy.deepcopy(config)
            spec_copy = copy.deepcopy(spec)
            mutate(config_copy, spec_copy)
            _, failures = validate_experiment(config_copy, spec_copy, [])
            codes = {failure.code for failure in failures}
            self.assertIn(expected_code, codes)
            found_codes.update(codes)
        self.assertTrue(set(cases).issubset(found_codes))

    def test_edit_validation_codes(self):
        config, spec, _ = load_samples()
        _, failures = validate_experiment(
            config,
            spec,
            [
                {
                    "corrections": [
                        {"row_id": 0, "column": "not_a_column", "value": 1},
                        {"row_id": 0, "column": "age", "value": "not-a-number"},
                    ]
                }
            ],
        )
        codes = {failure.code for failure in failures}
        self.assertIn("CFG_UNKNOWN_COLUMN", codes)
        self.assertIn("CFG_TYPE_MISMATCH", codes)

    def test_invalid_run_enters_failure_list_and_report(self):
        config, spec, _ = load_samples()
        broken = copy.deepcopy(config)
        broken["runtime"]["thread_pool_size"] = 0
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = Engine(root / "state", tmp_dir=root / "tmp", report_dir=root / "reports").run(
                broken,
                spec,
                [],
                report_name="invalid.json",
            )
            self.assertEqual(result.status, "failed")
            self.assertEqual(result.exit_code, 2)
            self.assertIn("CFG_CONCURRENCY_INVALID", {failure["business_code"] for failure in result.failures})
            report = json.loads(Path(result.report_path).read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            self.assertIn("failures", report)


class BinningTests(unittest.TestCase):
    def test_binning_is_deterministic_and_bin_count_changes_content(self):
        config, spec, _ = load_samples()
        validated, failures = validate_experiment(config, spec, [])
        self.assertEqual(failures, [])
        assert validated is not None
        data = prepare_data(validated, spec, [])
        first = compute_binning(validated, data)
        second = compute_binning(validated, data)
        self.assertEqual(first, second)

        changed_config = copy.copy(validated)
        changed_config = changed_config.__class__(**{**validated.__dict__, "bin_count": 7})
        changed = compute_binning(changed_config, data)
        age_first = first["columns"]["age"]
        age_changed = changed["columns"]["age"]
        self.assertNotEqual(age_first["boundaries"], age_changed["boundaries"])
        self.assertNotEqual(age_first["bin_indices"], age_changed["bin_indices"])
        self.assertNotEqual(age_first["content_fingerprint"], age_changed["content_fingerprint"])
        self.assertEqual(age_first["boundaries"], sorted(age_first["boundaries"]))


class EngineTests(unittest.TestCase):
    def test_cache_invalidation_matrix(self):
        config, spec, _ = load_samples()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine = Engine(root / "state", tmp_dir=root / "tmp", report_dir=root / "reports")
            initial = engine.run(config, spec, [], report_name="initial.json")
            self.assertEqual(initial.status, "completed")
            self.assertEqual(initial.cache_audit["reused"], [])

            grown_edits = [{"tree_count_delta": 3}]
            grown = engine.run(config, spec, grown_edits, report_name="grown.json")
            self.assertEqual(grown.status, "completed")
            self.assertEqual(grown.cache_audit["reused"][:9], ["binning"] + [f"tree:{index}" for index in range(8)])
            for index in range(8):
                self.assertIn(f"tree:{index}", grown.cache_audit["reused"])
            for index in range(8, 11):
                self.assertIn(f"tree:{index}", grown.cache_audit["recomputed"])
            self.assertIn("ensemble", grown.cache_audit["recomputed"])
            self.assertIn("contributions", grown.cache_audit["recomputed"])

            learning_edits = grown_edits + [{"learning_rate": 0.1}]
            learning_rate_changed = engine.run(config, spec, learning_edits, report_name="lr.json")
            self.assertEqual(learning_rate_changed.status, "completed")
            for index in range(11):
                self.assertIn(f"tree:{index}", learning_rate_changed.cache_audit["reused"])
            self.assertIn("binning", learning_rate_changed.cache_audit["reused"])
            self.assertIn("ensemble", learning_rate_changed.cache_audit["recomputed"])
            self.assertIn("contributions", learning_rate_changed.cache_audit["recomputed"])

            depth_edits = learning_edits + [{"max_depth": 5}]
            depth_changed = engine.run(config, spec, depth_edits, report_name="depth.json")
            self.assertEqual(depth_changed.status, "completed")
            self.assertIn("binning", depth_changed.cache_audit["reused"])
            for index in range(11):
                self.assertIn(f"tree:{index}", depth_changed.cache_audit["recomputed"])

            bin_edits = depth_edits + [{"bin_count": 7}]
            bin_changed = engine.run(config, spec, bin_edits, report_name="bins.json")
            self.assertEqual(bin_changed.status, "completed")
            self.assertIn("binning", bin_changed.cache_audit["recomputed"])
            for index in range(11):
                self.assertIn(f"tree:{index}", bin_changed.cache_audit["recomputed"])
            self.assertNotEqual(
                initial.artifacts["binning"]["columns"]["income"]["bin_indices"],
                bin_changed.artifacts["binning"]["columns"]["income"]["bin_indices"],
            )

            correction_edits = bin_edits + [{"corrections": [{"row_id": 0, "column": "income", "value": 123456.0}]}]
            corrected = engine.run(config, spec, correction_edits, report_name="corrected.json")
            self.assertEqual(corrected.status, "completed")
            self.assertIn("binning", corrected.cache_audit["recomputed"])
            for index in range(11):
                self.assertIn(f"tree:{index}", corrected.cache_audit["recomputed"])

    def test_failure_isolation_report_and_contributions(self):
        config, spec, _ = load_samples()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = Engine(root / "state", tmp_dir=root / "tmp", report_dir=root / "reports").run(
                config,
                spec,
                [],
                faults={1: "split"},
                report_name="fault.json",
            )
            self.assertEqual(result.status, "completed")
            self.assertEqual(len(result.failures), 1)
            failure = result.failures[0]
            self.assertEqual(failure["tree_index"], 1)
            self.assertEqual(failure["stage"], "split")
            self.assertEqual(failure["business_code"], "TREE_SPLIT_FAILURE")
            self.assertEqual(failure["exception_type"], "InjectedTreeFailure")
            ensemble = result.artifacts["ensemble"]
            contributions = result.artifacts["contributions"]
            self.assertNotIn(1, ensemble["applied_trees"])
            self.assertEqual(ensemble["failed_trees"], [1])
            self.assertNotIn("1", contributions["samples"][0]["by_tree"])
            for sample in contributions["samples"][:10]:
                attributed = math.fsum(sample["by_feature"].values())
                expected = sample["raw_score"] - sample["base_score"]
                self.assertAlmostEqual(attributed, expected, places=7)
            report = json.loads(Path(result.report_path).read_text(encoding="utf-8"))
            for key in (
                "timings_ms",
                "cache_audit",
                "reused",
                "recomputed",
                "failures",
                "recovery",
                "recovery_point",
                "metrics",
                "integration",
            ):
                self.assertIn(key, report)
            self.assertIsNotNone(report["integration"]["metrics"])

    def test_random_forest_mode_is_deterministic_across_concurrency(self):
        config, spec, _ = load_samples()
        config = copy.deepcopy(config)
        config["model"]["kind"] = "random_forest"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = Engine(root / "state-1", tmp_dir=root / "tmp", report_dir=root / "reports").run(
                config,
                spec,
                [],
                concurrency_override=1,
            )
            second = Engine(root / "state-4", tmp_dir=root / "tmp", report_dir=root / "reports").run(
                config,
                spec,
                [],
                concurrency_override=4,
            )
            self.assertEqual(first.status, "completed")
            self.assertEqual(second.status, "completed")
            self.assertEqual(first.artifacts["trees"], second.artifacts["trees"])
            self.assertEqual(first.artifacts["ensemble"], second.artifacts["ensemble"])
            self.assertEqual(first.artifacts["contributions"], second.artifacts["contributions"])

    def test_equivalence_and_concurrency_verification(self):
        config, spec, edits = load_samples()
        with tempfile.TemporaryDirectory() as temporary:
            verification, exit_code = run_verification(
                config_data=config,
                spec_data=spec,
                batches=edits,
                work_dir=temporary,
                parallel_workers=3,
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(verification["all_equal"])
            self.assertEqual(verification["scenario_count"], 5)
            self.assertTrue(verification["concurrency"]["equal"])
            for scenario in verification["scenarios"]:
                self.assertTrue(scenario["equal"], scenario.get("differences"))

    def test_graceful_interrupt_and_resume(self):
        config, spec, _ = load_samples()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine = Engine(root / "state", tmp_dir=root / "tmp", report_dir=root / "reports")
            stop_event = threading.Event()
            timer = threading.Timer(0.06, stop_event.set)
            timer.start()
            interrupted = engine.run(
                config,
                spec,
                [],
                report_name="interrupted.json",
                stop_event=stop_event,
                tree_delay_ms=100,
            )
            timer.cancel()
            self.assertEqual(interrupted.status, "interrupted")
            self.assertEqual(interrupted.exit_code, 0)
            self.assertGreater(len(interrupted.artifacts["trees"]), 0)
            self.assertLess(len(interrupted.artifacts["trees"]), 8)
            self.assertIsNotNone(interrupted.report["recovery"]["committed"])
            self.assertTrue(interrupted.cache_audit["deferred"])

            resumed = engine.run(config, spec, [], report_name="resumed.json")
            self.assertEqual(resumed.status, "completed")
            completed = len(interrupted.artifacts["trees"])
            for index in range(completed):
                self.assertIn(f"tree:{index}", resumed.cache_audit["reused"])
            self.assertIsNotNone(resumed.report["recovery"]["loaded"])
            self.assertEqual(resumed.report["recovery"]["loaded"]["completed_trees"], list(range(completed)))

    def test_incomplete_snapshot_is_ignored(self):
        config, spec, _ = load_samples()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = root / "state"
            result = Engine(state, tmp_dir=root / "tmp", report_dir=root / "reports").run(config, spec, [])
            self.assertEqual(result.status, "completed")
            manager = SnapshotManager(state)
            version = manager.current_version()
            self.assertGreater(version, 0)
            incomplete = manager.snapshots_root / ".tmp-snapshot-999999-deadbeef"
            incomplete.mkdir()
            (incomplete / "manifest.json").write_text('{"complete": false}', encoding="utf-8")
            loaded = manager.load_current()
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.version, version)

    def test_snapshot_commit_rejects_stale_expected_version(self):
        config, spec, _ = load_samples()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = root / "state"
            Engine(state, tmp_dir=root / "tmp", report_dir=root / "reports").run(config, spec, [])
            manager = SnapshotManager(state)
            snapshot = manager.load_current()
            self.assertIsNotNone(snapshot)
            assert snapshot is not None
            with self.assertRaises(SnapshotVersionChanged):
                manager.commit(
                    expected_version=snapshot.version - 1,
                    config_hash=snapshot.manifest["config_hash"],
                    data_fingerprint=snapshot.manifest["data_fingerprint"],
                    binning=snapshot.stages["binning"],
                    trees=snapshot.trees,
                    ensemble=snapshot.stages.get("ensemble"),
                    contributions=snapshot.stages.get("contributions"),
                )


class CliSignalTests(unittest.TestCase):
    def test_sigterm_returns_zero_then_resumes(self):
        config, spec, _ = load_samples()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command = [
                sys.executable,
                "-m",
                "tree_workbench",
                "run",
                "--config",
                str(ROOT / "samples" / "experiment.yaml"),
                "--data-spec",
                str(ROOT / "samples" / "data_spec.json"),
                "--state-dir",
                str(root / "state"),
                "--tmp-dir",
                str(root / "tmp"),
                "--report-dir",
                str(root / "reports"),
                "--tree-delay-ms",
                "1000",
            ]
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            time.sleep(0.8)
            process.send_signal(signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=15)
            self.assertEqual(process.returncode, 0, stderr + stdout)
            report = json.loads((root / "reports" / "report-00-initial.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "interrupted")
            self.assertIsNotNone(report["recovery"]["committed"])

            resume_command = [item for item in command]
            delay_index = resume_command.index("--tree-delay-ms")
            del resume_command[delay_index : delay_index + 2]
            resumed = subprocess.run(resume_command, cwd=ROOT, text=True, capture_output=True, timeout=30)
            self.assertEqual(resumed.returncode, 0, resumed.stderr + resumed.stdout)
            latest = json.loads((root / "reports" / "latest.json").read_text(encoding="utf-8"))
            self.assertEqual(latest["status"], "completed")
            self.assertIsNotNone(latest["recovery"]["loaded"])
            self.assertTrue(latest["recovery"]["loaded"]["completed_trees"])

    def test_sigkill_recovers_only_complete_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command = [
                sys.executable,
                "-m",
                "tree_workbench",
                "run",
                "--config",
                str(ROOT / "samples" / "experiment.yaml"),
                "--data-spec",
                str(ROOT / "samples" / "data_spec.json"),
                "--state-dir",
                str(root / "state"),
                "--tmp-dir",
                str(root / "tmp"),
                "--report-dir",
                str(root / "reports"),
                "--tree-delay-ms",
                "1200",
            ]
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            deadline = time.time() + 10
            manager = SnapshotManager(root / "state")
            while time.time() < deadline and manager.current_version() == 0:
                time.sleep(0.05)
            self.assertGreater(manager.current_version(), 0)
            process.kill()
            process.communicate(timeout=5)
            self.assertLess(process.returncode, 0)

            loaded = manager.load_current()
            self.assertIsNotNone(loaded)
            self.assertTrue(loaded.manifest["complete"])
            self.assertTrue(loaded.manifest["completed_trees"])

            resume_command = [item for item in command]
            delay_index = resume_command.index("--tree-delay-ms")
            del resume_command[delay_index : delay_index + 2]
            resumed = subprocess.run(resume_command, cwd=ROOT, text=True, capture_output=True, timeout=30)
            self.assertEqual(resumed.returncode, 0, resumed.stderr + resumed.stdout)
            latest = json.loads((root / "reports" / "latest.json").read_text(encoding="utf-8"))
            self.assertEqual(latest["status"], "completed")
            self.assertTrue(latest["recovery"]["loaded"]["completed_trees"])


if __name__ == "__main__":
    unittest.main()
