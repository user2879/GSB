from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from feature_pipeline.compare import compare_states
from feature_pipeline.engine import PipelineEngine
from feature_pipeline.state import SnapshotStore

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


class RecoveryAndInjectionTests(unittest.TestCase):
    def test_graceful_stage_stop_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = PipelineEngine(root / "recovering")
            stopped, code = engine.run(
                config_path=SAMPLES / "config.json",
                events_path=SAMPLES / "events.jsonl",
                features_path=SAMPLES / "features.json",
                labels_path=SAMPLES / "labels.json",
                reset=True,
                stop_after="events",
            )
            self.assertEqual(code, 0)
            self.assertTrue(stopped["stopped_gracefully"])
            self.assertEqual(stopped["completed_stages"], ["events"])
            resumed, code = engine.run(
                config_path=SAMPLES / "config.json",
                events_path=SAMPLES / "events.jsonl",
                features_path=SAMPLES / "features.json",
                labels_path=SAMPLES / "labels.json",
            )
            self.assertEqual(code, 0)
            self.assertTrue(resumed["recovery_point"]["resumed"])
            self.assertEqual(resumed["recovery_point"]["completed_stages"], ["events"])

            full_engine = PipelineEngine(root / "full")
            full_engine.run(
                config_path=SAMPLES / "config.json",
                events_path=SAMPLES / "events.jsonl",
                features_path=SAMPLES / "features.json",
                labels_path=SAMPLES / "labels.json",
                reset=True,
            )
            recovered_state, _ = SnapshotStore(root / "recovering").recover()
            full_state, _ = SnapshotStore(root / "full").recover()
            self.assertTrue(compare_states(recovered_state, full_state)["equal"])

    def test_incomplete_generation_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "state"
            engine = PipelineEngine(root)
            engine.run(
                config_path=SAMPLES / "config.json",
                events_path=SAMPLES / "events_base_10.jsonl",
                features_path=SAMPLES / "features_base.json",
                labels_path=SAMPLES / "labels_base.json",
                reset=True,
            )
            complete, _ = SnapshotStore(root).recover()
            incomplete = root / "snapshots" / "g999999"
            incomplete.mkdir()
            (incomplete / "manifest.json").write_text("{broken", encoding="utf-8")
            (root / "CURRENT").write_text("g999999\n", encoding="ascii")
            recovered, info = SnapshotStore(root).recover()
            self.assertEqual(recovered.generation, complete.generation)
            self.assertEqual(info["source"], "highest-valid-generation")

    def test_real_sigterm_finishes_current_stage_with_exit_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads((SAMPLES / "config.json").read_text(encoding="utf-8"))
            config["stage_delay_ms"] = 350
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            state_dir = root / "state"
            report_path = root / "report.json"
            command = [
                sys.executable,
                "-m",
                "feature_pipeline",
                "run",
                "--config",
                str(config_path),
                "--events",
                str(SAMPLES / "events.jsonl"),
                "--features",
                str(SAMPLES / "features.json"),
                "--labels",
                str(SAMPLES / "labels.json"),
                "--state-dir",
                str(state_dir),
                "--report",
                str(report_path),
                "--reset",
            ]
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            current = state_dir / "CURRENT"
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline and not current.exists():
                time.sleep(0.02)
            process.terminate()
            stdout, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 0, f"stdout={stdout}\nstderr={stderr}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(report["stopped_gracefully"])
            self.assertIn(report["stop_reason"], {"signal", "stop_after:None"})
            self.assertIn(report["completed_stages"], (["events"], ["events", "features"]))

    def test_all_injection_kinds_hit_and_are_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            engine = PipelineEngine(Path(directory) / "state")
            engine.run(
                config_path=SAMPLES / "config.json",
                events_path=SAMPLES / "events.jsonl",
                features_path=SAMPLES / "features.json",
                labels_path=SAMPLES / "labels.json",
                reset=True,
            )
            report, code = engine.run(
                config_path=SAMPLES / "config.json",
                events_path=SAMPLES / "events.jsonl",
                features_path=SAMPLES / "features.json",
                labels_path=SAMPLES / "labels.json",
                edits_path=SAMPLES / "edits_injection.json",
                injections_path=SAMPLES / "injections.json",
            )
            self.assertEqual(code, 0)
            self.assertEqual(len(report["injected_items"]), 5)
            self.assertTrue(all(item["hit"] for item in report["injected_items"]))
            by_id = {item["injection_id"]: item for item in report["injected_items"]}
            self.assertEqual(by_id["INJ-01"]["business_code"], "E_EVENT_INVALID")
            self.assertEqual(by_id["INJ-03"]["business_code"], "E_SCHEMA_UNKNOWN_FIELD")
            self.assertEqual(by_id["INJ-05"]["business_code"], "E_TIMESTAMP_INVALID")
            self.assertTrue(report["failures"])

    def test_feature_stage_failure_is_event_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = json.loads((SAMPLES / "features.json").read_text(encoding="utf-8"))
            target = next(
                feature
                for feature in features["features"]
                if feature["name"] == "item_latest_dwell_8s"
            )
            target["source_field"] = "context.does_not_exist"
            features_path = root / "features-bad-field.json"
            features_path.write_text(json.dumps(features), encoding="utf-8")
            report, code = PipelineEngine(root / "state").run(
                config_path=SAMPLES / "config.json",
                events_path=SAMPLES / "events.jsonl",
                features_path=features_path,
                labels_path=SAMPLES / "labels.json",
                reset=True,
            )
            self.assertEqual(code, 0)
            feature_failures = [
                item for item in report["failures"] if item["code"] == "E_FEATURE_COMPUTE"
            ]
            self.assertEqual(len(feature_failures), 24)
            self.assertTrue(all(item["stage"] == "feature" for item in feature_failures))
            self.assertTrue(all(item["event_id"].startswith("e") for item in feature_failures))
            self.assertEqual(report["sample_count"], 4)


if __name__ == "__main__":
    unittest.main()
