from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from urllib.parse import quote

from ner_editor.snapshot import SnapshotManager
from tests.helpers import command, make_engine

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main.py"
SAMPLES = ROOT / "samples"


def run_cli(*args: str, timeout: int = 90):
    return subprocess.run(
        [sys.executable, str(MAIN), *args],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        timeout=timeout,
    )


class CliSnapshotTests(unittest.TestCase):
    def test_resume_from_generation_and_equivalence(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            report_one = workspace / "part1.json"
            first = run_cli(
                "--workspace",
                str(workspace),
                "--run-id",
                "resume-test",
                "--max-commands",
                "3",
                "--report",
                str(report_one),
                "--quiet",
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            part_one = json.loads(report_one.read_text(encoding="utf-8"))
            self.assertEqual(part_one["global_version"], 3)
            manifest = part_one["snapshot"]["manifest"]
            self.assertEqual(manifest["global_version"], 3)
            self.assertEqual(manifest["command_position"], 3)
            self.assertEqual(manifest["completed_samples"], sorted(part_one["samples"][i]["sample_id"] for i in range(len(part_one["samples"]))))
            self.assertTrue(manifest["config_hash"])
            self.assertTrue(manifest["model_version"])
            self.assertTrue(manifest["stage_versions"]["merge"])

            report_two = workspace / "part2.json"
            second = run_cli(
                "--workspace",
                str(workspace),
                "--run-id",
                "resume-test",
                "--resume",
                "--verify-full",
                "--report",
                str(report_two),
                "--quiet",
            )
            self.assertEqual(second.returncode, 0, second.stderr)
            part_two = json.loads(report_two.read_text(encoding="utf-8"))
            self.assertTrue(part_two["resumed_from_snapshot"])
            self.assertIsNone(part_two["resume_failure"])
            self.assertEqual(part_two["global_version"], 12)
            self.assertEqual([item["command_position"] for item in part_two["command_responses"]], list(range(1, 13)))
            self.assertTrue(part_two["equivalence"]["passed"], part_two["equivalence"]["differences"][:2])

    def test_invalid_inputs_have_business_codes_and_fatal_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            corpus = workspace / "corpus.jsonl"
            corpus.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "good", "chars": "张 三", "labels": "B E", "types": "PER PER"}, ensure_ascii=False),
                        json.dumps({"id": "bad", "chars": "张 三", "labels": "B", "types": "PER"}, ensure_ascii=False),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            commands = workspace / "commands.json"
            commands.write_text(
                json.dumps(
                    [
                        {"type": "not-a-command"},
                        {"type": "emission_adjust", "target": {"dimension": 0, "label": "B"}},
                    ]
                ),
                encoding="utf-8",
            )
            report_path = workspace / "invalid-report.json"
            completed = run_cli(
                "--workspace",
                str(workspace),
                "--corpus",
                str(corpus),
                "--commands",
                str(commands),
                "--report",
                str(report_path),
                "--no-snapshot",
                "--quiet",
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(len(report["samples"]), 1)
            self.assertGreaterEqual(report["code_counts"].get("E_CORPUS_INVALID", 0), 1)
            self.assertGreaterEqual(report["code_counts"].get("E_COMMAND_INVALID", 0), 2)

            bad_model = json.loads((SAMPLES / "model.json").read_text(encoding="utf-8"))
            bad_model["feature_dim"] = 0
            model_path = workspace / "bad-model.json"
            model_path.write_text(json.dumps(bad_model, ensure_ascii=False), encoding="utf-8")
            fatal_report = workspace / "fatal.json"
            fatal = run_cli(
                "--workspace",
                str(workspace),
                "--model",
                str(model_path),
                "--report",
                str(fatal_report),
                "--no-snapshot",
                "--quiet",
            )
            self.assertEqual(fatal.returncode, 2)
            fatal_data = json.loads(fatal_report.read_text(encoding="utf-8"))
            self.assertTrue(fatal_data["fatal"])
            self.assertIn("E_MODEL_PARAM_INVALID", fatal_data["code_counts"])

    def test_corrupt_snapshot_is_rejected_and_run_recovers_from_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            first_report = workspace / "first.json"
            first = run_cli(
                "--workspace",
                str(workspace),
                "--run-id",
                "corrupt-test",
                "--max-commands",
                "1",
                "--report",
                str(first_report),
                "--quiet",
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            pointer = workspace / "artifacts" / "snapshots" / "corrupt-test" / "CURRENT"
            pointer.write_text("gen-999999\n", encoding="utf-8")
            recovered_report = workspace / "recovered.json"
            recovered = run_cli(
                "--workspace",
                str(workspace),
                "--run-id",
                "corrupt-test",
                "--resume",
                "--max-commands",
                "1",
                "--report",
                str(recovered_report),
                "--quiet",
            )
            self.assertEqual(recovered.returncode, 0, recovered.stderr)
            report = json.loads(recovered_report.read_text(encoding="utf-8"))
            self.assertFalse(report["resumed_from_snapshot"])
            self.assertEqual(report["resume_failure"]["code"], "E_SNAPSHOT_INVALID")
            self.assertGreaterEqual(report["code_counts"].get("E_SNAPSHOT_INVALID", 0), 1)

    @unittest.skipIf(os.name == "nt", "POSIX signal exit mapping is platform-specific")
    def test_sigint_finishes_current_command_snapshot_then_resumes(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            report_one = workspace / "signal1.json"
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(MAIN),
                    "--workspace",
                    str(workspace),
                    "--run-id",
                    "signal-test",
                    "--report",
                    str(report_one),
                    "--quiet",
                ],
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            time.sleep(0.8)
            process.send_signal(signal.SIGINT)
            stdout, stderr = process.communicate(timeout=90)
            self.assertEqual(process.returncode, 130, stderr)
            part_one = json.loads(report_one.read_text(encoding="utf-8"))
            self.assertEqual(part_one["exit_code"], 130)
            self.assertGreaterEqual(part_one["global_version"], 1)
            self.assertTrue(part_one["snapshot"]["manifest"]["completed_samples"])
            self.assertTrue(part_one["snapshot"]["manifest"]["stage_versions"]["merge"])

            report_two = workspace / "signal2.json"
            resumed = run_cli(
                "--workspace",
                str(workspace),
                "--run-id",
                "signal-test",
                "--resume",
                "--verify-full",
                "--report",
                str(report_two),
                "--quiet",
            )
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            part_two = json.loads(report_two.read_text(encoding="utf-8"))
            self.assertTrue(part_two["resumed_from_snapshot"])
            self.assertTrue(part_two["equivalence"]["passed"])

    def test_snapshot_versions_reuse_unchanged_stage_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            engine = make_engine(sample_limit=2, budget_ms=60000, workers=2)
            manager = SnapshotManager(workspace, "hardlink-test")
            try:
                engine.settle()
                first = manager.commit(engine, reason="first")
                self.assertEqual(first["status"], "committed")
                engine.apply_command(command(1, "noop"))
                engine.settle()
                second = manager.commit(engine, reason="after noop")
                self.assertEqual(second["status"], "committed")
                self.assertEqual(second["changed_stages"], [])
                filename = quote("s001", safe="") + ".json"
                first_path = Path(first["generation_path"]) / "stages" / "feature" / filename
                second_path = Path(second["generation_path"]) / "stages" / "feature" / filename
                self.assertTrue(first_path.exists() and second_path.exists())
                if os.name != "nt":
                    self.assertEqual(first_path.stat().st_ino, second_path.stat().st_ino)
                self.assertEqual(
                    (workspace / "artifacts" / "snapshots" / "hardlink-test" / "CURRENT").read_text(encoding="utf-8").strip(),
                    f"gen-{second['generation']:06d}",
                )
            finally:
                engine.close()


if __name__ == "__main__":
    unittest.main()
