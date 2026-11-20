from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

import yaml

from scorecard_app.engine import ScorecardEngine
from scorecard_app.errors import E_SNAPSHOT_STALE

from tests.helpers import ROOT, base_inputs, clone, content_view


def _write_inputs(root: Path, experiment: dict[str, Any], schema: dict[str, Any]) -> tuple[Path, Path]:
    exp_path = root / "experiment.yaml"
    data_path = root / "data.yaml"
    exp_path.write_text(yaml.safe_dump(experiment, sort_keys=False), encoding="utf-8")
    data_path.write_text(yaml.safe_dump(schema, sort_keys=False), encoding="utf-8")
    return exp_path, data_path


def _cli_run(
    exp_path: Path,
    data_path: Path,
    state: Path,
    report: Path,
    *,
    env: dict[str, str] | None = None,
    extra: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        "scorecard_app",
        "run",
        "--config",
        str(exp_path),
        "--data",
        str(data_path),
        "--state-dir",
        str(state),
        "--report",
        str(report),
    ]
    if extra:
        command.extend(extra)
    environment = os.environ.copy()
    if env:
        environment.update(env)
    return subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _wait_for_path(path: Path, process: subprocess.Popen[str], timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists():
            return True
        if process.poll() is not None:
            return path.exists()
        time.sleep(0.03)
    return False


class RecoveryTests(unittest.TestCase):
    def test_expected_global_version_is_checked_without_changing_model_identity(self) -> None:
        experiment, schema = base_inputs(rows=220)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            state = root / "state"
            initial = ScorecardEngine(experiment, schema, state).run(report_path=root / "initial.json")
            version = initial["snapshot"]["version"]

            matching = clone(experiment)
            matching["snapshot"]["expected_global_version"] = version
            resumed = ScorecardEngine(matching, schema, state).run(report_path=root / "matching.json")
            self.assertEqual(resumed["status"], "ok")
            self.assertEqual(resumed["recovery"]["resume_version"], version)

            mismatching = clone(experiment)
            mismatching["snapshot"]["expected_global_version"] = version - 1
            stale = ScorecardEngine(mismatching, schema, state).run(report_path=root / "stale-version.json")
            self.assertEqual(stale["error"], "snapshot_stale")
            self.assertTrue(any(item["code"] == E_SNAPSHOT_STALE for item in stale["failures"]))

    def test_hard_exit_after_committed_stage_recovers_exact_result(self) -> None:
        experiment, schema = base_inputs(rows=300)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            exp_path, data_path = _write_inputs(root, experiment, schema)
            state = root / "state"
            killed = _cli_run(
                exp_path,
                data_path,
                state,
                root / "killed.json",
                env={"SCORECARD_FAILPOINT": "after_binning_stage"},
            )
            self.assertEqual(killed.returncode, 137)
            manifest = json.loads((state / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["last_completed_stage"], "binning")
            self.assertEqual(manifest["global_version"], 2)

            resumed = _cli_run(exp_path, data_path, state, root / "resumed.json")
            self.assertEqual(resumed.returncode, 0, resumed.stderr + resumed.stdout)
            resumed_report = json.loads((root / "resumed.json").read_text(encoding="utf-8"))
            self.assertEqual(resumed_report["recovery"]["resume_version"], 2)
            self.assertEqual(resumed_report["recovery"]["resumed_from_stage"], "binning")
            self.assertEqual(resumed_report["snapshot"]["version"], 5)
            full = ScorecardEngine(clone(experiment), clone(schema), root / "full").run(
                report_path=root / "full.json"
            )
            self.assertEqual(content_view(resumed_report), content_view(full))

    def test_hard_exit_after_column_cache_recovers_exact_result(self) -> None:
        experiment, schema = base_inputs(rows=300)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            exp_path, data_path = _write_inputs(root, experiment, schema)
            state = root / "state"
            killed_report = root / "killed.json"
            killed = _cli_run(
                exp_path,
                data_path,
                state,
                killed_report,
                env={"SCORECARD_FAILPOINT": "after_column_cache:binning:age"},
            )
            self.assertNotEqual(killed.returncode, 0)
            self.assertTrue((state / "manifest.json").exists())

            resumed_report = root / "resumed.json"
            resumed_process = _cli_run(exp_path, data_path, state, resumed_report)
            self.assertEqual(resumed_process.returncode, 0, resumed_process.stderr + resumed_process.stdout)
            resumed = json.loads(resumed_report.read_text(encoding="utf-8"))
            self.assertEqual(resumed["status"], "ok")
            self.assertGreaterEqual(resumed["snapshot"]["version"], 5)
            self.assertTrue(resumed["recovery"]["resume_version"] >= 1)

            full = ScorecardEngine(clone(experiment), clone(schema), root / "full").run(
                report_path=root / "full.json"
            )
            self.assertEqual(content_view(resumed), content_view(full))

    @unittest.skipIf(os.name == "nt", "SIGKILL is not available on Windows")
    def test_actual_sigkill_then_restart_recovers_exact_result(self) -> None:
        experiment, schema = base_inputs(rows=300)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            exp_path, data_path = _write_inputs(root, experiment, schema)
            state = root / "state"
            environment = os.environ.copy()
            environment["SCORECARD_TEST_TASK_DELAY_MS"] = "120"
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "scorecard_app",
                    "run",
                    "--config",
                    str(exp_path),
                    "--data",
                    str(data_path),
                    "--state-dir",
                    str(state),
                    "--report",
                    str(root / "killed.json"),
                ],
                cwd=ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                self.assertTrue(_wait_for_path(state / "manifest.json", process))
                self.assertIsNone(process.poll())
                os.kill(process.pid, signal.SIGKILL)
                process.communicate(timeout=20)
            finally:
                if process.poll() is None:
                    process.kill()
            self.assertEqual(process.returncode, -signal.SIGKILL)

            resumed = _cli_run(exp_path, data_path, state, root / "resumed.json")
            self.assertEqual(resumed.returncode, 0, resumed.stderr + resumed.stdout)
            resumed_report = json.loads((root / "resumed.json").read_text(encoding="utf-8"))
            self.assertEqual(resumed_report["status"], "ok")
            full = ScorecardEngine(clone(experiment), clone(schema), root / "full").run(
                report_path=root / "full.json"
            )
            self.assertEqual(content_view(resumed_report), content_view(full))

    @unittest.skipIf(os.name == "nt", "POSIX signals are used in this test")
    def test_sigterm_commits_current_work_and_exits_zero(self) -> None:
        experiment, schema = base_inputs(rows=300)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            exp_path, data_path = _write_inputs(root, experiment, schema)
            state = root / "state"
            report_path = root / "interrupted.json"
            environment = os.environ.copy()
            environment["SCORECARD_TEST_TASK_DELAY_MS"] = "180"
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "scorecard_app",
                    "run",
                    "--config",
                    str(exp_path),
                    "--data",
                    str(data_path),
                    "--state-dir",
                    str(state),
                    "--report",
                    str(report_path),
                ],
                cwd=ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                self.assertTrue(_wait_for_path(state / "manifest.json", process))
                self.assertIsNone(process.poll())
                process.send_signal(signal.SIGTERM)
                stdout, stderr = process.communicate(timeout=30)
            finally:
                if process.poll() is None:
                    process.kill()
            self.assertEqual(process.returncode, 0, stderr + stdout)
            self.assertTrue(report_path.exists())
            interrupted = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertIn(interrupted["status"], {"interrupted", "ok"})

            resumed = _cli_run(exp_path, data_path, state, root / "resumed.json")
            self.assertEqual(resumed.returncode, 0, resumed.stderr + resumed.stdout)
            resumed_report = json.loads((root / "resumed.json").read_text(encoding="utf-8"))
            full = ScorecardEngine(clone(experiment), clone(schema), root / "full").run(
                report_path=root / "full.json"
            )
            self.assertEqual(content_view(resumed_report), content_view(full))

    def test_stale_snapshot_is_rejected_then_explicit_revision_reuses_cache(self) -> None:
        experiment, schema = base_inputs(rows=300)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            state = root / "state"
            initial = ScorecardEngine(experiment, schema, state).run(report_path=root / "initial.json")
            self.assertEqual(initial["status"], "ok")
            original_manifest = (state / "manifest.json").read_bytes()
            original_version = initial["snapshot"]["version"]

            changed = clone(experiment)
            changed["binning"]["max_bins"] = 5
            stale = ScorecardEngine(changed, schema, state).run(report_path=root / "stale.json")
            self.assertEqual(stale["status"], "error")
            self.assertEqual(stale["error"], "snapshot_stale")
            self.assertTrue(any(item["code"] == E_SNAPSHOT_STALE for item in stale["failures"]))
            self.assertTrue(stale["recovery"]["refused"])
            self.assertEqual((state / "manifest.json").read_bytes(), original_manifest)
            self.assertEqual(stale["snapshot"]["version"], original_version)

            revised = ScorecardEngine(changed, schema, state, allow_revision=True).run(
                report_path=root / "revised.json", allow_revision=True
            )
            self.assertEqual(revised["status"], "ok")
            self.assertTrue(revised["recovery"]["revision_started"])
            self.assertIn("income", revised["cache"]["by_column"])
            full = ScorecardEngine(clone(changed), clone(schema), root / "full").run(
                report_path=root / "full.json"
            )
            self.assertEqual(content_view(revised), content_view(full))

    def test_data_description_change_is_detected_as_stale(self) -> None:
        experiment, schema = base_inputs(rows=220)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            state = root / "state"
            initial = ScorecardEngine(experiment, schema, state).run(report_path=root / "initial.json")
            manifest_before = (state / "manifest.json").read_bytes()

            changed_schema = clone(schema)
            for column in changed_schema["columns"]:
                if column["name"] == "income":
                    column["seed_offset"] = 9
                    break
            stale = ScorecardEngine(experiment, changed_schema, state).run(report_path=root / "stale-data.json")
            self.assertEqual(stale["error"], "snapshot_stale")
            self.assertTrue(any(item["code"] == E_SNAPSHOT_STALE for item in stale["failures"]))
            self.assertEqual((state / "manifest.json").read_bytes(), manifest_before)
            self.assertNotEqual(initial["input"]["data_fingerprint"], stale["input"]["data_fingerprint"])


if __name__ == "__main__":
    unittest.main()
