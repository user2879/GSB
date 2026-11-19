from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from .common import (
    SAMPLES,
    adapter_payloads,
    cli_command,
    cli_environment,
    current_manifest,
    read_report,
    run_workspace,
    send_graceful_signal,
    subprocess_signal_flags,
    wait_for,
)


class SnapshotTests(unittest.TestCase):
    def _start(self, workspace: Path, extra_environment: dict[str, str]) -> subprocess.Popen[str]:
        environment = cli_environment()
        environment.update(extra_environment)
        return subprocess.Popen(
            cli_command(SAMPLES, workspace),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
            **subprocess_signal_flags(),
        )

    def test_graceful_signal_finishes_current_stage_and_exits_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "work"
            process = self._start(
                workspace,
                {"PEFT_TEST_HOLD_STAGE": "merge", "PEFT_TEST_HOLD_MS": "5000"},
            )
            try:
                wait_for(
                    lambda: bool(
                        list((workspace / "snapshot" / ".tmp").glob("stage-merge-*"))
                    )
                )
                send_graceful_signal(process)
                stdout, stderr = process.communicate(timeout=30)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate(timeout=10)
            self.assertEqual(process.returncode, 0, stderr)
            summary = json.loads(stdout.strip().splitlines()[-1])
            self.assertEqual(summary["status"], "graceful_stop")
            self.assertEqual(summary["exit_code"], 0)
            self.assertEqual(summary["shutdown"]["completed_stage"], "merge")
            manifest = current_manifest(workspace)
            self.assertIn("merge", manifest["stages"])
            self.assertNotIn("export", manifest["stages"])
            report = read_report(workspace)
            self.assertEqual(report["status"], "graceful_stop")

    def test_hard_kill_during_partial_training_resumes_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clean_workspace = root / "clean"
            self.assertEqual(run_workspace(SAMPLES, clean_workspace).exit_code, 0)
            clean_adapters = adapter_payloads(read_report(clean_workspace), clean_workspace)

            workspace = root / "work"
            process = self._start(
                workspace,
                {
                    "PEFT_TEST_HOLD_AFTER_STAGE": "training",
                    "PEFT_TEST_HOLD_AFTER_MS": "15000",
                },
            )
            try:
                def partial_stage_ready() -> bool:
                    current = current_manifest(workspace)
                    training = current.get("stages", {}).get("training")
                    if not training:
                        return False
                    index_path = (
                        workspace / "snapshot" / training["path"] / "index.json"
                    )
                    if not index_path.is_file():
                        return False
                    index = json.loads(index_path.read_text(encoding="utf-8"))
                    return index.get("complete") is False

                wait_for(partial_stage_ready)
                checkpoint_version = int(current_manifest(workspace)["global_version"])
                process.kill()
                process.communicate(timeout=10)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate(timeout=10)
            self.assertNotEqual(process.returncode, 0)
            self.assertEqual(current_manifest(workspace)["global_version"], checkpoint_version)

            environment = cli_environment()
            restarted = subprocess.run(
                cli_command(SAMPLES, workspace),
                capture_output=True,
                text=True,
                env=environment,
                timeout=60,
            )
            self.assertEqual(restarted.returncode, 0, restarted.stderr)
            summary = json.loads(restarted.stdout.strip().splitlines()[-1])
            self.assertEqual(summary["recovery"]["started_from_version"], checkpoint_version)
            report = read_report(workspace)
            self.assertEqual(report["status"], "success")
            resumed_adapters = adapter_payloads(report, workspace)
            for task_name in clean_adapters:
                self.assertEqual(
                    resumed_adapters[task_name]["content_hash"],
                    clean_adapters[task_name]["content_hash"],
                )

    def test_hard_kill_during_merge_never_publishes_partial_merge(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "work"
            process = self._start(
                workspace,
                {"PEFT_TEST_HOLD_STAGE": "merge", "PEFT_TEST_HOLD_MS": "15000"},
            )
            try:
                wait_for(
                    lambda: bool(
                        list((workspace / "snapshot" / ".tmp").glob("stage-merge-*"))
                    )
                )
                checkpoint_version = int(current_manifest(workspace)["global_version"])
                process.kill()
                process.communicate(timeout=10)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate(timeout=10)
            self.assertEqual(current_manifest(workspace)["global_version"], checkpoint_version)
            self.assertNotIn("merge", current_manifest(workspace)["stages"])
            restarted = subprocess.run(
                cli_command(SAMPLES, workspace),
                capture_output=True,
                text=True,
                env=cli_environment(),
                timeout=60,
            )
            self.assertEqual(restarted.returncode, 0, restarted.stderr)
            summary = json.loads(restarted.stdout.strip().splitlines()[-1])
            self.assertEqual(summary["recovery"]["started_from_version"], checkpoint_version)
            manifest = current_manifest(workspace)
            self.assertIn("merge", manifest["stages"])
            self.assertIn("export", manifest["stages"])


if __name__ == "__main__":
    unittest.main()
