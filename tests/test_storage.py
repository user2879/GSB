from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from spectral_diag.canonical import canonical_bytes
from spectral_diag.config import RunConfig
from spectral_diag.errors import DiagnosticError
from spectral_diag.pipeline import execute_pipeline
from spectral_diag.storage import Workspace, atomic_write_bytes, atomic_write_json


class StorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.workspace = Workspace(self.root).ensure()
        self.result = execute_pipeline(RunConfig())

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_normal_commit_writes_closed_manifest(self) -> None:
        self.workspace.begin("normal")
        commit = self.workspace.stage_and_commit("normal", self.result)
        current = self.workspace.load_current()
        self.assertTrue(commit["closed"])
        self.assertTrue(current["manifest"]["closed"])
        self.assertTrue(current["closed_marker"]["closed"])
        self.assertEqual(current["report"]["result_sha256"], self.result["result_sha256"])
        self.assertFalse(self.workspace.active_path.exists())
        self.assertTrue((self.workspace.snapshots / "snap-normal" / "scene.json").exists())

    def test_uncommitted_snapshot_is_recovered_as_orphan(self) -> None:
        self.workspace.begin("stable")
        stable = self.workspace.stage_and_commit("stable", self.result)

        class ForcedTermination(Exception):
            pass

        def stop(code: int) -> None:
            raise ForcedTermination(code)

        with self.assertRaises(ForcedTermination):
            self.workspace.begin("crash")
            self.workspace.stage_and_commit(
                "crash", self.result, crash_after_stage=True, exit_fn=stop
            )
        recovery = self.workspace.recover()
        current = self.workspace.load_current()
        self.assertEqual(current["manifest"]["snapshot_id"], stable["snapshot_id"])
        self.assertGreaterEqual(recovery["orphan_count"], 1)
        self.assertEqual(recovery["interrupted_run"]["run_id"], "crash")
        self.assertFalse(recovery["normal_exit_report_fabricated"])
        self.assertFalse(recovery["orphans"][0]["closed"])

    def test_real_forced_termination_returns_86_and_recovers(self) -> None:
        self.workspace.begin("stable")
        stable = self.workspace.stage_and_commit("stable", self.result)
        config_path = self.root / "crash-config.json"
        atomic_write_bytes(config_path, canonical_bytes(RunConfig().to_dict()))
        project_root = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(project_root) + os.pathsep + environment.get("PYTHONPATH", "")
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "spectral_diag",
                "run",
                "--workspace",
                str(self.root),
                "--run-id",
                "crash-real",
                "--crash-after-stage",
                "--config-json",
                str(config_path),
            ],
            cwd=str(project_root),
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        self.assertEqual(process.returncode, 86, process.stderr)
        recovery = self.workspace.recover()
        self.assertEqual(recovery["current"]["snapshot_id"], stable["snapshot_id"])
        self.assertEqual(recovery["interrupted_run"]["run_id"], "crash-real")

    def test_snapshot_version_mismatch_is_explicit(self) -> None:
        self.workspace.begin("stable")
        self.workspace.stage_and_commit("stable", self.result)
        pointer = json.loads((self.root / "CURRENT.json").read_text(encoding="utf-8"))
        pointer["schema_version"] = 999
        atomic_write_json(self.root / "CURRENT.json", pointer)
        with self.assertRaises(DiagnosticError) as context:
            self.workspace.load_current()
        self.assertEqual(context.exception.code.value, "E_SNAPSHOT_VERSION")

    def test_tampered_snapshot_file_fails_integrity_check(self) -> None:
        self.workspace.begin("stable")
        self.workspace.stage_and_commit("stable", self.result)
        matrix_path = self.workspace.snapshots / "snap-stable" / "matrix.json"
        with open(matrix_path, "ab") as handle:
            handle.write(b"tamper")
        with self.assertRaises(DiagnosticError):
            self.workspace.load_current()

    def test_committed_history_is_not_recovered_as_orphan(self) -> None:
        self.workspace.begin("first")
        first = self.workspace.stage_and_commit("first", self.result)
        second_result = dict(self.result)
        second_result["run_id"] = "second"
        self.workspace.begin("second")
        second = self.workspace.stage_and_commit("second", second_result)
        recovery = self.workspace.recover()
        self.assertEqual(recovery["orphan_count"], 0)
        self.assertEqual(recovery["current"]["snapshot_id"], second["snapshot_id"])
        pointer = json.loads((self.root / "CURRENT.json").read_text(encoding="utf-8"))
        self.assertIn(first["snapshot_id"], pointer["history"])


if __name__ == "__main__":
    unittest.main()
