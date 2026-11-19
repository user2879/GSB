from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from copy import deepcopy
from pathlib import Path

from mesh_audit.persistence import StateStore
from mesh_audit.sample import deterministic_mesh_payload, deterministic_repair_script


ROOT = Path(__file__).resolve().parents[1]


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    return environment


def _run_cli(arguments: list[str], timeout: float = 40.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "mesh_audit", *arguments],
        cwd=ROOT,
        env=_environment(),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _wait_for(path: Path, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.005)
    raise AssertionError(f"timed out waiting for {path}")


def _core_payloads(state_dir: Path) -> dict[int, dict]:
    store = StateStore(state_dir)
    manifest = store.load_manifest()
    records = store.valid_snapshots_for_resume(manifest["version_id"])
    result: dict[int, dict] = {}
    for record in records:
        assert record.payload is not None
        payload = record.payload
        result[record.step] = {
            "fixed": payload["mesh"]["fixed"],
            "reference": payload["mesh"]["reference"],
            "faces": payload["mesh"]["faces"],
            "execution_count": payload["execution_count"],
            "operator_effect": payload["operator_effect"],
            "vertex_errors": payload["error"]["vertex_errors"],
            "propagation": payload["propagation"],
        }
    return result


@unittest.skipIf(sys.platform == "win32", "POSIX signal integration test")
class SignalInterruptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.mesh_path = self.root / "mesh.json"
        self.script_path = self.root / "script.json"
        self.mesh_path.write_text(json.dumps(deterministic_mesh_payload()), encoding="utf-8")
        script = deepcopy(deterministic_repair_script())
        script["operators"][-1]["iterations"] = 4000
        script["operators"][-1]["lambda"] = 1.0e-9
        self.script_path.write_text(json.dumps(script), encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _arguments(self, label: str) -> list[str]:
        return [
            "run",
            "--mesh",
            str(self.mesh_path),
            "--script",
            str(self.script_path),
            "--state",
            str(self.root / f"state-{label}"),
            "--output",
            str(self.root / f"output-{label}"),
        ]

    def _baseline(self) -> dict[int, dict]:
        completed = _run_cli(self._arguments("baseline"))
        self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
        return _core_payloads(self.root / "state-baseline")

    def test_sigterm_finishes_current_operator_then_resumes(self) -> None:
        baseline = self._baseline()
        state = self.root / "state-term" / "current"
        process = subprocess.Popen(
            [sys.executable, "-m", "mesh_audit", *self._arguments("term")],
            cwd=ROOT,
            env=_environment(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _wait_for(state / "step-0003.json")
        time.sleep(0.25)
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=30)
        self.assertEqual(process.returncode, 143, stdout + stderr)
        self.assertTrue((state / "step-0004.json").exists())
        self.assertEqual(StateStore(self.root / "state-term").inspect().invalid_snapshot_count, 0)

        resumed = _run_cli(self._arguments("term"))
        self.assertEqual(resumed.returncode, 0, resumed.stderr + resumed.stdout)
        output = json.loads(resumed.stdout)
        self.assertEqual(output["resumed_from_step"], 4)
        self.assertEqual(_core_payloads(self.root / "state-term"), baseline)

    def test_sigkill_resumes_only_from_valid_persisted_snapshots(self) -> None:
        baseline = self._baseline()
        state = self.root / "state-kill" / "current"
        process = subprocess.Popen(
            [sys.executable, "-m", "mesh_audit", *self._arguments("kill")],
            cwd=ROOT,
            env=_environment(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _wait_for(state / "step-0003.json")
        process.kill()
        process.communicate(timeout=10)
        self.assertNotEqual(process.returncode, 0)
        inspection = StateStore(self.root / "state-kill").inspect()
        self.assertEqual(inspection.invalid_snapshot_count, 0)
        self.assertGreaterEqual(inspection.latest_contiguous_valid_step, 3)
        self.assertLessEqual(inspection.latest_contiguous_valid_step, 4)

        resumed = _run_cli(self._arguments("kill"))
        self.assertEqual(resumed.returncode, 0, resumed.stderr + resumed.stdout)
        self.assertEqual(_core_payloads(self.root / "state-kill"), baseline)


if __name__ == "__main__":
    unittest.main()
