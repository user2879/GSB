from __future__ import annotations

import json
from pathlib import Path
import os
import signal
import subprocess
import sys
import time

import pytest

from softmax_layernorm.cli import main


def test_bundled_verification_cli_passes(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    incremental = tmp_path / "incremental"
    full = tmp_path / "full"
    report = tmp_path / "verification.json"
    exit_code = main(
        [
            "verify",
            "--calls",
            str(root / "samples" / "calls.json"),
            "--tensor",
            str(root / "samples" / "tensor.json"),
            "--commands",
            str(root / "samples" / "commands.json"),
            "--config",
            str(root / "samples" / "config.json"),
            "--incremental-output",
            str(incremental),
            "--full-output",
            str(full),
            "--report",
            str(report),
        ]
    )
    assert exit_code == 0
    verification = json.loads(report.read_text(encoding="utf-8"))
    assert verification["passed"] is True
    assert verification["checks"]["arrays_equal"] is True
    assert verification["checks"]["failures_equal"] is True
    assert verification["checks"]["summary_equal"] is True
    run_report = json.loads((incremental / "reports" / "latest.json").read_text(encoding="utf-8"))
    assert run_report["snapshot_version"] >= 1
    assert len(run_report["commands"]) == 9
    assert all("response_latency_ms" in command for command in run_report["commands"])
    assert all("over_budget" in command for command in run_report["commands"])


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal delivery test")
def test_graceful_signal_finishes_snapshot_and_exits_zero(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "signal-run"
    run_dir.mkdir()
    tensor = {
        "shape": [180, 120],
        "seed": 11,
        "distribution": {"kind": "normal", "loc": 0, "scale": 1},
        "special_values": [],
    }
    calls = {
        "calls": [
            {
                "call_id": "signal-call",
                "sequence": 1,
                "rows": 180,
                "cols": 120,
                "softmax_axis": 1,
                "layernorm_axis": 1,
                "epsilon": 1e-5,
                "stabilize": True,
            }
        ]
    }
    commands = []
    for index in range(80):
        commands.append(
            {
                "command_id": f"signal-{index:03d}",
                "type": "replace_rows",
                "start": 0,
                "end": 4,
                "values": [[float(index + row + col) for col in range(120)] for row in range(4)],
            }
        )
    config = {
        "interaction_budget_ms": 2.0,
        "thread_pool_workers": 4,
        "snapshot_interval_commands": 100,
        "layernorm_scale": 1.0,
        "snapshot_root": str(run_dir / "output"),
        "resume": True,
    }
    paths = {}
    for name, value in (("tensor", tensor), ("calls", calls), ("commands", {"commands": commands}), ("config", config)):
        path = run_dir / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[name] = path

    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "softmax_layernorm.cli",
            "run",
            "--fresh",
            "--calls",
            str(paths["calls"]),
            "--tensor",
            str(paths["tensor"]),
            "--commands",
            str(paths["commands"]),
            "--config",
            str(paths["config"]),
            "--output",
            str(run_dir / "output"),
        ],
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    current_path = run_dir / "output" / "current.json"
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline and process.poll() is None and not current_path.exists():
        time.sleep(0.005)
    assert current_path.exists(), "initial atomic snapshot was not produced"
    process.send_signal(signal.SIGTERM)
    stdout, stderr = process.communicate(timeout=30)
    assert process.returncode == 0, f"stdout={stdout}\nstderr={stderr}"
    report = json.loads((run_dir / "output" / "reports" / "latest.json").read_text(encoding="utf-8"))
    assert report["exit_reason"] == "signal"
    current = json.loads(current_path.read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "output" / current["manifest"]).read_text(encoding="utf-8"))
    assert manifest["snapshot_version"] == current["snapshot_version"]
