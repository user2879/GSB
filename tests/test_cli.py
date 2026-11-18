from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run_demo(tmp_path: Path, injection: str):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    command = [
        sys.executable,
        "-m",
        "slowprof",
        "demo",
        "--samples-dir",
        str(ROOT / "samples"),
        "--runtime-dir",
        str(tmp_path / "runtime"),
        "--inject",
        injection,
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed, json.loads(completed.stdout)


@pytest.mark.parametrize(
    ("injection", "code", "family", "field", "exit_code"),
    [
        ("zero-weight", "E_SAMPLE", "E_SAMPLE", "weight", 2),
        ("depth-exceeded", "E_DEPTH", "E_SAMPLE", "frames", 3),
        ("missing-version", "E_VERSION", None, "metric_version", 4),
    ],
)
def test_cli_injections_are_logged(tmp_path, injection, code, family, field, exit_code):
    completed, payload = _run_demo(tmp_path, injection)
    assert completed.returncode == exit_code
    anomaly = payload["injected_anomaly"]
    assert anomaly["code"] == code
    assert anomaly["field"] == field
    if family is not None:
        assert anomaly["family"] == family
    log_path = tmp_path / "runtime" / "logs" / "slowprof.jsonl"
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    error_record = [record for record in records if record.get("event") == "error"][-1]
    assert error_record["code"] == code
    assert error_record["field"] == field
    assert error_record["injected"] == injection


def test_cli_parity_command(tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "slowprof",
            "parity",
            "--samples",
            str(ROOT / "samples" / "large.ndjson"),
            "--start-ms",
            "1710000000000",
            "--end-ms",
            "1710001000000",
            "--log-path",
            str(tmp_path / "logs" / "parity.jsonl"),
        ],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["byte_equal"] is True
    assert payload["full_path"] == "full"
    assert payload["streaming_path"] == "streaming"

