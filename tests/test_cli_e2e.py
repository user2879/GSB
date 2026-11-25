from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cli_verify_end_to_end(tmp_path):
    output = tmp_path / "report.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "vscan",
            "verify",
            "--spec",
            str(ROOT / "examples" / "spec.json"),
            "--data",
            str(tmp_path / "data"),
            "--ledger",
            str(tmp_path / "ledger.sqlite"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["ok"] is True
    assert summary["full_rebuild_verification"]["consistent"] is True
    assert summary["scheduler"]["max_observed_inflight"] <= 3
    assert summary["scheduler"]["max_observed_decoded_bytes"] <= 1200
    report = json.loads(output.read_text(encoding="utf-8"))
    assert len(report["queries"]) == 2
    assert len(report["batch_results"]) == 16
    assert report["columns"]
