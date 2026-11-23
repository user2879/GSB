from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_cli_writes_json_report_and_stdout(tmp_path):
    output = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rerank_lab",
            "--sample-dir",
            "samples/default",
            "--output-dir",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    stdout_report = json.loads(result.stdout)
    file_report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert stdout_report == file_report
    assert stdout_report["summary"]["status"] == "ok"
