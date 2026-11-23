from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_cli_generate_and_run(tmp_path: Path) -> None:
    samples = tmp_path / "samples"
    state = tmp_path / "state"
    output = tmp_path / "output"
    generated = subprocess.run(
        [sys.executable, "-m", "ancient_vlayout", "generate", "--samples", str(samples)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "config_hash" in generated.stdout
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ancient_vlayout",
            "run",
            "--samples",
            str(samples),
            "--state-dir",
            str(state),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    assert summary["images"] == 4
    assert summary["characters"] == 80
    assert summary["recognized"] == 80
    assert summary["failures"] == 0
    assert summary["pending"] == 0
    for name in (
        "recognition.json",
        "report.json",
        "diagnostics.json",
        "snapshot_manifest.json",
        "run_summary.json",
    ):
        assert (output / name).is_file()
