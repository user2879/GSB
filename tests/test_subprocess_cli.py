import json
import subprocess
import sys

from conftest import ROOT


def test_cli_runs_in_subprocess_with_temp_state(tmp_path):
    state = tmp_path / "state"
    output = tmp_path / "ledger.json"
    report = tmp_path / "report.json"
    cmd = [
        sys.executable, "-m", "qcalib.cli", "run",
        "--weights", str(ROOT / "samples" / "weights.jsonl"),
        "--calibration", str(ROOT / "samples" / "calibration.json"),
        "--config", str(ROOT / "configs" / "pipeline.yaml"),
        "--state", str(state),
        "--output", str(output),
        "--report", str(report),
    ]
    completed = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload["layers"]) == 4
    assert state.exists() and report.exists()
    diagnostic = json.loads(report.read_text(encoding="utf-8"))
    assert diagnostic["layer_count"] == 4


def test_cli_diff_subprocess_return_codes(tmp_path):
    left = tmp_path / "a.json"
    right = tmp_path / "b.json"
    left.write_text('{"x": 1}', encoding="utf-8")
    right.write_text('{"x": 1}', encoding="utf-8")
    ok = subprocess.run(
        [sys.executable, "-m", "qcalib.cli", "diff", str(left), str(right)],
        cwd=ROOT, capture_output=True, text=True)
    assert ok.returncode == 0
    right.write_text('{"x": 2}', encoding="utf-8")
    mismatch = subprocess.run(
        [sys.executable, "-m", "qcalib.cli", "diff", str(left), str(right)],
        cwd=ROOT, capture_output=True, text=True)
    assert mismatch.returncode == 3
