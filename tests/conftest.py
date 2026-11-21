from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_cli(*args: str | Path, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-m", "vc_workbench", *(str(arg) for arg in args)]
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout)


@pytest.fixture(scope="session")
def baseline(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    root = tmp_path_factory.mktemp("baseline")
    state = root / "state"
    output = root / "output.wav"
    report = root / "report.json"
    result = run_cli(
        "run",
        "--samples",
        SAMPLES,
        "--config",
        SAMPLES / "config.json",
        "--commands",
        SAMPLES / "commands.json",
        "--state",
        state,
        "--output",
        output,
        "--report",
        report,
        "--fresh",
    )
    assert result.returncode == 0, result.stderr + result.stdout
    return {
        "root": root,
        "state": state,
        "output": output,
        "report_path": report,
        "report": json.loads(report.read_text(encoding="utf-8")),
    }


@pytest.fixture
def samples_copy(tmp_path: Path) -> Path:
    import shutil

    target = tmp_path / "samples"
    shutil.copytree(SAMPLES, target)
    return target
