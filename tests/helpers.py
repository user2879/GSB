from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


def sample(name: str) -> Path:
    return SAMPLES / f"{name}.json"


def run_cli(*args: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [sys.executable, "-m", "txdiag", *args],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

