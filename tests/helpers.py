from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from batch_audit.demo import create_demo_workspace


def make_workspace(root: Path, **kwargs: Any):
    options = {
        "seed": 20261120,
        "sample_count": 12,
        "input_dim": 6,
        "hidden_dim": 8,
        "output_dim": 3,
        "supported_slots": (0, 1, 2, 3, 4),
    }
    options.update(kwargs)
    workspace = root / "workspace"
    created = create_demo_workspace(workspace, **options)
    return workspace, created


def run_cli(args: list[str], *, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    return subprocess.run(
        [sys.executable, "-m", "batch_audit.cli", *args],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def assignment_map(report: dict[str, Any]) -> dict[str, tuple[int, int, str, str]]:
    result: dict[str, tuple[int, int, str, str]] = {}
    for slot in report["slots"]:
        for member in slot["members"]:
            result[member["sample_id"]] = (
                slot["batch_slot"],
                member["rank"],
                member["assignment_id"],
                slot["output_digest"],
            )
    return result
