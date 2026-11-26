from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from mdet_render.samples import create_sample_workspace
from mdet_render.workspace import Workspace


@pytest.fixture
def sample_root(tmp_path: Path) -> Path:
    root = tmp_path / "samples"
    create_sample_workspace(root, width=32, height=24, frames=3, denoise_steps=4, max_workers=2)
    return root


def read_config(root: Path) -> dict[str, Any]:
    return json.loads((root / "config.json").read_text(encoding="utf-8"))


def write_config(root: Path, config: dict[str, Any]) -> None:
    (root / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n",
                                      encoding="utf-8")


def edit_config(root: Path, editor: Callable[[dict[str, Any]], None]) -> None:
    config = read_config(root)
    editor(config)
    write_config(root, config)


def load_workspace(root: Path) -> Workspace:
    return Workspace.load(root / "config.json")


def ledger_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ledger_tasks(path: Path, run_id: str) -> list[dict[str, Any]]:
    return [entry for entry in ledger_entries(path)
            if entry.get("run_id") == run_id and entry.get("event") == "task"]


def computed_keys(path: Path, run_id: str) -> list[str]:
    return [entry["stable_key"] for entry in ledger_tasks(path, run_id)
            if entry["outcome"] == "computed"]


def task_by_stable(path: Path, run_id: str) -> dict[str, dict[str, Any]]:
    return {entry["stable_key"]: entry for entry in ledger_tasks(path, run_id)}
