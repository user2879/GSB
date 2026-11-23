from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


@pytest.fixture
def sample_copy(tmp_path: Path):
    counter = {"value": 0}

    def _copy(name: str = "default") -> Path:
        counter["value"] += 1
        target = tmp_path / f"sample-{name}-{counter['value']}"
        shutil.copytree(SAMPLES / name, target)
        return target

    return _copy


def patch_config(sample_dir: Path, **changes: Any) -> dict[str, Any]:
    path = sample_dir / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config.update(changes)
    path.write_text(
        json.dumps(config, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return config


def write_commands(sample_dir: Path, commands: list[dict[str, Any]]) -> None:
    (sample_dir / "commands.jsonl").write_text(
        "".join(
            json.dumps(command, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for command in commands
        ),
        encoding="utf-8",
        newline="\n",
    )
