from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from gsscene.sample import generate_sample
from gsscene.util import atomic_write_json, read_json


@pytest.fixture(scope="session")
def base_sample(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("base-sample")
    return generate_sample(root / "sample", width=16, height=16, views=6, force=True)


def copy_sample(source: Path, target: Path) -> Path:
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    return target


def write_edits(sample: Path, edits: list[dict[str, Any]]) -> None:
    with (sample / "edits.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for edit in edits:
            handle.write(json.dumps(edit, sort_keys=True, separators=(",", ":")) + "\n")


def update_config(sample: Path, **changes: Any) -> dict[str, Any]:
    config = read_json(sample / "config.json")
    config.update(changes)
    atomic_write_json(sample / "config.json", config)
    return config


def update_injections(sample: Path, injections: list[dict[str, Any]]) -> None:
    atomic_write_json(sample / "injections.json", injections)
