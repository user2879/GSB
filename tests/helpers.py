from __future__ import annotations

import copy
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from mgirecon.config import default_config, save_config
from mgirecon.synth import initialize_project


def make_project(prefix: str = "mgirecon-test-", config: dict[str, Any] | None = None) -> tuple[tempfile.TemporaryDirectory, Path, dict[str, Any]]:
    temporary = tempfile.TemporaryDirectory(prefix=prefix)
    root = Path(temporary.name)
    effective = config if config is not None else default_config()
    initialize_project(root, config=effective)
    return temporary, root, copy.deepcopy(effective)


def copy_project(source: Path, prefix: str = "mgirecon-copy-") -> tuple[tempfile.TemporaryDirectory, Path]:
    temporary = tempfile.TemporaryDirectory(prefix=prefix)
    target = Path(temporary.name) / "project"
    shutil.copytree(source, target)
    snapshot = target / ".mgirecon"
    if snapshot.exists():
        shutil.rmtree(snapshot)
    return temporary, target


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
