from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from tile_anomaly.edits import clone_sample


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASE_SAMPLE = REPOSITORY_ROOT / "sample"


def make_workspace(name: str = "tile-test") -> Path:
    root = Path(tempfile.mkdtemp(prefix=f"{name}-"))
    clone_sample(BASE_SAMPLE, root)
    artifacts = root / "artifacts"
    if artifacts.exists():
        shutil.rmtree(artifacts)
    return root


def config_path(root: Path) -> Path:
    return root / "config" / "detection.yaml"


def read_config(root: Path) -> dict[str, Any]:
    return json.loads(config_path(root).read_text(encoding="utf-8"))


def write_config(root: Path, value: dict[str, Any]) -> None:
    config_path(root).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def set_config_value(root: Path, dotted_path: str, value: Any) -> None:
    from tile_anomaly.util import deep_set

    config = read_config(root)
    deep_set(config, dotted_path, value)
    write_config(root, config)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

