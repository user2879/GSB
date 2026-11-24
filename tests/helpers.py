from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


def copy_samples(destination: Path) -> Path:
    target = destination / "samples"
    shutil.copytree(SAMPLES, target)
    return target


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sample_paths(base: Path) -> dict[str, str]:
    return {
        "catalog": str(base / "base" / "catalog.json"),
        "views": str(base / "views.json"),
        "views_v2": str(base / "views_v2.json"),
        "changes": str(base / "changes.ndjson"),
    }
