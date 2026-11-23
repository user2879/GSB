from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inpaint_pyramid.config import load_commands, load_config
from inpaint_pyramid.samplegen import generate_sample


def make_sample(
    directory: str | Path,
    *,
    size: int = 64,
    levels: int = 3,
    tile_size: int = 16,
    budget: int = 3,
    threads: int = 4,
) -> tuple[Any, Any, dict[str, Any]]:
    directory = Path(directory)
    manifest = generate_sample(
        directory,
        size=size,
        levels=levels,
        tile_size=tile_size,
        tile_budget=budget,
        threads=threads,
    )
    bundle = load_config(directory / "config.json", directory)
    stream = load_commands(directory / "commands.json")
    return bundle, stream, manifest


def clone_bundle(bundle: Any, *, budget: int | None = None, threads: int | None = None) -> Any:
    cloned = copy.copy(bundle)
    if budget is not None:
        cloned.recompute_tile_budget = int(budget)
    if threads is not None:
        cloned.threads = int(threads)
    return cloned


def write_json(path: str | Path, value: Any) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
