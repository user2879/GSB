"""Shared helpers for the test suite."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mstc.config import load_config
from mstc.loader import load_manifest
from mstc.paths import to_path
from mstc.pipeline import Pipeline
from mstc.samplegen import generate_sample

ROOT = Path(__file__).resolve().parents[1]


def make_workspace(tmp_path: Path, edge: int = 16, overlap: int = 4,
                   grid=(3, 3), config_name: str = "config.yaml") -> Path:
    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    generate_sample(ws, grid[0], grid[1], edge=edge, overlap=overlap)
    shutil.copy(ROOT / "configs" / config_name, ws / config_name)
    return ws


def run(ws: Path, config_name: str = "config.yaml", version: int = 1):
    config = load_config(ws / config_name)
    manifest = load_manifest(to_path(config.manifest_path, ws), ws)
    grid = (max(t.row for t in manifest.tiles) + 1,
            max(t.col for t in manifest.tiles) + 1)
    return Pipeline(config, ws, global_version=version).run(
        grid, manifest.tile_edge, manifest.overlap), config
