"""Deterministic local sample workspace generator."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .pngio import write_png
from .util import atomic_write_json


CLASS_NAMES = {0: "farmland", 1: "water", 2: "building", 3: "road"}


def _class_raster(width: int, height: int) -> np.ndarray:
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    labels = np.zeros((height, width), dtype=np.uint8)

    block_x = x % 26
    block_y = y % 26
    roads = (block_x < 3) | (block_y < 3)
    labels[roads] = 3

    building_cells = (
        (block_x >= 8)
        & (block_x < 21)
        & (block_y >= 8)
        & (block_y < 21)
        & (((x // 26 + y // 26) % 2) == 0)
    )
    labels[building_cells] = 2

    river_center = 0.55 * height + 7.0 * np.sin(x / 13.0) + 3.0 * np.sin(x / 5.0)
    water = np.abs(y - river_center) <= 4.5
    labels[water] = 1
    return labels


def _band_raster(width: int, height: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    labels = _class_raster(width, height)
    rng = np.random.default_rng(seed)

    base = np.array(
        [
            [66.0, 114.0, 184.0, 96.0],
            [24.0, 48.0, 18.0, 18.0],
            [148.0, 141.0, 151.0, 176.0],
            [116.0, 116.0, 119.0, 61.0],
        ],
        dtype=np.float64,
    )
    bands = base[labels].copy()
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

    texture = np.stack(
        [
            7.0 * np.sin(x / 4.1) + 4.0 * np.cos(y / 6.3),
            8.0 * np.sin((x + y) / 5.7),
            9.0 * np.cos((x - y) / 7.1),
            5.0 * np.sin(x / 8.0) + 6.0 * np.cos(y / 5.0),
        ],
        axis=2,
    )
    elevation_gradient = (x / max(1, width - 1)) * 22.0
    bands[:, :, 3] += elevation_gradient
    bands += texture
    bands += rng.normal(0.0, 3.5, size=bands.shape)
    return labels, np.clip(np.rint(bands), 0, 255).astype(np.uint8)


def _tile_path(tile_id: str) -> str:
    return f"tiles/{tile_id}.png"


def _build_neighbors(tiles: list[dict[str, Any]]) -> None:
    for tile in tiles:
        x0, y0, x1, y1 = tile["bounds"]
        neighbors: list[str] = []
        for other in tiles:
            if other is tile:
                continue
            ox0, oy0, ox1, oy1 = other["bounds"]
            overlap = min(x1, ox1) > max(x0, ox0) and min(y1, oy1) > max(y0, oy0)
            if not overlap:
                continue
            horizontal = tile["row"] == other["row"] and abs(tile["col"] - other["col"]) == 1
            vertical = tile["col"] == other["col"] and abs(tile["row"] - other["row"]) == 1
            if horizontal or vertical:
                neighbors.append(other["tile_id"])
        tile["neighbors"] = sorted(neighbors)


def generate_sample_workspace(
    output_dir: Path,
    *,
    width: int = 48,
    height: int = 48,
    rows: int = 2,
    cols: int = 3,
    overlap: int = 8,
    seed: int = 20261122,
    patch_batches: int = 2,
    include_extra_tile: bool = True,
) -> dict[str, Any]:
    if width <= 0 or height <= 0 or rows <= 0 or cols <= 0:
        raise ValueError("width, height, rows and cols must be positive")
    if overlap < 0 or overlap >= min(width, height):
        raise ValueError("overlap must satisfy 0 <= overlap < min(width, height)")
    if patch_batches < 0:
        raise ValueError("patch_batches must be non-negative")

    output_dir = Path(output_dir)
    (output_dir / "tiles").mkdir(parents=True, exist_ok=True)
    (output_dir / "pool").mkdir(parents=True, exist_ok=True)

    stride_x = width - overlap
    stride_y = height - overlap
    tiles: list[dict[str, Any]] = []
    for row in range(rows):
        for col in range(cols):
            x0 = col * stride_x
            y0 = row * stride_y
            tile_id = f"r{row}c{col}"
            tiles.append(
                {
                    "tile_id": tile_id,
                    "row": row,
                    "col": col,
                    "path": _tile_path(tile_id),
                    "bounds": [x0, y0, x0 + width, y0 + height],
                    "band_order_version": "bov-1",
                }
            )
    _build_neighbors(tiles)

    max_x = max(tile["bounds"][2] for tile in tiles)
    max_y = max(tile["bounds"][3] for tile in tiles)
    if include_extra_tile:
        extra_x = cols * stride_x
        extra_y = 0
        max_x = max(max_x, extra_x + width)
        max_y = max(max_y, extra_y + height)
    max_x = int(math.ceil(max_x))
    max_y = int(math.ceil(max_y))
    _, global_bands = _band_raster(max_x, max_y, seed)

    for tile in tiles:
        x0, y0, x1, y1 = tile["bounds"]
        write_png(output_dir / tile["path"], global_bands[y0:y1, x0:x1])

    if include_extra_tile:
        extra = {
            "tile_id": "r0c3",
            "row": 0,
            "col": cols,
            "path": _tile_path("r0c3"),
            "bounds": [cols * stride_x, 0, cols * stride_x + width, height],
            "band_order_version": "bov-1",
            "neighbors": [],
        }
        x0, y0, x1, y1 = extra["bounds"]
        write_png(output_dir / "pool" / "tile_r0c3.png", global_bands[y0:y1, x0:x1])

    index = {
        "schema_version": 1,
        "index_version": "idx-0001",
        "band_order_version": "bov-1",
        "bands": ["red", "green", "nir", "elevation"],
        "tile_size": [width, height],
        "tile_overlap": overlap,
        "global_bounds": [0, 0, max_x, max_y],
        "tiles": tiles,
    }
    fusion_versions = {
        "schema_version": 1,
        "band_order_versions": ["bov-1", "bov-2"],
        "effective_order": ["f1", "f2", "f3"],
        "strategies": {
            "f1": {
                "version": "f1",
                "order": 1,
                "weight_power": 1.0,
                "scale": 1.0,
                "tie_break": "row_col_lex",
                "description": "linear boundary-distance weighting",
            },
            "f2": {
                "version": "f2",
                "order": 2,
                "weight_power": 2.0,
                "scale": 1.0,
                "tie_break": "row_col_lex",
                "description": "quadratic boundary-distance weighting",
            },
            "f3": {
                "version": "f3",
                "order": 3,
                "weight_power": 0.5,
                "scale": 2.0,
                "tie_break": "row_col_lex",
                "description": "square-root boundary-distance weighting",
            },
        },
    }

    batches: list[dict[str, Any]] = []
    patch_targets = [
        ("r0c0", [10, 10, 21, 21], 1101),
        ("r1c1", [18, 16, 31, 29], 1202),
        ("r0c2", [5, 7, 19, 20], 1303),
        ("r1c0", [28, 19, 41, 39], 1404),
    ]
    for batch_index in range(patch_batches):
        operations = []
        for target_index in range(2):
            source = patch_targets[(batch_index * 2 + target_index) % len(patch_targets)]
            tile_id, rect, patch_seed = source
            row = int(tile_id[1])
            col = int(tile_id[3])
            operations.append(
                {
                    "tile_id": tile_id,
                    "row": row,
                    "col": col,
                    "rect": rect,
                    "seed": patch_seed + batch_index * 100,
                }
            )
        batches.append({"batch": batch_index + 1, "operations": operations})
    patch_plan = {"schema_version": 1, "batches": batches}

    atomic_write_json(output_dir / "index.json", index)
    atomic_write_json(output_dir / "fusion_versions.json", fusion_versions)
    atomic_write_json(output_dir / "patch_plan.json", patch_plan)
    return {
        "index": index,
        "fusion_versions": fusion_versions,
        "patch_plan": patch_plan,
        "tiles": len(tiles),
        "size": [width, height],
        "rows": rows,
        "cols": cols,
        "overlap": overlap,
        "seed": seed,
    }
