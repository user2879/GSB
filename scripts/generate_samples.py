#!/usr/bin/env python3
"""Generate the deterministic local sample set.

The generator uses a fixed-seed numpy Generator and pure integer pixel values.
It creates a grid of four-band RGBA PNG tiles (red, green, nir, elevation), a
tile index JSON with spatial bounds and symmetric adjacency, a segmentation
config JSON, an injection configuration and an edit-sequence script input.

Usage:
    python scripts/generate_samples.py --out data/sample_set
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

SEED = 20261120
TILE_SIZE = 32
BAND_COUNT = 4
GRID_ROWS = 3
GRID_COLS = 3
OVERLAP = 8
CLASS_RATIOS = {"farmland": 0.40, "water": 0.22, "building": 0.20, "road": 0.18}

# Integer band palettes per ground-truth class (R, G, NIR, elevation).
PALETTE = {
    1: (70, 110, 200, 90),    # farmland: high NIR -> high NDVI
    2: (30, 120, 20, 60),     # water: green >> NIR -> strong positive NDWI
    3: (170, 150, 140, 210),  # building: bright, high elevation
    4: (130, 105, 80, 120),   # road: NIR < green slightly, low NDVI
}


def _class_raster(rng: np.random.Generator, size: int) -> np.ndarray:
    ratios = CLASS_RATIOS
    cumulative = np.cumsum(list(ratios.values()))
    draws = rng.random((size, size))
    classes = np.ones((size, size), dtype=np.uint8)
    for idx, threshold in enumerate(cumulative[:-1], start=2):
        classes[draws > threshold] = idx
    # Add deterministic road-like lines so connected components are non-trivial.
    classes[8, :] = 4
    classes[:, 24] = 4
    return classes


def _bands_for_tile(row: int, col: int) -> np.ndarray:
    rng = np.random.default_rng(SEED + row * 1009 + col * 9176)
    classes = _class_raster(rng, TILE_SIZE)
    bands = np.zeros((TILE_SIZE, TILE_SIZE, BAND_COUNT), dtype=np.uint8)
    for code, palette in PALETTE.items():
        mask = classes == code
        for band, base in enumerate(palette):
            noise = rng.integers(-7, 8, size=(TILE_SIZE, TILE_SIZE), dtype=np.int16)
            values = np.clip(base + noise, 0, 255).astype(np.uint8)
            bands[:, :, band][mask] = values[mask]
    # Deterministic elevation ridge to create slope structure.
    ridge = np.clip(40 + 6 * np.sin(np.arange(TILE_SIZE) / 3.0), 0, 255).astype(np.uint8)
    bands[:, :, 3] = np.maximum(bands[:, :, 3], ridge[:, None])
    return bands


def write_samples(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir = out_dir / "tiles"
    tiles_dir.mkdir(exist_ok=True)
    step = TILE_SIZE - OVERLAP
    index_tiles = []
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            bands = _bands_for_tile(row, col)
            file_name = f"tiles/tile_r{row:03d}_c{col:03d}.png"
            Image.fromarray(bands, mode="RGBA").save(out_dir / file_name)
            x0, y0 = col * step, row * step
            neighbors = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = row + dr, col + dc
                if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS:
                    neighbors.append([nr, nc])
            index_tiles.append({
                "row": row,
                "col": col,
                "file": file_name,
                "bounds": [x0, y0, x0 + TILE_SIZE, y0 + TILE_SIZE],
                "neighbors": neighbors,
            })

    width = (GRID_COLS - 1) * step + TILE_SIZE
    height = (GRID_ROWS - 1) * step + TILE_SIZE
    index = {
        "band_order_version": "BO-V1",
        "band_order": ["red", "green", "nir", "elevation"],
        "tile_size": TILE_SIZE,
        "bands": BAND_COUNT,
        "global_bounds": [0, 0, width, height],
        "tiles": index_tiles,
    }
    with open(out_dir / "tile_index.json", "w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)

    config = {
        "window_size": 14,
        "stride": 10,
        "context_overlap": 2,
        "feature_thresholds": {
            "ndvi": 0.12,
            "ndwi": 0.08,
            "edge": 18.0,
            "slope": 6.0,
        },
        "class_weights": {
            "farmland": {"ndvi": 4.0, "ndwi": -2.0, "edge": -1.0, "slope": -1.0, "bias": 200.0},
            "water": {"ndvi": -3.0, "ndwi": 4.0, "edge": -1.0, "slope": 0.0, "bias": 100.0},
            "building": {"ndvi": -2.0, "ndwi": -2.0, "edge": 2.0, "slope": 3.0, "bias": 120.0},
            "road": {"ndvi": -3.0, "ndwi": -3.0, "edge": 3.0, "slope": 1.0, "bias": 60.0},
        },
        "min_component_area": 20,
        "fusion_strategy": "distance_weighted",
        "recompute_budget": 12,
        "workers": 3,
        "snapshot_commit_interval": 2,
    }
    with open(out_dir / "segmentation_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)

    injections = {
        "injections": [
            {"id": "INJ-001", "kind": "truncated_png", "target": [0, 1],
             "payload": {"keep_ratio": 0.4}},
            {"id": "INJ-002", "kind": "non_finite_band", "target": [1, 0], "payload": {}},
            {"id": "INJ-003", "kind": "asymmetric_adjacency", "target": [2, 2], "payload": {}},
            {"id": "INJ-004", "kind": "zero_budget", "payload": {}},
        ]
    }
    with open(out_dir / "injections.json", "w", encoding="utf-8") as handle:
        json.dump(injections, handle, indent=2, sort_keys=True)

    edit_sequence = {
        "description": "Deterministic edit sequence applied in order by the test harness.",
        "steps": [
            {"id": "edit-1-threshold", "type": "config",
             "patch": {"feature_thresholds": {"ndvi": 0.2}}},
            {"id": "edit-2-weights", "type": "config",
             "patch": {"class_weights": {"building": {"edge": 3.0}}}},
            {"id": "edit-3-adjacency", "type": "adjacency",
             "remove": [[0, 0], [0, 1]]},
            {"id": "edit-4-band-patch", "type": "band_patch", "tile": [1, 1],
             "band": 2, "seed_offset": 123},
            {"id": "edit-5-restore-adjacency", "type": "restore_adjacency",
             "edge": [[0, 0], [0, 1]]},
            {"id": "edit-6-threshold-again", "type": "config",
             "patch": {"feature_thresholds": {"ndwi": 0.15}}},
        ],
    }
    with open(out_dir / "edit_sequence.json", "w", encoding="utf-8") as handle:
        json.dump(edit_sequence, handle, indent=2, sort_keys=True)

    return {"index": index, "config": config, "out_dir": str(out_dir)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/sample_set")
    args = parser.parse_args()
    result = write_samples(Path(args.out))
    print(f"generated {len(result['index']['tiles'])} tiles in {args.out}")


if __name__ == "__main__":
    main()
