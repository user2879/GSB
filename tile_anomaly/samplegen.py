"""Deterministic local tile sample and configuration generation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .util import sha256_json, write_json


CLASS_FARMLAND = 0
CLASS_WATER = 1
CLASS_BUILDING = 2
CLASS_ROAD = 3

CLASS_NAMES = {
    CLASS_FARMLAND: "farmland",
    CLASS_WATER: "water",
    CLASS_BUILDING: "building",
    CLASS_ROAD: "road",
}


def _ellipse_mask(rows: np.ndarray, cols: np.ndarray, center_row: int, center_col: int, radius_row: int, radius_col: int) -> np.ndarray:
    return (
        ((rows - center_row) / radius_row) ** 2
        + ((cols - center_col) / radius_col) ** 2
        <= 1.0
    )


def _tile_fields(
    tile_row: int,
    tile_col: int,
    tile_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    tile_number = tile_row * 1000 + tile_col
    generator = np.random.default_rng(seed + tile_number)
    rows, cols = np.mgrid[0:tile_size, 0:tile_size]
    class_map = np.full((tile_size, tile_size), CLASS_FARMLAND, dtype=np.int16)

    water_row = int(generator.integers(tile_size // 4, 3 * tile_size // 4))
    water_col = int(generator.integers(tile_size // 4, 3 * tile_size // 4))
    water_radius_row = int(generator.integers(tile_size // 8, tile_size // 4))
    water_radius_col = int(generator.integers(tile_size // 8, tile_size // 4))
    water_mask = _ellipse_mask(rows, cols, water_row, water_col, water_radius_row, water_radius_col)
    class_map[water_mask] = CLASS_WATER

    for _ in range(4):
        building_row = int(generator.integers(5, tile_size - 12))
        building_col = int(generator.integers(5, tile_size - 12))
        height = int(generator.integers(4, 12))
        width = int(generator.integers(5, 14))
        building_mask = (
            (rows >= building_row)
            & (rows < min(tile_size, building_row + height))
            & (cols >= building_col)
            & (cols < min(tile_size, building_col + width))
            & (class_map != CLASS_WATER)
        )
        class_map[building_mask] = CLASS_BUILDING

    road_column = int(generator.integers(tile_size // 5, 4 * tile_size // 5))
    road_width = int(generator.integers(3, 6))
    vertical_road = np.abs(cols - road_column) <= road_width // 2
    road_row = int(generator.integers(tile_size // 5, 4 * tile_size // 5))
    horizontal_road = np.abs(rows - road_row) <= road_width // 2
    road_mask = (vertical_road | horizontal_road) & (class_map != CLASS_WATER)
    class_map[road_mask] = CLASS_ROAD

    global_rows = tile_row * tile_size + rows
    global_cols = tile_col * tile_size + cols
    elevation = np.full((tile_size, tile_size), 0.32, dtype=np.float64)
    elevation += 0.025 * np.sin(global_rows / 10.0) + 0.02 * np.cos(global_cols / 13.0)
    elevation[class_map == CLASS_WATER] = 0.045
    elevation[class_map == CLASS_BUILDING] = 0.68
    elevation[class_map == CLASS_ROAD] = 0.27
    elevation += 0.004 * np.sin((global_rows + global_cols) / 3.7)
    return class_map, elevation


def _render_tile(tile_row: int, tile_col: int, tile_size: int, seed: int) -> dict[str, np.ndarray]:
    class_map, elevation = _tile_fields(tile_row, tile_col, tile_size, seed)
    generator = np.random.default_rng(seed + tile_row * 7919 + tile_col * 104729)
    rows, cols = np.mgrid[0:tile_size, 0:tile_size]
    global_rows = tile_row * tile_size + rows
    global_cols = tile_col * tile_size + cols

    red = np.full((tile_size, tile_size), 96.0)
    green = np.full((tile_size, tile_size), 142.0)
    blue = np.full((tile_size, tile_size), 70.0)
    noise_scale = np.full((tile_size, tile_size), 3.5)

    farmland_texture = 5.0 * np.sin((global_rows * 1.7 + global_cols * 0.7))
    red += farmland_texture
    green += farmland_texture
    blue += farmland_texture * 0.35

    water = class_map == CLASS_WATER
    red[water] = 26.0
    green[water] = 70.0
    blue[water] = 146.0
    noise_scale[water] = 1.2

    building = class_map == CLASS_BUILDING
    red[building] = 188.0
    green[building] = 156.0
    blue[building] = 118.0
    noise_scale[building] = 2.0
    building_shadow = building & (((global_rows + global_cols) % 9) < 2)
    red[building_shadow] -= 38.0
    green[building_shadow] -= 32.0
    blue[building_shadow] -= 24.0

    road = class_map == CLASS_ROAD
    red[road] = 112.0
    green[road] = 112.0
    blue[road] = 108.0
    noise_scale[road] = 1.8
    road_mark = road & ((global_cols % 18) < 2)
    red[road_mark] = 218.0
    green[road_mark] = 210.0
    blue[road_mark] = 172.0

    illumination = 7.0 * np.sin(global_cols / 31.0) + 5.0 * np.cos(global_rows / 27.0)
    red += illumination + generator.normal(0.0, noise_scale)
    green += illumination + generator.normal(0.0, noise_scale)
    blue += illumination * 0.8 + generator.normal(0.0, noise_scale)

    elevation += generator.normal(0.0, 0.003, size=(tile_size, tile_size))
    elevation = np.clip(elevation, 0.0, 1.0)
    return {
        "red": np.clip(np.rint(red), 0, 255).astype(np.uint8),
        "green": np.clip(np.rint(green), 0, 255).astype(np.uint8),
        "blue": np.clip(np.rint(blue), 0, 255).astype(np.uint8),
        "elevation": np.rint(elevation * 65535.0).astype(np.uint16),
        "class": class_map.astype(np.uint8),
    }


def _apply_anomaly_injections(
    layers_by_tile: dict[str, dict[str, np.ndarray]],
    injections: list[dict[str, Any]],
) -> None:
    for injection in injections:
        tile_id = injection["tile_id"]
        layers = layers_by_tile[tile_id]
        row_start, col_start, height, width = injection["rect"]
        patch = np.s_[row_start : row_start + height, col_start : col_start + width]
        for band, offset in injection.get("band_offsets", {}).items():
            values = layers[band].astype(np.int64) + int(offset)
            maximum = 65535 if band == "elevation" else 255
            layers[band] = np.clip(values, 0, maximum).astype(layers[band].dtype)


def _save_tile(tile_dir: Path, layers: dict[str, np.ndarray]) -> None:
    tile_dir.mkdir(parents=True, exist_ok=True)
    for band in ("red", "green", "blue"):
        Image.fromarray(layers[band], mode="L").save(tile_dir / f"{band}.png", optimize=False)
    Image.fromarray(layers["elevation"], mode="I;16").save(
        tile_dir / "elevation.png", optimize=False
    )


def default_config(sample_root: Path, seed: int, worker_count: int) -> dict[str, Any]:
    del seed
    return {
        "schema_version": 1,
        "tile": {
            "size": 64,
            "bands": ["red", "green", "blue", "elevation"],
            "band_set_versions": ["bsv-2026-01", "bsv-2026-02"],
        },
        "paths": {
            "manifest": "../input/manifest.json",
            "tiles_dir": "../input/tiles",
            "state_dir": "../artifacts/state",
            "output_dir": "../artifacts/output",
        },
        "window": {"size": 16, "step": 8, "context_overlap": 4},
        "quantization": {
            "fraction_bits": 12,
            "gear": 2,
            "allowed_gears": [1, 2, 3, 4],
            "allowed_fraction_bits": [8, 10, 12, 14],
            "dimension_scales_milli": {
                "vegetation_index": 1000,
                "normalized_water_index": 1000,
                "texture_gradient": 1000,
                "elevation_slope": 1000,
            },
        },
        "detector": {
            "threshold_permille": 530,
            "kde_bandwidth_milli": 1400,
            "distance_weight_milli": 600,
            "density_weight_milli": 400,
            "final_score_step_micro": 1000,
            "minimum_robust_scale_micro": 1000000,
        },
        "fusion": {
            "strategy_version": "fusion-distance-score-v1",
            "overlap_width": 8,
            "min_component_pixels": 6,
        },
        "scheduler": {
            "max_workers": max(1, min(worker_count, 8)),
            "global_window_budget": 4096,
            "tile_window_quota": 128,
            "fusion_recompute_limit": 32,
        },
        "sample_root_label": str(sample_root.name),
    }


def generate_sample(output_dir: Path, seed: int = 20261126, overwrite: bool = True) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    input_dir = output_dir / "input"
    config_dir = output_dir / "config"
    tiles_dir = input_dir / "tiles"
    config_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir.mkdir(parents=True, exist_ok=True)

    tile_size = 64
    tile_rows = 3
    tile_cols = 3
    anomaly_injections = [
        {
            "tile_id": "tile_r0c2",
            "rect": [18, 20, 18, 18],
            "band_offsets": {"red": 58, "green": -35, "blue": -25, "elevation": 18000},
            "reason": "deterministic_high_reflectance_building_variant",
        },
        {
            "tile_id": "tile_r2c0",
            "rect": [36, 8, 14, 20],
            "band_offsets": {"red": -50, "green": 62, "blue": -18, "elevation": 9000},
            "reason": "deterministic_vegetation_stress_patch",
        },
    ]

    all_layers: dict[str, dict[str, np.ndarray]] = {}
    tiles: list[dict[str, Any]] = []
    for tile_row in range(tile_rows):
        for tile_col in range(tile_cols):
            tile_id = f"tile_r{tile_row}c{tile_col}"
            layers = _render_tile(tile_row, tile_col, tile_size, seed)
            all_layers[tile_id] = layers
            tiles.append(
                {
                    "id": tile_id,
                    "row": tile_row,
                    "col": tile_col,
                    "size": tile_size,
                    "band_set_version": "bsv-2026-01",
                    "bands": {
                        "red": f"{tile_id}/red.png",
                        "green": f"{tile_id}/green.png",
                        "blue": f"{tile_id}/blue.png",
                        "elevation": f"{tile_id}/elevation.png",
                    },
                    "neighbors": [],
                }
            )

    _apply_anomaly_injections(all_layers, anomaly_injections)
    for tile in tiles:
        tile_id = tile["id"]
        _save_tile(tiles_dir / tile_id, all_layers[tile_id])
        class_path = input_dir / "classes" / f"{tile_id}.png"
        class_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(all_layers[tile_id]["class"], mode="L").save(class_path, optimize=False)

    by_position = {(tile["row"], tile["col"]): tile for tile in tiles}
    for tile in tiles:
        row = tile["row"]
        col = tile["col"]
        neighbor_ids = []
        for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = by_position.get((row + delta_row, col + delta_col))
            if neighbor is not None:
                neighbor_ids.append(neighbor["id"])
        tile["neighbors"] = sorted(neighbor_ids)

    manifest = {
        "schema_version": 1,
        "seed": seed,
        "grid": {"rows": tile_rows, "cols": tile_cols, "tile_size": tile_size},
        "tiles": tiles,
    }
    write_json(input_dir / "manifest.json", manifest)
    write_json(input_dir / "anomaly_injections.json", {"injections": anomaly_injections})

    config = default_config(output_dir, seed, worker_count=8)
    (config_dir / "detection.yaml").write_text(
        json.dumps(config, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    fault_injection = {
        "schema_version": 1,
        "tile_faults": {
            "tile_r0c1": {
                "kind": "nonfinite_band",
                "band": "elevation",
                "row": 7,
                "col": 9,
                "value": "nan",
            },
            "tile_r1c0": {
                "kind": "self_neighbor",
                "neighbor": "tile_r1c0",
            },
            "tile_r2c1": {
                "kind": "window_not_divisible",
                "window_size": 15,
            },
            "tile_r2c2": {
                "kind": "quant_gear_out_of_range",
                "gear": 99,
            },
        },
    }
    write_json(config_dir / "fault_injection.json", fault_injection)

    edit_commands = {
        "schema_version": 1,
        "commands": [
            {
                "op": "replace_band",
                "tile_id": "tile_r1c2",
                "band": "elevation",
                "method": "offset",
                "value": 9000,
            },
            {
                "op": "replace_band",
                "tile_id": "tile_r1c2",
                "band": "red",
                "method": "patch_add",
                "rect": [22, 26, 14, 14],
                "value": 42,
            },
            {
                "op": "set_config",
                "path": "detector.threshold_permille",
                "value": 600,
            },
        ],
    }
    write_json(input_dir / "edit_commands.json", edit_commands)

    generation_summary = {
        "seed": seed,
        "tile_count": len(tiles),
        "tile_size": tile_size,
        "layers_per_tile": 4,
        "classes": list(CLASS_NAMES.values()),
        "config_hash": sha256_json(config),
        "manifest_hash": sha256_json(manifest),
        "sample_root": str(output_dir),
    }
    write_json(output_dir / "generation_summary.json", generation_summary)
    return generation_summary


def sample_class_map(tile_row: int, tile_col: int, tile_size: int, seed: int) -> np.ndarray:
    class_map, _ = _tile_fields(tile_row, tile_col, tile_size, seed)
    return class_map


def class_frequency(layers: dict[str, np.ndarray]) -> dict[str, float]:
    class_map = layers["class"]
    total = float(class_map.size)
    return {
        name: float(np.count_nonzero(class_map == class_id)) / total
        for class_id, name in CLASS_NAMES.items()
    }


def sampled_class_frequencies(class_map: np.ndarray) -> dict[str, float]:
    total = float(class_map.size)
    return {
        name: float(np.count_nonzero(class_map == class_id)) / total
        for class_id, name in CLASS_NAMES.items()
    }
