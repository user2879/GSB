"""Deterministic local sample generator (no network or external data)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _smooth_field(rng: np.random.Generator, size: int, grid: int) -> np.ndarray:
    low = rng.normal(0.0, 1.0, size=(grid, grid, 3))
    coordinates = np.linspace(0.0, grid - 1, size, dtype=np.float64)
    left = np.floor(coordinates).astype(np.int64)
    right = np.minimum(left + 1, grid - 1)
    frac = coordinates - left
    x_left = low[:, left, :]
    x_right = low[:, right, :]
    horizontal = x_left * (1.0 - frac)[None, :, None] + x_right * frac[None, :, None]
    y_left = horizontal[left, :, :]
    y_right = horizontal[right, :, :]
    return y_left * (1.0 - frac)[:, None, None] + y_right * frac[:, None, None]


def _source_image(seed: int, size: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    phase = rng.uniform(-np.pi, np.pi, size=3)
    image = np.zeros((size, size, 3), dtype=np.float64)
    for channel in range(3):
        image[..., channel] = (
            92.0
            + 22.0 * np.sin(2.0 * np.pi * x / (8.0 + 2.0 * channel) + phase[channel])
            + 15.0 * np.cos(2.0 * np.pi * y / (19.0 - channel) - phase[channel])
            + 9.0 * np.sin(2.0 * np.pi * (x + 2.0 * y) / (43.0 + 3.0 * channel))
        )
    image += 15.0 * _smooth_field(rng, size, 8)
    image += 7.0 * _smooth_field(rng, size, 16)
    image += rng.normal(0.0, 3.0, size=(size, size, 3))
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def _style_image(seed: int, size: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    image = np.empty((size, size, 3), dtype=np.float64)
    image[..., 0] = 40.0 + 190.0 * (x / max(1, size - 1)) + 35.0 * np.sin(y / 3.0)
    image[..., 1] = 45.0 + 175.0 * (y / max(1, size - 1)) + 45.0 * np.cos(x / 5.0)
    image[..., 2] = 80.0 + 120.0 * (((x // 4) + (y // 4)) % 2) + 30.0 * np.sin((x + y) / 7.0)
    image += 10.0 * _smooth_field(rng, size, 6)
    image += rng.normal(0.0, 2.0, size=(size, size, 3))
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def _rect_levels(base_rect: list[int], factors: list[int]) -> list[list[int]]:
    x, y, width, height = base_rect
    cumulative = 1
    result: list[list[int]] = []
    for factor in factors:
        cumulative *= factor
        scaled = [
            x // cumulative,
            y // cumulative,
            max(1, (width + cumulative - 1) // cumulative),
            max(1, (height + cumulative - 1) // cumulative),
        ]
        result.append(scaled)
    return result


def generate_sample(
    output_dir: str | Path,
    *,
    seed: int = 20261123,
    size: int = 128,
    levels: int = 4,
    tile_size: int = 16,
    tile_budget: int = 8,
    threads: int = 4,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    factors = [1] + [2] * (levels - 1)
    cumulative = 1
    sizes: list[tuple[int, int]] = []
    for factor in factors:
        cumulative *= factor
        if size % cumulative:
            raise ValueError("sample size is incompatible with pyramid factors")
        sizes.append((size // cumulative, size // cumulative))
    for width, height in sizes:
        if width % tile_size or height % tile_size:
            raise ValueError("sample level size is not divisible by tile size")
    if min(width for width, _ in sizes) < tile_size:
        raise ValueError("coarsest sample level is smaller than one tile")

    base_rng = np.random.default_rng(seed)
    style = _style_image(seed + 9001, size)
    Image.fromarray(style, mode="RGB").save(output_dir / "style_ref.png", format="PNG")

    source_names: list[str] = []
    defect = [int(size * 0.31), int(size * 0.29), max(8, size // 4), max(8, size // 5)]
    dx, dy, dw, dh = defect
    for index in range(3):
        image = _source_image(seed + index * 101, size)
        corruption_rng = np.random.default_rng(seed + 500 + index)
        noise = corruption_rng.integers(-6, 7, size=(dh, dw, 3), dtype=np.int16)
        image[dy : dy + dh, dx : dx + dw] = np.clip(
            118 + noise, 0, 255
        ).astype(np.uint8)
        name = f"image_{index + 1:02d}.png"
        Image.fromarray(image, mode="RGB").save(output_dir / name, format="PNG")
        source_names.append(name)

    level_rects = _rect_levels(defect, factors)
    mask_entries: list[dict[str, Any]] = []
    for level, rect in enumerate(level_rects):
        mask_entries.append({"layer": level, "rects": [rect]})

    config = {
        "version": 1,
        "image": {"source": source_names[0], "style": "style_ref.png"},
        "pyramid": {
            "levels": levels,
            "downsample_factors": factors,
            "tile_size": tile_size,
            "tile_order": "row_major",
        },
        "masks": mask_entries,
        "repair": {
            "iterations_by_layer": [2 + level for level in range(levels)],
            "guide_strength_by_layer": [
                round(0.12 + 0.18 * level, 4) for level in range(levels)
            ],
        },
        "retention": {
            "mask_fill": "diffuse",
            "style_inside": "mapped",
            "style_outside": "mapped",
        },
        "style": {
            "quant_levels_by_layer": [min(256, 6 + 4 * level) for level in range(levels)],
            "weights_by_layer": [
                [round(0.08 + 0.025 * level, 4), round(0.06 + 0.02 * level, 4), round(0.04 + 0.015 * level, 4)]
                for level in range(levels)
            ],
            "composition_weights": [1.0 / (level + 1) for level in range(levels)],
        },
        "fixed_point": {"fraction_bits": 12},
        "scheduler": {"threads": threads, "recompute_tile_budget": tile_budget},
        "snapshot": {"commit_every_commands": 1},
    }

    draft = base_rng.integers(0, max(1, size // 4), size=4, dtype=np.int64).tolist()
    edit_layer = max(0, levels - 2)
    source_layer_size = sizes[edit_layer][0]
    tile_edit_rect = [
        min(source_layer_size - 4, max(1, draft[0])),
        min(source_layer_size - 4, max(1, draft[1])),
        min(4, source_layer_size),
        min(3, source_layer_size),
    ]
    coarse_rect = level_rects[-1]
    commands: list[dict[str, Any]] = [
        {
            "id": "edit-mask-local",
            "type": "set_mask_rect",
            "layer": edit_layer,
            "rect": tile_edit_rect,
        },
        {
            "id": "edit-iterations",
            "type": "set_iterations",
            "layer": min(1, levels - 1),
            "value": 5,
        },
        {
            "id": "edit-guide",
            "type": "set_guide_strength",
            "layer": min(2, levels - 1),
            "value": 0.42,
        },
        {
            "id": "invalid-quant-skipped",
            "type": "set_quant_levels",
            "layer": min(1, levels - 1),
            "levels": 1,
        },
        {
            "id": "edit-style-weight",
            "type": "set_style_weight",
            "layer": levels - 1,
            "weights": [0.18, 0.10, 0.06],
        },
        {
            "id": "edit-quant",
            "type": "set_quant_levels",
            "layer": min(1, levels - 1),
            "levels": 7,
        },
        {
            "id": "restore-coarse-mask",
            "type": "set_mask_rect",
            "layer": levels - 1,
            "rect": coarse_rect,
        },
    ]
    for index in range(8):
        commands.append({"id": f"drain-{index + 1:02d}", "type": "noop"})

    config_path = output_dir / "config.json"
    commands_path = output_dir / "commands.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    commands_path.write_text(json.dumps(commands, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "seed": seed,
        "size": [size, size],
        "levels": levels,
        "factors": factors,
        "level_sizes": [list(value) for value in sizes],
        "tile_size": tile_size,
        "sources": source_names,
        "style": "style_ref.png",
        "defect": defect,
        "config": config_path.name,
        "commands": commands_path.name,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest

