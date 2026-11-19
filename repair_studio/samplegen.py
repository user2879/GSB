from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_COMMANDS: list[dict[str, Any]] = [
    {"op": "set_repair_radius", "value": 4},
    {"op": "set_style_weights", "weights": [0.2, 0.7, 0.5]},
    {"op": "set_quant_levels", "value": 8},
    {"op": "set_mask_rect", "rect": {"x": 44, "y": 28, "width": 72, "height": 54}},
    {"op": "set_repair_radius", "value": 2},
    {"op": "set_style_weights", "weights": [0.6, 0.25, 0.8]},
    {"op": "set_quant_levels", "value": 32},
    {"op": "set_mask_rect", "rect": {"x": 36, "y": 20, "width": 82, "height": 62}},
    {"op": "set_preserve_mask", "value": False},
    {"op": "set_repair_radius", "value": 3},
]


def _procedural_image(width: int, height: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    rows = np.arange(height, dtype=np.int64)[:, None]
    columns = np.arange(width, dtype=np.int64)[None, :]
    red = (28 + (columns * 150) // max(1, width - 1) + (rows * 25) // max(1, height - 1)) % 256
    green = (35 + (rows * 170) // max(1, height - 1) + (columns * 20) // max(1, width - 1)) % 256
    blue = (65 + ((columns + rows) * 120) // max(1, width + height - 2)) % 256
    base = np.stack(
        [
            np.broadcast_to(red, (height, width)),
            np.broadcast_to(green, (height, width)),
            np.broadcast_to(blue, (height, width)),
        ],
        axis=2,
    )
    noise = generator.integers(-16, 17, size=(height, width, 3), dtype=np.int16)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def _style_image(width: int, height: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    rows = np.arange(height, dtype=np.int64)[:, None]
    columns = np.arange(width, dtype=np.int64)[None, :]
    wave = ((columns * 12 + rows * 7) // 9 + (columns * columns) // max(1, width * 2)) % 96
    red = 155 + wave
    green = 90 + (wave * 3) // 4
    blue = 45 + (wave * 2) // 3
    base = np.stack(
        [
            np.broadcast_to(red, (height, width)),
            np.broadcast_to(green, (height, width)),
            np.broadcast_to(blue, (height, width)),
        ],
        axis=2,
    )
    noise = generator.integers(-9, 10, size=(height, width, 3), dtype=np.int16)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def _damage_image(image: np.ndarray, rect: dict[str, int], seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    damaged = image.copy()
    x_value = rect["x"]
    y_value = rect["y"]
    width_value = rect["width"]
    height_value = rect["height"]
    defect = generator.integers(0, 256, size=(height_value, width_value, 3), dtype=np.uint8)
    damaged[y_value : y_value + height_value, x_value : x_value + width_value] = defect
    return damaged


def _write_png(path: Path, pixels: np.ndarray) -> None:
    Image.fromarray(pixels).save(path, format="PNG", compress_level=6, optimize=False)


def generate_sample(
    directory: Path,
    width: int = 192,
    height: int = 128,
    seed: int = 20261120,
    mask: dict[str, int] | None = None,
    command_overrides: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    if mask is None:
        mask = {"x": 48, "y": 32, "width": 64, "height": 48}
    clean = _procedural_image(width, height, seed)
    damaged = _damage_image(clean, mask, seed + 17)
    style = _style_image(width, height, seed + 101)
    _write_png(directory / "image.png", damaged)
    _write_png(directory / "style.png", style)

    chunk_rows = 8 if height >= 64 else max(1, height // 4)
    config = {
        "schema_version": 1,
        "seed": seed,
        "width": width,
        "height": height,
        "image_file": "image.png",
        "style_file": "style.png",
        "commands_file": "commands.json",
        "mask": mask,
        "repair": {"radius": 3, "iterations": 40 if height <= 160 else 80},
        "style": {
            "quant_levels": 16,
            "channel_weights": [0.35, 0.75, 0.55],
            "preserve_mask": True,
        },
        "interaction": {
            "latency_budget_ms": 30,
            "chunk_rows": chunk_rows,
            "thread_workers": 4,
            "snapshot_interval": 2,
        },
        "fixed_bits": 16,
    }
    (directory / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    commands = command_overrides if command_overrides is not None else list(DEFAULT_COMMANDS)
    (directory / "commands.json").write_text(
        json.dumps({"commands": commands}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"directory": str(directory), "width": width, "height": height, "seed": seed, "mask": mask}


def generate_default_samples(root: Path) -> list[dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=True)
    first = generate_sample(root / "procedural_a", width=192, height=128, seed=20261120)
    commands_second = list(DEFAULT_COMMANDS)
    commands_second[0] = {"op": "set_repair_radius", "value": 2}
    commands_second[-1] = {"op": "set_repair_radius", "value": 5}
    second = generate_sample(
        root / "procedural_b",
        width=160,
        height=96,
        seed=20261121,
        mask={"x": 30, "y": 18, "width": 70, "height": 52},
        command_overrides=commands_second,
    )
    return [first, second]
