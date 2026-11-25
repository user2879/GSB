"""Deterministic, dependency-light synthetic vertical layout generator."""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw

from .constants import (
    DEFAULT_STRATEGIES,
    FIXED_PIXEL_MODE,
    FIXED_RESOLUTION,
    QUANT_STEPS_ALLOWED,
    FIXED_BITS_ALLOWED,
)
from .utils import stable_hash, write_json


GLYPHS = {
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
}


@dataclass(frozen=True)
class ImageSpec:
    name: str
    columns: int
    chars_per_column: int
    noise_density: float
    frequency: float


DEFAULT_IMAGE_SPECS = (
    ImageSpec("small_layout_01.png", 2, 3, 0.0012, 5.0),
    ImageSpec("medium_layout_01.png", 3, 6, 0.0020, 4.0),
    ImageSpec("large_layout_01.png", 5, 10, 0.0030, 3.0),
)


def generate_samples(
    root: str | Path,
    *,
    seed: int = 20261126,
    specs: Iterable[ImageSpec] = DEFAULT_IMAGE_SPECS,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create PNGs, truth, two annotation versions, and a manifest."""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    images_dir = root / "images"
    truth_dir = root / "truth"
    versions_dir = root / "keypoints"
    for directory in (images_dir, truth_dir, versions_dir):
        if directory.exists():
            shutil.rmtree(directory)
    for directory in (images_dir, truth_dir, versions_dir / "v1", versions_dir / "v2"):
        directory.mkdir(parents=True, exist_ok=True)

    manifest_images: list[dict[str, Any]] = []
    for image_index, spec in enumerate(specs):
        rng = random.Random(seed + image_index * 104_729)
        image, truth = _render_image(spec, seed=seed, image_index=image_index, rng=rng)
        image_path = images_dir / spec.name
        image.save(image_path, format="PNG", optimize=False, compress_level=6)
        truth_path = truth_dir / f"{Path(spec.name).stem}.json"
        write_json(truth_path, truth)

        version_v1 = {
            "schema_version": 1,
            "version": "v1",
            "image": spec.name,
            "keypoints": [dict(point) for point in truth["keypoints"]],
        }
        version_v2 = {
            "schema_version": 1,
            "version": "v2",
            "image": spec.name,
            "keypoints": [_shifted_point(point, image_index) for point in truth["keypoints"]],
        }
        v1_path = versions_dir / "v1" / f"{Path(spec.name).stem}.json"
        v2_path = versions_dir / "v2" / f"{Path(spec.name).stem}.json"
        write_json(v1_path, version_v1)
        write_json(v2_path, version_v2)
        manifest_images.append(
            {
                "name": spec.name,
                "path": f"images/{spec.name}",
                "truth_path": f"truth/{Path(spec.name).stem}.json",
                "keypoint_versions": {
                    "v1": f"keypoints/v1/{Path(spec.name).stem}.json",
                    "v2": f"keypoints/v2/{Path(spec.name).stem}.json",
                },
                "group": "small" if spec.columns <= 2 else "medium" if spec.columns <= 3 else "large",
            }
        )

    manifest = {
        "schema_version": 1,
        "seed": seed,
        "resolution": {"width": FIXED_RESOLUTION[0], "height": FIXED_RESOLUTION[1]},
        "pixel_mode": FIXED_PIXEL_MODE,
        "images": manifest_images,
    }
    write_json(root / "manifest.json", manifest)
    if config_path is not None:
        write_default_config(config_path)
    return manifest


def write_default_config(path: str | Path) -> dict[str, Any]:
    config = {
        "schema_version": 1,
        "resolution": {"width": FIXED_RESOLUTION[0], "height": FIXED_RESOLUTION[1]},
        "pixel_mode": FIXED_PIXEL_MODE,
        "tier": "auto",
        "window": {"width": 64, "height": 64, "step_x": 32, "step_y": 32},
        "cc": {
            "min_area": 10,
            "max_aspect": 3.2,
            "merge_gap": 2,
            "split_ratio": 1.9,
            "ink_threshold": 192,
        },
        "precision": {"fixed_bits": 8, "quant_step": 2},
        "max_workers": 4,
        "point_budget": 96,
        "keypoint_version": "v1",
        "strategies": DEFAULT_STRATEGIES,
        "fixed_bits_allowed": list(FIXED_BITS_ALLOWED),
        "quant_steps_allowed": list(QUANT_STEPS_ALLOWED),
    }
    write_json(path, config)
    return config


def _render_image(
    spec: ImageSpec,
    *,
    seed: int,
    image_index: int,
    rng: random.Random,
) -> tuple[Image.Image, dict[str, Any]]:
    width, height = FIXED_RESOLUTION
    canvas = np.full((height, width), 238, dtype=np.uint8)
    image_rng = random.Random(rng.randrange(0, 2**31))

    column_width = 76
    column_gap = 22
    right_margin = 54
    top_margin = 96
    vertical_spacing = 64 if spec.chars_per_column <= 8 else 60
    reading_index = 0
    columns: list[dict[str, Any]] = []
    keypoints: list[dict[str, Any]] = []

    for column_index in range(spec.columns):
        right = width - right_margin - column_index * (column_width + column_gap)
        left = right - column_width
        center_x = (left + right - 1) // 2
        column_start_reading = reading_index
        char_centers: list[tuple[int, int]] = []
        for char_index in range(spec.chars_per_column):
            center_y = top_margin + char_index * vertical_spacing
            glyph = str((column_index * 3 + char_index + image_index) % 10)
            angle = image_rng.uniform(-spec.frequency, spec.frequency)
            jitter_x = image_rng.randint(-2, 2)
            jitter_y = image_rng.randint(-2, 2)
            actual_x = center_x + jitter_x
            actual_y = center_y + jitter_y
            _draw_glyph(
                canvas,
                glyph,
                actual_x,
                actual_y,
                angle,
                image_rng,
                occlude=(spec.chars_per_column >= 6 and char_index % 4 == 2),
            )
            char_centers.append((actual_x, actual_y))
            keypoints.append(
                {
                    "id": f"k{reading_index:06d}",
                    "column": f"c{column_index}",
                    "x": int(actual_x),
                    "y": int(actual_y),
                    "reading_index": reading_index,
                    "char": f"G{glyph}",
                }
            )
            reading_index += 1
        y_values = [center[1] for center in char_centers]
        columns.append(
            {
                "id": f"c{column_index}",
                "x0": int(left),
                "x1": int(right),
                "y0": int(max(0, min(y_values) - 30)),
                "y1": int(min(height, max(y_values) + 30)),
                "char_count": spec.chars_per_column,
                "start_reading_index": column_start_reading,
                "order": column_index,
            }
        )

    _add_noise(canvas, image_rng, spec.noise_density)
    image = Image.fromarray(canvas, mode=FIXED_PIXEL_MODE)
    truth = {
        "schema_version": 1,
        "image": spec.name,
        "resolution": {"width": width, "height": height},
        "seed": stable_hash([seed, image_index, spec.name], length=16),
        "columns": columns,
        "keypoints": keypoints,
    }
    return image, truth


def _draw_glyph(
    canvas: np.ndarray,
    glyph: str,
    center_x: int,
    center_y: int,
    angle: float,
    rng: random.Random,
    *,
    occlude: bool,
) -> None:
    scale = 4
    pattern = GLYPHS[glyph]
    tile = Image.new("L", (44, 52), 238)
    draw = ImageDraw.Draw(tile)
    origin_x, origin_y = 8, 10
    gray = rng.randint(12, 55)
    for row_index, row in enumerate(pattern):
        for column_index, bit in enumerate(row):
            if bit != "1":
                continue
            x = origin_x + column_index * scale + rng.randint(-1, 1)
            y = origin_y + row_index * scale + rng.randint(-1, 1)
            line_width = 4 if rng.random() > 0.18 else 3
            draw.rectangle((x, y, x + line_width - 1, y + line_width - 1), fill=gray)
    rotated = tile.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=238)
    tile_array = np.asarray(rotated)
    tile_mask = tile_array < 192
    target_y0 = center_y - 26
    target_x0 = center_x - 22
    height, width = canvas.shape
    ys, xs = np.nonzero(tile_mask)
    ys = ys + target_y0
    xs = xs + target_x0
    valid = (ys >= 0) & (ys < height) & (xs >= 0) & (xs < width)
    canvas[ys[valid], xs[valid]] = tile_array[tile_mask][valid]

    if occlude:
        occluder_y = center_y + (1 if center_y % 2 else 0)
        x0 = max(0, center_x - 14)
        x1 = min(width - 1, center_x + 14)
        canvas[max(0, occluder_y) : min(height, occluder_y + 2), x0 : x1 + 1] = 238


def _add_noise(canvas: np.ndarray, rng: random.Random, density: float) -> None:
    height, width = canvas.shape
    count = int(width * height * density)
    for _ in range(count):
        x = rng.randrange(0, width)
        y = rng.randrange(0, height)
        canvas[y, x] = rng.randint(40, 170)
    for _ in range(max(4, count // 180)):
        x = rng.randrange(1, width - 2)
        y = rng.randrange(1, height - 2)
        canvas[y : y + 2, x : x + 2] = rng.randint(55, 175)


def _shifted_point(point: dict[str, Any], image_index: int) -> dict[str, Any]:
    delta_x = 1 if image_index % 2 == 0 else -1
    delta_y = 1 if point["reading_index"] % 2 == 0 else -1
    return {
        **point,
        "x": point["x"] + delta_x,
        "y": point["y"] + delta_y,
    }
