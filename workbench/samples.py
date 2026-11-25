from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .image_io import write_png


SAMPLE_WIDTH = 128
SAMPLE_HEIGHT = 96


def _rects_for_index(index: int) -> list[dict[str, Any]]:
    rect_sets = (
        [
            {"id": "defect_alpha", "x": 18, "y": 16, "width": 24, "height": 15},
            {"id": "defect_beta", "x": 82, "y": 50, "width": 23, "height": 17},
            {"id": "defect_gamma", "x": 52, "y": 76, "width": 20, "height": 12},
        ],
        [
            {"id": "defect_alpha", "x": 28, "y": 20, "width": 22, "height": 14},
            {"id": "defect_beta", "x": 76, "y": 54, "width": 22, "height": 16},
            {"id": "defect_gamma", "x": 58, "y": 78, "width": 18, "height": 11},
        ],
        [
            {"id": "defect_alpha", "x": 14, "y": 24, "width": 23, "height": 14},
            {"id": "defect_beta", "x": 88, "y": 56, "width": 20, "height": 15},
            {"id": "defect_gamma", "x": 46, "y": 79, "width": 19, "height": 11},
        ],
    )
    return [dict(item) for item in rect_sets[index % len(rect_sets)]]


def generate_sample(index: int, seed: int, image_path: Path, mask_path: Path) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    width, height = SAMPLE_WIDTH, SAMPLE_HEIGHT

    yy, xx = np.mgrid[0:height, 0:width]
    base = np.empty((height, width, 3), dtype=np.uint8)
    base[..., 0] = np.clip(42 + xx * 0.72 + yy * 0.10, 0, 255).astype(np.uint8)
    base[..., 1] = np.clip(74 + yy * 0.82 + np.sin(xx / 9.0) * 18, 0, 255).astype(np.uint8)
    base[..., 2] = np.clip(118 - xx * 0.34 + yy * 0.35, 0, 255).astype(np.uint8)

    image = Image.fromarray(base, mode="RGB")
    draw = ImageDraw.Draw(image, "RGBA")

    for block_y in range(0, height, 16):
        for block_x in range(0, width, 16):
            if int(rng.integers(0, 2)):
                color = tuple(int(v) for v in rng.integers(40, 220, size=3))
                draw.rectangle([block_x, block_y, block_x + 15, block_y + 15], fill=color + (70,))
                if int(rng.integers(0, 2)):
                    draw.line(
                        [block_x, block_y + 15, block_x + 15, block_y],
                        fill=(255, 255, 255, 90), width=1,
                    )

    for _ in range(14):
        x0, y0 = (int(rng.integers(0, width - 20)), int(rng.integers(0, height - 20)))
        x1 = x0 + int(rng.integers(8, 34))
        y1 = y0 + int(rng.integers(8, 28))
        color = tuple(int(v) for v in rng.integers(30, 245, size=3))
        shape = int(rng.integers(0, 3))
        if shape == 0:
            draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        elif shape == 1:
            draw.ellipse([x0, y0, x1, y1], fill=color + (150,), outline=(20, 20, 20, 180))
        else:
            draw.polygon([(x0, y1), ((x0 + x1) // 2, y0), (x1, y1)], fill=color + (170,))

    pixels = np.asarray(image, dtype=np.int16)
    noise = rng.normal(0, 5.0, size=pixels.shape)
    pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(pixels, mode="RGB")
    draw = ImageDraw.Draw(image, "RGBA")

    rectangles = _rects_for_index(index)
    for rect in rectangles:
        x, y, w, h = rect["x"], rect["y"], rect["width"], rect["height"]
        draw.rectangle([x, y, x + w - 1, y + h - 1], fill=(12, 10, 16, 235), outline=(255, 70, 90, 255), width=1)
        for _ in range(max(8, w * h // 18)):
            px = int(rng.integers(x, x + w))
            py = int(rng.integers(y, y + h))
            color = tuple(int(v) for v in rng.integers(0, 255, size=3))
            draw.point((px, py), fill=color)

    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    write_png(image_path, np.asarray(image, dtype=np.uint8))
    mask_document = {
        "schema_version": 1,
        "image": image_path.name,
        "width": width,
        "height": height,
        "coordinate_convention": "x/y are top-left; width/height are exclusive in positive directions",
        "rectangles": rectangles,
    }
    mask_path.write_text(json.dumps(mask_document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return mask_document


def default_config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "global_version": 1,
        "seed": 28525,
        "image": {
            "path": "../samples/images/sample_01.png",
            "width": SAMPLE_WIDTH,
            "height": SAMPLE_HEIGHT,
            "channels": 3,
        },
        "mask": {"path": "../samples/masks/sample_01.mask.json"},
        "pyramid": {
            "levels": 4,
            "downsample_factors": [2, 2, 2],
            "tile_sizes": [4, 8, 16, 32],
        },
        "fixed_point": {"fractional_bits_by_level": [8, 8, 8, 8]},
        "quantization": {"levels_by_level": [8, 16, 32, 64]},
        "inpaint": {
            "iterations_by_level": [18, 8, 5, 3],
            "guidance_strength_by_level": [0.0, 0.72, 0.62, 0.52],
        },
        "style": {
            "strength_by_level": [0.34, 0.44, 0.54, 0.62],
            "weights_by_level": [
                [1.22, 1.00, 0.82],
                [1.16, 1.02, 0.86],
                [1.10, 1.00, 0.92],
                [1.06, 1.00, 0.96],
            ],
        },
        "budget": {"tiles_per_batch": 12},
        "concurrency": {"max_workers": 4},
        "snapshot": {"dir": "../state"},
    }


def default_commands() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "commands": [
            {
                "id": "cmd-01-modify-alpha",
                "op": "upsert_rect",
                "rectangle": {"id": "defect_alpha", "x": 20, "y": 17, "width": 36, "height": 22},
            },
            {
                "id": "cmd-02-add-temp",
                "op": "upsert_rect",
                "rectangle": {"id": "defect_temp", "x": 48, "y": 70, "width": 24, "height": 18},
            },
            {"id": "cmd-03-delete-temp", "op": "delete_rect", "rectangle_id": "defect_temp"},
            {"id": "cmd-04-iterations-l2", "op": "set_iterations", "level": 2, "iterations": 7},
            {"id": "cmd-05-guidance-l1", "op": "set_guidance", "level": 1, "strength": 0.84},
            {"id": "cmd-06-style-weight-l2", "op": "set_style_weight", "level": 2,
             "weights": [1.34, 0.92, 0.72]},
            {"id": "cmd-07-quant-l3", "op": "set_quantization", "level": 3,
             "levels": 16, "fractional_bits": 8},
            {"id": "cmd-08-guidance-l3", "op": "set_guidance", "level": 3, "strength": 0.68},
        ],
    }


def default_injection() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "note": "Coordinates target the sample geometry: level 3 has 3 rows x 4 columns of 32px tiles.",
        "injections": [
            {
                "id": "INJ-001-NONFINITE",
                "kind": "nonfinite_pixels",
                "level": 3,
                "row": 0,
                "col": 0,
                "stage": "repair",
            },
            {
                "id": "INJ-002-BAD-MASK",
                "kind": "invalid_mask_rectangle",
                "level": 3,
                "row": 0,
                "col": 1,
                "stage": "repair",
                "rect": {"id": "injected_bad", "x": 900, "y": 4, "width": 12, "height": 8},
            },
            {
                "id": "INJ-003-BAD-QUANT",
                "kind": "quantization_out_of_range",
                "level": 3,
                "row": 1,
                "col": 0,
                "stage": "style",
                "levels": 7,
                "fractional_bits": 8,
            },
        ],
    }


def invalid_config_example() -> dict[str, Any]:
    data = default_config()
    data["pyramid"]["downsample_factors"] = [3, 2, 2]
    return data


def invalid_mask_example() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "image": "sample_01.png",
        "width": SAMPLE_WIDTH,
        "height": SAMPLE_HEIGHT,
        "rectangles": [
            {"id": "out_of_range", "x": 110, "y": 80, "width": 40, "height": 30},
        ],
    }


def generate_workspace(root: str | Path, samples: int = 3) -> list[Path]:
    root = Path(root)
    written: list[Path] = []
    configs_dir = root / "configs"
    images_dir = root / "samples" / "images"
    masks_dir = root / "samples" / "masks"
    examples_dir = root / "examples"
    for directory in (configs_dir, images_dir, masks_dir, examples_dir, root / "state"):
        directory.mkdir(parents=True, exist_ok=True)

    base_seed = int(default_config()["seed"])
    for index in range(1, samples + 1):
        stem = f"sample_{index:02d}"
        image_path = images_dir / f"{stem}.png"
        mask_path = masks_dir / f"{stem}.mask.json"
        generate_sample(index - 1, base_seed + index * 1009, image_path, mask_path)
        written.extend([image_path, mask_path])

    files = {
        configs_dir / "workbench.json": default_config(),
        configs_dir / "edit_stream.json": default_commands(),
        configs_dir / "injection.json": default_injection(),
        examples_dir / "invalid_config.json": invalid_config_example(),
        examples_dir / "invalid_mask.json": invalid_mask_example(),
    }
    for path, data in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written.append(path)
    return written
