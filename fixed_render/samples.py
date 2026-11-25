from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .hashing import derive_seed
from .miniyaml import dump_yaml


VOCABULARY = (
    "entity_cube",
    "entity_sphere",
    "entity_cylinder",
    "entity_pyramid",
    "entity_grid",
    "entity_wave",
    "attr_red",
    "attr_blue",
    "attr_green",
    "attr_yellow",
    "attr_bright",
    "attr_dark",
    "attr_smooth",
    "attr_rough",
    "attr_fast",
    "attr_slow",
    "attr_warm",
    "attr_cool",
    "motion_pan",
    "motion_zoom",
    "motion_rotate",
    "motion_drift",
    "quality_sharp",
    "quality_soft",
)


def generate_all_samples(root: str | Path = ".") -> dict[str, str]:
    root = Path(root).resolve()
    vocab_path = root / "samples" / "vocab" / "controlled_vocab.txt"
    image_a = root / "samples" / "images" / "condition_a.png"
    image_b = root / "samples" / "images" / "condition_b.png"
    motion_path = root / "samples" / "motion" / "vectors.csv"
    config_path = root / "configs" / "render.yaml"
    commands_path = root / "commands" / "edit_sequence.yaml"
    for path in [vocab_path.parent, image_a.parent, motion_path.parent, config_path.parent, commands_path.parent]:
        path.mkdir(parents=True, exist_ok=True)

    vocab_path.write_text("\n".join(VOCABULARY) + "\n", encoding="utf-8")
    generate_png(101, image_a)
    generate_png(202, image_b)
    generate_motion(303, motion_path, block_rows=8, block_cols=8)
    config_path.write_text(dump_yaml(sample_config()), encoding="utf-8")
    commands_path.write_text(dump_yaml({"commands": sample_commands()}), encoding="utf-8")
    return {
        "vocab": str(vocab_path),
        "image_a": str(image_a),
        "image_b": str(image_b),
        "motion": str(motion_path),
        "config": str(config_path),
        "commands": str(commands_path),
    }


def generate_png(seed: int, path: str | Path, size: int = 64) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    pixels = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            pixels[y, x] = (
                (seed + x * 3 + y * 5) % 96 + 24,
                (seed * 2 + x * 5 - y * 2) % 96 + 24,
                (seed * 3 - x * 2 + y * 7) % 96 + 24,
            )
    image = Image.fromarray(pixels, "RGB")
    draw = ImageDraw.Draw(image)

    for _ in range(10):
        x0 = rng.randrange(0, size - 12)
        y0 = rng.randrange(0, size - 12)
        x1 = x0 + rng.randrange(6, 24)
        y1 = y0 + rng.randrange(6, 24)
        color = (rng.randrange(120, 256), rng.randrange(80, 220), rng.randrange(80, 256))
        shape = rng.randrange(3)
        if shape == 0:
            draw.rectangle((x0, y0, x1, y1), outline=color, width=2)
        elif shape == 1:
            draw.ellipse((x0, y0, x1, y1), fill=color)
        else:
            draw.polygon([(x0, y1), ((x0 + x1) // 2, y0), (x1, y1)], fill=color)

    block = 8
    for by in range(0, size, block):
        for bx in range(0, size, block):
            if ((bx // block) + (by // block) + seed) % 3 == 0:
                shade = 160 + ((bx * 7 + by * 11 + seed) % 80)
                for yy in range(by + 1, min(by + block - 1, size)):
                    for xx in range(bx + 1, min(bx + block - 1, size), 2):
                        image.putpixel((xx, yy), (shade, shade // 2, 255 - shade // 3))
    image.save(path, format="PNG", compress_level=6)
    return path


def generate_motion(
    seed: int, path: str | Path, block_rows: int = 8, block_cols: int = 8
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(derive_seed("motion", seed))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["block_row", "block_col", "dx_q8", "dy_q8"])
        for br in range(block_rows):
            for bc in range(block_cols):
                dx = int(rng.integers(-2, 3)) * 256 + int(rng.integers(-96, 97))
                dy = int(rng.integers(-2, 3)) * 256 + int(rng.integers(-96, 97))
                writer.writerow([br, bc, dx, dy])
    return path


def sample_config() -> dict[str, Any]:
    return {
        "seed": 20261126,
        "frame_width": 64,
        "frame_height": 64,
        "num_frames": 2,
        "latent_scale": 4,
        "latent_channels": 4,
        "text_dim": len(VOCABULARY),
        "image_width": 64,
        "image_height": 64,
        "image_grid": 4,
        "image_dim": 80,
        "motion_block_h": 8,
        "motion_block_w": 8,
        "motion_dim": 128,
        "fusion_dim": 32,
        "denoise_steps": 4,
        "tile_height": 8,
        "guidance_q8": 1536,
        "noise_amplitude": 96,
        "fusion_weights": {"text": 3, "image": 2, "motion": 1},
        "fixed_frac_bits": 8,
        "quant_mode": "round_half_up",
        "preview_budget_ms": 250,
        "lease_ttl_ms": 5000,
        "executor_host": "127.0.0.1",
        "executor_port": 48765,
        "max_slots": 2,
        "conditions": {
            "vocab_path": "../samples/vocab/controlled_vocab.txt",
            "prompt_tokens": ["entity_cube", "attr_red", "attr_bright", "quality_sharp"],
            "image_path": "../samples/images/condition_a.png",
            "motion_path": "../samples/motion/vectors.csv",
        },
    }


def sample_commands() -> list[dict[str, Any]]:
    return [
        {"op": "set_prompt", "tokens": ["entity_sphere", "attr_blue", "attr_cool"]},
        {"op": "set_image", "path": "samples/images/condition_b.png"},
        {"op": "delete_image"},
        {"op": "set_image", "path": "samples/images/condition_a.png"},
        {"op": "tune_motion", "block_row": 2, "block_col": 3, "dx_delta_q8": 512, "dy_delta_q8": -256},
        {"op": "set_weights", "weights": {"text": 1, "image": 4, "motion": 2}},
        {"op": "set_denoise_steps", "value": 6},
        {"op": "set_tile_height", "value": 16},
        {"op": "set_quant", "frac_bits": 12, "mode": "round_half_even"},
    ]
