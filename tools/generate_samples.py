"""Generate all local, deterministic sample assets used by the demo and tests."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


def draw_texture(path: Path, seed: int, resolution: int, variant: int) -> None:
    rng = np.random.default_rng(seed)
    image = Image.new("RGB", (resolution, resolution), (18 + variant * 3, 22, 30 - variant * 2))
    draw = ImageDraw.Draw(image)
    palette = [
        (234, 116, 72),
        (58, 156, 190),
        (218, 190, 88),
        (104, 198, 142),
        (170, 116, 214),
    ]
    for item_index in range(9):
        color = palette[(item_index + variant) % len(palette)]
        x0 = int((item_index * 17 + variant * 9 + 5) % resolution)
        y0 = int((item_index * 29 + variant * 13 + 7) % resolution)
        radius = 4 + ((item_index * 7 + variant * 3) % 13)
        if item_index % 3 == 0:
            draw.ellipse((x0 - radius, y0 - radius, x0 + radius, y0 + radius), fill=color)
        elif item_index % 3 == 1:
            draw.rectangle((x0, y0, x0 + radius + 7, y0 + radius), fill=color)
        else:
            draw.line((0, y0, resolution, y0 + radius), fill=color, width=1 + (item_index % 4))
    point_pixels = np.asarray(image, dtype=np.uint8).copy()
    for point_index in range(36):
        x = int((point_index * 37 + seed // 7 + variant * 11) % resolution)
        y = int((point_index * 53 + seed // 13 + variant * 17) % resolution)
        point_pixels[y, x] = np.asarray(palette[(point_index + variant) % len(palette)], dtype=np.uint8)

    array = point_pixels.astype(np.int64)
    yy, xx = np.mgrid[0:resolution, 0:resolution]
    wave = (
        np.sin((xx + variant * 5) * 0.23) * 13
        + np.cos((yy - variant * 7) * 0.19) * 11
        + np.sin((xx + yy) * 0.11) * 6
    )
    noise = rng.integers(-5, 6, size=(resolution, resolution, 1))
    array = np.clip(array + wave[:, :, None] + noise, 0, 255).astype(np.uint8)
    Image.fromarray(array, mode="RGB").save(path, format="PNG")


def main() -> None:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    resolution = 64
    draw_texture(SAMPLES / "base_texture.png", seed=20261126, resolution=resolution, variant=1)
    draw_texture(SAMPLES / "base_texture_v2.png", seed=20261127, resolution=resolution, variant=2)

    config = {
        "name": "layered-fixed-sample",
        "texture": "base_texture.png",
        "resolution_tiers": [32, 64, 96, 128],
        "resolution": 64,
        "seed": 20261126,
        "algorithm": "value2d",
        "noise": {
            "layer_count": 5,
            "layers": [
                {"frequency": 1.5, "amplitude": 1.0, "seed": 101},
                {"frequency": 3.25, "amplitude": 0.85, "seed": 202},
                {"frequency": 6.0, "amplitude": 0.7, "seed": 303},
                {"frequency": 10.5, "amplitude": 0.55, "seed": 404},
                {"frequency": 16.0, "amplitude": 0.4, "seed": 505},
            ],
        },
        "fixed": {
            "fraction_bits": 12,
            "quant_step": 4,
            "tie_rule": "ties_to_even",
            "guard_bits": 2,
        },
        "budgets": {"layer_points": 4096, "batch_points": 65536},
        "blend_mode": "weighted",
        "preview_budget_ms": 500,
        "thread_pool_size": 4,
        "commit_interval_commands": 3,
        "error_tolerance_quant_steps": 0,
    }
    (SAMPLES / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    low_budget = json.loads(json.dumps(config))
    low_budget["name"] = "layered-fixed-low-budget"
    low_budget["budgets"] = {"layer_points": 1024, "batch_points": 3072}
    low_budget["preview_budget_ms"] = 5
    low_budget["thread_pool_size"] = 3
    (SAMPLES / "config_low_budget.json").write_text(json.dumps(low_budget, indent=2) + "\n", encoding="utf-8")

    material = {
        "name": "four-channel-sample",
        "metadata": {
            "tie_rule": "ties_to_even",
            "equal_weight_rule": "lower_layer_index",
        },
        "channels": {
            "diffuse": {
                "base_weight": 1.0,
                "layers": [
                    {"index": 0, "weight": 0.35},
                    {"index": 1, "weight": 0.25},
                    {"index": 3, "weight": 0.15},
                ],
            },
            "roughness": {
                "base_weight": 0.8,
                "layers": [
                    {"index": 1, "weight": 0.4},
                    {"index": 2, "weight": 0.6},
                    {"index": 4, "weight": 0.3},
                ],
            },
            "metallic": {
                "base_weight": 0.7,
                "layers": [
                    {"index": 0, "weight": 0.2},
                    {"index": 2, "weight": 0.3},
                    {"index": 3, "weight": 0.5},
                ],
            },
            "normal": {
                "base_weight": 0.9,
                "layers": [
                    {"index": 1, "weight": 0.4},
                    {"index": 3, "weight": 0.3},
                    {"index": 4, "weight": 0.5},
                ],
            },
        },
    }
    (SAMPLES / "material.json").write_text(json.dumps(material, indent=2) + "\n", encoding="utf-8")

    commands = [
        {"command": "set_channel_weight", "channel": "roughness", "layer": 2, "weight": 0.9},
        {"command": "set_layer_param", "layer": 1, "frequency": 4.5, "amplitude": 0.9},
        {"command": "set_blend_mode", "mode": "max"},
        {"command": "set_quant_step", "step": 8},
        {"command": "replace_texture", "path": "samples/base_texture_v2.png"},
        {"command": "set_frac_bits", "bits": 12},
        {"command": "set_channel_weight", "channel": "diffuse", "layer": 3, "weight": 0.45},
        {"command": "set_quant_step", "step": 4},
        {"command": "set_blend_mode", "mode": "weighted"},
        {"command": "commit"},
    ]
    (SAMPLES / "commands.jsonl").write_text(
        "".join(json.dumps(item, separators=(",", ":")) + "\n" for item in commands), encoding="utf-8"
    )

    injections = {
        "injections": [
            {
                "layer": 2,
                "stage": "layer_quantize",
                "code": "E_INJECTED_LAYER_QUANTIZE",
                "message": "sample fault at layer 2 quantization",
            },
            {
                "layer": 1,
                "stage": "material_mix",
                "channel": "diffuse",
                "code": "E_INJECTED_MATERIAL_MIX",
                "message": "sample fault while reading layer 1 for diffuse mixing",
            },
        ]
    }
    (SAMPLES / "injections.json").write_text(json.dumps(injections, indent=2) + "\n", encoding="utf-8")

    print(f"wrote samples to {SAMPLES}")


if __name__ == "__main__":
    main()
