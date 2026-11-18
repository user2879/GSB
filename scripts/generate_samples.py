"""Generate a deterministic local sample bundle (PNGs + JSON descriptions).

The PNGs are produced from fixed integer seeds and explicit control points; no
network resource is used.  The script also writes an editable edit-sequence file
used by the incremental/recovery tests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def splitmix(value: np.ndarray) -> np.ndarray:
    value = (value + np.uint64(0x9E3779B97F4A7C15)) & MASK64
    value = ((value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & MASK64
    value = ((value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & MASK64
    return value ^ (value >> np.uint64(31))


def hash_grid(width: int, height: int, seed: int) -> np.ndarray:
    ys, xs = np.meshgrid(np.arange(height, dtype=np.uint64), np.arange(width, dtype=np.uint64), indexing="ij")
    coords = (xs * np.uint64(0x9E3779B1) + ys * np.uint64(0x85EBCA77) + np.uint64(seed)) & MASK64
    return splitmix(splitmix(coords))


def make_texture(width: int, height: int, seed: int, control_points: list[tuple[int, int, int]]) -> np.ndarray:
    base = (hash_grid(width, height, seed).astype(np.uint32) % np.uint32(96)).astype(np.uint8)
    image = np.repeat(base[:, :, None], 3, axis=2)
    for cx, cy, radius in control_points:
        ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius**2
        image[mask] = np.minimum(image[mask].astype(np.uint16) + 120, 255).astype(np.uint8)
    return image


def default_config(size: int, workers: int, budget: int) -> dict:
    small_tier = max(8, size // 2)
    if small_tier >= size:
        small_tier = size // 2
    return {
        "name": "deterministic-sample",
        "resolution_tiers": [small_tier, size],
        "active_tier": size,
        "noise_layers": [
            {"source_texture": "tex_a", "frequency": 4, "amplitude": 0.25, "seed": 101, "fingerprint": 11},
            {"source_texture": "tex_a", "frequency": 8, "amplitude": 0.125, "seed": 202, "fingerprint": 22},
            {"source_texture": "tex_b", "frequency": 4, "amplitude": 0.1875, "seed": 303, "fingerprint": 33},
        ],
        "blend_mode": "weighted",
        "fractional_bits": 8,
        "quantize": True,
        "rounding": "half_even",
        "workers": workers,
        "point_budget": budget,
        "snapshot_interval": 2,
        "tolerances": {"noise_overlay": 4, "quantize": 2, "material_blend": 4, "channel_compose": 2},
        "bin_width": 16,
    }


def default_materials() -> list[dict]:
    return [
        {
            "name": "mat_stone",
            "layers": ["layer00", "layer01", "layer02"],
            "weights": [0.5, 0.3, 0.2],
            "blend_mode": "weighted",
            "channels": {
                "diffuse": {"base": 0.10, "scale": 0.70, "bias": 0.02},
                "roughness": {"base": 0.20, "scale": 0.50, "bias": 0.05},
                "metalness": {"base": 0.00, "scale": 0.20, "bias": 0.00},
                "normal": {"base": 0.05, "scale": 0.60, "bias": 0.03},
            },
        },
        {
            "name": "mat_metal",
            "layers": ["layer00", "layer02"],
            "weights": [0.5, 0.5],
            "blend_mode": "weighted",
            "channels": {
                "diffuse": {"base": 0.05, "scale": 0.50, "bias": 0.01},
                "roughness": {"base": 0.10, "scale": 0.30, "bias": 0.02},
                "metalness": {"base": 0.30, "scale": 0.60, "bias": 0.00},
                "normal": {"base": 0.02, "scale": 0.40, "bias": 0.01},
            },
        },
    ]


def write_bundle(out_dir: Path, size: int, workers: int, budget: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    textures = {
        "tex_a": make_texture(size, size, seed=7, control_points=[(6, 6, 3), (20, 14, 4)]),
        "tex_b": make_texture(size, size, seed=11, control_points=[(10, 18, 4), (22, 24, 2)]),
    }
    for name, pixels in textures.items():
        Image.fromarray(pixels, mode="RGB").save(out_dir / f"{name}.png")
    with (out_dir / "generation_config.json").open("w", encoding="utf-8") as handle:
        json.dump(default_config(size, workers, budget), handle, indent=2, sort_keys=True)
    with (out_dir / "materials.json").open("w", encoding="utf-8") as handle:
        json.dump(default_materials(), handle, indent=2, sort_keys=True)
    edits = [
        {"op": "replace_texture", "texture": "tex_a", "seed": 17},
        {"op": "set_layer", "layer_index": 1, "frequency": 12, "amplitude": 0.09},
        {"op": "set_config", "blend_mode": "additive"},
        {"op": "set_config", "fractional_bits": 10},
        {"op": "set_config", "blend_mode": "weighted", "fractional_bits": 8},
    ]
    with (out_dir / "edit_sequence.json").open("w", encoding="utf-8") as handle:
        json.dump(edits, handle, indent=2)
    fault_injections = [
        {"file": "broken.png", "kind": "truncated_png"},
        {"file": "notpng.png", "kind": "bad_signature"},
        {"config": "bad_config.json", "kind": "bad_resolution_tiers"},
    ]
    with (out_dir / "fault_injections.json").open("w", encoding="utf-8") as handle:
        json.dump(fault_injections, handle, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic texmat samples")
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--budget", type=int, default=100000)
    args = parser.parse_args()
    write_bundle(Path(args.out), args.size, args.workers, args.budget)
    print(f"sample bundle written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
