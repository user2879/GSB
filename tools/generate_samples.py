#!/usr/bin/env python3
"""Fixed-seed local sample generator.

Produces a self-contained sample directory containing:
  * base PNG textures at every configured resolution tier (uint8, P or RGB),
  * materials.json with per-channel layered weights,
  * generation_config.json with layers, fixed-point tiers, budget and threads.

Everything is derived from integer seeds; no network or external assets are
used.  Re-running with the same seed reproduces every file byte-for-byte.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from texmat_pipeline.noise import pattern_control_points, value_noise_field  # noqa: E402

BASE_TEXTURES = ("stone", "wood", "metal")
LAYER_COUNT = 4


def _grayscale_field(seed: int, width: int, height: int) -> np.ndarray:
    field = np.zeros((height, width), dtype=np.float64)
    points = pattern_control_points(seed, width, height)
    yy, xx = np.mgrid[0:height, 0:width]
    for px, py, radius, strength in points:
        dist2 = (xx - px) ** 2 + (yy - py) ** 2
        field += strength * np.exp(-dist2 / (2.0 * (radius * max(width, height)) ** 2))
    field += 0.5 * value_noise_field(width, height, frequency=4.0 / max(width, height), fingerprint=seed * 7919 + 13)
    field = field - field.min()
    if field.max() > 0:
        field = field / field.max()
    return np.clip(field, 0.0, 1.0)


def write_base_textures(out_dir: Path, resolution: int, seed: int) -> None:
    for tex_idx, name in enumerate(BASE_TEXTURES):
        tex_seed = seed + tex_idx * 101
        gray = _grayscale_field(tex_seed, resolution, resolution)
        if name == "metal":
            # RGB pattern texture to exercise mode validation/conversion
            r = gray
            g = np.clip(gray * 0.7 + 0.15, 0, 1)
            b = np.clip(gray * 0.4 + 0.3, 0, 1)
            rgb = (np.stack([r, g, b], axis=-1) * 255).round().astype(np.uint8)
            img = Image.fromarray(rgb, mode="RGB")
        else:
            img = Image.fromarray((gray * 255).round().astype(np.uint8), mode="L")
        img.save(out_dir / f"{name}_{resolution}.png", format="PNG", optimize=True)


def build_config(seed: int, tiers: list[int], layers: int, threads: int,
                 point_budget: int, frac_bits: int, round_mode: str, blend_mode: str) -> dict:
    layer_defs = []
    for i in range(layers):
        layer_defs.append(
            {
                "fingerprint": (seed * 104729 + i * 4099 + 7) & 0x7FFFFFFF,
                "frequency": round(2.0 + i * 1.75, 4),
                "amplitude": round(0.5 / (i + 1), 4),
                "base_texture": BASE_TEXTURES[i % len(BASE_TEXTURES)],
            }
        )
    return {
        "resolution_tiers": tiers,
        "layers": layer_defs,
        "blend_mode": blend_mode,
        "fixed_point": {
            "frac_bits": frac_bits,
            "unit_levels": 256,
            "signed_levels": 512,
            "round_mode": round_mode,
        },
        "tolerance": 2e-3,
        "threads": threads,
        "point_budget": point_budget,
        "snapshot_interval": 2,
    }


def build_materials(seed: int, layers: int) -> dict:
    materials = []
    for m, mid in enumerate(("mat_slate", "mat_bronze")):
        weights: dict[str, dict[str, float]] = {}
        for c, channel in enumerate(("diffuse", "roughness", "metallic", "normal")):
            chan: dict[str, float] = {}
            for i in range(layers):
                # deterministic, mostly positive weights; equal weights appear
                # for the slate material so boundary decisions are exercised
                if mid == "mat_slate" and channel in ("diffuse", "normal"):
                    w = 1.0
                else:
                    w = float(((seed + i * 13 + c * 7 + m * 3) % 5) + 1)
                chan[str(i)] = w
            weights[channel] = chan
        materials.append({"material_id": mid, "layer_weights": weights})
    return {"materials": materials}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="sample output directory")
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--tiers", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--layers", type=int, default=LAYER_COUNT)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--point-budget", type=int, default=2048)
    parser.add_argument("--frac-bits", type=int, default=8)
    parser.add_argument("--round-mode", default="half_even")
    parser.add_argument("--blend-mode", default="normalized_weight")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for resolution in args.tiers:
        write_base_textures(out, resolution, args.seed)
    (out / "generation_config.json").write_text(
        json.dumps(
            build_config(
                args.seed, args.tiers, args.layers, args.threads,
                args.point_budget, args.frac_bits, args.round_mode, args.blend_mode,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "materials.json").write_text(
        json.dumps(build_materials(args.seed, args.layers), indent=2), encoding="utf-8"
    )
    print(f"generated sample set in {out} (tiers={args.tiers}, layers={args.layers})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
