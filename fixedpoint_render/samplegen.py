"""Deterministic local sample generation (no network, no GPU).

All samples derive from a single integer seed.  Prompts come from the
controlled vocabulary, PNGs are fixed resolution/mode and motion tables have one
block entry per configured tile row.
"""
from __future__ import annotations

import json
import os

from PIL import Image

from .config import RenderConfig
from .rng import DeterministicRandom


def generate_workspace(workdir: str, config: RenderConfig,
                       frame_count: int, seed: int = 20261127) -> str:
    for sub in ("prompts", "images", "motion"):
        os.makedirs(os.path.join(workdir, sub), exist_ok=True)
    with open(os.path.join(workdir, "config.json"), "w",
              encoding="utf-8") as fh:
        json.dump(config.to_dict(), fh, sort_keys=True, indent=1)

    rng = DeterministicRandom(f"sample-gen|seed={seed}|frames={frame_count}")
    for frame in range(frame_count):
        fid = f"frame_{frame:02d}"
        # Prompt: 2-4 controlled vocab terms.
        term_count = 2 + rng.next_uint(min(3, len(config.vocab)))
        terms = [config.vocab[rng.next_uint(len(config.vocab))]
                 for _ in range(term_count)]
        with open(os.path.join(workdir, "prompts", fid + ".txt"),
                  "w", encoding="utf-8") as fh:
            fh.write(" ".join(terms))

        # PNG: deterministic per-pixel integer pattern, fixed size/mode.
        channels = config.channels
        buffer = bytearray()
        for row in range(config.height):
            for col in range(config.width):
                for ch in range(channels):
                    value = (frame * 31 + row * 7 + col * 13 +
                             ch * 17 + rng.next_uint(64)) % 256
                    buffer.append(value)
        image = Image.frombytes(config.pixel_mode,
                                (config.width, config.height), bytes(buffer))
        image.save(os.path.join(workdir, "images", fid + ".png"),
                   format="PNG", compress_level=6)

        # Motion: one tile entry per tile row.
        tiles = []
        for tile in range(config.tile_rows):
            tiles.append({
                "tile": tile,
                "vx": rng.next_signed(20),
                "vy": rng.next_signed(20),
            })
        motion = {"frame": frame, "tile_rows": config.tile_rows,
                  "frame_cover": 40 + rng.next_uint(61), "tiles": tiles}
        with open(os.path.join(workdir, "motion", fid + ".json"),
                  "w", encoding="utf-8") as fh:
            json.dump(motion, fh, sort_keys=True, indent=1)
    return workdir


def default_config() -> RenderConfig:
    return RenderConfig(
        vocab=("city", "rain", "neon", "street", "night", "crowd",
               "vehicle", "fog"),
        width=24, height=16, pixel_mode="RGB", channels=3, tile_rows=4,
        condition_dim=8, latent_dim=8, frac_bits=12, qmode="trunc",
        weights={"text": "0.30", "image": "0.45", "motion": "0.25"},
        guidance="0.50", per_frame_budget=6, global_step_budget=20,
        noise_seed=20261127, max_workers=4,
    )
