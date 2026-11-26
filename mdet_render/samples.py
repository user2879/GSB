"""Deterministic local sample workspace generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .formats import v2_to_v1
from .png import encode_rgb_png
from .util import atomic_write_bytes, atomic_write_text, canonical_json, sha256_hex

ENTITY_TERMS = [
    "cube", "sphere", "torus", "prism", "ring", "leaf", "stone", "cloud", "wave", "star",
    "crystal", "shell", "feather", "gear", "pebble", "bloom", "shard", "dune", "arc", "lattice",
]
ATTRIBUTE_TERMS = [
    "crimson", "azure", "gold", "ivory", "violet", "emerald", "amber", "silver", "warm", "cool",
    "glossy", "matte", "rough", "smooth", "metal", "glass", "soft", "hard", "striped", "speckled",
]
VOCAB_TERMS = ENTITY_TERMS + ATTRIBUTE_TERMS


def _prompt(prompt_id: str, entities: list[str], attributes: list[str]) -> dict[str, Any]:
    return {"prompt_id": prompt_id, "entities": entities, "attributes": attributes}


def _synthetic_rgb(width: int, height: int, seed: int) -> np.ndarray:
    y = np.arange(height, dtype=np.int64)[:, None]
    x = np.arange(width, dtype=np.int64)[None, :]
    red = (x * 5 + y * 2 + seed * 17) % 256
    green = (x * 2 + y * 7 + seed * 29) % 256
    blue = ((x ^ y) * 3 + seed * 11) % 256
    pixels = np.stack([red, green, blue], axis=2).astype(np.uint8)
    # Add a flat block and a few edges so gradient statistics are non-trivial.
    pixels[height // 4: height // 2, width // 3: 2 * width // 3] = np.array(
        [(31 + seed * 7) % 256, (127 + seed * 3) % 256, (211 + seed * 5) % 256], dtype=np.uint8
    )
    pixels[1::5, ::7] = np.array(
        [(250 - seed * 3) % 256, (19 + seed * 2) % 256, (90 + seed) % 256], dtype=np.uint8
    )
    return pixels


def _motion_table(rows: int, cols: int, seed: int) -> dict[str, Any]:
    rng = np.random.Generator(np.random.PCG64(seed))
    values = rng.integers(-160, 161, size=(rows, cols, 2), dtype=np.int64)
    return {
        "version": 1,
        "block_rows": rows,
        "block_cols": cols,
        "vector_dim": 2,
        "seed": seed,
        "blocks": values.tolist(),
    }


def create_sample_workspace(root: str | Path = "samples", *, width: int = 64, height: int = 48,
                            frames: int = 3, denoise_steps: int = 8,
                            max_workers: int = 3) -> dict[str, Any]:
    root = Path(root).resolve()
    prompts_dir = root / "prompts"
    images_dir = root / "images"
    out_dir = root / "out"
    ledger_dir = root / "ledger"
    for directory in (root, prompts_dir, images_dir, out_dir, ledger_dir):
        directory.mkdir(parents=True, exist_ok=True)
    vocabulary = {
        "version": 1,
        "terms": VOCAB_TERMS,
        "entities": ENTITY_TERMS,
        "attributes": ATTRIBUTE_TERMS,
    }
    atomic_write_text(root / "vocabulary.json", canonical_json(vocabulary) + "\n")
    prompt_bodies = [
        _prompt("frame-0", ["cube", "sphere", "torus"], ["crimson", "glossy", "metal"]),
        _prompt("frame-1", ["wave", "stone"], ["azure", "matte", "smooth", "cool"]),
        _prompt("frame-2", ["crystal", "ring", "leaf"], ["gold", "glass", "hard"]),
        _prompt("frame-3", ["dune", "arc"], ["warm", "silver", "rough"]),
    ]
    written_prompts: list[str] = []
    for index in range(frames):
        target = prompts_dir / f"frame{index}.json"
        atomic_write_text(target, json.dumps(prompt_bodies[index], ensure_ascii=False,
                                             indent=2, sort_keys=True) + "\n")
        written_prompts.append(str(target))
    for index in range(frames):
        pixels = _synthetic_rgb(width, height, seed=index + 1)
        atomic_write_bytes(images_dir / f"frame{index}.png", encode_rgb_png(pixels))
    atomic_write_bytes(images_dir / "frame0_alt.png", encode_rgb_png(_synthetic_rgb(width, height, 97)))
    motion = _motion_table(4, 5, seed=2026)
    atomic_write_text(root / "motion.json", json.dumps(motion, indent=2, sort_keys=True) + "\n")
    point_budget = width * height * 3 * 3
    config = {
        "schema_version": 1,
        "resolution": {"width": width, "height": height},
        "pixel_mode": "RGB",
        "condition_dim": 40,
        "vocabulary": "vocabulary.json",
        "motion": "motion.json",
        "frames": [
            {
                "frame": index,
                "prompt": f"prompts/frame{index}.json",
                "image": f"images/frame{index}.png",
            }
            for index in range(frames)
        ],
        "fixed_point": {"fraction_bits": 10, "storage_bits": 32, "quant_level": 2},
        "fusion": {"weights": {"image": 0.75, "motion": 0.5, "text": 1.0}},
        "image_encoding": {"grid_rows": 4, "grid_cols": 2},
        "motion_encoding": {"block_rows": 4, "block_cols": 5},
        "noise": {"seed": 7719, "table_size": 8},
        "latent": {"seed": 481516},
        "denoise": {"steps": denoise_steps, "guidance_strength": 0.875},
        "decode": {"block_rows": 4, "projection_seed": 9001},
        "execution": {"max_workers": max_workers, "point_budget": point_budget,
                      "max_denoise_steps": 32},
        "lease": {"ttl_ms": 10_000},
    }
    atomic_write_text(root / "config.json", json.dumps(config, indent=2, sort_keys=True) + "\n")
    commands = [
        {"op": "set_quantization", "fraction_bits": 8, "storage_bits": 32, "quant_level": 1},
        {"op": "replace_prompt", "frame": 1,
         "prompt": _prompt("frame-1-edited", ["cloud", "gear"], ["violet", "glass", "soft"])},
        {"op": "replace_image", "frame": 0, "path": "images/frame0_alt.png"},
        {"op": "delete_image", "frame": 2},
        {"op": "set_weight", "condition": "image", "weight": 1.125},
        {"op": "set_denoise_steps", "steps": 6},
        {"op": "set_decode_block_rows", "block_rows": 5},
        {"op": "set_lease", "ttl_ms": 12_000},
    ]
    atomic_write_text(root / "commands.json", json.dumps(commands, indent=2, sort_keys=False) + "\n")
    lease_params = {
        "ttl_ms": 250,
        "initial_now_ms": 1000,
        "advance_ms_before_renew": 150,
        "advance_ms_to_expiry": 300,
        "replacement_ttl_ms": 400,
    }
    atomic_write_text(root / "lease.json", json.dumps(lease_params, indent=2, sort_keys=True) + "\n")
    intermediate_v2 = {
        "format_version": 2,
        "records": [
            {
                "stage": "denoise",
                "frame": 0,
                "step": 3,
                "block": 0,
                "guide_strength": 0.875,
                "noise_scheme": "perlin",
                "precision_tag": "sample-v2-high-precision",
                "quantization_bits": 10,
                "quantization_level": 2,
                "residual": -12,
                "digest": sha256_hex(canonical_json({"sample": "denoise", "frame": 0, "step": 3})),
                "extra": {"sample_note": "round-trip me"},
            },
            {
                "stage": "block",
                "frame": 0,
                "step": 3,
                "block": 0,
                "guide_strength": 0.8754,
                "noise_scheme": "uniform",
                "precision_tag": "sample-v2-block",
                "quantization_bits": 8,
                "quantization_level": 1,
                "residual": 5,
                "digest": sha256_hex(canonical_json({"sample": "block", "frame": 0, "row": 0})),
            },
        ],
    }
    v1_text, sidecar = v2_to_v1(intermediate_v2)
    atomic_write_text(root / "intermediate_v2.json",
                      json.dumps(intermediate_v2, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    atomic_write_text(root / "intermediate.v1", v1_text)
    atomic_write_text(root / "intermediate.v1.sidecar.json",
                      json.dumps(sidecar, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    manifest = {
        "config": str(root / "config.json"),
        "vocabulary": str(root / "vocabulary.json"),
        "motion": str(root / "motion.json"),
        "commands": str(root / "commands.json"),
        "lease": str(root / "lease.json"),
        "intermediate_v2": str(root / "intermediate_v2.json"),
        "intermediate_v1": str(root / "intermediate.v1"),
        "intermediate_v1_sidecar": str(root / "intermediate.v1.sidecar.json"),
        "prompts": written_prompts,
        "images": sorted(str(path) for path in images_dir.glob("*.png")),
        "out_dir": str(out_dir),
        "ledger_dir": str(ledger_dir),
        "sample_sha256": sha256_hex(canonical_json({"config": config, "vocabulary": vocabulary,
                                                    "motion": motion, "commands": commands})),
    }
    atomic_write_text(root / "sample_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def default_sample_root(cwd: str | Path | None = None) -> Path:
    base = Path(cwd) if cwd is not None else Path.cwd()
    return (base / "samples").resolve()
