"""Deterministic local sample workspace generator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .canonical import atomic_write_bytes, atomic_write_json, atomic_write_text, canonical_json
from .exchange import v1_to_v2, v2_to_v1
from .png import encode_png_rgb


def _draw_image(seed: int, width: int = 48, height: int = 32) -> np.ndarray:
    """Draw deterministic geometric primitives and texture blocks."""

    rng = np.random.default_rng(seed)
    y = np.arange(height, dtype=np.int64)[:, None]
    x = np.arange(width, dtype=np.int64)[None, :]
    base = np.empty((height, width, 3), dtype=np.int64)
    base[:, :, 0] = 22 + (x * 3 + y * 2 + seed) % 90
    base[:, :, 1] = 28 + (x + y * 4 + seed * 3) % 96
    base[:, :, 2] = 34 + (x * 5 - y + seed * 7) % 104

    # Texture blocks are deliberately aligned to the condition encoder tiles.
    for ty in range(0, height, 8):
        for tx in range(0, width, 8):
            colour = rng.integers(24, 205, size=3, dtype=np.int64)
            shade = int(rng.integers(55, 100))
            block = base[ty : ty + 8, tx : tx + 8, :]
            base[ty : ty + 8, tx : tx + 8, :] = (block * shade + colour * (100 - shade)) // 100

    centre_x = int(12 + rng.integers(0, width - 24))
    centre_y = int(8 + rng.integers(0, height - 16))
    radius = int(5 + rng.integers(0, 8))
    circle = (x - centre_x) ** 2 + (y - centre_y) ** 2 <= radius**2
    circle_colour = rng.integers(80, 255, size=3, dtype=np.int64)
    base[circle] = (base[circle] + circle_colour * 2) // 3

    rect_x = int(rng.integers(0, width - 10))
    rect_y = int(rng.integers(0, height - 8))
    rect_w = int(rng.integers(5, 16))
    rect_h = int(rng.integers(4, 12))
    rect = (x >= rect_x) & (x < min(width, rect_x + rect_w)) & (y >= rect_y) & (y < min(height, rect_y + rect_h))
    rect_colour = rng.integers(30, 230, size=3, dtype=np.int64)
    base[rect] = (base[rect] + rect_colour) // 2

    # A small set of bright horizontal/vertical strokes changes gradients.
    stroke_x = int(rng.integers(0, width))
    stroke_y = int(rng.integers(0, height))
    stroke_length = int(rng.integers(3, min(18, height)))
    for offset in range(stroke_length):
        yy = (stroke_y + offset) % height
        xx = (stroke_x + offset * 2) % width
        base[yy, xx, :] = np.array([245, 220, 110], dtype=np.int64)
    return np.clip(base, 0, 255).astype(np.uint8)


def _motion_table(seed: int, width: int, height: int, block_size: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    rows = (height + block_size - 1) // block_size
    cols = (width + block_size - 1) // block_size
    vectors: list[list[list[int]]] = []
    for row in range(rows):
        vector_row: list[list[int]] = []
        for col in range(cols):
            dx = int(rng.integers(-3, 4))
            dy = int(rng.integers(-3, 4))
            vector_row.append([dx, dy])
        vectors.append(vector_row)
    return {
        "schema": "mmfm-motion-v1",
        "width": width,
        "height": height,
        "block_size": block_size,
        "vectors": vectors,
    }


def render_config() -> dict[str, Any]:
    return {
        "schema": "mmfm-render-config-v1",
        "seed": 20261127,
        "frame_count": 3,
        "condition_dim": 16,
        "latent_dim": 16,
        "image": {"width": 48, "height": 32, "mode": "RGB", "tile_size": 8},
        "motion": {"block_size": 8},
        "denoise_steps": 3,
        "chunk_rows": 2,
        "fusion_weights_permille": {"text": 1000, "image": 1000, "motion": 800},
        "quant_profile": "q8",
        "allowed_quant_profiles": ["q4", "q8", "q12"],
        "quant_profiles": {
            "q4": {"fraction_bits": 4, "scale": 16},
            "q8": {"fraction_bits": 8, "scale": 256},
            "q12": {"fraction_bits": 12, "scale": 4096},
        },
        "allowed_fraction_bits": [4, 8, 12],
        "fixed_point": {"milli_scale": 1000, "allowed_milli_scales": [1000]},
        "noise_scheme": "lcg",
        "noise_schemes": {"lcg": 1, "counter": 2, "xor": 3},
        "guidance_permille": 1000,
        "projection": {
            "version": "linear-v1",
            "coefficient_radius": 32,
            "denominator_scale": 32,
        },
        "exchange_version": "v2",
        "max_workers": 4,
        "max_denoise_points": 100,
        "vocab": "vocab.json",
    }


def vocabulary() -> dict[str, Any]:
    return {
        "schema": "mmfm-vocab-v1",
        "version": "1",
        "terms": [
            {"term": "crystal", "kind": "entity", "index": 0, "weight": 2},
            {"term": "fox", "kind": "entity", "index": 1, "weight": 2},
            {"term": "koi", "kind": "entity", "index": 2, "weight": 2},
            {"term": "orbit", "kind": "entity", "index": 3, "weight": 2},
            {"term": "amber", "kind": "attribute", "index": 4, "weight": 1},
            {"term": "cobalt", "kind": "attribute", "index": 5, "weight": 1},
            {"term": "soft", "kind": "attribute", "index": 6, "weight": 1},
            {"term": "sharp", "kind": "attribute", "index": 7, "weight": 1},
            {"term": "glowing", "kind": "attribute", "index": 8, "weight": 1},
            {"term": "matte", "kind": "attribute", "index": 9, "weight": 1},
            {"term": "rippled", "kind": "attribute", "index": 10, "weight": 1},
            {"term": "faceted", "kind": "attribute", "index": 11, "weight": 1},
        ],
    }


def conditions_manifest() -> dict[str, Any]:
    return {
        "schema": "mmfm-conditions-v1",
        "texts": {
            "text_a": {"entities": ["crystal"], "attributes": ["amber", "soft"]},
            "text_b": {"entities": ["fox"], "attributes": ["cobalt", "glowing"]},
            "text_c": {"entities": ["koi", "orbit"], "attributes": ["rippled", "faceted"]},
        },
        "images": {
            "img_a": "images/geometry_a.png",
            "img_b": "images/geometry_b.png",
            "img_shared": "images/geometry_shared.png",
        },
        "motions": {
            "mot_a": "motion/motion_a.json",
            "mot_b": "motion/motion_b.json",
            "mot_c": "motion/motion_c.json",
        },
        "frames": [
            {
                "frame": 1,
                "text": "text_a",
                "image": "img_a",
                "motion": "mot_a",
                "guidance_permille": 1000,
                "seed_offset": 1009,
            },
            {
                "frame": 2,
                "text": "text_b",
                "image": "img_shared",
                "motion": "mot_b",
                "guidance_permille": 1200,
                "seed_offset": 2018,
            },
            {
                "frame": 3,
                "text": "text_c",
                "image": "img_shared",
                "motion": "mot_c",
                "guidance_permille": 800,
                "seed_offset": 3027,
            },
        ],
    }


def edit_streams() -> dict[str, dict[str, Any]]:
    return {
        "selective_image": {
            "commands": [
                {
                    "op": "set_image",
                    "id": "img_shared",
                    "path": "images/geometry_shared_replacement.png",
                }
            ]
        },
        "fusion_weight": {
            "commands": [
                {"op": "set_fusion_weight", "condition_type": "image", "value_permille": 1250}
            ]
        },
        "chunk_rows": {"commands": [{"op": "set_chunk_rows", "value": 3}]},
        "denoise_steps": {"commands": [{"op": "set_denoise_steps", "value": 4}]},
        "quant_profile": {"commands": [{"op": "set_quant_profile", "value": "q12"}]},
        "exchange_v1": {"commands": [{"op": "set_exchange_version", "value": "v1"}]},
        "budget_zero": {"commands": [{"op": "set_budget", "value": 0}]},
    }


def exchange_sample() -> dict[str, Any]:
    return {
        "schema": "mmfm-stage-v2",
        "stage": "denoise",
        "provenance": {"sample": "local-deterministic", "version": "v2"},
        "records": [
            {
                "stage": "denoise",
                "frame": 1,
                "step": 1,
                "guidance": 1.2345,
                "noise_scheme": "lcg",
                "quant_profile": "q8",
                "refs": ["condition:1:text", "fusion:1"],
                "digest": "sample-digest-frame-1-step-1",
                "source_version": "v2",
                "missing_fields": [],
                "precision_labels": {"source": "v2-json"},
                "metadata": {"sample": True},
            },
            {
                "stage": "denoise",
                "frame": 2,
                "step": 1,
                "guidance": 0.8004,
                "noise_scheme": "counter",
                "quant_profile": "q12",
                "refs": ["condition:2:image", "fusion:2"],
                "digest": "sample-digest-frame-2-step-1",
                "source_version": "v2",
                "missing_fields": [],
                "precision_labels": {"source": "v2-json"},
                "metadata": {"sample": True},
            },
        ],
    }


def generate_samples(root: str | Path) -> dict[str, Any]:
    base = Path(root)
    configs = base / "configs"
    samples = base / "samples"
    images = samples / "images"
    motion = samples / "motion"
    edits = samples / "edits"
    exchange = samples / "exchange"
    for directory in (configs, samples, images, motion, edits, exchange, base / "snapshots", base / "reports"):
        directory.mkdir(parents=True, exist_ok=True)

    config = render_config()
    vocab = vocabulary()
    manifest = conditions_manifest()
    atomic_write_json(configs / "vocab.json", vocab)
    atomic_write_json(configs / "render.json", config)
    try:
        import yaml  # type: ignore

        atomic_write_text(
            configs / "render.yaml",
            yaml.safe_dump(config, sort_keys=True, allow_unicode=True, default_flow_style=False),
        )
        yaml_written = True
    except ImportError:
        yaml_written = False

    image_specs = {
        "geometry_a.png": 4101,
        "geometry_b.png": 4202,
        "geometry_shared.png": 4303,
        "geometry_shared_replacement.png": 4404,
    }
    for name, seed in image_specs.items():
        pixels = _draw_image(seed, width=config["image"]["width"], height=config["image"]["height"])
        atomic_write_bytes(images / name, encode_png_rgb(pixels))

    for name, seed in {"motion_a.json": 5101, "motion_b.json": 5202, "motion_c.json": 5303}.items():
        atomic_write_json(
            motion / name,
            _motion_table(
                seed,
                width=config["image"]["width"],
                height=config["image"]["height"],
                block_size=config["motion"]["block_size"],
            ),
        )
    atomic_write_json(samples / "conditions.json", manifest)
    for name, stream in edit_streams().items():
        atomic_write_json(edits / f"{name}.json", stream)
    sample_v2 = exchange_sample()
    atomic_write_json(exchange / "denoise_v2.json", sample_v2)
    profile_bits = {
        name: definition["fraction_bits"]
        for name, definition in config["quant_profiles"].items()
    }
    sample_v1, sidecar = v2_to_v1(
        sample_v2,
        noise_schemes=config["noise_schemes"],
        profiles=profile_bits,
        allow_missing=False,
    )
    atomic_write_text(exchange / "denoise_v1.txt", sample_v1)
    atomic_write_json(exchange / "denoise_v1.sidecar.json", sidecar)
    restored_v2 = v1_to_v2(
        sample_v1,
        sidecar=sidecar,
        noise_code_map=config["noise_schemes"],
        profile_bits=profile_bits,
    )
    atomic_write_json(exchange / "denoise_v1_roundtrip_v2.json", restored_v2)
    return {
        "config": str(configs / "render.json"),
        "config_yaml": str(configs / "render.yaml") if yaml_written else None,
        "vocab": str(configs / "vocab.json"),
        "conditions": str(samples / "conditions.json"),
        "images": sorted(str(path) for path in images.glob("*.png")),
        "motions": sorted(str(path) for path in motion.glob("*.json")),
        "edits": sorted(str(path) for path in edits.glob("*.json")),
        "exchange_v2": str(exchange / "denoise_v2.json"),
        "exchange_v1": str(exchange / "denoise_v1.txt"),
        "exchange_v1_sidecar": str(exchange / "denoise_v1.sidecar.json"),
        "exchange_v1_roundtrip_v2": str(exchange / "denoise_v1_roundtrip_v2.json"),
        "manifest_digest": __import__("hashlib").sha256(canonical_json(manifest).encode("utf-8")).hexdigest(),
    }
