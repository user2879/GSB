from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw

from .config import load_config
from .resources import (
    load_command_stream,
    load_image,
    load_injections,
    load_motion_table,
    load_prompt_set,
)
from .util import atomic_write_bytes, atomic_write_json


def _draw_sample_image(path: Path, seed: int, palette: tuple[tuple[int, int, int], ...]) -> None:
    rng = np.random.default_rng(seed)
    width = height = 32
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, :, :] = np.asarray(palette[0], dtype=np.uint8)
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 15, 31), fill=palette[1])
    draw.rectangle((16, 0, 31, 31), fill=palette[2])
    draw.ellipse((4, 4, 18, 18), outline=palette[3], width=2)
    draw.rectangle((17, 15, 29, 27), outline=palette[3], width=2)
    draw.line((0, 30, 31, 1), fill=palette[4], width=1)
    for block_y in range(0, 32, 4):
        for block_x in range(0, 32, 4):
            tint = rng.integers(-18, 19, size=3, dtype=np.int64)
            block = np.asarray(image.crop((block_x, block_y, block_x + 4, block_y + 4)), dtype=np.int64)
            block = np.clip(block + tint.reshape(1, 1, 3), 0, 255).astype(np.uint8)
            image.paste(Image.fromarray(block), (block_x, block_y))
    atomic_write_bytes(path, _png_bytes(image))


def _png_bytes(image: Image.Image) -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG", compress_level=6, optimize=False)
    return buffer.getvalue()


def _sample_motion(seed: int, frame_count: int, rows: int, cols: int, delta: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.integers(-3, 4, size=(frame_count, rows, cols, 2), dtype=np.int32)
    if delta:
        values = np.clip(values + rng.integers(-delta, delta + 1, size=values.shape), -6, 6).astype(np.int32)
    return values


def generate_sample_workspace(root: str | Path) -> dict[str, str]:
    target = Path(root).resolve()
    target.mkdir(parents=True, exist_ok=True)

    config: dict[str, Any] = {
        "version": 1,
        "frame": {"width": 32, "height": 32, "channels": 3, "count": 3},
        "latent": {"height": 8, "width": 8, "channels": 4},
        "condition": {
            "dim": 8,
            "frac_bits": 8,
            "allowed_frac_bits": [4, 8, 12],
            "quantization_level": 2,
            "allowed_quantization_levels": [1, 2, 4],
        },
        "denoise": {
            "steps": 5,
            "seed": 20261127,
            "noise_seed": 7788,
            "guidance": 1.25,
            "batch_point_budget": 3,
        },
        "render": {
            "block_rows": 4,
            "max_concurrency": 4,
            "interaction_latency_budget_ms": 250,
        },
        "image": {"block_size": 8},
        "motion": {"block_width": 8, "block_height": 8},
        "fusion": {"weights": {"text": 1.0, "image": 0.5, "motion": 0.75}},
        "vocabulary": [
            {"term": "circle", "kind": "entity"},
            {"term": "square", "kind": "entity"},
            {"term": "triangle", "kind": "entity"},
            {"term": "ring", "kind": "entity"},
            {"term": "red", "kind": "attribute"},
            {"term": "blue", "kind": "attribute"},
            {"term": "green", "kind": "attribute"},
            {"term": "matte", "kind": "attribute"},
            {"term": "gloss", "kind": "attribute"},
            {"term": "small", "kind": "attribute"},
            {"term": "large", "kind": "attribute"},
            {"term": "dense", "kind": "attribute"},
        ],
        "resources": {
            "prompts": "prompts.json",
            "image": "image.png",
            "motion": "motion.npy",
        },
        "cache": {"snapshot_version": 1},
    }
    config_path = target / "config.yaml"
    atomic_write_bytes(
        config_path,
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True).encode("utf-8"),
    )

    prompts = {
        "prompts": {
            "p0": {"entities": ["circle"], "attributes": ["red", "matte"]},
            "p1": {"entities": ["triangle"], "attributes": ["blue", "gloss", "large"]},
            "p2": {"entities": ["ring"], "attributes": ["green", "small", "dense"]},
        }
    }
    atomic_write_json(target / "prompts.json", prompts)

    _draw_sample_image(
        target / "image.png",
        91001,
        ((24, 34, 48), (154, 48, 58), (35, 84, 146), (238, 214, 140), (230, 92, 48)),
    )
    _draw_sample_image(
        target / "image_alt.png",
        91002,
        ((36, 20, 58), (48, 150, 96), (206, 92, 34), (96, 224, 210), (245, 228, 92)),
    )
    motion_base = _sample_motion(81001, 3, 4, 4)
    motion_alt = _sample_motion(81002, 3, 4, 4, delta=2)
    np.save(target / "motion.npy", motion_base, allow_pickle=False)
    np.save(target / "motion_alt.npy", motion_alt, allow_pickle=False)

    commands = [
        {"op": "set_prompt", "prompt": "p1"},
        {"op": "set_image", "path": "image_alt.png"},
        {"op": "set_motion", "path": "motion_alt.npy"},
        {"op": "set_weights", "weights": {"text": 1.5, "image": 0.25, "motion": 0.5}},
        {"op": "set_denoise_steps", "steps": 6},
        {"op": "set_block_rows", "rows": 2},
        {"op": "set_quant_level", "level": 4},
        {"op": "delete_image"},
        {"op": "set_prompt", "prompt": "p2"},
        {"op": "set_guidance", "guidance": 1.75},
        {"op": "set_seed", "seed": 20261225},
        {"op": "set_image", "path": "image.png"},
        {"op": "set_block_rows", "rows": 4},
    ]
    command_text = "\n".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) for item in commands) + "\n"
    atomic_write_bytes(target / "commands.jsonl", command_text.encode("utf-8"))

    injections = [
        {
            "id": "INJ-DIM",
            "stage": "fusion",
            "kind": "dimension_mismatch",
            "target": {"frame": 1},
        },
        {
            "id": "INJ-NONFINITE",
            "stage": "denoise",
            "kind": "nonfinite_latent",
            "target": {"frame": 1, "step": 2},
        },
        {
            "id": "INJ-ROW-OOB",
            "stage": "decode_row",
            "kind": "block_row_oob",
            "target": {"frame": 0, "row": 2},
        },
        {
            "id": "INJ-QUANT-OOB",
            "stage": "quantization",
            "kind": "quant_level_oob",
            "value": 9,
        },
        {
            "id": "INJ-STALE-SNAP",
            "stage": "snapshot",
            "kind": "stale_snapshot_version",
            "value": 999,
        },
    ]
    atomic_write_json(target / "injections.json", injections)
    return {
        "root": str(target),
        "config": str(config_path),
        "prompts": str(target / "prompts.json"),
        "image": str(target / "image.png"),
        "image_alt": str(target / "image_alt.png"),
        "motion": str(target / "motion.npy"),
        "motion_alt": str(target / "motion_alt.npy"),
        "commands": str(target / "commands.jsonl"),
        "injections": str(target / "injections.json"),
    }


def validate_sample_workspace(config_path: str | Path, commands_path: str | Path | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    prompt_set = load_prompt_set(config.resources.prompts, config)
    image = None if config.resources.image is None else load_image(config.resources.image, config)
    motion = load_motion_table(config.resources.motion, config)
    commands = [] if commands_path is None else load_command_stream(commands_path)
    return {
        "ok": True,
        "config_fingerprint": config.config_fingerprint,
        "vocabulary_size": len(config.vocabulary),
        "prompt_count": len(prompt_set.prompts),
        "image": None if image is None else {"size": [image.width, image.height], "mode": image.mode},
        "motion_shape": list(motion.shape),
        "commands": len(commands),
    }
