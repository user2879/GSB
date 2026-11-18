from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable

import yaml


DEFAULT_CASE: dict[str, Any] = {
    "schema_version": 1,
    "project": {"name": "test-case"},
    "prompt": {"samples_path": "prompts.yaml", "embedding_dim": 4},
    "render": {
        "denoise_steps": 3,
        "guidance_scale": 2.0,
        "resolution_tiers": [
            {"name": "tiny", "width": 8, "height": 8},
            {"name": "small", "width": 16, "height": 16},
        ],
        "batch_sizes": [1, 2, 4],
        "latent_block_size": 64,
        "cache_budget_blocks": 4,
        "seed_max": 1000,
        "noise_table": {"values": [0.17, 0.23, 0.31], "expected_length": 3},
        "guidance_table": {"values": [0.42, 0.37, 0.29], "expected_length": 3},
    },
    "img2img": {
        "enabled": True,
        "replace_early_steps": 1,
        "initial_image_path": "initial.ppm",
    },
    "runtime": {
        "temp_dir": "runtime",
        "concurrency": 4,
        "preview_budget_ms": 10,
        "snapshot_interval": 1,
    },
    "strategy": {"version": "strategy-table-v1"},
}


SAMPLE: dict[str, Any] = {
    "schema_version": 1,
    "embedding_dim": 4,
    "vocab": ["amber", "arch", "cloud", "forest", "glass", "river"],
    "seeds": [11, 13],
    "prompts": [
        {"id": "p0", "tokens": ["amber", "cloud"]},
        {"id": "p1", "tokens": ["forest", "river"]},
    ],
}


def write_ppm(path: Path, width: int = 16, height: int = 16) -> None:
    body = bytearray()
    for row in range(height):
        for column in range(width):
            body.extend(((row * 11) % 256, (column * 17) % 256, ((row + column) * 9) % 256))
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + bytes(body))


def write_case(
    root: Path,
    mutate_config: Callable[[dict[str, Any]], None] | None = None,
    mutate_sample: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(DEFAULT_CASE)
    sample = copy.deepcopy(SAMPLE)
    if mutate_config is not None:
        mutate_config(config)
    if mutate_sample is not None:
        mutate_sample(sample)
    write_ppm(root / "initial.ppm")
    (root / "prompts.yaml").write_text(yaml.safe_dump(sample, sort_keys=False), encoding="utf-8")
    config_path = root / "workbench.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path
