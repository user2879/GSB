from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .config import default_config

SEED = 20261127


def make_sample(index: int, seed: int = SEED) -> dict[str, Any]:
    rng = random.Random(seed + index * 1000003)
    dimension = 8
    train_length = 16
    target_length = 64
    sample_id = f"s{index:03d}"
    query_position = 18 + (index * 5) % 28
    heavy = index == 12
    key_count = 12 if heavy else 5 + (index % 4)

    positions = {0, min(target_length - 1, query_position + 18)}
    for offset in range(1, train_length - 2):
        for candidate in (query_position - offset, query_position + offset):
            if 0 <= candidate < target_length and len(positions) < key_count:
                positions.add(candidate)
    while len(positions) < key_count:
        positions.add(rng.randrange(target_length))
    key_positions = sorted(positions)[:key_count]

    def tiny_vector() -> list[float]:
        return [rng.randint(-4, 4) / 4096.0 for _ in range(dimension)]

    return {
        "id": sample_id,
        "dimension": dimension,
        "query_position": query_position,
        "train_length": train_length,
        "target_length": target_length,
        "query_vector": tiny_vector(),
        "key_positions": key_positions,
        "key_vectors": [tiny_vector() for _ in key_positions],
    }


def invalid_sample_lines() -> list[str]:
    duplicate = {
        "id": "s001",
        "dimension": 8,
        "query_position": 3,
        "train_length": 16,
        "target_length": 64,
        "query_vector": [0.1] * 8,
        "key_positions": [1, 2],
        "key_vectors": [[0.1] * 8, [0.2] * 8],
    }
    negative_position = {
        "id": "bad-negative-qpos",
        "dimension": 8,
        "query_position": -2,
        "train_length": 16,
        "target_length": 64,
        "query_vector": [0.1] * 8,
        "key_positions": [1, 2],
        "key_vectors": [[0.1] * 8, [0.2] * 8],
    }
    duplicate_keys = {
        "id": "bad-duplicate-keys",
        "dimension": 8,
        "query_position": 9,
        "train_length": 16,
        "target_length": 64,
        "query_vector": [0.1] * 8,
        "key_positions": [4, 4, 8],
        "key_vectors": [[0.1] * 8, [0.2] * 8, [0.3] * 8],
    }
    wrong_vector = {
        "id": "bad-vector-length",
        "dimension": 8,
        "query_position": 9,
        "train_length": 16,
        "target_length": 64,
        "query_vector": [0.1, float("nan")],
        "key_positions": [4],
        "key_vectors": [[0.1] * 8],
    }
    length_order = {
        "id": "bad-length-order",
        "dimension": 7,
        "query_position": 9,
        "train_length": 80,
        "target_length": 64,
        "query_vector": [0.1] * 7,
        "key_positions": [4],
        "key_vectors": [[0.1] * 7],
    }
    return [
        '{"id":"bad-json", "query_position": ',
        json.dumps(duplicate, sort_keys=True),
        json.dumps(negative_position, sort_keys=True),
        json.dumps(duplicate_keys, sort_keys=True),
        json.dumps(wrong_vector, sort_keys=True),
        json.dumps(length_order, sort_keys=True),
    ]


def build_command_stream(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    replacement = json.loads(json.dumps(next(item for item in samples if item["id"] == "s002")))
    replacement["query_vector"][0] = replacement["query_vector"][0] + 3.0 / 4096.0
    replacement["key_vectors"][0][1] = replacement["key_vectors"][0][1] - 2.0 / 4096.0
    extra_scheme = {
        "name": "octave_demo",
        "kind": "blockwave",
        "base_freq": 8000.0,
        "scale": 0.75,
        "block_size": 2,
        "enabled": True,
    }
    return [
        {"op": "set_budget", "batch_point_budget": 10**9, "sample_point_quota": 10**9},
        {"op": "run_batch"},
        {"op": "switch_scheme", "name": "linear_ramp"},
        {"op": "replace_sample", "sample": replacement},
        {"op": "delete_sample", "id": "s009"},
        {"op": "set_base_freq", "scheme": "sinusoidal", "base_freq": 5000.0},
        {"op": "set_scale", "scheme": "blockwave", "scale": 1.25},
        {"op": "set_block_size", "scheme": "sinusoidal", "block_size": 4},
        {"op": "run_batch"},
        {"op": "set_frac_bits", "frac_bits": 12},
        {"op": "set_quant_tier", "quant_tier": "Q16"},
        {"op": "run_batch"},
        {"op": "set_lengths", "train_length": 12, "target_length": 80},
        {"op": "run_batch"},
        {"op": "add_scheme", "scheme": extra_scheme},
        {"op": "remove_scheme", "name": "linear_ramp"},
        {"op": "set_workers", "workers": 2},
        {"op": "run_batch"},
    ]


def generate_project_files(root: str | Path = ".") -> dict[str, Path]:
    root = Path(root)
    for directory in ("samples", "configs", "commands", "state", "out", "scripts", "tests", "docs"):
        (root / directory).mkdir(parents=True, exist_ok=True)

    samples = [make_sample(index) for index in range(1, 13)]
    sample_path = root / "samples" / "attention_samples.jsonl"
    with sample_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, sort_keys=True) + "\n")
        for line in invalid_sample_lines():
            handle.write(line + "\n")

    config = default_config()
    config_path = root / "configs" / "demo_config.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    commands = build_command_stream(samples)
    command_path = root / "commands" / "demo_commands.jsonl"
    command_path.write_text(
        "\n".join(json.dumps(item, sort_keys=True) for item in commands) + "\n",
        encoding="utf-8",
    )
    return {"samples": sample_path, "config": config_path, "commands": command_path}
