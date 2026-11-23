"""Generate deterministic local sample data for the diagnostics workbench."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
DIMENSION = 32


def load_vectors(rng: np.random.Generator, count: int) -> list[list[float]]:
    vectors: list[list[float]] = []
    for _ in range(count):
        vector = rng.standard_normal(DIMENSION) * 0.9
        # Deliberate large components exercise clipping in coarse quantization.
        if rng.random() < 0.16:
            vector[int(rng.integers(0, DIMENSION))] = 4.25
        if rng.random() < 0.12:
            vector[int(rng.integers(0, DIMENSION))] = -3.75
        vectors.append([round(float(value), 12) for value in vector])
    return vectors


def make_sample(
    rng: np.random.Generator,
    index: int,
    *,
    query_position: int,
    distances: list[int],
    sample_id: str | None = None,
) -> dict:
    key_positions: list[int] = []
    for distance in distances:
        if query_position - distance >= 0 and (index + distance) % 3 != 0:
            key_positions.append(query_position - distance)
        elif query_position + distance < 256:
            key_positions.append(query_position + distance)
        else:
            key_positions.append(max(0, query_position - distance))
    count = 1 + len(key_positions)
    vectors = load_vectors(rng, count)
    return {
        "sample_id": sample_id or f"s{index + 1:04d}",
        "query_position": query_position,
        "key_positions": key_positions,
        "query_vector": vectors[0],
        "key_vectors": vectors[1:],
    }


def generate_main_samples() -> list[dict]:
    rng = np.random.default_rng(20261124)
    rows: list[dict] = []
    for index in range(48):
        query_position = 66 + (index * 19) % 164
        key_count = 3 + index % 5
        if index % 8 == 0:
            boundary_distances = [62, 63, 64, 65, 66, 95, 96, 127]
            distances = boundary_distances[:key_count]
        else:
            distances = [2 + (index * 7 + key * 13) % 118 for key in range(key_count)]
            if index % 3 == 0:
                distances[0] = 63
                if len(distances) > 1:
                    distances[1] = 64
        rows.append(
            make_sample(
                rng,
                index,
                query_position=query_position,
                distances=distances,
            )
        )
    return rows


def write_jsonl(path: Path, rows: list[dict] | list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            if isinstance(row, str):
                handle.write(row + "\n")
            else:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    config = {
        "scheme": "rope",
        "base_frequency": 10000.0,
        "train_context_length": 64,
        "extrapolation_target_length": 256,
        "dimension": DIMENSION,
        "scale_factor": 2.0,
        "precision_level": "fp32",
        "precision_levels": ["fp16", "fp32", "fp64"],
        "dimension_block_size": 8,
        "fixed_decimal_bits": 10,
        "fixed_decimal_widths": [6, 8, 10, 12, 14],
        "quantization_level": "balanced",
        "quantization_levels": ["fine", "balanced", "coarse"],
        "tolerances": {
            "position_vector": 0.0005,
            "dot_score": 0.002,
            "attention_weight": 0.0005,
        },
        "tolerance_candidates": {
            "position_vector": [0.0001, 0.0005, 0.002, 0.01],
            "dot_score": [0.0005, 0.002, 0.01, 0.05],
            "attention_weight": [0.0001, 0.0005, 0.002, 0.01],
        },
        "scheme_candidates": ["rope", "linear", "ntk"],
        "thread_pool_size": 4,
        "point_budget": 5000,
        "sample_quota": 32,
        "snapshot_interval": 12,
    }
    (SAMPLES / "position_encoding.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    rows = generate_main_samples()
    write_jsonl(SAMPLES / "attention_samples.jsonl", rows)
    write_jsonl(SAMPLES / "budget_attention_samples.jsonl", rows[:5] + [rows[10], rows[20]])

    commands = [
        {"command": "run", "params": {}},
        {"command": "set", "patch": {"scheme": "linear"}},
        {"command": "run", "params": {}},
        {"command": "set", "patch": {"scheme": "ntk"}},
        {"command": "run", "params": {}},
        {"command": "set", "patch": {"quantization_level": "coarse"}},
        {"command": "run", "params": {}},
        {"command": "set", "patch": {"quantization_level": "fine", "fixed_decimal_bits": 6}},
        {"command": "run", "params": {}},
        {
            "command": "sensitivity",
            "params": {
                "scheme_candidates": ["rope", "linear", "ntk"],
                "precision_levels": ["fp16", "fp32", "fp64"],
                "quantization_levels": ["fine", "balanced", "coarse"],
                "fixed_decimal_widths": [6, 8, 10, 12, 14],
                "data_scales": [0.25, 0.5, 0.75, 1.0],
            },
        },
        {"command": "report", "params": {}},
    ]
    write_jsonl(SAMPLES / "commands.jsonl", commands)

    budget_commands = [
        {"command": "set", "patch": {"scheme": "rope", "precision_level": "fp32", "quantization_level": "balanced"}},
        {"command": "run", "params": {"point_budget": 7, "sample_quota": 32}},
        {"command": "run", "params": {"point_budget": 7, "sample_quota": 32}},
        {"command": "run", "params": {"point_budget": 500, "sample_quota": 3}},
        {"command": "run", "params": {"point_budget": 500, "sample_quota": 32}},
        {"command": "report", "params": {}},
    ]
    write_jsonl(SAMPLES / "budget_commands.jsonl", budget_commands)

    invalid_rows = [
        json.dumps(rows[0], ensure_ascii=False, separators=(",", ":")),
        json.dumps({**rows[1], "sample_id": "s0001"}, ensure_ascii=False, separators=(",", ":")),
        json.dumps({**rows[2], "query_position": -1}, ensure_ascii=False, separators=(",", ":")),
        json.dumps(
            {
                **rows[3],
                "key_positions": [10, -2],
                "key_vectors": rows[3]["key_vectors"][:2],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).replace("]", "]", 1),
        '{"sample_id":"s9001","query_position":3,"key_positions":[0],"query_vector":[NaN],"key_vectors":[[0.0]]}',
        json.dumps(
            {
                **rows[4],
                "query_vector": rows[4]["query_vector"][:-1],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "not-json",
        json.dumps(rows[5], ensure_ascii=False, separators=(",", ":")),
    ]
    write_jsonl(SAMPLES / "invalid_samples.jsonl", invalid_rows)

    invalid_commands = [
        '{"command":"run","params":{}}',
        '{"command":"set","patch":{"dimension":33}}',
        '{"command":"unknown"}',
        "not-json",
        '{"command":"run","params":{"point_budget":0}}',
    ]
    write_jsonl(SAMPLES / "invalid_commands.jsonl", invalid_commands)


if __name__ == "__main__":
    main()
