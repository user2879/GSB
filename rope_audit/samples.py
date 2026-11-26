from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import is_real_number
from .errors import ConfigInvalid


def validate_sample(obj: Any, dimension: int, line: int | None = None) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ConfigInvalid("sample must be a JSON object", line=line, field="$")

    sample_id = obj.get("id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ConfigInvalid("sample id must be a non-empty string", line=line, field="id")

    query_position = obj.get("query_position")
    if not isinstance(query_position, int) or isinstance(query_position, bool) or query_position < 0:
        raise ConfigInvalid("query_position must be a non-negative integer", line=line, field="query_position", sample_id=sample_id)

    train_length = obj.get("train_length")
    if not isinstance(train_length, int) or isinstance(train_length, bool) or train_length <= 0:
        raise ConfigInvalid("train_length must be a positive integer marker", line=line, field="train_length", sample_id=sample_id)

    key_positions = obj.get("key_positions")
    if not isinstance(key_positions, list) or not key_positions:
        raise ConfigInvalid("key_positions must be a non-empty array", line=line, field="key_positions", sample_id=sample_id)
    for index, position in enumerate(key_positions):
        if not isinstance(position, int) or isinstance(position, bool) or position < 0:
            raise ConfigInvalid(
                "key_positions must contain non-negative integers",
                line=line,
                field=f"key_positions[{index}]",
                sample_id=sample_id,
            )
    if len(set(key_positions)) != len(key_positions):
        raise ConfigInvalid("key_positions must be unique", line=line, field="key_positions", sample_id=sample_id)
    if any(int(key_positions[index]) >= int(key_positions[index + 1]) for index in range(len(key_positions) - 1)):
        raise ConfigInvalid(
            "key_positions must be strictly ascending",
            line=line,
            field="key_positions",
            sample_id=sample_id,
        )

    query_vector = obj.get("query_vector")
    if not isinstance(query_vector, list) or len(query_vector) != dimension:
        actual = len(query_vector) if isinstance(query_vector, list) else "not-array"
        raise ConfigInvalid(
            f"query_vector length must equal dimension {dimension}, got {actual}",
            line=line,
            field="query_vector",
            sample_id=sample_id,
        )

    def validate_vector(vector: Any, field: str) -> list[float]:
        if not isinstance(vector, list) or len(vector) != dimension:
            actual = len(vector) if isinstance(vector, list) else "not-array"
            raise ConfigInvalid(
                f"{field} length must equal dimension {dimension}, got {actual}",
                line=line,
                field=field,
                sample_id=sample_id,
            )
        normalized: list[float] = []
        for index, value in enumerate(vector):
            if not is_real_number(value):
                raise ConfigInvalid(
                    f"{field}[{index}] must be a finite number",
                    line=line,
                    field=f"{field}[{index}]",
                    sample_id=sample_id,
                )
            normalized.append(float(value))
        return normalized

    query = validate_vector(query_vector, "query_vector")
    key_vectors_raw = obj.get("key_vectors")
    if not isinstance(key_vectors_raw, list) or len(key_vectors_raw) != len(key_positions):
        raise ConfigInvalid(
            "key_vectors must have the same length as key_positions",
            line=line,
            field="key_vectors",
            sample_id=sample_id,
        )
    key_vectors = [validate_vector(vector, f"key_vectors[{index}]") for index, vector in enumerate(key_vectors_raw)]

    seed = obj.get("seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
        raise ConfigInvalid("seed must be a non-negative integer when provided", line=line, field="seed", sample_id=sample_id)

    return {
        "id": sample_id,
        "seed": int(seed) if seed is not None else None,
        "query_position": int(query_position),
        "train_length": int(train_length),
        "key_positions": [int(position) for position in key_positions],
        "query_vector": query,
        "key_vectors": key_vectors,
    }


def load_samples(path: str | Path, dimension: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    seen: set[str] = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ConfigInvalid(
                    f"invalid sample JSON: {exc.msg}",
                    line=line_number,
                    field="$",
                ) from exc
            sample = validate_sample(obj, dimension, line_number)
            if sample["id"] in seen:
                raise ConfigInvalid(
                    f"duplicate sample id {sample['id']!r}",
                    line=line_number,
                    field="id",
                    sample_id=sample["id"],
                )
            seen.add(sample["id"])
            samples.append(sample)
    return sorted(samples, key=lambda item: item["id"])


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def generate_samples(config: dict[str, Any], count: int = 6) -> list[dict[str, Any]]:
    dimension = int(config["dimension"])
    train_length = int(config["train_length"])
    root_seed = int(config["seed"])
    distances = [1, 4, 8, 13, 19, 26]
    samples: list[dict[str, Any]] = []
    for index in range(count):
        seed = root_seed + 1009 * (index + 1)
        rng = np.random.default_rng(seed)
        query_position = 32 + 8 * index
        key_positions = [query_position - distance for distance in reversed(distances)]
        query_vector = np.round(rng.normal(0.0, 0.38, dimension), 6).astype(np.float64)
        key_vectors = np.round(rng.normal(0.0, 0.38, (len(distances), dimension)), 6).astype(np.float64)
        sample = {
            "id": f"s{index + 1:03d}",
            "seed": seed,
            "query_position": int(query_position),
            "train_length": train_length,
            "key_positions": [int(value) for value in key_positions],
            "query_vector": [float(value) for value in query_vector],
            "key_vectors": [[float(value) for value in vector] for vector in key_vectors],
        }
        samples.append(validate_sample(sample, dimension))
    return sorted(samples, key=lambda item: item["id"])
