"""Shared deterministic serialization and sample-generation helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def array_digest(values: np.ndarray) -> str:
    array = np.asarray(values, dtype="<f4")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def stable_seed(seed: int, *parts: Any) -> int:
    material = "|".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def atomic_write_json(path: str | Path, value: Any, *, mode: int = 0o600) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, mode)
    os.replace(temporary, destination)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def generate_sample_payload(
    *,
    seed: int = 1729,
    count: int = 12,
    dim: int = 8,
    prefix: str = "s",
) -> dict[str, Any]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
        raise ValueError("dim must be a positive integer")
    rng = np.random.default_rng(seed)
    matrix = np.round(rng.standard_normal((count, dim)), 8).astype("<f4")
    return {
        "format_version": 1,
        "seed": seed,
        "dim": dim,
        "count": count,
        "samples": [
            {"id": f"{prefix}{index:04d}", "values": matrix[index].tolist()}
            for index in range(count)
        ],
    }
