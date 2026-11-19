"""Deterministic JSON serialization and hashing helpers."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

FLOAT_DECIMALS = 12
EPS = 1.0e-12


def _clean_float(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"Non-finite float cannot be serialized: {value!r}")
    value = round(float(value), FLOAT_DECIMALS)
    if abs(value) < 10.0 ** (-FLOAT_DECIMALS):
        return 0.0
    return value


def to_builtin(value: Any) -> Any:
    """Recursively convert NumPy values and normalize floating-point output."""

    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_builtin(value.tolist())
    if isinstance(value, (np.floating, float)):
        return _clean_float(float(value))
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        to_builtin(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def content_hash(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def file_hash(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def pretty_json(value: Any) -> str:
    return json.dumps(
        to_builtin(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ) + "\n"


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(pretty_json(value).encode("utf-8"))


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def with_content_hash(record: dict[str, Any]) -> dict[str, Any]:
    """Return a copy carrying a hash of every other field."""

    payload = {key: value for key, value in record.items() if key != "content_hash"}
    result = dict(payload)
    result["content_hash"] = content_hash(payload)
    return result
