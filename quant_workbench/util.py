"""Deterministic serialization, hashing and numeric helpers."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _normalize_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_normalize_json(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _normalize_json(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float cannot be canonicalized")
        # Avoid the two JSON spellings of zero in fingerprints.
        return 0.0 if value == 0.0 else value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        _normalize_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def fingerprint(value: Any) -> str:
    return sha256_text(canonical_json(value))


def stage_cache_key(stage: str, *parts: Any) -> str:
    return fingerprint({"stage": stage, "parts": list(parts)})


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def round_significant(value: float, digits: int) -> float:
    """Round a float64 scale to a fixed number of significant decimal digits.

    Formatting in scientific notation avoids intermediate ``10**exponent``
    overflow for subnormal finite inputs.
    """

    value = float(value)
    if value == 0.0 or not math.isfinite(value):
        return 0.0 if value == 0.0 else value
    rounded = float(f"{value:.{max(0, digits - 1)}e}")
    return 0.0 if rounded == 0.0 else rounded


def round_half_away(values: np.ndarray) -> np.ndarray:
    """Vectorized deterministic rounding (ties away from zero)."""

    values = np.asarray(values, dtype=np.float64)
    return np.copysign(np.floor(np.abs(values) + 0.5), values)


def clean_float(value: float) -> float:
    value = float(value)
    if value == 0.0:
        return 0.0
    return value


def clean_list(values: Any) -> list[Any]:
    return [_normalize_json(item) for item in np.asarray(values).tolist()]


def ensure_parent(path: str | Path) -> Path:
    result = Path(path)
    result.parent.mkdir(parents=True, exist_ok=True)
    return result


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write a file through a sibling temporary file and os.replace."""

    import os
    import tempfile

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, destination)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
