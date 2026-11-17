"""Hashing, JSON normalization and atomic file helpers."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, is_dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np


def jsonable(value: Any) -> Any:
    """Return a JSON-safe value while preserving ordinary float precision."""
    if is_dataclass(value) and not isinstance(value, type):
        return jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_value(value: Any) -> str:
    import hashlib

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    """Write a JSON document through a same-directory temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def max_abs_diff(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return math.inf
    if not left:
        return 0.0
    return max(abs(float(a) - float(b)) for a, b in zip(left, right))


def stable_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    return float(value)


@dataclass(frozen=True)
class Timing:
    started_ns: int
    elapsed_ms: float


class Span:
    """Context manager recording monotonic elapsed milliseconds."""

    def __init__(self) -> None:
        self.elapsed_ms = 0.0
        self._start = 0

    def __enter__(self) -> "Span":
        import time

        self._start = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        import time

        self.elapsed_ms = (time.perf_counter_ns() - self._start) / 1_000_000.0
