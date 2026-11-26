"""Canonical JSON, hashing, numeric snapshots and atomic persistence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_json(value: Any) -> str:
    return digest_bytes(canonical_json(value).encode("utf-8"))


def digest_text(value: str) -> str:
    return digest_bytes(value.encode("utf-8"))


def digest_array(array: Any) -> str:
    arr = np.asarray(array, dtype="<f8")
    header = canonical_json({"dtype": "float64-le", "shape": list(arr.shape)})
    h = hashlib.sha256(header.encode("ascii"))
    h.update(b"\0")
    h.update(np.ascontiguousarray(arr).tobytes(order="C"))
    return h.hexdigest()


def digest_int_array(array: Any) -> str:
    arr = np.asarray(array, dtype="<i8")
    header = canonical_json({"dtype": "int64-le", "shape": list(arr.shape)})
    h = hashlib.sha256(header.encode("ascii"))
    h.update(b"\0")
    h.update(np.ascontiguousarray(arr).tobytes(order="C"))
    return h.hexdigest()


def normalize_float(value: float, digits: int = 12) -> float:
    if not math.isfinite(value):
        raise ValueError("non-finite value cannot be persisted")
    rounded = round(float(value), digits)
    return 0.0 if rounded == 0 else rounded


def rounded_list(array: Any, digits: int = 12) -> list[Any]:
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim == 0:
        return normalize_float(float(arr), digits)  # type: ignore[return-value]
    return [rounded_list(row, digits) for row in arr]


def assert_finite(array: Any, *, label: str, code: str) -> None:
    from .errors import QLabError

    arr = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise QLabError(code, f"{label} contains non-finite values", label=label)


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_json(path: str | Path, value: Any) -> None:
    payload = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    atomic_write_bytes(path, payload)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def monotonic_ms() -> int:
    return time.monotonic_ns() // 1_000_000


def monotonic_seconds() -> float:
    return time.monotonic()


def stable_sorted(items: Iterable[Any], key) -> list[Any]:
    return sorted(items, key=key)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

