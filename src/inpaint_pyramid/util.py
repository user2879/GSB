"""Small deterministic utility functions."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def array_fingerprint(array: np.ndarray) -> str:
    """Fingerprint shape, dtype and C-order bytes; independent of object identity."""
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(f"{arr.shape}|{arr.dtype.str}|C|".encode("ascii"))
    digest.update(arr.tobytes(order="C"))
    return digest.hexdigest()


def bool_fingerprint(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array.astype(np.uint8, copy=False))
    digest = hashlib.sha256()
    digest.update(f"{arr.shape}|bool|C|".encode("ascii"))
    digest.update(arr.tobytes(order="C"))
    return digest.hexdigest()


def stable_digest_parts(*parts: Any) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(canonical_json(part).encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def tile_sort_key(key: tuple[int, int, int]) -> tuple[int, int, int]:
    return key


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def jsonable_float(value: float) -> float:
    return float(round(float(value), 12))


def clip_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def chunks(values: Iterable[Any], size: int) -> Iterable[list[Any]]:
    chunk: list[Any] = []
    for value in values:
        chunk.append(value)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

