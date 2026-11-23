from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

import numpy as np


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest_bytes(*chunks: bytes) -> str:
    hasher = hashlib.sha256()
    for chunk in chunks:
        hasher.update(len(chunk).to_bytes(8, "little"))
        hasher.update(chunk)
    return hasher.hexdigest()


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def array_bytes(array: np.ndarray, dtype: str | np.dtype | None = None) -> bytes:
    target = np.asarray(array, dtype=dtype) if dtype is not None else np.asarray(array)
    canonical = np.ascontiguousarray(target, dtype=target.dtype.newbyteorder("<"))
    return canonical.tobytes(order="C")


def fingerprint_arrays(*arrays: np.ndarray, prefix: str = "") -> str:
    chunks: list[bytes] = [prefix.encode("utf-8")] if prefix else []
    for array in arrays:
        normalized = np.ascontiguousarray(array)
        chunks.append(str(normalized.dtype.str).encode("ascii"))
        chunks.append(repr(tuple(normalized.shape)).encode("ascii"))
        chunks.append(array_bytes(normalized))
    return digest_bytes(*chunks)


def fingerprint_rows(rows: Iterable[np.ndarray]) -> str:
    chunks: list[bytes] = []
    for row in rows:
        normalized = np.ascontiguousarray(row, dtype="<f8")
        chunks.append(array_bytes(normalized))
    return digest_bytes(*chunks)

