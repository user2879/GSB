"""Canonical JSON and binary hashing helpers.

All persisted hashes use explicit little-endian dtypes and canonical JSON so a
ledger can be recomputed on Windows without hash-randomisation or host-endian
surprises.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes."""

    text = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return text.encode("utf-8")


def pretty_json_text(value: Any) -> str:
    """Return deterministic, human-readable JSON text."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )


def sha256_bytes(payload: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(bytes(payload)).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _normalise_dtype(dtype: str | np.dtype[Any]) -> str:
    array_dtype = np.dtype(dtype)
    if array_dtype.kind == "f":
        return "<f8" if array_dtype.itemsize > 4 else "<f4"
    if array_dtype.kind in {"u", "i", "b"}:
        size = max(1, array_dtype.itemsize)
        prefix = "u" if array_dtype.kind in {"u", "b"} else "i"
        return f"<{prefix}{size}"
    return array_dtype.str


def hash_array(array: np.ndarray, dtype: str | np.dtype[Any]) -> str:
    """Hash shape, explicit dtype and C-order bytes."""

    normalised = np.ascontiguousarray(np.asarray(array, dtype=dtype))
    dtype_spec = _normalise_dtype(normalised.dtype)
    header = f"vgcover-array-v1|{dtype_spec}|{','.join(map(str, normalised.shape))}".encode(
        "ascii"
    )
    return sha256_bytes(header + b"\0" + normalised.astype(dtype_spec, copy=False).tobytes())


def hash_float32(array: np.ndarray) -> str:
    return hash_array(array, "<f4")


def hash_uint8(array: np.ndarray) -> str:
    return hash_array(array, "<u1")


def hash_int32(array: np.ndarray) -> str:
    return hash_array(array, "<i4")


def hash_bytes_chunks(chunks: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

