"""Canonical, platform-stable content hashing helpers."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import numpy as np

_QUANT_DECIMALS = 12
_QUANT_SCALE = 10**_QUANT_DECIMALS


def _normal_float(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite float cannot be content-hashed")
    value = round(value, _QUANT_DECIMALS)
    if value == 0.0:
        return 0.0
    return value


def _canonical_array(array: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(array)
    if arr.dtype.kind == "f":
        work = np.asarray(arr, dtype="<f8")
    elif arr.dtype.kind == "c":
        work = np.asarray(arr, dtype="<c16")
    elif arr.dtype.kind == "b":
        work = np.asarray(arr, dtype="|u1")
    elif arr.dtype.kind in "iu":
        work = np.asarray(arr, dtype="<i8")
    else:
        raise TypeError(f"unsupported array dtype for hashing: {arr.dtype}")
    work = np.ascontiguousarray(work)
    return {
        "__array__": True,
        "dtype": work.dtype.str,
        "shape": list(work.shape),
        "sha256": hashlib.sha256(work.tobytes(order="C")).hexdigest(),
    }


def canonicalize(value: Any) -> Any:
    """Convert values to a JSON-compatible canonical structure."""

    if dataclasses.is_dataclass(value):
        return canonicalize(dataclasses.asdict(value))
    if isinstance(value, np.ndarray):
        return _canonical_array(value)
    if isinstance(value, np.generic):
        return canonicalize(value.item())
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): canonicalize(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [canonicalize(item) for item in value]
    if isinstance(value, set):
        return sorted(canonicalize(item) for item in value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"__bytes_sha256__": hashlib.sha256(bytes(value)).hexdigest(), "length": len(bytes(value))}
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return _normal_float(value)
    raise TypeError(f"unsupported value for canonicalization: {type(value).__name__}")


def canonical_json(value: Any, *, pretty: bool = False) -> str:
    canonical = canonicalize(value)
    if pretty:
        return json.dumps(
            canonical,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"
    return json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def bytes_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text_atomic(path: str | os.PathLike[str], text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def write_bytes_atomic(path: str | os.PathLike[str], data: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with open(temporary, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
