"""Small deterministic and crash-safe helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
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
    ) + "\n"


def json_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_array_hash(*arrays: tuple[np.ndarray, str]) -> str:
    digest = hashlib.sha256()
    for array, dtype in arrays:
        contiguous = np.ascontiguousarray(array, dtype=dtype)
        digest.update(str(contiguous.shape).encode("ascii") + b"\0")
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp-", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: str | os.PathLike[str], value: Any) -> None:
    atomic_write_bytes(path, json_bytes(value))


def read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def make_readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def stable_sorted_keys(values: Iterable[str]) -> list[str]:
    return sorted(values, key=lambda item: item.encode("utf-8"))


def float32_quantization_coordinate_bound(values: np.ndarray) -> np.ndarray:
    """Half an adjacent float32 spacing, represented in float64."""

    as_float32 = np.asarray(values, dtype=np.float32)
    spacing = np.spacing(np.abs(as_float32)).astype(np.float64)
    # ``spacing(0)`` is the smallest subnormal; half of it is a safe bound too.
    return 0.5 * spacing


def point_quantization_bound(points: np.ndarray) -> np.ndarray:
    coordinate = float32_quantization_coordinate_bound(np.asarray(points, dtype=np.float32))
    return np.linalg.norm(coordinate, axis=1)

