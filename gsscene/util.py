"""Deterministic hashing, JSON and atomic file helpers."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def hash_value(value: Any) -> str:
    return sha256_text(canonical_json(value))


def _canonical_array(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.dtype.byteorder not in ("=", "|"):
        arr = arr.astype(arr.dtype.newbyteorder("="), copy=False)
    if arr.dtype.kind == "f" and arr.dtype.itemsize > 8:
        arr = arr.astype(np.float64, copy=False)
    return np.ascontiguousarray(arr)


def array_fingerprint(array: np.ndarray) -> str:
    arr = _canonical_array(array)
    digest = hashlib.sha256()
    digest.update(arr.dtype.str.encode("ascii"))
    digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    digest.update(arr.tobytes(order="C"))
    return digest.hexdigest()


def arrays_fingerprint(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        digest.update(name.encode("utf-8"))
        digest.update(bytes.fromhex(array_fingerprint(arrays[name])))
    return digest.hexdigest()


def file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def ensure_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, _canonical_array(array), allow_pickle=False)
    return buffer.getvalue()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def atomic_write_json(path: Path, value: Any, *, pretty: bool = True) -> None:
    if pretty:
        text = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n"
    else:
        text = canonical_json(value) + "\n"
    atomic_write_bytes(path, text.encode("utf-8"))


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write a deterministic NPZ (fixed ZIP timestamps, sorted members)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temp_name, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for name in sorted(arrays):
                info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = 0o600 << 16
                archive.writestr(info, _npy_bytes(np.asarray(arrays[name])))
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def remove_tree(path: Path) -> None:
    import shutil

    if path.exists():
        shutil.rmtree(path)


def stable_sorted_ints(values: Iterable[int]) -> list[int]:
    return sorted(int(value) for value in values)


def jsonable(value: Any) -> Any:
    """Convert numpy containers to deterministic JSON-compatible values."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return value
