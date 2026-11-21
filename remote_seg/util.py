"""Canonical serialization, hashing, filesystem and arithmetic helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_hash(value: Any, *, prefix: str = "") -> str:
    payload = (prefix + canonical_json(value)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def bytes_hash(data: bytes, *, prefix: str = "") -> str:
    digest = hashlib.sha256()
    digest.update(prefix.encode("utf-8"))
    digest.update(data)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_content_fingerprint(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def rounded_float(value: float, digits: int = 12) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite float in canonical output: {value!r}")
    result = round(value, digits)
    return 0.0 if result == 0 else result


def rounded_mapping(values: Iterable[tuple[Any, float]]) -> dict[str, float]:
    return {str(key): rounded_float(value) for key, value in values}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        _fsync_dir(path.parent)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n")


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def deep_set(mapping: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    cursor: Any = mapping
    for index, part in enumerate(parts):
        is_last = index == len(parts) - 1
        if isinstance(cursor, list):
            list_index = int(part)
            if is_last:
                cursor[list_index] = value
            else:
                cursor = cursor[list_index]
            continue
        if not isinstance(cursor, dict):
            raise TypeError(f"cannot traverse non-container at {part!r}")
        if is_last:
            cursor[part] = value
        else:
            next_part = parts[index + 1]
            if part not in cursor:
                cursor[part] = [] if next_part.isdigit() else {}
            cursor = cursor[part]


def relative_posix(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()
