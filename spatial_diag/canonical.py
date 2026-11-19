"""Canonical serialization, hashing, and atomic file helpers.

All persisted JSON is produced from this module.  Floats are rounded to a
fixed number of decimal places before hashing so that harmless last-bit
variation in linear algebra does not leak into the content identity.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


FLOAT_DIGITS = 12


def json_safe(value: Any, *, digits: int = FLOAT_DIGITS) -> Any:
    """Return a deterministic JSON-compatible representation of *value*."""

    if isinstance(value, dict):
        return {str(k): json_safe(v, digits=digits) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [json_safe(v, digits=digits) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist(), digits=digits)
    if isinstance(value, np.generic):
        return json_safe(value.item(), digits=digits)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"non-finite float cannot be serialized: {number!r}")
        rounded = round(number, digits)
        return 0.0 if rounded == 0.0 else rounded
    return str(value)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        json_safe(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            json_safe(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def hash_file_set(paths: Iterable[Path], root: Path | None = None) -> str:
    """Hash a set of files in stable relative-path order."""

    root = root or Path(os.path.commonpath([str(p) for p in paths]))
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.relative_to(root).as_posix()):
        rel = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(rel)
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def file_records(paths: Iterable[Path], root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(paths, key=lambda p: p.relative_to(root).as_posix()):
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    try:
        directory_fd = os.open(str(path.parent), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    except OSError:
        pass
    finally:
        os.close(directory_fd)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, pretty_json_bytes(value))


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

