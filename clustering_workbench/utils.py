"""Small deterministic, atomic, and serialization helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def hash_object(prefix: str, value: Any) -> str:
    return f"{prefix}-{sha256_text(canonical_json(value))[:24]}"


def hash_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.astype("<f8", copy=False).tobytes())
    return digest.hexdigest()


def hash_sequence(values: Iterable[Any]) -> str:
    return sha256_text(canonical_json(list(values)))


def json_safe(value: Any) -> Any:
    """Convert NumPy-rich values into strict JSON-compatible values."""
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value))
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.chmod(temp_name, 0o644)
        except OSError:
            pass
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, canonical_json(json_safe(value)) + "\n")


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.chmod(temp_name, 0o644)
        except OSError:
            pass
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def atomic_write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        with open(temp_name, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.chmod(temp_name, 0o644)
        except OSError:
            pass
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(base))
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def new_temp_dir(parent: Path, prefix: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / f".{prefix}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def safe_rmtree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


def truncate_text(text: str, limit: int = 800) -> str:
    return text if len(text) <= limit else text[:limit] + "...<truncated>"
