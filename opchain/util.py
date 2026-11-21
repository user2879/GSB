"""Small deterministic and filesystem helpers used across the application."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def tensor_fingerprint(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(b"|")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"|")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_bytes(path: str | os.PathLike[str], payload: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def atomic_write_text(path: str | os.PathLike[str], text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | os.PathLike[str], value: Any) -> None:
    atomic_write_text(path, json.dumps(to_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=True) + "\n")


def atomic_replace_directory(staging: Path, current: Path) -> None:
    """Replace a directory as a unit, retaining a recoverable backup."""

    backup = current.with_name(current.name + ".previous")
    if not current.exists() and backup.exists():
        os.replace(backup, current)
    if backup.exists():
        shutil.rmtree(backup)
    if current.exists():
        os.replace(current, backup)
    try:
        os.replace(staging, current)
    except BaseException:
        if backup.exists() and not current.exists():
            os.replace(backup, current)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def ensure_private_workspace(path: str | os.PathLike[str]) -> Path:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    for child in ("artifacts", "snapshots", "tmp"):
        (root / child).mkdir(parents=True, exist_ok=True)
    return root


@contextmanager
def workspace_lock(path: str | os.PathLike[str]):
    """Serialize workspace mutations across threads and processes on POSIX."""

    import fcntl

    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
