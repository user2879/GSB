"""Small deterministic and atomic I/O helpers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Any


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return format(value, "f")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def content_fingerprint(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        _fsync_dir(path.parent)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                                       default=_json_default) + "\n")


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_replace_directory(source: Path, target: Path) -> None:
    """Install source as target, moving an existing target aside first.

    This is used only as a best-effort convenience helper. Snapshot generations
    themselves are committed through a unique directory rename plus an atomic
    CURRENT pointer, so an interrupted process never exposes a half-written
    generation.
    """

    target.parent.mkdir(parents=True, exist_ok=True)
    backup = target.with_name(f".{target.name}.old.{os.getpid()}")
    if backup.exists():
        shutil.rmtree(backup)
    if target.exists():
        os.replace(target, backup)
    try:
        os.replace(source, target)
        _fsync_dir(target.parent)
    finally:
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)


def deep_copy_json(value: Any) -> Any:
    """Copy a JSON-compatible value without sharing mutable containers."""

    return json.loads(canonical_json(value))

