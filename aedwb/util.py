"""Deterministic JSON, hashing, and filesystem helpers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    """Return a stable, strict JSON representation used for fingerprints."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def write_json_atomic(path: str | Path, value: Any) -> None:
    """Atomically replace a JSON file in its destination directory."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=str(path.parent)
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        write_json(temporary, value)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def replace_directory(temporary: str | Path, target: str | Path) -> None:
    """Replace a directory as a whole, with a portable Windows-friendly swap."""

    temporary = Path(temporary)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = target.parent / f".{target.name}.old-{uuid.uuid4().hex}"
    if target.exists():
        os.replace(target, backup)
    try:
        os.replace(temporary, target)
    except Exception:
        if backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
    finally:
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)


def clean_temporary_directories(root: str | Path) -> None:
    """Remove abandoned temporary stage directories from an interrupted run."""

    root = Path(root)
    if not root.exists():
        return
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith(".") and ".tmp-" in child.name:
            shutil.rmtree(child, ignore_errors=True)


def json_safe_float(value: float, digits: int = 12) -> float:
    """Bound float noise in reports while retaining deterministic values."""

    rounded = round(float(value), digits)
    if rounded == 0.0:
        return 0.0
    return rounded


def json_safe_vector(values: Any, digits: int = 12) -> list[float]:
    return [json_safe_float(value, digits=digits) for value in values]
