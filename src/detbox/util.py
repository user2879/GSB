"""Deterministic hashing, atomic file replacement and small helpers."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_hash(data: Any) -> str:
    """Stable SHA-256 fingerprint of any JSON-serialisable payload."""
    payload = canonical_json(data)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_hash(path: str | Path, chunk: int = 1 << 16) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write via a sibling temp file then os.replace (atomic on local FS)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", suffix=".part", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: str | Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False))


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def all_finite(values: Any) -> bool:
    if isinstance(values, dict):
        return all(all_finite(key) and all_finite(val) for key, val in values.items())
    if isinstance(values, (list, tuple)):
        return all(all_finite(item) for item in values)
    if isinstance(values, (int, float)) and not isinstance(values, bool):
        return math.isfinite(float(values))
    return True


def replace_dir_atomic(target: Path, builder) -> None:
    """Stage a directory under target's parent then atomically swap it in.

    ``builder`` receives the staging directory path and populates it.  The
    previous directory (if any) is moved aside and removed only after the
    replacement commits, so recovery always sees a complete tree.
    """
    import shutil

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.staging-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    backup = target.parent / f".{target.name}.old-{os.getpid()}"
    try:
        builder(staging)
        if target.exists():
            os.replace(target, backup) if os.path.isfile(target) else _move_dir(target, backup)
        _move_dir(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)


def _move_dir(source: Path, destination: Path) -> None:
    import shutil

    try:
        os.replace(source, destination)
    except OSError:
        shutil.move(str(source), str(destination))
