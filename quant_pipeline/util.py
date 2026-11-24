"""Deterministic serialization, hashing, rounding, and filesystem helpers."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
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


def pretty_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"


def sha256_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_write_json(path: str | Path, value: Any, *, pretty: bool = True) -> None:
    atomic_write_text(path, pretty_json(value) if pretty else canonical_json(value))


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def replace_directory(stage_dir: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a directory, never overwriting files one by one."""

    stage_dir = stage_dir.resolve()
    parent = stage_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f".{stage_dir.name}.", suffix=".tmp", dir=parent))
    backup_dir = parent / f".{stage_dir.name}.previous"
    try:
        (tmp_dir / "data.json").write_text(pretty_json(payload), encoding="utf-8", newline="\n")
        (tmp_dir / "COMMITTED").write_text("committed\n", encoding="utf-8", newline="\n")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        if stage_dir.exists():
            os.replace(stage_dir, backup_dir)
        os.replace(tmp_dir, stage_dir)
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
    except Exception:
        if not stage_dir.exists() and backup_dir.exists():
            os.replace(backup_dir, stage_dir)
        raise
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def recover_directory(stage_dir: Path) -> None:
    backup_dir = stage_dir.parent / f".{stage_dir.name}.previous"
    if not stage_dir.exists() and backup_dir.exists():
        os.replace(backup_dir, stage_dir)


def finite_float(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return False
    return math.isfinite(float(value))


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def round_values(values: np.ndarray, mode: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if mode == "nearest_even":
        return np.rint(values)
    if mode == "nearest_away":
        return np.sign(values) * np.floor(np.abs(values) + 0.5)
    if mode == "floor":
        return np.floor(values)
    if mode == "ceil":
        return np.ceil(values)
    if mode == "truncate":
        return np.trunc(values)
    raise ValueError(f"unsupported rounding mode: {mode}")


def softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def ndarray_to_list(values: np.ndarray) -> list[Any]:
    return np.asarray(values).tolist()


def unique_in_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

