"""Deterministic I/O, hashing, and atomic commit helpers."""

from __future__ import annotations

import hashlib
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
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_fingerprint(array: np.ndarray) -> str:
    array64 = np.asarray(array, dtype="<f8", order="C")
    digest = hashlib.sha256()
    digest.update(canonical_bytes(list(array64.shape)))
    digest.update(array64.tobytes(order="C"))
    return digest.hexdigest()


def combined_fingerprint(*parts: Any) -> str:
    digest = hashlib.sha256()
    for part in parts:
        if isinstance(part, bytes):
            payload = part
        elif isinstance(part, str):
            payload = part.encode("utf-8")
        else:
            payload = canonical_bytes(part)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: str | Path) -> list[Any]:
    rows: list[Any] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
    return rows


def load_yaml(path: str | Path) -> Any:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:  # JSON is a valid YAML 1.2 subset and keeps runtime deps small.
        return json.loads(text)
    return yaml.safe_load(text)


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def publish_directory(temp_dir: str | Path, final_dir: str | Path) -> None:
    """Publish a fully prepared directory with one rename operation."""

    temp_path = Path(temp_dir)
    final_path = Path(final_dir)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if final_path.exists():
        raise FileExistsError(final_path)
    os.replace(temp_path, final_path)


def remove_tree(path: str | Path) -> None:
    target = Path(path)
    if target.exists():
        shutil.rmtree(target)


def finite_float(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def normalize_order(values: Iterable[str]) -> list[str]:
    return sorted(set(values))


def stable_round_half_away(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.sign(values) * np.floor(np.abs(values) + 0.5)


def stable_round_half_even(values: np.ndarray) -> np.ndarray:
    return np.rint(np.asarray(values, dtype=np.float64))


def result_digest(payload: Any) -> str:
    return sha256_json(payload)
