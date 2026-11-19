"""Canonical JSON, hashing and deterministic array persistence helpers."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _normalise(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return float(f"{value:.12g}")
    if isinstance(value, dict):
        return {str(k): _normalise(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalise(v) for v in value]
    return value


def canonical_json_bytes(value: Any) -> bytes:
    normal = _normalise(value)
    return (
        json.dumps(
            normal,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_json(path: str | Path, value: Any) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value)
    tmp = target.with_name(target.name + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, target)
    return hashlib.sha256(payload).hexdigest()


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return buffer.getvalue()


def write_npy(path: str | Path, array: np.ndarray) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = npy_bytes(array)
    tmp = target.with_name(target.name + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, target)
    return hashlib.sha256(payload).hexdigest()


def read_npy(path: str | Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def pose_frame_hashes(poses: np.ndarray, decimals: int = 6) -> list[str]:
    """Stable per-frame hash of joint positions quantised at 1e-6 metres."""
    scale = 10**decimals
    hashes: list[str] = []
    for frame in np.asarray(poses, dtype=np.float64):
        quantised = np.rint(frame * scale).astype("<i8", copy=False)
        hashes.append(hashlib.sha256(quantised.tobytes(order="C")).hexdigest())
    return hashes


def array_digest(array: np.ndarray) -> str:
    return sha256_bytes(npy_bytes(np.asarray(array)))


def rolling_digest(items: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for item in items:
        digest.update(item.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()
