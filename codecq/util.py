"""Shared deterministic helpers: canonical JSON, hashing, atomic writes."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


def canonical_json(obj) -> str:
    """Serialize to a canonical, byte-stable JSON string."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_obj(obj) -> str:
    return sha256_hex(canonical_json(obj).encode("utf-8"))


def hash_array(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("ascii"))
    h.update(b"|")
    h.update(repr(tuple(arr.shape)).encode("ascii"))
    h.update(b"|")
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def round6(x: float) -> float:
    """Round to 6 decimals; normalize -0.0 so JSON bytes stay stable."""
    v = round(float(x), 6)
    return 0.0 if v == 0 else v


def atomic_write(path, data: bytes) -> None:
    """Write data to path atomically (tmp file + fsync + os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def clean_stale_tmp(directory) -> None:
    """Remove leftover *.tmp files; they are never valid committed state."""
    directory = Path(directory)
    if not directory.is_dir():
        return
    for entry in directory.rglob("*.tmp"):
        entry.unlink()
