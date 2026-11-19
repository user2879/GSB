"""Deterministic hashing and canonical serialization helpers.

All on-disk artifacts that participate in reproducibility are serialized in
little-endian binary or canonical JSON, and hashed with BLAKE2b.  Floats are
always converted through ``repr`` on IEEE-754 doubles, which is stable across
CPython 3.10+ on x86_64 Windows / Linux.
"""
from __future__ import annotations

import hashlib
import io
import json
from typing import Any

import numpy as np

HASH_NAME = "blake2b-256"


def blake2b_hex(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=32).hexdigest()


def digest_file(path: str) -> str:
    h = hashlib.blake2b(digest_size=32)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_array(arr: np.ndarray) -> str:
    return blake2b_hex(np.ascontiguousarray(arr).tobytes(order="C"))


def _canonical_float(value: float) -> float:
    # -0.0 and 0.0 hash identically once dumped via repr; normalize explicitly.
    if value == 0.0:
        return 0.0
    return float(value)


def canonical_json_bytes(obj: Any) -> bytes:
    """Deterministic JSON: sorted keys, fixed separators, allow_nan disabled."""

    def default(o: Any) -> Any:
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not JSON canonicalizable: {type(o)!r}")

    def convert(o: Any) -> Any:
        if isinstance(o, dict):
            return {str(k): convert(v) for k, v in sorted(o.items(), key=lambda kv: str(kv[0]))}
        if isinstance(o, (list, tuple)):
            return [convert(v) for v in o]
        if isinstance(o, float):
            return _canonical_float(o)
        if isinstance(o, np.floating):
            return _canonical_float(float(o))
        if isinstance(o, np.integer):
            return int(o)
        return o

    text = json.dumps(
        convert(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=default,
    )
    return text.encode("utf-8")


def write_canonical_json(path: str, obj: Any) -> str:
    data = canonical_json_bytes(obj)
    atomic_write_bytes(path, data)
    return blake2b_hex(data)


def atomic_write_bytes(path: str, data: bytes) -> None:
    """Write via temp file + fsync + os.replace so readers never see partial data."""
    import os

    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def stable_stream_hash(items: list[Any]) -> str:
    """Hash a sequence of canonicalizable items in order."""
    buf = io.BytesIO()
    for item in items:
        chunk = canonical_json_bytes(item)
        buf.write(len(chunk).to_bytes(8, "little"))
        buf.write(chunk)
    return blake2b_hex(buf.getvalue())
