"""Deterministic content hashing.

Hashes are SHA-256 hex digests over canonical little-endian byte
representations.  They never depend on dictionary insertion order, thread
scheduling or block boundaries.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Mapping

import numpy as np


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash(path: str | os.PathLike[str], chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(obj: Any) -> bytes:
    """Deterministic JSON encoding used for both hashing and on-disk output."""

    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def array_bytes(arr: np.ndarray) -> bytes:
    a = np.ascontiguousarray(arr, dtype=np.float64)
    return a.tobytes()


def frame_hash(
    frame_index: int,
    rotations: np.ndarray,
    root_translation: np.ndarray,
    meta: Mapping[str, Any] | None = None,
) -> str:
    """Hash one output frame, including a small structured metadata block."""

    digest = hashlib.sha256()
    digest.update(b"FRAME\x01")
    digest.update(int(frame_index).to_bytes(8, "little", signed=False))
    digest.update(array_bytes(rotations))
    digest.update(array_bytes(root_translation))
    if meta:
        digest.update(canonical_json_bytes(meta))
    return digest.hexdigest()


def animation_hash(
    schema_version: str,
    interp: str,
    quat_order: str,
    joint_names: list[str],
    times: np.ndarray,
    rotations: np.ndarray,
    root_translations: np.ndarray,
    parameter_version: str,
) -> str:
    """Whole-animation content hash (covers every declared channel)."""

    digest = hashlib.sha256()
    digest.update(b"ANIM\x02")
    digest.update(
        canonical_json_bytes(
            {
                "schema_version": schema_version,
                "interp": interp,
                "quat_order": quat_order,
                "joint_names": list(joint_names),
                "parameter_version": parameter_version,
            }
        )
    )
    digest.update(array_bytes(np.asarray(times, dtype=np.float64)))
    digest.update(array_bytes(rotations))
    digest.update(array_bytes(root_translations))
    return digest.hexdigest()


def hash_dict(d: Mapping[str, Any]) -> str:
    return sha256_hex(canonical_json_bytes(d))
