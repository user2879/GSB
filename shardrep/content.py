"""Deterministic shard byte generation without any network dependency."""

from __future__ import annotations

import hashlib

try:  # NumPy is a Windows requirement but optional for the pure-Python fallback.
    import numpy as _np
except ImportError:  # pragma: no cover - exercised on the current headless host
    _np = None


def generate_shard_bytes(seed: int, logical_id: str, byte_count: int) -> bytes:
    """Return a byte-identical stream with or without NumPy.

    NumPy, when present, is used as the byte-buffer materialisation layer. The
    entropy stream itself is SHAKE-256 so the committed local sample remains
    portable across Python/NumPy versions and operating systems.
    """

    if byte_count <= 0:
        raise ValueError("byte_count must be positive")
    material = f"shardrep-shake256-v1\x00{seed}\x00{logical_id}\x00{byte_count}".encode("utf-8")
    raw = hashlib.shake_256(material).digest(byte_count)
    if _np is not None:
        return _np.frombuffer(raw, dtype=_np.uint8).tobytes()
    return raw


def deterministic_replacement(seed: int, shard_id: str, byte_count: int) -> bytes:
    material = f"shardrep-replacement-v1\x00{seed}\x00{shard_id}\x00{byte_count}".encode("utf-8")
    return hashlib.shake_256(material).digest(byte_count)


def content_digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()

