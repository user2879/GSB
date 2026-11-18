"""Deterministic hashing, canonical serialization and atomic file writes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile

_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_MASK64 = 0xFFFFFFFFFFFFFFFF


def fnv1a64(data: bytes) -> int:
    """64-bit FNV-1a hash, fully specified and independent of Python's hash()."""
    value = _FNV_OFFSET
    for byte in data:
        value ^= byte
        value = (value * _FNV_PRIME) & _MASK64
    return value


def stable_hash(text: str, seed: int = 0) -> int:
    """Deterministic 64-bit hash of a unicode string with an integer seed."""
    payload = seed.to_bytes(8, "little", signed=False) + text.encode("utf-8")
    return fnv1a64(payload)


def canonical_json(obj) -> str:
    """JSON serialization with sorted keys and stable separators (no whitespace)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint(obj) -> str:
    """16-byte blake2b content fingerprint of a JSON-serialisable object."""
    return hashlib.blake2b(canonical_json(obj).encode("utf-8"), digest_size=16).hexdigest()


def keyed_fingerprint(parts: list) -> str:
    """Fingerprint of an ordered list of already-derived fingerprints/strings."""
    joined = "|".join(str(part) for part in parts)
    return hashlib.blake2b(joined.encode("utf-8"), digest_size=16).hexdigest()


def double_hex(value: float) -> str:
    """Exact hex representation of a double, used as the reference diff value."""
    return float(value).hex()


def atomic_write_bytes(path: str, payload: bytes) -> None:
    """Write payload via a temp file in the same directory then os.replace."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def atomic_write_json(path: str, obj) -> None:
    atomic_write_bytes(
        path,
        (json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
