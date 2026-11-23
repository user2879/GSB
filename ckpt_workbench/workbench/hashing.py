"""Deterministic canonical encoding and content digests.

Everything is encoded with explicit field tags and big-endian fixed-width
integer framing so identical logical content always hashes to identical bytes
across runs and machines. No platform-dependent float repr is used for arrays;
NumPy arrays are hashed from their little-endian raw bytes.
"""

import hashlib
import struct

import numpy as np


def _u32(value):
    return struct.pack(">I", int(value))


def _u64(value):
    return struct.pack(">Q", int(value))


def _tag(tag):
    return tag.encode("ascii")


def canon_bytes(value):
    """Encode bytes/str/int/float/bool/None/list/tuple/dict/np arrays canonically."""
    if isinstance(value, (bytes, bytearray)):
        return _tag("B") + _u64(len(value)) + bytes(value)
    if isinstance(value, str):
        raw = value.encode("utf-8")
        return _tag("S") + _u64(len(raw)) + raw
    if isinstance(value, bool):
        return _tag("b") + (b"\x01" if value else b"\x00")
    if isinstance(value, int):
        raw = str(value).encode("ascii")
        return _tag("i") + _u64(len(raw)) + raw
    if isinstance(value, float):
        raw = repr(value).encode("ascii")
        return _tag("f") + _u64(len(raw)) + raw
    if value is None:
        return _tag("N")
    if isinstance(value, np.ndarray):
        arr = np.ascontiguousarray(value)
        header = _tag("A") + canon_bytes(str(arr.dtype)) + canon_bytes(list(arr.shape))
        return header + _u64(arr.nbytes) + arr.tobytes(order="C")
    if isinstance(value, (list, tuple)):
        out = _tag("L") + _u64(len(value))
        for item in value:
            out += canon_bytes(item)
        return out
    if isinstance(value, dict):
        out = _tag("D") + _u64(len(value))
        for key in sorted(value.keys()):
            out += canon_bytes(str(key)) + canon_bytes(value[key])
        return out
    raise TypeError(f"cannot canonically encode type {type(value)!r}")


def digest(value):
    """Return a stable sha256 hex digest for any canonically encodable value."""
    return hashlib.sha256(canon_bytes(value)).hexdigest()


def digest_parts(parts):
    """Digest an ordered list of already-canonical parts under one framing tag."""
    h = hashlib.sha256()
    h.update(_tag("SEQ"))
    h.update(_u64(len(parts)))
    for part in parts:
        encoded = part if isinstance(part, (bytes, bytearray)) else canon_bytes(part)
        h.update(_u64(len(encoded)))
        h.update(encoded)
    return h.hexdigest()


def short(value, width=12):
    return value[:width]
