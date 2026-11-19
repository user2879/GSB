"""Canonical binary encodings and SHA-256 helpers."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import struct
from typing import Any, Iterable, Sequence


def pack_floats(values: Iterable[float]) -> bytes:
    vals = tuple(float(v) for v in values)
    if not vals:
        return b""
    if any(not math.isfinite(v) for v in vals):
        raise ValueError("non-finite float in deterministic payload")
    return struct.pack("<%dd" % len(vals), *vals)


def unpack_floats(data: bytes) -> tuple[float, ...]:
    if not data:
        return ()
    if len(data) % 8:
        raise ValueError("float payload length must be a multiple of 8")
    return struct.unpack("<%dd" % (len(data) // 8), data)


def pack_ints(values: Iterable[int]) -> bytes:
    vals = tuple(int(v) for v in values)
    if not vals:
        return b""
    return struct.pack("<%di" % len(vals), *vals)


def unpack_ints(data: bytes) -> tuple[int, ...]:
    if not data:
        return ()
    if len(data) % 4:
        raise ValueError("int payload length must be a multiple of 4")
    return struct.unpack("<%di" % (len(data) // 4), data)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def canonical_json_bytes(value: Any) -> bytes:
    text = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return text.encode("utf-8")


def canonical_json_hash(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def b64encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def b64decode(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"), validate=True)


def float_bits(value: float) -> bytes:
    return struct.pack("<d", float(value))


def digest_sequence(chunks: Sequence[bytes]) -> str:
    h = hashlib.sha256()
    for chunk in chunks:
        h.update(struct.pack("<Q", len(chunk)))
        h.update(chunk)
    return h.hexdigest()


def quantize(values: Iterable[float], decimals: int) -> tuple[float, ...]:
    # round() is deterministic here (binary64); -0.0 is canonicalized to 0.0.
    out = []
    for value in values:
        q = round(float(value), decimals)
        out.append(0.0 if q == 0.0 else q)
    return tuple(out)


def tolerant_float_hash(values: Iterable[float], decimals: int = 11) -> str:
    return sha256_bytes(pack_floats(quantize(values, decimals)))


def array_max_abs_diff(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("arrays must have the same length")
    return max((abs(float(x) - float(y)) for x, y in zip(a, b)), default=0.0)

