"""Canonical serialization and exact floating-point fingerprints."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    text = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    )
    return (text + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_value(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def float_bits(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("non-finite float cannot be fingerprinted")
    return struct.pack(">d", float(value)).hex()


def float_hex(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("non-finite float cannot be fingerprinted")
    return float(value).hex()


def exact_float(value: float) -> dict[str, str]:
    return {"hex": float_hex(value), "bits": float_bits(value)}

