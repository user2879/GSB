"""Deterministic serialisation, hashing and clock helpers."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any


def canonical_json(value: Any) -> str:
    """Stable byte representation: sorted keys, compact separators, no whitespace."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_of(value: Any) -> str:
    return sha256_hex(canonical_json(value))


def monotonic_ms() -> float:
    return time.monotonic() * 1000.0


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int((time.time() % 1) * 1000):03d}Z"
