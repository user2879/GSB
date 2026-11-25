from __future__ import annotations

import hashlib
import json
import time
from typing import Any


def canonical_bytes(value: Any) -> bytes:
    """Return the unique byte representation used by all digests and comparisons."""

    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_value(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def monotonic_ns() -> int:
    return time.monotonic_ns()


def monotonic_ms() -> int:
    return monotonic_ns() // 1_000_000


def elapsed_ms(start_ns: int, end_ns: int | None = None) -> float:
    end = monotonic_ns() if end_ns is None else end_ns
    return round((end - start_ns) / 1_000_000.0, 3)


def ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ZeroDivisionError("denominator must be positive")
    return -((-numerator) // denominator)
