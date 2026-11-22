from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_canonical(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def true_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def round_div(numerator: int, denominator: int) -> int:
    """Round a rational integer division half away from zero, deterministically."""
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    sign = -1 if numerator < 0 else 1
    amount = abs(numerator)
    return sign * ((amount + denominator // 2) // denominator)
