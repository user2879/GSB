"""Canonical JSON and content fingerprints."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from fractions import Fraction
from typing import Any


def json_default(value: Any):
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, Fraction):
        return format(value.numerator, "d") + "/" + format(value.denominator, "d")
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def canonical_dumps(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=json_default,
        allow_nan=False,
    )


def fingerprint(payload: Any) -> str:
    body = canonical_dumps(payload).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def short_fingerprint(payload: Any, length: int = 12) -> str:
    return fingerprint(payload)[:length]


def cache_key(stage: str, payload: Any) -> str:
    return f"{stage}:{fingerprint(payload)}"


def fingerprint_lenient(payload: Any) -> str:
    """Fingerprint invalid inputs containing non-finite float sentinels."""
    body = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=json_default,
        allow_nan=True,
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()
