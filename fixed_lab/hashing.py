"""Canonical JSON and SHA-256 content fingerprints."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from typing import Any


def decimal_fingerprint_value(value: Decimal) -> str:
    if value == 0:
        return "0"
    normalized = value.normalize()
    text = format(normalized, "f")
    if text in {"-0", "-0.0"}:
        return "0"
    return text


def to_canonical_data(value: Any) -> Any:
    if isinstance(value, Decimal):
        return {"__decimal__": decimal_fingerprint_value(value)}
    if isinstance(value, dict):
        return {str(k): to_canonical_data(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [to_canonical_data(v) for v in value]
    return value


def canonical_dumps(value: Any) -> str:
    return json.dumps(
        to_canonical_data(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_dumps(value).encode("utf-8")).hexdigest()


def short_key(prefix: str, material: Any, length: int = 18) -> str:
    return f"{prefix}:v2:{canonical_sha256(material)[:length]}"
