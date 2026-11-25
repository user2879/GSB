from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from typing import Any


def canonical_json(value: Any) -> str:
    """Return a stable UTF-8 JSON representation used by every fingerprint."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def digest(value: Any) -> str:
    return digest_bytes(canonical_json(value).encode("utf-8"))


def stable_key(*parts: Any) -> str:
    return digest({"parts": list(parts)})


def fixed_string(value: int, decimal_places: int, scale: int = 1_000_000) -> str:
    if scale <= 0:
        raise ValueError("scale must be positive")
    return f"{Decimal(value) / Decimal(scale):.{decimal_places}f}"


def text_digest(text: str) -> str:
    return digest_bytes(text.encode("utf-8"))

