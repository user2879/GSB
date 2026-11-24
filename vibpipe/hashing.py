from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=_json_default,
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return format(value, "f")
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, (set, tuple)):
        return sorted(value) if isinstance(value, set) else list(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint(value: Any) -> str:
    return sha256_text(canonical_json(value))


def short_key(*parts: Any, length: int = 16) -> str:
    return sha256_text("|".join(canonical_json(part) for part in parts))[:length]
