from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_dumps(value: Any) -> str:
    """Serialize a value with one stable cross-run representation."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def _json_default(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, (set, tuple)):
        return sorted(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def fingerprint(value: Any, length: int | None = None) -> str:
    raw = canonical_dumps(value).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    return digest if length is None else digest[:length]


def prefixed_key(prefix: str, *parts: Any) -> str:
    material = canonical_dumps(list(parts))
    return f"{prefix}_{hashlib.sha256(material.encode('utf-8')).hexdigest()[:24]}"


def stable_sort_key(value: Any) -> tuple:
    if isinstance(value, dict):
        return tuple((k, stable_sort_key(v)) for k, v in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(stable_sort_key(v) for v in value)
    return (type(value).__name__, value)
