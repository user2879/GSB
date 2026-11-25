from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    )


def _json_default(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, (set, tuple)):
        return sorted(value, key=str)
    raise TypeError(f"Object of type {type(value)!r} is not fingerprintable")


def fingerprint(value: Any, length: int | None = None) -> str:
    raw = canonical_json(value).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    return digest if length is None else digest[:length]


def named_fingerprint(name: str, value: Any) -> str:
    return f"{name}:{fingerprint(value, 20)}"


def derive_cache_key(stage: str, **parts: Any) -> str:
    payload = {"stage": stage, "parts": parts}
    return f"{stage}-{fingerprint(payload, 24)}"


def stable_seed(*parts: Any) -> int:
    material = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:16], "big", signed=False)


def stable_sort_key(request_id: str, stage: str) -> str:
    return f"{request_id}\x1f{stage}"
