"""Deterministic serialization and digest helpers.

All hashing goes through canonical JSON with sorted keys and compact separators,
so the same logical object always produces the same digest on every machine and
Python version.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> str:
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_default,
    )


def _default(obj: Any) -> Any:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    raise TypeError(f"object of type {type(obj)!r} is not digest serializable")


def digest_of(obj: Any) -> str:
    payload = canonical_json(obj).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def short_digest(obj: Any) -> str:
    return digest_of(obj)[:12]
