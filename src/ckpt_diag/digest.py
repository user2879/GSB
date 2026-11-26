"""Deterministic canonical serialization and content digests.

All summaries are digested over canonical JSON (sorted keys, compact
separators, no whitespace) so they are bit-for-bit identical regardless of
dictionary insertion order or traversal order.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def digest(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
