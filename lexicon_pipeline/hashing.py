"""Canonical JSON hashing and stable scheduling keys."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint(value: Any, length: int | None = 16) -> str:
    digest = sha256_text(canonical_json(value))
    return digest if length is None else digest[:length]


def stable_key(identifier: str, stage: str) -> str:
    """Stable sort key derived from a document identifier and stage name."""
    return fingerprint({"doc_id": identifier, "stage": stage}, 20)


def content_fingerprint(value: Any) -> str:
    return fingerprint(value, 20)
