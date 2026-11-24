"""Stable canonical JSON serialization and SHA-256 fingerprints."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    """Deterministic JSON text independent of dict insertion order."""

    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def sha256_hex(value: Any) -> str:
    data = canonical_json(value)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def short_digest(value: Any, length: int = 16) -> str:
    digest = sha256_hex(value)
    return digest[:length]


def fingerprint_files(texts: dict) -> str:
    """Fingerprint a mapping of file name -> textual content."""

    return sha256_hex({name: texts[name] for name in sorted(texts)})
