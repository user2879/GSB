"""Deterministic content fingerprints and cache-key construction.

All fingerprints are SHA-256 hex digests over canonical JSON / raw bytes. The
canonicalization is fully specified here so two processes always derive the
same key from the same logical inputs.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

JSON_SEPARATORS = (",", ":")


def canonical_json(obj: Any) -> str:
    """Serialize with sorted keys and stable float/string encoding."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=JSON_SEPARATORS,
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def fingerprint_json(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def fingerprint_strings(values: Iterable[str]) -> str:
    return sha256_text("\u001f".join(sorted(values)))


def content_fingerprint(png_bytes: bytes) -> str:
    """Fingerprint of the raw image container bytes."""
    return sha256_bytes(png_bytes)


def scale_tier(num_samples: int) -> str:
    """Stable scale-tier identifier derived from the matrix sample count."""
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    buckets = (0, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
    tier = 0
    for idx, edge in enumerate(buckets):
        if num_samples >= edge:
            tier = idx
    return f"T{tier:02d}_n{num_samples}"


def cache_key(kind: str, *parts: Any) -> str:
    """Build a content-addressed cache key for a stage artifact."""
    rendered = [kind]
    for part in parts:
        if isinstance(part, (dict, list, tuple)):
            rendered.append(fingerprint_json(part))
        else:
            rendered.append(str(part))
    return f"{kind}:{sha256_text('|'.join(rendered))}"
