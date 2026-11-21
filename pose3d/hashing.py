"""Canonical JSON and content-fingerprint helpers.

All cache keys and snapshot versions are derived from canonical JSON.  Floats
are emitted by Python's shortest-round-trip float representation, which keeps
the full binary64 value while remaining stable across runs.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def canonical_json(value: Any) -> str:
    """Return a deterministic JSON representation.

    Non-finite values are represented explicitly because cache keys must still
    be computable for malformed input observations before validation rejects
    them.
    """

    def sanitize(item: Any) -> Any:
        if isinstance(item, float):
            if math.isnan(item):
                return {"__float__": "nan"}
            if math.isinf(item):
                return {"__float__": "inf" if item > 0 else "-inf"}
            return item
        if isinstance(item, dict):
            return {str(k): sanitize(v) for k, v in sorted(item.items(), key=lambda kv: str(kv[0]))}
        if isinstance(item, (list, tuple)):
            return [sanitize(v) for v in item]
        if hasattr(item, "tolist"):
            return sanitize(item.tolist())
        return item

    return json.dumps(
        sanitize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def combine_fingerprints(items: Iterable[str]) -> str:
    return sha256_json(sorted(items))
