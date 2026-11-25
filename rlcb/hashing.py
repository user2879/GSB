"""Canonical JSON serialization and content digests.

All byte-stable comparisons in the project funnel through ``canonical_json`` so
that dict traversal order can never influence a conclusion or its digest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
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
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"object of type {type(value)!r} is not JSON serializable")


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def digest_concat(*parts: Any) -> str:
    """Stable digest over an ordered set of already-canonical parts."""
    hasher = hashlib.sha256()
    for part in parts:
        if not isinstance(part, str):
            part = canonical_json(part)
        hasher.update(part.encode("utf-8"))
        hasher.update(b"\x1e")
    return "sha256:" + hasher.hexdigest()


def file_sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return "sha256:" + hasher.hexdigest()
