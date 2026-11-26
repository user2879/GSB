from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    """Serialize a value in a stable, cross-process canonical form."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def fingerprint(value: Any, prefix: str = "") -> str:
    digest = hashlib.sha256(canonical_bytes(value)).hexdigest()
    return f"{prefix}{digest}" if prefix else digest


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def short_key(key: str, length: int = 16) -> str:
    return key[:length]
