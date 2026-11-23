"""Canonical JSON serialization and content fingerprints."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data with a stable key/order representation."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def fingerprint_text(*parts: Any) -> str:
    return hashlib.sha256(
        "\x1f".join(str(part) for part in parts).encode("utf-8")
    ).hexdigest()


def task_key(stage: str, sequence_id: str, row: int | None = None, col: int | None = None) -> str:
    row_part = "*" if row is None else f"{row:08d}"
    col_part = "*" if col is None else f"{col:08d}"
    return f"{stage}:{sequence_id}:{row_part}:{col_part}"
