"""Canonical serialization and digest helpers shared by ledger/snapshot/replay."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def chained_digest(previous: str, value: Any) -> str:
    material = f"{previous}\n".encode("ascii") + canonical_bytes(value)
    return hashlib.sha256(material).hexdigest()
