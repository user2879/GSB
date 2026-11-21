from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_bytes(value: Any) -> bytes:
    """Return the one canonical JSON representation used by all components."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_text(value: Any) -> str:
    return canonical_bytes(value).decode("utf-8")


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_text(value: str) -> str:
    return digest_bytes(value.encode("utf-8"))


def digest_canonical(value: Any) -> str:
    return digest_bytes(canonical_bytes(value))


def stable_bucket(seed: int, rule_version: int, trace_id: str, span_id: str, node_id: str) -> int:
    material = f"{seed}\x1f{rule_version}\x1f{trace_id}\x1f{span_id}\x1f{node_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % 10_000
