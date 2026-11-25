from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Iterable


def canonical_json(value: Any) -> str:
    """Return the one canonical JSON representation used by the ledger."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def sha256_digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def monotonic_ms() -> int:
    """Milliseconds from the local monotonic clock, never wall-clock UTC."""

    return time.monotonic_ns() // 1_000_000


def stable_sorted(items: Iterable[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    return sorted(items, key=lambda item: tuple(str(item.get(key, "")) for key in keys))

