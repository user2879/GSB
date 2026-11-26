from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def fingerprint(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stable_seed(base: int, parts: Iterable[str]) -> int:
    payload = "\x1f".join([str(base), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def sorted_records(records: Iterable[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return sorted(records, key=lambda record: str(record.get(key, "")))
