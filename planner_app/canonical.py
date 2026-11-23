"""Byte-stable JSON and content-hash helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_text(value: Any) -> str:
    return canonical_bytes(value).decode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def content_digest(document: dict[str, Any], field: str = "content_digest") -> str:
    payload = {key: value for key, value in document.items() if key != field}
    return digest(payload)


def with_content_digest(document: dict[str, Any]) -> dict[str, Any]:
    result = dict(document)
    result.pop("content_digest", None)
    result["content_digest"] = content_digest(result)
    return result


def atomic_write_json(path: Path, document: Any) -> None:
    """Write JSON without exposing a partially written snapshot file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    data = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def stable_sorted(values: Iterable[str]) -> list[str]:
    return sorted(values, key=lambda item: item.encode("utf-8"))


def first_difference(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first differing top-level field in sorted key order."""

    for key in sorted(set(left) | set(right), key=lambda item: item.encode("utf-8")):
        if left.get(key) != right.get(key):
            return {"field": key, "snapshot": left.get(key), "current": right.get(key)}
    return None
