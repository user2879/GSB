"""Shared deterministic helpers (canonical bytes, hashing, JSON I/O)."""
from __future__ import annotations

import hashlib
import json
from typing import Any

# Canonical JSON: sorted keys, compact separators, ensure_ascii=False with
# explicit UTF-8 encoding so normalization is byte-identical on every OS and
# independent of dict insertion order.
def canonical_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_digest(obj: Any) -> str:
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def digest_parts(parts: list[Any]) -> str:
    return hashlib.sha256(b"|".join(canonical_bytes(p) for p in parts)).hexdigest()


def stable_id(prefix: str, parts: list[Any]) -> str:
    return prefix + "_" + digest_parts(parts)[:16]


def read_json(path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")


def read_ndjson(path) -> list[tuple[int, str]]:
    """Return list of (1-based line number, raw line text)."""
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            text = raw.strip()
            if text:
                out.append((lineno, text))
    return out
