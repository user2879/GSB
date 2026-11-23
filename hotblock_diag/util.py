"""Deterministic hashing / canonical JSON helpers."""
from __future__ import annotations

import hashlib
import json
from typing import Any


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def short_hash(data: bytes) -> str:
    return sha256_hex(data)[:16]


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def json_hash(obj: Any) -> str:
    return short_hash(canonical_json_bytes(obj))


def dump_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(obj, handle, sort_keys=True, indent=2, ensure_ascii=False,
                  allow_nan=False)
        handle.write("\n")
