"""Canonical JSON serialization and digests; never depends on dict/hash order."""

import hashlib
import json


def canonical_bytes(obj) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def canonical_text(obj) -> str:
    return canonical_bytes(obj).decode("utf-8")


def digest(obj) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(obj)).hexdigest()


def digest_bytes(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()
