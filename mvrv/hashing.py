from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_object(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def short_fingerprint(value: Any, length: int = 16) -> str:
    if isinstance(value, (bytes, bytearray)):
        digest = hashlib.sha256(bytes(value)).hexdigest()
    elif isinstance(value, str):
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    else:
        digest = sha256_object(value)
    return digest[:length]


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
