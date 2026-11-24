from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_dumps(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_bytes(value: Any) -> bytes:
    if isinstance(value, str):
        return value.encode("utf-8")
    return canonical_dumps(value).encode("utf-8")


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def fingerprint(value: Any) -> str:
    return sha256_hex(canonical_bytes(value))


def prefixed_digest(value: Any) -> str:
    return "sha256:" + fingerprint(value)


def derive_cache_key(
    *,
    stage: str,
    upstream_fingerprint: str,
    quota_version: Any,
    capacity_version: Any,
) -> str:
    preimage = {
        "stage": stage,
        "upstream_fingerprint": upstream_fingerprint,
        "quota_version": quota_version,
        "capacity_version": capacity_version,
    }
    return "ck_" + fingerprint(preimage)[:24]
