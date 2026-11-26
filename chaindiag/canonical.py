from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    """Serialize data in a stable, hash-friendly form.

    ``sort_keys`` is safe because all payloads are JSON-compatible.  Tuples are
    converted by the default hook to lists.
    """

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda obj: list(obj) if isinstance(obj, tuple) else str(obj),
    )


def sha256_hex(value: Any) -> str:
    payload = canonical_json(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def content_fingerprint(value: Any) -> str:
    return sha256_hex(value)


def cache_key(*, upstream: Any, operator_fingerprint: str, fixed_point: dict[str, Any],
              exchange_version: str, stage: str, identity: dict[str, Any] | None = None) -> str:
    return sha256_hex(
        {
            "schema": "chaindiag-cache-v1",
            "stage": stage,
            "identity": identity or {},
            "upstream": upstream,
            "operator_fingerprint": operator_fingerprint,
            "fixed_point": fixed_point,
            "exchange_version": exchange_version,
        }
    )
