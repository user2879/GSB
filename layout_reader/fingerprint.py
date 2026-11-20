"""Canonical serialization and content fingerprints."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def fingerprint(value: Any) -> str:
    return sha256_text(canonical_json(value))


def stage_cache_key(
    stage: str,
    implementation_version: int,
    upstream: Any,
    parameters: Any,
    fault_items: list[dict[str, Any]] | None = None,
) -> str:
    """Derive a cache key from upstream content and stage parameters.

    Fault items are part of the key because an injected input fault changes the
    effective input of the stage.  Adding or removing an injection must not
    silently reuse a clean artifact.
    """

    return fingerprint(
        {
            "stage": stage,
            "implementation_version": implementation_version,
            "upstream": upstream,
            "parameters": parameters,
            "faults": fault_items or [],
        }
    )

