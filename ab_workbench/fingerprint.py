"""Content fingerprints and derived cache keys.

Fingerprints use BLAKE2b (stable across runs, no PYTHONHASHSEED dependence).
Canonical JSON serialization is sorted and separator-fixed so that logically
identical payloads always produce identical bytes.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List

CALIBER_VERSION = "caliber-v1"


def canonical_json(obj: Any) -> str:
    return json.dumps(
        obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=_default
    )


def _default(obj: Any) -> Any:
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise TypeError(f"object not fingerprintable: {type(obj)!r}")


def fingerprint_bytes(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def fingerprint_text(text: str) -> str:
    return fingerprint_bytes(text.encode("utf-8"))


def fingerprint_json(obj: Any) -> str:
    return fingerprint_text(canonical_json(obj))


def fingerprint_file(path: str) -> str:
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def file_set_fingerprint(paths: Iterable[str]) -> str:
    """Order-independent fingerprint over a set of input files."""
    parts: List[str] = [fingerprint_file(p) for p in sorted(paths)]
    return fingerprint_text("|".join(parts))


def derive_cache_key(
    stage: str,
    upstream_fingerprint: str,
    config_fingerprint: str,
    caliber: str = CALIBER_VERSION,
    scope: str = "",
) -> str:
    payload = {
        "stage": stage,
        "upstream": upstream_fingerprint,
        "config": config_fingerprint,
        "caliber": caliber,
        "scope": scope,
    }
    return f"{stage}:{fingerprint_json(payload)}"


def stable_sort_key(namespace: str, identifier: str) -> str:
    """Deterministic per-task ordering key derived from id and stage name."""
    return fingerprint_text(f"{namespace}::{identifier}")
