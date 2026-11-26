"""Deterministic content fingerprints and derived cache keys."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable


def _canonical(payload: Any) -> str:
    """Stable JSON encoding: sorted keys, no whitespace surprises."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(payload: Any) -> str:
    """Short stable sha256 hex digest of any JSON-serialisable payload."""
    raw = _canonical(payload).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def text_fingerprint(text_id: str, tokens: Iterable[str], channel: str) -> str:
    return digest(
        {
            "kind": "text",
            "text_id": text_id,
            "tokens": tuple(tokens),
            "channel": channel,
        }
    )


def entry_fingerprint(
    entry_id: str,
    term: Iterable[str],
    weight: int,
    intent: str,
) -> str:
    return digest(
        {
            "kind": "trigger_entry",
            "entry_id": entry_id,
            "term": tuple(term),
            "weight": weight,
            "intent": intent,
        }
    )


def chain_fingerprint(revision_ids_in_order: Iterable[str], head_id: str) -> str:
    return digest(
        {
            "kind": "revision_chain",
            "ordered_revisions": tuple(revision_ids_in_order),
            "head": head_id,
        }
    )


def composite_key(parts: Iterable[str]) -> str:
    """Combine ordered child fingerprints into one cache key."""
    return digest({"kind": "composite", "parts": tuple(parts)})


def cache_key(kind: str, *parts: str) -> str:
    return digest({"kind": kind, "parts": tuple(parts)})
