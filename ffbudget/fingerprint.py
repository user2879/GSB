"""Content fingerprints and deterministic cache keys."""
from __future__ import annotations

import hashlib
import json


def canonical_json(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def content_fingerprint(payload) -> str:
    """SHA-256 over the canonical JSON encoding of a payload."""
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8"))
    return digest.hexdigest()


def cache_key(parts: list[str]) -> str:
    joined = "|".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def view_cache_key(slot_key: str, content_fp: str, authority_fp: str,
                   staleness_fp: str, width_tier: str) -> str:
    return cache_key(["view", slot_key, content_fp, authority_fp,
                      staleness_fp, width_tier])


def score_cache_key(slot_key: str, content_fp: str, authority_fp: str,
                    staleness_fp: str, width_tier: str) -> str:
    return cache_key(["score", slot_key, content_fp, authority_fp,
                      staleness_fp, width_tier])


def ranking_cache_key(request_id: str, slot_fingerprints: list[str],
                      width_tier: str) -> str:
    return cache_key(["ranking", request_id, *sorted(slot_fingerprints),
                      width_tier])
