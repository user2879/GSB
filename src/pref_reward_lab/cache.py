from __future__ import annotations

from typing import Any

from .canonical import canonical_json, fingerprint

CACHE_SCHEMA_VERSION = 1
STAGES = ("features", "scoring", "conversion")


def empty_cache() -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "features": {},
        "scoring": {},
        "conversion": {},
        "metrics": {},
        "strategy": {},
        "bucket_refs": {},
        "pair_content_fingerprints": {},
    }


def normalize_cache(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != CACHE_SCHEMA_VERSION:
        return empty_cache()
    result = empty_cache()
    for key in (
        "features",
        "scoring",
        "conversion",
        "metrics",
        "strategy",
        "bucket_refs",
        "pair_content_fingerprints",
    ):
        candidate = value.get(key)
        if isinstance(candidate, dict):
            result[key] = candidate
    return result


def cache_lookup(cache: dict[str, Any], stage: str, pair_id: str, key: str) -> Any | None:
    entry = cache.get(stage, {}).get(pair_id)
    if not isinstance(entry, dict):
        return None
    if entry.get("key") != key:
        return None
    return entry.get("value")


def cache_store(cache: dict[str, Any], stage: str, pair_id: str, key: str, value: Any) -> None:
    cache[stage][pair_id] = {"key": key, "value": value}


def metrics_lookup(cache: dict[str, Any], key: str) -> Any | None:
    entry = cache.get("metrics", {}).get(key)
    if not isinstance(entry, dict):
        return None
    return entry.get("value")


def metrics_store(cache: dict[str, Any], key: str, value: Any) -> None:
    cache["metrics"] = {key: {"key": key, "value": value}}


def build_bucket_refs(feature_values: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    refs: dict[str, set[str]] = {}
    for pair_id, value in feature_values.items():
        for bucket in value.get("buckets", {}):
            refs.setdefault(str(bucket), set()).add(pair_id)
    return {bucket: sorted(refs[bucket]) for bucket in sorted(refs, key=lambda item: int(item))}


def invalidation_summary(
    old_cache: dict[str, Any],
    new_cache: dict[str, Any],
    *,
    recomputed: dict[str, list[str]],
    reused: dict[str, int],
    affected_buckets: list[int],
) -> dict[str, Any]:
    old_pairs = old_cache.get("pair_content_fingerprints", {})
    new_pairs = new_cache.get("pair_content_fingerprints", {})
    replaced = sorted(
        pair_id
        for pair_id in set(old_pairs) & set(new_pairs)
        if old_pairs[pair_id] != new_pairs[pair_id]
    )
    deleted = sorted(set(old_pairs) - set(new_pairs))
    added = sorted(set(new_pairs) - set(old_pairs))
    return {
        "strategy_reused": bool(
            old_cache.get("strategy")
            and old_cache.get("strategy", {}).get("key")
            == new_cache.get("strategy", {}).get("key")
        ),
        "changed_pair_ids": replaced,
        "deleted_pair_ids": deleted,
        "added_pair_ids": added,
        "affected_bucket_ids": sorted(affected_buckets),
        "recomputed_pair_ids": {
            stage: sorted(set(recomputed.get(stage, []))) for stage in STAGES
        },
        "reused_counts": {stage: int(reused.get(stage, 0)) for stage in STAGES},
        "recomputed_counts": {
            stage: len(set(recomputed.get(stage, []))) for stage in STAGES
        },
        "metrics_recomputed": bool(recomputed.get("metrics")),
        "reused_existing_snapshot_cache": bool(old_cache.get("pair_content_fingerprints")),
    }


def data_fingerprint(
    *,
    raw_pairs: dict[str, Any],
    exchange_v1: dict[str, Any],
    exchange_v2: dict[str, Any],
    source_version: str,
) -> str:
    return fingerprint(
        {
            "raw_pairs": {key: raw_pairs[key] for key in sorted(raw_pairs)},
            "exchange_v1": {key: exchange_v1[key] for key in sorted(exchange_v1)},
            "exchange_v2": {key: exchange_v2[key] for key in sorted(exchange_v2)},
            "source_version": source_version,
        },
        prefix="dataset:",
    )


def pair_exchange_fingerprint(value: dict[str, Any]) -> str:
    return fingerprint(value, prefix="exchange:")


def metrics_cache_key(payload: dict[str, Any], config_fingerprint: str) -> str:
    return fingerprint(
        {"payload": payload, "config": config_fingerprint, "metrics_schema": 1},
        prefix="metrics:",
    )


def cache_digest(cache: dict[str, Any]) -> str:
    return fingerprint(canonical_json(cache), prefix="cache:")
