from __future__ import annotations

from collections import defaultdict
from typing import Any

from .canonical import fingerprint
from .config import bucket_config_fingerprint


BUCKET_PRODUCT = "inverted_bucket"


def bucket_product_id(fingerprint_key: str) -> str:
    return f"bucket:{fingerprint_key}"


def _entry_from_pair(pair: dict[str, Any]) -> dict[str, Any]:
    return {
        "entry_id": pair["product_id"],
        "pair_cache_key": pair["cache_key"],
        "clip_id": pair["clip_id"],
        "track_id": pair["track_id"],
        "version": int(pair["version"]),
        "anchor_frame": int(pair["anchor_frame"]),
        "target_frame": int(pair["target_frame"]),
        "strength_q": int(pair["strength_q"]),
        "fingerprint_key": pair["fingerprint_key"],
    }


def build_buckets(pairs: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        grouped[pair["fingerprint_key"]].append(_entry_from_pair(pair))

    capacity = int(config["bucket_capacity"])
    bucket_config_fp = bucket_config_fingerprint(config)
    buckets: list[dict[str, Any]] = []

    for fingerprint_key in sorted(grouped):
        entries = grouped[fingerprint_key]
        retention_order = sorted(
            entries,
            key=lambda entry: (
                -int(entry["strength_q"]),
                str(entry["track_id"]),
                int(entry["version"]),
                int(entry["anchor_frame"]),
                int(entry["target_frame"]),
                str(entry["entry_id"]),
                str(entry["pair_cache_key"]),
            ),
        )
        retained_rank = retention_order[:capacity]
        evicted_rank = retention_order[capacity:]
        retained_ids = {entry["entry_id"] for entry in retained_rank}
        retained = sorted(
            (entry for entry in entries if entry["entry_id"] in retained_ids),
            key=lambda entry: (
                str(entry["track_id"]),
                int(entry["version"]),
                int(entry["anchor_frame"]),
                int(entry["target_frame"]),
                str(entry["entry_id"]),
            ),
        )
        evicted: list[dict[str, Any]] = []
        for rank, entry in enumerate(evicted_rank, start=capacity + 1):
            evicted.append(
                {
                    **entry,
                    "eviction_reason": "bucket_capacity_exceeded",
                    "retention_rank": rank,
                    "capacity": capacity,
                }
            )

        all_pair_keys = sorted(entry["pair_cache_key"] for entry in entries)
        cache_key = fingerprint(
            {
                "product_type": BUCKET_PRODUCT,
                "product_id": bucket_product_id(fingerprint_key),
                "fingerprint_key": fingerprint_key,
                "config_fingerprint": bucket_config_fp,
                "capacity": capacity,
                "all_pair_cache_keys": all_pair_keys,
                "retained_pair_cache_keys": sorted(entry["pair_cache_key"] for entry in retained),
            },
            "bucketkey_",
        )
        buckets.append(
            {
                "product_type": BUCKET_PRODUCT,
                "product_id": bucket_product_id(fingerprint_key),
                "cache_key": cache_key,
                "fingerprint_key": fingerprint_key,
                "bucket_config_fingerprint": bucket_config_fp,
                "capacity": capacity,
                "total_entry_count": len(entries),
                "retained_entry_count": len(retained),
                "entries": retained,
                "evicted_entries": evicted,
                "upstream_refs": [{"type": "fingerprint_pair", "key": key} for key in all_pair_keys]
                + [{"type": "bucket_config", "key": bucket_config_fp}],
            }
        )

    buckets.sort(key=lambda bucket: bucket["fingerprint_key"])
    return buckets


def bucket_index(buckets: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {bucket["fingerprint_key"]: bucket for bucket in buckets}
