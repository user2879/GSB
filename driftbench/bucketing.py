from __future__ import annotations

import hashlib
from typing import Any

from .concurrency import ParallelRunner
from .fingerprint import canonical_json, digest, labelled_digest
from .models import Config, Exposure


def bucket_hash(seed: str, user_id: str, attribute_value: Any | None, bucket_count: int) -> int:
    payload = "\x1f".join(
        (
            seed,
            user_id,
            "__missing__" if attribute_value is None else canonical_json(attribute_value),
        )
    )
    integer = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest(), "big")
    return integer % bucket_count


def group_for_bucket(bucket: int, config: Config) -> str:
    # Fixed, auditable 50/50 parity mapping; bucket parity is stable across machines.
    return config.control_group if bucket % 2 == 0 else config.experiment_group


def bucketing_content(users: list[str], properties: dict[str, dict[str, Any]], config: Config) -> dict[str, Any]:
    return {
        "users": [
            {
                "user_id": user_id,
                "attribute": properties.get(user_id, {}).get(config.bucket_property_key),
            }
            for user_id in sorted(users)
        ],
        "bucket_hash_seed": config.bucket_hash_seed,
        "bucket_count": config.bucket_count,
        "bucket_property_key": config.bucket_property_key,
        "fallback_policy": config.fallback_policy,
        "missing_attribute_group": config.missing_attribute_group,
        "control_group": config.control_group,
        "experiment_group": config.experiment_group,
    }


def compute_bucketing(
    exposures: list[Exposure],
    properties: dict[str, dict[str, Any]],
    config: Config,
    runner: ParallelRunner,
    *,
    cache_key: str,
) -> dict[str, Any]:
    users = sorted({event.user_id for event in exposures})
    content = bucketing_content(users, properties, config)
    content_fingerprint = digest(content)

    def assign(user_id: str) -> dict[str, Any]:
        values = properties.get(user_id)
        attribute_value: Any | None = None
        fallback_reason: str | None = None
        if values is None:
            fallback_reason = "missing_user"
        elif config.bucket_property_key not in values:
            fallback_reason = "missing_attribute"
        else:
            attribute_value = values[config.bucket_property_key]

        bucket = bucket_hash(config.bucket_hash_seed, user_id, attribute_value, config.bucket_count)
        if fallback_reason and config.fallback_policy == "fixed_group":
            group = config.missing_attribute_group
        else:
            group = group_for_bucket(bucket, config)
        if fallback_reason and config.fallback_policy == "user_id_only":
            bucket = bucket_hash(config.bucket_hash_seed, user_id, None, config.bucket_count)
            group = group_for_bucket(bucket, config)
        return {
            "user_id": user_id,
            "bucket": bucket,
            "group": group,
            "attribute_value": attribute_value,
            "fallback": fallback_reason is not None,
            "fallback_reason": fallback_reason,
            "fallback_policy": config.fallback_policy if fallback_reason else None,
        }

    assignments = runner.map_tasks(
        [(f"bucketing|{user_id}", lambda uid=user_id: assign(uid)) for user_id in users]
    )
    assignment_map = {item["user_id"]: item for item in assignments}
    fallbacks = [item for item in assignments if item["fallback"]]
    group_members = {
        group: [item["user_id"] for item in assignments if item["group"] == group]
        for group in config.group_order
    }
    return {
        "stage": "bucketing",
        "cache_key": cache_key,
        "content_fingerprint": content_fingerprint,
        "config_fingerprint": config.fingerprint(),
        "assignments": assignment_map,
        "groups": {
            group: {
                "user_count": len(group_members[group]),
                "user_fingerprint": digest(sorted(group_members[group])),
            }
            for group in config.group_order
        },
        "fallbacks": fallbacks,
        "fallback_count": len(fallbacks),
        "mapping_rule": "even_bucket=control;odd_bucket=experiment",
    }


def bucketing_cache_key(
    exposures: list[Exposure], properties: dict[str, dict[str, Any]], config: Config
) -> str:
    users = sorted({event.user_id for event in exposures})
    return "bucketing:" + labelled_digest("bucketing-v1", bucketing_content(users, properties, config))
