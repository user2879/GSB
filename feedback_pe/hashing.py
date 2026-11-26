"""Canonical JSON and content-fingerprint helpers."""
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
        default=_json_default,
    )


def _json_default(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint(namespace: str, value: Any) -> str:
    payload = canonical_json(value)
    return sha256_text(f"{namespace}:{payload}")


def short_fp(value: str, length: int = 12) -> str:
    return value[:length]


def task_sort_key(sample_id: str, scheme_name: str, stage: str) -> str:
    return f"{sample_id}::{scheme_name}::{stage}"


def task_sort_hash(sort_key: str) -> str:
    return sha256_text("task:" + sort_key)


def point_cost(sample: dict[str, Any]) -> int:
    return int(sample["dimension"]) * len(sample["key_positions"])


def meter_sample_points(sample: dict[str, Any], quota: int) -> dict[str, Any]:
    """Meter dimension/key units and stop at the first quota overflow."""
    dimension = int(sample["dimension"])
    key_count = len(sample["key_positions"])
    projected = dimension * key_count
    consumed = 0
    exceeded_at = None
    for _key_index in range(key_count):
        for _dimension_index in range(dimension):
            consumed += 1
            if consumed > int(quota):
                exceeded_at = consumed
                break
        if exceeded_at is not None:
            break
    return {
        "projected_cost": projected,
        "metered_points": consumed,
        "exceeded": exceeded_at is not None,
        "exceeded_at": exceeded_at,
        "quota": int(quota),
    }
