"""Reconciled inverted-index ledger assembled from shard receipts."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

from .canonical import fingerprint


def _without_elapsed(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_elapsed(item)
            for key, item in value.items()
            if key != "elapsed_ms_monotonic"
        }
    if isinstance(value, list):
        return [_without_elapsed(item) for item in value]
    return value


def business_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove wall/monotonic timing fields for deterministic comparisons."""

    return _without_elapsed(dict(value))


def assemble_ledger(
    *,
    global_version: int,
    config_hash: str,
    index_revision: int,
    summaries: Mapping[str, Mapping[str, Any]],
    receipt_arbitrations: Iterable[Mapping[str, Any]],
    failures: Iterable[Mapping[str, Any]],
    consumed_budget: int,
) -> dict[str, Any]:
    buckets: list[dict[str, Any]] = []
    for shard_id in sorted(summaries):
        summary = summaries[shard_id]
        for bucket in sorted(summary.get("buckets", []), key=lambda item: item["bucket_index"]):
            item = dict(bucket)
            item["shard_id"] = shard_id
            buckets.append(item)

    arbitration_list = sorted(
        (dict(item) for item in receipt_arbitrations),
        key=lambda item: (
            str(item.get("shard_id", "")),
            str(item.get("request_id", "")),
            int(item.get("served_revision", 0)),
        ),
    )
    decision_counts = Counter(str(item.get("decision", "unknown")) for item in arbitration_list)
    code_counts = Counter(str(item.get("code", "unknown")) for item in arbitration_list)
    failure_list = sorted(
        (dict(item) for item in failures),
        key=lambda item: (
            str(item.get("code", "")),
            str(item.get("object_type", "")),
            str(item.get("object_id", "")),
            str(item.get("field", "")),
            str(item.get("message", "")),
        ),
    )
    business_code_counts = Counter(str(item.get("code", "unknown")) for item in failure_list)
    ledger = {
        "schema_version": 1,
        "global_version": global_version,
        "config_hash": config_hash,
        "index_revision": index_revision,
        "entries_count": sum(int(summary.get("entries_count", 0)) for summary in summaries.values()),
        "consumed_budget": consumed_budget,
        "shard_entry_counts": {
            shard_id: int(summaries[shard_id].get("entries_count", 0))
            for shard_id in sorted(summaries)
        },
        "buckets": buckets,
        "receipt_arbitrations": arbitration_list,
        "arbitration_stats": {
            "total": len(arbitration_list),
            "accepted": int(decision_counts.get("accepted", 0)),
            "accepted_compatible": int(decision_counts.get("accepted_compatible", 0)),
            "rejected": int(decision_counts.get("rejected", 0)),
            "decision_counts": dict(sorted(decision_counts.items())),
            "code_counts": dict(sorted(code_counts.items())),
        },
        "failure_list": failure_list,
        "business_code_counts": dict(sorted(business_code_counts.items())),
    }
    ledger["business_digest"] = fingerprint(business_projection(ledger))
    return ledger

