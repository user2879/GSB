from __future__ import annotations

import json
from typing import Any, Callable

from .canonical import canonical_bytes, sha256_bytes
from .errors import AppError
from .ledger import Ledger
from .simulation import compute_aggregate


COMPARE_FIELDS = ("scan_rows", "byte_digest", "summary_digest", "groups")


def _first_aggregate_difference(
    expected: dict[str, Any], actual: dict[str, Any]
) -> tuple[str, Any, Any] | None:
    for field in COMPARE_FIELDS:
        if expected.get(field) != actual.get(field):
            return field, expected.get(field), actual.get(field)
    return None


def _comparison_pair(cache: dict[str, Any] | None, authoritative: dict[str, Any]) -> dict[str, Any]:
    return {
        "scan_rows": {
            "cache": None if cache is None else cache.get("scan_rows"),
            "authoritative": authoritative.get("scan_rows"),
        },
        "byte_digest": {
            "cache": None if cache is None else cache.get("byte_digest"),
            "authoritative": authoritative.get("byte_digest"),
        },
    }


def _preview_error(
    row: dict[str, Any],
    source: str,
    field: str,
    cache_value: Any,
    authoritative_value: Any,
    cache: dict[str, Any] | None,
    authoritative: dict[str, Any],
    checked_before: int,
) -> AppError:
    return AppError(
        "E_PREVIEW",
        "账本缓存摘要与权威重算不一致",
        field,
        row["shard_id"],
        {
            "cluster_id": row["cluster_id"],
            "task_id": row["task_id"],
            "shard_id": row["shard_id"],
            "dependency": source,
            "field": field,
            "cache_value": cache_value,
            "authoritative_value": authoritative_value,
            "cache": cache,
            "authoritative": authoritative,
            "comparison": _comparison_pair(cache, authoritative),
            "records_checked_before_mismatch": checked_before,
        },
    )


def verify_ledger_authoritative(
    ledger: Ledger,
    dispatch: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Recompute every committed shard with NumPy and compare ledger cache.

    When ``dispatch`` is provided, the same shard is first sent to its live
    loopback node. The node result must agree with the local NumPy authority.
    """

    plan = ledger.run_plan()
    plan_shards = {shard["shard_id"]: shard for shard in plan.get("shards", [])}
    rows = [row for row in ledger.current_shards(include_failed=False) if row["outcome"] == "completed"]
    checked = 0
    replay_digests: list[str] = []
    for row in rows:
        shard = plan_shards.get(row["shard_id"])
        if shard is None:
            raise AppError(
                "E_PREVIEW",
                "账本分片在运行计划中不存在",
                "shard_id",
                row["shard_id"],
                {
                    "cluster_id": row["cluster_id"],
                    "task_id": row["task_id"],
                    "shard_id": row["shard_id"],
                    "comparison": _comparison_pair(None, {"scan_rows": None, "byte_digest": None}),
                },
            )
        authoritative = compute_aggregate(
            simulation_seed=plan["simulation_seed"],
            cluster_id=shard["cluster_id"],
            revision_version=shard["revision_version"],
            task_id=shard["task_id"],
            shard_id=shard["shard_id"],
            config_keys=shard["config_keys"],
            config_values=shard["config_values"],
        )
        if dispatch is not None:
            try:
                replay = dispatch(shard)
            except AppError as exc:
                raise AppError(
                    "E_PREVIEW",
                    "权威重放无法完成分片下发",
                    "dispatch",
                    row["shard_id"],
                    {
                        "cluster_id": row["cluster_id"],
                        "task_id": row["task_id"],
                        "shard_id": row["shard_id"],
                        "dependency": "node_replay",
                        "cause": exc.to_dict(),
                    },
                ) from None
            node_aggregate = replay["aggregate"]
            difference = _first_aggregate_difference(authoritative, node_aggregate)
            if difference is not None:
                field, expected, actual = difference
                raise _preview_error(
                    row,
                    "node_replay_vs_local_numpy",
                    field,
                    actual,
                    expected,
                    node_aggregate,
                    authoritative,
                    checked,
                )
            replay_digests.append(sha256_bytes(canonical_bytes(node_aggregate)))

        cache = json.loads(row["aggregate_json"]) if row["aggregate_json"] else None
        difference = _first_aggregate_difference(cache or {}, authoritative)
        if difference is not None:
            field, cache_value, authoritative_value = difference
            raise _preview_error(
                row,
                "ledger_cache_vs_local_numpy",
                field,
                cache_value,
                authoritative_value,
                cache,
                authoritative,
                checked,
            )
        checked += 1
    return {
        "records_checked": checked,
        "node_replays": len(replay_digests),
        "replay_digest": sha256_bytes(canonical_bytes(replay_digests)) if replay_digests else None,
        "first_mismatch": None,
        "authority": "numpy_local" + ("+loopback_node" if dispatch is not None else ""),
    }


def plan_replay_digest(plan: dict[str, Any]) -> str:
    return sha256_bytes(canonical_bytes(plan))
