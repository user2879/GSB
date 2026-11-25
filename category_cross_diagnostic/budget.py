"""Cardinality and encoding-point budget adjudication."""

from __future__ import annotations

from typing import Any, Iterable

from .config import Strategy
from .constants import ERROR_POINT_BUDGET_EXHAUSTED
from .enumeration import Combo
from .hashing import sha256_hex


def cardinality_fingerprint_params(strategy: Strategy) -> str:
    return sha256_hex(
        "cardinality-params-v1",
        "true_tuple_count-v1",
        "collision_excess=true_unique-unique_code",
        "null_semantics=any_missing",
    )


def dictionary_fingerprint_params(strategy: Strategy) -> str:
    return sha256_hex(
        "dictionary-params-v1",
        "blake2b-256",
        "canonical-json-v1",
        "bit-mask-v1",
        strategy.dictionary_preview_limit,
    )


def decision_fingerprint_params(strategy: Strategy) -> str:
    return sha256_hex(
        "decision-params-v1",
        strategy.cardinality_budget,
        strategy.point_budget,
        strategy.over_budget_action,
        strategy.truncate_buckets,
        "point_cost=row_count*encoding_width",
        "order=stable_combination_order",
    )


def adjudicate_cardinality(
    unique_code_count: int,
    strategy: Strategy,
    *,
    truncated_bucket_count: int | None = None,
) -> dict[str, Any]:
    budget = strategy.cardinality_budget
    if unique_code_count <= budget:
        return {
            "verdict": "within_budget",
            "retained": True,
            "disposition": "retained",
            "degradations": [],
            "dominant_constraint": "none",
        }

    if strategy.over_budget_action == "skip":
        detail = {
            "type": "cardinality_skip",
            "code": "E_CARDINALITY_BUDGET_SKIP",
            "observed_unique_code_count": unique_code_count,
            "cardinality_budget": budget,
            "budget_excess": unique_code_count - budget,
        }
        return {
            "verdict": "over_budget_skip",
            "retained": False,
            "disposition": "skipped",
            "degradations": [detail],
            "dominant_constraint": "cardinality_budget",
        }

    bucket_count = int(
        strategy.truncate_buckets if strategy.truncate_buckets is not None else budget
    )
    retained_buckets = (
        truncated_bucket_count
        if truncated_bucket_count is not None
        else min(unique_code_count, bucket_count)
    )
    detail = {
        "type": "hash_bucket_truncation",
        "code": "E_CARDINALITY_BUDGET_TRUNCATE",
        "observed_unique_code_count": unique_code_count,
        "cardinality_budget": budget,
        "budget_excess": unique_code_count - budget,
        "bucket_count": bucket_count,
        "retained_bucket_count": retained_buckets,
        "removed_dictionary_entries": max(0, unique_code_count - retained_buckets),
    }
    return {
        "verdict": "over_budget_truncate",
        "retained": True,
        "disposition": "truncated",
        "degradations": [detail],
        "dominant_constraint": "cardinality_budget",
    }


def truncated_occupancy(entries: Iterable[dict[str, int]], bucket_count: int) -> int:
    return len({int(entry["code"]) % bucket_count for entry in entries})


def point_cost(row_count: int, encoding_width: int) -> int:
    return int(row_count) * int(encoding_width)


def adjudicate_combo(
    combo: Combo,
    column_fingerprints: dict[str, str],
    cardinality: dict[str, Any],
    encoding_entries: Iterable[dict[str, int]],
    *,
    row_count: int,
    encoding_width: int,
    consumed_points: int,
    strategy: Strategy,
    width_gear_fp: str,
) -> dict[str, Any]:
    card_decision = adjudicate_cardinality(
        int(cardinality["unique_code_count"]), strategy
    )
    if card_decision["verdict"] == "over_budget_truncate":
        bucket_count = (
            strategy.truncate_buckets
            if strategy.truncate_buckets is not None
            else strategy.cardinality_budget
        )
        occupied = truncated_occupancy(encoding_entries, int(bucket_count))
        card_decision = adjudicate_cardinality(
            int(cardinality["unique_code_count"]),
            strategy,
            truncated_bucket_count=occupied,
        )

    cost = point_cost(row_count, encoding_width)
    consumed_after = consumed_points + cost
    point_exhausted = consumed_after > strategy.point_budget
    point_info = {
        "cost": cost,
        "row_count": row_count,
        "encoding_width": encoding_width,
        "budget": strategy.point_budget,
        "consumed_before": consumed_points,
        "consumed_after": consumed_after,
        "remaining_before": strategy.point_budget - consumed_points,
        "remaining_after": strategy.point_budget - consumed_after,
        "status": "exhausted" if point_exhausted else "charged",
        "exceeded_by": max(0, consumed_after - strategy.point_budget),
    }

    sequence = [
        {
            "order": 1,
            "constraint": "cardinality_budget",
            "verdict": card_decision["verdict"],
            "disposition": card_decision["disposition"],
        },
        {
            "order": 2,
            "constraint": "point_budget",
            "verdict": "exhausted" if point_exhausted else "within_budget",
            "charged_points": cost,
        },
    ]

    if card_decision["disposition"] == "skipped":
        dominant = "cardinality_budget"
    elif point_exhausted:
        dominant = "point_budget"
    elif card_decision["disposition"] == "truncated":
        dominant = "cardinality_budget"
    else:
        dominant = "none"

    codes: list[str] = []
    if card_decision["verdict"] == "over_budget_skip":
        codes.append("E_CARDINALITY_BUDGET_SKIP")
    elif card_decision["verdict"] == "over_budget_truncate":
        codes.append("E_CARDINALITY_BUDGET_TRUNCATE")
    if point_exhausted:
        codes.append(ERROR_POINT_BUDGET_EXHAUSTED)

    active_fp = combo.width_fingerprint(width_gear_fp)
    return {
        "combo_id": combo.combo_id,
        "columns": list(combo.columns),
        "order": combo.order,
        "combo_fingerprint": active_fp,
        "base_fingerprint": combo.base_fingerprint,
        "column_fingerprints": {
            name: column_fingerprints[name] for name in combo.columns
        },
        "cardinality_budget": strategy.cardinality_budget,
        "point_budget": strategy.point_budget,
        "cardinality_verdict": card_decision["verdict"],
        "disposition": card_decision["disposition"],
        "retained": card_decision["retained"],
        "degradations": card_decision["degradations"],
        "decision_sequence": sequence,
        "point": point_info,
        "point_exhausted": point_exhausted,
        "joint_constraint_triggered": bool(
            card_decision["degradations"] and point_exhausted
        ),
        "dominant_constraint": dominant,
        "terminal_constraint": "point_budget" if point_exhausted else None,
        "decision_codes": codes,
    }


def pending_record(combo: Combo, width_gear_fp: str, index: int, reason: str) -> dict[str, Any]:
    code = (
        ERROR_POINT_BUDGET_EXHAUSTED
        if reason == ERROR_POINT_BUDGET_EXHAUSTED
        else reason
    )
    return {
        "queue_index": index,
        "combo_id": combo.combo_id,
        "columns": list(combo.columns),
        "order": combo.order,
        "combo_fingerprint": combo.width_fingerprint(width_gear_fp),
        "base_fingerprint": combo.base_fingerprint,
        "stable_task_key": combo.stable_sort_key,
        "code": code,
        "reason": reason,
    }
